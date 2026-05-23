import numpy as np
import math
from itertools import combinations, product
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark import SparkConf, SparkContext
# Bigger socket timeout because the venv lives under OneDrive on Windows:
# Python worker cold-start can take longer than the 15 s default and JVM
# would otherwise abort the task with
#   SocketTimeoutException: Timed out while waiting for the Python worker to connect back
conf = (SparkConf()
        .setMaster("local[*]")
        .setAppName("SimHash")
        .set("spark.python.worker.faulthandler.enabled", "true")
        .set("spark.python.authenticate.socketTimeout", "300s"))
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# ----------------------------------------------------------------
# Env vars (set by run_all.py). Fall back to hard-coded defaults.
#   BENCH_N_VALUES = "1000,10000"
#   BENCH_SUFFIX   = "_sanity"
#   BENCH_MEM_OPT  ignored (original code has no mem-opt path)
# ----------------------------------------------------------------
SUFFIX = os.environ.get("BENCH_SUFFIX", "")
_env_n = os.environ.get("BENCH_N_VALUES")
if _env_n is not None:
    N_VALUES = [int(x.strip()) for x in _env_n.split(",") if x.strip()]
else:
    N_VALUES = [100]

# CARICA DATI E CALCOLA SIMHASH

def load_fvecs(filename):
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    n = len(data) // (d + 1)
    data = data.reshape(n, d + 1)
    return data[:, 1:].view(np.float32)

D_full = load_fvecs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sift", "sift_base.fvecs"))
d = D_full.shape[1]  # dimensione vettori: 128

print(f"[config] N_VALUES={N_VALUES}")
print(f"[config] SUFFIX={SUFFIX!r}")

# broadcast dei dati (riassegnato dentro al loop)
D_bc = None

def compute_simhash(i,m):
    # prendi il vettore i-esimo
    vector = D_bc.value[i]          # shape (128,)
    R = R_full_bc.value[:m, :]
    # proietta il vettore sulle m direzioni random
    w = np.dot(R, vector)  # shape (64,)

    # applica sign: 1 se positivo, 0 se negativo
    signature = np.zeros(m, dtype=int)
    for j in range(m):
        if w[j] > 0:
            signature[j] = 1
        else:
            signature[j] = 0

    return (i, signature)

def make_bands(item,b,r):
    doc_id    = item[0]
    signature = item[1]

    result = []
    for band_index in range(b):
        # prende i bit di questa band
        start     = band_index * r
        end       = start + r
        band_bits = tuple(signature[start:end])

        # chiave = (posizione band, valore band)
        band_key  = (band_index, band_bits)
        result.append((band_key, doc_id))

    return result

def get_candidate_pairs(item):
    #band_key = item[0]         # non serve ma per chiarezza
    doc_ids  = list(item[1])   # lista di doc che condividono questa band
    # genera tutte le coppie
    if len(doc_ids) > 300:
        return []

    if len(doc_ids) < 2:
        return []
    
    pairs = []
    for pair in combinations(doc_ids, 2):
        pairs.append(pair)

    return pairs

def compute_hamming(pair,m,sig_bc):
    doc1 = pair[0]
    doc2 = pair[1]
    sig1 = sig_bc.value[doc1]
    sig2 = sig_bc.value[doc2]
    # conta i bit diversi
    differing_bits = 0
    for j in range(m):
        if sig1[j] != sig2[j]:
            differing_bits += 1
    return (doc1, doc2, differing_bits)

def compute_cosine_similarity(pair):
    doc1 = pair[0]
    doc2 = pair[1]
    vec1 = D_bc.value[doc1]
    vec2 = D_bc.value[doc2]   
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return cosine_sim


m_values = [64]          # numero di bit della firma
k_values = [0,2,4,8]                       # numero massimo bit diversi
b_values = [4,8]
rows = []

_plot_colors = {64: "blue", 128: "green", 256: "orange", 512: "red", 1024: "purple"}

def _save_plots():
    if not rows:
        return
    df_live = pd.DataFrame(rows)
    n_plot  = df_live["n"].max()
    df_plot = df_live[df_live["n"] == n_plot]
    fig, axes = plt.subplots(3, len(b_values), figsize=(20, 15))
    fig.suptitle(f"SimHash + LSH results at n = {n_plot}", fontsize=14)
    for i, b in enumerate(b_values):
        df_b = df_plot[df_plot["b"] == b]
        ax_time, ax_quality, ax_cos_h = axes[0, i], axes[1, i], axes[2, i]
        for m in m_values:
            df_bm = df_b[df_b["m"] == m]
            ax_time.plot(df_bm["k"],    df_bm["time_sec"],
                         color=_plot_colors[m], marker="o", label=f"m={m}")
            ax_quality.plot(df_bm["k"], df_bm["mean_cosine"],
                            color=_plot_colors[m], marker="o", label=f"m={m}")
            ax_cos_h.plot(df_bm["k"],   df_bm["cosine_hamming"],
                          color=_plot_colors[m], marker="o", label=f"m={m}")
        ax_time.set_title(f"Time - b={b}");       ax_time.set_xlabel("k");    ax_time.set_ylabel("time (s)");                 ax_time.legend()
        ax_quality.set_title(f"Quality - b={b}"); ax_quality.set_xlabel("k"); ax_quality.set_ylabel("average cosine similarity"); ax_quality.legend()
        ax_cos_h.set_title(f"Quality - b={b}");   ax_cos_h.set_xlabel("k");   ax_cos_h.set_ylabel("average cos(h*pi/m)");      ax_cos_h.legend()
    plt.tight_layout()
    plt.savefig(f"sim_hash{SUFFIX}.png")
    plt.savefig(f"sim_hash{SUFFIX}_n{n_plot}.png")
    plt.close(fig)

m_max = max(m_values)
np.random.seed(42)
R_full = np.random.randn(m_max, d) 
R_full_bc = sc.broadcast(R_full)

for n_test in N_VALUES:
  print(f"\n=== Running n={n_test} ===")
  D = D_full[:n_test]
  n = D.shape[0]
  if D_bc is not None:
      D_bc.unpersist()
  D_bc = sc.broadcast(D)

  for m in m_values: 
    for b in b_values:
        # ricalcola R e signatures solo per ogni (b,m)
        # RDD di indici -> RDD di (doc_id, firma)
        indices_rdd    = sc.parallelize(range(n))
        signatures_rdd = indices_rdd.map(lambda i: compute_simhash(i,m)).cache()

        #  BANDED APPROACH 
        r = int(m // b)  # bit per band 
        # crea b coppie per ogni documento
        bands_rdd = signatures_rdd.flatMap(lambda i: make_bands(i, b, r))
        # raggruppa per (band_index, band_bits)
        grouped_rdd = bands_rdd.groupByKey()
        # genera coppie candidate
        candidates_rdd = grouped_rdd.flatMap(get_candidate_pairs)
        # rimuovi duplicati
        candidates_rdd = candidates_rdd.distinct()

        # broadcast dizionario firme per calcolo hamming
        signatures_dict    = dict(signatures_rdd.collect())
        signatures_dict_bc = sc.broadcast(signatures_dict)

        # calcola distanza di hamming per ogni coppia
        results_rdd = candidates_rdd.map(lambda i: compute_hamming(i,m,signatures_dict_bc))
        results_rdd = results_rdd.cache()

        for k in k_values:
            start = time.time()
            # filtra coppie simili
            similar_pairs_rdd = results_rdd.filter(lambda x: x[2] <= k)

            docs_rdd = similar_pairs_rdd.flatMap(lambda x: [x[0], x[1]]) \
                                   .distinct()
            num_docs = docs_rdd.count()

            mean_hamming = results_rdd.filter(lambda x: x[2] <= k) \
                           .map(lambda x: x[2]) \
                           .mean()
            
            

            cos_hamming_rdd = results_rdd.filter(lambda x: x[2] <= k) \
                                          .map(lambda x: math.cos(x[2] * math.pi / m))

            mean_cos_hamming = cos_hamming_rdd.mean()
            
            end = time.time()
            elapsed = end - start
            # calcola cosine similarity media sulle coppie simili
            cosine_rdd = (
                similar_pairs_rdd
                .map(lambda x: (x[0], x[1]))
                .map(compute_cosine_similarity)
                .cache()
            )

            count = cosine_rdd.count()

            if count > 0:
                mean_cosine = cosine_rdd.mean()
            else:
                mean_cosine = 0.0

            rows.append({
                "n":              n_test,
                "b":              b,
                "m":              m,
                "k":              k,
                "num_docs":       num_docs,
                "time_sec":       elapsed,
                "mean_cosine":    mean_cosine,
                "cosine_hamming": mean_cos_hamming,
            })

            # crash-safe partial save dopo ogni (b, m, k)
            pd.DataFrame(rows).to_csv(f"results_partial{SUFFIX}.csv", index=False)

  # crash-safe partial save dopo ogni n
  pd.DataFrame(rows).to_csv(f"results_partial{SUFFIX}.csv", index=False)
  _save_plots()

df = pd.DataFrame(rows)
df.to_csv(f"results_final{SUFFIX}.csv", index=False)
print(df)



# PLOT
# -------------------------------------------------------

colors = {64: "blue", 128: "green", 256: "orange", 512: "red", 1024: "purple"}

n_plot  = df["n"].max()
df_plot = df[df["n"] == n_plot]

fig, axes = plt.subplots(3, len(b_values), figsize=(20, 15))
fig.suptitle(f"SimHash + LSH results at n = {n_plot}", fontsize=14)

for i, b in enumerate(b_values):

    df_b = df_plot[df_plot["b"] == b]
    ax_time    = axes[0, i]  # prima riga: tempo
    ax_quality = axes[1, i]  # seconda riga: qualità
    ax_cos_h = axes[2,i] 

    for m in m_values:

        df_bm = df_b[df_b["m"] == m]
        # plot tempo
        ax_time.plot(
            df_bm["k"],
            df_bm["time_sec"],
            color=colors[m],
            marker="o",
            label=f"m={m}"
        )
        # plot qualità
        ax_quality.plot(
            df_bm["k"],
            df_bm["mean_cosine"],
            color=colors[m],
            marker="o",
            label=f"m={m}"
        )
        #plot cos (h*PI/m)
        ax_cos_h.plot(
            df_bm["k"],
            df_bm["cosine_hamming"],
            color=colors[m],
            marker="o",
            label=f"m={m}"
        )

    # titoli e label
    ax_time.set_title(f"Time - b={b}")
    ax_time.set_xlabel("k")
    ax_time.set_ylabel("tempo (s)")
    ax_time.legend()

    ax_quality.set_title(f"Quality - b={b}")
    ax_quality.set_xlabel("k")
    ax_quality.set_ylabel("average cosine similarity")
    ax_quality.legend()

    ax_cos_h.set_title(f"Quality - b={b}")
    ax_cos_h.set_xlabel("k")
    ax_cos_h.set_ylabel("average cos(h*pi/m)")
    ax_cos_h.legend()

plt.tight_layout()
plt.savefig(f"sim_hash{SUFFIX}.png")
plt.show()