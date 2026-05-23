import numpy as np
import math
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

# ===============================================================
# HOW TO RUN / TESTING WORKFLOW
# ===============================================================
# This script benchmarks SimHash + LSH banding on the SIFT1M dataset
# across several dataset sizes (N_VALUES, defined below).
#
# Recommended order of runs:
#
# 1) QUICK SANITY CHECK (~ 30-60 seconds total).
#    Verify the pipeline runs end-to-end without errors and produces
#    a non-empty DataFrame and the PNG plots.
#       N_VALUES = [1_000, 10_000]
#       USE_MEMORY_OPTIMIZATIONS = False
#
# 2) MEDIUM SCALABILITY RUN (~ 5-15 minutes).
#    Useful to inspect the shape of the time-vs-n curve before
#    committing to the full run.
#       N_VALUES = [10_000, 50_000, 100_000]
#       USE_MEMORY_OPTIMIZATIONS = False
#
# 3) FULL REPORT RUN (~ 20-60 minutes on a modern laptop).
#    This produces the numbers and plots that go into the report.
#    Memory optimizations are required for the 1M point to fit
#    comfortably in RAM.
#       N_VALUES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
#       USE_MEMORY_OPTIMIZATIONS = True
#
# Outputs of any run:
#   - results_partial.csv  : updated after EVERY n value (crash-safe)
#   - results_final.csv    : written once the full loop completes
#   - sim_hash.png         : per-b plots of time/quality/cos(h*pi/m),
#                            filtered on the largest n
#   - scalability.png      : log-log time vs n, the key plot for the
#                            report (one curve per b, plus O(n) and
#                            O(n^2) reference lines)
#
# Tips:
#   - If the 1M run hits OOM, drop the last entry of N_VALUES; the
#     remaining four points are already enough for a clean log-log fit.
#   - Watch the Spark UI at http://localhost:4040 while the job runs:
#     screenshots of the DAG and stage breakdown are nice to include
#     in the report.
#   - To estimate the empirical complexity exponent from the scalability
#     curve, fit a line in log-log:
#         slope, _ = np.polyfit(np.log(df_b["n"]), np.log(df_b["time_sec"]), 1)
#     A slope near 1 means linear, near 2 means quadratic.
# ===============================================================

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ===============================================================
# [LOG] LOGGING + TIMING INFRASTRUCTURE.
#
# - All [profile ...] / [config] / etc. lines go BOTH to stdout AND to
#   a per-run log file under logs/. So you can scroll the console for
#   a quick look and grep the log file later.
# - A second CSV file (profile_timings<suffix>.csv) accumulates one
#   row per (n, b, phase) pair: easy to load with pandas and plot
#   afterwards.
# - Spark event log is enabled and written under logs/spark-events:
#   you can replay it with the Spark History Server if you want a
#   per-stage breakdown.
# ===============================================================
import csv
import datetime as _dt
import atexit as _atexit

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR  = os.path.join(_HERE, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_SPARK_EVT_DIR = os.path.join(_LOG_DIR, "spark-events")
os.makedirs(_SPARK_EVT_DIR, exist_ok=True)
_RUN_STAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_SUFFIX_FOR_LOG = os.environ.get("BENCH_SUFFIX", "")
_LOG_PATH = os.path.join(_LOG_DIR,
    f"ass3_23-05{_SUFFIX_FOR_LOG}_{_RUN_STAMP}.log")
_TIMINGS_CSV = os.path.join(_HERE,
    f"profile_timings{_SUFFIX_FOR_LOG}.csv")

logger = logging.getLogger("bench")
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
_fh  = logging.FileHandler(_LOG_PATH, encoding="utf-8")
_fh.setFormatter(_fmt)
_sh  = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)

_TIMINGS_FP = open(_TIMINGS_CSV, "w", encoding="utf-8", newline="")
_TIMINGS_W  = csv.writer(_TIMINGS_FP)
_TIMINGS_W.writerow(["timestamp", "n", "b", "phase", "seconds", "extra"])
_TIMINGS_FP.flush()
_atexit.register(lambda: _TIMINGS_FP.close())

def log_phase(n, b, phase, seconds, extra=""):
    """Print one timing line and append it to the CSV."""
    bstr = "-" if b is None else str(b)
    logger.info(f"[profile n={n} b={bstr}] {phase}={seconds:.3f}s  {extra}".rstrip())
    _TIMINGS_W.writerow([_dt.datetime.now().isoformat(timespec="seconds"),
                         n, bstr, phase, f"{seconds:.6f}", extra])
    _TIMINGS_FP.flush()

logger.info(f"[log] log file       = {_LOG_PATH}")
logger.info(f"[log] timings CSV    = {_TIMINGS_CSV}")
logger.info(f"[log] spark events   = {_SPARK_EVT_DIR}")

_T_SCRIPT_START = time.time()

# [PERF] Spark configuration tuning.
# Setting driver/executor memory and a few perf-related flags HERE (via
# SparkConf) is more robust than relying on PYSPARK_DRIVER_MEMORY env
# vars set outside the script: SparkConf is honored even when the JVM
# is launched directly by PySpark (no spark-submit).
#
# Memory: defaults are 1 GB driver / 1 GB executor, which is tight for
# the n=1M run. With 64 GB on this machine we can afford 8 GB each.
# Tune down to 4g/4g if you run on a smaller laptop.
from pyspark import SparkContext, SparkConf

# Spark wants a file:/// URI for the event log dir on Windows.
_evt_uri = "file:///" + _SPARK_EVT_DIR.replace("\\", "/")

conf = (
    SparkConf()
        .setAppName("SimHash")
        .setMaster("local[*]")
        .set("spark.driver.memory",   "8g")
        .set("spark.executor.memory", "8g")
        # KryoSerializer is faster than the default Java serializer for
        # the kind of small Python objects we shuffle (band keys, pairs).
        .set("spark.serializer",
             "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer.max", "512m")
        # Default shuffle partitions for Spark SQL (we don't use SQL much,
        # but cap it at the core count to match our RDD partitioning rule).
        .set("spark.sql.shuffle.partitions", str(os.cpu_count() or 4))
        # On local[*] data is always local; don't wait for locality hints.
        .set("spark.locality.wait", "0s")
        # Reuse Python worker processes across tasks: avoids the cost of
        # spawning a fresh worker for every short task.
        .set("spark.python.worker.reuse", "true")
        # Compress shuffle and spilled blocks: less disk I/O at the cost
        # of a bit of CPU. On Windows where disk I/O is more expensive
        # (winutils, antivirus) this is a clear win.
        .set("spark.shuffle.compress",       "true")
        .set("spark.shuffle.spill.compress", "true")
        .set("spark.rdd.compress",           "true")
        # [LOG] event log: replayable with the Spark History Server.
        .set("spark.eventLog.enabled", "true")
        .set("spark.eventLog.dir",     _evt_uri)
)
_t_spark = time.time()
sc = SparkContext(conf=conf)
sc._conf.set("spark.python.worker.faulthandler.enabled", "true")
sc.setLogLevel("ERROR")
log_phase("-", None, "spark_init", time.time() - _t_spark,
          extra=f"defaultParallelism={sc.defaultParallelism}")

# ===============================================================
# CONFIG FLAG
# ===============================================================
# Set to True to use the memory-optimized paths (recommended for n >= ~500k).
# Set to False to use the simpler default paths (fine for n up to ~100k).
# Both versions live in the same file so you can compare them.
#
# [AUTOMATION] Defaults below can be overridden via environment variables
# so the script can be driven by run_all.py without editing this file:
#   BENCH_MEM_OPT  = "true" / "false"
#   BENCH_N_VALUES = comma-separated ints, e.g. "1000,10000,100000"
#   BENCH_SUFFIX   = string appended to output filenames, e.g. "_sanity"
# If an env var is missing, the hard-coded default is used.
USE_MEMORY_OPTIMIZATIONS = False

_env_mem = os.environ.get("BENCH_MEM_OPT")
if _env_mem is not None:
    USE_MEMORY_OPTIMIZATIONS = _env_mem.strip().lower() in ("1", "true", "yes")

SUFFIX = os.environ.get("BENCH_SUFFIX", "")

# ===============================================================
# LOAD DATA
# ===============================================================

def load_fvecs(filename):
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    n = len(data) // (d + 1)
    data = data.reshape(n, d + 1)
    return data[:, 1:].view(np.float32)

_t_load = time.time()
D_full = load_fvecs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sift", "sift_base.fvecs"))
d = D_full.shape[1]
log_phase("-", None, "dataset_load", time.time() - _t_load,
          extra=f"shape={D_full.shape}")

# [SCALABILITY] List of dataset sizes to benchmark.
# Start small to verify correctness, then scale up. The final report plot
# shows time vs n in log-log scale (expect roughly linear-to-sub-quadratic
# slope for a well-tuned LSH; pure brute force would be quadratic).
N_VALUES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# [AUTOMATION] override from env var if provided (used by run_all.py)
_env_n = os.environ.get("BENCH_N_VALUES")
if _env_n is not None:
    N_VALUES = [int(x.strip()) for x in _env_n.split(",") if x.strip()]

logger.info(f"[config] USE_MEMORY_OPTIMIZATIONS={USE_MEMORY_OPTIMIZATIONS}")
logger.info(f"[config] N_VALUES={N_VALUES}")
logger.info(f"[config] SUFFIX={SUFFIX!r}")

# D_bc and norms_bc are (re)assigned inside the n loop below.
D_bc     = None
norms_bc = None

# ===============================================================
# PARAMETERS
# ===============================================================

m_values = [64]            # ASSUMPTION: m is a multiple of 64 (signature -> uint64)
k_values = [0, 2, 4, 8]
b_values = [4, 8]          # ASSUMPTION: r = m/b divides 64 (band fits in a word)
rows = []

m_max = max(m_values)
np.random.seed(42)
R_full = np.random.randn(m_max, d).astype(np.float32)
R_full_bc = sc.broadcast(R_full)

# [MOD 2] PARTITION COUNT SCALED TO DATASET SIZE.
# Before: PARTS = sc.defaultParallelism * 4 (e.g. 128) fixed for every n.
# With n=1000 that means ~8 elements/partition: each task spends ~1.5 s
# on Python-worker overhead (Py4J, serialization, worker boot) to do a
# few ms of real work. collect() of a 8 KB result took 2.4 min just
# because it was 128 tasks.
# Now: aim for ~5000 elements per partition, capped between 2 and
# defaultParallelism (no point creating more tasks than cores when each
# task is dense). On big n we still saturate the cores; on small n we
# pay only a handful of task overheads.
_TARGET_PER_PART = 5000

def parts_for(n):
    return max(2, min(sc.defaultParallelism, n // _TARGET_PER_PART + 1))

# ===============================================================
# [MOD 3] VECTORIZED SIMHASH PER PARTITION.
#
# Before: a per-element map with a Python for-loop over the 64 bits of
# the signature. Per document this paid:
#   - one broadcast access
#   - one R[:m] slicing
#   - 64 Python iterations to set the bits
#
# Now: each partition gathers its indices into a NumPy array and does
# ONE matrix multiplication (BLAS, SIMD vectorized) for the whole
# partition. Then packbits compresses the boolean bits into bytes, and
# a view reinterprets them as uint64: a 64-bit signature becomes ONE
# 64-bit integer (8 bytes) instead of a NumPy array.
#
# NOTE: packbits + view permutes the bits relative to the "logical"
# signature numbering. This does not matter for Hamming distance
# (XOR + popcount is invariant under permutation) and it does not
# matter for banding as long as all documents undergo the SAME
# permutation (band collisions are preserved).
# ===============================================================

def simhash_partition(iterator, m):
    R       = R_full_bc.value[:m, :]            # (m, d)
    D_local = D_bc.value                        # (n, d)
    idx     = np.fromiter(iterator, dtype=np.int64)
    if idx.size == 0:
        return iter([])
    block  = D_local[idx]                       # (batch, d)
    bits   = (block @ R.T) > 0                  # (batch, m) bool
    packed = np.packbits(bits.astype(np.uint8), axis=1, bitorder="big")
    num_words = m // 64
    sigs_u64  = packed.view(np.uint64).reshape(-1, num_words)
    if num_words == 1:
        return zip(idx.tolist(), sigs_u64[:, 0].tolist())
    else:
        return zip(idx.tolist(), [tuple(row) for row in sigs_u64.tolist()])

# ===============================================================
# [MOD 4] VECTORIZED BANDING + COMPACT KEYS.
#
# Before: make_bands emitted, for each document, b keys of the form
# (band_index, tuple_of_r_ints). The bit tuple is heavy and bloats the
# groupByKey shuffle.
#
# Now: the signature is a uint64; each band is extracted with shift+mask
# (a single machine instruction). The band key becomes
# (band_index, integer), much smaller to serialize.
# ===============================================================

def emit_bands_partition(iterator, b, r):
    items = list(iterator)
    if not items:
        return iter([])
    ids  = np.array([x[0] for x in items], dtype=np.int64)
    sigs = np.array([x[1] for x in items], dtype=np.uint64)   # (batch,)
    mask = np.uint64((1 << r) - 1)
    out = []
    for band_idx in range(b):
        shift     = np.uint64(band_idx * r)
        band_vals = (sigs >> shift) & mask                    # (batch,) uint64
        for did, bv in zip(ids.tolist(), band_vals.tolist()):
            out.append(((band_idx, int(bv)), did))
    return iter(out)

# ===============================================================
# [MOD 5] CANDIDATE PAIRS with intra-bucket dedup and canonical pairs.
#
# Before: combinations could be called on a list with duplicates;
# non-canonical pairs; the final distinct had to deal with (a,b)
# vs (b,a) as well.
#
# Now: set() first, sort after: combinations yields only (min, max).
# The final distinct does less work.
# ===============================================================

# Hard cap on bucket size: drops random/dense bands and keeps shuffle
# bounded. Must match MAX_GROUP used by aggregateByKey below.
MAX_GROUP = 300

def _seq_op(acc, doc_id):
    if len(acc) < MAX_GROUP:
        acc.append(doc_id)
    return acc

def _comb_op(a, b):
    if len(a) >= MAX_GROUP:
        return a
    a.extend(b[:MAX_GROUP - len(a)])
    return a

def get_candidate_pairs(item):
    doc_ids = list(set(item[1]))             # intra-bucket dedup
    if len(doc_ids) < 2 or len(doc_ids) > MAX_GROUP:
        return []
    doc_ids.sort()                           # canonical pairs (min,max)
    return list(combinations(doc_ids, 2))

# ===============================================================
# [MOD 6] HAMMING + COSINE IN A SINGLE PASS (default version).
#
# Before: for each k you triggered 4-5 separate Spark actions
# (count, mean, mean, count, mean), each rebuilding the DAG from
# results_rdd. Also, Hamming used a Python for-loop over 64 bits.
#
# Now: one Spark pass computes (h, cos) for every candidate pair.
# Hamming = popcount of XOR between two uint64 (one XOR + bin().count('1');
# on Python 3.10+ int.bit_count() would be even faster). Cosine = dot
# product of vectors with pre-computed norms. The result is collect()ed
# to the driver and all per-k statistics are computed locally in NumPy.
#
# WARNING: this version returns a Python list of tuples from collect().
# Each tuple is ~150 bytes of Python overhead. For ~50M pairs that means
# ~7.5 GB on the driver. Switch USE_MEMORY_OPTIMIZATIONS = True to use
# the NumPy-batched variant below.
# ===============================================================

def hamming_and_cosine_partition(iterator, sigs_bc):
    D_local     = D_bc.value
    norms_local = norms_bc.value
    sigs_local  = sigs_bc.value
    out = []
    for (d1, d2) in iterator:
        s1 = sigs_local[d1]
        s2 = sigs_local[d2]
        h  = bin(int(s1) ^ int(s2)).count("1")
        v1 = D_local[d1]; v2 = D_local[d2]
        cos = float(np.dot(v1, v2) / (norms_local[d1] * norms_local[d2]))
        out.append((d1, d2, h, cos))
    return iter(out)

# ===============================================================
# [FIX B - improvement for memory] HAMMING + COSINE, NUMPY-BATCHED.
#
# Same logic as the function above, but each partition emits ONE
# structured NumPy array instead of yielding individual tuples.
# Two big wins:
#   1) the "Python list of tuples" representation never exists, so
#      driver memory drops from ~7.5 GB to ~1.5 GB at 50M pairs;
#   2) Hamming and cosine are computed vectorized over the partition
#      (einsum for dot products, vectorized XOR), faster than the
#      per-pair Python loop.
#
# Note on popcount: NumPy has no native popcount for uint64. We still
# use bin().count('1') on the XOR result, but applied via a list
# comprehension to a vector that has been computed in one shot. If
# bottlenecked, one can use np.unpackbits + sum, or gmpy2.popcount.
# ===============================================================

def hamming_and_cosine_partition_np(iterator, sigs_bc):
    pairs = list(iterator)
    if not pairs:
        return iter([])
    sigs_local  = sigs_bc.value
    D_local     = D_bc.value
    norms_local = norms_bc.value

    d1 = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    d2 = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))

    # signatures can be either a dict (default) or an ndarray (fix A):
    # both support fancy indexing, but the ndarray case is much faster.
    s1 = sigs_local[d1] if isinstance(sigs_local, np.ndarray) else np.array(
        [sigs_local[i] for i in d1.tolist()], dtype=np.uint64)
    s2 = sigs_local[d2] if isinstance(sigs_local, np.ndarray) else np.array(
        [sigs_local[i] for i in d2.tolist()], dtype=np.uint64)

    xor = s1 ^ s2
    # popcount over a vector of uint64
    h = np.fromiter((bin(int(x)).count("1") for x in xor),
                    dtype=np.int32, count=len(xor))

    v1 = D_local[d1]
    v2 = D_local[d2]
    dots = np.einsum("ij,ij->i", v1, v2)
    cos  = (dots / (norms_local[d1] * norms_local[d2])).astype(np.float32)

    dt = [("d1", "i8"), ("d2", "i8"), ("h", "i4"), ("cos", "f4")]
    out = np.empty(len(pairs), dtype=dt)
    out["d1"] = d1; out["d2"] = d2; out["h"] = h; out["cos"] = cos
    # yield a single element per partition: the whole array
    return iter([out])

# ===============================================================
# MAIN LOOP
#
# [MOD 7] LOOP INVERTED: m OUTER, b INNER.
# Before: outer was for m, for b, and signatures were recomputed
# inside the b loop. But signatures depend ONLY on m. Computing them
# once per m and reusing across b halves the work (with two b values:
# 2x; with more b values the win scales).
# ===============================================================

for m in m_values:

    # ===============================================================
    # [SCALABILITY] OUTER LOOP ON DATASET SIZE.
    # The whole pipeline (broadcast of D, signatures, banding, distance)
    # is re-run for each n in N_VALUES. Broadcasts of D and norms are
    # rebuilt every iteration because the data slice changes; signatures
    # are anyway recomputed because they depend on D.
    # ===============================================================
    for n_test in N_VALUES:
        logger.info(f"=== Running n={n_test}, m={m} ===")
        # [PROFILE] per-phase timing block
        _phase = {}
        _t = time.time()
        _t_n_start = _t

        D = D_full[:n_test]
        n = D.shape[0]
        parts = parts_for(n)
        logger.info(f"    parts={parts} (~{n // parts} elements/partition)")

        # rebuild D and norms broadcasts for this n
        if D_bc is not None:
            D_bc.unpersist()
        if norms_bc is not None:
            norms_bc.unpersist()
        norms    = np.linalg.norm(D, axis=1).astype(np.float32)
        D_bc     = sc.broadcast(D)
        norms_bc = sc.broadcast(norms)
        _phase["broadcast_D"] = time.time() - _t; _t = time.time()

        # signatures: computed once per (m, n)
        indices_rdd = sc.parallelize(range(n), numSlices=parts)
        sig_rdd     = indices_rdd.mapPartitions(lambda it, mm=m: simhash_partition(it, mm))

        # [MOD 8] Collect + broadcast of signatures.
        sig_pairs = sig_rdd.collect()
        _phase["simhash"] = time.time() - _t; _t = time.time()

        if USE_MEMORY_OPTIMIZATIONS:
            # [FIX A - improvement for memory] STORE SIGNATURES AS NUMPY ARRAY.
            # The Python dict version costs ~120 bytes per entry (hash table,
            # boxed ints). For n = 1M that is ~120 MB on the driver, replicated
            # on every Python worker after broadcast. A flat uint64 ndarray is
            # 8 bytes per entry: 8 MB total, 15x less. Indexing semantics
            # (sigs[doc_id]) are identical.
            sigs_store = np.zeros(n, dtype=np.uint64)
            for did, s in sig_pairs:
                sigs_store[did] = s
        else:
            # default: Python dict (simple, but heavy at scale)
            sigs_store = dict(sig_pairs)

        del sig_pairs
        sigs_bc = sc.broadcast(sigs_store)
        _phase["broadcast_sigs"] = time.time() - _t

        log_phase(n_test, None, "broadcast_D",    _phase["broadcast_D"])
        log_phase(n_test, None, "simhash",        _phase["simhash"])
        log_phase(n_test, None, "broadcast_sigs", _phase["broadcast_sigs"])

        for b in b_values:
            r = m // b
            _tb = time.time()

            # rebuild RDD from the in-driver signatures: emit_bands_partition
            # expects (doc_id, signature) tuples.
            if isinstance(sigs_store, np.ndarray):
                sig_items = list(zip(range(n), sigs_store.tolist()))
            else:
                sig_items = list(sigs_store.items())
            sig_items_rdd = sc.parallelize(sig_items, numSlices=parts)

            # banding (compact keys)
            bands_rdd = sig_items_rdd.mapPartitions(
                lambda it, bb=b, rr=r: emit_bands_partition(it, bb, rr)
            )

            # group by (band_index, band_value) with a cap on bucket size:
            # aggregateByKey trims oversized buckets during shuffle instead
            # of materializing the full list (lighter shuffle than groupByKey).
            grouped_rdd = bands_rdd.aggregateByKey(
                [], _seq_op, _comb_op, numPartitions=parts
            )

            # candidate pairs, global dedup
            candidates_rdd = (
                grouped_rdd.flatMap(get_candidate_pairs)
                           .distinct(numPartitions=parts)
            )
            # [PROFILE] force candidate materialization to time it separately
            n_candidates = candidates_rdd.cache().count()
            _t_cand = time.time() - _tb
            _tb = time.time()

            # [MOD 9 / FIX B switch] one Spark pass to build (d1, d2, h, cos).
            t_start = time.time()

            if USE_MEMORY_OPTIMIZATIONS:
                # FIX B: each partition returns ONE NumPy structured array.
                # collect() then returns a list of arrays (one per partition)
                # which we concatenate. Peak memory is dominated by the final
                # array, not by an intermediate Python-list-of-tuples.
                dt = [("d1", "i8"), ("d2", "i8"), ("h", "i4"), ("cos", "f4")]
                chunks = (
                    candidates_rdd
                      .mapPartitions(lambda it: hamming_and_cosine_partition_np(it, sigs_bc))
                      .collect()
                )
                arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=dt)
                del chunks
            else:
                # default: list of tuples, then convert to NumPy structured array
                pairs_list = (
                    candidates_rdd
                      .mapPartitions(lambda it: hamming_and_cosine_partition(it, sigs_bc))
                      .collect()
                )
                dt = [("d1", "i8"), ("d2", "i8"), ("h", "i4"), ("cos", "f4")]
                arr = np.array(pairs_list, dtype=dt) if pairs_list else np.empty(0, dtype=dt)
                del pairs_list

            t_compute = time.time() - t_start

            log_phase(n_test, b, "banding_candidates", _t_cand,
                      extra=f"pairs={n_candidates}")
            log_phase(n_test, b, "compute", t_compute,
                      extra=f"pairs={n_candidates}")

            candidates_rdd.unpersist()

            # [MOD 10] PER-k STATISTICS ON THE DRIVER (NumPy).
            # Before: each k = 4-5 Spark actions (count, mean, ...) each with
            # its own scheduling overhead. Now: NumPy filter + NumPy mean,
            # microseconds.
            for k in k_values:
                t0   = time.time()
                mask = arr["h"] <= k
                sub  = arr[mask]
                if len(sub) > 0:
                    num_docs         = len(np.unique(np.concatenate([sub["d1"], sub["d2"]])))
                    mean_cosine      = float(sub["cos"].mean())
                    mean_cos_hamming = float(np.cos(sub["h"].astype(np.float64) * np.pi / m).mean())
                else:
                    num_docs         = 0
                    mean_cosine      = 0.0
                    mean_cos_hamming = 0.0
                t1 = time.time()

                # Spark cost was paid ONCE per (m,b); we spread it uniformly
                # over the k values so the "per-k time" column is comparable
                # to the original script.
                elapsed = (t_compute / len(k_values)) + (t1 - t0)

                rows.append({
                    # [SCALABILITY] add n to the row so we can group/plot by n
                    "n":              n_test,
                    "b":              b,
                    "m":              m,
                    "k":              k,
                    "num_docs":       num_docs,
                    "time_sec":       elapsed,
                    "mean_cosine":    mean_cosine,
                    # [MOD 11] column name with underscore (see plot too).
                    # Before it was "cosine hamming" (with space) on write but
                    # "cosine_hamming" (with underscore) on read -> KeyError.
                    "cosine_hamming": mean_cos_hamming,
                })

        # [SCALABILITY] release the per-(m,n) signatures broadcast
        sigs_bc.unpersist()

        # [LOG] total wall-clock time for this n
        log_phase(n_test, None, "total_n", time.time() - _t_n_start)

        # [SCALABILITY] save partial CSV after every n: if the script crashes
        # later (OOM on the big run), you still have the smaller-n results.
        pd.DataFrame(rows).to_csv(f"results_partial{SUFFIX}.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(f"results_final{SUFFIX}.csv", index=False)
print(df)
log_phase("-", None, "script_total", time.time() - _T_SCRIPT_START)

# ===============================================================
# PLOT
# [MOD 12] subplots(3, ...) instead of subplots(2, ...).
# Before: 2 rows were declared but 3 were used (ax_cos_h = axes[2,i])
# -> IndexError on first access. Now rows are consistent.
#
# [SCALABILITY] The per-(b,k) plots below now filter on the largest n
# (n_plot), so they show the "final" results. Mixing rows from different
# n in the same plot would be meaningless.
# ===============================================================

colors = {64: "blue", 128: "green", 256: "orange", 512: "red", 1024: "purple"}

n_plot   = df["n"].max()
df_plot  = df[df["n"] == n_plot]

fig, axes = plt.subplots(3, len(b_values), figsize=(20, 15))
fig.suptitle(f"SimHash + LSH results at n = {n_plot}", fontsize=14)

for i, b in enumerate(b_values):
    df_b = df_plot[df_plot["b"] == b]
    ax_time    = axes[0, i]
    ax_quality = axes[1, i]
    ax_cos_h   = axes[2, i]

    for m in m_values:
        df_bm = df_b[df_b["m"] == m]
        ax_time.plot(df_bm["k"],    df_bm["time_sec"],
                     color=colors[m], marker="o", label=f"m={m}")
        ax_quality.plot(df_bm["k"], df_bm["mean_cosine"],
                        color=colors[m], marker="o", label=f"m={m}")
        ax_cos_h.plot(df_bm["k"],   df_bm["cosine_hamming"],
                      color=colors[m], marker="o", label=f"m={m}")

    ax_time.set_title(f"Time - b={b}")
    ax_time.set_xlabel("k"); ax_time.set_ylabel("time (s)")
    ax_time.legend()

    ax_quality.set_title(f"Quality - b={b}")
    ax_quality.set_xlabel("k"); ax_quality.set_ylabel("average cosine similarity")
    ax_quality.legend()

    ax_cos_h.set_title(f"Quality - b={b}")
    ax_cos_h.set_xlabel("k"); ax_cos_h.set_ylabel("average cos(h*pi/m)")
    ax_cos_h.legend()

plt.tight_layout()
plt.savefig(f"sim_hash{SUFFIX}.png")
plt.show()

# ===============================================================
# [SCALABILITY] SCALABILITY PLOT: time vs n in log-log.
# For each (b, k) pair we plot how the running time grows with the
# dataset size. A straight line in log-log scale means a polynomial
# relationship: time ~ n^a, where a is the slope. A well-tuned LSH
# should give slopes well below 2 (brute force baseline).
# We average over k_values for a cleaner picture; one curve per b.
# ===============================================================

fig2, ax = plt.subplots(1, 1, figsize=(10, 7))

# pick a representative k for the curve: the middle one (more pairs
# than k=0, less than the most permissive). Change to suit the report.
k_repr = k_values[len(k_values) // 2]
df_scal = df[df["k"] == k_repr]

for b in b_values:
    df_b = df_scal[df_scal["b"] == b].sort_values("n")
    if len(df_b) == 0:
        continue
    ax.loglog(df_b["n"], df_b["time_sec"], marker="o", label=f"b={b}")

# reference slopes: linear (n) and quadratic (n^2)
if len(N_VALUES) >= 2:
    n_ref = np.array(sorted(N_VALUES), dtype=float)
    t0    = df_scal["time_sec"].min() if len(df_scal) else 1.0
    ax.loglog(n_ref, t0 * (n_ref / n_ref[0]),
              linestyle="--", color="gray", alpha=0.6, label="O(n) reference")
    ax.loglog(n_ref, t0 * (n_ref / n_ref[0]) ** 2,
              linestyle=":",  color="gray", alpha=0.6, label="O(n^2) reference")

ax.set_xlabel("n (number of documents)")
ax.set_ylabel("time (s)")
ax.set_title(f"Scalability: time vs dataset size  (m={m_values[0]}, k={k_repr})")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(f"scalability{SUFFIX}.png")
plt.show()
