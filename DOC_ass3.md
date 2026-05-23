# Documentazione di `ass3_23-05.py`

Script di benchmark per **SimHash + LSH a banding** sul dataset SIFT1M, eseguito con PySpark in locale (`local[*]`). Misura tempo e qualità (cosine medio, `cos(h·π/m)`) al variare di `n`, `m`, `b`, `k`, e produce CSV + grafici PNG.

---

## 1. Configurazione e setup

### Import e silenziamento log
- `warnings.filterwarnings("ignore")` e logger `py4j` / `pyspark` portati a `ERROR` per output pulito.
- `PYSPARK_PYTHON` e `PYSPARK_DRIVER_PYTHON` forzati a `sys.executable` → worker e driver usano lo **stesso interprete** del venv (evita mismatch versioni).

### `SparkContext`
- Creato in modalità `local[*]` (usa tutti i core disponibili).
- `spark.python.worker.faulthandler.enabled = true` → tracce Python utili in caso di segfault del worker.
- Log level `ERROR`.

### Variabili d'ambiente di override (per `run_all.py`)
| Var | Tipo | Default | Effetto |
|---|---|---|---|
| `BENCH_MEM_OPT` | `true`/`false` | `False` | Attiva i percorsi memory-optimized (FIX A, FIX B). |
| `BENCH_N_VALUES` | `"1000,10000,…"` | `[10k, 50k, 100k, 500k, 1M]` | Lista delle dimensioni di dataset da testare. |
| `BENCH_SUFFIX` | stringa | `""` | Suffisso aggiunto ai nomi dei file di output. |

---

## 2. Funzioni

### `load_fvecs(filename) -> np.ndarray`
Legge un file in formato `.fvecs` (formato SIFT/INRIA).

Layout binario: per ogni vettore, un `int32` con la dimensione `d`, seguito da `d` `float32`. La funzione:
1. Legge l'intero file come `int32`.
2. Inferisce `d` dal primo elemento e calcola `n` da `len/(d+1)`.
3. Fa il `reshape` in `(n, d+1)`.
4. Restituisce le colonne `[:, 1:]` reinterpretate come `float32` via `.view(np.float32)` (zero-copy).

Output: matrice `(n, d)` `float32` con i vettori SIFT. Nel codice `D_full` riceve l'intero dataset (1M × 128).

---

### `simhash_partition(iterator, m) -> iterator[(doc_id, signature)]`
Calcola le **firme SimHash** a `m` bit per tutti i documenti della partizione corrente.

Algoritmo (vettorizzato per partizione, non per elemento):
1. `R = R_full_bc.value[:m, :]` → matrice random `(m, d)` (proiezioni iperpiani), broadcast condivisa.
2. Raccoglie tutti gli `idx` della partizione in un array NumPy.
3. `block = D_local[idx]` → matrice `(batch, d)` dei vettori.
4. `bits = (block @ R.T) > 0` → matrice booleana `(batch, m)`: bit `i` = 1 se il vettore sta nel semispazio positivo dell'iperpiano `i`.
5. `packbits` comprime gli `m` bool in `m/8` byte; `.view(np.uint64)` reinterpreta in word da 64 bit.
6. Se `m == 64` restituisce `(doc_id, uint64)`; altrimenti `(doc_id, tuple_di_uint64)`.

**Perché vettorizzata**: invece di un loop Python `for bit in range(m)` per ogni documento (lento), una **singola moltiplicazione matrice-matrice** (BLAS, SIMD) per tutta la partizione. La permutazione introdotta da `packbits` è irrilevante: l'hamming distance è invariante per permutazione di bit e il banding è coerente se applicata a tutti i documenti.

---

### `emit_bands_partition(iterator, b, r) -> iterator[((band_idx, band_val), doc_id)]`
Genera le **chiavi di banding** per LSH (con `b` bande di `r = m/b` bit ciascuna).

1. Materializza la partizione in due array NumPy: `ids` (`int64`) e `sigs` (`uint64`).
2. Per ogni banda `band_idx ∈ [0, b)`:
   - `shift = band_idx · r`, `mask = (1 << r) − 1`.
   - `band_vals = (sigs >> shift) & mask` → estrae i `r` bit della banda in un colpo solo, su tutta la partizione (operazione bitwise vettorizzata).
   - Emette `((band_idx, int(band_val)), doc_id)` per ogni documento.

**Perché compatto**: la chiave è `(int, int)` (~28 byte serializzati) invece di `(int, tuple_di_r_bit)` (centinaia di byte). Riduce drasticamente lo shuffle del successivo `groupByKey`.

Vincolo: richiede `r ≤ 64` (la banda deve stare in un `uint64`).

---

### `get_candidate_pairs(item) -> list[(d1, d2)]`
Da un bucket LSH (`item = (key, iterable_of_doc_ids)`) produce le **coppie candidate** all'interno del bucket.

1. `doc_ids = list(set(item[1]))` → dedup intra-bucket (un doc può finire più volte nella stessa banda se la sua firma collide più volte… in realtà no per banda singola, ma `set` blinda contro duplicati upstream).
2. Filtro: se meno di 2 docs **o più di 300**, scarta. Il taglio a 300 è un **safety net anti-bucket-mostro**: un bucket con 1000 elementi genererebbe 500k coppie da solo.
3. `doc_ids.sort()` → tutte le coppie generate da `combinations` sono in forma canonica `(min, max)`, così il successivo `.distinct()` non vede `(a,b)` e `(b,a)` come distinte.

---

### `hamming_and_cosine_partition(iterator, sigs_bc) -> iterator[(d1, d2, h, cos)]`
**Versione default** (no mem opt): calcola distanza di Hamming sulle firme e cosine similarity sui vettori originali, per ogni coppia candidata.

Per ogni coppia `(d1, d2)`:
- `h = bin(s1 ^ s2).count("1")` → popcount XOR sulle firme uint64.
- `cos = dot(v1, v2) / (||v1|| · ||v2||)` con norme precalcolate (broadcast `norms_bc`).
- Yield `(d1, d2, h, cos)`.

**Pro**: semplice. **Contro**: emette tuple Python — ~150 byte per coppia. Con 50M coppie → ~7.5 GB sul driver dopo `collect()`. Usare la variante NumPy per dataset grandi.

---

### `hamming_and_cosine_partition_np(iterator, sigs_bc) -> iterator[np.ndarray]`
**Versione memory-optimized** (FIX B): stessa logica, ma emette **un singolo array NumPy strutturato per partizione**.

1. Materializza la partizione in due array `d1`, `d2` (`int64`).
2. Recupera firme `s1`, `s2` via fancy indexing (veloce se `sigs_local` è già un `ndarray`, vedi FIX A; con `dict` fa fallback a list comprehension).
3. `xor = s1 ^ s2`, poi popcount riga per riga con `bin(int(x)).count("1")` (NumPy non ha popcount nativo per uint64).
4. `dots = np.einsum("ij,ij->i", v1, v2)` → prodotti scalari batch.
5. `cos = dots / (norms[d1] · norms[d2])` in `float32`.
6. Costruisce un array strutturato con dtype `[("d1","i8"),("d2","i8"),("h","i4"),("cos","f4")]` (28 byte/riga) e lo emette come **singolo elemento**.

**Vantaggi**:
- Memoria driver: ~1.5 GB invece di ~7.5 GB a 50M coppie.
- I prodotti scalari sono BLAS-vettorizzati invece di un loop Python.
- La lista-di-tuple-Python non esiste mai.

---

## 3. Loop principale

Struttura (annidata da fuori a dentro):
```
for m in m_values:           # 1 valore: m=64
  for n_test in N_VALUES:    # dimensioni dataset
    # broadcast D, norms
    # calcola firme (dipendono solo da m, riusate per ogni b)
    for b in b_values:       # b=4, 8
      # banding -> groupByKey -> coppie candidate -> distinct
      # 1 sola pass Spark per (h, cos) di tutte le coppie
      for k in k_values:     # k=0, 2, 4, 8
        # filtro NumPy (in driver) + statistiche locali
```

**Inversione `m` esterno, `b` interno** (MOD 7): le firme dipendono solo da `m`, quindi calcolarle una volta per `m` e riutilizzarle per tutti i `b` dimezza il lavoro (con 2 valori di `b`).

### Gestione broadcast
- `D_bc` e `norms_bc` rilasciati con `.unpersist()` e rifatti a ogni `n_test`.
- `sigs_bc` rilasciato a fine ciclo `n_test`.

### Storage firme
- Default: `dict[doc_id -> uint64]` (~120 byte/entry → 120 MB a 1M docs).
- FIX A (`USE_MEMORY_OPTIMIZATIONS=True`): `np.ndarray(n, dtype=uint64)` (~8 MB a 1M docs, 15× meno).

### Calcolo statistiche per k
Sul driver, in NumPy:
- `mask = arr["h"] <= k`, `sub = arr[mask]`.
- `num_docs = len(unique(concatenate([sub.d1, sub.d2])))`.
- `mean_cosine = sub["cos"].mean()`.
- `mean_cos_hamming = np.cos(sub.h · π / m).mean()` → stima trigonometrica della cosine dal solo numero di bit diversi.
- Il tempo Spark `t_compute` viene **diviso uniformemente** tra i `k` valori per avere una colonna `time_sec` comparabile al codice originale.

### Output incrementali
- Dopo ogni `n_test` salva `results_partial<SUFFIX>.csv` (crash-safe: se la run grande va in OOM hai comunque le `n` piccole).
- A fine loop salva `results_final<SUFFIX>.csv`.

---

## 4. Grafici

### `sim_hash<SUFFIX>.png` (subplot 3 × len(b_values))
Filtra le righe sulla `n` massima (`n_plot = df.n.max()`) e per ogni `b` produce 3 plot in funzione di `k`:
1. **Tempo** (`time_sec`).
2. **Qualità** (`mean_cosine`).
3. **Qualità via Hamming** (`cos(h·π/m)`).
Una curva per ogni valore di `m`.

### `scalability<SUFFIX>.png`
Log-log di `time_sec` vs `n` a `k = k_repr` (il `k` mediano), una curva per `b`. Aggiunge due rette di riferimento:
- `O(n)` (slope 1)
- `O(n²)` (slope 2)

Per stimare l'esponente empirico:
```python
slope, _ = np.polyfit(np.log(df_b["n"]), np.log(df_b["time_sec"]), 1)
```
Un LSH ben tarato dà slope ben sotto 2.

---

## 5. Parametri e assunzioni

| Parametro | Valore | Vincolo |
|---|---|---|
| `m_values` | `[64]` | multiplo di 64 (firma sta in `uint64`) |
| `k_values` | `[0, 2, 4, 8]` | soglie Hamming |
| `b_values` | `[4, 8]` | `r = m/b` deve dividere 64 |
| `PARTS` | `defaultParallelism × 4` | bilancia overhead scheduling |
| seed | 42 | riproducibilità di `R_full` |

---

## 6. Output

| File | Contenuto |
|---|---|
| `results_partial<SUFFIX>.csv` | Aggiornato dopo ogni `n` (crash-safe). |
| `results_final<SUFFIX>.csv` | Risultato finale. |
| `sim_hash<SUFFIX>.png` | Tempo/qualità vs `k` alla `n` massima. |
| `scalability<SUFFIX>.png` | Tempo vs `n` (log-log), curva chiave per il report. |

Colonne del CSV: `n, b, m, k, num_docs, time_sec, mean_cosine, cosine_hamming`.
