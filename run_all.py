"""
run_all.py
==========
Driver script: runs ass3_23-05.py three times in a row, with different
presets, on a single laptop. Each preset runs as a fresh Python/Spark
subprocess so the JVM and broadcast caches start clean every time.

Behavior:
  - Presets are executed in INCREASING size order (sanity -> medium -> full).
    Rationale: if something is broken, you see it within ~1 minute on
    the sanity run, before wasting time on the heavy ones.
  - FAIL-FAST: if a preset crashes (non-zero exit code), the driver
    stops immediately and surfaces the error. Subsequent presets are
    NOT executed. Results from earlier presets are kept on disk.

Each preset produces its own set of output files, distinguished by the
SUFFIX appended to filenames:
    results_final_<suffix>.csv
    results_partial_<suffix>.csv
    sim_hash_<suffix>.png
    scalability_<suffix>.png

Usage:
    python run_all.py

Edit SCRIPT_PATH and PRESETS below if you want to change paths or
configurations.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DRIVER_LOG = os.path.join(LOG_DIR, f"run_all_{RUN_STAMP}.log")

# Tag for log/output filenames based on selected worker (set later in main)
WORKER_TAG = ""


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


_driver_log_fp = open(DRIVER_LOG, "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _driver_log_fp)
sys.stderr = _Tee(sys.__stderr__, _driver_log_fp)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

# Worker scripts available. Selected via CLI:
#   python run_all.py            -> default "23"
#   python run_all.py 23         -> ass3_23-05.py
#   python run_all.py ruben      -> ass3_ruben.py
#   python run_all.py <filename> -> any explicit filename relative to HERE
WORKERS = {
    "23":    os.path.join(HERE, "ass3_23-05.py"),
    "ruben": os.path.join(HERE, "ass3_ruben.py"),
}

def _resolve_script():
    if len(sys.argv) < 2:
        return WORKERS["23"]
    arg = sys.argv[1].strip()
    if arg in WORKERS:
        return WORKERS[arg]
    p = arg if os.path.isabs(arg) else os.path.join(HERE, arg)
    return p

SCRIPT_PATH = _resolve_script()

# Presets, in execution order (smallest first).
# Each preset is a dict with:
#   name      : human-readable label
#   n_values  : comma-separated list of dataset sizes
#   mem_opt   : "true" / "false"
#   suffix    : appended to output filenames
PRESETS = [
    {"name": "sanity",        "n_values": "1000,10000",                          "mem_opt": "true",  "suffix": "_sanity"},
    {"name": "medium_naive",  "n_values": "100000",                              "mem_opt": "false", "suffix": "_medium_naive"},
    {"name": "medium_memopt", "n_values": "100000",                              "mem_opt": "true",  "suffix": "_medium_memopt"},
    {"name": "full",          "n_values": "10000,50000,100000,500000,1000000",   "mem_opt": "true",  "suffix": "_full"},
]

# ---------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------

def run_preset(preset):
    """
    Launch ass3_23-05.py as a subprocess with environment variables
    set from `preset`. Returns the elapsed wall-clock seconds.
    Raises subprocess.CalledProcessError on non-zero exit code.
    """
    env = os.environ.copy()
    # Avoid BLAS oversubscription: PySpark already spawns one Python worker
    # per core, and each worker calls NumPy which would otherwise spin up
    # another thread pool per BLAS call -> N x N threads. Pin BLAS to 1.
    env["OMP_NUM_THREADS"]        = "1"
    env["OPENBLAS_NUM_THREADS"]   = "1"
    env["MKL_NUM_THREADS"]        = "1"
    env["NUMEXPR_NUM_THREADS"]    = "1"
    env["BENCH_N_VALUES"] = preset["n_values"]
    env["BENCH_MEM_OPT"]  = preset["mem_opt"]
    # prefix the worker tag to BENCH_SUFFIX so outputs from different
    # workers don't overwrite each other (e.g. "_ass3_ruben_sanity").
    env["BENCH_SUFFIX"]   = f"_{WORKER_TAG.rstrip('_')}{preset['suffix']}"

    print("=" * 70)
    print(f"PRESET: {preset['name']}")
    print(f"  n_values = {preset['n_values']}")
    print(f"  mem_opt  = {preset['mem_opt']}")
    print(f"  suffix   = {preset['suffix']}")
    print("=" * 70, flush=True)

    preset_log = os.path.join(
        LOG_DIR, f"{WORKER_TAG}{preset['name']}_{RUN_STAMP}.log"
    )
    print(f"  log file = {preset_log}", flush=True)

    t0 = time.time()
    with open(preset_log, "w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            [sys.executable, "-u", SCRIPT_PATH],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.__stdout__.write(line)
            sys.__stdout__.flush()
            logf.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, SCRIPT_PATH)
    return time.time() - t0


def main():
    global WORKER_TAG
    if not os.path.isfile(SCRIPT_PATH):
        print(f"ERROR: cannot find worker script at {SCRIPT_PATH}")
        sys.exit(1)
    WORKER_TAG = os.path.splitext(os.path.basename(SCRIPT_PATH))[0] + "_"
    print(f"[driver] worker script = {SCRIPT_PATH}")

    timings = []
    t_global = time.time()

    for preset in PRESETS:
        try:
            elapsed = run_preset(preset)
            timings.append((preset["name"], elapsed, "OK"))
            print(f"\n[{preset['name']}] completed in {elapsed:.1f} s\n",
                  flush=True)
        except subprocess.CalledProcessError as e:
            # FAIL-FAST: stop the whole run on first error.
            timings.append((preset["name"], time.time() - t_global, "FAILED"))
            print(f"\nERROR: preset '{preset['name']}' failed "
                  f"with exit code {e.returncode}.", file=sys.stderr)
            print("Stopping the driver (fail-fast mode).", file=sys.stderr)
            _print_summary(timings, time.time() - t_global)
            sys.exit(e.returncode)

    _print_summary(timings, time.time() - t_global)


def _print_summary(timings, total):
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, elapsed, status in timings:
        print(f"  {name:<10}  {elapsed:>8.1f} s   {status}")
    print(f"  {'TOTAL':<10}  {total:>8.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
