"""
Multi-trace benchmark for all cache eviction plugins.

Usage (from the libCacheSim root):
    .venv/bin/python plugins/benchmark.py

Traces tested:
  • data/cloudPhysicsIO.vscsi          (block, VSCSI format, included in repo)
  • data/traces/*.oracleGeneral*.zst   (Twitter KV, oracleGeneral binary format)
"""

import importlib.util
import sys
import time
from pathlib import Path

from libcachesim import PluginCache, TraceReader, TraceType

# ── locate plugins ─────────────────────────────────────────────────────────────

PLUGINS_DIR = Path(__file__).parent
REPO_ROOT   = PLUGINS_DIR.parent

ALGORITHMS = [
    ("FIFO",      PLUGINS_DIR / "plugin_fifo.py"),
    ("SIEVE",     PLUGINS_DIR / "plugin_sieve.py"),
    ("S3-FIFO",   PLUGINS_DIR / "plugin_s3fifo.py"),
    ("ARC",       PLUGINS_DIR / "plugin_arc.py"),
    ("W-TinyLFU", PLUGINS_DIR / "plugin_wtinylfu.py"),
    ("S3-SIEVE",  PLUGINS_DIR / "plugin_s3sieve.py"),
]

# ── locate traces ──────────────────────────────────────────────────────────────

def find_traces() -> list[tuple[str, Path, TraceType]]:
    traces = []

    # 1. Bundled VSCSI trace
    vscsi = REPO_ROOT / "data" / "cloudPhysicsIO.vscsi"
    if vscsi.exists():
        traces.append(("cloudPhysicsIO", vscsi, TraceType.VSCSI_TRACE))

    # 2. Downloaded oracleGeneral traces (compressed or not)
    trace_dir = REPO_ROOT / "data" / "traces"
    if trace_dir.exists():
        for f in sorted(trace_dir.iterdir()):
            if "oracleGeneral" in f.name:
                traces.append((f.stem.split(".")[0], f, TraceType.ORACLE_GENERAL_TRACE))

    return traces

# ── plugin loader ──────────────────────────────────────────────────────────────

def load_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("plugin", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── runner ─────────────────────────────────────────────────────────────────────

def run_one(name: str, mod, trace_path: Path, trace_type: TraceType,
            cache_size: int) -> tuple[float, float, float]:
    """Returns (req_miss_ratio, byte_miss_ratio, elapsed_sec)."""
    cache = PluginCache(
        cache_size=cache_size,
        cache_init_hook=mod.init_hook,
        cache_hit_hook=mod.hit_hook,
        cache_miss_hook=mod.miss_hook,
        cache_eviction_hook=mod.eviction_hook,
        cache_remove_hook=mod.remove_hook,
        cache_free_hook=mod.free_hook,
        cache_name=name.lower().replace("-", ""),
    )
    reader = TraceReader(trace=str(trace_path), trace_type=trace_type)
    t0 = time.perf_counter()
    req_mr, byte_mr = cache.process_trace(reader)
    elapsed = time.perf_counter() - t0
    return req_mr, byte_mr, elapsed

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    traces = find_traces()
    if not traces:
        print("No traces found.  Run this script from the libCacheSim root.")
        sys.exit(1)

    # Pre-load all plugin modules
    plugins = []
    for alg_name, path in ALGORITHMS:
        if not path.exists():
            print(f"  [skip] {alg_name}: {path} not found")
            continue
        plugins.append((alg_name, load_plugin(path)))

    # For each trace, test at three cache sizes (1MB, 16MB, 128MB)
    cache_sizes = {
        "1MB":   1 * 1024 * 1024,
        "16MB":  16 * 1024 * 1024,
        "128MB": 128 * 1024 * 1024,
    }

    summary: dict[str, list[float]] = {name: [] for name, _ in plugins}

    for trace_name, trace_path, trace_type in traces:
        for size_label, cache_size in cache_sizes.items():
            print(f"\n{'═'*70}")
            print(f"  Trace: {trace_name}   Cache: {size_label}")
            print(f"{'═'*70}")
            print(f"  {'Algorithm':<12}  {'Req miss':>10}  {'Byte miss':>10}  {'Time (s)':>9}")
            print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}")

            for alg_name, mod in plugins:
                try:
                    req_mr, byte_mr, elapsed = run_one(
                        alg_name, mod, trace_path, trace_type, cache_size)
                    print(f"  {alg_name:<12}  {req_mr:>10.4f}  {byte_mr:>10.4f}  {elapsed:>9.2f}")
                    summary[alg_name].append(req_mr)
                except Exception as e:
                    print(f"  {alg_name:<12}  ERROR: {e}")

    # Overall average miss ratio per algorithm
    print(f"\n{'═'*70}")
    print("  OVERALL AVERAGE REQUEST MISS RATIO (lower = better)")
    print(f"{'═'*70}")
    ranked = sorted(
        [(name, sum(mrs) / len(mrs)) for name, mrs in summary.items() if mrs],
        key=lambda x: x[1]
    )
    for rank, (name, avg_mr) in enumerate(ranked, 1):
        bar = "█" * int((1 - avg_mr) * 40)
        print(f"  #{rank}  {name:<12}  avg req miss = {avg_mr:.4f}  {bar}")


if __name__ == "__main__":
    main()
