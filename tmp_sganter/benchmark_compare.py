"""Compare per-iteration times between ZBH1 and dynamic_mb benchmark runs.

Usage:
    python tmp_sganter/benchmark_compare.py \
        --zbh1 zbh1_run.log \
        --dynamic dynamic_mb_run.log \
        [--warmup 10]
"""
import argparse
import re
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)


def parse_log(path: Path) -> list[tuple[int, float]]:
    """Return list of (iteration, elapsed_ms) from a training log."""
    out = []
    with open(path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    s = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    return {
        "n": n,
        "mean": mean,
        "median": s[n // 2],
        "std": std,
        "min": s[0],
        "max": s[-1],
        "p25": s[n // 4],
        "p75": s[(3 * n) // 4],
    }


def print_stats(name: str, st: dict):
    if st["n"] == 0:
        print(f"  {name}: no iterations found")
        return
    print(f"  {name}:")
    print(f"    n       = {st['n']}")
    print(f"    mean    = {st['mean']:8.2f} ms")
    print(f"    median  = {st['median']:8.2f} ms")
    print(f"    std     = {st['std']:8.2f} ms")
    print(f"    min     = {st['min']:8.2f} ms")
    print(f"    p25     = {st['p25']:8.2f} ms")
    print(f"    p75     = {st['p75']:8.2f} ms")
    print(f"    max     = {st['max']:8.2f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zbh1", required=True, type=Path)
    p.add_argument("--dynamic", required=True, type=Path)
    p.add_argument("--warmup", type=int, default=10,
                   help="Skip the first N iterations (default: 10)")
    args = p.parse_args()

    for path in (args.zbh1, args.dynamic):
        if not path.exists():
            sys.exit(f"error: {path} not found")

    zbh1 = parse_log(args.zbh1)
    dyn = parse_log(args.dynamic)

    print(f"Raw iterations parsed: zbh1={len(zbh1)}, dynamic={len(dyn)}")

    # Filter: skip warmup (keep iter > warmup)
    zbh1_vals = [t for (i, t) in zbh1 if i > args.warmup]
    dyn_vals = [t for (i, t) in dyn if i > args.warmup]

    print(f"After skipping first {args.warmup} warmup iterations: "
          f"zbh1={len(zbh1_vals)}, dynamic={len(dyn_vals)}")
    print()

    zbh1_st = stats(zbh1_vals)
    dyn_st = stats(dyn_vals)

    print("=" * 60)
    print("ZBH1 (baseline)")
    print_stats("iteration time", zbh1_st)
    print()
    print("dynamic_mb")
    print_stats("iteration time", dyn_st)
    print("=" * 60)

    if zbh1_st["n"] and dyn_st["n"]:
        print()
        print("Comparison (dynamic_mb vs ZBH1):")
        for key in ("mean", "median", "p25", "p75"):
            zb = zbh1_st[key]
            dy = dyn_st[key]
            delta = dy - zb
            pct = (delta / zb) * 100 if zb else 0
            sign = "faster" if delta < 0 else "slower"
            print(f"  {key:7s}: {zb:8.2f} ms  ->  {dy:8.2f} ms   "
                  f"({abs(delta):+6.2f} ms, {pct:+6.2f}%, {sign})")

        # Throughput-oriented comparison (samples/sec)
        # Assumes same global_batch_size — which is true for these runs.
        print()
        speedup = zbh1_st["mean"] / dyn_st["mean"]
        print(f"Speedup (zbh1_mean / dynamic_mean): {speedup:.3f}x")


if __name__ == "__main__":
    main()
