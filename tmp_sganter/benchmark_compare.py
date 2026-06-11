"""Compare per-iteration times across N labeled training-log files.

Usage:
    python tmp_sganter/benchmark_compare.py \\
        --log "ZBH1=path/to/zbh1.log=[4]*16" \\
        --log "dyn_eq=path/to/dyn_eq.log=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]" \\
        --log "dyn_var=path/to/dyn_var.log=[1,2,3,3,3,4,4,9,10,8,5,4,3,2,2,1]" \\
        [--warmup 10] [--baseline-label ZBH1]

Format of --log: "LABEL=PATH=MB_SIZES" (= as separator). MB_SIZES is a freeform
string (it is only used as a column value in the output table, never parsed).
The first --log is the baseline against which other rows compute the
percentage delta unless --baseline-label is given.

Outputs both a per-config summary block and a Markdown comparison table
that can be pasted directly into a report.
"""
import argparse
import re
import sys
from pathlib import Path

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)


def parse_log(path: Path) -> list[tuple[int, float]]:
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


def parse_log_spec(spec: str) -> tuple[str, Path, str]:
    """Split LABEL=PATH=MB_SIZES on the first two '=' so that MB_SIZES may
    contain '=' characters itself (it usually won't, but be safe)."""
    parts = spec.split("=", 2)
    if len(parts) != 3:
        sys.exit(
            f"error: --log expects LABEL=PATH=MB_SIZES, got: {spec!r}"
        )
    label, path, mbsizes = parts
    return label.strip(), Path(path.strip()), mbsizes.strip()


def fmt_delta(value: float, base: float, key: str) -> str:
    if base == 0:
        return ""
    delta = value - base
    pct = (delta / base) * 100
    return f"{pct:+.1f} %"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--log",
        action="append",
        required=True,
        metavar="LABEL=PATH=MB_SIZES",
        help="repeatable; provide one --log per run to compare",
    )
    p.add_argument("--warmup", type=int, default=10,
                   help="Skip the first N iterations (default: 10)")
    p.add_argument("--baseline-label", default=None,
                   help="Label of the row to use as the comparison baseline "
                        "(default: the first --log)")
    args = p.parse_args()

    specs = [parse_log_spec(s) for s in args.log]
    if len(specs) < 1:
        sys.exit("error: need at least one --log")

    rows = []
    for label, path, mbsizes in specs:
        if not path.exists():
            sys.exit(f"error: {path} not found (label={label})")
        iters = parse_log(path)
        vals = [t for (i, t) in iters if i > args.warmup]
        rows.append({
            "label": label,
            "path": path,
            "mbsizes": mbsizes,
            "raw_count": len(iters),
            "stats": stats(vals),
        })

    print("Parsed iterations per log "
          f"(post-warmup of {args.warmup} iters):")
    for r in rows:
        st = r["stats"]
        print(f"  {r['label']:<20s} raw={r['raw_count']:>4d}  "
              f"after-warmup={st['n']:>4d}  ({r['path']})")
    print()

    # Per-row summary blocks
    for r in rows:
        print("=" * 60)
        print(f"{r['label']}   mb_sizes={r['mbsizes']}")
        print_stats("iteration time", r["stats"])
    print("=" * 60)
    print()

    # Comparison table
    baseline_label = args.baseline_label or rows[0]["label"]
    baseline = next((r for r in rows if r["label"] == baseline_label), None)
    if baseline is None:
        sys.exit(f"error: baseline-label {baseline_label!r} not found")
    base_st = baseline["stats"]

    print(f"Comparison table  (baseline = {baseline_label}, "
          f"deltas are vs baseline median)")
    print()
    header = (f"| {'Config':<18s} | {'mb_sizes':<48s} | "
              f"{'median':>10s} | {'mean':>10s} | {'std':>8s} | "
              f"{'vs baseline':>11s} |")
    sep = "|" + "-" * (len(header) - 2) + "|"
    sep = ("|" + "-" * 20 + "|" + "-" * 50 + "|"
           + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|"
           + "-" * 13 + "|")
    print(header)
    print(sep)
    for r in rows:
        st = r["stats"]
        if st["n"] == 0:
            print(f"| {r['label']:<18s} | {r['mbsizes']:<48s} | "
                  f"{'(empty)':>10s} | {'':>10s} | {'':>8s} | {'':>11s} |")
            continue
        is_base = r["label"] == baseline_label
        delta_str = "baseline" if is_base else fmt_delta(
            st["median"], base_st["median"], "median")
        med_str = f"{st['median']:.1f} ms"
        mean_str = f"{st['mean']:.1f} ms"
        std_str = f"{st['std']:.1f} ms"
        # Highlight baseline with double-star? Keep plain for readability.
        print(f"| {r['label']:<18s} | {r['mbsizes']:<48s} | "
              f"{med_str:>10s} | {mean_str:>10s} | {std_str:>8s} | "
              f"{delta_str:>11s} |")


if __name__ == "__main__":
    main()
