"""Parse the nsys CUDA GPU trace CSV into results/latency_breakdown.md.

REMOTE step (needs a real trace). Buckets rows into H2D / kernel / D2H by
the Name column, sums per-iteration durations, writes the breakdown table.

Usage: python profiling/latency_report.py \
           [--csv profiling/estimator_fp16_cuda_gpu_trace.csv] [--iters 1000]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bucket(name):
    if "HtoD" in name:
        return "H2D"
    if "DtoH" in name:
        return "D2H"
    return "kernel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default=str(ROOT / "profiling" / "estimator_fp16_cuda_gpu_trace.csv"))
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    sums = {"H2D": 0.0, "kernel": 0.0, "D2H": 0.0}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            dur_col = next((k for k in row if "Duration" in k), None)
            name_col = next((k for k in row if "Name" in k), None)
            if not dur_col or not name_col or not row[dur_col]:
                continue
            sums[bucket(row[name_col])] += float(row[dur_col])

    # nsys durations are ns; convert summed-total -> per-iteration us
    per_iter = {k: v / 1e3 / args.iters for k, v in sums.items()}
    e2e = sum(per_iter.values())

    out = ROOT / "results" / "latency_breakdown.md"
    lines = [
        "# Latency breakdown (TensorRT FP16, per-iteration GPU time)",
        "",
        f"Source: `{Path(args.csv).name}`, {args.iters} iterations.",
        "",
        "| stage | mean (us) |",
        "|---|---|",
        *(f"| {k} | {v:.2f} |" for k, v in per_iter.items()),
        f"| **GPU total** | **{e2e:.2f}** |",
        "",
        "Wall-clock e2e (incl. launch overhead) comes from the estimator",
        "binary's own cudaEvent report; run `python profiling/budget.py",
        "--e2e-us <value>` to express it against the symbol/slot budget.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for k, v in per_iter.items():
        print(f"  {k}: {v:.2f} us")


if __name__ == "__main__":
    main()
