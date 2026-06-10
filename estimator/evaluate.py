"""NMSE-vs-SNR curves at low and high Doppler: LS vs LMMSE vs CNN.

Generates fresh fixed-SNR eval batches (seed 2, never seen in training),
runs all three estimators per (Doppler, SNR) cell, writes
results/nmse_curves.png + results/nmse_summary.json, prints the NMSE gap.

Usage: python estimator/evaluate.py [--ckpt estimator/checkpoints/model.pt]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from baselines import Baselines, load_cfgs, nmse  # noqa: E402
from generate import gen_batch  # noqa: E402
from model import load_model  # noqa: E402

SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=str(ROOT / "estimator" / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    sys_cfg, ch_cfg = load_cfgs()
    sweep = ch_cfg["doppler_sweep_hz"]
    bins = [float(min(sweep)), float(max(sweep))]  # low, high
    bl = Baselines(sys_cfg, ch_cfg)
    model = load_model(args.ckpt)

    curves = {}
    for f_d in bins:
        gen = torch.Generator().manual_seed(args.seed)
        curves[str(int(f_d))] = {}
        for snr in SNRS:
            y, h, _ = gen_batch(args.n, f_d, snr, sys_cfg, ch_cfg, gen)
            no = 10.0 ** (-snr / 10.0)
            h_ls = bl.ls(y, no)  # LS curve AND model input (see model.py)
            with torch.no_grad():
                x2 = torch.stack([h_ls.real, h_ls.imag], dim=1)
                p2 = model(x2)
                h_cnn = torch.complex(p2[:, 0], p2[:, 1])
            curves[str(int(f_d))][str(snr)] = {
                "ls": nmse(h_ls, h),
                "lmmse": nmse(bl.lmmse(y, no, f_d), h),
                "model": nmse(h_cnn, h),
            }
            c = curves[str(int(f_d))][str(snr)]
            print(f"f_d={f_d:6.0f} Hz  SNR={snr:4.0f} dB  "
                  f"LS={c['ls']:.4f}  LMMSE={c['lmmse']:.4f}  CNN={c['model']:.4f}")

    high = str(int(bins[1]))
    ls_m = sum(c["ls"] for c in curves[high].values()) / len(SNRS)
    cnn_m = sum(c["model"] for c in curves[high].values()) / len(SNRS)
    gap_db = 10 * math.log10(cnn_m / ls_m)
    print(f"\nNMSE gap at {high} Hz (CNN vs LS): {gap_db:+.2f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, f_d in zip(axes, bins):
        cell = curves[str(int(f_d))]
        for key, label, style in (("ls", "LS (lin interp)", "o--"),
                                  ("lmmse", "LMMSE (genie stats)", "s--"),
                                  ("model", "CNN", "^-")):
            ax.semilogy(SNRS, [cell[str(s)][key] for s in SNRS], style, label=label)
        ax.set_title(f"Doppler {f_d:.0f} Hz")
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("NMSE")
    axes[0].legend()
    fig.suptitle("OFDM channel estimation, TDL-C, comb-4 pilots")
    fig.tight_layout()
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    fig.savefig(res / "nmse_curves.png", dpi=150)

    (res / "nmse_summary.json").write_text(json.dumps(
        {"curves": curves, "gap_db_high_doppler": gap_db,
         "n_per_cell": args.n, "seed": args.seed}, indent=2))
    print(f"wrote {res/'nmse_curves.png'} and nmse_summary.json")


if __name__ == "__main__":
    main()
