"""Produce INT8 calibration set + benchmark sample input (LOCAL, fast).

calib.npy: 512 LS-interp grids drawn evenly across Doppler bins from
data/train — INT8 scales must see the full fading range, not just slow
channels. sample_input.npy: one grid for the C++ benchmark + parity check.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_CALIB = 512


def main():
    d = ROOT / "data" / "train"
    ls = np.load(d / "ls_interp.npy", mmap_mode="r")
    dop = np.load(d / "doppler_hz.npy")
    rng = np.random.default_rng(11)

    idx = []
    bins = np.unique(dop)
    per = N_CALIB // len(bins)
    for b in bins:
        cand = np.where(dop == b)[0]
        idx.extend(rng.choice(cand, per, replace=False))
    idx = np.sort(np.array(idx))

    calib = np.asarray(ls[idx], dtype=np.float32)
    out = ROOT / "inference"
    np.save(out / "calib.npy", calib)
    np.save(out / "sample_input.npy", np.asarray(ls[:1], dtype=np.float32))
    print(f"calib.npy {calib.shape} ({len(bins)} doppler bins x {per}), "
          f"sample_input.npy (1, 2, 14, 64)")
    sys.exit(0)


if __name__ == "__main__":
    main()
