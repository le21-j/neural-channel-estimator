"""P1-PROMPT-3 EXIT check: PyTorch vs ONNX Runtime parity (FP32, atol=1e-3).

Runs identical inputs through the PyTorch checkpoint and model.onnx via
ONNX Runtime, asserts elementwise agreement, prints max abs diff.
FP32 only — FP16/INT8 tolerances are Prompt-4 territory.

Usage: python scripts/verify_parity.py [--onnx model.onnx] [--n 64]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "estimator"))
from model import load_model  # noqa: E402

ATOL = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default=str(ROOT / "model.onnx"))
    ap.add_argument("--ckpt", type=str,
                    default=str(ROOT / "estimator" / "checkpoints" / "model.pt"))
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    model = load_model(args.ckpt)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    print(f"onnx input '{iname}' shape {ishape}")

    rng = np.random.default_rng(7)
    # mix of real LS-grid stats and pure noise inputs
    ls = np.load(ROOT / "data" / "train" / "ls_interp.npy", mmap_mode="r")
    xs = np.concatenate([
        np.asarray(ls[: args.n // 2], dtype=np.float32),
        rng.standard_normal((args.n - args.n // 2, 2, 14, 64)).astype(np.float32),
    ])

    max_diff = 0.0
    for x in xs:
        x1 = x[None]  # batch=1, matches exported static shape
        with torch.no_grad():
            ref = model(torch.from_numpy(x1)).numpy()
        out = sess.run(None, {iname: x1})[0]
        max_diff = max(max_diff, float(np.abs(ref - out).max()))

    print(f"parity over {args.n} inputs (FP32): max abs diff = {max_diff:.3e}")
    assert np.isfinite(max_diff)
    if max_diff > ATOL:
        print(f"EXIT CHECK FAIL: max abs diff {max_diff:.3e} > atol {ATOL}")
        sys.exit(1)
    print(f"EXIT CHECK PASS (atol={ATOL})")


if __name__ == "__main__":
    main()
