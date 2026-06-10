"""P1-PROMPT-4 EXIT helper (REMOTE): TensorRT output vs ONNX Runtime.

Run on the GPU host after ./estimator wrote trt_out.npy. FP16 engine vs FP32
ONNX-RT reference: atol=1e-2 (FP16 rounding budget — NOT the 1e-3 FP32 bar
from Prompt 3). INT8 gets no parity assert here; its accuracy story is the
per-Doppler NMSE table in Prompt 5.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ATOL_FP16 = 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "model.onnx"))
    ap.add_argument("--input", default=str(ROOT / "inference" / "sample_input.npy"))
    ap.add_argument("--trt-out", default=str(ROOT / "inference" / "trt_out.npy"))
    ap.add_argument("--atol", type=float, default=ATOL_FP16)
    args = ap.parse_args()

    x = np.load(args.input)
    trt = np.load(args.trt_out)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    diff = float(np.abs(ref - trt).max())
    print(f"TRT vs ONNX-RT: max abs diff = {diff:.3e} (atol {args.atol})")
    if diff > args.atol:
        print("PARITY FAIL")
        sys.exit(1)
    print("PARITY PASS")


if __name__ == "__main__":
    main()
