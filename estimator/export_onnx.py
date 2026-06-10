"""Export the trained ChannelCNN to model.onnx.

Pinned opset 17. Static shapes — batch=1, per-slot inference (one engine
execution per OFDM slot is the deployment model; TensorRT prefers static).
Input  "ls_grid"   float32 (1, 2, 14, 64)  — LS linear-interp grid (re/im)
Output "h_refined" float32 (1, 2, 14, 64)  — refined channel estimate

Usage: python estimator/export_onnx.py [--ckpt ...] [--out model.onnx]
"""
import argparse
from pathlib import Path

import onnx
import torch

from model import load_model

ROOT = Path(__file__).resolve().parents[1]
OPSET = 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=str(ROOT / "estimator" / "checkpoints" / "model.pt"))
    ap.add_argument("--out", type=str, default=str(ROOT / "model.onnx"))
    args = ap.parse_args()

    model = load_model(args.ckpt)
    dummy = torch.zeros(1, 2, 14, 64)
    torch.onnx.export(model, dummy, args.out, opset_version=OPSET,
                      input_names=["ls_grid"], output_names=["h_refined"],
                      dynamo=False)

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 2, 14, 64], f"input shape {dims} != [1, 2, 14, 64]"
    print(f"exported {args.out}: opset {OPSET}, input {dims}, "
          f"{len(m.graph.node)} nodes")


if __name__ == "__main__":
    main()
