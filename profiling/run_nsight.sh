#!/usr/bin/env bash
# REMOTE GPU host only (nsys is Linux/CUDA tooling — blocked on macOS by hook).
# Profiles the TensorRT estimator and exports a CSV trace for latency_report.py.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=inference/build/estimator
[ -x "$BIN" ] || { echo "build first: see inference/README.md"; exit 1; }

nsys profile -o profiling/estimator_fp16 --force-overwrite true \
    "$BIN" --onnx model.onnx --input inference/sample_input.npy --fp16 --iters 1000

nsys stats --report cuda_gpu_trace --format csv \
    --output profiling/estimator_fp16 profiling/estimator_fp16.nsys-rep

echo "trace: profiling/estimator_fp16_cuda_gpu_trace.csv"
echo "next: python profiling/latency_report.py"
