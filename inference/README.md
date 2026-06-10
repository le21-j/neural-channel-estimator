# inference/ — standalone C++ TensorRT engine

**GPU_STEP — builds on a Linux + CUDA + TensorRT 10.x host only.**
Written and reviewed on macOS; never compiled there (see CLAUDE.md GPU gates).

## Prep (local, before shipping to the GPU host)

```bash
python scripts/make_calib_npy.py     # -> inference/calib.npy + sample_input.npy
```

Ship: `model.onnx`, `inference/`, `scripts/compare_trt_onnx.py`,
`inference/calib.npy`, `inference/sample_input.npy`.

## Build + run (remote)

```bash
cd inference
cmake -B build -DCMAKE_BUILD_TYPE=Release   # -DTENSORRT_ROOT=... if needed
cmake --build build -j

./build/estimator --onnx ../model.onnx --input sample_input.npy --fp16
./build/estimator --onnx ../model.onnx --input sample_input.npy \
    --int8 --calib calib.npy
python ../scripts/compare_trt_onnx.py        # parity vs ONNX-RT (atol 1e-2)
```

## Layout

| file | role |
|---|---|
| `src/trt_engine.{h,cpp}` | build from ONNX / deserialize cache; FP16/INT8 flags |
| `src/buffers.{h,cpp}` | RAII pinned-host + device memory, explicit async H2D/D2H |
| `src/calibrator.{h,cpp}` | `IInt8EntropyCalibrator2` fed from `calib.npy` |
| `src/npy.h` | minimal float32 C-order .npy load/save (calibration + IO) |
| `src/main.cpp` | arg parse, warmup, N-iter benchmark, per-stage mean/p99 |

Per-stage timing uses `cudaEvent`s around H2D / `enqueueV3` / D2H on one
stream — the breakdown feeds `results/latency_breakdown.md` in Prompt 5.
