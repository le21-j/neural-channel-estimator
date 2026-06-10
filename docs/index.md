# Tutorial: neural-channel-estimator

A neural OFDM **channel estimator**: a small CNN learns to reconstruct the full
radio-channel grid from a few **pilot** measurements, beating the classical
LS/LMMSE estimators exactly where they struggle — fast-moving (high-Doppler)
channels — and (in later prompts) gets deployed as a C++ TensorRT engine under
a 5G slot-time budget.

**Source spec:** `../portfolio-projects.md` (Project 1) · **State:** see `Active:` line in `CLAUDE.md`

## Core abstractions

```mermaid
flowchart TD
    A0["OFDM numerology & configs"] --> A1["Dataset generator (Sionna TDL-C)"]
    A1 --> A2["Baseline estimators (LS / LMMSE)"]
    A1 --> A3["Residual CNN (ChannelCNN)"]
    A2 -->|"LS grid = model input"| A3
    A2 --> A4["Evaluation & EXIT checks"]
    A3 --> A4
    A5["Agent harness & GPU gates"] -.->|"sequences all work"| A0
```

## Chapters

1. [OFDM numerology & configs](01_ofdm_numerology_and_configs.md)
2. [Dataset generator](02_dataset_generator.md)
3. [Baseline estimators: LS & LMMSE](03_baseline_estimators.md)
4. [Residual CNN](04_residual_cnn.md)
5. [Harness, EXIT checks & GPU gates](05_harness_and_gpu_gates.md)
6. [ONNX export & parity](06_onnx_export_and_parity.md)

Chapters 1–6 cover code that exists today (Prompts 1–3 PASSED). TensorRT
inference and profiling chapters get added as Prompts 4–5 land.

---
*Maintained in the style of [PocketFlow-Tutorial-Codebase-Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge); updated at each prompt EXIT.*
