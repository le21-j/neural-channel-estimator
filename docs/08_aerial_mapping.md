---
type: kernel-doc
title: "Chapter 8: NVIDIA Aerial pyAerial mapping & slot-budget framing"
tags: [channel-est, aerial, mapping]
timestamp: 2026-06-17
---

# Chapter 8: NVIDIA Aerial pyAerial mapping & slot-budget framing

This chapter explains how this repo's CNN channel estimator maps to
NVIDIA Aerial's pluggable channel-estimation slot, frames the 5G-NR
slot-time budget, motivates the 5G-NR DMRS pilot option, and traces
the ONNX → TensorRT engine path already present in `inference/`.

---

## 1. Where this estimator fits in pyAerial

NVIDIA Aerial (cuPHY / pyAerial 24.3) structures the L1 pipeline as a
sequence of pluggable **processing slots**, one per OFDM symbol group.
The channel-estimation slot receives the de-mapped received resource
grid and must return a full-grid channel estimate $\hat{H}$ before the
PDSCH equalizer runs. The default chain is:

```
LS pilot extraction  →  [MMSE or CNN]  →  full-grid Ĥ  →  equalizer
```

The bracketed stage is the **pluggable channel-estimator**.  cuPHY
exposes this via `ChannelEstimatorPlugin` (pyAerial 24.3 Python
binding); a user-supplied TensorRT engine can replace the default
MMSE kernel.

**This repo's `ChannelCNN` is that replacement.**  The mapping is
one-to-one:

| pyAerial concept | This repo | Notes |
|---|---|---|
| Pluggable channel-est slot | `estimator/model.py` `ChannelCNN` | residual CNN over LS-interp grid |
| Input tensor `ls_grid` (1, 2, S, F) float32 | `estimator/export_onnx.py` input name | LS linear-interp pre-stage outside CNN |
| Output tensor `h_refined` (1, 2, S, F) float32 | ONNX output name | real/imag stacked (TensorRT-friendly) |
| TensorRT FP16 engine | `inference/src/TrtEngine.*` | built from `model.onnx` via `trtexec` |
| Per-slot execution (static shape) | `torch.onnx.export(..., dynamo=False)` | static (1,2,14,64) lets TRT optimize hardest |
| INT8 calibration | `scripts/make_calib_npy.py` + `NpyEntropyCalibrator` | 85 grids/Doppler bin, all bins covered |

The LS pre-stage (pilot extraction + linear interp across subcarriers
and symbols) is a fixed, lightweight cuPHY kernel — it runs **before**
the CNN engine, exactly mirroring this repo's `baselines.py:ls_estimate`
which produces the input the CNN expects.

---

## 2. 5G-NR slot-budget framing

For $\mu = 1$ (30 kHz SCS):

$$T_{\text{slot}} = 0.5\,\text{ms} = 500\,\mu\text{s}$$
$$T_{\text{symbol}} = T_{\text{slot}} / 14 \approx 35.7\,\mu\text{s}$$

The channel-estimator must finish **within one symbol period** (35.7 µs)
so the equalizer can start before the next symbol arrives.

`profiling/budget.py` converts any measured latency into budget fractions:

```
python profiling/budget.py --e2e-us <X>
```

Example (illustrative, not measured here):

```
e2e = 12.00 us
  =   33.6% of one OFDM symbol (35.7 us)    ← PARKED: real X from remote GPU
  =    2.4% of one slot (500 us)
```

> **PARKED placeholder.** The actual measured latency X µs for this CNN
> (FP16 TensorRT, A100/H100) will be filled in by the remote GPU session
> (P1-PROMPT-5).  No numbers are invented here.  The budget frame itself
> (X µs = Y % of 35.7 µs symbol) is the conceptual tool; run
> `profiling/budget.py --e2e-us <measured>` to populate Y after the
> remote run.

Why the 35.7 µs bound matters:

- cuPHY pipelines L1 processing symbol-by-symbol in streaming mode.
- The CNN must not become the critical-path gate.  The 85k-param,
  ~120 MFLOP `ChannelCNN` (6 × Conv3×3 at width 48) was explicitly
  sized to cost single-digit µs on a datacenter GPU (vs >500k-param
  U-Nets that exceed the budget).  See `estimator/model.py` docstring.

---

## 3. Why 5G-NR DMRS pilots (`--dmrs` flag)

The default pilot pattern in this repo (`pilot_comb_spacing=4`, comb-4)
is numerology-compatible but not spec-identical to NR PDSCH DMRS.

3GPP TS 38.211 §6.4.1.1 defines **DMRS Type 1** for PDSCH:

- **CDM group 0, port p=0**: pilots on every **even** subcarrier
  (comb-2: subcarriers 0, 2, 4, …) within the DMRS OFDM symbols.
- **DMRS symbols**: positions 2 and 11 within a normal-CP slot
  (same as this repo's `pilot_symbol_indices: [2, 11]`).
- Port mapping (§6.4.1.1.3-1): CDM group 0 = subcarriers
  $\{k : k \equiv 0 \pmod{2}\}$; CDM group 1 = odd subcarriers.
  Single-port (p=0) occupies CDM group 0 only.

Result: **64 pilot REs/slot** (vs 32 for comb-4) — 7.1% pilot overhead
vs 3.6%.  The denser pattern improves delay-domain coverage but leaves
fewer data REs per slot.

`data/generate.py --dmrs` activates this pattern as an additive flag
(`--dmrs`, default OFF) so all P1-1..3 PASSED exits remain unaffected.
The output array shapes are identical `(N, 2, 14, 64)` — only the pilot
mask changes inside `gen_batch`.

**Hiring-manager relevance**: demonstrating spec-correct NR DMRS
(not a generic comb pattern) is the difference between a toy estimator
and one that slots into cuPHY's actual pilot extraction chain, where
the pilot positions are hard-coded to §6.4.1.1 tables.

---

## 4. ONNX → TensorRT engine path

The full export + deployment chain is already implemented:

```
estimator/export_onnx.py   →  model.onnx   (opset 17, static (1,2,14,64))
                                  │
          trtexec --fp16          │   (remote GPU: P1-PROMPT-4)
                                  ▼
                          engine_fp16.trt
                                  │
          inference/src/main.cpp  │   TrtEngine::deserialize()
          (cudaEvent timing)      ▼
                     per-stage mean/p99 timings
                                  │
          profiling/budget.py     ▼
                     X µs = Y% of 35.7 µs symbol   ← PARKED
```

Key design decisions repeated here for Aerial context:

| Decision | Reason for Aerial fit |
|---|---|
| Opset 17 pinned | TensorRT 10 parser requires explicit opset; "latest" breaks serialization |
| Static batch=1 | One engine call per slot; dynamic shapes disable TRT kernel fusion |
| FP16 engine | cuPHY processes in FP16 on-device; avoids conversion overhead |
| INT8 calibration from all Doppler bins | Aerial sees full vehicular range; single-Doppler calib clips fast-fading activations |
| complex → 2-ch real | TensorRT complex64 support is thin; 2-ch real is the standard workaround |

---

## 5. Connecting budget.py to this chapter

After the remote GPU session populates a real latency number:

```bash
python profiling/budget.py --e2e-us <measured>
```

Replace the PARKED placeholder line in §2 above with the real output.
Then update `README.md` latency table (P1-5 EXIT).

Back to [index](index.md)
