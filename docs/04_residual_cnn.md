# Chapter 4: Residual CNN

`estimator/model.py` holds the learner: an 85k-parameter CNN that *refines*
the LS estimate instead of estimating from scratch.

## The input contract (important!)

The model does **not** see the sparse pilot grid. It sees the
[LS linear-interpolated grid](03_baseline_estimators.md) and predicts a
correction:

```python
def forward(self, x):   # (B, 2, 14, 64) LS grid -> refined estimate
    return x + self.net(x)
```

Analogy: instead of asking an artist to paint a map from 32 dots, hand them
the rough interpolated map and ask them to fix the smudges.

## Why residual? (a failure that taught us)

The first attempt fed the raw sparse grid (3.6% non-zero) straight into the
CNN. After 60 epochs it was **worse than LS by ~1 dB** at 1600 Hz — an honest
EXIT failure, reported as such. Switching to LS-input + residual output
converged in 30 epochs to val NMSE 0.094 (from 0.361). The lesson is baked
into the model docstring so nobody retries the sparse path by accident.

## Architecture vs latency budget

6 × conv3x3 at width 48 → **85.3k params, ~120 MFLOPs** per (14, 64) slot.
Receptive field 13×13 — deliberately just wide enough to span the 9-symbol
gap between pilot symbols [2, 11]. Small on purpose: the Prompt 4/5 TensorRT
engine must fit inside a 35.7 µs symbol budget, where a 500k-param U-Net
would not.

## Training (estimator/train.py)

- Loss = batch NMSE, $\|\hat{H}-H\|^2 / \|H\|^2$ — the eval metric itself.
- **Doppler-stratified val split**: 15% held out *per Doppler bin*, so val
  isn't dominated by easy slow-fading samples.
- LS inputs precomputed once and cached (`ls_interp.npy`).
- W&B logging, offline mode by default.

Result on fresh eval cells (seed 2): beats LS at **every** SNR at 1600 Hz,
mean gap **−1.96 dB**, within ~5% of genie LMMSE at ≥10 dB.

Next: [Chapter 5 — Harness, EXIT checks & GPU gates](05_harness_and_gpu_gates.md)
