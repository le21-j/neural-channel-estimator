# NMSE vs quantization (CNN estimator)

Mean NMSE over SNR 0–25 dB (eval cells of 200 samples, seed 2).
FP32 = local PyTorch/ONNX-RT (parity-verified). FP16/INT8 require the
TensorRT engines — **PLACEHOLDER: awaiting remote GPU run** (the remote
runbook evaluates all six Doppler bins through both engines).

| Doppler (Hz) | FP32 | FP16 | INT8 |
|---|---|---|---|
| 10   | 0.0593 | PLACEHOLDER | PLACEHOLDER |
| 100  | PLACEHOLDER (local eval optional) | PLACEHOLDER | PLACEHOLDER |
| 300  | PLACEHOLDER (local eval optional) | PLACEHOLDER | PLACEHOLDER |
| 600  | PLACEHOLDER (local eval optional) | PLACEHOLDER | PLACEHOLDER |
| 1000 | PLACEHOLDER (local eval optional) | PLACEHOLDER | PLACEHOLDER |
| 1600 | 0.2769 | PLACEHOLDER | PLACEHOLDER |

Why INT8 is broken out **by Doppler bin**: quantization noise hits channel
estimators hardest exactly where LS already fails — fast fading. One
aggregate INT8 number would hide where the damage concentrates; the
per-bin column is the physics result this table exists to show.
