# Chapter 3: Baseline estimators: LS & LMMSE

`estimator/baselines.py` is the bar the CNN must clear. Both estimators come
from the **installed Sionna API** (probed first, per spec) rather than
hand-rolled math.

## LS — least squares

At each pilot RE the channel is just measured: $\hat{H} = Y/X$ (divide received
by known pilot). Everything between pilots is **linear interpolation**.
Cheap, assumption-free, and blind to channel statistics — at 1600 Hz Doppler
it floors around NMSE 0.26 because two pilot symbols can't track the fading.

```python
self.ls_est = LSChannelEstimator(self.rg, interpolation_type="lin")
```

## LMMSE — the genie baseline

LMMSE filters the LS estimate using the channel's *known statistics*
(spec closed form: $H_{lmmse} = R_{hh}(R_{hh}+\sigma^2 I)^{-1} H_{ls}$).
We hand it **analytic** TDL covariance matrices — true frequency covariance
from the delay profile, true time covariance from the exact Doppler:

```python
cov_freq = tdl_freq_cov_mat("C", 30e3, 64, 300e-9, precision="single")
cov_time = tdl_time_cov_mat("C", speed, fc, sym_dur, 14, precision="single")
interp   = LMMSEInterpolator(pilot_pattern, cov_time, cov_freq, order="f-t")
```

That makes it a *genie*: in the field nobody knows the Doppler exactly. Beating
LS is the EXIT bar; getting near genie-LMMSE is the flex.

## A measured decision: interpolation order

`order="f-t"` (frequency first, then time) measured **0.189** NMSE at 1600 Hz
vs 0.268 for `"t-f"`. Reason: frequency interpolation works from 16
well-spaced comb pilots (well-conditioned), time interpolation from only 2
symbols (nearly blind) — do the reliable axis first.

> Self-test: `python estimator/baselines.py` checks LS accuracy at high SNR
> and asserts LMMSE never loses to LS.

Next: [Chapter 4 — Residual CNN](04_residual_cnn.md)
