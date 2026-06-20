---
type: kernel-doc
title: "Chapter 1: OFDM numerology & configs"
tags: [channel-est, ofdm, numerology, configs]
timestamp: 2026-06-17
---

# Chapter 1: OFDM numerology & configs

Everything in this repo hangs off three small YAML files in `config/`. Think of
them as the *physics constitution*: every script re-derives its expectations
from these numbers, so changing a value here re-shapes the whole pipeline.

## The numerology (system.yaml)

5G numerology $\mu=1$ means **30 kHz subcarrier spacing** (SCS). That fixes time:

- slot = 0.5 ms, containing **14 OFDM symbols**
- mean symbol period incl. cyclic prefix = $0.5\,\text{ms}/14 \approx 35.7\,\mu s$
- useful symbol $T_u = 1/\Delta f = 33.33\,\mu s$

```yaml
mu: 1
scs_khz: 30
symbols_per_slot: 14
fft_size: 64       # 64 subcarriers, no guards
cp: 8              # samples @ fs = 64*30kHz = 1.92 MHz -> 4.17 us
pilot_symbol_indices: [2, 11]
pilot_comb_spacing: 4
```

Why `cp: 8`? The cyclic prefix must outlast the channel's delay spread —
TDL-C at 300 ns rms has ~2.6 µs max excess delay; 8 samples = 4.17 µs covers it.

## The pilot pattern

Pilots are *known* symbols sprinkled on the grid so the receiver can sample the
channel. Ours is **comb-4**: every 4th subcarrier, on symbols 2 and 11 only —
32 pilot resource elements out of 896 (3.6% overhead). Like checking road
conditions from 32 weather stations and inferring the whole map.

## The channel (channel.yaml)

```yaml
profile: TDL-C
delay_spread_ns: 300
carrier_frequency_ghz: 3.5
doppler_sweep_hz: [10, 100, 300, 600, 1000, 1600]
snr_db_range: [0, 25]
```

Doppler is the villain of this project: 1600 Hz ≈ 494 km/h at 3.5 GHz. The
channel then changes *within one slot* — and the two pilot symbols can't track
it by interpolation alone. That's the gap the CNN exploits.

> Every consumer (generator, EXIT checks, baselines) loads these files fresh —
> there are no hard-coded copies of these numbers anywhere else.

Next: [Chapter 2 — Dataset generator](02_dataset_generator.md)
