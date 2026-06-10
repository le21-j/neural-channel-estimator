# Dataset format

Produced by `python data/generate.py --n 1000` (Sionna 2.0.1, PyTorch backend).
All arrays land in `data/dataset/` as `.npy`.

## Artifacts

| file | shape | dtype | content |
|---|---|---|---|
| `rx_pilots.npy` | `(N, 2, 14, 64)` | float32 | received grid masked to pilot REs (zeros elsewhere) |
| `true_H.npy` | `(N, 2, 14, 64)` | float32 | true per-RE channel from `GenerateOFDMChannel` |
| `doppler_hz.npy` | `(N,)` | float32 | per-sample Doppler bin (for stratified splits) |
| `snr_db.npy` | `(N,)` | float32 | per-sample SNR, uniform draw from `snr_db_range` |

Axis order: `(sample, complex_channel, ofdm_symbol, subcarrier)`.
**Channel 0 = real part, channel 1 = imag part** (complex-as-2ch-real — TensorRT/ONNX
friendly, see CLAUDE.md). Reconstruct: `H = a[:, 0] + 1j * a[:, 1]`.

## Generation model

- Grid: `config/system.yaml` — mu=1, 30 kHz SCS, 14 symbols/slot, fft_size=64, cp=8 samples.
- Pilots: comb-4 in frequency (subcarriers 0,4,...,60) on OFDM symbols **[2, 11]**;
  fixed unit-power QPSK drawn from `torch.Generator().manual_seed(42)` (`PILOT_SEED`) —
  identical across all samples so baselines can reproduce them. 32 pilot REs/slot (3.6%).
- Data REs: random unit-power QPSK (not stored; estimators see pilots only).
- Channel: TDL-C, 300 ns delay spread, fc=3.5 GHz; one TDL instance per Doppler bin with
  `min_speed = max_speed = f_D * c / fc`. Bins: 10/100/300/600/1000/1600 Hz, N split evenly.
- Receive: frequency-domain `y = H * x + n`, complex AWGN with `no = 10^(-snr_db/10)`
  (unit signal power), then masked to pilot REs.

## Sanity (measured at generation)

`|corr(sym0, sym13)|` of `true_H`: 1.000 @ 10 Hz, 0.616 @ 1600 Hz — slot-time
decorrelation grows with Doppler as expected.
