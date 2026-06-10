"""Generate (rx_pilots, true_H) dataset from a Sionna 2.x TDL-C OFDM link.

Grid from config/system.yaml (mu=1, 30 kHz SCS, 14 symbols, comb-4 pilots on
symbols [2, 11]); channel physics from sionna.phy.channel.tr38901.TDL via
GenerateOFDMChannel (PyTorch backend). The received grid is formed in the
frequency domain as y = H * x + n, which is exactly what ApplyOFDMChannel
computes — done explicitly here so pilot placement stays config-driven.

Usage: python data/generate.py --n 1000 [--seed 0] [--out data/dataset]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import ResourceGrid

ROOT = Path(__file__).resolve().parents[1]
C = 3e8
PILOT_SEED = 42  # fixed: Prompt-2 baselines must reproduce the same pilots
CHUNK = 250


def load_cfgs():
    sys_cfg = yaml.safe_load((ROOT / "config" / "system.yaml").read_text())
    ch_cfg = yaml.safe_load((ROOT / "config" / "channel.yaml").read_text())
    return sys_cfg, ch_cfg


def qpsk(shape, gen):
    bits = torch.randint(0, 2, shape + (2,), generator=gen, dtype=torch.float32)
    sym = (1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])
    return (sym / np.sqrt(2)).to(torch.complex64)


def build_tx_grid(n, sys_cfg, gen):
    """Unit-power QPSK grid: fixed pilots at comb REs, random data elsewhere."""
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    x = qpsk((n, S, F), gen)
    pilot_gen = torch.Generator().manual_seed(PILOT_SEED)
    sc = torch.arange(0, F, sys_cfg["pilot_comb_spacing"])
    for sym in sys_cfg["pilot_symbol_indices"]:
        x[:, sym, sc] = qpsk((len(sc),), pilot_gen)  # same pilots every sample
    mask = torch.zeros(S, F, dtype=torch.bool)
    mask[torch.tensor(sys_cfg["pilot_symbol_indices"]).unsqueeze(1), sc] = True
    return x, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "dataset"))
    args = ap.parse_args()

    sys_cfg, ch_cfg = load_cfgs()
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    fc = ch_cfg["carrier_frequency_ghz"] * 1e9
    sweep = ch_cfg["doppler_sweep_hz"]
    snr_lo, snr_hi = ch_cfg["snr_db_range"]

    rg = ResourceGrid(num_ofdm_symbols=S, fft_size=F,
                      subcarrier_spacing=sys_cfg["scs_khz"] * 1e3,
                      cyclic_prefix_length=sys_cfg["cp"])

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # round-robin split of n across doppler bins
    counts = [args.n // len(sweep)] * len(sweep)
    for i in range(args.n - sum(counts)):
        counts[i] += 1

    rx_list, h_list, dop_list, snr_list = [], [], [], []
    for f_d, cnt in zip(sweep, counts):
        v = f_d * C / fc  # m/s
        tdl = TDL(model=ch_cfg["profile"].split("-")[1],
                  delay_spread=ch_cfg["delay_spread_ns"] * 1e-9,
                  carrier_frequency=fc, min_speed=v, max_speed=v)
        gen_h = GenerateOFDMChannel(tdl, rg)
        done = 0
        while done < cnt:
            b = min(CHUNK, cnt - done)
            h = gen_h(batch_size=b).reshape(b, S, F)          # complex64
            x, mask = build_tx_grid(b, sys_cfg, gen)
            snr_db = torch.rand(b, generator=gen) * (snr_hi - snr_lo) + snr_lo
            no = 10.0 ** (-snr_db / 10.0)
            noise = torch.sqrt(no / 2).reshape(b, 1, 1) * (
                torch.randn(b, S, F, generator=gen)
                + 1j * torch.randn(b, S, F, generator=gen))
            y = h * x + noise.to(torch.complex64)
            rx_list.append(torch.where(mask, y, torch.zeros((), dtype=y.dtype)))
            h_list.append(h)
            dop_list.append(torch.full((b,), float(f_d)))
            snr_list.append(snr_db)
            done += b
        print(f"doppler {f_d:>5.0f} Hz (v={v*3.6:6.1f} km/h): {cnt} realizations")

    def to_2ch(t):  # complex (N,S,F) -> float32 (N,2,S,F)
        t = torch.cat(t)
        return torch.stack([t.real, t.imag], dim=1).numpy().astype(np.float32)

    out = {"rx_pilots": to_2ch(rx_list), "true_H": to_2ch(h_list),
           "doppler_hz": torch.cat(dop_list).numpy().astype(np.float32),
           "snr_db": torch.cat(snr_list).numpy().astype(np.float32)}

    exp = (args.n, 2, S, F)
    for name in ("rx_pilots", "true_H"):
        assert out[name].shape == exp, f"{name} shape {out[name].shape} != {exp}"
        assert not np.isnan(out[name]).any(), f"NaNs in {name}"
    assert out["doppler_hz"].shape == (args.n,) and out["snr_db"].shape == (args.n,)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in out.items():
        np.save(out_dir / f"{name}.npy", arr)
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
