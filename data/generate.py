"""Generate (rx_pilots, true_H) dataset from a Sionna 2.x TDL-C OFDM link.

Grid from config/system.yaml (mu=1, 30 kHz SCS, 14 symbols, comb-4 pilots on
symbols [2, 11]); channel physics from sionna.phy.channel.tr38901.TDL via
GenerateOFDMChannel (PyTorch backend). The received grid is formed in the
frequency domain as y = H * x + n, which is exactly what ApplyOFDMChannel
computes — done explicitly here so pilot placement stays config-driven.

Usage:
  python data/generate.py --n 1000 [--seed 0] [--out data/dataset]
  python data/generate.py --n 1000 --dmrs [--out data/dataset_dmrs]

--dmrs flag: switch from the default comb-4 pilots to a 5G-NR Type-1 DMRS
  (3GPP TS 38.211 §6.4.1.1) comb-2 pattern — every 2nd subcarrier on the same
  pilot symbols [2, 11] (mapping table §6.4.1.1.3-1, port p=0, cdm group 0).
  Doubles pilot density (32 -> 64 pilot REs/slot, 7.1% overhead) at the cost
  of data RE capacity, matching the NR physical downlink shared channel (PDSCH)
  DMRS configuration type 1.  Output array shapes are unchanged; only the RE
  mask differs.  Default: OFF (existing comb-4 behaviour preserved).
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


def build_tx_grid(n, sys_cfg, gen, dmrs=False):
    """Unit-power QPSK grid: fixed pilots at comb REs, random data elsewhere.

    dmrs=False (default): comb-4 from system.yaml (pilot_comb_spacing=4).
    dmrs=True : 5G-NR Type-1 DMRS comb-2, port p=0, CDM group 0
                (3GPP TS 38.211 §6.4.1.1.3-1: subcarriers 0,2,4,...,F-2
                 on pilot_symbol_indices; same fixed-QPSK seeding convention).
    """
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    x = qpsk((n, S, F), gen)
    pilot_gen = torch.Generator().manual_seed(PILOT_SEED)
    if dmrs:
        # NR DMRS Type-1, port 0, CDM group 0: even subcarriers (comb-2)
        sc = torch.arange(0, F, 2)
    else:
        sc = torch.arange(0, F, sys_cfg["pilot_comb_spacing"])
    for sym in sys_cfg["pilot_symbol_indices"]:
        x[:, sym, sc] = qpsk((len(sc),), pilot_gen)  # same pilots every sample
    mask = torch.zeros(S, F, dtype=torch.bool)
    mask[torch.tensor(sys_cfg["pilot_symbol_indices"]).unsqueeze(1), sc] = True
    return x, mask


def gen_batch(n, f_d, snr, sys_cfg, ch_cfg, gen, dmrs=False):
    """n realizations at Doppler f_d. snr: (lo, hi) uniform draw or fixed float, dB.

    Returns (y_masked, h) complex64 (n, S, F) and snr_db (n,).
    """
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    fc = ch_cfg["carrier_frequency_ghz"] * 1e9
    v = f_d * C / fc  # m/s
    rg = ResourceGrid(num_ofdm_symbols=S, fft_size=F,
                      subcarrier_spacing=sys_cfg["scs_khz"] * 1e3,
                      cyclic_prefix_length=sys_cfg["cp"])
    tdl = TDL(model=ch_cfg["profile"].split("-")[1],
              delay_spread=ch_cfg["delay_spread_ns"] * 1e-9,
              carrier_frequency=fc, min_speed=v, max_speed=v)
    h = GenerateOFDMChannel(tdl, rg)(batch_size=n).reshape(n, S, F)  # complex64
    x, mask = build_tx_grid(n, sys_cfg, gen, dmrs=dmrs)
    if isinstance(snr, (tuple, list)):
        snr_db = torch.rand(n, generator=gen) * (snr[1] - snr[0]) + snr[0]
    else:
        snr_db = torch.full((n,), float(snr))
    no = 10.0 ** (-snr_db / 10.0)
    noise = torch.sqrt(no / 2).reshape(n, 1, 1) * (
        torch.randn(n, S, F, generator=gen)
        + 1j * torch.randn(n, S, F, generator=gen))
    y = h * x + noise.to(torch.complex64)
    y_masked = torch.where(mask, y, torch.zeros((), dtype=y.dtype))
    return y_masked, h, snr_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--dmrs", action="store_true",
                    help="Use 5G-NR Type-1 DMRS comb-2 pilot pattern (default: comb-4)")
    args = ap.parse_args()

    sys_cfg, ch_cfg = load_cfgs()
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    fc = ch_cfg["carrier_frequency_ghz"] * 1e9
    sweep = ch_cfg["doppler_sweep_hz"]
    snr_range = tuple(ch_cfg["snr_db_range"])

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    pilot_desc = ("NR DMRS Type-1 comb-2 (§6.4.1.1.3-1, port 0)"
                  if args.dmrs else "comb-4 (system.yaml)")
    print(f"pilot pattern: {pilot_desc}")

    # round-robin split of n across doppler bins
    counts = [args.n // len(sweep)] * len(sweep)
    for i in range(args.n - sum(counts)):
        counts[i] += 1

    rx_list, h_list, dop_list, snr_list = [], [], [], []
    for f_d, cnt in zip(sweep, counts):
        done = 0
        while done < cnt:
            b = min(CHUNK, cnt - done)
            y_masked, h, snr_db = gen_batch(b, f_d, snr_range, sys_cfg, ch_cfg, gen,
                                            dmrs=args.dmrs)
            rx_list.append(y_masked)
            h_list.append(h)
            dop_list.append(torch.full((b,), float(f_d)))
            snr_list.append(snr_db)
            done += b
        v = f_d * C / fc
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
