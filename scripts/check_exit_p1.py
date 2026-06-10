"""P1-PROMPT-1 EXIT check: dataset exists, no NaNs, shapes match config.

Run AFTER `python data/generate.py --n 1000`. Independent verification —
loads artifacts from disk and re-derives expected shapes from config/*.yaml.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
N_EXPECTED = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def fail(msg):
    print(f"EXIT CHECK FAIL: {msg}")
    sys.exit(1)


sys_cfg = yaml.safe_load((ROOT / "config" / "system.yaml").read_text())
ch_cfg = yaml.safe_load((ROOT / "config" / "channel.yaml").read_text())

for k in ("fft_size", "cp"):
    if sys_cfg.get(k) is None:
        fail(f"config/system.yaml `{k}` still null — Prompt 1 must set it")
for k in ("doppler_sweep_hz", "snr_db_range"):
    if ch_cfg.get(k) is None:
        fail(f"config/channel.yaml `{k}` still null — Prompt 1 must set it")

S = sys_cfg["symbols_per_slot"]   # 14
F = sys_cfg["fft_size"]
exp_grid = (N_EXPECTED, 2, S, F)  # complex split into 2 real channels

ds = ROOT / "data" / "dataset"
arrs = {}
for name in ("rx_pilots", "true_H", "doppler_hz", "snr_db"):
    p = ds / f"{name}.npy"
    if not p.exists():
        fail(f"missing artifact {p}")
    arrs[name] = np.load(p)

for name in ("rx_pilots", "true_H"):
    a = arrs[name]
    if a.shape != exp_grid:
        fail(f"{name} shape {a.shape} != expected {exp_grid}")
    if a.dtype != np.float32:
        fail(f"{name} dtype {a.dtype} != float32")
    if np.isnan(a).any():
        fail(f"{name} contains NaNs")
    if not np.isfinite(a).all():
        fail(f"{name} contains infs")

for name in ("doppler_hz", "snr_db"):
    a = arrs[name]
    if a.shape != (N_EXPECTED,):
        fail(f"{name} shape {a.shape} != ({N_EXPECTED},)")
    if np.isnan(a).any():
        fail(f"{name} contains NaNs")

if set(np.unique(arrs["doppler_hz"])) - set(float(d) for d in ch_cfg["doppler_sweep_hz"]):
    fail("doppler_hz values outside config sweep")

if np.allclose(arrs["true_H"], 0):
    fail("true_H all zeros — channel not applied")
pilot_energy = float(np.square(arrs["rx_pilots"]).mean())
if pilot_energy == 0.0:
    fail("rx_pilots all zeros")

print("EXIT CHECK PASS")
for name, a in arrs.items():
    print(f"  {name}: shape={a.shape} dtype={a.dtype}")
print(f"  rx_pilots mean energy: {pilot_energy:.4e}")
print(f"  doppler bins: {sorted(set(arrs['doppler_hz'].tolist()))}")
