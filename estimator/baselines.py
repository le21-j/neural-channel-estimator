"""LS and LMMSE channel-estimation baselines on the comb-4 pilot grid.

Uses the installed Sionna 2.x estimators (probed via
dir(sionna.phy.ofdm.channel_estimation)): LSChannelEstimator with linear
interpolation for LS, and LSChannelEstimator + LMMSEInterpolator with
ANALYTIC TDL covariance matrices (tdl_freq_cov_mat / tdl_time_cov_mat) for
LMMSE — i.e. genie channel statistics, the strongest honest baseline.
Closed form per spec: H_lmmse = R_hh (R_hh + sigma^2 I)^-1 H_ls.

Pilots are reproduced bit-exact from data/generate.py (PILOT_SEED=42,
comb-4 on symbols [2, 11], draw order: symbol-major, subcarrier-ascending —
matching Sionna's row-major PilotPattern fill order).
"""
import sys
from pathlib import Path

import torch

from sionna.phy.ofdm import (LMMSEInterpolator, LSChannelEstimator,
                             PilotPattern, ResourceGrid, tdl_freq_cov_mat,
                             tdl_time_cov_mat)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from generate import PILOT_SEED, load_cfgs, qpsk  # noqa: E402


def build_pilot_pattern(sys_cfg):
    """Comb-4 PilotPattern identical to the dataset generator's pilots."""
    S, F = sys_cfg["symbols_per_slot"], sys_cfg["fft_size"]
    sc = torch.arange(0, F, sys_cfg["pilot_comb_spacing"])
    mask = torch.zeros(1, 1, S, F, dtype=torch.bool)
    pilot_gen = torch.Generator().manual_seed(PILOT_SEED)
    vals = []
    for sym in sys_cfg["pilot_symbol_indices"]:
        mask[0, 0, sym, sc] = True
        vals.append(qpsk((len(sc),), pilot_gen))
    pilots = torch.cat(vals).reshape(1, 1, -1)
    return PilotPattern(mask, pilots)


def build_rg(sys_cfg):
    return ResourceGrid(num_ofdm_symbols=sys_cfg["symbols_per_slot"],
                        fft_size=sys_cfg["fft_size"],
                        subcarrier_spacing=sys_cfg["scs_khz"] * 1e3,
                        cyclic_prefix_length=sys_cfg["cp"],
                        pilot_pattern=build_pilot_pattern(sys_cfg))


class Baselines:
    """LS (linear interp) and per-Doppler genie-statistics LMMSE estimators."""

    def __init__(self, sys_cfg, ch_cfg):
        self.sys_cfg, self.ch_cfg = sys_cfg, ch_cfg
        self.rg = build_rg(sys_cfg)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="lin")
        model = ch_cfg["profile"].split("-")[1]
        self.cov_freq = tdl_freq_cov_mat(model, sys_cfg["scs_khz"] * 1e3,
                                         sys_cfg["fft_size"],
                                         ch_cfg["delay_spread_ns"] * 1e-9,
                                         precision="single")
        self._lmmse = {}  # f_d -> estimator (cov_time depends on Doppler)

    def _lmmse_est(self, f_d):
        if f_d not in self._lmmse:
            sys_cfg, ch_cfg = self.sys_cfg, self.ch_cfg
            fc = ch_cfg["carrier_frequency_ghz"] * 1e9
            speed = f_d * 3e8 / fc
            # 37.5 us = normal-CP symbol duration; distinct from system.yaml
            # symbol_us=35.7 (mean over slot incl. extended-CP symbol 0).
            # Correct quantity for tdl_time_cov_mat — do NOT "fix" to 35.7.
            sym_dur = (sys_cfg["fft_size"] + sys_cfg["cp"]) / (
                sys_cfg["fft_size"] * sys_cfg["scs_khz"] * 1e3)
            cov_time = tdl_time_cov_mat(ch_cfg["profile"].split("-")[1], speed,
                                        fc, sym_dur, sys_cfg["symbols_per_slot"],
                                        precision="single")
            # f-t: frequency interp first (16 comb pilots, well-conditioned),
            # then time — measured 0.189 vs 0.268 NMSE for t-f at 1600 Hz
            interp = LMMSEInterpolator(self.rg.pilot_pattern, cov_time,
                                       self.cov_freq, order="f-t")
            self._lmmse[f_d] = LSChannelEstimator(self.rg, interpolator=interp)
        return self._lmmse[f_d]

    @staticmethod
    def _io(est, y, no):
        # y complex (B, S, F) -> sionna layout (B, rx=1, rx_ant=1, S, F)
        h_hat, _ = est(y.unsqueeze(1).unsqueeze(1), torch.as_tensor(no))
        return h_hat.reshape(y.shape)  # (B, S, F)

    def ls(self, y, no):
        return self._io(self.ls_est, y, no)

    def lmmse(self, y, no, f_d):
        return self._io(self._lmmse_est(f_d), y, no)


def nmse(h_hat, h):
    """E[||H_hat - H||^2] / E[||H||^2] over the full grid."""
    return float((abs(h_hat - h) ** 2).sum() / (abs(h) ** 2).sum())


if __name__ == "__main__":
    # self-test: high SNR, LS should be accurate at low Doppler; LMMSE <= LS
    from generate import gen_batch
    sys_cfg, ch_cfg = load_cfgs()
    bl = Baselines(sys_cfg, ch_cfg)
    gen = torch.Generator().manual_seed(123)
    for f_d in (10.0, 1600.0):
        y, h, _ = gen_batch(64, f_d, 20.0, sys_cfg, ch_cfg, gen)
        no = 10.0 ** (-20.0 / 10.0)
        e_ls, e_lm = nmse(bl.ls(y, no), h), nmse(bl.lmmse(y, no, f_d), h)
        print(f"f_d={f_d:6.0f} Hz @ 20 dB:  LS NMSE={e_ls:.4f}  LMMSE NMSE={e_lm:.4f}")
        assert e_lm <= e_ls * 1.05, "LMMSE should not lose to LS"
    print("baselines self-test OK")
