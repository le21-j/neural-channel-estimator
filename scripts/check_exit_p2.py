"""P1-PROMPT-2 EXIT check: model beats LS at high Doppler.

Reads results/nmse_summary.json (written by estimator/evaluate.py) and
asserts results/nmse_curves.png exists. Prints the NMSE gap.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(msg):
    print(f"EXIT CHECK FAIL: {msg}")
    sys.exit(1)


png = ROOT / "results" / "nmse_curves.png"
js = ROOT / "results" / "nmse_summary.json"
if not png.exists():
    fail(f"missing {png}")
if not js.exists():
    fail(f"missing {js}")

summary = json.loads(js.read_text())
curves = summary["curves"]
high = str(max(int(k) for k in curves))   # highest doppler bin evaluated
cells = curves[high]

worse = [snr for snr, c in cells.items() if c["model"] >= c["ls"]]
ls_mean = sum(c["ls"] for c in cells.values()) / len(cells)
model_mean = sum(c["model"] for c in cells.values()) / len(cells)
lmmse_mean = sum(c["lmmse"] for c in cells.values()) / len(cells)
gap_db = 10 * math.log10(model_mean / ls_mean)

print(f"high-Doppler bin: {high} Hz over SNRs {sorted(cells, key=float)}")
print(f"mean NMSE  LS={ls_mean:.4f}  LMMSE={lmmse_mean:.4f}  model={model_mean:.4f}")
print(f"NMSE gap (model vs LS): {gap_db:+.2f} dB")
if worse:
    print(f"note: model >= LS at SNR(s) {worse}")

if model_mean >= ls_mean:
    fail(f"model mean NMSE {model_mean:.4f} does not beat LS {ls_mean:.4f} at {high} Hz")
print("EXIT CHECK PASS")
