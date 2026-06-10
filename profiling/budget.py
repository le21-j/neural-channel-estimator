"""Express measured e2e latency against the 5G mu=1 timing budget (LOCAL-safe).

Symbol budget: 35.7 us (mean symbol period incl. CP = 0.5 ms slot / 14).
Slot budget: 0.5 ms. Reads numbers from config/system.yaml — no hardcoding.

Usage: python profiling/budget.py --e2e-us 12.3
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-us", type=float, required=True,
                    help="measured end-to-end latency in microseconds")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "system.yaml").read_text())
    slot_us = cfg["slot_ms"] * 1e3
    symbol_us = cfg["symbol_us"]

    print(f"e2e = {args.e2e_us:.2f} us")
    print(f"  = {100 * args.e2e_us / symbol_us:6.1f}% of one OFDM symbol "
          f"({symbol_us} us)")
    print(f"  = {100 * args.e2e_us / slot_us:6.1f}% of one slot ({slot_us:.0f} us)")


if __name__ == "__main__":
    main()
