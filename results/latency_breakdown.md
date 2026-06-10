# Latency breakdown

PLACEHOLDER: awaiting remote GPU run.

Populate on the GPU host:

```bash
bash profiling/run_nsight.sh
python profiling/latency_report.py
python profiling/budget.py --e2e-us <e2e from ./estimator output>
```

This file is overwritten by `latency_report.py` with the per-stage
H2D / kernel / D2H table. No numbers are hand-entered, ever.
