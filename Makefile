.PHONY: data train export cpp profile
data:
	.venv/bin/python data/generate.py --n 1000
	.venv/bin/python scripts/check_exit_p1.py 1000
train:
	WANDB_SILENT=true .venv/bin/python estimator/train.py
export:
	.venv/bin/python estimator/export_onnx.py
	.venv/bin/python scripts/verify_parity.py
cpp profile:
	@echo "GPU_STEP: requires Linux + CUDA/TensorRT host. Write-only on macOS."
