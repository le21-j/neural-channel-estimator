.PHONY: data train export cpp profile
data:
	.venv/bin/python data/generate.py --n 1000
	.venv/bin/python scripts/check_exit_p1.py 1000
train export:
	@echo "not implemented — run /run-prompt N in Claude Code"
cpp profile:
	@echo "GPU_STEP: requires Linux + CUDA/TensorRT host. Write-only on macOS."
