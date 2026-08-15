# RACER-C seed-99 active GPU benchmark entry point

Status: **pre-freeze technical benchmark only**.

The active target is the user's local Windows laptop with an NVIDIA GeForce RTX
4060 Laptop GPU. The former Linux RTX-4090 assumption is superseded and must not
be used for the current benchmark.

Follow the complete PowerShell instructions in:

`seed99_gpu_benchmark_windows_rtx4060_runbook.md`

The active default lock is:

`configs/racer_c/gpu_environment_lock.yaml`

It is byte-for-byte equivalent to the explicit platform lock:

`configs/racer_c/gpu_environment_windows_rtx4060.yaml`

The workflow may read labels only for `D_dev`. It must not generate policy,
conformal, or test predictions, calculate a performance metric, run seeds
101--110, or create a protocol tag.
