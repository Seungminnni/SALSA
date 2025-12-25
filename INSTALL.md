# SALSA — Installation & Run Guide

## Quick (conda)
Recommended: use the included `requirements.txt` with conda:

```bash
conda create --name lattice_env --file requirements.txt
conda activate lattice_env
```

This will get compatible binaries (including PyTorch + CUDA) for many setups.

## Quick (pip)
If you prefer pip/venv, create a venv and install generic dependencies, but install PyTorch with the official command matching your CUDA:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-pip.txt
# Example for CUDA 11.3 (change to version that matches your GPU/drivers):
pip install --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.9.0+cu113 torchvision==0.10.0+cu113
```

## Required / Recommended packages
- numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, tqdm, joblib
- PyTorch (install the wheel that matches your CUDA toolkit)
- `matplotlib` is required to generate `metrics_*.png` via `plot_metrics.py`.

## Optional but useful
- `git-lfs` — recommended for storing large checkpoints. Install system-wide then run:
  ```bash
  git lfs install
  git lfs track "*.pth" "*.pt" "*.ckpt"
  ```
  If large model files are already committed, consider `git lfs migrate import --include="*.pth,*.pt"` (CAUTION: rewrites history).

- NVIDIA Apex (optional) — provides mixed precision and fused ops. Install from source:
  https://github.com/NVIDIA/apex

## Running SALSA
- Default training (GPU):
  ```bash
  python3 train.py --exp_name myrun
  ```
- CPU only (for debugging / small runs):
  ```bash
  python3 train.py --cpu True --exp_name myrun_cpu
  ```
- Change RLWE parameters (example N=50, Q=4099):
  ```bash
  python3 train.py --N 50 --Q 4099 --exp_name myrun_n50_q4099
  ```
- Export data (no training):
  ```bash
  python3 train.py --export_data True --dump_path /tmp/salsa_export --exp_name export_test
  ```

## Evaluation & plotting
- Use `plot_metrics.py` to make a PNG from a `train.log`:
  ```bash
  python3 plot_metrics.py --log path/to/train.log --out metrics.png
  ```

## Running on different machines / cluster
- Use `--num_workers` and `--batch_size` to tune for each machine.
- SLURM: provided `slurm_params/*.json` examples; use the slurm wrapper if your cluster uses it.
- For multi-node/multi-gpu, use standard PyTorch Distributed patterns and set `--master_port` when needed.

## Troubleshooting
- If an evaluation metric is missing you may see a KeyError; if it occurs I can patch `src/trainer.py` to skip saving best models when metrics are missing.

