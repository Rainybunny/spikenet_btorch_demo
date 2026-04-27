#!/bin/bash
#SBATCH --job-name=chen2021_dyn_scan
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=spikenet_btorch_demo/static_gaussian/slurm_logs/%x_%j.out
#SBATCH --error=spikenet_btorch_demo/static_gaussian/slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p spikenet_btorch_demo/static_gaussian/slurm_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="spikenet_btorch_demo/static_gaussian/scan_chen2021_dynamics_${RUN_TAG}"

# One job, small parallelism (2 workers) to avoid heavy cluster usage.
conda run -n spike_gpu python spikenet_btorch_demo/scan_chen2021_dynamics.py \
  --device cuda \
  --workers 2 \
  --T-ms 1200 \
  --stim-start-ms 200 \
  --output-root "${OUT_ROOT}" \
  --run-full-on-minimal \
  --full-T-ms 10000 \
  --full-stim-start-ms 4000

echo "Scan finished. Outputs: ${OUT_ROOT}"
