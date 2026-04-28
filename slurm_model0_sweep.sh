#!/bin/bash
#SBATCH --job-name=chen2021_sweep2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Chen & Gong 2021 parameter sweep (round 2) with ChemSyn model-0 synapse.
#
# New in this sweep:
#   ee_scale  : EE mean weight g_mu scaling — key for bump attractor formation
#   inh_scale : focus on lower range (0.5-1.0) which helps bump formation
#   ext_scale : broader range including higher drives
#   noise_cv  : noise to enable attractor switching
#
# tau_scale dropped: not critical for switching.
# Each run: T_ms=10000, stim_start=4000ms.  4 parallel workers.

# Hardcode script directory to avoid Slurm working-dir resolution issue.
SCRIPT_DIR="/home/wangxukang/mice_visual_lab/spikenet_btorch_demo"

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spike_gpu

PYTHON=$(conda run -n spike_gpu which python)
OUT_ROOT="${SCRIPT_DIR}/static_gaussian/scan_model0_sweep2"

echo "============================================================"
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: ${PYTHON}"
echo "Output root: ${OUT_ROOT}"
echo "Start: $(date)"
echo "============================================================"

conda run -n spike_gpu python "${SCRIPT_DIR}/scan_chen2021_dynamics.py" \
    --python-exec "${PYTHON}" \
    --device cuda \
    --seed 42 \
    --T-ms 10000 \
    --stim-start-ms 4000 \
    --workers 4 \
    --output-root "${OUT_ROOT}" \
    --ext-levels  0.80 1.00 1.20 1.50 2.00 \
    --noise-levels 0.00 0.10 0.20 0.30 \
    --inh-levels  0.50 0.60 0.70 0.80 0.90 1.00 \
    --tau-levels  1.00 \
    --ee-levels   1.00 1.50 2.00 3.00 4.00

echo "============================================================"
echo "Sweep done: $(date)"
echo "Results: ${OUT_ROOT}/scan_summary.csv"
echo "============================================================"
