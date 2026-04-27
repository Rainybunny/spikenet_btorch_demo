# SpikeNet BTorch Demostration

## Install

```bash
mkdir test && cd test
git clone https://github.com/Rainybunny/spikenet_btorch_demo.git # this repo
git clone https://github.com/Rainybunny/SpikeNet.git # SpikeNet
git clone https://github.com/Rainybunny/btorch.git # btorch

git clone https://github.com/CFXTGJD/snnTracker.git
cd snnTracker
git checkout origin/data-eval-visualization
```

## Conda environment (reference btorch/AGENTS.md)

Use a dedicated conda env for reproducible testing.

```bash
conda create -n spike python=3.12 -y
conda run -n spike pip install numpy matplotlib scipy
conda run -n spike pip install torch --index-url https://download.pytorch.org/whl/cpu

# btorch runtime dependencies required by the demo scripts
conda run -n spike pip install jaxtyping spikingjelly pandas
```

## Chen and Gong 2021 static double-peak test (unscaled)

This test uses the unscaled Chen and Gong 2021 network size (hw=31, N_i=1000)
with two static Gaussian peaks:

1. one at center: (0, 0)
2. one at corner: (pi, pi)

Smoke run (fast verification):

```bash
conda run -n spike python spikenet_btorch_demo/chen_gong_2021_static_double_peak.py \
	--T-ms 1200 \
	--stim-start-ms 200 \
	--output-dir static_gaussian/chen_gong_2021_double_peak_smoke
```

Full run (for Levy-flight-style transition observation):

```bash
conda run -n spike python spikenet_btorch_demo/chen_gong_2021_static_double_peak.py \
	--T-ms 10000 \
	--stim-start-ms 4000 \
	--output-dir static_gaussian/chen_gong_2021_double_peak_infer
```

Expected outputs in the selected output directory:

1. `wta_levy_overview.png`: input map + early/late activity + winner timeline
2. `wta_levy_raster.png`: excitatory raster
3. `infer_summary.txt`: key statistics and transition summary
<<<<<<< HEAD
4. `activation_metadata.npz`: per-time-step activation metadata for future video rendering

## Dynamics parameter sweep in one Slurm job

This repo now includes a one-job dynamics sweep workflow for four factors:

1. external input strength (`ext_scale`)
2. noise level (`noise_cv`, Poisson-lambda jitter)
3. inhibitory strength (`inh_scale` for `g_EI/g_II`)
4. time constants (`tau_scale` for `tau_ampa_rec/tau_gaba_rec/tau_ampa_ext`)

Submit one Slurm job (small parallelism, 2 workers inside the job):

```bash
sbatch spikenet_btorch_demo/slurm_scan_chen2021_dynamics.sbatch
```

The job performs:

1. smoke-level sweep (`T=1200ms`) across the four factor groups
2. automatic minimal reproducible set selection
3. full-length run (`T=10000ms`) on the selected minimal set

Key outputs under `spikenet_btorch_demo/static_gaussian/scan_chen2021_dynamics_<timestamp>/`:

1. `scan_summary.csv`: all sweep cases and metrics
2. `minimal_repro_set.csv`: selected minimal reproducible set
3. `full/minimal_full_summary.csv`: full-length results for minimal set
4. `logs/*.log`: per-case stdout/stderr logs
=======
4. `activation_metadata.npz`: per-time-step activation metadata for future video rendering
>>>>>>> 1cb92294be061c82f0c4395da47712a51d4bbb2f
