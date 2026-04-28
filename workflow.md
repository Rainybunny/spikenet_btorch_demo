# SpikeNet BTorch Demo Workflow (Current Status)

## 1. Scope and Goal

Current task scope:

1. Reproduce Chen and Gong 2021 unscaled static double-peak experiment in btorch.
2. Add parameter sweep workflow (external input strength, noise, inhibitory strength, time constants).
3. Run sweep on Slurm in one job with small internal parallelism.
4. Diagnose why expected Levy-like switching is not observed and why activity is overly high.

Target behavior:

1. Stable GPU execution.
2. Reasonable firing-rate regime (not near refractory-limit saturation).
3. Reproducible winner-transition dynamics (Levy-like switching evidence if model fidelity is sufficient).

## 2. Completed Work

### 2.1 Environment and execution pipeline

1. Built clean GPU environment using nightly PyTorch for RTX 5090 compatibility.
2. Added one-job Slurm sweep script with low resource usage policy.
3. Confirmed cluster partition and submission settings for current server.

### 2.2 Experiment and outputs

1. Implemented Chen2021 static double-peak script with output artifacts:
   - overview figure
   - raster figure
   - infer summary
   - activation metadata
2. Added CLI sweep controls:
   - ext_scale
   - noise_cv
   - inh_scale
   - tau_scale
3. Added batch scan runner and CSV summarization:
   - scan_summary.csv
   - minimal_repro_set.csv
   - full/minimal_full_summary.csv

### 2.3 Network-construction fixes already applied

Compared against SpikeNet model scripts and utilities, then corrected:

1. Connection sampling rule:
   - from simple distance-decayed Bernoulli
   - to per-pre out-degree sampling + distance-weighted target sampling (closer to SpikeNet lattice pipeline)
2. Spatial geometry:
   - excitatory coordinates switched to centered lattice-like construction
   - inhibitory coordinates switched to quasi-lattice style (instead of plain uniform random)
3. EE degree variability control:
   - added low-CV degree control entry to reduce unrealistic out-degree spread

### 2.4 Synapse dynamics fidelity upgrade (model-0)

Root cause of over-activity identified and fixed:

1. **Root cause**: Current btorch synapse used simple exponential PSC (each spike → immediate full conductance). SpikeNet ChemSyn model-0 uses a pre-synaptic transmitter release pulse with a gating variable `s[i]`:
   - Each spike extends `trans_left[i]` by `steps_trans = Dt_trans / dt` steps
   - Each active step contributes `K_trans * (1 - s[i]) * K_weight` to post-syn conductance
   - `s[i]` builds up during release and decays with `tau_decay`
   - At high firing rates, `s[i]` → 1, so `(1 - s[i])` → 0: natural saturation prevents runaway

2. **Fix implemented** in `chen_gong_2021_static_double_peak.py`:
   - Added `Dt_trans_AMPA = 1.0 ms`, `Dt_trans_GABA = 1.0 ms` config fields
   - Added `s_pre` [1, n_total] and `trans_left` [1, n_total] tensors in sim loop
   - Each step: compute release from `s_pre`, update `s_pre`, pass `release` to synapse channels
   - AMPA channel receives `release` with E-neuron gating; GABA with I-neuron gating
   - Separate per-neuron decay: E-neurons decay with tau_ampa_rec, I-neurons with tau_gaba_rec

3. **Result** (smoke run, T=2000ms, seed=42, CPU):
   - Mean firing rate: **4.4 Hz** (was 200-225+ Hz before)
   - Network enters competitive balance: center 48.9% / corner 51.1%
   - No winner switches detected in 2s window (too short)

4. **Full-length run** (T=10000ms, stim_start=4000ms, seed=42, CPU):
   - Mean firing rate: **4.49 Hz** (stable regime)
   - center_win_fraction: 0.497, corner_win_fraction: 0.503 (near tie)
   - winner_switches: 0 (both groups compete but no min-dwell transitions yet)
   - Interpretation: network is in correct activity regime but near the transition point;
     parameter tuning via sweep needed to find robust switching conditions

## 3. Current Results Snapshot

After model-0 synapse fix:

1. Mean firing rate now in physiological range (~4.5 Hz mean; active neurons ~15-30 Hz).
2. Center/corner competition is balanced, suggesting correct dynamical regime.
3. Winner switches not yet observed; likely requires:
   - Stronger external drive (ext_scale > 1) to push firing rates higher
   - Or moderate noise (noise_cv > 0) to trigger transitions
   - Or adjusted inhibitory balance

## 4. Key Problems (Current Status)

## Problem A: RESOLVED - Persistent over-activity

Root cause identified and fixed: missing ChemSyn model-0 transmitter gating.
Mean rate now 4.5 Hz vs 200+ Hz before.

## Problem B: Transition behavior (In progress)

Symptom: winner_switches = 0 in baseline run.
Status: Network is now in correct regime. Need parameter sweep to find switching conditions.
Likely fix: scale ext_scale up (e.g. 1.0-1.4) or add noise_cv (0.1-0.3).

## Problem C: Reproduction confidence still incomplete

Single-seed runs dominate current evidence.
Pending: multi-repeat statistics after finding working parameter conditions.

## 5. Pending Actions (Planned)

Priority order:

1. ~~Implement synapse-dynamics fidelity upgrade~~ DONE.
2. Run parameter sweep on GPU (Slurm job 14832 submitted: `slurm_model0_sweep.sh`):
   - ext_scale: 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4
   - noise_cv: 0.0, 0.1, 0.2, 0.3
   - inh_scale: 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6
   - tau_scale: 0.7, 0.85, 1.0, 1.2, 1.4
   - T_ms=10000, stim_start=4000, seed=42, device=cuda
3. After sweep: identify conditions with winner_switches > 0.
4. Run 5+ repeats with different seeds on best conditions.
5. Evaluate center-first Levy-like transition criterion.

## 6. Acceptance Criteria for "Issue Resolved"

Minimum criteria:

1. Mean firing rate leaves pathological high-activity regime and matches expected qualitative regime from SpikeNet reference behavior. **ACHIEVED (4.5 Hz vs 200+ Hz).**
2. At least one condition shows non-trivial winner switching (winner_switches > 0) with interpretable center-corner temporal dynamics.
3. Behavior reproduces across repeats (not only a single seed).
4. Final report includes both positive and negative conditions with quantitative summary tables.

## 7. Notes

1. ChemSyn model-0 pre-syn gating is the critical ingredient for correct dynamics regime.
2. SpikeNet default parameters: Dt_trans_AMPA=1.0ms, Dt_trans_GABA=1.0ms, tau_decay_AMPA=5.0ms, tau_decay_GABA=3.0ms. Our demo uses tau_ampa_rec=5.8ms, tau_gaba_rec=6.5ms from Chen&Gong 2021 calibration.
3. External input handling (btorch: instantaneous accumulation + exp decay) vs SpikeNet (spread over steps_trans): minor difference, does not cause over-activity.
4. Slurm output: `chen2021_model0_sweep_{JOBID}.out` in demo directory.
