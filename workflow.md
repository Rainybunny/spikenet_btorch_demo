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

## 3. Current Results Snapshot

From finished sweep outputs and follow-up smoke runs:

1. Mean firing rate remains high across settings (roughly 200-225+ Hz; some post-fix smoke runs even higher).
2. winner_switches remains 0 in tested runs.
3. levy_like_transition remains no in tested runs.
4. Corner region dominates winner fraction in most runs.

Interpretation:

1. The "overly active neurons" issue is real and reproducible.
2. Connectivity-construction fixes alone did not solve this regime mismatch.

## 4. Key Problems (Focus)

## Problem A (Highest Priority): Persistent over-activity

Symptom:

1. Population firing rates remain near a high-activity regime across broad sweeps.
2. Dynamics are biased toward persistent dominance rather than balanced switching.

Why this is critical:

1. It invalidates intended dynamical regime for Levy-like transition analysis.
2. Downstream winner metrics become non-diagnostic if network is effectively saturated.

Most likely root cause (current hypothesis):

1. Remaining fidelity gap between current btorch synapse dynamics and SpikeNet ChemSyn model-0 transmitter/release process.
2. Current implementation uses simplified exponential PSC accumulation compared to SpikeNet release buffering/rise-decay handling.

## Problem B: Transition behavior mismatch

Symptom:

1. No robust winner switching in sweep/full runs.
2. Center and corner competition does not reproduce expected temporal handover pattern.

Why this matters:

1. Main scientific target (WTA-induced Levy-flight-like behavior) is not yet reproduced.

Likely dependency:

1. Problem B is probably downstream of Problem A (wrong dynamical regime).

## Problem C: Reproduction confidence still incomplete

Symptom:

1. Single-seed runs dominate current evidence.
2. No multi-repeat statistics or confidence intervals yet.

Why this matters:

1. Even if behavior appears in one run, robustness cannot be claimed.

## 5. Pending Actions (Planned)

Priority order:

1. Implement synapse-dynamics fidelity upgrade to align with SpikeNet ChemSyn model-0 release/update behavior.
2. Re-run smoke and full-length baseline after fidelity upgrade.
3. Re-run one-job parameter sweep under corrected dynamics.
4. Evaluate transition metrics and firing-rate regime again.
5. Add repeated-run statistics (at least 5 repeats for key conditions).

## 6. Acceptance Criteria for "Issue Resolved"

Minimum criteria:

1. Mean firing rate leaves pathological high-activity regime and matches expected qualitative regime from SpikeNet reference behavior.
2. At least one condition shows non-trivial winner switching (winner_switches > 0) with interpretable center-corner temporal dynamics.
3. Behavior reproduces across repeats (not only a single seed).
4. Final report includes both positive and negative conditions with quantitative summary tables.

## 7. Notes

1. This document is a status snapshot up to the current turn.
2. Main blocker is no longer environment or Slurm orchestration; it is model-fidelity and dynamics-regime mismatch.
