"""Chen and Gong 2021 static double-peak WTA test (btorch reproduction).

This script runs the unscaled Chen and Gong 2021 network (hw=31, N_i=1000)
with two static Gaussian feedforward inputs:
- one at scene center (0, 0)
- one at scene corner (pi, pi)

The goal is to reproduce a winner-take-all style Levy-flight-like transition:
center-adjacent neurons dominate first, then are suppressed, and corner-adjacent
neurons become dominant later.

Outputs are saved in static_gaussian-compatible style:
- infer_summary.txt
- wta_levy_overview.png
- wta_levy_raster.png
- activation_metadata.npz (time-step activation metadata for future videos)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for subdir in ("btorch",):
    p = ROOT / subdir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btorch.models import environ, functional
from btorch.models.neurons.spikenet import SpikeNetNeuron
from btorch.models.synapse import SpikeNetCompositePSC


@dataclass
class ChenGong2021Cfg:
    # Unscaled model size
    hw: int = 31
    N_i: int = 1000
    seed: int = 42

    # Simulation
    dt: float = 0.1
    T_ms: float = 10000.0
    stim_start_ms: float = 4000.0

    # Neuron (Chen & Gong 2021 / SpikeNet defaults)
    tau_ref: float = 4.0
    v_threshold: float = -50.0
    v_reset: float = -60.0
    v_lk: float = -70.0
    c_m: float = 0.25
    g_lk: float = 0.0167
    spike_freq_adapt: bool = True
    dg_k: float = 0.003
    tau_k: float = 80.0
    v_k: float = -85.0

    # Connectivity
    delay_max_ms: float = 4.0
    P0_init: float = 0.08
    tau_c_EE: float = 8.0
    tau_c_IE: float = 10.0
    tau_c_I: float = 20.0
    P_ei: float = 0.20
    P_ie: float = 0.20
    P_ii: float = 0.40
    ee_degree_cv: float = 0.20

    # SpikeNet ChemSyn model-0 transmitter release pulse duration (ms)
    # Dt_trans = 1.0ms → steps_trans = 10 at dt=0.1ms
    # The (1-s) gating variable prevents runaway excitation at high firing rates.
    Dt_trans_AMPA: float = 1.0
    Dt_trans_GABA: float = 1.0

    # Conductances (uS)
    g_mu: float = 4e-3
    g_EI: float = 13.5e-3
    g_IE: float = 5e-3
    g_II: float = 25e-3
    g_balance: float = 0.98

    # Reversal potentials (mV)
    E_ampa: float = 0.0
    E_gaba: float = -80.0

    # Synaptic decay (ms)
    tau_ampa_rec: float = 5.8
    tau_gaba_rec: float = 6.5
    tau_ampa_ext: float = 5.8

    # External input
    N_ext: int = 1000
    g_ext: float = 2e-3
    rate_ext_E: float = 0.85
    rate_ext_I: float = 1.0
    # Extra per-step input-rate jitter (coefficient of variation) for sweep use.
    lambda_noise_cv: float = 0.0

    # Initial condition from Chen2021 script (writeInitVHDF5 p_fire)
    init_fire_prob_e: float = 0.1
    init_fire_prob_i: float = 0.0

    # Double static Gaussian stimulus
    qisig: float = 0.6
    center_pos_yx: tuple[float, float] = (0.0, 0.0)
    corner_pos_yx: tuple[float, float] = (math.pi, math.pi)
    center_contrast: float = 1.0
    corner_contrast: float = 1.0

    # Winner metadata extraction
    region_sigma_mult: float = 1.5
    ema_tau_ms: float = 100.0
    winner_eps_hz: float = 0.5
    min_dwell_ms: float = 120.0


DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _torus_dist2_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return (diff ** 2).sum(axis=2)


def _sample_edges_distance_weighted(
    coords_pre: np.ndarray,
    coords_post: np.ndarray,
    p0: float,
    tau_c: float,
    spacing: float,
    rng: np.random.Generator,
    forbid_self: bool,
    degree_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample directed edges with SpikeNet-like lattice rule.

    For each pre neuron i:
    1) sample out-degree k_i ~ Poisson(N_post * p0)
    2) sample k_i unique post neurons with probabilities proportional to
       exp(-dist(i,j) / tau_c), where dist is torus Euclidean distance
       measured in lattice steps.
    """
    n_pre = coords_pre.shape[0]
    n_post = coords_post.shape[0]
    mean_out = max(0.0, n_post * p0)
    tau_rad = max(tau_c * spacing, 1e-9)

    diff = coords_post[np.newaxis, :, :] - coords_pre[:, np.newaxis, :]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt((diff**2).sum(axis=2), dtype=np.float32)  # [N_pre, N_post]
    weights = np.exp(-dist / tau_rad, dtype=np.float64)

    if forbid_self and n_pre == n_post and np.array_equal(coords_pre, coords_post):
        np.fill_diagonal(weights, 0.0)

    pre_list: list[np.ndarray] = []
    post_list: list[np.ndarray] = []
    for i_pre in range(n_pre):
        if degree_std is None:
            k = int(rng.poisson(mean_out))
        else:
            k = int(round(rng.normal(loc=mean_out, scale=max(float(degree_std), 1e-9))))
        if k <= 0:
            continue

        row = weights[i_pre]
        valid = row > 0.0
        n_valid = int(valid.sum())
        if n_valid <= 0:
            continue
        if k > n_valid:
            k = n_valid

        cand = np.flatnonzero(valid)
        prob = row[cand]
        prob_sum = prob.sum()
        if prob_sum <= 0.0:
            continue
        prob = prob / prob_sum

        chosen = rng.choice(cand, size=k, replace=False, p=prob)
        pre_list.append(np.full((k,), i_pre, dtype=np.int32))
        post_list.append(chosen.astype(np.int32))

    if not pre_list:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return np.concatenate(pre_list), np.concatenate(post_list)


def _quasi_lattice_2d(n: int, hw: int, rng: np.random.Generator) -> np.ndarray:
    """Python version of SpikeNet's quasi_lattice_2D in lattice coordinates."""
    w = 2 * hw + 1

    x = np.zeros((n,), dtype=np.float32)
    while x[0] == 0.0 or x[-1] == float(w):
        x = np.sort((rng.random(n, dtype=np.float32) * w).astype(np.float32))

    seg_n = int(round(math.sqrt(n)))
    segs = np.linspace(0.0, float(w), seg_n + 1, dtype=np.float32)
    y_chunks: list[np.ndarray] = []
    for i in range(seg_n):
        mask = (x >= segs[i]) & (x < segs[i + 1])
        x_seg = x[mask]
        if x_seg.size == 0:
            continue
        y_tmp = (rng.random(x_seg.size, dtype=np.float32) * w).astype(np.float32)
        y_order = np.argsort(y_tmp)
        y_tmp = y_tmp[y_order]
        x[mask] = x_seg[y_order]
        y_chunks.append(y_tmp)

    y = np.concatenate(y_chunks).astype(np.float32)
    lattice = np.stack([x, y], axis=1) - (w / 2.0)
    return lattice.astype(np.float32)


def _sparse_delay_buckets(
    post: np.ndarray,
    pre: np.ndarray,
    vals: np.ndarray,
    delays: np.ndarray,
    n_neuron: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    buckets: dict[int, torch.Tensor] = {}
    if post.size == 0:
        return buckets
    for d in np.unique(delays):
        mask = delays == d
        rows = torch.as_tensor(post[mask], dtype=torch.long, device=device)
        cols = torch.as_tensor(pre[mask], dtype=torch.long, device=device)
        vs = torch.as_tensor(vals[mask], dtype=torch.float32, device=device)
        w = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vs,
            size=(n_neuron, n_neuron),
        ).to_sparse_csr()
        buckets[int(d)] = w
    return buckets


def _torus_r2_to_center(coords: np.ndarray, center_yx: np.ndarray) -> np.ndarray:
    d = coords - center_yx
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return (d ** 2).sum(axis=1)


def _double_input_gain(coords_e: np.ndarray, cfg: ChenGong2021Cfg) -> np.ndarray:
    c1 = np.asarray(cfg.center_pos_yx, dtype=np.float32)
    c2 = np.asarray(cfg.corner_pos_yx, dtype=np.float32)
    r1_2 = _torus_r2_to_center(coords_e, c1)
    r2_2 = _torus_r2_to_center(coords_e, c2)
    sig2 = max(cfg.qisig * cfg.qisig, 1e-9)
    gain = (
        cfg.center_contrast * np.exp(-0.5 * r1_2 / sig2)
        + cfg.corner_contrast * np.exp(-0.5 * r2_2 / sig2)
    )
    return gain.astype(np.float32)


def _region_mask(coords_e: np.ndarray, center_yx: tuple[float, float], radius: float) -> np.ndarray:
    r2 = _torus_r2_to_center(coords_e, np.asarray(center_yx, dtype=np.float32))
    return r2 <= radius * radius


def build_network(cfg: ChenGong2021Cfg) -> tuple[SpikeNetNeuron, SpikeNetCompositePSC, int, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n_e = (2 * cfg.hw + 1) ** 2
    n_total = n_e + cfg.N_i

    delay_max_steps = max(1, int(round(cfg.delay_max_ms / cfg.dt)))
    spacing = 2 * np.pi / (2 * cfg.hw + 1)

    # SpikeNet uses centered lattice coordinates: (-hw:hw) * 2*pi/(2*hw+1)
    x1d = (np.arange(-cfg.hw, cfg.hw + 1, dtype=np.float32) * spacing).astype(np.float32)
    yy, xx = np.meshgrid(x1d, x1d, indexing="ij")
    coords_e = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    coords_i = (_quasi_lattice_2d(cfg.N_i, cfg.hw, rng) * spacing).astype(np.float32)

    lognormal_sigma = 0.3
    mu_p = math.log(cfg.g_mu) - 0.5 * (lognormal_sigma**2)

    exc_post_all: list[np.ndarray] = []
    exc_pre_all: list[np.ndarray] = []
    exc_val_all: list[np.ndarray] = []
    exc_del_all: list[np.ndarray] = []

    inh_post_all: list[np.ndarray] = []
    inh_pre_all: list[np.ndarray] = []
    inh_val_all: list[np.ndarray] = []
    inh_del_all: list[np.ndarray] = []

    # EE (SpikeNet-like: poisson out-degree + distance-weighted targets)
    print(f"  EE edges (N_e={n_e})...", flush=True)
    pre_ee, post_ee = _sample_edges_distance_weighted(
        coords_pre=coords_e,
        coords_post=coords_e,
        p0=cfg.P0_init,
        tau_c=cfg.tau_c_EE,
        spacing=spacing,
        rng=rng,
        forbid_self=True,
        degree_std=cfg.ee_degree_cv * n_e * cfg.P0_init,
    )
    n_ee = len(post_ee)
    w_ee = rng.lognormal(mu_p, lognormal_sigma, n_ee).astype(np.float32)
    del_ee = rng.integers(1, delay_max_steps + 1, n_ee).astype(np.int32)
    exc_post_all.append(post_ee)
    exc_pre_all.append(pre_ee)
    exc_val_all.append(w_ee)
    exc_del_all.append(del_ee)
    print(f"    EE edges: {n_ee:,}", flush=True)

    # Chen & Gong 2021: I->E mean weight depends on EE input to each E neuron.
    ee_input = np.zeros(n_e, dtype=np.float32)
    np.add.at(ee_input, post_ee, w_ee)

    # I->E (GABA)
    print("  I->E edges...", flush=True)
    pre_ie_i, post_ie_e = _sample_edges_distance_weighted(
        coords_pre=coords_i,
        coords_post=coords_e,
        p0=cfg.P_ie,
        tau_c=cfg.tau_c_I,
        spacing=spacing,
        rng=rng,
        forbid_self=False,
    )
    n_ie = len(post_ie_e)

    in_count_ie = np.bincount(post_ie_e, minlength=n_e).astype(np.float32)
    mu_ie_by_target = np.zeros(n_e, dtype=np.float32)
    valid = in_count_ie > 0
    mu_ie_by_target[valid] = (
        ee_input[valid] / in_count_ie[valid] * (cfg.g_EI / cfg.g_mu) * cfg.g_balance
    )
    mu_ie = mu_ie_by_target[post_ie_e]
    std_ie = mu_ie / 4.0
    w_ie = np.abs(rng.normal(loc=mu_ie, scale=std_ie)).astype(np.float32)

    del_ie = rng.integers(1, delay_max_steps + 1, n_ie).astype(np.int32)
    inh_post_all.append(post_ie_e)
    inh_pre_all.append(n_e + pre_ie_i)
    inh_val_all.append(w_ie)
    inh_del_all.append(del_ie)
    print(f"    I->E edges: {n_ie:,}", flush=True)

    # E->I (AMPA)
    print("  E->I edges...", flush=True)
    pre_ei_e, post_ei_i = _sample_edges_distance_weighted(
        coords_pre=coords_e,
        coords_post=coords_i,
        p0=cfg.P_ei,
        tau_c=cfg.tau_c_IE,
        spacing=spacing,
        rng=rng,
        forbid_self=False,
    )
    n_ei = len(post_ei_i)
    del_ei = rng.integers(1, delay_max_steps + 1, n_ei).astype(np.int32)
    exc_post_all.append(n_e + post_ei_i)
    exc_pre_all.append(pre_ei_e)
    exc_val_all.append(np.full(n_ei, cfg.g_IE, dtype=np.float32))
    exc_del_all.append(del_ei)
    print(f"    E->I edges: {n_ei:,}", flush=True)

    # I->I (GABA)
    print("  I->I edges...", flush=True)
    pre_ii_i, post_ii_i = _sample_edges_distance_weighted(
        coords_pre=coords_i,
        coords_post=coords_i,
        p0=cfg.P_ii,
        tau_c=cfg.tau_c_I,
        spacing=spacing,
        rng=rng,
        forbid_self=True,
    )
    n_ii = len(post_ii_i)
    del_ii = rng.integers(1, delay_max_steps + 1, n_ii).astype(np.int32)
    inh_post_all.append(n_e + post_ii_i)
    inh_pre_all.append(n_e + pre_ii_i)
    inh_val_all.append(np.full(n_ii, cfg.g_II, dtype=np.float32))
    inh_del_all.append(del_ii)
    print(f"    I->I edges: {n_ii:,}", flush=True)

    print("  Assembling sparse delay buckets...", flush=True)
    exc_weights = _sparse_delay_buckets(
        np.concatenate(exc_post_all),
        np.concatenate(exc_pre_all),
        np.concatenate(exc_val_all),
        np.concatenate(exc_del_all),
        n_total,
        device=DEVICE,
    )
    inh_weights = _sparse_delay_buckets(
        np.concatenate(inh_post_all),
        np.concatenate(inh_pre_all),
        np.concatenate(inh_val_all),
        np.concatenate(inh_del_all),
        n_total,
        device=DEVICE,
    )

    param = lambda v: torch.full((n_total,), v, dtype=DTYPE, device=DEVICE)

    neuron = SpikeNetNeuron(
        n_neuron=n_total,
        v_threshold=param(cfg.v_threshold),
        v_reset=param(cfg.v_reset),
        v_lk=param(cfg.v_lk),
        c_m=param(cfg.c_m),
        g_lk=param(cfg.g_lk),
        tau_ref=param(cfg.tau_ref),
        spike_freq_adapt=cfg.spike_freq_adapt,
        dg_k=param(cfg.dg_k),
        tau_k=param(cfg.tau_k),
        v_k=param(cfg.v_k),
        dtype=DTYPE,
    )

    synapse = SpikeNetCompositePSC(
        n_neuron=n_total,
        exc_weights_by_delay=exc_weights,
        inh_weights_by_delay=inh_weights,
        tau_ampa=cfg.tau_ampa_rec,
        tau_gaba=cfg.tau_gaba_rec,
        E_ampa=cfg.E_ampa,
        E_gaba=cfg.E_gaba,
        use_sparse=True,
        use_circular_buffer=False,
    )

    neuron.to(device=DEVICE, dtype=DTYPE)
    synapse.to(device=DEVICE, dtype=DTYPE)

    return neuron, synapse, n_e, coords_e


def _apply_initial_condition(
    neuron: SpikeNetNeuron,
    n_e: int,
    cfg: ChenGong2021Cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Mimic Chen2021 random initial condition (p_fire for E/I populations)."""
    n_total = neuron.size
    n_i = n_total - n_e

    if cfg.init_fire_prob_e <= 0.0 and cfg.init_fire_prob_i <= 0.0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(int(cfg.seed) + 12345)

    with torch.no_grad():
        e_mask = torch.rand((n_e,), generator=gen, device=DEVICE) < float(cfg.init_fire_prob_e)
        if n_i > 0:
            i_mask = torch.rand((n_i,), generator=gen, device=DEVICE) < float(cfg.init_fire_prob_i)
        else:
            i_mask = torch.zeros((0,), dtype=torch.bool, device=DEVICE)

        if e_mask.any():
            neuron.v[0, :n_e] = torch.where(
                e_mask,
                neuron.v_threshold[:n_e] + 1.0,
                neuron.v[0, :n_e],
            )
        if i_mask.any():
            neuron.v[0, n_e:] = torch.where(
                i_mask,
                neuron.v_threshold[n_e:] + 1.0,
                neuron.v[0, n_e:],
            )

    return (
        torch.nonzero(e_mask, as_tuple=False).squeeze(1).to(torch.int32).cpu().numpy(),
        torch.nonzero(i_mask, as_tuple=False).squeeze(1).to(torch.int32).cpu().numpy(),
    )


def _smooth_rate(step_counts: np.ndarray, n_cells: int, dt_ms: float, tau_ms: float) -> np.ndarray:
    alpha = math.exp(-dt_ms / max(tau_ms, 1e-6))
    inst_hz = step_counts.astype(np.float32) / max(n_cells, 1) / (dt_ms * 1e-3)
    out = np.empty_like(inst_hz, dtype=np.float32)
    prev = 0.0
    for i, val in enumerate(inst_hz):
        prev = alpha * prev + (1.0 - alpha) * float(val)
        out[i] = prev
    return out


def _winner_labels(
    center_hz: np.ndarray,
    corner_hz: np.ndarray,
    eps_hz: float,
) -> np.ndarray:
    diff = center_hz - corner_hz
    winner = np.full(center_hz.shape, -1, dtype=np.int8)
    winner[diff > eps_hz] = 0
    winner[diff < -eps_hz] = 1
    return winner


def _count_switches(winner: np.ndarray, start_step: int, min_dwell_steps: int) -> int:
    seq = winner[start_step:]
    if seq.size == 0:
        return 0

    compressed: list[int] = []
    i = 0
    while i < seq.size:
        label = int(seq[i])
        j = i + 1
        while j < seq.size and seq[j] == label:
            j += 1
        run_len = j - i
        if label >= 0 and run_len >= min_dwell_steps:
            compressed.append(label)
        i = j

    if len(compressed) <= 1:
        return 0

    switches = 0
    prev = compressed[0]
    for cur in compressed[1:]:
        if cur != prev:
            switches += 1
        prev = cur
    return switches


def _firing_rate_grid(
    spike_t: np.ndarray,
    spike_n: np.ndarray,
    n_e: int,
    step_start: int,
    step_end: int,
    dt_ms: float,
    hw: int,
) -> np.ndarray:
    duration_s = max((step_end - step_start) * dt_ms * 1e-3, 1e-6)
    counts = np.zeros(n_e, dtype=np.float32)
    mask = (spike_t >= step_start) & (spike_t < step_end)
    if np.any(mask):
        np.add.at(counts, spike_n[mask], 1)
    rates = counts / duration_s
    return rates.reshape(2 * hw + 1, 2 * hw + 1)


def _plot_winner_background(ax: plt.Axes, times_s: np.ndarray, winner: np.ndarray) -> None:
    for label, color in ((0, "#fff5d9"), (1, "#dff4ff")):
        idx = np.flatnonzero(winner == label)
        if idx.size == 0:
            continue
        seg_start = idx[0]
        seg_prev = idx[0]
        for cur in idx[1:]:
            if cur == seg_prev + 1:
                seg_prev = cur
                continue
            x0 = times_s[seg_start]
            x1 = times_s[min(seg_prev + 1, times_s.size - 1)]
            ax.axvspan(x0, x1, color=color, alpha=0.25, lw=0)
            seg_start = cur
            seg_prev = cur
        x0 = times_s[seg_start]
        x1 = times_s[min(seg_prev + 1, times_s.size - 1)]
        ax.axvspan(x0, x1, color=color, alpha=0.25, lw=0)


def run_simulation(
    neuron: SpikeNetNeuron,
    synapse: SpikeNetCompositePSC,
    n_e: int,
    coords_e: np.ndarray,
    cfg: ChenGong2021Cfg,
) -> dict[str, np.ndarray]:
    n_total = neuron.size
    n_i = n_total - n_e

    t_steps = int(cfg.T_ms / cfg.dt)
    stim_on_step = int(cfg.stim_start_ms / cfg.dt)

    center_radius = cfg.region_sigma_mult * cfg.qisig
    center_mask_np = _region_mask(coords_e, cfg.center_pos_yx, center_radius)
    corner_mask_np = _region_mask(coords_e, cfg.corner_pos_yx, center_radius)

    center_mask_t = torch.as_tensor(center_mask_np, dtype=torch.bool, device=DEVICE)
    corner_mask_t = torch.as_tensor(corner_mask_np, dtype=torch.bool, device=DEVICE)

    input_gain = _double_input_gain(coords_e, cfg)
    input_rate_on = cfg.rate_ext_E + input_gain

    # Poisson event count per step matches SpikeNet's implementation.
    lambda_e_base = torch.full(
        (1, n_e),
        cfg.N_ext * cfg.rate_ext_E * cfg.dt / 1000.0,
        dtype=DTYPE,
        device=DEVICE,
    )
    lambda_e_on = torch.as_tensor(
        cfg.N_ext * input_rate_on * cfg.dt / 1000.0,
        dtype=DTYPE,
        device=DEVICE,
    ).unsqueeze(0)

    lambda_i = torch.full(
        (1, n_i),
        cfg.N_ext * cfg.rate_ext_I * cfg.dt / 1000.0,
        dtype=DTYPE,
        device=DEVICE,
    )

    functional.init_net_state(neuron, batch_size=1, device=DEVICE, dtype=DTYPE)
    functional.init_net_state(synapse, batch_size=1, device=DEVICE, dtype=DTYPE)
    init_active_e, init_active_i = _apply_initial_condition(neuron, n_e=n_e, cfg=cfg)

    alpha_ext = math.exp(-cfg.dt / cfg.tau_ampa_ext)
    ext_gs = torch.zeros(1, n_total, device=DEVICE, dtype=DTYPE)

    # SpikeNet ChemSyn model-0 pre-synaptic transmitter release state.
    # s_pre[i]: gating variable; release per step = K_trans*(1-s_pre[i]) when active.
    # trans_left[i]: remaining release steps after a spike.
    steps_trans_ampa = max(1, int(round(cfg.Dt_trans_AMPA / cfg.dt)))
    steps_trans_gaba = max(1, int(round(cfg.Dt_trans_GABA / cfg.dt)))
    K_trans_ampa = float(1.0 / steps_trans_ampa)
    K_trans_gaba = float(1.0 / steps_trans_gaba)
    exp_decay_ampa = math.exp(-cfg.dt / cfg.tau_ampa_rec)
    exp_decay_gaba = math.exp(-cfg.dt / cfg.tau_gaba_rec)
    # Per-neuron decay vector: E-neurons use AMPA tau, I-neurons use GABA tau
    s_pre_decay = torch.ones(1, n_total, device=DEVICE, dtype=DTYPE)
    s_pre_decay[:, :n_e] = exp_decay_ampa
    s_pre_decay[:, n_e:] = exp_decay_gaba
    s_pre = torch.zeros(1, n_total, device=DEVICE, dtype=DTYPE)
    trans_left = torch.zeros(1, n_total, device=DEVICE, dtype=torch.int32)

    spike_t_parts: list[torch.Tensor] = []
    spike_n_parts: list[torch.Tensor] = []
    center_counts_t = torch.zeros((t_steps,), dtype=torch.int32, device=DEVICE)
    corner_counts_t = torch.zeros((t_steps,), dtype=torch.int32, device=DEVICE)

    with torch.no_grad(), environ.context(dt=cfg.dt):
        for t in range(t_steps):
            lambda_e = lambda_e_on if t >= stim_on_step else lambda_e_base
            if cfg.lambda_noise_cv > 0.0:
                noise_e = 1.0 + cfg.lambda_noise_cv * torch.randn_like(lambda_e)
                noise_i = 1.0 + cfg.lambda_noise_cv * torch.randn_like(lambda_i)
                lambda_e_step = torch.clamp(lambda_e * noise_e, min=0.0)
                lambda_i_step = torch.clamp(lambda_i * noise_i, min=0.0)
            else:
                lambda_e_step = lambda_e
                lambda_i_step = lambda_i

            poi_e = torch.poisson(lambda_e_step)
            poi_i = torch.poisson(lambda_i_step)

            ext_gs = alpha_ext * ext_gs
            ext_gs[:, :n_e] = ext_gs[:, :n_e] + poi_e * cfg.g_ext
            ext_gs[:, n_e:] = ext_gs[:, n_e:] + poi_i * cfg.g_ext

            v_now = neuron.v
            rec_current = synapse.current(v_now)
            ext_current = ext_gs * (cfg.E_ampa - v_now)
            z = neuron(rec_current + ext_current)

            # --- ChemSyn model-0 pre-synaptic gating ---
            # New spikes extend the release window.
            fired = (z > 0).to(torch.int32)
            # E neurons: steps_trans_ampa; I neurons: steps_trans_gaba
            steps_per_neuron = torch.full_like(trans_left, steps_trans_ampa)
            steps_per_neuron[:, n_e:] = steps_trans_gaba
            trans_left = trans_left + fired * steps_per_neuron
            active = trans_left > 0
            # Release factor: K_trans * (1 - s_pre) where active, else 0.
            # Use AMPA K_trans for E neurons, GABA K_trans for I neurons.
            K_trans_e = K_trans_ampa
            K_trans_g = K_trans_gaba
            k_trans = torch.full_like(s_pre, K_trans_e)
            k_trans[:, n_e:] = K_trans_g
            release = active.float() * k_trans * (1.0 - s_pre)
            # Update s_pre: increment where active, then decay all.
            s_pre = torch.where(active, s_pre + k_trans * (1.0 - s_pre), s_pre) * s_pre_decay
            # Decrement trans_left counter.
            trans_left = torch.clamp(trans_left - active.to(torch.int32), min=0)
            # Pass release tensor to synapse channels (AMPA uses E pre-weights,
            # GABA uses I pre-weights; zero weights mask the other population).
            ampa_psc = synapse.ampa(release)
            gaba_psc = synapse.gaba(release)
            synapse.psc = ampa_psc + gaba_psc

            fired_e = z[0, :n_e] > 0
            center_counts_t[t] = fired_e[center_mask_t].sum()
            corner_counts_t[t] = fired_e[corner_mask_t].sum()

            fired_idx = fired_e.nonzero(as_tuple=False).squeeze(1)
            if fired_idx.numel() > 0:
                spike_t_parts.append(
                    torch.full(
                        (fired_idx.numel(),),
                        t,
                        dtype=torch.int32,
                        device=DEVICE,
                    )
                )
                spike_n_parts.append(fired_idx.to(torch.int32))

    if spike_t_parts:
        spike_t = torch.cat(spike_t_parts).cpu().numpy()
        spike_n = torch.cat(spike_n_parts).cpu().numpy()
    else:
        spike_t = np.empty((0,), dtype=np.int32)
        spike_n = np.empty((0,), dtype=np.int32)

    center_counts = center_counts_t.cpu().numpy().astype(np.int16)
    corner_counts = corner_counts_t.cpu().numpy().astype(np.int16)

    center_hz = _smooth_rate(center_counts, int(center_mask_np.sum()), cfg.dt, cfg.ema_tau_ms)
    corner_hz = _smooth_rate(corner_counts, int(corner_mask_np.sum()), cfg.dt, cfg.ema_tau_ms)
    winner = _winner_labels(center_hz, corner_hz, cfg.winner_eps_hz)

    return {
        "spike_t": np.asarray(spike_t, dtype=np.int32),
        "spike_n": np.asarray(spike_n, dtype=np.int32),
        "center_counts": center_counts,
        "corner_counts": corner_counts,
        "center_hz": center_hz,
        "corner_hz": corner_hz,
        "winner": winner,
        "center_mask": np.flatnonzero(center_mask_np).astype(np.int32),
        "corner_mask": np.flatnonzero(corner_mask_np).astype(np.int32),
        "input_gain": input_gain,
        "input_rate_on": input_rate_on.astype(np.float32),
        "stim_on_step": np.asarray([stim_on_step], dtype=np.int32),
        "region_radius": np.asarray([center_radius], dtype=np.float32),
        "init_active_e": init_active_e,
        "init_active_i": init_active_i,
        "lambda_e_base": np.asarray([cfg.N_ext * cfg.rate_ext_E * cfg.dt / 1000.0], dtype=np.float32),
        "lambda_i_base": np.asarray([cfg.N_ext * cfg.rate_ext_I * cfg.dt / 1000.0], dtype=np.float32),
    }


def save_outputs(
    cfg: ChenGong2021Cfg,
    coords_e: np.ndarray,
    n_e: int,
    result: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    spike_t = result["spike_t"]
    spike_n = result["spike_n"]
    center_hz = result["center_hz"]
    corner_hz = result["corner_hz"]
    winner = result["winner"]

    t_steps = int(cfg.T_ms / cfg.dt)
    times_s = np.arange(t_steps, dtype=np.float32) * (cfg.dt * 1e-3)
    stim_on_step = int(result["stim_on_step"][0])

    early_start = stim_on_step
    early_end = min(t_steps, stim_on_step + int(1000.0 / cfg.dt))
    late_start = max(stim_on_step, t_steps - int(1000.0 / cfg.dt))
    late_end = t_steps

    input_grid = result["input_rate_on"].reshape(2 * cfg.hw + 1, 2 * cfg.hw + 1)
    early_grid = _firing_rate_grid(spike_t, spike_n, n_e, early_start, early_end, cfg.dt, cfg.hw)
    late_grid = _firing_rate_grid(spike_t, spike_n, n_e, late_start, late_end, cfg.dt, cfg.hw)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    im = ax.imshow(
        input_grid,
        origin="lower",
        cmap="Blues",
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        aspect="equal",
    )
    ax.set_title("Input rate map when stimulus is ON (kHz)")
    ax.set_xlabel("x (rad)")
    ax.set_ylabel("y (rad)")
    ax.plot(cfg.center_pos_yx[1], cfg.center_pos_yx[0], "r+", ms=10, mew=2)
    ax.plot(cfg.corner_pos_yx[1], cfg.corner_pos_yx[0], "g+", ms=10, mew=2)
    fig.colorbar(im, ax=ax, label="kHz")

    ax = axes[0, 1]
    im = ax.imshow(
        early_grid,
        origin="lower",
        cmap="hot",
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        aspect="equal",
    )
    ax.set_title(
        f"Early firing map [{early_start * cfg.dt:.0f}, {early_end * cfg.dt:.0f}] ms"
    )
    ax.set_xlabel("x (rad)")
    ax.set_ylabel("y (rad)")
    fig.colorbar(im, ax=ax, label="Hz")

    ax = axes[1, 0]
    im = ax.imshow(
        late_grid,
        origin="lower",
        cmap="hot",
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        aspect="equal",
    )
    ax.set_title(
        f"Late firing map [{late_start * cfg.dt:.0f}, {late_end * cfg.dt:.0f}] ms"
    )
    ax.set_xlabel("x (rad)")
    ax.set_ylabel("y (rad)")
    fig.colorbar(im, ax=ax, label="Hz")

    ax = axes[1, 1]
    _plot_winner_background(ax, times_s, winner)
    ax.plot(times_s, center_hz, color="#cc2f2f", lw=1.5, label="center-region rate")
    ax.plot(times_s, corner_hz, color="#1b77b9", lw=1.5, label="corner-region rate")
    ax.axvline(cfg.stim_start_ms * 1e-3, color="k", ls="--", lw=1.0, label="stim start")
    ax.set_title("Winner timeline (EMA-smoothed region rates)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.legend(loc="upper right")

    fig.suptitle("Chen and Gong 2021 unscaled: static double-peak WTA transition", fontsize=13)
    fig.tight_layout()
    overview_path = out_dir / "wta_levy_overview.png"
    fig.savefig(overview_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig_r, ax_r = plt.subplots(figsize=(11, 4))
    if spike_t.size > 0:
        ax_r.scatter(spike_t * cfg.dt * 1e-3, spike_n, s=0.2, c="k", alpha=0.25, rasterized=True)
    ax_r.axvline(cfg.stim_start_ms * 1e-3, color="tab:red", ls="--", lw=1)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_ylabel("Excitatory neuron index")
    ax_r.set_title("Spike raster (E population)")
    fig_r.tight_layout()
    raster_path = out_dir / "wta_levy_raster.png"
    fig_r.savefig(raster_path, dpi=170, bbox_inches="tight")
    plt.close(fig_r)

    metadata_path = out_dir / "activation_metadata.npz"
    np.savez(
        metadata_path,
        spike_t=spike_t,
        spike_n=spike_n,
        time_s=times_s,
        center_counts=result["center_counts"],
        corner_counts=result["corner_counts"],
        center_hz=center_hz,
        corner_hz=corner_hz,
        winner=winner,
        center_mask=result["center_mask"],
        corner_mask=result["corner_mask"],
        input_gain=result["input_gain"],
        input_rate_on=result["input_rate_on"],
        coords_e=coords_e.astype(np.float32),
        center_pos=np.asarray(cfg.center_pos_yx, dtype=np.float32),
        corner_pos=np.asarray(cfg.corner_pos_yx, dtype=np.float32),
        dt_ms=np.asarray([cfg.dt], dtype=np.float32),
        stim_on_step=result["stim_on_step"],
        region_radius=result["region_radius"],
        init_active_e=result["init_active_e"],
        init_active_i=result["init_active_i"],
        lambda_e_base=result["lambda_e_base"],
        lambda_i_base=result["lambda_i_base"],
    )

    total_spikes = int(spike_t.size)
    mean_rate_hz = total_spikes / (n_e * cfg.T_ms * 1e-3)

    center_after = winner[stim_on_step:] == 0
    corner_after = winner[stim_on_step:] == 1
    valid_after = (winner[stim_on_step:] >= 0)
    denom_valid = max(int(valid_after.sum()), 1)
    center_fraction = float(center_after.sum()) / denom_valid
    corner_fraction = float(corner_after.sum()) / denom_valid

    idx_center = np.flatnonzero(winner[stim_on_step:] == 0)
    idx_corner = np.flatnonzero(winner[stim_on_step:] == 1)
    first_center_ms = None if idx_center.size == 0 else (stim_on_step + int(idx_center[0])) * cfg.dt
    first_corner_ms = None if idx_corner.size == 0 else (stim_on_step + int(idx_corner[0])) * cfg.dt

    min_dwell_steps = max(1, int(cfg.min_dwell_ms / cfg.dt))
    switches = _count_switches(winner, stim_on_step, min_dwell_steps)

    levy_like = (
        first_center_ms is not None
        and first_corner_ms is not None
        and first_center_ms < first_corner_ms
        and switches >= 1
    )

    summary_path = out_dir / "infer_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("model: Chen_and_Gong_2021_unscaled\n")
        f.write(f"output_dir: {out_dir}\n")
        f.write(f"n_neuron: {n_e + cfg.N_i}\n")
        f.write(f"hw: {cfg.hw}, N_i: {cfg.N_i}\n")
        f.write(f"dt_ms: {cfg.dt:.4f}\n")
        f.write(f"T_ms: {cfg.T_ms:.1f}\n")
        f.write(f"stim_start_ms: {cfg.stim_start_ms:.1f}\n")
        f.write(f"sampler: poisson\n")
        f.write(f"lambda_noise_cv: {cfg.lambda_noise_cv:.4f}\n")
        f.write(f"init_fire_prob_e: {cfg.init_fire_prob_e:.3f}\n")
        f.write(f"init_fire_prob_i: {cfg.init_fire_prob_i:.3f}\n")
        f.write(f"rate_ext_E: {cfg.rate_ext_E:.6f}\n")
        f.write(f"rate_ext_I: {cfg.rate_ext_I:.6f}\n")
        f.write(f"g_EI: {cfg.g_EI:.6f}\n")
        f.write(f"g_II: {cfg.g_II:.6f}\n")
        f.write(f"g_mu: {cfg.g_mu:.6f}\n")
        f.write(f"tau_ampa_rec: {cfg.tau_ampa_rec:.6f}\n")
        f.write(f"tau_gaba_rec: {cfg.tau_gaba_rec:.6f}\n")
        f.write(f"tau_ampa_ext: {cfg.tau_ampa_ext:.6f}\n")
        f.write(f"Dt_trans_AMPA: {cfg.Dt_trans_AMPA:.4f}\n")
        f.write(f"Dt_trans_GABA: {cfg.Dt_trans_GABA:.4f}\n")
        f.write(f"steps_trans_ampa: {max(1, int(round(cfg.Dt_trans_AMPA / cfg.dt)))}\n")
        f.write(f"synapse_model: chemsyn_model0\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"total_spikes: {total_spikes}\n")
        f.write(f"mean_rate_hz: {mean_rate_hz:.6f}\n")
        f.write(
            "first_center_win_ms: {}\n".format("nan" if first_center_ms is None else f"{first_center_ms:.3f}")
        )
        f.write(
            "first_corner_win_ms: {}\n".format("nan" if first_corner_ms is None else f"{first_corner_ms:.3f}")
        )
        f.write(f"winner_switches: {switches}\n")
        f.write(f"center_win_fraction: {center_fraction:.6f}\n")
        f.write(f"corner_win_fraction: {corner_fraction:.6f}\n")
        f.write(f"levy_like_transition: {'yes' if levy_like else 'no'}\n")
        f.write(f"overview_figure: {overview_path.name}\n")
        f.write(f"raster_figure: {raster_path.name}\n")
        f.write(f"activation_metadata: {metadata_path.name}\n")

    print(f"Saved: {overview_path}")
    print(f"Saved: {raster_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chen and Gong 2021 unscaled static double-peak WTA test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T-ms", type=float, default=10000.0)
    parser.add_argument("--stim-start-ms", type=float, default=4000.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="static_gaussian/chen_gong_2021_double_peak_infer",
    )
    parser.add_argument(
        "--ext-scale",
        type=float,
        default=1.0,
        help="Scale factor for rate_ext_E/rate_ext_I.",
    )
    parser.add_argument(
        "--noise-cv",
        type=float,
        default=0.0,
        help="Per-step Gaussian jitter CV for Poisson lambda.",
    )
    parser.add_argument(
        "--inh-scale",
        type=float,
        default=1.0,
        help="Scale factor for inhibitory conductances g_EI/g_II.",
    )
    parser.add_argument(
        "--tau-scale",
        type=float,
        default=1.0,
        help="Scale factor for tau_ampa_rec/tau_gaba_rec/tau_ampa_ext.",
    )
    parser.add_argument(
        "--ee-scale",
        type=float,
        default=1.0,
        help="Scale factor for EE mean weight g_mu (controls bump attractor strength).",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ChenGong2021Cfg()
    cfg = replace(cfg, seed=args.seed, T_ms=args.T_ms, stim_start_ms=args.stim_start_ms)
    cfg.rate_ext_E *= args.ext_scale
    cfg.rate_ext_I *= args.ext_scale
    cfg.lambda_noise_cv = max(0.0, args.noise_cv)
    cfg.g_EI *= args.inh_scale
    cfg.g_II *= args.inh_scale
    cfg.tau_ampa_rec *= args.tau_scale
    cfg.tau_gaba_rec *= args.tau_scale
    cfg.tau_ampa_ext *= args.tau_scale
    cfg.g_mu *= args.ee_scale

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    global DEVICE
    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("=" * 72)
    print("Chen and Gong 2021 static double-peak test")
    print(f"  hw={cfg.hw}, N_e={(2 * cfg.hw + 1) ** 2}, N_i={cfg.N_i}")
    print(f"  dt={cfg.dt} ms, T={cfg.T_ms:.1f} ms, stim_start={cfg.stim_start_ms:.1f} ms")
    print(f"  centers(y,x) = {cfg.center_pos_yx}, {cfg.corner_pos_yx}")
    print(f"  device={DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  cuda_name={torch.cuda.get_device_name(0)}")
    print("=" * 72)

    t0 = time.perf_counter()
    neuron, synapse, n_e, coords_e = build_network(cfg)
    print(f"Network built in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    result = run_simulation(neuron, synapse, n_e, coords_e, cfg)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"Simulation done in {time.perf_counter() - t1:.2f}s")

    out_dir = Path(__file__).parent / args.output_dir
    save_outputs(cfg, coords_e, n_e, result, out_dir)


if __name__ == "__main__":
    main()
