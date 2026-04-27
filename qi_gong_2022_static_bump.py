"""Qi & Gong 2022 — Static Gaussian bump attractor experiment (btorch reproduction).

Reproduces the static Gaussian Poisson-input experiment from main_Qi_and_Gong_2022.m:
- Scaled network: hw=10 → N_e=441, N_i=300 (paper: hw=31 → N_e=3969, N_i=1000)
- dt=0.1ms, simulate 2 seconds (T=20000 steps, paper: 10s)
- Static Gaussian rate map: rate = rate_ext_E * (1 + contrast * exp(-0.5*r²/σ²))
- Sweep σ over linspace(0, π, 10), 3 representative values shown
- No training; pure forward simulation under torch.no_grad()
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
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


# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    # Network size (scaled down)
    hw: int = 10
    N_i: int = 300
    seed: int = 42
    # Simulation
    dt: float = 0.1          # ms
    T_ms: float = 2000.0     # total simulation time (ms)
    # Neuron (Qi 2022 values)
    tau_ref: float = 4.0
    v_threshold: float = -50.0
    v_reset: float = -60.0
    v_lk: float = -70.0
    c_m: float = 0.25
    g_lk: float = 0.0167
    spike_freq_adapt: bool = True
    dg_k: float = 0.01
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
    # Weights (µS)
    g_mu: float = 4e-3
    g_EI: float = 13.5e-3
    g_IE: float = 5e-3
    g_II: float = 25e-3
    # Current scaling (used only for backward-compat; not used in conductance mode)
    exc_scale: float = 65.0
    inh_scale: float = -15.0
    # Reversal potentials (mV) — SpikeNet ChemSyn.cpp: V_ex=0, V_in=-80
    E_ampa: float = 0.0
    E_gaba: float = -80.0
    # External input (paper values)
    N_ext: int = 1000
    g_ext: float = 2e-3
    rate_ext_E: float = 0.85   # kHz
    rate_ext_I: float = 1.0    # kHz
    tau_ampa_ext: float = 5.0  # ms (AMPA decay for external PSC)
    contrast: float = 0.5


CFG = Cfg()
DEVICE = torch.device("cpu")
DTYPE = torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder (adapted from qi_gong_2022_full_scale.py)
# ──────────────────────────────────────────────────────────────────────────────

def _torus_dist2_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return (diff ** 2).sum(axis=2)


def _sparse_delay_buckets(
    post: np.ndarray,
    pre: np.ndarray,
    vals: np.ndarray,
    delays: np.ndarray,
    n_neuron: int,
) -> dict[int, torch.Tensor]:
    buckets: dict[int, torch.Tensor] = {}
    for d in np.unique(delays):
        mask = delays == d
        rows = torch.as_tensor(post[mask], dtype=torch.long)
        cols = torch.as_tensor(pre[mask], dtype=torch.long)
        vs = torch.as_tensor(vals[mask], dtype=torch.float32)
        W = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vs, size=(n_neuron, n_neuron),
        ).to_sparse_csr()
        buckets[int(d)] = W
    return buckets


def build_network(cfg: Cfg):
    rng = np.random.default_rng(cfg.seed)
    N_e = (2 * cfg.hw + 1) ** 2
    n_neuron = N_e + cfg.N_i
    delay_max_steps = max(1, int(round(cfg.delay_max_ms / cfg.dt)))
    spacing = 2 * np.pi / (2 * cfg.hw + 1)

    x1d = np.linspace(-np.pi, np.pi, 2 * cfg.hw + 1, endpoint=False)
    yy, xx = np.meshgrid(x1d, x1d, indexing="ij")
    coords_e = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    coords_i = rng.uniform(-np.pi, np.pi, (cfg.N_i, 2)).astype(np.float32)

    lognormal_sigma = 0.3
    mu_p = math.log(cfg.g_mu) - 0.5 * lognormal_sigma ** 2

    exc_post_all, exc_pre_all, exc_val_all, exc_del_all = [], [], [], []
    inh_post_all, inh_pre_all, inh_val_all, inh_del_all = [], [], [], []

    # EE
    print(f"  EE (N_e={N_e})...", flush=True)
    sigma2_EE = max((cfg.tau_c_EE * spacing) ** 2, 1e-9)
    d2_ee = _torus_dist2_matrix(coords_e)
    p_ee = cfg.P0_init * np.exp(-0.5 * d2_ee / sigma2_EE)
    np.fill_diagonal(p_ee, 0.0)
    mask_ee = rng.random(d2_ee.shape) < p_ee
    post_ee, pre_ee = np.where(mask_ee)
    n_ee = len(post_ee)
    w_ee = rng.lognormal(mu_p, lognormal_sigma, n_ee).astype(np.float32)
    del_ee = rng.integers(1, delay_max_steps + 1, n_ee).astype(np.int32)
    exc_post_all.append(post_ee); exc_pre_all.append(pre_ee)
    exc_val_all.append(w_ee); exc_del_all.append(del_ee)
    print(f"    EE edges: {n_ee:,}", flush=True)
    del d2_ee, p_ee, mask_ee

    # I→E (GABA, inhibitory)
    print("  I→E...", flush=True)
    sigma2_I = max((cfg.tau_c_I * spacing) ** 2, 1e-9)
    diff_ie = coords_e[:, np.newaxis, :] - coords_i[np.newaxis, :, :]
    diff_ie = (diff_ie + np.pi) % (2 * np.pi) - np.pi
    d2_ie = (diff_ie ** 2).sum(axis=2)
    p_ie = cfg.P_ie * np.exp(-0.5 * d2_ie / sigma2_I)
    mask_ie = rng.random(d2_ie.shape) < p_ie
    post_ie_e, pre_ie_i = np.where(mask_ie)
    n_ie = len(post_ie_e)
    del_ie = rng.integers(1, delay_max_steps + 1, n_ie).astype(np.int32)
    inh_post_all.append(post_ie_e); inh_pre_all.append(N_e + pre_ie_i)
    inh_val_all.append(np.full(n_ie, cfg.g_EI, np.float32))
    inh_del_all.append(del_ie)
    print(f"    I→E edges: {n_ie:,}", flush=True)
    del d2_ie, p_ie, mask_ie

    # E→I (AMPA, excitatory)
    print("  E→I...", flush=True)
    sigma2_IE = max((cfg.tau_c_IE * spacing) ** 2, 1e-9)
    diff_ei = coords_i[:, np.newaxis, :] - coords_e[np.newaxis, :, :]
    diff_ei = (diff_ei + np.pi) % (2 * np.pi) - np.pi
    d2_ei = (diff_ei ** 2).sum(axis=2)
    p_ei = cfg.P_ei * np.exp(-0.5 * d2_ei / sigma2_IE)
    mask_ei = rng.random(d2_ei.shape) < p_ei
    post_ei_i, pre_ei_e = np.where(mask_ei)
    n_ei = len(post_ei_i)
    del_ei = rng.integers(1, delay_max_steps + 1, n_ei).astype(np.int32)
    exc_post_all.append(N_e + post_ei_i); exc_pre_all.append(pre_ei_e)
    exc_val_all.append(np.full(n_ei, cfg.g_IE, np.float32))
    exc_del_all.append(del_ei)
    print(f"    E→I edges: {n_ei:,}", flush=True)
    del d2_ei, p_ei, mask_ei

    # I→I (GABA)
    print("  I→I...", flush=True)
    d2_ii = _torus_dist2_matrix(coords_i)
    p_ii = cfg.P_ii * np.exp(-0.5 * d2_ii / sigma2_I)
    np.fill_diagonal(p_ii, 0.0)
    mask_ii = rng.random(d2_ii.shape) < p_ii
    post_ii_i, pre_ii_i = np.where(mask_ii)
    n_ii = len(post_ii_i)
    del_ii = rng.integers(1, delay_max_steps + 1, n_ii).astype(np.int32)
    inh_post_all.append(N_e + post_ii_i); inh_pre_all.append(N_e + pre_ii_i)
    inh_val_all.append(np.full(n_ii, cfg.g_II, np.float32))
    inh_del_all.append(del_ii)
    print(f"    I→I edges: {n_ii:,}", flush=True)
    del d2_ii, p_ii, mask_ii

    print("  Assembling sparse matrices...", flush=True)
    exc_weights = _sparse_delay_buckets(
        np.concatenate(exc_post_all), np.concatenate(exc_pre_all),
        np.concatenate(exc_val_all), np.concatenate(exc_del_all), n_neuron,
    )
    inh_weights = _sparse_delay_buckets(
        np.concatenate(inh_post_all), np.concatenate(inh_pre_all),
        np.concatenate(inh_val_all), np.concatenate(inh_del_all), n_neuron,
    )

    param = lambda v: torch.full((n_neuron,), v, dtype=DTYPE)
    neuron = SpikeNetNeuron(
        n_neuron=n_neuron,
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
        n_neuron=n_neuron,
        exc_weights_by_delay=exc_weights,
        inh_weights_by_delay=inh_weights,
        tau_ampa=5.0,
        tau_gaba=3.0,
        E_ampa=cfg.E_ampa,
        E_gaba=cfg.E_gaba,
        use_sparse=True,
        use_circular_buffer=False,
    )
    return neuron, synapse, N_e, coords_e


# ──────────────────────────────────────────────────────────────────────────────
# External input utilities
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_rate_map(coords_e: np.ndarray, sigma: float, cfg: Cfg) -> torch.Tensor:
    """Gaussian rate map for excitatory neurons on the torus.

    rate_i = rate_ext_E * (1 + contrast * exp(-0.5 * r_i² / sigma²))
    where r_i is toroidal distance from map centre (origin).
    If sigma == 0, constant input (flat map).
    """
    N_e = len(coords_e)
    if sigma < 1e-9:
        rate = np.full(N_e, cfg.rate_ext_E, dtype=np.float32)
    else:
        # Toroidal distance from origin
        d = coords_e.copy()
        d = (d + np.pi) % (2 * np.pi) - np.pi
        r2 = (d ** 2).sum(axis=1)
        rate = cfg.rate_ext_E * (1.0 + cfg.contrast * np.exp(-0.5 * r2 / sigma**2))
    return torch.as_tensor(rate, dtype=DTYPE, device=DEVICE)  # [N_e]


def double_gaussian_rate_map(
    coords_e: np.ndarray,
    sigma: float,
    cfg: Cfg,
    center1: np.ndarray | None = None,
    center2: np.ndarray | None = None,
) -> torch.Tensor:
    """Double-Gaussian rate map: two bumps on the torus.

    rate_i = rate_ext_E * (1 + contrast * (G(r1_i, sigma) + G(r2_i, sigma)))

    Defaults: center1=(0,0), center2=(pi,0) — maximally separated in x.
    Both distances are computed with toroidal wrapping.
    """
    if center1 is None:
        center1 = np.array([0.0, 0.0], dtype=np.float32)
    if center2 is None:
        center2 = np.array([np.pi, 0.0], dtype=np.float32)

    N_e = len(coords_e)
    if sigma < 1e-9:
        rate = np.full(N_e, cfg.rate_ext_E, dtype=np.float32)
    else:
        def _torus_r2(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
            d = coords - center
            d = (d + np.pi) % (2 * np.pi) - np.pi
            return (d ** 2).sum(axis=1)

        r1_2 = _torus_r2(coords_e, center1)
        r2_2 = _torus_r2(coords_e, center2)
        rate = cfg.rate_ext_E * (
            1.0 + cfg.contrast * (
                np.exp(-0.5 * r1_2 / sigma**2) + np.exp(-0.5 * r2_2 / sigma**2)
            )
        )
    return torch.as_tensor(rate, dtype=DTYPE, device=DEVICE)  # [N_e]


# ──────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    neuron: SpikeNetNeuron,
    synapse: SpikeNetCompositePSC,
    N_e: int,
    rate_e: torch.Tensor,    # [N_e] Hz (kHz * 1e3)... actually kHz
    cfg: Cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single simulation and return spike times and neuron IDs."""
    n_neuron = neuron.size
    N_i = n_neuron - N_e
    T = int(cfg.T_ms / cfg.dt)
    dt = cfg.dt

    functional.init_net_state(neuron, batch_size=1, device=DEVICE, dtype=DTYPE)
    functional.init_net_state(synapse, batch_size=1, device=DEVICE, dtype=DTYPE)

    # External conductance state: filtered Poisson input, shape [1, n_neuron] (µS)
    alpha_ext = math.exp(-dt / cfg.tau_ampa_ext)
    # In SpikeNet C++: Poisson(N_ext * rate_kHz * dt / 1000) events per step per neuron.
    # Each event contributes g_ext (µS) conductance; current = g_ext * (E_ampa - V).
    ext_weight_e = cfg.g_ext    # µS per aggregate spike event (AMPA external)
    ext_weight_i = cfg.g_ext    # µS per aggregate spike event (AMPA external to I)
    ext_gs = torch.zeros(1, n_neuron, device=DEVICE, dtype=DTYPE)  # µS

    # Poisson mean per step = N_ext * rate_kHz * dt_ms / 1000
    prob_e = cfg.N_ext * rate_e * dt / 1000.0     # [N_e] per-neuron Poisson probability
    prob_i = cfg.N_ext * cfg.rate_ext_I * dt / 1000.0  # scalar

    # Spike recording
    spike_t: list[int] = []
    spike_n: list[int] = []

    with torch.no_grad(), environ.context(dt=dt):
        for t in range(T):
            # Generate external Poisson spikes
            poi_e = torch.bernoulli(prob_e.expand(1, -1))   # [1, N_e]
            poi_i = torch.bernoulli(torch.full((1, N_i), prob_i, device=DEVICE))

            # Exponential filter on external conductance (µS)
            ext_gs = alpha_ext * ext_gs
            ext_gs[:, :N_e] = ext_gs[:, :N_e] + poi_e * ext_weight_e
            ext_gs[:, N_e:] = ext_gs[:, N_e:] + poi_i * ext_weight_i

            # Voltage-dependent currents (nA): I = gs * (E_rev - V)
            v_now = neuron.v                                 # [1, n_neuron] mV
            rec_current = synapse.current(v_now)             # [1, n_neuron] nA
            ext_current = ext_gs * (cfg.E_ampa - v_now)     # [1, n_neuron] nA

            # Neuron step
            total_input = rec_current + ext_current          # [1, n_neuron] nA
            z = neuron(total_input)                          # [1, n_neuron]

            # Update synapse with current spikes
            synapse(z)

            # Record spikes (excitatory only)
            fired_e = z[0, :N_e].nonzero(as_tuple=False).squeeze(1)
            if fired_e.numel() > 0:
                spike_t.extend([t] * fired_e.numel())
                spike_n.extend(fired_e.tolist())

    return np.array(spike_t, dtype=np.int32), np.array(spike_n, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Main: sweep σ and plot
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = CFG
    N_e = (2 * cfg.hw + 1) ** 2
    T = int(cfg.T_ms / cfg.dt)

    print("=" * 60)
    print(f"Qi & Gong 2022 Static Bump Experiment")
    print(f"  hw={cfg.hw}, N_e={N_e}, N_i={cfg.N_i}, n_total={N_e + cfg.N_i}")
    print(f"  dt={cfg.dt} ms, T={T} steps ({cfg.T_ms} ms)")
    print("=" * 60)

    print("\nBuilding network...", flush=True)
    t0 = time.perf_counter()
    neuron, synapse, N_e, coords_e = build_network(cfg)
    print(f"  Done in {time.perf_counter() - t0:.1f}s", flush=True)

    # σ sweep: pick 3 representative values from linspace(0, π, 10)
    sigma_all = np.linspace(0, np.pi, 10)
    sigma_indices = [2, 5, 8]   # narrow, medium, wide
    sigmas = sigma_all[sigma_indices]

    fig, axes = plt.subplots(2, len(sigmas), figsize=(4 * len(sigmas), 7))
    fig.suptitle("Qi & Gong 2022 — Static Gaussian Bump Attractor\n"
                 f"hw={cfg.hw}, N_e={N_e}, N_i={cfg.N_i}, T={cfg.T_ms:.0f}ms",
                 fontsize=12)

    for col, sigma in enumerate(sigmas):
        print(f"\n--- σ = {sigma:.3f} rad ---", flush=True)
        rate_e = gaussian_rate_map(coords_e, sigma, cfg)
        print(f"  rate range: [{rate_e.min():.3f}, {rate_e.max():.3f}] kHz", flush=True)

        t_sim = time.perf_counter()
        spike_t, spike_n = run_simulation(neuron, synapse, N_e, rate_e, cfg)
        elapsed = time.perf_counter() - t_sim
        if len(spike_t) > 0:
            rate_hz = len(spike_t) / (N_e * cfg.T_ms * 1e-3)
            print(f"  {len(spike_t):,} spikes, mean rate {rate_hz:.1f} Hz, sim {elapsed:.1f}s")
        else:
            print(f"  0 spikes (network silent), sim {elapsed:.1f}s")

        # ── Plot 1: Spatial firing rate map (last 500ms) ──
        ax_map = axes[0, col]
        win_start = max(0, T - int(500 / cfg.dt))
        mask_win = spike_t >= win_start
        st_win, sn_win = spike_t[mask_win], spike_n[mask_win]
        fire_rate = np.zeros(N_e)
        np.add.at(fire_rate, sn_win, 1)
        fire_rate /= (500e-3)  # spikes per second in last 500ms

        grid = fire_rate.reshape(2 * cfg.hw + 1, 2 * cfg.hw + 1)
        im = ax_map.imshow(grid, origin="lower", cmap="hot",
                           extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
        ax_map.set_title(f"σ = {sigma:.2f} rad")
        ax_map.set_xlabel("x (rad)")
        ax_map.set_ylabel("y (rad)")
        plt.colorbar(im, ax=ax_map, label="Hz (last 500ms)")

        # ── Plot 2: Input rate map ──
        ax_in = axes[1, col]
        rate_np = rate_e.numpy().reshape(2 * cfg.hw + 1, 2 * cfg.hw + 1)
        im2 = ax_in.imshow(rate_np, origin="lower", cmap="Blues",
                            extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
        ax_in.set_title(f"Input rate (kHz)")
        ax_in.set_xlabel("x (rad)")
        plt.colorbar(im2, ax=ax_in, label="kHz")

    plt.tight_layout()
    out_path = Path(__file__).parent / "qi_gong_2022_static_bump.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Extra: spike raster for last σ
    fig2, ax = plt.subplots(figsize=(10, 4))
    if len(spike_t) > 0:
        raster_t = spike_t * cfg.dt  # convert to ms
        ax.scatter(raster_t, spike_n, s=0.3, c="k", alpha=0.3, rasterized=True)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index (E pop)")
    ax.set_title(f"Spike raster — σ={sigmas[-1]:.2f} rad")
    raster_path = Path(__file__).parent / "qi_gong_2022_static_raster.png"
    plt.savefig(raster_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {raster_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Double-Gaussian experiment
    # Two bumps: center1=(0,0), center2=(π,0) — maximally separated in x.
    # Sweep the same 3 representative σ values.
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Double-Gaussian bump experiment")
    center1 = np.array([0.0,    0.0], dtype=np.float32)
    center2 = np.array([np.pi,  0.0], dtype=np.float32)
    print(f"  center1 = (0, 0),  center2 = (π, 0)")
    print("=" * 60)

    fig3, axes3 = plt.subplots(3, len(sigmas), figsize=(4 * len(sigmas), 10))
    fig3.suptitle(
        "Double-Gaussian Bump — center1=(0,0), center2=(π,0)\n"
        f"hw={cfg.hw}, N_e={N_e}, N_i={cfg.N_i}, T={cfg.T_ms:.0f}ms",
        fontsize=12,
    )

    for col, sigma in enumerate(sigmas):
        print(f"\n--- σ = {sigma:.3f} rad ---", flush=True)
        rate_e = double_gaussian_rate_map(coords_e, sigma, cfg, center1, center2)
        print(f"  rate range: [{rate_e.min():.3f}, {rate_e.max():.3f}] kHz", flush=True)

        t_sim = time.perf_counter()
        spike_t, spike_n = run_simulation(neuron, synapse, N_e, rate_e, cfg)
        elapsed = time.perf_counter() - t_sim
        if len(spike_t) > 0:
            rate_hz = len(spike_t) / (N_e * cfg.T_ms * 1e-3)
            print(f"  {len(spike_t):,} spikes, mean rate {rate_hz:.1f} Hz, sim {elapsed:.1f}s")
        else:
            print(f"  0 spikes (network silent), sim {elapsed:.1f}s")

        win_start = max(0, T - int(500 / cfg.dt))
        mask_win = spike_t >= win_start
        sn_win = spike_n[mask_win]
        fire_rate = np.zeros(N_e)
        np.add.at(fire_rate, sn_win, 1)
        fire_rate /= 500e-3
        grid = fire_rate.reshape(2 * cfg.hw + 1, 2 * cfg.hw + 1)

        # Row 0: firing rate map
        ax0 = axes3[0, col]
        im0 = ax0.imshow(grid, origin="lower", cmap="hot",
                         extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
        ax0.set_title(f"σ = {sigma:.2f} rad")
        ax0.set_xlabel("x (rad)"); ax0.set_ylabel("y (rad)")
        plt.colorbar(im0, ax=ax0, label="Hz (last 500ms)")

        # Mark bump centers
        for ax_mark in (ax0,):
            ax_mark.plot(center1[1], center1[0], "b+", ms=10, mew=2, label="c1")
            ax_mark.plot(
                np.clip(center2[1], -np.pi, np.pi),
                np.clip(center2[0], -np.pi, np.pi),
                "g+", ms=10, mew=2, label="c2",
            )

        # Row 1: input rate map
        ax1 = axes3[1, col]
        rate_np = rate_e.numpy().reshape(2 * cfg.hw + 1, 2 * cfg.hw + 1)
        im1 = ax1.imshow(rate_np, origin="lower", cmap="Blues",
                          extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
        ax1.set_title("Input rate (kHz)")
        ax1.set_xlabel("x (rad)")
        plt.colorbar(im1, ax=ax1, label="kHz")

        # Row 2: 1-D profile along x-axis (y=0 slice)
        ax2 = axes3[2, col]
        hw = cfg.hw
        mid_row = hw   # index of y=0 row (centre)
        x_vals = np.linspace(-np.pi, np.pi, 2 * hw + 1, endpoint=False)
        ax2.plot(x_vals, grid[mid_row, :], "r-", lw=1.5, label="firing rate (Hz)")
        ax2_r = ax2.twinx()
        ax2_r.plot(x_vals, rate_np[mid_row, :], "b--", lw=1, label="input (kHz)")
        ax2.set_xlabel("x (rad)")
        ax2.set_ylabel("Firing rate (Hz)", color="r")
        ax2_r.set_ylabel("Input rate (kHz)", color="b")
        ax2.axvline(center1[1], color="b", ls=":", lw=1)
        ax2.axvline(np.pi,      color="g", ls=":", lw=1)
        ax2.set_title("x-profile (y=0)")

    plt.tight_layout()
    dg_path = Path(__file__).parent / "qi_gong_2022_double_gaussian.png"
    plt.savefig(dg_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {dg_path}")


if __name__ == "__main__":
    main()
