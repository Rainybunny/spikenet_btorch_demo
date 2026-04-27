"""Qi & Gong 2022 cortical model — full-scale experiment.

Network: N_e = (2*31+1)^2 = 3969 excitatory + N_i = 1000 inhibitory = 4969 neurons.
Synapse: btorch.models.synapse.SpikeNetCompositePSC (AMPA + GABA channels,
         sparse CSR weight matrices, per-delay-step connectivity).
Neuron:  btorch.models.neurons.spikenet.SpikeNetNeuron with spike-freq adaptation.
Task:    Two-Gaussian spike motion detection (train + eval).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for subdir in ("btorch",):
    p = ROOT / subdir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btorch.models import environ, functional, rnn
from btorch.models.neurons.spikenet import SpikeNetNeuron
from btorch.models.synapse import SpikeNetCompositePSC


# ──────────────────────────────────────────────────────────────────────────────
# Qi & Gong 2022 network parameters (paper-original scale)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QiGong2022Config:
    # Topology (hw=31 → N_e = (2*31+1)^2 = 3969, N_i = 1000)
    hw: int = 31
    N_i: int = 1000
    seed: int = 42
    # Simulation
    dt: float = 0.1          # ms
    # Neuron parameters (SpikeNet defaults + Qi2022: tau_ref=4ms, SFA on)
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
    # Connectivity (matching Qi2022 MATLAB parameters)
    delay_max_ms: float = 4.0   # random uniform delays in (0, delay_max_ms]
    P0_init: float = 0.08       # EE base connection probability
    tau_c_EE: float = 8.0       # EE spatial scale (grid steps)
    tau_c_IE: float = 10.0      # E→I spatial scale
    tau_c_I: float = 20.0       # I→E, I→I spatial scale
    P_ei: float = 0.20          # E→I connection probability
    P_ie: float = 0.20          # I→E connection probability
    P_ii: float = 0.40          # I→I connection probability
    # Synaptic weights (µS)
    g_mu: float = 4e-3          # mean EE log-normal weight
    g_EI: float = 13.5e-3       # I→E (GABA)
    g_IE: float = 5e-3          # E→I (AMPA)
    g_II: float = 25e-3         # I→I (GABA)
    # Current scaling (align µS conductances with btorch LIF input current)
    exc_scale: float = 65.0
    inh_scale: float = -15.0


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder (vectorised — no per-neuron Python loops)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeNetGraph:
    dt: float
    n_neuron: int
    v_threshold: torch.Tensor
    v_reset: torch.Tensor
    v_lk: torch.Tensor
    c_m: torch.Tensor
    g_lk: torch.Tensor
    tau_ref: torch.Tensor
    spike_freq_adapt: bool
    dg_k: torch.Tensor
    tau_k: torch.Tensor
    v_k: torch.Tensor
    exc_weights_by_delay: dict[int, torch.Tensor]
    inh_weights_by_delay: dict[int, torch.Tensor]
    n_exc_edges: int = 0
    n_inh_edges: int = 0


def _torus_dist2_matrix(coords: np.ndarray) -> np.ndarray:
    """Vectorised pairwise squared toroidal distance on (-pi, pi)^2.

    Returns d2[j, i] = squared distance from neuron i to neuron j.
    Shape: [N, N].
    """
    # d[j,i] = coords[j] - coords[i]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]   # [N, N, 2]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return (diff ** 2).sum(axis=2)                                 # [N, N]


def _sparse_delay_buckets(
    post: np.ndarray,   # post-synaptic indices in [0, n_neuron)
    pre:  np.ndarray,   # pre-synaptic  indices in [0, n_neuron)
    vals: np.ndarray,   # connection weights (already scaled)
    delays: np.ndarray, # integer delay steps
    n_neuron: int,
) -> dict[int, torch.Tensor]:
    """Build per-delay sparse COO→CSR weight matrices of size [n_neuron, n_neuron]."""
    buckets: dict[int, torch.Tensor] = {}
    for d in np.unique(delays):
        mask = delays == d
        rows = torch.as_tensor(post[mask], dtype=torch.long)
        cols = torch.as_tensor(pre[mask],  dtype=torch.long)
        vs   = torch.as_tensor(vals[mask], dtype=torch.float32)
        W = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vs,
            size=(n_neuron, n_neuron),
        ).to_sparse_csr()
        buckets[int(d)] = W
    return buckets


def build_qi_gong_2022_graph(cfg: QiGong2022Config) -> SpikeNetGraph:
    rng = np.random.default_rng(cfg.seed)

    N_e = (2 * cfg.hw + 1) ** 2
    n_neuron = N_e + cfg.N_i
    delay_max_steps = max(1, int(round(cfg.delay_max_ms / cfg.dt)))
    spacing = 2 * np.pi / (2 * cfg.hw + 1)

    # ── 2-D toroidal lattice coordinates ──────────────────────────────
    x1d = np.linspace(-np.pi, np.pi, 2 * cfg.hw + 1, endpoint=False)
    yy, xx = np.meshgrid(x1d, x1d, indexing="ij")
    coords_e = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

    coords_i = rng.uniform(-np.pi, np.pi, (cfg.N_i, 2)).astype(np.float32)

    lognormal_sigma = 0.3
    mu_p = math.log(cfg.g_mu) - 0.5 * lognormal_sigma ** 2

    exc_post_all: list[np.ndarray] = []
    exc_pre_all:  list[np.ndarray] = []
    exc_val_all:  list[np.ndarray] = []
    exc_del_all:  list[np.ndarray] = []

    inh_post_all: list[np.ndarray] = []
    inh_pre_all:  list[np.ndarray] = []
    inh_val_all:  list[np.ndarray] = []
    inh_del_all:  list[np.ndarray] = []

    # ── EE connections (full vectorised pairwise) ──────────────────────
    print(f"  Building EE connectivity (N_e={N_e})...", flush=True)
    sigma2_EE = max((cfg.tau_c_EE * spacing) ** 2, 1e-9)
    d2_ee = _torus_dist2_matrix(coords_e)          # [N_e, N_e]
    p_ee  = cfg.P0_init * np.exp(-0.5 * d2_ee / sigma2_EE)
    np.fill_diagonal(p_ee, 0.0)
    mask_ee  = rng.random(d2_ee.shape) < p_ee      # [N_e(post), N_e(pre)]
    post_ee, pre_ee = np.where(mask_ee)
    n_ee = len(post_ee)
    w_ee  = rng.lognormal(mu_p, lognormal_sigma, n_ee).astype(np.float32)
    del_ee = rng.integers(1, delay_max_steps + 1, n_ee).astype(np.int32)
    exc_post_all.append(post_ee)
    exc_pre_all.append(pre_ee)
    exc_val_all.append(cfg.exc_scale * w_ee)
    exc_del_all.append(del_ee)
    print(f"    EE edges: {n_ee:,}", flush=True)
    del d2_ee, p_ee, mask_ee

    # ── I→E connections (post=excitatory, pre=inhibitory, GABA) ────────
    print("  Building I→E connectivity...", flush=True)
    sigma2_I = max((cfg.tau_c_I * spacing) ** 2, 1e-9)
    # d2_ie[j_e, j_i] = dist( excitatory_j, inhibitory_ji )
    diff_ie = (
        coords_e[:, np.newaxis, :] - coords_i[np.newaxis, :, :]
    )   # [N_e, N_i, 2]
    diff_ie = (diff_ie + np.pi) % (2 * np.pi) - np.pi
    d2_ie = (diff_ie ** 2).sum(axis=2)             # [N_e, N_i]
    p_ie  = cfg.P_ie * np.exp(-0.5 * d2_ie / sigma2_I)
    mask_ie = rng.random(d2_ie.shape) < p_ie       # [N_e(post), N_i(pre)]
    post_ie_e, pre_ie_i = np.where(mask_ie)
    post_ie = post_ie_e                             # global idx (excitatory)
    pre_ie  = N_e + pre_ie_i                        # global idx (inhibitory)
    n_ie = len(post_ie)
    del_ie = rng.integers(1, delay_max_steps + 1, n_ie).astype(np.int32)
    inh_post_all.append(post_ie)
    inh_pre_all.append(pre_ie)
    inh_val_all.append(np.full(n_ie, cfg.inh_scale * cfg.g_EI, dtype=np.float32))
    inh_del_all.append(del_ie)
    print(f"    I→E edges: {n_ie:,}", flush=True)
    del d2_ie, p_ie, mask_ie

    # ── E→I connections (post=inhibitory, pre=excitatory, AMPA) ────────
    print("  Building E→I connectivity...", flush=True)
    sigma2_IE = max((cfg.tau_c_IE * spacing) ** 2, 1e-9)
    # d2_ei[j_i, j_e] = dist( inhibitory_ji, excitatory_j )
    diff_ei = (
        coords_i[:, np.newaxis, :] - coords_e[np.newaxis, :, :]
    )   # [N_i, N_e, 2]
    diff_ei = (diff_ei + np.pi) % (2 * np.pi) - np.pi
    d2_ei = (diff_ei ** 2).sum(axis=2)             # [N_i, N_e]
    p_ei  = cfg.P_ei * np.exp(-0.5 * d2_ei / sigma2_IE)
    mask_ei = rng.random(d2_ei.shape) < p_ei       # [N_i(post), N_e(pre)]
    post_ei_i, pre_ei_e = np.where(mask_ei)
    post_ei = N_e + post_ei_i                       # global idx (inhibitory)
    pre_ei  = pre_ei_e                              # global idx (excitatory)
    n_ei = len(post_ei)
    del_ei = rng.integers(1, delay_max_steps + 1, n_ei).astype(np.int32)
    exc_post_all.append(post_ei)
    exc_pre_all.append(pre_ei)
    exc_val_all.append(np.full(n_ei, cfg.exc_scale * cfg.g_IE, dtype=np.float32))
    exc_del_all.append(del_ei)
    print(f"    E→I edges: {n_ei:,}", flush=True)
    del d2_ei, p_ei, mask_ei

    # ── I→I connections (both global indices in [N_e, n_neuron)) ───────
    print("  Building I→I connectivity...", flush=True)
    d2_ii = _torus_dist2_matrix(coords_i)           # [N_i, N_i]
    p_ii  = cfg.P_ii * np.exp(-0.5 * d2_ii / sigma2_I)
    np.fill_diagonal(p_ii, 0.0)
    mask_ii  = rng.random(d2_ii.shape) < p_ii
    post_ii_i, pre_ii_i = np.where(mask_ii)
    post_ii = N_e + post_ii_i
    pre_ii  = N_e + pre_ii_i
    n_ii = len(post_ii)
    del_ii = rng.integers(1, delay_max_steps + 1, n_ii).astype(np.int32)
    inh_post_all.append(post_ii)
    inh_pre_all.append(pre_ii)
    inh_val_all.append(np.full(n_ii, cfg.inh_scale * cfg.g_II, dtype=np.float32))
    inh_del_all.append(del_ii)
    print(f"    I→I edges: {n_ii:,}", flush=True)
    del d2_ii, p_ii, mask_ii

    # ── Assemble sparse delay-bucket matrices ──────────────────────────
    print("  Assembling sparse matrices...", flush=True)
    exc_weights = _sparse_delay_buckets(
        np.concatenate(exc_post_all),
        np.concatenate(exc_pre_all),
        np.concatenate(exc_val_all),
        np.concatenate(exc_del_all),
        n_neuron,
    )
    inh_weights = _sparse_delay_buckets(
        np.concatenate(inh_post_all),
        np.concatenate(inh_pre_all),
        np.concatenate(inh_val_all),
        np.concatenate(inh_del_all),
        n_neuron,
    )

    param_1d = lambda v: torch.full((n_neuron,), v, dtype=torch.float32)
    return SpikeNetGraph(
        dt=cfg.dt,
        n_neuron=n_neuron,
        v_threshold=param_1d(cfg.v_threshold),
        v_reset=param_1d(cfg.v_reset),
        v_lk=param_1d(cfg.v_lk),
        c_m=param_1d(cfg.c_m),
        g_lk=param_1d(cfg.g_lk),
        tau_ref=param_1d(cfg.tau_ref),
        spike_freq_adapt=cfg.spike_freq_adapt,
        dg_k=param_1d(cfg.dg_k),
        tau_k=param_1d(cfg.tau_k),
        v_k=param_1d(cfg.v_k),
        exc_weights_by_delay=exc_weights,
        inh_weights_by_delay=inh_weights,
        n_exc_edges=int(n_ee + n_ei),
        n_inh_edges=int(n_ie + n_ii),
    )


# ──────────────────────────────────────────────────────────────────────────────
# btorch model
# ──────────────────────────────────────────────────────────────────────────────

class QiGong2022Backbone(nn.Module):
    """Full-scale Qi & Gong 2022 cortical network as btorch backbone."""

    def __init__(self, graph: SpikeNetGraph, input_dim: int, dtype: torch.dtype):
        super().__init__()
        self.graph = graph
        self.dt = graph.dt

        self.input_proj = nn.Linear(input_dim, graph.n_neuron, bias=False, dtype=dtype)
        # Small init so external current doesn't saturate the network immediately
        nn.init.normal_(self.input_proj.weight, std=0.01)

        neuron = SpikeNetNeuron(
            n_neuron=graph.n_neuron,
            neuron_model="lif",
            v_threshold=graph.v_threshold.to(dtype=dtype),
            v_reset=graph.v_reset.to(dtype=dtype),
            v_lk=graph.v_lk.to(dtype=dtype),
            c_m=graph.c_m.to(dtype=dtype),
            g_lk=graph.g_lk.to(dtype=dtype),
            tau_ref=graph.tau_ref.to(dtype=dtype),
            spike_freq_adapt=graph.spike_freq_adapt,
            dg_k=graph.dg_k.to(dtype=dtype),
            tau_k=graph.tau_k.to(dtype=dtype),
            v_k=graph.v_k.to(dtype=dtype),
            dtype=dtype,
        )

        synapse = SpikeNetCompositePSC(
            n_neuron=graph.n_neuron,
            exc_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.exc_weights_by_delay.items()},
            inh_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.inh_weights_by_delay.items()},
            tau_ampa=5.0,
            tau_gaba=3.0,
            use_sparse=True,
            use_circular_buffer=False,  # autograd-compatible
        )

        self.brain = rnn.RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            update_state_names=("neuron.v",),
        )
        functional.init_net_state(
            self.brain, batch_size=1, device=torch.device("cpu"), dtype=dtype
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        functional.reset_net(self.brain, batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, events: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = events.flatten(start_dim=2)          # [T, B, H*W]
        x = self.input_proj(x)                   # [T, B, N]
        with environ.context(dt=self.dt):
            spike, states = self.brain(x)        # [T, B, N]
        return spike, states


class QiGong2022Detector(nn.Module):
    def __init__(self, backbone: QiGong2022Backbone, n_object: int = 2, hidden: int = 256):
        super().__init__()
        self.backbone = backbone
        self.n_object = n_object
        N = backbone.graph.n_neuron
        self.head = nn.Sequential(
            nn.Linear(N, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_object * 4),
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.backbone.reset_state(batch_size, device, dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        spike, _ = self.backbone(events)         # [T, B, N]
        T, B, N = spike.shape
        pred = self.head(spike.reshape(T * B, N)).reshape(T, B, self.n_object, 4)
        return torch.sigmoid(pred)


# ──────────────────────────────────────────────────────────────────────────────
# Two-Gaussian motion dataset
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionCfg:
    timesteps: int = 40
    height: int = 48
    width: int = 64
    sigma: float = 3.0
    base_rate: float = 0.01
    peak_rate: float = 0.35
    speed_min: float = 0.30
    speed_max: float = 0.90


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _make_sample(cfg: MotionCfg, rng: np.random.Generator):
    T, H, W = cfg.timesteps, cfg.height, cfg.width
    margin = 4.0 * cfg.sigma
    centers = rng.uniform(margin, [[W - margin], [H - margin]], (2, 2)).astype(np.float32)
    speeds  = rng.uniform(cfg.speed_min, cfg.speed_max, 2)
    angles  = rng.uniform(0.0, 2 * math.pi, 2)
    vel     = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

    yg, xg = np.mgrid[0:H, 0:W]
    spikes = np.zeros((T, H, W), np.float32)
    boxes  = np.zeros((T, 2, 4), np.float32)

    for t in range(T):
        rate = np.full((H, W), cfg.base_rate, np.float32)
        for o in range(2):
            cx, cy = float(centers[o, 0]), float(centers[o, 1])
            d2 = (xg - cx) ** 2 + (yg - cy) ** 2
            rate += cfg.peak_rate * np.exp(-0.5 * d2 / max(cfg.sigma ** 2, 1e-6)).astype(np.float32)
            bw = max(4.0, 4.0 * cfg.sigma)
            bh = bw
            boxes[t, o] = [
                np.clip(cx - bw / 2, 0, W - bw) / W,
                np.clip(cy - bh / 2, 0, H - bh) / H,
                bw / W, bh / H,
            ]
            for dim in range(2):
                nv = centers[o, dim] + vel[o, dim]
                lo = margin if dim == 0 else margin
                hi = (W if dim == 0 else H) - margin
                if nv < lo:
                    nv = lo + (lo - nv); vel[o, dim] *= -1
                elif nv > hi:
                    nv = hi - (nv - hi); vel[o, dim] *= -1
                centers[o, dim] = nv
        spikes[t] = (rng.random((H, W)) < np.clip(rate, 0, 0.95)).astype(np.float32)
    return spikes, boxes


class MotionDataset(Dataset):
    def __init__(self, n: int, cfg: MotionCfg, seed: int):
        rng = np.random.default_rng(seed)
        self.samples = [
            (torch.as_tensor(s), torch.as_tensor(b))
            for s, b in (_make_sample(cfg, rng) for _ in range(n))
        ]

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ──────────────────────────────────────────────────────────────────────────────
# Loss and metrics
# ──────────────────────────────────────────────────────────────────────────────

def perm_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    tgt_s = tgt[:, :, [1, 0], :]
    l1 = F.smooth_l1_loss(pred, tgt,   reduction="none").mean((-2, -1))
    l2 = F.smooth_l1_loss(pred, tgt_s, reduction="none").mean((-2, -1))
    return torch.minimum(l1, l2).mean()


def iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax2 = a[..., 0] + a[..., 2]; ay2 = a[..., 1] + a[..., 3]
    bx2 = b[..., 0] + b[..., 2]; by2 = b[..., 1] + b[..., 3]
    ix  = (torch.minimum(ax2, bx2) - torch.maximum(a[..., 0], b[..., 0])).clamp(0)
    iy  = (torch.minimum(ay2, by2) - torch.maximum(a[..., 1], b[..., 1])).clamp(0)
    inter = ix * iy
    ua = a[..., 2].clamp(0) * a[..., 3].clamp(0)
    ub = b[..., 2].clamp(0) * b[..., 3].clamp(0)
    return torch.where(ua + ub - inter > 0, inter / (ua + ub - inter), torch.zeros_like(inter))


def metrics(pred: torch.Tensor, tgt: torch.Tensor):
    tgt_s   = tgt[:, :, [1, 0], :]
    iou_d   = iou_xywh(pred, tgt).mean((-2, -1))
    iou_s   = iou_xywh(pred, tgt_s).mean((-2, -1))
    best    = torch.maximum(iou_d, iou_s)
    return float(best.mean()), float((best > 0.5).float().mean())


def prep(batch, device, dtype):
    s, b = batch
    return (
        s.permute(1, 0, 2, 3).to(device=device, dtype=dtype),
        b.permute(1, 0, 2, 3).to(device=device, dtype=dtype),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, scheduler, device, dtype):
    model.train()
    tot = dict(loss=0.0, iou=0.0, hit=0.0, n=0)
    for batch in loader:
        sp, bx = prep(batch, device, dtype)
        model.reset_state(int(sp.shape[1]), device, dtype)
        pred = model(sp)
        loss = perm_loss(pred, bx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        iou, hit = metrics(pred.detach(), bx)
        tot["loss"] += float(loss); tot["iou"] += iou; tot["hit"] += hit; tot["n"] += 1
    if scheduler is not None:
        scheduler.step()
    d = max(tot["n"], 1)
    return {k: tot[k] / d for k in ("loss", "iou", "hit")}


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    tot = dict(loss=0.0, iou=0.0, hit=0.0, n=0)
    for batch in loader:
        sp, bx = prep(batch, device, dtype)
        model.reset_state(int(sp.shape[1]), device, dtype)
        pred = model(sp)
        loss = perm_loss(pred, bx)
        iou, hit = metrics(pred, bx)
        tot["loss"] += float(loss); tot["iou"] += iou; tot["hit"] += hit; tot["n"] += 1
    d = max(tot["n"], 1)
    return {k: tot[k] / d for k in ("loss", "iou", "hit")}


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _px(boxes, W, H):
    o = boxes.copy()
    o[..., 0] *= W; o[..., 1] *= H; o[..., 2] *= W; o[..., 3] *= H
    return o


def plot_overlay(path: Path, spikes, gt, pred):
    T, H, W = spikes.shape
    idxs = np.linspace(0, T - 1, min(8, T), dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    gt_px = _px(gt, W, H); pred_px = _px(pred, W, H)
    for ax in axes.flat: ax.axis("off")
    for ax, t in zip(axes.flat, idxs):
        ax.imshow(spikes[t], cmap="gray", vmin=0, vmax=1)
        for o in range(2):
            ax.add_patch(patches.Rectangle(
                (gt_px[t, o, 0], gt_px[t, o, 1]), gt_px[t, o, 2], gt_px[t, o, 3],
                lw=1.8, edgecolor="lime", facecolor="none"))
            ax.add_patch(patches.Rectangle(
                (pred_px[t, o, 0], pred_px[t, o, 1]), pred_px[t, o, 2], pred_px[t, o, 3],
                lw=1.5, edgecolor="cyan", ls="--", facecolor="none"))
        ax.set_title(f"t={t}"); ax.axis("off")
    fig.suptitle("Qi & Gong 2022 full-scale — GT (lime) vs Pred (cyan)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history, path: Path):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, ttl in zip(axes, ("loss", "iou", "hit"), ("Loss", "Mean IoU", "Hit@0.5")):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], label="train")
        ax.plot(epochs, [r[f"val_{key}"] for r in history], label="val")
        ax.set(title=ttl, xlabel="Epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Qi & Gong 2022 full-scale training curves")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype  = torch.float32

    net_cfg = QiGong2022Config(hw=args.hw, N_i=args.N_i, seed=args.seed)
    N_e = (2 * args.hw + 1) ** 2
    print(f"Qi & Gong 2022 full-scale: hw={args.hw}, N_e={N_e}, N_i={args.N_i}, "
          f"total={N_e + args.N_i}")
    t0 = time.time()
    graph = build_qi_gong_2022_graph(net_cfg)
    print(f"  Graph built in {time.time()-t0:.1f}s  "
          f"(exc_edges={graph.n_exc_edges:,}, inh_edges={graph.n_inh_edges:,})")

    data_cfg = MotionCfg(
        timesteps=args.timesteps, height=args.height, width=args.width,
        sigma=args.sigma, base_rate=args.base_rate, peak_rate=args.peak_rate,
        speed_min=args.speed_min, speed_max=args.speed_max,
    )
    train_set = MotionDataset(args.train_samples, data_cfg, seed=args.seed)
    val_set   = MotionDataset(args.val_samples,   data_cfg, seed=args.seed + 7)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    input_dim = data_cfg.height * data_cfg.width
    backbone  = QiGong2022Backbone(graph=graph, input_dim=input_dim, dtype=dtype)
    model     = QiGong2022Detector(backbone=backbone, hidden=256).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}  (input_proj + head only)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, dtype)
        va = evaluate(model, val_loader, device, dtype)
        elapsed = time.time() - t_ep

        row = dict(epoch=epoch,
                   train_loss=tr["loss"], train_iou=tr["iou"], train_hit=tr["hit"],
                   val_loss=va["loss"],   val_iou=va["iou"],   val_hit=va["hit"])
        history.append(row)

        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    args=vars(args), epoch=epoch, history=history)
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save(ckpt, out_dir / "checkpoint_best.pth")

        print(
            f"epoch={epoch:03d}/{args.epochs}  "
            f"train loss={tr['loss']:.4f} iou={tr['iou']:.4f} hit={tr['hit']:.4f}  "
            f"val loss={va['loss']:.4f} iou={va['iou']:.4f} hit={va['hit']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

    hist_path = out_dir / "train_history.txt"
    with hist_path.open("w") as f:
        for r in history:
            f.write(
                "epoch={epoch} "
                "train_loss={train_loss:.6f} train_iou={train_iou:.6f} train_hit={train_hit:.6f} "
                "val_loss={val_loss:.6f} val_iou={val_iou:.6f} val_hit={val_hit:.6f}\n"
                .format(**r)
            )

    plot_curves(history, out_dir / "training_curves.png")
    print(f"\nBest val IoU: {best_iou:.4f}")
    print(f"Saved to: {out_dir}")


@torch.no_grad()
def run_infer(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype  = torch.float32

    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved     = ckpt.get("args", {})
    hw        = saved.get("hw",  args.hw)
    N_i       = saved.get("N_i", args.N_i)

    net_cfg  = QiGong2022Config(hw=hw, N_i=N_i, seed=saved.get("seed", args.seed))
    print(f"Rebuilding graph: hw={hw}, N_e={(2*hw+1)**2}, N_i={N_i}...")
    graph    = build_qi_gong_2022_graph(net_cfg)

    data_cfg = MotionCfg(
        timesteps=args.timesteps, height=args.height, width=args.width,
        sigma=args.sigma, base_rate=args.base_rate, peak_rate=args.peak_rate,
        speed_min=args.speed_min, speed_max=args.speed_max,
    )
    backbone = QiGong2022Backbone(graph=graph, input_dim=data_cfg.height * data_cfg.width, dtype=dtype)
    model    = QiGong2022Detector(backbone=backbone, hidden=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    infer_set = MotionDataset(args.num_sequences, data_cfg, seed=args.seed + 123)
    loader    = DataLoader(infer_set, batch_size=1, shuffle=False)

    all_gt, all_pred = [], []
    first_sp = first_gt = first_pred = None

    for batch in loader:
        sp, bx = prep(batch, device, dtype)
        model.reset_state(1, device, dtype)
        pred = model(sp)
        gt_np   = bx[:, 0].cpu().numpy()
        pred_np = pred[:, 0].cpu().numpy()
        sp_np   = sp[:, 0].cpu().numpy()
        all_gt.append(gt_np); all_pred.append(pred_np)
        if first_sp is None:
            first_sp, first_gt, first_pred = sp_np, gt_np, pred_np

    gt_arr   = np.stack(all_gt)
    pred_arr = np.stack(all_pred)
    mean_iou, hit_rate = metrics(torch.as_tensor(pred_arr), torch.as_tensor(gt_arr))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if first_sp is not None:
        plot_overlay(out_dir / "overlay.png", first_sp, first_gt, first_pred)

    np.save(out_dir / "pred_boxes.npy", pred_arr)
    np.save(out_dir / "gt_boxes.npy",   gt_arr)

    summary = out_dir / "infer_summary.txt"
    with summary.open("w") as f:
        f.write(f"checkpoint:       {args.checkpoint}\n")
        f.write(f"n_sequences:      {args.num_sequences}\n")
        f.write(f"n_neuron:         {graph.n_neuron}\n")
        f.write(f"hw={hw}, N_i={N_i}\n")
        f.write(f"mean_iou:         {mean_iou:.6f}\n")
        f.write(f"hit_rate_iou>0.5: {hit_rate:.6f}\n")

    print(f"mean_iou={mean_iou:.4f}, hit_rate={hit_rate:.4f}")
    print(f"Results saved to: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hw",         type=int,   default=31)
    p.add_argument("--N-i",        type=int,   default=1000, dest="N_i")
    p.add_argument("--height",     type=int,   default=48)
    p.add_argument("--width",      type=int,   default=64)
    p.add_argument("--timesteps",  type=int,   default=40)
    p.add_argument("--sigma",      type=float, default=3.0)
    p.add_argument("--base-rate",  type=float, default=0.01)
    p.add_argument("--peak-rate",  type=float, default=0.35)
    p.add_argument("--speed-min",  type=float, default=0.30)
    p.add_argument("--speed-max",  type=float, default=0.90)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--cpu",        action="store_true")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qi & Gong 2022 full-scale (N=4969) SpikeNet→btorch motion detection"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    _add_common(tr)
    tr.add_argument("--train-samples", type=int,   default=800)
    tr.add_argument("--val-samples",   type=int,   default=160)
    tr.add_argument("--batch-size",    type=int,   default=4)
    tr.add_argument("--epochs",        type=int,   default=60)
    tr.add_argument("--lr",            type=float, default=2e-3)
    tr.add_argument("--output-dir",    type=str,   default="output/qi_gong_2022_full")

    inf = sub.add_parser("infer")
    _add_common(inf)
    inf.add_argument("--checkpoint",    type=str, required=True)
    inf.add_argument("--num-sequences", type=int, default=40)
    inf.add_argument("--output-dir",    type=str, default="output/qi_gong_2022_full_infer")

    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)


if __name__ == "__main__":
    main()
