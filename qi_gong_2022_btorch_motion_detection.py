"""Qi & Gong 2022 cortical network model migrated to btorch.

Architecture (scaled down from the original paper):
  - 2D spatially embedded excitatory population: N_e = (2*hw+1)^2
  - Inhibitory population: N_i neurons
  - 4 synapse types: EE (AMPA), IE->E (GABA), E->I (AMPA), II (GABA)
  - Spike-frequency adaptation on all neurons
  - Distance-dependent EE connectivity on toroidal 2D lattice
  - Log-normal EE weights; fixed weights for other pathways

The network backbone is plugged into btorch's RecurrentNN scheduler, then
trained/evaluated on the two-Gaussian spike motion detection task.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
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
    candidate = ROOT / subdir
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from btorch.models import environ, functional, rnn
from btorch.models.neurons.spikenet import SpikeNetNeuron


# ---------------------------------------------------------------------------
# Qi & Gong 2022 model parameters
# ---------------------------------------------------------------------------

@dataclass
class QiGong2022Config:
    # Network topology
    hw: int = 7              # half-width of 2D excitatory lattice; N_e=(2*hw+1)^2
    N_i: int = 100           # inhibitory population size
    seed: int = 42

    # Simulation
    dt: float = 0.1          # ms

    # Neuron (all populations share same params; from SpikeNet defaults + Qi2022)
    tau_ref: float = 4.0     # ms (Qi2022 uses tau_ref=4)
    v_threshold: float = -50.0
    v_reset: float = -60.0
    v_lk: float = -70.0
    c_m: float = 0.25
    g_lk: float = 0.0167
    # SFA parameters (writeSpikeFreqAdptHDF5 in MATLAB → SpikeNetNeuron.spike_freq_adapt)
    spike_freq_adapt: bool = True
    dg_k: float = 0.01
    tau_k: float = 80.0
    v_k: float = -85.0

    # Connectivity
    delay_max_ms: float = 4.0   # random delays in [dt, delay_max_ms]
    P0_init: float = 0.08       # EE base connection probability
    tau_c_EE: float = 8.0       # EE spatial scale (grid units, ~sigma of Gaussian kernel)
    tau_c_IE: float = 10.0      # E->I spatial scale
    tau_c_I: float = 20.0       # I->E, I->I spatial scale
    P_ei: float = 0.20          # E->I probability (P_mat[0,1]*2=0.2)
    P_ie: float = 0.20          # I->E probability (P_mat[1,0]*2=0.2)
    P_ii: float = 0.40          # I->I probability (P_mat[1,1]*2=0.4)

    # Synaptic weights (µS, same notation as MATLAB)
    g_mu: float = 4e-3          # mean EE log-normal weight
    g_EI: float = 13.5e-3       # I->E (GABA weight)
    g_IE: float = 5e-3          # E->I (AMPA weight)
    g_II: float = 25e-3         # I->I (GABA weight)

    # Scaling factors (from existing btorch-SpikeNet alignment demos)
    excitatory_scale: float = 65.0
    inhibitory_scale: float = -15.0


# ---------------------------------------------------------------------------
# Graph dataclass (mirrors existing gaussian_motion_detection demo)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Qi & Gong 2022 graph builder
# ---------------------------------------------------------------------------

def _accumulate_weight(
    bucket: dict[int, np.ndarray],
    delay_step: int,
    dst: int,
    src: int,
    value: float,
    n_neuron: int,
) -> None:
    if delay_step not in bucket:
        bucket[delay_step] = np.zeros((n_neuron, n_neuron), dtype=np.float32)
    bucket[delay_step][dst, src] += value


def _torus_dist2(
    coords: np.ndarray,          # [N, 2] in (-pi, pi)
    j: int,
) -> np.ndarray:                 # [N]
    d = coords - coords[j]
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return (d ** 2).sum(axis=1)


def build_qi_gong_2022_graph(cfg: QiGong2022Config) -> SpikeNetGraph:
    rng = np.random.default_rng(cfg.seed)

    N_e = (2 * cfg.hw + 1) ** 2
    n_neuron = N_e + cfg.N_i
    delay_max_steps = max(1, int(round(cfg.delay_max_ms / cfg.dt)))

    # 2D toroidal lattice coordinates for excitatory population, in (-pi, pi)
    x1d = np.linspace(-np.pi, np.pi, 2 * cfg.hw + 1, endpoint=False)
    yy, xx = np.meshgrid(x1d, x1d, indexing="ij")
    coords_e = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)  # [N_e, 2]

    # Random placement for inhibitory neurons (quasi-uniform on torus)
    coords_i = rng.uniform(-np.pi, np.pi, (cfg.N_i, 2)).astype(np.float32)

    exc_bucket: dict[int, np.ndarray] = {}
    inh_bucket: dict[int, np.ndarray] = {}

    # ---- EE connections: distance-dependent Gaussian kernel ----
    # sigma in radians corresponds to tau_c_EE grid steps
    spacing = 2 * np.pi / (2 * cfg.hw + 1)
    sigma_EE = cfg.tau_c_EE * spacing
    sigma2_EE = max(sigma_EE ** 2, 1e-6)

    lognormal_sigma = 0.3  # log-normal spread (matches Qi2022 distribution)
    mu_p = math.log(cfg.g_mu) - 0.5 * lognormal_sigma ** 2

    for j in range(N_e):
        d2 = _torus_dist2(coords_e, j)
        p_conn = cfg.P0_init * np.exp(-0.5 * d2 / sigma2_EE)
        p_conn[j] = 0.0  # no self-connections
        mask = rng.random(N_e) < p_conn
        srcs = np.where(mask)[0]
        if len(srcs) == 0:
            continue
        weights = rng.lognormal(mu_p, lognormal_sigma, len(srcs)).astype(np.float32)
        delays = rng.integers(1, delay_max_steps + 1, len(srcs))
        for idx, s in enumerate(srcs):
            _accumulate_weight(
                exc_bucket, int(delays[idx]), j, int(s),
                cfg.excitatory_scale * float(weights[idx]), n_neuron,
            )

    # ---- I->E connections (GABA, pop_pre=inhibitory, pop_post=excitatory) ----
    sigma_I = cfg.tau_c_I * spacing
    sigma2_I = max(sigma_I ** 2, 1e-6)

    for j in range(N_e):  # post = excitatory neuron j
        # distance from each inhibitory neuron to j
        dy = coords_i[:, 0] - coords_e[j, 0]
        dx = coords_i[:, 1] - coords_e[j, 1]
        dy = (dy + np.pi) % (2 * np.pi) - np.pi
        dx = (dx + np.pi) % (2 * np.pi) - np.pi
        d2 = dy ** 2 + dx ** 2
        p_conn = cfg.P_ie * np.exp(-0.5 * d2 / sigma2_I)
        mask = rng.random(cfg.N_i) < p_conn
        srcs_i = np.where(mask)[0]
        if len(srcs_i) == 0:
            continue
        delays = rng.integers(1, delay_max_steps + 1, len(srcs_i))
        for idx, si in enumerate(srcs_i):
            src_global = N_e + int(si)
            _accumulate_weight(
                inh_bucket, int(delays[idx]), j, src_global,
                cfg.inhibitory_scale * cfg.g_EI, n_neuron,
            )

    # ---- E->I connections (AMPA) ----
    sigma_IE = cfg.tau_c_IE * spacing
    sigma2_IE = max(sigma_IE ** 2, 1e-6)

    for ji in range(cfg.N_i):  # post = inhibitory neuron ji
        dst_global = N_e + ji
        dy = coords_e[:, 0] - coords_i[ji, 0]
        dx = coords_e[:, 1] - coords_i[ji, 1]
        dy = (dy + np.pi) % (2 * np.pi) - np.pi
        dx = (dx + np.pi) % (2 * np.pi) - np.pi
        d2 = dy ** 2 + dx ** 2
        p_conn = cfg.P_ei * np.exp(-0.5 * d2 / sigma2_IE)
        mask = rng.random(N_e) < p_conn
        srcs_e = np.where(mask)[0]
        if len(srcs_e) == 0:
            continue
        delays = rng.integers(1, delay_max_steps + 1, len(srcs_e))
        for idx, se in enumerate(srcs_e):
            _accumulate_weight(
                exc_bucket, int(delays[idx]), dst_global, int(se),
                cfg.excitatory_scale * cfg.g_IE, n_neuron,
            )

    # ---- I->I connections (GABA) ----
    for ji in range(cfg.N_i):  # post = inhibitory neuron ji
        dst_global = N_e + ji
        dy = coords_i[:, 0] - coords_i[ji, 0]
        dx = coords_i[:, 1] - coords_i[ji, 1]
        dy = (dy + np.pi) % (2 * np.pi) - np.pi
        dx = (dx + np.pi) % (2 * np.pi) - np.pi
        d2 = dy ** 2 + dx ** 2
        p_conn = cfg.P_ii * np.exp(-0.5 * d2 / sigma2_I)
        p_conn[ji] = 0.0
        mask = rng.random(cfg.N_i) < p_conn
        srcs_i2 = np.where(mask)[0]
        if len(srcs_i2) == 0:
            continue
        delays = rng.integers(1, delay_max_steps + 1, len(srcs_i2))
        for idx, si in enumerate(srcs_i2):
            src_global = N_e + int(si)
            _accumulate_weight(
                inh_bucket, int(delays[idx]), dst_global, src_global,
                cfg.inhibitory_scale * cfg.g_II, n_neuron,
            )

    def to_tensor_dict(data: dict[int, np.ndarray]) -> dict[int, torch.Tensor]:
        if not data:
            return {0: torch.zeros((n_neuron, n_neuron), dtype=torch.float32)}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

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
        exc_weights_by_delay=to_tensor_dict(exc_bucket),
        inh_weights_by_delay=to_tensor_dict(inh_bucket),
    )


# ---------------------------------------------------------------------------
# btorch synapse modules
# ---------------------------------------------------------------------------

class DelayedLinearPSC(nn.Module):
    def __init__(self, n_neuron: int, weights_by_delay: dict[int, torch.Tensor], tau_syn: float):
        super().__init__()
        self.n_neuron = n_neuron
        self.delay_keys = sorted(int(k) for k in weights_by_delay)
        self.max_delay = max(self.delay_keys) if self.delay_keys else 0
        self.buffer_len = self.max_delay + 1
        self.tau_syn = float(tau_syn)
        for delay in self.delay_keys:
            self.register_buffer(f"w_{delay}", weights_by_delay[delay].clone(), persistent=False)
        self._cursor = 0
        self._history: torch.Tensor | None = None
        self.psc: torch.Tensor | None = None

    def _ensure_state(self, z: torch.Tensor) -> None:
        need = (
            self._history is None or self.psc is None
            or self._history.shape[1:] != z.shape
            or self._history.device != z.device
            or self._history.dtype != z.dtype
        )
        if need:
            self._history = torch.zeros((self.buffer_len, *z.shape), device=z.device, dtype=z.dtype)
            self.psc = torch.zeros_like(z)
            self._cursor = 0

    def init_state(self, batch_size: int, device=None, dtype=None, **kw) -> None:
        self._history = torch.zeros((self.buffer_len, batch_size, self.n_neuron), device=device, dtype=dtype)
        self.psc = torch.zeros((batch_size, self.n_neuron), device=device, dtype=dtype)
        self._cursor = 0

    def reset(self, batch_size: int, device=None, dtype=None, **kw) -> None:
        self.init_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_state(z)
        self._history[self._cursor] = z
        driven = torch.zeros_like(z)
        for delay in self.delay_keys:
            idx = (self._cursor - delay) % self.buffer_len
            z_delayed = self._history[idx]
            weight = getattr(self, f"w_{delay}")
            driven = driven + F.linear(z_delayed, weight)
        dt = float(environ.get("dt"))
        alpha = math.exp(-dt / max(self.tau_syn, 1e-6))
        self.psc = alpha * self.psc + (1.0 - alpha) * driven
        self._cursor = (self._cursor + 1) % self.buffer_len
        return self.psc


class QiGong2022Synapse(nn.Module):
    def __init__(
        self,
        n_neuron: int,
        exc_weights_by_delay: dict[int, torch.Tensor],
        inh_weights_by_delay: dict[int, torch.Tensor],
    ):
        super().__init__()
        self.n_neuron = n_neuron
        self.exc = DelayedLinearPSC(n_neuron, exc_weights_by_delay, tau_syn=5.0)
        self.inh = DelayedLinearPSC(n_neuron, inh_weights_by_delay, tau_syn=3.0)
        self.psc: torch.Tensor = torch.zeros((1, n_neuron), dtype=torch.float32)

    def init_state(self, batch_size: int, device=None, dtype=None, **kw) -> None:
        self.exc.init_state(batch_size=batch_size, device=device, dtype=dtype)
        self.inh.init_state(batch_size=batch_size, device=device, dtype=dtype)
        self.psc = torch.zeros((batch_size, self.n_neuron), device=device, dtype=dtype)

    def reset(self, batch_size: int, device=None, dtype=None, **kw) -> None:
        self.init_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.psc = self.exc(z) + self.inh(z)
        return self.psc


# ---------------------------------------------------------------------------
# btorch backbone and detector
# ---------------------------------------------------------------------------

class QiGong2022Backbone(nn.Module):
    """SpikeNet Qi & Gong 2022 cortical model as btorch backbone."""

    def __init__(self, graph: SpikeNetGraph, input_dim: int, dtype: torch.dtype):
        super().__init__()
        self.graph = graph
        self.dt = graph.dt
        self.input_proj = nn.Linear(input_dim, graph.n_neuron, bias=False, dtype=dtype)

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

        synapse = QiGong2022Synapse(
            n_neuron=graph.n_neuron,
            exc_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.exc_weights_by_delay.items()},
            inh_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.inh_weights_by_delay.items()},
        )

        self.brain = rnn.RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            update_state_names=("neuron.v",),
        )
        functional.init_net_state(self.brain, batch_size=1, device=torch.device("cpu"), dtype=dtype)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        functional.reset_net(self.brain, batch_size=batch_size, device=device, dtype=dtype)
        self.brain.synapse.reset(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, events: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = events.flatten(start_dim=2)
        x = self.input_proj(x)
        with environ.context(dt=self.dt):
            spike, states = self.brain(x)
        return spike, states


class QiGong2022MotionDetector(nn.Module):
    def __init__(self, backbone: QiGong2022Backbone, n_object: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.n_object = n_object
        self.head = nn.Sequential(
            nn.Linear(backbone.graph.n_neuron, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_object * 4),
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.backbone.reset_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        spike, _ = self.backbone(events)
        t, b, n = spike.shape
        pred = self.head(spike.reshape(t * b, n)).reshape(t, b, self.n_object, 4)
        return torch.sigmoid(pred)


# ---------------------------------------------------------------------------
# Two-Gaussian motion dataset (same as existing demo)
# ---------------------------------------------------------------------------

@dataclass
class GaussianMotionConfig:
    timesteps: int = 40
    height: int = 48
    width: int = 64
    sigma: float = 2.5
    base_rate: float = 0.01
    peak_rate: float = 0.30
    speed_min: float = 0.35
    speed_max: float = 0.95


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gaussian_map(xg, yg, cx, cy, sigma):
    return np.exp(-0.5 * ((xg - cx) ** 2 + (yg - cy) ** 2) / max(sigma ** 2, 1e-6))


def _box_from_center(cx, cy, sigma, width, height):
    bw = max(4.0, 4.0 * sigma)
    bh = max(4.0, 4.0 * sigma)
    x = float(np.clip(cx - bw / 2, 0.0, width - bw))
    y = float(np.clip(cy - bh / 2, 0.0, height - bh))
    return np.array([x / width, y / height, bw / width, bh / height], dtype=np.float32)


def _reflect(v, lo, hi):
    if v < lo:
        return lo + (lo - v)
    if v > hi:
        return hi - (v - hi)
    return v


def generate_two_gaussian_sequence(cfg: GaussianMotionConfig, rng: np.random.Generator):
    T, H, W = cfg.timesteps, cfg.height, cfg.width
    margin = 4.0 * cfg.sigma
    centers = rng.uniform(margin, [W - margin, H - margin], (2, 2)).astype(np.float32)
    velocities = np.zeros((2, 2), dtype=np.float32)
    for o in range(2):
        sp = rng.uniform(cfg.speed_min, cfg.speed_max)
        ang = rng.uniform(0.0, 2 * math.pi)
        velocities[o] = [sp * math.cos(ang), sp * math.sin(ang)]

    yg, xg = np.mgrid[0:H, 0:W]
    spikes = np.zeros((T, H, W), dtype=np.float32)
    boxes = np.zeros((T, 2, 4), dtype=np.float32)

    for t in range(T):
        rate = np.full((H, W), cfg.base_rate, dtype=np.float32)
        for o in range(2):
            cx, cy = float(centers[o, 0]), float(centers[o, 1])
            rate += cfg.peak_rate * _gaussian_map(xg, yg, cx, cy, cfg.sigma).astype(np.float32)
            boxes[t, o] = _box_from_center(cx, cy, cfg.sigma, W, H)
            nx = _reflect(centers[o, 0] + velocities[o, 0], margin, W - margin)
            ny = _reflect(centers[o, 1] + velocities[o, 1], margin, H - margin)
            if abs(nx - (centers[o, 0] + velocities[o, 0])) > 1e-6:
                velocities[o, 0] *= -1
            if abs(ny - (centers[o, 1] + velocities[o, 1])) > 1e-6:
                velocities[o, 1] *= -1
            centers[o] = [nx, ny]
        spikes[t] = (rng.random((H, W)) < np.clip(rate, 0.0, 0.95)).astype(np.float32)

    return spikes, boxes


class TwoGaussianDataset(Dataset):
    def __init__(self, n: int, cfg: GaussianMotionConfig, seed: int):
        rng = np.random.default_rng(seed)
        self.samples = [
            (torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))
            for s, b in (generate_two_gaussian_sequence(cfg, rng) for _ in range(n))
        ]

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def perm_invariant_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t_swap = target[:, :, [1, 0], :]
    l1 = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(-1, -2))
    l2 = F.smooth_l1_loss(pred, t_swap, reduction="none").mean(dim=(-1, -2))
    return torch.minimum(l1, l2).mean()


def box_iou_xywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px2, py2 = pred[..., 0] + pred[..., 2], pred[..., 1] + pred[..., 3]
    tx2, ty2 = target[..., 0] + target[..., 2], target[..., 1] + target[..., 3]
    inter = (torch.minimum(px2, tx2) - torch.maximum(pred[..., 0], target[..., 0])).clamp(0) * \
            (torch.minimum(py2, ty2) - torch.maximum(pred[..., 1], target[..., 1])).clamp(0)
    area_p = pred[..., 2].clamp(0) * pred[..., 3].clamp(0)
    area_t = target[..., 2].clamp(0) * target[..., 3].clamp(0)
    union = area_p + area_t - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def motion_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    t_swap = target[:, :, [1, 0], :]
    iou_d = box_iou_xywh(pred, target).mean(dim=(-1, -2))
    iou_s = box_iou_xywh(pred, t_swap).mean(dim=(-1, -2))
    best = torch.maximum(iou_d, iou_s)
    return float(best.mean()), float((best > 0.5).float().mean())


def _prep(batch, device, dtype):
    s, b = batch
    return s.permute(1, 0, 2, 3).to(device=device, dtype=dtype), \
           b.permute(1, 0, 2, 3).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, dtype):
    model.train()
    totals = dict(loss=0.0, iou=0.0, hit=0.0, n=0)
    for batch in loader:
        sp, bx = _prep(batch, device, dtype)
        model.reset_state(int(sp.shape[1]), device, dtype)
        pred = model(sp)
        loss = perm_invariant_loss(pred, bx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iou, hit = motion_metrics(pred.detach(), bx)
        totals["loss"] += float(loss); totals["iou"] += iou; totals["hit"] += hit; totals["n"] += 1
    d = max(totals["n"], 1)
    return {k: totals[k] / d for k in ("loss", "iou", "hit")}


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    totals = dict(loss=0.0, iou=0.0, hit=0.0, n=0)
    for batch in loader:
        sp, bx = _prep(batch, device, dtype)
        model.reset_state(int(sp.shape[1]), device, dtype)
        pred = model(sp)
        loss = perm_invariant_loss(pred, bx)
        iou, hit = motion_metrics(pred, bx)
        totals["loss"] += float(loss); totals["iou"] += iou; totals["hit"] += hit; totals["n"] += 1
    d = max(totals["n"], 1)
    return {k: totals[k] / d for k in ("loss", "iou", "hit")}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _denorm(boxes, W, H):
    out = boxes.copy()
    out[..., 0] *= W; out[..., 1] *= H; out[..., 2] *= W; out[..., 3] *= H
    return out


def plot_overlay(out_path: Path, spikes: np.ndarray, gt_boxes: np.ndarray, pred_boxes: np.ndarray):
    T, H, W = spikes.shape
    idxs = np.linspace(0, T - 1, num=min(8, T), dtype=int)
    cols, rows = 4, int(math.ceil(len(idxs) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    gt_px = _denorm(gt_boxes, W, H)
    pred_px = _denorm(pred_boxes, W, H)
    for ax in axes.flat: ax.axis("off")
    for ax, t in zip(axes.flat, idxs):
        ax.imshow(spikes[t], cmap="gray", vmin=0, vmax=1)
        for o in range(2):
            ax.add_patch(patches.Rectangle((gt_px[t,o,0], gt_px[t,o,1]), gt_px[t,o,2], gt_px[t,o,3],
                                           lw=1.8, edgecolor="lime", facecolor="none"))
            ax.add_patch(patches.Rectangle((pred_px[t,o,0], pred_px[t,o,1]), pred_px[t,o,2], pred_px[t,o,3],
                                           lw=1.5, edgecolor="cyan", ls="--", facecolor="none"))
        ax.set_title(f"t={int(t)}")
        ax.axis("off")
    fig.suptitle("Qi & Gong 2022 btorch — GT (lime) vs Pred (cyan)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(history: list[dict], out_path: Path):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ("loss", "iou", "hit"), ("Loss", "IoU", "Hit@0.5")):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], label="train")
        ax.plot(epochs, [r[f"val_{key}"] for r in history], label="val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Qi & Gong 2022 btorch — Training Curves")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    net_cfg = QiGong2022Config(hw=args.hw, N_i=args.N_i, seed=args.seed)
    print(f"Building Qi & Gong 2022 graph: hw={args.hw}, N_e={(2*args.hw+1)**2}, N_i={args.N_i} ...")
    graph = build_qi_gong_2022_graph(net_cfg)
    n_exc_syn = sum(v.abs().sum().item() > 0 for v in graph.exc_weights_by_delay.values())
    n_inh_syn = sum(v.abs().sum().item() > 0 for v in graph.inh_weights_by_delay.values())
    print(f"  n_neuron={graph.n_neuron}, exc_delay_buckets={len(graph.exc_weights_by_delay)}, inh_delay_buckets={len(graph.inh_weights_by_delay)}")

    data_cfg = GaussianMotionConfig(
        timesteps=args.timesteps, height=args.height, width=args.width,
        sigma=args.sigma, base_rate=args.base_rate, peak_rate=args.peak_rate,
        speed_min=args.speed_min, speed_max=args.speed_max,
    )
    train_set = TwoGaussianDataset(args.train_samples, data_cfg, seed=args.seed)
    val_set   = TwoGaussianDataset(args.val_samples,   data_cfg, seed=args.seed + 7)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    backbone = QiGong2022Backbone(graph=graph, input_dim=data_cfg.height * data_cfg.width, dtype=dtype)
    model = QiGong2022MotionDetector(backbone=backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, dtype)
        va = evaluate(model, val_loader, device, dtype)
        row = dict(epoch=epoch, train_loss=tr["loss"], train_iou=tr["iou"], train_hit=tr["hit"],
                   val_loss=va["loss"], val_iou=va["iou"], val_hit=va["hit"])
        history.append(row)

        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    args=vars(args), epoch=epoch, history=history, net_cfg=vars(net_cfg))
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save(ckpt, out_dir / "checkpoint_best.pth")

        print(f"epoch={epoch:03d} "
              f"train_loss={tr['loss']:.4f} train_iou={tr['iou']:.4f} train_hit={tr['hit']:.4f} "
              f"val_loss={va['loss']:.4f} val_iou={va['iou']:.4f} val_hit={va['hit']:.4f}")

    hist_path = out_dir / "train_history.txt"
    with hist_path.open("w") as f:
        for r in history:
            f.write("epoch={epoch} train_loss={train_loss:.6f} train_iou={train_iou:.6f} "
                    "train_hit={train_hit:.6f} val_loss={val_loss:.6f} val_iou={val_iou:.6f} "
                    "val_hit={val_hit:.6f}\n".format(**r))

    plot_training_curve(history, out_dir / "training_curves.png")
    print(f"Saved to: {out_dir}")


@torch.no_grad()
def run_infer(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    hw  = saved_args.get("hw",  args.hw)
    N_i = saved_args.get("N_i", args.N_i)

    net_cfg = QiGong2022Config(hw=hw, N_i=N_i, seed=saved_args.get("seed", args.seed))
    graph = build_qi_gong_2022_graph(net_cfg)

    data_cfg = GaussianMotionConfig(
        timesteps=args.timesteps, height=args.height, width=args.width,
        sigma=args.sigma, base_rate=args.base_rate, peak_rate=args.peak_rate,
        speed_min=args.speed_min, speed_max=args.speed_max,
    )

    backbone = QiGong2022Backbone(graph=graph, input_dim=data_cfg.height * data_cfg.width, dtype=dtype)
    model = QiGong2022MotionDetector(backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    infer_set = TwoGaussianDataset(args.num_sequences, data_cfg, seed=args.seed + 123)
    loader = DataLoader(infer_set, batch_size=1, shuffle=False)

    all_gt, all_pred = [], []
    first_spikes = first_gt = first_pred = None

    for batch in loader:
        sp, bx = _prep(batch, device, dtype)
        model.reset_state(1, device, dtype)
        pred = model(sp)
        gt_np = bx[:, 0].cpu().numpy()
        pr_np = pred[:, 0].cpu().numpy()
        sp_np = sp[:, 0].cpu().numpy()
        all_gt.append(gt_np); all_pred.append(pr_np)
        if first_spikes is None:
            first_spikes, first_gt, first_pred = sp_np, gt_np, pr_np

    gt_arr   = np.stack(all_gt)
    pred_arr = np.stack(all_pred)
    mean_iou, hit_rate = motion_metrics(torch.as_tensor(pred_arr), torch.as_tensor(gt_arr))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if first_spikes is not None:
        plot_overlay(out_dir / "overlay.png", first_spikes, first_gt, first_pred)

    np.save(out_dir / "pred_boxes.npy", pred_arr)
    np.save(out_dir / "gt_boxes.npy",   gt_arr)

    summary = out_dir / "infer_summary.txt"
    with summary.open("w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"n_sequences: {args.num_sequences}\n")
        f.write(f"n_neuron: {graph.n_neuron}\n")
        f.write(f"hw: {hw}, N_i: {N_i}\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"hit_rate_iou>0.5: {hit_rate:.6f}\n")

    print(f"mean_iou={mean_iou:.4f}, hit_rate={hit_rate:.4f}")
    print(f"Results saved to: {out_dir}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hw",         type=int,   default=7)
    p.add_argument("--N-i",        type=int,   default=100, dest="N_i")
    p.add_argument("--height",     type=int,   default=48)
    p.add_argument("--width",      type=int,   default=64)
    p.add_argument("--timesteps",  type=int,   default=40)
    p.add_argument("--sigma",      type=float, default=2.5)
    p.add_argument("--base-rate",  type=float, default=0.01)
    p.add_argument("--peak-rate",  type=float, default=0.30)
    p.add_argument("--speed-min",  type=float, default=0.35)
    p.add_argument("--speed-max",  type=float, default=0.95)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--cpu",        action="store_true")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qi & Gong 2022 cortical network in btorch — two-Gaussian motion detection")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    _add_common(tr)
    tr.add_argument("--train-samples", type=int,   default=200)
    tr.add_argument("--val-samples",   type=int,   default=50)
    tr.add_argument("--batch-size",    type=int,   default=8)
    tr.add_argument("--epochs",        type=int,   default=20)
    tr.add_argument("--lr",            type=float, default=1e-3)
    tr.add_argument("--output-dir",    type=str,   default="output/qi_gong_2022_train")

    inf = sub.add_parser("infer")
    _add_common(inf)
    inf.add_argument("--checkpoint",    type=str, required=True)
    inf.add_argument("--num-sequences", type=int, default=20)
    inf.add_argument("--output-dir",    type=str, default="output/qi_gong_2022_infer")

    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)


if __name__ == "__main__":
    main()
