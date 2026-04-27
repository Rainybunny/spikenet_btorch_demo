from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
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
for subdir in ("btorch", "SpikeNet"):
    candidate = ROOT / subdir
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from spikenet_py.config import SimulationConfig, load_config_h5

from btorch.models import environ, functional, rnn
from btorch.models.neurons.spikenet import SpikeNetNeuron


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
    exc_weights_by_delay: dict[int, torch.Tensor]
    inh_weights_by_delay: dict[int, torch.Tensor]
    nmda_weights_by_delay: dict[int, torch.Tensor]


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


def _population_offsets(pop_sizes: list[int]) -> np.ndarray:
    offsets = np.zeros((len(pop_sizes) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(pop_sizes, dtype=np.int64))
    return offsets


def _fill_population_param(
    dest: np.ndarray,
    start: int,
    size: int,
    value: float,
) -> None:
    dest[start : start + size] = value


def _accumulate_weight(
    bucket: dict[int, np.ndarray],
    delay_step: int,
    dst_idx: int,
    src_idx: int,
    value: float,
    n_neuron: int,
) -> None:
    if delay_step not in bucket:
        bucket[delay_step] = np.zeros((n_neuron, n_neuron), dtype=np.float32)
    bucket[delay_step][dst_idx, src_idx] += value


def spikenet_config_to_graph(config_path: Path) -> SpikeNetGraph:
    config: SimulationConfig = load_config_h5(str(config_path))
    pop_sizes = [int(x) for x in config.pop_sizes]
    offsets = _population_offsets(pop_sizes)
    n_neuron = int(offsets[-1])

    v_threshold = np.zeros((n_neuron,), dtype=np.float32)
    v_reset = np.zeros((n_neuron,), dtype=np.float32)
    v_lk = np.zeros((n_neuron,), dtype=np.float32)
    c_m = np.zeros((n_neuron,), dtype=np.float32)
    g_lk = np.zeros((n_neuron,), dtype=np.float32)
    tau_ref = np.zeros((n_neuron,), dtype=np.float32)

    for pop in config.populations:
        start = int(offsets[pop.index])
        size = int(pop.size)
        params = pop.params

        _fill_population_param(v_threshold, start, size, float(params.get("V_th", -50.0)))
        _fill_population_param(v_reset, start, size, float(params.get("V_rt", -60.0)))
        _fill_population_param(v_lk, start, size, float(params.get("V_lk", -70.0)))
        _fill_population_param(c_m, start, size, float(params.get("Cm", 0.25)))
        _fill_population_param(g_lk, start, size, float(params.get("g_lk", 0.0167)))
        _fill_population_param(tau_ref, start, size, float(params.get("tau_ref", 2.0)))

    exc_bucket: dict[int, np.ndarray] = {}
    inh_bucket: dict[int, np.ndarray] = {}
    nmda_bucket: dict[int, np.ndarray] = {}

    # Empirical scaling factors borrowed from existing SpikeNet/btorch alignment demos.
    excitatory_scale = 65.0
    inhibitory_scale = -15.0
    nmda_scale = 25.0

    for syn in config.synapses:
        pre_off = int(offsets[syn.pop_pre])
        post_off = int(offsets[syn.pop_post])

        src = pre_off + np.asarray(syn.i, dtype=np.int64)
        dst = post_off + np.asarray(syn.j, dtype=np.int64)
        weights = np.asarray(syn.k, dtype=np.float32)
        delays = np.clip(np.round(np.asarray(syn.d, dtype=np.float64) / float(config.dt)).astype(np.int64), 0, None)

        for s, d, w, delay_step in zip(src, dst, weights, delays, strict=False):
            if s < 0 or s >= n_neuron or d < 0 or d >= n_neuron:
                continue
            if syn.syn_type == 0:
                _accumulate_weight(exc_bucket, int(delay_step), int(d), int(s), excitatory_scale * float(w), n_neuron)
            elif syn.syn_type == 1:
                _accumulate_weight(inh_bucket, int(delay_step), int(d), int(s), inhibitory_scale * float(w), n_neuron)
            elif syn.syn_type == 2:
                _accumulate_weight(nmda_bucket, int(delay_step), int(d), int(s), nmda_scale * float(w), n_neuron)

    def to_tensor_dict(data: dict[int, np.ndarray]) -> dict[int, torch.Tensor]:
        if not data:
            return {0: torch.zeros((n_neuron, n_neuron), dtype=torch.float32)}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    return SpikeNetGraph(
        dt=float(config.dt),
        n_neuron=n_neuron,
        v_threshold=torch.as_tensor(v_threshold),
        v_reset=torch.as_tensor(v_reset),
        v_lk=torch.as_tensor(v_lk),
        c_m=torch.as_tensor(c_m),
        g_lk=torch.as_tensor(g_lk),
        tau_ref=torch.as_tensor(tau_ref),
        exc_weights_by_delay=to_tensor_dict(exc_bucket),
        inh_weights_by_delay=to_tensor_dict(inh_bucket),
        nmda_weights_by_delay=to_tensor_dict(nmda_bucket),
    )


class DelayedLinearPSC(nn.Module):
    def __init__(
        self,
        n_neuron: int,
        weights_by_delay: dict[int, torch.Tensor],
        tau_syn: float,
    ) -> None:
        super().__init__()
        self.n_neuron = int(n_neuron)
        self.delay_keys = sorted(int(k) for k in weights_by_delay)
        self.max_delay = max(self.delay_keys) if self.delay_keys else 0
        self.buffer_len = self.max_delay + 1
        self.tau_syn = float(tau_syn)

        for delay in self.delay_keys:
            self.register_buffer(
                f"w_{delay}",
                weights_by_delay[delay].clone(),
                persistent=False,
            )

        self._cursor = 0
        self._history: torch.Tensor | None = None
        self.psc: torch.Tensor | None = None

    def _ensure_state(self, z: torch.Tensor) -> None:
        need_init = (
            self._history is None
            or self.psc is None
            or self._history.shape[1:] != z.shape
            or self._history.device != z.device
            or self._history.dtype != z.dtype
        )
        if need_init:
            self._history = torch.zeros(
                (self.buffer_len, *z.shape),
                device=z.device,
                dtype=z.dtype,
            )
            self.psc = torch.zeros_like(z)
            self._cursor = 0

    def init_state(self, batch_size: int, device: torch.device = None, dtype: torch.dtype = None, **kwargs) -> None:
        template = torch.zeros((batch_size, self.n_neuron), device=device, dtype=dtype)
        self._history = torch.zeros((self.buffer_len, *template.shape), device=device, dtype=dtype)
        self.psc = torch.zeros_like(template)
        self._cursor = 0

    def reset(self, batch_size: int, device: torch.device = None, dtype: torch.dtype = None, **kwargs) -> None:
        self.init_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure_state(z)
        assert self._history is not None
        assert self.psc is not None

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


class CompositeSpikeNetSynapse(nn.Module):
    def __init__(
        self,
        n_neuron: int,
        exc_weights_by_delay: dict[int, torch.Tensor],
        inh_weights_by_delay: dict[int, torch.Tensor],
        nmda_weights_by_delay: dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.n_neuron = int(n_neuron)
        self.exc = DelayedLinearPSC(n_neuron, exc_weights_by_delay, tau_syn=5.0)
        self.inh = DelayedLinearPSC(n_neuron, inh_weights_by_delay, tau_syn=3.0)
        self.nmda = DelayedLinearPSC(n_neuron, nmda_weights_by_delay, tau_syn=80.0)
        self.psc: torch.Tensor = torch.zeros((1, n_neuron), dtype=torch.float32)

    def init_state(self, batch_size: int, device: torch.device = None, dtype: torch.dtype = None, **kwargs) -> None:
        self.exc.init_state(batch_size=batch_size, device=device, dtype=dtype)
        self.inh.init_state(batch_size=batch_size, device=device, dtype=dtype)
        self.nmda.init_state(batch_size=batch_size, device=device, dtype=dtype)
        self.psc = torch.zeros((batch_size, self.n_neuron), device=device, dtype=dtype)

    def reset(self, batch_size: int, device: torch.device = None, dtype: torch.dtype = None, **kwargs) -> None:
        self.init_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.psc = self.exc(z) + self.inh(z) + self.nmda(z)
        return self.psc


class SpikeNetBtorchBackbone(nn.Module):
    def __init__(
        self,
        graph: SpikeNetGraph,
        input_dim: int,
        dtype: torch.dtype,
    ) -> None:
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
            dtype=dtype,
        )

        synapse = CompositeSpikeNetSynapse(
            n_neuron=graph.n_neuron,
            exc_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.exc_weights_by_delay.items()},
            inh_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.inh_weights_by_delay.items()},
            nmda_weights_by_delay={k: v.to(dtype=dtype) for k, v in graph.nmda_weights_by_delay.items()},
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

    def forward(self, events: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # events: [T, B, H, W]
        x = events.flatten(start_dim=2)
        x = self.input_proj(x)

        with environ.context(dt=self.dt):
            spike, states = self.brain(x)
        return spike, states


class SpikeNetMotionDetector(nn.Module):
    def __init__(
        self,
        backbone: SpikeNetBtorchBackbone,
        n_object: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_object = int(n_object)
        self.feature_dim = backbone.graph.n_neuron
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_object * 4),
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.backbone.reset_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: [T, B, H, W]
        spike, states = self.backbone(events)
        t, b, n = spike.shape
        pred = self.head(spike.reshape(t * b, n)).reshape(t, b, self.n_object, 4)
        return torch.sigmoid(pred)


def _gaussian_map(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    d2 = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    return np.exp(-0.5 * d2 / max(sigma * sigma, 1e-6))


def _box_from_center(
    center_x: float,
    center_y: float,
    sigma: float,
    width: int,
    height: int,
) -> np.ndarray:
    box_w = max(4.0, 4.0 * sigma)
    box_h = max(4.0, 4.0 * sigma)
    x = float(np.clip(center_x - box_w / 2.0, 0.0, width - box_w))
    y = float(np.clip(center_y - box_h / 2.0, 0.0, height - box_h))

    return np.array([
        x / width,
        y / height,
        box_w / width,
        box_h / height,
    ], dtype=np.float32)


def _reflect(value: float, low: float, high: float) -> float:
    if value < low:
        return low + (low - value)
    if value > high:
        return high - (value - high)
    return value


def generate_two_gaussian_peak_sequence(
    cfg: GaussianMotionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    t_steps, h, w = cfg.timesteps, cfg.height, cfg.width
    margin = 4.0 * cfg.sigma

    centers = np.array(
        [
            [rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)],
            [rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)],
        ],
        dtype=np.float32,
    )

    velocities = np.zeros((2, 2), dtype=np.float32)
    for obj in range(2):
        speed = rng.uniform(cfg.speed_min, cfg.speed_max)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        velocities[obj, 0] = speed * math.cos(angle)
        velocities[obj, 1] = speed * math.sin(angle)

    y_grid, x_grid = np.mgrid[0:h, 0:w]

    spikes = np.zeros((t_steps, h, w), dtype=np.float32)
    boxes = np.zeros((t_steps, 2, 4), dtype=np.float32)

    for t in range(t_steps):
        rate = np.full((h, w), cfg.base_rate, dtype=np.float32)

        for obj in range(2):
            cx, cy = float(centers[obj, 0]), float(centers[obj, 1])
            gauss = _gaussian_map(x_grid, y_grid, cx, cy, cfg.sigma)
            rate += cfg.peak_rate * gauss.astype(np.float32)
            boxes[t, obj] = _box_from_center(cx, cy, cfg.sigma, w, h)

            nx = centers[obj, 0] + velocities[obj, 0]
            ny = centers[obj, 1] + velocities[obj, 1]
            nx_ref = _reflect(float(nx), margin, w - margin)
            ny_ref = _reflect(float(ny), margin, h - margin)
            if abs(nx_ref - nx) > 1e-6:
                velocities[obj, 0] *= -1.0
            if abs(ny_ref - ny) > 1e-6:
                velocities[obj, 1] *= -1.0
            centers[obj, 0] = nx_ref
            centers[obj, 1] = ny_ref

        rate = np.clip(rate, 0.0, 0.95)
        spikes[t] = (rng.random((h, w)) < rate).astype(np.float32)

    return spikes, boxes


class TwoGaussianMotionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        num_sequences: int,
        cfg: GaussianMotionConfig,
        seed: int,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        rng = np.random.default_rng(seed)

        for _ in range(num_sequences):
            spikes, boxes = generate_two_gaussian_peak_sequence(cfg, rng)
            self.samples.append(
                (
                    torch.as_tensor(spikes, dtype=torch.float32),
                    torch.as_tensor(boxes, dtype=torch.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def permutation_invariant_box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [T, B, 2, 4], normalized xywh
    target_swap = target[:, :, [1, 0], :]
    loss_direct = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(-1, -2))
    loss_swap = F.smooth_l1_loss(pred, target_swap, reduction="none").mean(dim=(-1, -2))
    return torch.minimum(loss_direct, loss_swap).mean()


def box_iou_xywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [..., 4], normalized xywh
    px1 = pred[..., 0]
    py1 = pred[..., 1]
    px2 = pred[..., 0] + pred[..., 2]
    py2 = pred[..., 1] + pred[..., 3]

    tx1 = target[..., 0]
    ty1 = target[..., 1]
    tx2 = target[..., 0] + target[..., 2]
    ty2 = target[..., 1] + target[..., 3]

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = area_p + area_t - inter

    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def batch_motion_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    target_swap = target[:, :, [1, 0], :]

    iou_direct = box_iou_xywh(pred, target).mean(dim=(-1, -2))
    iou_swap = box_iou_xywh(pred, target_swap).mean(dim=(-1, -2))

    best_iou = torch.maximum(iou_direct, iou_swap)
    hit = (best_iou > 0.5).float()
    return float(best_iou.mean().item()), float(hit.mean().item())


def _prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    spikes, boxes = batch
    # [B, T, H, W] -> [T, B, H, W]
    spikes_t = spikes.permute(1, 0, 2, 3).to(device=device, dtype=dtype)
    # [B, T, 2, 4] -> [T, B, 2, 4]
    boxes_t = boxes.permute(1, 0, 2, 3).to(device=device, dtype=dtype)
    return spikes_t, boxes_t


def train_one_epoch(
    model: SpikeNetMotionDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_hit = 0.0
    total_batches = 0

    for batch in loader:
        spikes_t, boxes_t = _prepare_batch(batch, device=device, dtype=dtype)
        batch_size = int(spikes_t.shape[1])
        model.reset_state(batch_size=batch_size, device=device, dtype=dtype)

        pred = model(spikes_t)
        loss = permutation_invariant_box_loss(pred, boxes_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iou, hit = batch_motion_metrics(pred.detach(), boxes_t)
        total_loss += float(loss.item())
        total_iou += iou
        total_hit += hit
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "iou": total_iou / denom,
        "hit": total_hit / denom,
    }


@torch.no_grad()
def evaluate(
    model: SpikeNetMotionDetector,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_hit = 0.0
    total_batches = 0

    for batch in loader:
        spikes_t, boxes_t = _prepare_batch(batch, device=device, dtype=dtype)
        batch_size = int(spikes_t.shape[1])
        model.reset_state(batch_size=batch_size, device=device, dtype=dtype)

        pred = model(spikes_t)
        loss = permutation_invariant_box_loss(pred, boxes_t)
        iou, hit = batch_motion_metrics(pred, boxes_t)

        total_loss += float(loss.item())
        total_iou += iou
        total_hit += hit
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "iou": total_iou / denom,
        "hit": total_hit / denom,
    }


def _denorm_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = boxes.copy()
    out[..., 0] *= width
    out[..., 1] *= height
    out[..., 2] *= width
    out[..., 3] *= height
    return out


def write_mot_txt(
    out_path: Path,
    boxes_seq: np.ndarray,
    width: int,
    height: int,
    conf: float = 1.0,
) -> None:
    # boxes_seq: [S, T, 2, 4], normalized xywh
    boxes_px = _denorm_boxes(boxes_seq, width=width, height=height)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_id = 1
    with out_path.open("w", encoding="utf-8") as f:
        for seq_id in range(boxes_px.shape[0]):
            for t in range(boxes_px.shape[1]):
                for obj in range(2):
                    x, y, w, h = boxes_px[seq_id, t, obj]
                    track_id = seq_id * 10 + obj + 1
                    f.write(
                        f"{frame_id},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.3f},-1,-1,-1\n"
                    )
                frame_id += 1


def plot_overlay(
    out_path: Path,
    spikes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
) -> None:
    # spikes: [T, H, W], boxes: [T, 2, 4], normalized
    t_steps, h, w = spikes.shape
    idx = np.linspace(0, t_steps - 1, num=min(8, t_steps), dtype=int)

    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    gt_px = _denorm_boxes(gt_boxes, width=w, height=h)
    pred_px = _denorm_boxes(pred_boxes, width=w, height=h)

    for ax in axes.flat:
        ax.axis("off")

    for ax, t in zip(axes.flat, idx, strict=False):
        ax.imshow(spikes[t], cmap="gray", vmin=0, vmax=1)
        for obj in range(2):
            gx, gy, gw, gh = gt_px[t, obj]
            px, py, pw, ph = pred_px[t, obj]

            ax.add_patch(
                patches.Rectangle(
                    (gx, gy),
                    gw,
                    gh,
                    linewidth=1.8,
                    edgecolor="lime",
                    facecolor="none",
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (px, py),
                    pw,
                    ph,
                    linewidth=1.5,
                    edgecolor="cyan",
                    linestyle="--",
                    facecolor="none",
                )
            )

        ax.set_title(f"frame {int(t)}")
        ax.axis("off")

    fig.suptitle("Two-Gaussian Spike Motion: GT (lime) vs Pred (cyan)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_model(
    config_path: Path,
    input_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SpikeNetMotionDetector:
    graph = spikenet_config_to_graph(config_path)
    backbone = SpikeNetBtorchBackbone(graph=graph, input_dim=input_dim, dtype=dtype)
    model = SpikeNetMotionDetector(backbone=backbone)
    return model.to(device)


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    data_cfg = GaussianMotionConfig(
        timesteps=args.timesteps,
        height=args.height,
        width=args.width,
        sigma=args.sigma,
        base_rate=args.base_rate,
        peak_rate=args.peak_rate,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )

    train_set = TwoGaussianMotionDataset(args.train_samples, data_cfg, seed=args.seed)
    val_set = TwoGaussianMotionDataset(args.val_samples, data_cfg, seed=args.seed + 7)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(
        config_path=Path(args.spikenet_config),
        input_dim=data_cfg.height * data_cfg.width,
        dtype=dtype,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device=device, dtype=dtype)
        val_stats = evaluate(model, val_loader, device=device, dtype=dtype)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_hit": train_stats["hit"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_hit": val_stats["hit"],
        }
        history.append(row)

        ckpt_latest = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "history": history,
        }
        torch.save(ckpt_latest, out_dir / "checkpoint_latest.pth")

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            torch.save(ckpt_latest, out_dir / "checkpoint_best.pth")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} train_iou={train_stats['iou']:.4f} train_hit={train_stats['hit']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_iou={val_stats['iou']:.4f} val_hit={val_stats['hit']:.4f}"
        )

    history_path = out_dir / "train_history.txt"
    with history_path.open("w", encoding="utf-8") as f:
        for row in history:
            f.write(
                "epoch={epoch} train_loss={train_loss:.6f} train_iou={train_iou:.6f} train_hit={train_hit:.6f} "
                "val_loss={val_loss:.6f} val_iou={val_iou:.6f} val_hit={val_hit:.6f}\n".format(**row)
            )

    print(f"Saved checkpoints and logs to: {out_dir}")


def run_infer(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    data_cfg = GaussianMotionConfig(
        timesteps=args.timesteps,
        height=args.height,
        width=args.width,
        sigma=args.sigma,
        base_rate=args.base_rate,
        peak_rate=args.peak_rate,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )

    model = build_model(
        config_path=Path(args.spikenet_config),
        input_dim=data_cfg.height * data_cfg.width,
        dtype=dtype,
        device=device,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    infer_set = TwoGaussianMotionDataset(args.num_sequences, data_cfg, seed=args.seed + 123)
    infer_loader = DataLoader(infer_set, batch_size=1, shuffle=False)

    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    first_spikes: np.ndarray | None = None
    first_gt: np.ndarray | None = None
    first_pred: np.ndarray | None = None

    with torch.no_grad():
        for batch in infer_loader:
            spikes_t, boxes_t = _prepare_batch(batch, device=device, dtype=dtype)
            model.reset_state(batch_size=1, device=device, dtype=dtype)
            pred = model(spikes_t)

            gt_np = boxes_t[:, 0].detach().cpu().numpy()
            pred_np = pred[:, 0].detach().cpu().numpy()
            spikes_np = spikes_t[:, 0].detach().cpu().numpy()

            all_gt.append(gt_np)
            all_pred.append(pred_np)

            if first_spikes is None:
                first_spikes = spikes_np
                first_gt = gt_np
                first_pred = pred_np

    gt_arr = np.stack(all_gt, axis=0)
    pred_arr = np.stack(all_pred, axis=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_mot = out_dir / "pred_mot.txt"
    gt_mot = out_dir / "gt_mot.txt"
    write_mot_txt(pred_mot, pred_arr, width=data_cfg.width, height=data_cfg.height)
    write_mot_txt(gt_mot, gt_arr, width=data_cfg.width, height=data_cfg.height)

    if first_spikes is not None and first_gt is not None and first_pred is not None:
        plot_overlay(
            out_path=out_dir / "gaussian_motion_overlay.png",
            spikes=first_spikes,
            gt_boxes=first_gt,
            pred_boxes=first_pred,
        )

    # quick quality summary
    pred_t = torch.as_tensor(pred_arr, dtype=torch.float32)
    gt_t = torch.as_tensor(gt_arr, dtype=torch.float32)
    mean_iou, hit_rate = batch_motion_metrics(pred_t, gt_t)

    summary_path = out_dir / "infer_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"num_sequences: {args.num_sequences}\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"hit_rate_iou>0.5: {hit_rate:.6f}\n")
        f.write(f"pred_mot: {pred_mot}\n")
        f.write(f"gt_mot: {gt_mot}\n")

    print(f"Inference outputs saved to: {out_dir}")
    print(f"mean_iou={mean_iou:.4f}, hit_rate={hit_rate:.4f}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load SpikeNet structure into btorch, train/infer on two-Gaussian event motion, and export MOT-style boxes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--spikenet-config", type=str, required=True, help="SpikeNet HDF5 config path used to build network structure.")
        p.add_argument("--height", type=int, default=48)
        p.add_argument("--width", type=int, default=64)
        p.add_argument("--timesteps", type=int, default=40)
        p.add_argument("--sigma", type=float, default=2.5)
        p.add_argument("--base-rate", type=float, default=0.01)
        p.add_argument("--peak-rate", type=float, default=0.30)
        p.add_argument("--speed-min", type=float, default=0.35)
        p.add_argument("--speed-max", type=float, default=0.95)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    train_p = subparsers.add_parser("train", help="Train detector on synthetic two-Gaussian event sequences.")
    add_common(train_p)
    train_p.add_argument("--train-samples", type=int, default=256)
    train_p.add_argument("--val-samples", type=int, default=64)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--output-dir", type=str, default="output/gaussian_motion_train")

    infer_p = subparsers.add_parser("infer", help="Run inference and export MOT-style predictions.")
    add_common(infer_p)
    infer_p.add_argument("--checkpoint", type=str, required=True)
    infer_p.add_argument("--num-sequences", type=int, default=16)
    infer_p.add_argument("--output-dir", type=str, default="output/gaussian_motion_infer")

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
