from __future__ import annotations

import argparse
import dataclasses
import math
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.models.neurons.spikenet import SpikeNetNeuron


@dataclasses.dataclass
class EdgeSpec:
    src: int
    dst: int
    syn_type: int  # 0=AMPA, 1=GABA
    weight: float
    delay_ms: float


@dataclasses.dataclass
class CaseSpec:
    name: str
    title: str
    dt: float
    step_tot: int
    init_v: list[float]
    i_ext_mean: list[float]
    edges: list[EdgeSpec]

    @property
    def n_node(self) -> int:
        return len(self.init_v)


def build_cases() -> list[CaseSpec]:
    case_a = CaseSpec(
        name="case_a_single_neuron",
        title="(a) Single Neuron",
        dt=0.1,
        step_tot=500,
        init_v=[-70.0],
        i_ext_mean=[0.52],
        edges=[],
    )

    case_b = CaseSpec(
        name="case_b_two_neurons",
        title="(b) Upstream-Downstream Two Neurons",
        dt=0.1,
        step_tot=600,
        init_v=[-70.0, -70.0],
        i_ext_mean=[0.56, 0.0],
        edges=[
            EdgeSpec(src=0, dst=1, syn_type=0, weight=0.08, delay_ms=0.4),
        ],
    )

    edges_c: list[EdgeSpec] = []
    # Excitatory ring.
    for i in range(10):
        edges_c.append(
            EdgeSpec(
                src=i,
                dst=(i + 1) % 10,
                syn_type=0,
                weight=0.05,
                delay_ms=0.2 + 0.1 * (i % 3),
            )
        )
    # Long-range excitatory shortcuts.
    for i in [0, 2, 4, 6, 8]:
        edges_c.append(
            EdgeSpec(
                src=i,
                dst=(i + 3) % 10,
                syn_type=0,
                weight=0.035,
                delay_ms=0.5,
            )
        )
    # Inhibitory hub-like projections.
    for dst in [1, 3, 5, 9]:
        edges_c.append(
            EdgeSpec(
                src=7,
                dst=dst,
                syn_type=1,
                weight=0.06,
                delay_ms=0.3,
            )
        )

    i_ext_c = [0.0] * 10
    i_ext_c[0] = 0.56
    i_ext_c[4] = 0.53
    i_ext_c[7] = 0.50

    case_c = CaseSpec(
        name="case_c_ten_neurons_nontrivial",
        title="(c) Ten Neurons with Non-trivial Topology",
        dt=0.1,
        step_tot=700,
        init_v=[-70.0] * 10,
        i_ext_mean=i_ext_c,
        edges=edges_c,
    )

    return [case_a, case_b, case_c]


def write_case_h5(case: CaseSpec, case_path: Path) -> None:
    case_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(case_path, "w") as h5:
        h5.create_dataset(
            "/config/Net/INIT001/N",
            data=np.ones((case.n_node,), dtype=np.int32),
        )
        h5.create_dataset(
            "/config/Net/INIT002/dt",
            data=np.asarray([case.dt], dtype=np.float64),
        )
        h5.create_dataset(
            "/config/Net/INIT002/step_tot",
            data=np.asarray([case.step_tot], dtype=np.int32),
        )

        for pop_idx in range(case.n_node):
            pop_root = f"/config/pops/pop{pop_idx}"
            h5.create_group(f"{pop_root}/SAMP003")
            h5.create_dataset(
                f"{pop_root}/SETINITV/external_init_V",
                data=np.asarray([case.init_v[pop_idx]], dtype=np.float64),
            )
            if abs(case.i_ext_mean[pop_idx]) > 1e-12:
                h5.create_dataset(
                    f"{pop_root}/INIT004/mean",
                    data=np.asarray([case.i_ext_mean[pop_idx]], dtype=np.float64),
                )
                h5.create_dataset(
                    f"{pop_root}/INIT004/std",
                    data=np.asarray([0.0], dtype=np.float64),
                )

        h5.create_dataset(
            "/config/syns/n_syns",
            data=np.asarray([len(case.edges)], dtype=np.int32),
        )
        for syn_idx, edge in enumerate(case.edges):
            syn_root = f"/config/syns/syn{syn_idx}/INIT006"
            h5.create_dataset(
                f"{syn_root}/type", data=np.asarray([edge.syn_type], dtype=np.int32)
            )
            h5.create_dataset(
                f"{syn_root}/i_pre", data=np.asarray([edge.src], dtype=np.int32)
            )
            h5.create_dataset(
                f"{syn_root}/j_post", data=np.asarray([edge.dst], dtype=np.int32)
            )
            h5.create_dataset(f"{syn_root}/I", data=np.asarray([0], dtype=np.int32))
            h5.create_dataset(f"{syn_root}/J", data=np.asarray([0], dtype=np.int32))
            h5.create_dataset(
                f"{syn_root}/K", data=np.asarray([edge.weight], dtype=np.float64)
            )
            h5.create_dataset(
                f"{syn_root}/D", data=np.asarray([edge.delay_ms], dtype=np.float64)
            )
            h5.create_group(f"/config/syns/syn{syn_idx}/SAMP004")


def run_cpp_simulator(simulator: Path, case_path: Path) -> Path:
    stem = case_path.with_suffix("")
    before = set(stem.parent.glob(f"{stem.name}_*_out.h5"))

    proc = subprocess.run(
        [str(simulator), str(case_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ simulator failed for {case_path.name}.\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )

    after = set(stem.parent.glob(f"{stem.name}_*_out.h5"))
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not created:
        raise RuntimeError(
            f"No output *_out.h5 detected for {case_path.name}.\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )
    return created[-1]


def read_cpp_results(out_h5: Path, n_node: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(out_h5, "r") as h5:
        v_series = []
        spike_series = []
        for idx in range(n_node):
            v = np.asarray(h5[f"/pop_result_{idx}/stats_V_mean"]).reshape(-1)
            spk = np.asarray(h5[f"/pop_result_{idx}/num_spikes_pop"]).reshape(-1)
            v_series.append(v)
            spike_series.append(spk)
    return np.stack(v_series, axis=0), np.stack(spike_series, axis=0)


class BtorchSynapseModel0:
    def __init__(self, edge: EdgeSpec, dt: float):
        self.src = edge.src
        self.dst = edge.dst
        self.syn_type = edge.syn_type
        self.weight = float(edge.weight)

        if self.syn_type == 0:
            tau_rise = 1.0
            tau_decay = 5.0
            self.v_rev = 0.0
        elif self.syn_type == 1:
            tau_rise = 1.0
            tau_decay = 3.0
            self.v_rev = -80.0
        else:
            raise ValueError(f"Unsupported syn_type for demo: {self.syn_type}")

        self.steps_trans = max(1, int(round(tau_rise / dt)))
        self.delay_steps = int(round(edge.delay_ms / dt))
        self.exp_step_decay = math.exp(-dt / tau_decay)
        self.k_trans = 1.0 / self.steps_trans

        self.buffer_steps = self.delay_steps + self.steps_trans + 1
        self.buffer = np.zeros((self.buffer_steps,), dtype=np.float64)
        self.trans_left = 0
        self.s_pre = 0.0
        self.gs_sum = 0.0

    def update(
        self,
        step: int,
        pre_spike: float,
        v_post: float,
    ) -> float:
        if pre_spike > 0.0:
            self.trans_left += self.steps_trans

        if self.trans_left > 0:
            t_ring = (step + self.delay_steps) % self.buffer_steps
            self.buffer[t_ring] += self.k_trans * (1.0 - self.s_pre) * self.weight
            self.trans_left -= 1
            self.s_pre += self.k_trans * (1.0 - self.s_pre)

        self.s_pre *= self.exp_step_decay

        t_now = step % self.buffer_steps
        self.gs_sum += self.buffer[t_now]
        self.gs_sum *= self.exp_step_decay
        self.buffer[t_now] = 0.0

        return -self.gs_sum * (v_post - self.v_rev)


def run_btorch_simulator(case: CaseSpec) -> tuple[np.ndarray, np.ndarray]:
    dt = case.dt
    dtype = torch.float64

    pops: list[SpikeNetNeuron] = []
    for i in range(case.n_node):
        neuron = SpikeNetNeuron(
            n_neuron=1,
            neuron_model="lif",
            v_threshold=-50.0,
            v_reset=-60.0,
            v_lk=-70.0,
            c_m=0.25,
            g_lk=0.0167,
            tau_ref=2.0,
            spike_freq_adapt=False,
            dtype=dtype,
        )
        neuron.init_state()
        neuron.v[...] = torch.as_tensor([case.init_v[i]], dtype=dtype)
        pops.append(neuron)

    synapses = [BtorchSynapseModel0(edge, dt=dt) for edge in case.edges]

    v_trace = np.zeros((case.n_node, case.step_tot), dtype=np.float64)
    spike_trace = np.zeros((case.n_node, case.step_tot), dtype=np.float64)

    for step in range(case.step_tot):
        spikes = []
        for pop in pops:
            can_spike = pop.ref_step_left == 0
            spike = ((pop.v >= pop.v_threshold) & can_spike).to(pop.v.dtype)
            spikes.append(spike)

        for idx, pop in enumerate(pops):
            spike = spikes[idx]
            pop.v = pop.v - (pop.v - pop.v_reset) * spike

            ref_steps = torch.round(torch.clamp_min(pop.tau_ref, 0.0) / dt).to(
                torch.int64
            )
            ref_next = torch.where(spike > 0, ref_steps, pop.ref_step_left)
            ref_next = torch.where(ref_next > 0, ref_next - 1, ref_next)
            pop.ref_step_left = ref_next

            spike_trace[idx, step] = float(spike.item())

        syn_current = np.zeros((case.n_node,), dtype=np.float64)
        for syn in synapses:
            pre_spike = float(spikes[syn.src].item())
            v_post = float(pops[syn.dst].v.item())
            syn_current[syn.dst] += syn.update(step=step, pre_spike=pre_spike, v_post=v_post)

        for idx, pop in enumerate(pops):
            i_ext = case.i_ext_mean[idx]
            x = torch.as_tensor([i_ext + syn_current[idx]], dtype=dtype)

            pop.i_k.zero_()
            pop.i_leak = -pop.g_lk * (pop.v - pop.v_lk)
            pop.i_input = x + pop.i_k
            vdot = (pop.i_leak + pop.i_input) / pop.c_m
            non_ref = pop.ref_step_left == 0
            pop.v = torch.where(non_ref, pop.v + vdot * dt, pop.v)
            v_trace[idx, step] = float(pop.v.item())

    return v_trace, spike_trace


def draw_topology(ax: plt.Axes, case: CaseSpec) -> None:
    n = case.n_node
    if n == 1:
        pos = {0: (0.0, 0.0)}
    elif n == 2:
        pos = {0: (-0.8, 0.0), 1: (0.8, 0.0)}
    else:
        pos = {
            i: (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        }

    for edge in case.edges:
        x0, y0 = pos[edge.src]
        x1, y1 = pos[edge.dst]
        color = "tab:blue" if edge.syn_type == 0 else "tab:red"
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.8),
        )

    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=250, color="white", edgecolors="black", zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, zorder=4)

    ax.set_title("Topology")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_case_comparison(
    case: CaseSpec,
    t_ms: np.ndarray,
    v_cpp: np.ndarray,
    v_bt: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2])
    ax_topo = fig.add_subplot(gs[0, 0])
    ax_dyn = fig.add_subplot(gs[0, 1])

    draw_topology(ax_topo, case)

    cmap = plt.cm.get_cmap("tab10", case.n_node)
    for i in range(case.n_node):
        color = cmap(i % 10)
        ax_dyn.plot(
            t_ms,
            v_cpp[i],
            color=color,
            lw=1.8,
            label=f"Neuron {i} C++",
        )
        ax_dyn.plot(
            t_ms,
            v_bt[i],
            color=color,
            lw=1.2,
            ls="--",
            label=f"Neuron {i} btorch",
        )

    ax_dyn.set_title("Membrane Potential Comparison")
    ax_dyn.set_xlabel("Time (ms)")
    ax_dyn.set_ylabel("V (mV)")
    ax_dyn.grid(alpha=0.3)

    if case.n_node <= 3:
        ax_dyn.legend(loc="best", fontsize=8)
    else:
        handles, labels = ax_dyn.get_legend_handles_labels()
        ax_dyn.legend(handles[:10], labels[:10], loc="upper right", fontsize=7)

    fig.suptitle(case.title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_summary(
    summary_rows: list[tuple[str, int, float, float]],
    output_path: Path,
) -> None:
    names = [r[0] for r in summary_rows]
    n_nodes = np.asarray([r[1] for r in summary_rows], dtype=np.int32)
    rmse = np.asarray([r[2] for r in summary_rows], dtype=np.float64)
    max_abs = np.asarray([r[3] for r in summary_rows], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    bw = 0.36

    ax1.bar(x - bw / 2, rmse, width=bw, label="Mean RMSE of V", color="tab:blue")
    ax1.bar(x + bw / 2, max_abs, width=bw, label="Max |ΔV|", color="tab:orange")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in n_nodes])
    ax1.set_ylabel("Voltage error (mV)")
    ax1.set_title("Error Summary Across Network Scales")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper left")

    for i, name in enumerate(names):
        ax1.text(x[i], max(rmse[i], max_abs[i]) * 1.02 + 1e-9, name, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def ensure_controlled_difference(summary_rows: list[tuple[str, int, float, float]]) -> None:
    # A soft guardrail: mean RMSE should not scale explosively with network size.
    # We consider it controlled if the largest-case RMSE is <= 2x single-neuron RMSE + 0.5mV.
    rows_sorted = sorted(summary_rows, key=lambda x: x[1])
    rmse_small = rows_sorted[0][2]
    rmse_large = rows_sorted[-1][2]
    if rmse_large > (2.0 * rmse_small + 0.5):
        raise RuntimeError(
            "Observed mismatch appears to scale with network size beyond the configured guardrail. "
            f"small-case RMSE={rmse_small:.4f}, large-case RMSE={rmse_large:.4f}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing SpikeNet/ and btorch/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Demo output directory.",
    )
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    simulator = workspace_root / "SpikeNet" / "simulator"
    if not simulator.exists():
        raise FileNotFoundError(f"SpikeNet simulator not found at: {simulator}")

    case_dir = args.output_dir / "cases"
    fig_dir = args.output_dir / "figures"
    result_dir = args.output_dir / "results"
    case_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, int, float, float]] = []
    cases = build_cases()

    for case in cases:
        case_path = case_dir / f"{case.name}_in.h5"
        write_case_h5(case, case_path)

        cpp_out = run_cpp_simulator(simulator=simulator, case_path=case_path)
        v_cpp, spk_cpp = read_cpp_results(cpp_out, n_node=case.n_node)

        v_bt, spk_bt = run_btorch_simulator(case)

        t_ms = np.arange(case.step_tot, dtype=np.float64) * case.dt

        plot_case_comparison(
            case=case,
            t_ms=t_ms,
            v_cpp=v_cpp,
            v_bt=v_bt,
            output_path=fig_dir / f"{case.name}.png",
        )

        dv = v_cpp - v_bt
        mean_rmse = float(np.mean(np.sqrt(np.mean(np.square(dv), axis=1))))
        max_abs = float(np.max(np.abs(dv)))

        summary_rows.append((case.name, case.n_node, mean_rmse, max_abs))

        with h5py.File(result_dir / f"{case.name}_comparison.h5", "w") as h5:
            h5.create_dataset("time_ms", data=t_ms)
            h5.create_dataset("v_cpp", data=v_cpp)
            h5.create_dataset("v_btorch", data=v_bt)
            h5.create_dataset("spike_cpp", data=spk_cpp)
            h5.create_dataset("spike_btorch", data=spk_bt)
            h5.create_dataset("dv", data=dv)

    ensure_controlled_difference(summary_rows)
    plot_error_summary(
        summary_rows=summary_rows,
        output_path=fig_dir / "error_summary.png",
    )

    print("=== SpikeNet (C++) vs btorch SpikeNetNeuron demo completed ===")
    for name, n_node, rmse, max_abs in summary_rows:
        print(
            f"{name}: N={n_node}, mean_RMSE(V)={rmse:.6f} mV, max_abs(V)={max_abs:.6f} mV"
        )
    print(f"Figures: {fig_dir}")
    print(f"Result tensors: {result_dir}")


if __name__ == "__main__":
    main()
