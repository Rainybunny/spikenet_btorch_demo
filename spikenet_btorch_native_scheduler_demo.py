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

from btorch.models import environ, functional, rnn, synapse
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
        name="native_case_a_single_neuron",
        title="Native (a) Single Neuron",
        dt=0.1,
        step_tot=500,
        init_v=[-70.0],
        i_ext_mean=[0.52],
        edges=[],
    )

    case_b = CaseSpec(
        name="native_case_b_two_neurons",
        title="Native (b) Upstream-Downstream Two Neurons",
        dt=0.1,
        step_tot=600,
        init_v=[-70.0, -70.0],
        i_ext_mean=[0.56, 0.0],
        edges=[
            EdgeSpec(src=0, dst=1, syn_type=0, weight=0.06, delay_ms=0.0),
        ],
    )

    edges_c: list[EdgeSpec] = []
    for i in range(10):
        edges_c.append(
            EdgeSpec(
                src=i,
                dst=(i + 1) % 10,
                syn_type=0,
                weight=0.035,
                delay_ms=0.0,
            )
        )
    for i in [0, 2, 4, 6, 8]:
        edges_c.append(
            EdgeSpec(
                src=i,
                dst=(i + 3) % 10,
                syn_type=0,
                weight=0.028,
                delay_ms=0.0,
            )
        )
    for dst in [1, 3, 5, 9]:
        edges_c.append(
            EdgeSpec(
                src=7,
                dst=dst,
                syn_type=1,
                weight=0.045,
                delay_ms=0.0,
            )
        )

    i_ext_c = [0.0] * 10
    i_ext_c[0] = 0.55
    i_ext_c[4] = 0.53
    i_ext_c[7] = 0.50

    case_c = CaseSpec(
        name="native_case_c_ten_neurons_nontrivial",
        title="Native (c) Ten Neurons with Non-trivial Topology",
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


class NativeCompositePSC(torch.nn.Module):
    def __init__(self, n_neuron: int, w_exc: np.ndarray, w_inh: np.ndarray):
        super().__init__()
        self.n_neuron = (n_neuron,)
        self.size = n_neuron

        lin_exc = torch.nn.Linear(n_neuron, n_neuron, bias=False, dtype=torch.float64)
        lin_inh = torch.nn.Linear(n_neuron, n_neuron, bias=False, dtype=torch.float64)
        with torch.no_grad():
            lin_exc.weight.copy_(torch.as_tensor(w_exc, dtype=torch.float64))
            lin_inh.weight.copy_(torch.as_tensor(w_inh, dtype=torch.float64))

        self.exc = synapse.ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=5.0,
            linear=lin_exc,
        )
        self.inh = synapse.ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=3.0,
            linear=lin_inh,
        )

    @property
    def psc(self) -> torch.Tensor:
        return self.exc.psc + self.inh.psc

    def init_state(self, *args, **kwargs):
        self.exc.init_state(*args, **kwargs)
        self.inh.init_state(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.exc.reset(*args, **kwargs)
        self.inh.reset(*args, **kwargs)

    def forward(self, z: torch.Tensor):
        _ = self.exc(z)
        _ = self.inh(z)
        return self.psc


def build_native_btorch_network(case: CaseSpec) -> rnn.RecurrentNN:
    n = case.n_node

    # Approximate conductance-based C++ currents with current-based btorch PSC.
    # At V_ref=-65mV: AMPA scale~65, GABA scale~-15.
    w_exc = np.zeros((n, n), dtype=np.float64)
    w_inh = np.zeros((n, n), dtype=np.float64)
    for edge in case.edges:
        if edge.syn_type == 0:
            w_exc[edge.dst, edge.src] += 65.0 * edge.weight
        elif edge.syn_type == 1:
            w_inh[edge.dst, edge.src] += -15.0 * edge.weight

    neuron = SpikeNetNeuron(
        n_neuron=n,
        neuron_model="lif",
        v_threshold=-50.0,
        v_reset=-60.0,
        v_lk=-70.0,
        c_m=0.25,
        g_lk=0.0167,
        tau_ref=2.0,
        spike_freq_adapt=False,
        dtype=torch.float64,
    )

    psc = NativeCompositePSC(n_neuron=n, w_exc=w_exc, w_inh=w_inh)
    brain = rnn.RecurrentNN(
        neuron=neuron,
        synapse=psc,
        update_state_names=("neuron.v", "synapse.exc.psc", "synapse.inh.psc"),
    )
    functional.init_net_state(brain, dtype=torch.float64)
    brain.neuron.v[...] = torch.as_tensor(case.init_v, dtype=torch.float64)
    return brain


def run_btorch_native(case: CaseSpec) -> tuple[np.ndarray, np.ndarray]:
    brain = build_native_btorch_network(case)
    x_seq = np.tile(np.asarray(case.i_ext_mean, dtype=np.float64), (case.step_tot, 1))
    x = torch.as_tensor(x_seq, dtype=torch.float64)

    with torch.no_grad():
        with environ.context(dt=case.dt):
            spike, states = brain(x)

    v_bt = states["neuron.v"].detach().cpu().numpy().T
    spk_bt = spike.detach().cpu().numpy().T
    return v_bt, spk_bt


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
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.8),
        )

    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=240, color="white", edgecolors="black", zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, zorder=4)

    ax.set_title("Topology")
    ax.set_aspect("equal")
    ax.axis("off")


def pick_plot_neurons(n_node: int) -> list[int]:
    if n_node <= 3:
        return list(range(n_node))
    return [0, n_node // 2, n_node - 1]


def plot_case(
    case: CaseSpec,
    t_ms: np.ndarray,
    v_cpp: np.ndarray,
    v_bt: np.ndarray,
    spk_cpp: np.ndarray,
    spk_bt: np.ndarray,
    output_path: Path,
) -> None:
    neurons = pick_plot_neurons(case.n_node)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2], height_ratios=[1.0, 1.0])
    ax_topo = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    draw_topology(ax_topo, case)

    cmap = plt.get_cmap("tab10")
    for idx, neu in enumerate(neurons):
        color = cmap(idx)
        ax_v.plot(t_ms, v_cpp[neu], color=color, lw=1.8, label=f"N{neu} C++")
        ax_v.plot(t_ms, v_bt[neu], color=color, lw=1.2, ls="--", label=f"N{neu} btorch")

    ax_v.set_title("Membrane Potential")
    ax_v.set_ylabel("V (mV)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="best", fontsize=8)

    # Spike-rate proxy over selected neurons.
    spk_cpp_sel = spk_cpp[neurons].mean(axis=0)
    spk_bt_sel = spk_bt[neurons].mean(axis=0)
    ax_s.plot(t_ms, spk_cpp_sel, color="black", lw=1.6, label="C++ mean spike")
    ax_s.plot(t_ms, spk_bt_sel, color="black", lw=1.2, ls="--", label="btorch mean spike")
    ax_s.set_title("Mean Spike Activity (selected neurons)")
    ax_s.set_xlabel("Time (ms)")
    ax_s.set_ylabel("Spike")
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="best", fontsize=8)

    fig.suptitle(case.title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_summary(rows: list[tuple[str, int, float, float, float]], output_path: Path) -> None:
    names = [r[0] for r in rows]
    n_nodes = np.asarray([r[1] for r in rows], dtype=np.int32)
    rmse = np.asarray([r[2] for r in rows], dtype=np.float64)
    max_abs = np.asarray([r[3] for r in rows], dtype=np.float64)
    spike_l1 = np.asarray([r[4] for r in rows], dtype=np.float64)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bw = 0.24
    ax.bar(x - bw, rmse, width=bw, label="mean RMSE(V)", color="tab:blue")
    ax.bar(x, max_abs, width=bw, label="max |ΔV|", color="tab:orange")
    ax.bar(x + bw, spike_l1, width=bw, label="mean |Δspike|", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in n_nodes])
    ax.set_ylabel("Error")
    ax.set_title("Native Scheduler Error Summary")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    for i, name in enumerate(names):
        top = max(rmse[i], max_abs[i], spike_l1[i])
        ax.text(x[i], top * 1.02 + 1e-9, name, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def ensure_migration_error_controlled(rows: list[tuple[str, int, float, float, float]]) -> None:
    # We allow native-scheduler differences, but guard against scale explosion.
    rows_sorted = sorted(rows, key=lambda x: x[1])
    rmse_small = rows_sorted[0][2]
    rmse_mid = rows_sorted[min(1, len(rows_sorted) - 1)][2]
    rmse_large = rows_sorted[-1][2]
    max_abs_large = rows_sorted[-1][3]

    # If the smallest case is nearly exact (common for single-neuron),
    # use the next size as scale reference to avoid divide-by-near-zero effects.
    rmse_ref = rmse_mid if rmse_small < 0.25 else rmse_small

    # Loose but explicit guardrails for migration sanity.
    if rmse_large > (2.8 * rmse_ref + 2.0):
        raise RuntimeError(
            "Voltage RMSE appears to scale too aggressively with network size. "
            f"ref={rmse_ref:.4f}, large={rmse_large:.4f}."
        )
    if max_abs_large > 25.0:
        raise RuntimeError(
            "Max voltage mismatch is too large for migration sanity check. "
            f"max_abs_large={max_abs_large:.4f} mV."
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
        default=Path(__file__).resolve().parent / "output_native",
        help="Output directory for native-scheduler demo.",
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

    rows: list[tuple[str, int, float, float, float]] = []
    for case in build_cases():
        case_path = case_dir / f"{case.name}_in.h5"
        write_case_h5(case, case_path)

        cpp_out = run_cpp_simulator(simulator=simulator, case_path=case_path)
        v_cpp, spk_cpp = read_cpp_results(cpp_out, n_node=case.n_node)
        v_bt, spk_bt = run_btorch_native(case)

        t_ms = np.arange(case.step_tot, dtype=np.float64) * case.dt
        dv = v_cpp - v_bt
        dspk = spk_cpp - spk_bt

        mean_rmse = float(np.mean(np.sqrt(np.mean(np.square(dv), axis=1))))
        max_abs = float(np.max(np.abs(dv)))
        spike_l1 = float(np.mean(np.abs(dspk)))
        rows.append((case.name, case.n_node, mean_rmse, max_abs, spike_l1))

        plot_case(
            case=case,
            t_ms=t_ms,
            v_cpp=v_cpp,
            v_bt=v_bt,
            spk_cpp=spk_cpp,
            spk_bt=spk_bt,
            output_path=fig_dir / f"{case.name}.png",
        )

        with h5py.File(result_dir / f"{case.name}_native_comparison.h5", "w") as h5:
            h5.create_dataset("time_ms", data=t_ms)
            h5.create_dataset("v_cpp", data=v_cpp)
            h5.create_dataset("v_btorch_native", data=v_bt)
            h5.create_dataset("spike_cpp", data=spk_cpp)
            h5.create_dataset("spike_btorch_native", data=spk_bt)
            h5.create_dataset("dv", data=dv)
            h5.create_dataset("dspk", data=dspk)

    print("=== Native scheduler comparison completed ===")
    for name, n_node, rmse, max_abs, spike_l1 in rows:
        print(
            f"{name}: N={n_node}, mean_RMSE(V)={rmse:.6f} mV, "
            f"max_abs(V)={max_abs:.6f} mV, mean_|Δspike|={spike_l1:.6f}"
        )

    plot_error_summary(rows, output_path=fig_dir / "error_summary_native.png")
    ensure_migration_error_controlled(rows)

    print(f"Figures: {fig_dir}")
    print(f"Result tensors: {result_dir}")


if __name__ == "__main__":
    main()
