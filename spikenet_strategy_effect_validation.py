from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from spikenet_btorch_native_scheduler_demo import (
    CaseSpec,
    build_cases,
    read_cpp_results,
    run_btorch_native,
    run_cpp_simulator,
    write_case_h5,
)


@dataclasses.dataclass
class StimulusSample:
    name: str
    label: int
    i_ext_mean: list[float]


def pick_base_case(cases: list[CaseSpec], case_name: str) -> CaseSpec:
    for c in cases:
        if c.name == case_name:
            return c
    available = ", ".join(c.name for c in cases)
    raise ValueError(f"Case '{case_name}' not found. Available: {available}")


def build_stimulus_bank(
    base_case: CaseSpec,
    n_class: int,
    samples_per_class: int,
    jitter_std: float,
    rng: np.random.Generator,
) -> list[StimulusSample]:
    n = base_case.n_node
    if n < 4:
        raise ValueError("Use a base case with at least 4 neurons for multi-class stimuli")

    anchors = [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (8, 0),
        (9, 2),
    ]
    if n_class > len(anchors):
        raise ValueError(f"n_class={n_class} exceeds supported anchors={len(anchors)}")

    base = np.asarray(base_case.i_ext_mean, dtype=np.float64)
    samples: list[StimulusSample] = []
    for c in range(n_class):
        hot_a, hot_b = anchors[c]
        hot_a %= n
        hot_b %= n
        for k in range(samples_per_class):
            ext = base.copy()
            ext[hot_a] += 0.085
            ext[hot_b] += 0.065
            ext = ext + rng.normal(loc=0.0, scale=jitter_std, size=n)
            ext = np.clip(ext, -0.02, 0.90)
            samples.append(
                StimulusSample(
                    name=f"stim_c{c}_s{k}",
                    label=c,
                    i_ext_mean=ext.tolist(),
                )
            )
    return samples


def make_case(base_case: CaseSpec, sample: StimulusSample) -> CaseSpec:
    return dataclasses.replace(
        base_case,
        name=f"{base_case.name}_{sample.name}",
        i_ext_mean=sample.i_ext_mean,
    )


def best_lag_and_aligned_rmse(
    v_ref: np.ndarray,
    v_cmp: np.ndarray,
    max_lag_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, t = v_ref.shape
    best_lags = np.zeros((n,), dtype=np.int32)
    best_rmse = np.full((n,), np.inf, dtype=np.float64)

    for i in range(n):
        a = v_ref[i]
        b = v_cmp[i]
        for lag in range(-max_lag_step, max_lag_step + 1):
            if lag >= 0:
                a_seg = a[lag:]
                b_seg = b[: t - lag]
            else:
                a_seg = a[: t + lag]
                b_seg = b[-lag:]

            if a_seg.size < 8:
                continue

            rmse = float(np.sqrt(np.mean(np.square(a_seg - b_seg))))
            if rmse < best_rmse[i]:
                best_rmse[i] = rmse
                best_lags[i] = lag

    return best_lags, best_rmse


def spike_time_jitter_ms(
    spk_ref: np.ndarray,
    spk_cmp: np.ndarray,
    dt: float,
) -> np.ndarray:
    jitters = []
    n = spk_ref.shape[0]
    for i in range(n):
        t_ref = np.flatnonzero(spk_ref[i] > 0.5)
        t_cmp = np.flatnonzero(spk_cmp[i] > 0.5)
        if t_ref.size == 0 or t_cmp.size == 0:
            continue
        # nearest-neighbor timing mismatch in ms
        for tr in t_ref:
            dif = np.abs(t_cmp - tr)
            jitters.append(float(np.min(dif) * dt))
    if not jitters:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(jitters, dtype=np.float64)


def extract_features(v: np.ndarray, spk: np.ndarray) -> np.ndarray:
    spike_count = spk.sum(axis=1)
    mean_v = v.mean(axis=1)
    std_v = v.std(axis=1)
    late_v = v[:, int(0.7 * v.shape[1]) :].mean(axis=1)
    return np.concatenate([spike_count, mean_v, std_v, late_v], axis=0)


def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = []
    test_idx = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx.copy()
        rng.shuffle(idx)
        n_train = max(1, int(round(train_ratio * idx.size)))
        n_train = min(n_train, idx.size - 1)
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    return np.asarray(sorted(train_idx)), np.asarray(sorted(test_idx))


def normalize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma, mu, sigma


def train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_class: int,
    seed: int,
    epoch: int = 300,
    lr: float = 0.05,
) -> tuple[torch.nn.Module, list[float]]:
    torch.manual_seed(seed)

    model = torch.nn.Linear(x_train.shape[1], n_class, bias=True, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    xt = torch.as_tensor(x_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    hist = []
    for _ in range(epoch):
        logits = model(xt)
        loss = criterion(logits, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hist.append(float(loss.item()))

    return model, hist


def eval_probe(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float64))
    pred = logits.argmax(dim=1).cpu().numpy()
    acc = float(np.mean(pred == y))
    return acc, pred


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> np.ndarray:
    cm = np.zeros((n_class, n_class), dtype=np.int32)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = values.size
    boots = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(low=0, high=n, size=n)
        boots[i] = float(np.mean(values[idx]))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    basis = vt[:2].T
    return x0 @ basis


def render_figures(
    out_fig: Path,
    raw_rmse: np.ndarray,
    aligned_rmse: np.ndarray,
    best_lags: np.ndarray,
    jitter_ms: np.ndarray,
    x_cpp: np.ndarray,
    x_bt: np.ndarray,
    labels: np.ndarray,
    cm_dict: dict[str, np.ndarray],
    acc_dict: dict[str, float],
    seed_acc: dict[str, np.ndarray],
    curves: dict[str, list[float]],
    ci_gap_1: tuple[float, float],
    ci_gap_2: tuple[float, float],
    ci_in_domain: tuple[float, float],
) -> None:
    out_fig.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot([raw_rmse, aligned_rmse], tick_labels=["Raw RMSE", "Lag-aligned RMSE"])
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("Voltage Error Decomposition")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / "01_rmse_decomposition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.arange(best_lags.min() - 0.5, best_lags.max() + 1.5, 1.0)
    ax.hist(best_lags, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Best lag (step)")
    ax.set_ylabel("Count")
    ax.set_title("Best Temporal Lag Distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / "02_lag_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if jitter_ms.size > 0:
        ax.hist(jitter_ms, bins=30, color="tab:orange", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Nearest spike timing error (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Spike Timing Jitter")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig / "03_spike_jitter_histogram.png", dpi=160)
    plt.close(fig)

    x_all = np.concatenate([x_cpp, x_bt], axis=0)
    z = pca_2d(x_all)
    z_cpp = z[: x_cpp.shape[0]]
    z_bt = z[x_cpp.shape[0] :]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(
            z_cpp[m, 0],
            z_cpp[m, 1],
            s=38,
            marker="o",
            color=cmap(int(c)),
            alpha=0.72,
            label=f"Class {c} C++" if c < 4 else None,
        )
        ax.scatter(
            z_bt[m, 0],
            z_bt[m, 1],
            s=48,
            marker="x",
            color=cmap(int(c)),
            alpha=0.85,
            label=f"Class {c} btorch" if c < 4 else None,
        )
    ax.set_title("Feature Space Overlay (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[:8], labels_leg[:8], fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_fig / "04_feature_overlay_pca.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.4))
    keys = [
        ("cpp_train_cpp_test", "Train C++ -> Test C++"),
        ("cpp_train_bt_test", "Train C++ -> Test btorch"),
        ("bt_train_cpp_test", "Train btorch -> Test C++"),
        ("bt_train_bt_test", "Train btorch -> Test btorch"),
    ]
    for ax, (key, title) in zip(axes.flat, keys, strict=True):
        cm = cm_dict[key]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(f"{title}\nAcc={acc_dict[key]:.3f}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_fig / "05_confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = [
        "C++->C++",
        "C++->btorch",
        "btorch->C++",
        "btorch->btorch",
    ]
    vals = [
        float(seed_acc["cpp_train_cpp_test"].mean()),
        float(seed_acc["cpp_train_bt_test"].mean()),
        float(seed_acc["bt_train_cpp_test"].mean()),
        float(seed_acc["bt_train_bt_test"].mean()),
    ]
    yerr = [
        float(seed_acc["cpp_train_cpp_test"].std()),
        float(seed_acc["cpp_train_bt_test"].std()),
        float(seed_acc["bt_train_cpp_test"].std()),
        float(seed_acc["bt_train_bt_test"].std()),
    ]
    x = np.arange(len(vals))
    ax.bar(
        x,
        vals,
        yerr=yerr,
        capsize=4,
        color=["tab:blue", "tab:cyan", "tab:orange", "tab:red"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Readout Capability Transfer")
    ax.grid(axis="y", alpha=0.3)

    gap1 = vals[0] - vals[1]
    gap2 = vals[3] - vals[2]
    in_gap = vals[3] - vals[0]
    ax.text(0.02, 0.16, f"Gap(C++ train)={gap1:.3f}, 95%CI=[{ci_gap_1[0]:.3f},{ci_gap_1[1]:.3f}]", transform=ax.transAxes, fontsize=9)
    ax.text(0.02, 0.10, f"Gap(btorch train)={gap2:.3f}, 95%CI=[{ci_gap_2[0]:.3f},{ci_gap_2[1]:.3f}]", transform=ax.transAxes, fontsize=9)
    ax.text(0.02, 0.04, f"In-domain(bt-cpp)={in_gap:.3f}, 95%CI=[{ci_in_domain[0]:.3f},{ci_in_domain[1]:.3f}]", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "06_accuracy_transfer.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(curves["cpp"], lw=1.6, label="Train on C++")
    ax.plot(curves["bt"], lw=1.6, label="Train on btorch")
    ax.set_title("Linear Probe Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig / "07_training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strategy-induced mismatch does not harm capability")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output_native" / "strategy_validation")
    parser.add_argument("--base-case", type=str, default="native_case_c_ten_neurons_nontrivial")
    parser.add_argument("--n-class", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--jitter-std", type=float, default=0.008)
    parser.add_argument("--max-lag-ms", type=float, default=2.0)
    parser.add_argument("--equiv-eps", type=float, default=0.08)
    parser.add_argument("--n-probe-seed", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    root = args.workspace_root.resolve()
    simulator = root / "SpikeNet" / "simulator"
    if not simulator.exists():
        raise FileNotFoundError(f"SpikeNet simulator not found at {simulator}")

    out = args.output_dir
    case_dir = out / "cases"
    pair_dir = out / "pair_results"
    fig_dir = out / "figures"
    result_dir = out / "results"
    for d in (case_dir, pair_dir, fig_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    base_case = pick_base_case(build_cases(), args.base_case)
    samples = build_stimulus_bank(
        base_case=base_case,
        n_class=args.n_class,
        samples_per_class=args.samples_per_class,
        jitter_std=args.jitter_std,
        rng=rng,
    )

    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    n_sample = len(samples)

    x_cpp = []
    x_bt = []
    raw_rmse = []
    aligned_rmse = []
    lags_all = []
    jitter_all = []

    max_lag_step = int(round(args.max_lag_ms / base_case.dt))

    for i, sample in enumerate(samples):
        case = make_case(base_case, sample)
        pair_path = pair_dir / f"pair_{i:04d}.h5"

        if args.reuse and pair_path.exists():
            with h5py.File(pair_path, "r") as h5:
                v_cpp = np.asarray(h5["v_cpp"])
                spk_cpp = np.asarray(h5["spike_cpp"])
                v_bt = np.asarray(h5["v_btorch_native"])
                spk_bt = np.asarray(h5["spike_btorch_native"])
        else:
            case_path = case_dir / f"{case.name}_in.h5"
            write_case_h5(case, case_path)
            cpp_out = run_cpp_simulator(simulator=simulator, case_path=case_path)
            v_cpp, spk_cpp = read_cpp_results(cpp_out, n_node=case.n_node)
            v_bt, spk_bt = run_btorch_native(case)

            with h5py.File(pair_path, "w") as h5:
                h5.create_dataset("v_cpp", data=v_cpp)
                h5.create_dataset("spike_cpp", data=spk_cpp)
                h5.create_dataset("v_btorch_native", data=v_bt)
                h5.create_dataset("spike_btorch_native", data=spk_bt)
                h5.create_dataset("label", data=np.asarray([sample.label], dtype=np.int32))

        x_cpp.append(extract_features(v_cpp, spk_cpp))
        x_bt.append(extract_features(v_bt, spk_bt))

        dv = v_cpp - v_bt
        raw_rmse.append(float(np.sqrt(np.mean(np.square(dv)))))
        lag_step, rmse_n = best_lag_and_aligned_rmse(v_cpp, v_bt, max_lag_step=max_lag_step)
        aligned_rmse.append(float(np.mean(rmse_n)))
        lags_all.append(lag_step)
        jitter_all.append(spike_time_jitter_ms(spk_cpp, spk_bt, dt=base_case.dt))

    x_cpp = np.asarray(x_cpp, dtype=np.float64)
    x_bt = np.asarray(x_bt, dtype=np.float64)
    raw_rmse = np.asarray(raw_rmse, dtype=np.float64)
    aligned_rmse = np.asarray(aligned_rmse, dtype=np.float64)
    lags_all = np.concatenate(lags_all) if lags_all else np.zeros((0,), dtype=np.int32)
    jitter_all = np.concatenate(jitter_all) if jitter_all else np.zeros((0,), dtype=np.float64)

    split_rng = np.random.default_rng(args.seed + 1)
    train_idx, test_idx = stratified_split(labels, args.train_ratio, split_rng)
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    x_cpp_train, x_cpp_test = x_cpp[train_idx], x_cpp[test_idx]
    x_bt_train, x_bt_test = x_bt[train_idx], x_bt[test_idx]

    x_cpp_train_n, x_cpp_test_n, mu_cpp, sig_cpp = normalize_train_test(x_cpp_train, x_cpp_test)
    x_bt_train_on_cpp = (x_bt_train - mu_cpp) / sig_cpp
    x_bt_test_on_cpp = (x_bt_test - mu_cpp) / sig_cpp

    x_bt_train_n, x_bt_test_n, mu_bt, sig_bt = normalize_train_test(x_bt_train, x_bt_test)
    x_cpp_train_on_bt = (x_cpp_train - mu_bt) / sig_bt
    x_cpp_test_on_bt = (x_cpp_test - mu_bt) / sig_bt

    seed_acc = {
        "cpp_train_cpp_test": [],
        "cpp_train_bt_test": [],
        "bt_train_cpp_test": [],
        "bt_train_bt_test": [],
    }

    curve_cpp: list[float] | None = None
    curve_bt: list[float] | None = None
    pred_cpp_cpp = pred_cpp_bt = pred_bt_cpp = pred_bt_bt = None

    for s in range(args.n_probe_seed):
        probe_cpp, cur_cpp = train_linear_probe(
            x_train=x_cpp_train_n,
            y_train=y_train,
            n_class=args.n_class,
            seed=args.seed + 100 + s,
        )
        probe_bt, cur_bt = train_linear_probe(
            x_train=x_bt_train_n,
            y_train=y_train,
            n_class=args.n_class,
            seed=args.seed + 500 + s,
        )

        acc_cpp_cpp, p_cpp_cpp = eval_probe(probe_cpp, x_cpp_test_n, y_test)
        acc_cpp_bt, p_cpp_bt = eval_probe(probe_cpp, x_bt_test_on_cpp, y_test)
        acc_bt_bt, p_bt_bt = eval_probe(probe_bt, x_bt_test_n, y_test)
        acc_bt_cpp, p_bt_cpp = eval_probe(probe_bt, x_cpp_test_on_bt, y_test)

        seed_acc["cpp_train_cpp_test"].append(acc_cpp_cpp)
        seed_acc["cpp_train_bt_test"].append(acc_cpp_bt)
        seed_acc["bt_train_cpp_test"].append(acc_bt_cpp)
        seed_acc["bt_train_bt_test"].append(acc_bt_bt)

        if s == 0:
            curve_cpp = cur_cpp
            curve_bt = cur_bt
            pred_cpp_cpp = p_cpp_cpp
            pred_cpp_bt = p_cpp_bt
            pred_bt_bt = p_bt_bt
            pred_bt_cpp = p_bt_cpp

    seed_acc = {k: np.asarray(v, dtype=np.float64) for k, v in seed_acc.items()}

    assert curve_cpp is not None and curve_bt is not None
    assert pred_cpp_cpp is not None and pred_cpp_bt is not None
    assert pred_bt_cpp is not None and pred_bt_bt is not None

    acc_dict = {
        "cpp_train_cpp_test": float(seed_acc["cpp_train_cpp_test"].mean()),
        "cpp_train_bt_test": float(seed_acc["cpp_train_bt_test"].mean()),
        "bt_train_cpp_test": float(seed_acc["bt_train_cpp_test"].mean()),
        "bt_train_bt_test": float(seed_acc["bt_train_bt_test"].mean()),
    }

    cm_dict = {
        "cpp_train_cpp_test": confusion(y_test, pred_cpp_cpp, args.n_class),
        "cpp_train_bt_test": confusion(y_test, pred_cpp_bt, args.n_class),
        "bt_train_cpp_test": confusion(y_test, pred_bt_cpp, args.n_class),
        "bt_train_bt_test": confusion(y_test, pred_bt_bt, args.n_class),
    }

    # Pairwise per-sample performance gaps on shared test labels (seed-0 model).
    gap_cpp = (pred_cpp_cpp == y_test).astype(np.float64) - (pred_cpp_bt == y_test).astype(np.float64)
    gap_bt = (pred_bt_bt == y_test).astype(np.float64) - (pred_bt_cpp == y_test).astype(np.float64)

    ci_rng = np.random.default_rng(args.seed + 4)
    ci_gap_cpp = bootstrap_ci(gap_cpp, n_boot=2000, alpha=0.05, rng=ci_rng)
    ci_gap_bt = bootstrap_ci(gap_bt, n_boot=2000, alpha=0.05, rng=ci_rng)
    ci_in_domain = bootstrap_ci(
        seed_acc["bt_train_bt_test"] - seed_acc["cpp_train_cpp_test"],
        n_boot=2000,
        alpha=0.05,
        rng=np.random.default_rng(args.seed + 40),
    )

    render_figures(
        out_fig=fig_dir,
        raw_rmse=raw_rmse,
        aligned_rmse=aligned_rmse,
        best_lags=lags_all,
        jitter_ms=jitter_all,
        x_cpp=x_cpp,
        x_bt=x_bt,
        labels=labels,
        cm_dict=cm_dict,
        acc_dict=acc_dict,
        seed_acc=seed_acc,
        curves={"cpp": curve_cpp, "bt": curve_bt},
        ci_gap_1=ci_gap_cpp,
        ci_gap_2=ci_gap_bt,
        ci_in_domain=ci_in_domain,
    )

    summary = {
        "config": {
            "base_case": args.base_case,
            "n_class": args.n_class,
            "samples_per_class": args.samples_per_class,
            "train_ratio": args.train_ratio,
            "jitter_std": args.jitter_std,
            "max_lag_ms": args.max_lag_ms,
            "equiv_eps": args.equiv_eps,
            "seed": args.seed,
        },
        "error_decomposition": {
            "raw_rmse_mean": float(raw_rmse.mean()),
            "raw_rmse_std": float(raw_rmse.std()),
            "aligned_rmse_mean": float(aligned_rmse.mean()),
            "aligned_rmse_std": float(aligned_rmse.std()),
            "aligned_improvement_ratio": float(
                1.0 - aligned_rmse.mean() / max(raw_rmse.mean(), 1e-12)
            ),
            "best_lag_step_mean": float(lags_all.mean()) if lags_all.size else 0.0,
            "best_lag_step_std": float(lags_all.std()) if lags_all.size else 0.0,
            "spike_jitter_ms_median": float(np.median(jitter_all)) if jitter_all.size else 0.0,
            "spike_jitter_ms_p90": float(np.quantile(jitter_all, 0.9)) if jitter_all.size else 0.0,
        },
        "capability_transfer": {
            **acc_dict,
            "std_cpp_train_cpp_test": float(seed_acc["cpp_train_cpp_test"].std()),
            "std_cpp_train_bt_test": float(seed_acc["cpp_train_bt_test"].std()),
            "std_bt_train_cpp_test": float(seed_acc["bt_train_cpp_test"].std()),
            "std_bt_train_bt_test": float(seed_acc["bt_train_bt_test"].std()),
            "gap_cpp_train_in_minus_cross": float(acc_dict["cpp_train_cpp_test"] - acc_dict["cpp_train_bt_test"]),
            "gap_bt_train_in_minus_cross": float(acc_dict["bt_train_bt_test"] - acc_dict["bt_train_cpp_test"]),
            "gap_cpp_train_ci95": [ci_gap_cpp[0], ci_gap_cpp[1]],
            "gap_bt_train_ci95": [ci_gap_bt[0], ci_gap_bt[1]],
            "in_domain_gap_bt_minus_cpp": float(
                acc_dict["bt_train_bt_test"] - acc_dict["cpp_train_cpp_test"]
            ),
            "in_domain_gap_ci95": [ci_in_domain[0], ci_in_domain[1]],
            "equivalent_cpp_train": bool(
                ci_gap_cpp[0] > -args.equiv_eps and ci_gap_cpp[1] < args.equiv_eps
            ),
            "equivalent_bt_train": bool(
                ci_gap_bt[0] > -args.equiv_eps and ci_gap_bt[1] < args.equiv_eps
            ),
            "equivalent_in_domain": bool(
                ci_in_domain[0] > -args.equiv_eps and ci_in_domain[1] < args.equiv_eps
            ),
            "n_probe_seed": int(args.n_probe_seed),
        },
        "paths": {
            "figures": str(fig_dir),
            "pair_results": str(pair_dir),
        },
        "n_samples": int(n_sample),
        "n_test": int(y_test.size),
    }

    with (result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Strategy effect validation completed ===")
    print(json.dumps(summary, indent=2))
    print(f"Figures: {fig_dir}")
    print(f"Summary: {result_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
