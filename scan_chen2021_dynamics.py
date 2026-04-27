from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "chen_gong_2021_static_double_peak.py"


@dataclass(frozen=True)
class SweepCase:
    name: str
    factor: str
    ext_scale: float
    noise_cv: float
    inh_scale: float
    tau_scale: float


def _build_cases(
    ext_levels: list[float],
    noise_levels: list[float],
    inh_levels: list[float],
    tau_levels: list[float],
) -> list[SweepCase]:
    cases: list[SweepCase] = [
        SweepCase(
            name="baseline",
            factor="baseline",
            ext_scale=1.0,
            noise_cv=0.0,
            inh_scale=1.0,
            tau_scale=1.0,
        )
    ]

    for v in ext_levels:
        if abs(v - 1.0) < 1e-12:
            continue
        cases.append(
            SweepCase(
                name=f"ext_{v:.2f}".replace(".", "p"),
                factor="ext",
                ext_scale=v,
                noise_cv=0.0,
                inh_scale=1.0,
                tau_scale=1.0,
            )
        )

    for v in noise_levels:
        if abs(v) < 1e-12:
            continue
        cases.append(
            SweepCase(
                name=f"noise_{v:.2f}".replace(".", "p"),
                factor="noise",
                ext_scale=1.0,
                noise_cv=v,
                inh_scale=1.0,
                tau_scale=1.0,
            )
        )

    for v in inh_levels:
        if abs(v - 1.0) < 1e-12:
            continue
        cases.append(
            SweepCase(
                name=f"inh_{v:.2f}".replace(".", "p"),
                factor="inh",
                ext_scale=1.0,
                noise_cv=0.0,
                inh_scale=v,
                tau_scale=1.0,
            )
        )

    for v in tau_levels:
        if abs(v - 1.0) < 1e-12:
            continue
        cases.append(
            SweepCase(
                name=f"tau_{v:.2f}".replace(".", "p"),
                factor="tau",
                ext_scale=1.0,
                noise_cv=0.0,
                inh_scale=1.0,
                tau_scale=v,
            )
        )

    return cases


def _parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _as_float(d: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except ValueError:
        return default


def _as_int(d: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(d.get(key, default)))
    except ValueError:
        return default


def _score_case(summary: dict[str, str]) -> float:
    levy = 1.0 if summary.get("levy_like_transition", "no").lower() == "yes" else 0.0
    switches = _as_int(summary, "winner_switches", 0)
    center = _as_float(summary, "center_win_fraction", 0.0)
    corner = _as_float(summary, "corner_win_fraction", 0.0)
    balance = 1.0 - abs(center - corner)
    return levy * 100.0 + min(switches, 30) * 2.0 + balance


def _run_one(
    case: SweepCase,
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, str]:
    out_dir = out_root / case.name
    log_dir = out_root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = shlex.split(args.python_exec) + [
        str(RUNNER),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--T-ms",
        str(args.T_ms),
        "--stim-start-ms",
        str(args.stim_start_ms),
        "--ext-scale",
        str(case.ext_scale),
        "--noise-cv",
        str(case.noise_cv),
        "--inh-scale",
        str(case.inh_scale),
        "--tau-scale",
        str(case.tau_scale),
        "--output-dir",
        str(out_dir),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    log_path = log_dir / f"{case.name}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout)
        f.write("\nSTDERR:\n")
        f.write(proc.stderr)

    summary_path = out_dir / "infer_summary.txt"
    summary = _parse_summary(summary_path)

    row: dict[str, str] = {
        "name": case.name,
        "factor": case.factor,
        "ext_scale": f"{case.ext_scale:.6f}",
        "noise_cv": f"{case.noise_cv:.6f}",
        "inh_scale": f"{case.inh_scale:.6f}",
        "tau_scale": f"{case.tau_scale:.6f}",
        "returncode": str(proc.returncode),
        "elapsed_s": f"{elapsed:.3f}",
        "log": str(log_path),
        "summary": str(summary_path),
        "levy_like_transition": summary.get("levy_like_transition", "missing"),
        "winner_switches": summary.get("winner_switches", ""),
        "center_win_fraction": summary.get("center_win_fraction", ""),
        "corner_win_fraction": summary.get("corner_win_fraction", ""),
        "first_center_win_ms": summary.get("first_center_win_ms", ""),
        "first_corner_win_ms": summary.get("first_corner_win_ms", ""),
        "mean_rate_hz": summary.get("mean_rate_hz", ""),
        "total_spikes": summary.get("total_spikes", ""),
    }
    return row


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _pick_minimal_set(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {r["name"]: r for r in rows}
    selected: list[dict[str, str]] = []
    if "baseline" in by_name:
        selected.append(by_name["baseline"])

    for factor in ("ext", "noise", "inh", "tau"):
        cand = [r for r in rows if r["factor"] == factor and r["returncode"] == "0"]
        if not cand:
            continue
        best = max(cand, key=lambda r: _score_case({k: v for k, v in r.items()}))
        selected.append(best)

    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in selected:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        unique.append(r)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-job dynamics sweep for Chen&Gong 2021 double-peak test"
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable or launcher command used to run each case.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T-ms", type=float, default=1200.0)
    parser.add_argument("--stim-start-ms", type=float, default=200.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(SCRIPT_DIR / "static_gaussian" / "scan_chen2021_dynamics"),
    )
    parser.add_argument(
        "--ext-levels",
        type=float,
        nargs="+",
        default=[0.85, 1.0, 1.15],
        help="Levels for external-input scaling.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.20],
        help="Levels for lambda jitter CV.",
    )
    parser.add_argument(
        "--inh-levels",
        type=float,
        nargs="+",
        default=[0.85, 1.0, 1.15],
        help="Levels for inhibitory-conductance scaling.",
    )
    parser.add_argument(
        "--tau-levels",
        type=float,
        nargs="+",
        default=[0.85, 1.0, 1.15],
        help="Levels for synaptic-time-constant scaling.",
    )
    parser.add_argument(
        "--run-full-on-minimal",
        action="store_true",
        help="Run full-length evaluation on the selected minimal set.",
    )
    parser.add_argument("--full-T-ms", type=float, default=10000.0)
    parser.add_argument("--full-stim-start-ms", type=float, default=4000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(args.ext_levels, args.noise_levels, args.inh_levels, args.tau_levels)
    print(f"[scan] total cases: {len(cases)}")
    print(f"[scan] output root: {out_root}")
    print(f"[scan] workers: {args.workers}")

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_to_case = {ex.submit(_run_one, c, args, out_root): c for c in cases}
        for fut in as_completed(fut_to_case):
            c = fut_to_case[fut]
            try:
                row = fut.result()
            except Exception as e:  # pragma: no cover
                row = {
                    "name": c.name,
                    "factor": c.factor,
                    "ext_scale": f"{c.ext_scale:.6f}",
                    "noise_cv": f"{c.noise_cv:.6f}",
                    "inh_scale": f"{c.inh_scale:.6f}",
                    "tau_scale": f"{c.tau_scale:.6f}",
                    "returncode": "-1",
                    "elapsed_s": "0.000",
                    "log": "",
                    "summary": "",
                    "levy_like_transition": f"error: {e}",
                    "winner_switches": "",
                    "center_win_fraction": "",
                    "corner_win_fraction": "",
                    "first_center_win_ms": "",
                    "first_corner_win_ms": "",
                    "mean_rate_hz": "",
                    "total_spikes": "",
                }
            rows.append(row)
            print(
                f"[done] {c.name:12s} rc={row['returncode']} "
                f"switches={row.get('winner_switches', '')} "
                f"levy={row.get('levy_like_transition', '')}"
            )

    rows.sort(key=lambda r: r["name"])
    summary_csv = out_root / "scan_summary.csv"
    _write_csv(rows, summary_csv)

    minimal_rows = _pick_minimal_set(rows)
    minimal_csv = out_root / "minimal_repro_set.csv"
    _write_csv(minimal_rows, minimal_csv)

    print(f"[scan] summary csv: {summary_csv}")
    print(f"[scan] minimal set csv: {minimal_csv}")

    if args.run_full_on_minimal and minimal_rows:
        print("[scan] running full-length evaluation on minimal set...")
        full_rows: list[dict[str, str]] = []
        for r in minimal_rows:
            case = SweepCase(
                name=r["name"] + "_full",
                factor=r["factor"],
                ext_scale=float(r["ext_scale"]),
                noise_cv=float(r["noise_cv"]),
                inh_scale=float(r["inh_scale"]),
                tau_scale=float(r["tau_scale"]),
            )
            full_args = argparse.Namespace(**vars(args))
            full_args.T_ms = args.full_T_ms
            full_args.stim_start_ms = args.full_stim_start_ms
            full_row = _run_one(case, full_args, out_root / "full")
            full_rows.append(full_row)
            print(
                f"[full] {case.name:16s} rc={full_row['returncode']} "
                f"switches={full_row.get('winner_switches', '')} "
                f"levy={full_row.get('levy_like_transition', '')}"
            )

        full_csv = out_root / "full" / "minimal_full_summary.csv"
        full_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(full_rows, full_csv)
        print(f"[scan] minimal full summary: {full_csv}")


if __name__ == "__main__":
    main()
