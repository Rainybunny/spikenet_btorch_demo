"""
make_winner_video.py  --  Visualize winner-switching dynamics from activation_metadata.npz

Layout
------
Top   : 63×63 per-neuron firing-rate heatmap (brightness = Hz).
        A dashed circle marks the 2σ region of the active winner peak in
        real time; torus wrap-around is handled for the corner peak.
Bottom: Centre vs Corner smoothed rate traces with a moving time cursor
        and winner-background shading.

Usage
-----
  conda run -n spike_gpu python make_winner_video.py \\
      [--meta PATH] [--output PATH] [--fps N] [--speed X] [--smooth-ms N]
      [--t-start S] [--t-end S] [--dpi N] [--vmax HZ]

Defaults target the inh_0p50 result with 3× slowdown and 15 fps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def build_rate_frames(
    spike_t: np.ndarray,     # sorted int64 spike step indices
    spike_n: np.ndarray,     # int32 neuron indices
    n_e: int,
    n_side: int,
    dt_ms: float,
    frame_centers: np.ndarray,  # int64 step indices of frame centres
    half_win: int,               # half-window in steps
) -> np.ndarray:
    """Return float32 (n_frames, n_side, n_side) firing-rate grids in Hz."""
    n_steps_total = int(spike_t[-1]) + 1 if len(spike_t) else 1
    duration_s = 2 * half_win * dt_ms * 1e-3

    n_frames = len(frame_centers)
    frames = np.zeros((n_frames, n_side, n_side), dtype=np.float32)

    for i, fc in enumerate(frame_centers):
        lo_step = int(fc) - half_win
        hi_step = int(fc) + half_win
        lo = int(np.searchsorted(spike_t, lo_step, side="left"))
        hi = int(np.searchsorted(spike_t, hi_step, side="right"))
        counts = np.zeros(n_e, dtype=np.float32)
        if hi > lo:
            np.add.at(counts, spike_n[lo:hi], 1)
        frames[i] = (counts / max(duration_s, 1e-9)).reshape(n_side, n_side)

    return frames


def frame_dominant_winner(winner: np.ndarray, fc: int, half_win: int) -> int:
    """Return majority winner label in [fc-half_win, fc+half_win]. -1 = tie."""
    n = len(winner)
    lo = max(0, fc - half_win)
    hi = min(n, fc + half_win)
    w = winner[lo:hi]
    active = w[w >= 0]
    if active.size == 0:
        return -1
    counts = np.bincount(active.astype(np.uint8), minlength=2)
    if counts[0] == counts[1]:
        return -1
    return int(np.argmax(counts))


def corner_circle_centres(corner_pos: np.ndarray, r: float) -> list[tuple[float, float]]:
    """All torus copies of corner_pos that intersect the [-π,π]² viewport."""
    y0, x0 = float(corner_pos[0]), float(corner_pos[1])
    centres = []
    for dy in (-2 * np.pi, 0.0, 2 * np.pi):
        for dx in (-2 * np.pi, 0.0, 2 * np.pi):
            cy, cx = y0 + dy, x0 + dx
            # keep only copies whose circle can intersect [-π,π]²
            if -np.pi - r <= cx <= np.pi + r and -np.pi - r <= cy <= np.pi + r:
                centres.append((cx, cy))
    return centres


# ── main ──────────────────────────────────────────────────────────────────────

def make_video(args: argparse.Namespace) -> None:
    # ── load metadata ─────────────────────────────────────────────────────────
    meta      = np.load(args.meta)
    spike_t   = meta["spike_t"].astype(np.int64)
    spike_n   = meta["spike_n"].astype(np.int32)
    coords_e  = meta["coords_e"]          # (n_e, 2) y,x in rad
    dt_ms     = float(meta["dt_ms"].flat[0])
    stim_on   = int(meta["stim_on_step"].flat[0])
    center_pos = meta["center_pos"]       # (2,) float32
    corner_pos = meta["corner_pos"]       # (2,) float32
    time_s    = meta["time_s"]
    center_hz = meta["center_hz"]
    corner_hz = meta["corner_hz"]
    winner    = meta["winner"].astype(np.int8)

    n_steps = len(time_s)
    n_e     = len(coords_e)
    n_side  = int(round(n_e ** 0.5))     # 63 for hw=31
    dt_s    = dt_ms * 1e-3

    # sort spikes by time once for fast searchsorted
    order  = np.argsort(spike_t, kind="stable")
    spike_t = spike_t[order]
    spike_n = spike_n[order]

    # ── frame parameters ──────────────────────────────────────────────────────
    frame_dt_s    = args.speed / args.fps          # sim-seconds per frame
    frame_dt_steps = max(1, int(round(frame_dt_s / dt_s)))

    smooth_half = max(1, int(round(args.smooth_ms * 1e-3 / (2 * dt_s))))

    step_start = max(0, int(args.t_start / dt_s))
    step_end   = min(n_steps - 1, int(args.t_end / dt_s))

    frame_centers = np.arange(step_start, step_end, frame_dt_steps, dtype=np.int64)
    n_frames = len(frame_centers)

    print(f"Sim window : {args.t_start:.1f} – {args.t_end:.1f} s")
    print(f"Speed      : {args.speed}× ({n_frames} frames at {args.fps} fps → "
          f"{n_frames / args.fps:.1f} s video)")
    print(f"Smooth win : {args.smooth_ms:.0f} ms ({2*smooth_half} steps)")

    # ── precompute rate grids ─────────────────────────────────────────────────
    print("Precomputing per-neuron rate grids …", flush=True)
    rate_grids = build_rate_frames(
        spike_t, spike_n, n_e, n_side, dt_ms, frame_centers, smooth_half
    )

    if args.vmax is not None:
        vmax = args.vmax
    else:
        vmax = float(np.percentile(rate_grids[rate_grids > 0], 99)) * 1.1
        vmax = max(vmax, 5.0)
    print(f"Colour scale: 0 – {vmax:.1f} Hz")

    # ── winner label per frame ────────────────────────────────────────────────
    frame_winners = np.array(
        [frame_dominant_winner(winner, int(fc), smooth_half) for fc in frame_centers],
        dtype=np.int8,
    )

    sigma_rad   = 0.6          # qisig used in network construction
    r_2sig      = 2.0 * sigma_rad

    COL_CENTER = "#FF5555"
    COL_CORNER = "#55AAFF"
    COL_TIE    = "#888888"
    BG         = "#111111"

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8), facecolor=BG)
    gs  = fig.add_gridspec(
        2, 2, height_ratios=[3, 1.2],
        width_ratios=[20, 1],
        hspace=0.12, wspace=0.04,
        left=0.08, right=0.96, top=0.94, bottom=0.08,
    )
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_cb   = fig.add_subplot(gs[0, 1])   # colorbar axis
    ax_rate = fig.add_subplot(gs[1, 0])

    # ── heatmap ───────────────────────────────────────────────────────────────
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    im = ax_map.imshow(
        rate_grids[0],
        origin="lower", extent=extent,
        cmap="hot", vmin=0, vmax=vmax,
        interpolation="bilinear", aspect="equal",
    )
    ax_map.set_facecolor(BG)
    ax_map.set_xlim(-np.pi, np.pi)
    ax_map.set_ylim(-np.pi, np.pi)
    ax_map.set_xlabel("x  (rad)", color="white", fontsize=10)
    ax_map.set_ylabel("y  (rad)", color="white", fontsize=10)
    ax_map.tick_params(colors="white", labelsize=8)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#555555")

    # colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Firing rate  (Hz)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.ax.set_facecolor(BG)

    # ── 2σ circle patches ─────────────────────────────────────────────────────
    # Center peak: one circle at (0,0)
    circ_center = mpatches.Circle(
        (float(center_pos[1]), float(center_pos[0])),
        r_2sig, fill=False, linewidth=2.2, linestyle="--",
        edgecolor=COL_CENTER, alpha=0, zorder=5,
    )
    ax_map.add_patch(circ_center)

    # Corner peak: draw at every torus copy that overlaps the viewport
    corner_centres = corner_circle_centres(corner_pos, r_2sig)
    circ_corner_list: list[mpatches.Circle] = []
    for (cx, cy) in corner_centres:
        c = mpatches.Circle(
            (cx, cy), r_2sig, fill=False, linewidth=2.2, linestyle="--",
            edgecolor=COL_CORNER, alpha=0, zorder=5,
        )
        ax_map.add_patch(c)
        circ_corner_list.append(c)

    # filled marker dots at peak centres
    dot_center = ax_map.plot(
        float(center_pos[1]), float(center_pos[0]),
        "o", ms=6, color=COL_CENTER, alpha=0, zorder=6,
    )[0]
    dot_corner = ax_map.plot(
        float(corner_pos[1]), float(corner_pos[0]),
        "o", ms=6, color=COL_CORNER, alpha=0, zorder=6,
    )[0]

    # title with time stamp and winner label
    title_obj = ax_map.set_title(
        "", color="white", fontsize=11, pad=5, loc="left",
    )
    winner_label = ax_map.text(
        0.98, 1.02, "", transform=ax_map.transAxes,
        ha="right", va="bottom", fontsize=12, fontweight="bold",
        color="white",
    )

    # ── rate panel ────────────────────────────────────────────────────────────
    ax_rate.set_facecolor(BG)
    ax_rate.tick_params(colors="white", labelsize=8)
    for sp in ax_rate.spines.values():
        sp.set_edgecolor("#555555")

    t_slice = time_s[step_start:step_end]
    c_slice = center_hz[step_start:step_end]
    k_slice = corner_hz[step_start:step_end]

    # winner background shading (per step)
    # fill in large chunks for efficiency
    w_slice = winner[step_start:step_end]
    for label, col in ((0, COL_CENTER), (1, COL_CORNER)):
        idx = np.flatnonzero(w_slice == label)
        if idx.size == 0:
            continue
        # find contiguous runs
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        ends   = np.concatenate([idx[breaks], [idx[-1]]])
        for s, e in zip(starts, ends):
            ax_rate.axvspan(t_slice[s], t_slice[min(e + 1, len(t_slice) - 1)],
                            color=col, alpha=0.12, lw=0, zorder=0)

    # stim-on marker
    stim_t = stim_on * dt_s
    if args.t_start < stim_t < args.t_end:
        ax_rate.axvline(stim_t, color="#FFFF88", linewidth=1, linestyle=":", alpha=0.6)
        ax_rate.text(stim_t + 0.05, 0.97, "stim on", transform=ax_rate.get_xaxis_transform(),
                     color="#FFFF88", fontsize=7, va="top")

    # full traces (dim background)
    ax_rate.plot(t_slice, c_slice, color=COL_CENTER, alpha=0.30, linewidth=0.8, zorder=1)
    ax_rate.plot(t_slice, k_slice, color=COL_CORNER, alpha=0.30, linewidth=0.8, zorder=1)

    # bright moving window (last N frames worth of trace)
    trail_steps = max(1, int(0.5 / dt_s))   # 0.5 s trail
    line_c, = ax_rate.plot([], [], color=COL_CENTER, linewidth=1.8, alpha=0.95, zorder=2)
    line_k, = ax_rate.plot([], [], color=COL_CORNER, linewidth=1.8, alpha=0.95, zorder=2)

    vline = ax_rate.axvline(x=t_slice[0], color="white", linewidth=1.2,
                            alpha=0.85, zorder=4)

    ymax_r = max(float(c_slice.max()), float(k_slice.max())) * 1.08
    ax_rate.set_xlim(args.t_start, args.t_end)
    ax_rate.set_ylim(0, ymax_r)
    ax_rate.set_xlabel("Time  (s)", color="white", fontsize=10)
    ax_rate.set_ylabel("Rate  (Hz)", color="white", fontsize=10)

    ax_rate.plot([], [], color=COL_CENTER, linewidth=2, label="Centre")
    ax_rate.plot([], [], color=COL_CORNER, linewidth=2, label="Corner")
    ax_rate.legend(loc="upper right", facecolor=BG, edgecolor="#555555",
                   labelcolor="white", fontsize=8, framealpha=0.85)

    # count label
    n_switches_total = int((np.diff(winner[stim_on:].astype(np.int16)) != 0).sum())
    fig.text(0.50, 0.965, "Chen & Gong 2021  —  inh_scale=0.50  —  winner switching dynamics",
             ha="center", va="top", color="white", fontsize=10)

    # ── update ────────────────────────────────────────────────────────────────
    all_artists = [im, circ_center, dot_center, dot_corner,
                   title_obj, winner_label, vline, line_c, line_k,
                   *circ_corner_list]

    def update(fi: int):
        fc    = int(frame_centers[fi])
        t_now = float(time_s[fc])
        w     = int(frame_winners[fi])

        # --- heatmap ---
        im.set_data(rate_grids[fi])

        # --- circles ---
        if w == 0:   # centre wins
            circ_center.set_alpha(0.90)
            dot_center.set_alpha(0.95)
            for c in circ_corner_list:
                c.set_alpha(0)
            dot_corner.set_alpha(0)
            wtext = "■ CENTRE"
            wcol  = COL_CENTER
        elif w == 1:  # corner wins
            circ_center.set_alpha(0)
            dot_center.set_alpha(0)
            for c in circ_corner_list:
                c.set_alpha(0.90)
            dot_corner.set_alpha(0.95)
            wtext = "■ CORNER"
            wcol  = COL_CORNER
        else:         # tie
            circ_center.set_alpha(0.30)
            dot_center.set_alpha(0.50)
            for c in circ_corner_list:
                c.set_alpha(0.30)
            dot_corner.set_alpha(0.50)
            wtext = "—"
            wcol  = COL_TIE

        # --- title ---
        stim_flag = "  [stim ON]" if fc >= stim_on else ""
        title_obj.set_text(f"t = {t_now:.3f} s{stim_flag}")
        winner_label.set_text(wtext)
        winner_label.set_color(wcol)

        # --- trailing bright rate lines ---
        lo_step = max(step_start, fc - trail_steps)
        lo_i    = lo_step - step_start
        hi_i    = fc - step_start
        line_c.set_data(t_slice[lo_i:hi_i], c_slice[lo_i:hi_i])
        line_k.set_data(t_slice[lo_i:hi_i], k_slice[lo_i:hi_i])

        # --- cursor ---
        vline.set_xdata([t_now, t_now])

        return all_artists

    # Render frames to numpy arrays, then encode with imageio-ffmpeg
    # (avoids dependency on a working system ffmpeg binary).
    import imageio_ffmpeg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()

    # Read actual rendered pixel dimensions from the canvas buffer.
    buf0 = fig.canvas.buffer_rgba()
    renderer = fig.canvas.get_renderer()
    actual_w = int(renderer.width)
    actual_h = int(renderer.height)

    # Align to macro-block size 16 required by libx264.
    enc_w = (actual_w // 16) * 16
    enc_h = (actual_h // 16) * 16

    print(f"Encoding → {out_path}  [{enc_w}×{enc_h}]", flush=True)

    gen = imageio_ffmpeg.write_frames(
        str(out_path),
        size=(enc_w, enc_h),
        fps=args.fps,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20"],
    )
    gen.send(None)

    for fi in range(n_frames):
        update(fi)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_np = np.frombuffer(buf, dtype=np.uint8).reshape(actual_h, actual_w, 4)
        # Crop to aligned size (bottom-right crop avoids edge artifacts).
        frame_rgb = frame_np[:enc_h, :enc_w, :3]
        gen.send(frame_rgb.tobytes())
        print(f"  frame {fi+1}/{n_frames}", end="\r", flush=True)

    gen.close()
    print()
    plt.close(fig)
    print(f"Done  →  {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--meta", type=str,
        default="static_gaussian/scan_model0_sweep2/inh_0p50/activation_metadata.npz",
        help="Path to activation_metadata.npz",
    )
    p.add_argument(
        "--output", type=str, default="inh_0p50_winner_video.mp4",
        help="Output video path",
    )
    p.add_argument(
        "--fps", type=int, default=15,
        help="Video frame rate (default 15)",
    )
    p.add_argument(
        "--speed", type=float, default=0.3,
        help="Simulated seconds per video second (default 0.3 = 3× slowdown)",
    )
    p.add_argument(
        "--smooth-ms", type=float, default=200.0,
        help="Sliding window for per-neuron rate estimation (ms, default 200)",
    )
    p.add_argument(
        "--t-start", type=float, default=3.5,
        help="Sim start time to render (s, default 3.5)",
    )
    p.add_argument(
        "--t-end", type=float, default=10.0,
        help="Sim end time to render (s, default 10.0)",
    )
    p.add_argument(
        "--dpi", type=int, default=120,
        help="Output DPI (default 120 → ~1080×960 px)",
    )
    p.add_argument(
        "--vmax", type=float, default=None,
        help="Heatmap colour scale max (Hz). Auto if omitted.",
    )
    return p.parse_args()


if __name__ == "__main__":
    make_video(parse_args())
