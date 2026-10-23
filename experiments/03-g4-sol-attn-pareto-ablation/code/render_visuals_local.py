#!/usr/bin/env python3
"""
Local Zero-CU Post-Processing & Visual Artifact Renderer for Experiment 03.
Runs on local Mac (0.00 CU) after pulling /content/exp03_g4_outputs/ from the G4 instance.

Generates:
  1. outputs/pareto_frontier_cost_speed_quality.png
  2. outputs/sol_attn_mechanisms_and_lifecycle_cu.png
  3. outputs/frame_metrics_curves_12runs.png
  4. outputs/keyframe_strip_12runs.png
  5. outputs/comparison_grid_12runs.mp4
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "code" else _SCRIPT_DIR
OUTPUT_DIR = EXP_DIR / "outputs"


FAMILY_COLORS: Dict[str, str] = {
    "Dense Baseline": "#1f77b4",
    "Dense Accelerated": "#ff7f0e",
    "Dense + Operator Fusion": "#d62728",
    "Native CK Sol-Attn (4K Chunked)": "#9467bd",
    "Native CK Sol-Attn (Direct Zero-Copy)": "#2ca02c",
    "Community Triton Sol-Attn": "#17becf",
    "Full-Stack Hybrid": "#e377c2",
}


def load_frames_from_mp4(mp4_path: Path, max_frames: int = 124) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames from {mp4_path}")
    return np.stack(frames, axis=0)


def render_pareto_frontier_chart(data: Dict[str, Any]) -> None:
    runs = data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.2), dpi=200)

    # Panel A: 8-Step DiT Denoise Latency (s) vs 4D Quality Score (Q_total)
    ax0 = axes[0]
    pareto_denoise = [r for r in runs if r.get("is_pareto_denoise_qtotal")]
    pareto_denoise.sort(key=lambda x: x["denoise_sec"])
    if len(pareto_denoise) >= 2:
        ax0.plot(
            [p["denoise_sec"] for p in pareto_denoise],
            [p["Q_total"] for p in pareto_denoise],
            linestyle="--", linewidth=2.5, color="#d62728", alpha=0.88,
            label="Denoise Pareto Frontier", zorder=3,
        )

    custom_offsets_a = {
        1: (-130, -26),
        2: (10, -26),
        3: (-55, 16),
        4: (-115, 16),
        5: (14, -22),
        6: (14, -30),
        7: (6, 18),
        8: (-125, 36),
        9: (12, 14),
        10: (12, -10),
        11: (14, -14),
        12: (-145, 6),
    }

    seen_families = set()
    for idx, r in enumerate(runs):
        fam = r["family"]
        c = FAMILY_COLORS.get(fam, "#333333")
        is_p = r.get("is_pareto_denoise_qtotal", False)
        lbl = fam if fam not in seen_families else None
        seen_families.add(fam)
        ax0.scatter(
            r["denoise_sec"], r["Q_total"],
            s=250 if is_p else 130,
            c=c,
            marker="*" if is_p else "o",
            edgecolors="#000000" if is_p else "#ffffff",
            linewidths=1.4 if is_p else 0.7,
            label=lbl,
            zorder=5 if is_p else 4,
        )
        dx, dy = custom_offsets_a.get(r["run_id"], (8, 8))
        ax0.annotate(
            f"R{r['run_id']:02d} {r['code']}\n({r['denoise_sec']:.1f}s, Q={r['Q_total']:.1f})",
            (r["denoise_sec"], r["Q_total"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.6,
            fontweight="bold" if is_p else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.88, lw=0.85),
            arrowprops=dict(arrowstyle="-", color=c, lw=0.75, alpha=0.75),
        )

    ax0.set_xlim(30, 83)
    ax0.set_ylim(64.5, 94.0)
    ax0.set_title("A. 8-Step DiT Denoise Latency vs. 4D Quality Score (Q_total)", fontsize=12.5, fontweight="bold")
    ax0.set_xlabel("8-Step DiT Denoising Latency on Blackwell G4 (seconds, lower is faster ←)", fontsize=11)
    ax0.set_ylabel("4D Composite Quality Score Q_total ∈ [0, 100] (higher is better ↑)", fontsize=11)
    ax0.grid(True, linestyle=":", alpha=0.55)
    ax0.legend(loc="lower right", fontsize=8.3, framealpha=0.92)

    # Panel B: Marginal Cost per Video (CU) vs Q_total (with S_fl2v annotated)
    ax1 = axes[1]
    pareto_e2e = [r for r in runs if r.get("is_pareto_e2e_qtotal")]
    pareto_e2e.sort(key=lambda x: x["cu_e2e"])
    if len(pareto_e2e) >= 2:
        ax1.plot(
            [p["cu_e2e"] for p in pareto_e2e],
            [p["Q_total"] for p in pareto_e2e],
            linestyle="--", linewidth=2.5, color="#2ca02c", alpha=0.88,
            label="Cost-Quality Pareto Frontier (CU vs Q_total)", zorder=3,
        )

    custom_offsets_b = {
        1: (-145, -26),
        2: (10, -26),
        3: (-65, 16),
        4: (-130, 16),
        5: (14, -22),
        6: (14, -30),
        7: (6, 18),
        8: (-135, 36),
        9: (12, 14),
        10: (12, -10),
        11: (14, -14),
        12: (-155, 6),
    }

    for idx, r in enumerate(runs):
        fam = r["family"]
        c = FAMILY_COLORS.get(fam, "#333333")
        is_p = r.get("is_pareto_e2e_qtotal", False)
        ax1.scatter(
            r["cu_e2e"], r["Q_total"],
            s=250 if is_p else 130,
            c=c,
            marker="*" if is_p else "o",
            edgecolors="#000000" if is_p else "#ffffff",
            linewidths=1.4 if is_p else 0.7,
            zorder=5 if is_p else 4,
        )
        dx, dy = custom_offsets_b.get(r["run_id"], (8, 8))
        ax1.annotate(
            f"R{r['run_id']:02d} {r['code']}\n({r['cu_e2e']:.4f} CU | FL2V={r['S_fl2v']:.1f})",
            (r["cu_e2e"], r["Q_total"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.6,
            fontweight="bold" if is_p else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.88, lw=0.85),
            arrowprops=dict(arrowstyle="-", color=c, lw=0.75, alpha=0.75),
        )

    ax1.set_xlim(0.066, 0.174)
    ax1.set_ylim(64.5, 94.0)

    ax1.set_title("B. Single-Video Marginal CU Cost (G4 @ 7.2046 CU/hr) vs. Quality", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Marginal Compute Cost per 124-Frame AV Video (Colab CU, lower is cheaper ←)", fontsize=11)
    ax1.set_ylabel("4D Composite Quality Score Q_total ∈ [0, 100] (higher is better ↑)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Experiment 03: MiniMax H3-Base-FL2VA (1344×768, 124f, 8-Step) Sol-Attn & Community Acceleration Pareto Frontier on Blackwell G4",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "pareto_frontier_cost_speed_quality.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def render_mechanisms_and_lifecycle_chart(data: Dict[str, Any]) -> None:
    micro = data["microbenchmark"]
    ops = micro["operators"]
    d1 = micro["discovery1_block_benchmark"]
    d2 = micro["discovery2_schedule_verification"]
    lc = data["lifecycle_telemetry"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2), dpi=200)

    # Panel A: Operator Microbenchmark at B=1, H=56, N=42058, D=128
    ax0 = axes[0]
    op_labels = [
        ("PyTorch SDPA\n(BF16 Dense)", ops["sdpa_cudnn_bf16"]["avg_ms"], "#1f77b4"),
        ("SageAttn 1.0.6\n(Triton INT8-QK)", ops["sage_triton_int8qk_bf16pv"]["avg_ms"], "#ff7f0e"),
        ("CK Pure-INT8\n(CUDA QK+PV)", ops["ck_dense_pure_int8"]["avg_ms"], "#e6550d"),
        ("CK Sol Direct\n(τ=1.0, tok=256)", ops["ck_sol_direct_t10_tok256"]["avg_ms"], "#31a354"),
        ("CK Sol Direct\n(τ=1.3, tok=256)", ops["ck_sol_direct_t13_tok256"]["avg_ms"], "#2ca02c"),
        ("CK Sol Direct\n(τ=1.3, tok=0)", ops["ck_sol_direct_t13_tok0"]["avg_ms"], "#006d2c"),
        ("Triton Sol\n(τ=1.3, INT8-QK)", ops["triton_sol_t13_int8qk_bf16pv"]["avg_ms"], "#17becf"),
        ("Triton Sol\n(τ=1.3, INT8-QK+PV)", ops["triton_sol_t13_int8qkpv"]["avg_ms"], "#08519c"),
    ]
    y_pos = np.arange(len(op_labels))
    ms_vals = [x[1] for x in op_labels]
    bars = ax0.barh(y_pos, ms_vals, color=[x[2] for x in op_labels], edgecolor="#222222", height=0.62)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([x[0] for x in op_labels], fontsize=8.8)
    ax0.invert_yaxis()
    ax0.set_xlabel("Single-Call Latency at N=42,058, H=56, D=128 (ms, lower is faster ←)", fontsize=10)
    ax0.set_title("A. Operator Latency & Effective TFLOPS (H3 Joint AV Shape)", fontsize=11.5, fontweight="bold")
    ax0.grid(True, axis="x", linestyle=":", alpha=0.55)
    for bar, (_, ms_v, _) in zip(bars, op_labels):
        tflops = micro["flops_per_call_tflops"] / (ms_v / 1000.0)
        ax0.text(
            ms_v + max(ms_vals) * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{ms_v:.1f} ms ({tflops:.0f} TFLOPS)",
            va="center", fontsize=8.2, fontweight="bold",
        )
    ax0.set_xlim(0, max(ms_vals) * 1.32)

    # Panel B: Discovery 1 (Chunked 2x-Pass vs Direct) & Discovery 2 (Flow-Shift Schedule Fix)
    ax1 = axes[1]
    steps_list = [s["step"] for s in d2["steps"]]
    p_stock = [s["stock_progress"] * 100.0 for s in d2["steps"]]
    p_fixed = [s["fixed_progress"] * 100.0 for s in d2["steps"]]
    ax1.plot(steps_list, p_stock, "o--", color="#d62728", linewidth=2.2, markersize=7,
             label=f"Stock _TauSchedule (1 - σ): {d2['stock_dense_steps_of_8']}/8 steps forced DENSE!")
    ax1.plot(steps_list, p_fixed, "s-", color="#2ca02c", linewidth=2.4, markersize=7,
             label=f"Corrected Unshifted (α=12.0): {d2['fixed_dense_steps_of_8']}/8 steps DENSE, 6/8 SPARSE")
    ax1.axhline(20.0, color="#333333", linestyle=":", linewidth=1.5, label="dense_percent = 20% Gate")
    ax1.fill_between(steps_list, 0, 20.0, color="#cccccc", alpha=0.35, label="Dense Warmup Zone (<20%)")
    ax1.set_xticks(steps_list)
    ax1.set_xticklabels([f"Step {s}\n(σ={d2['steps'][s]['sigma']:.2f})" for s in steps_list], fontsize=8)
    ax1.set_ylabel("Computed Sampling Progress (%)", fontsize=10)
    ax1.set_title("B. Discovery 1 & 2: Flow-Shift Schedule Audit & Producer Overhead", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    summary_box = (
        f"Discovery 1 (H3 DiTBlock @ N=42,058):\n"
        f"• Chunked 1st Sparse Step: {d1['chunked_cold_first_step_ms']:.1f} ms/blk "
        f"({d1['chunked_cold_qkv_calls_per_block']} qkv_proj calls = {d1['chunked_cold_qkv_calls_per_block']*50} GEMMs/step)\n"
        f"• Chunked Warm Step: {d1['chunked_warm_step_ms']:.1f} ms/blk "
        f"({d1['chunked_warm_qkv_calls_per_block']} qkv_proj calls = {d1['chunked_warm_qkv_calls_per_block']*50} GEMMs/step)\n"
        f"• Direct Zero-Copy Step: {d1['direct_zero_copy_step_ms']:.1f} ms/blk "
        f"({d1['direct_qkv_calls_per_block']} qkv_proj call = 50 GEMMs/step)"
    )
    ax1.text(
        0.03, 0.42, summary_box, transform=ax1.transAxes, fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc", ec="#b8860b", lw=1.0),
    )

    # Panel C: Session CU Focus Comparison (Exp 02 A100 vs Exp 02 G4 vs Exp 03 G4)
    ax2 = axes[2]
    if "session_theoretical_cu" not in lc or "session_core_inference_share_pct" not in lc:
        raise RuntimeError(
            "Missing session_theoretical_cu / session_core_inference_share_pct in lifecycle_telemetry; "
            "run `python3 run_g4_sol_attn_ablation.py --reconcile-ledger-only` first."
        )
    exp03_sess_cu = lc["session_theoretical_cu"]
    exp03_core_pct = lc["session_core_inference_share_pct"]
    exp03_lbl = f"Exp 03 G4\n(12 Runs, {exp03_sess_cu:.2f} CU)"
    sessions = [
        "Exp 02 A100\n(12 Runs, 7.31 CU)",
        "Exp 02 G4\n(4 Runs, 2.20 CU)",
        exp03_lbl,
    ]
    core_pcts = [46.58, 24.69, exp03_core_pct]
    non_core_pcts = [100.0 - x for x in core_pcts]

    x_idx = np.arange(len(sessions))
    b1 = ax2.bar(x_idx, core_pcts, width=0.52, color="#2ca02c", edgecolor="#222222", label="Core Ablation Inference (%)")
    b2 = ax2.bar(x_idx, non_core_pcts, width=0.52, bottom=core_pcts, color="#d9d9d9", edgecolor="#222222",
                 label="Bootstrap / Download / Warmup / Eval (%)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(sessions, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Share of Total Session Wall-Clock Time & CU (%)", fontsize=10)
    ax2.set_title("C. GPU Core Inference CU Focus: Exp 02 vs. Exp 03", fontsize=11.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax2.legend(loc="upper right", fontsize=8.5)
    for bar, pct in zip(b1, core_pcts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            pct / 2.0,
            f"Core Inference\n{pct:.1f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9.5,
        )

    fig.suptitle(
        "Experiment 03: Underlying Kernel Mechanisms, Source-Level Bug Audits & GPU Lifecycle CU Focus",
        fontsize=13.5, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUTPUT_DIR / "sol_attn_mechanisms_and_lifecycle_cu.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def render_frame_metrics_curves(data: Dict[str, Any]) -> None:
    runs = data["runs"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), dpi=180, sharex=True)
    cmap = plt.get_cmap("tab20")

    for idx, r in enumerate(runs):
        if r["run_id"] == 1:
            continue
        ref = r["ref_metrics"]
        psnr_t = ref["psnr_per_frame"]
        ssim_t = ref["ssim_per_frame"]
        lpips_t = ref["lpips_per_frame"]
        x = np.arange(len(psnr_t))
        c = cmap(idx)
        lw = 2.2 if r.get("is_pareto_e2e_qtotal") else 1.3
        ls = "-" if r.get("is_pareto_e2e_qtotal") else "--"
        lbl = f"R{r['run_id']:02d} {r['code']} (PSNR={ref['psnr_mean']:.1f}dB, SSIM={ref['ssim_mean']:.3f}, Q={r['Q_total']:.1f})"
        axes[0].plot(x, psnr_t, label=lbl, color=c, linewidth=lw, linestyle=ls)
        axes[1].plot(x, ssim_t, label=lbl, color=c, linewidth=lw, linestyle=ls)
        axes[2].plot(x, lpips_t, label=lbl, color=c, linewidth=lw, linestyle=ls)

    axes[0].set_ylabel("PSNR vs Run 01 (dB ↑)", fontsize=10.5, fontweight="bold")
    axes[0].set_title("Per-Frame Reference Fidelity Trajectories Across 124 Frames (Run 02 – Run 12 vs. Run 01 SDPA Baseline)", fontsize=12.5, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.55)
    axes[0].legend(loc="upper right", ncol=2, fontsize=7.8, framealpha=0.9)

    axes[1].set_ylabel("SSIM vs Run 01 (↑)", fontsize=10.5, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.55)

    axes[2].set_ylabel("LPIPS vs Run 01 (↓)", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Frame Index t ∈ [0, 123] (24 FPS, 5.17s Duration)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.55)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "frame_metrics_curves_12runs.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def render_keyframe_strip_and_grid_video(data: Dict[str, Any]) -> None:
    import imageio.v2 as imageio

    runs = data["runs"]
    all_frames: Dict[int, np.ndarray] = {}
    for r in runs:
        mp4_path = OUTPUT_DIR / r["mp4_file"]
        all_frames[r["run_id"]] = load_frames_from_mp4(mp4_path)

    T = all_frames[1].shape[0]
    kf_indices = [0, T // 4, T // 2, (3 * T) // 4, T - 1]
    thumb_w, thumb_h = 336, 192
    label_w = 340

    rows_img = []
    for r in runs:
        rid = r["run_id"]
        frames = all_frames[rid]
        row_cells = []

        # Left label banner
        banner = np.full((thumb_h, label_w, 3), 24, dtype=np.uint8)
        lines = [
            f"Run {rid:02d}: {r['code']}",
            f"{r['label'][:28]}",
            f"Denoise: {r['denoise_sec']:.2f}s ({r['denoise_speedup_vs_sdpa']:.2f}x)",
            f"E2E: {r['e2e_sec']:.2f}s ({r['cu_e2e']:.4f} CU)",
            f"Q_total: {r['Q_total']:.2f} | FL2V: {r['S_fl2v']:.2f}",
            f"PSNR: {r['ref_metrics']['psnr_mean']:.2f}dB | SSIM: {r['ref_metrics']['ssim_mean']:.3f}",
        ]
        for li, txt in enumerate(lines):
            color = (120, 255, 140) if r.get("is_pareto_e2e_qtotal") and li == 0 else (235, 235, 235)
            cv2.putText(
                banner, txt, (10, 26 + li * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
        row_cells.append(banner)

        for k_idx in kf_indices:
            cell = cv2.resize(frames[k_idx], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            pct = int(round(100.0 * k_idx / (T - 1)))
            cv2.putText(
                cell, f"f={k_idx} ({pct}%)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 230, 100), 1, cv2.LINE_AA,
            )
            row_cells.append(cell)

        rows_img.append(np.concatenate(row_cells, axis=1))

    strip_img = np.concatenate(rows_img, axis=0)
    strip_path = OUTPUT_DIR / "keyframe_strip_12runs.png"
    cv2.imwrite(str(strip_path), cv2.cvtColor(strip_img, cv2.COLOR_RGB2BGR))
    print(f"  ✅ Saved {strip_path}")

    # Render 3x4 12-Run Comparison Grid MP4 with Run 01 Stereo Audio
    cell_w, cell_h = 448, 256
    grid_cols, grid_rows = 4, 3
    tmp_grid_vid = OUTPUT_DIR / "comparison_grid_12runs.tmp_video.mp4"
    out_grid_mp4 = OUTPUT_DIR / "comparison_grid_12runs.mp4"

    writer = imageio.get_writer(
        str(tmp_grid_vid), fps=24, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16
    )
    for t in range(T):
        grid_frame = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
        for idx, r in enumerate(runs):
            rid = r["run_id"]
            gr = idx // grid_cols
            gc_idx = idx % grid_cols
            cell = cv2.resize(all_frames[rid][t], (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            # Semi-transparent top bar for readable HUD
            overlay = cell.copy()
            cv2.rectangle(overlay, (0, 0), (cell_w, 54), (0, 0, 0), -1)
            cell = cv2.addWeighted(overlay, 0.62, cell, 0.38, 0)
            hud1 = f"R{rid:02d} {r['code']} | Denoise {r['denoise_sec']:.1f}s ({r['denoise_speedup_vs_sdpa']:.2f}x)"
            hud2 = f"Cost {r['cu_e2e']:.4f} CU | Q={r['Q_total']:.1f} | FL2V={r['S_fl2v']:.1f}"
            cv2.putText(cell, hud1, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(cell, hud2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 255, 160), 1, cv2.LINE_AA)
            y0, y1 = gr * cell_h, (gr + 1) * cell_h
            x0, x1 = gc_idx * cell_w, (gc_idx + 1) * cell_w
            grid_frame[y0:y1, x0:x1] = cell
        writer.append_data(grid_frame)
    writer.close()

    run1_mp4 = OUTPUT_DIR / runs[0]["mp4_file"]
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_grid_vid),
            "-i", str(run1_mp4),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            str(out_grid_mp4),
        ],
        check=True,
    )
    tmp_grid_vid.unlink(missing_ok=True)
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise RuntimeError("ffprobe is required to verify 124 frames in comparison_grid_12runs.mp4")
    probe_out = subprocess.check_output(
        [
            ffprobe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
            str(out_grid_mp4),
        ],
        text=True,
    ).strip()
    if T != 124 or int(probe_out) != 124:
        raise RuntimeError(f"Expected 124 frames in {out_grid_mp4.name}, got T={T}, nb_read_frames={probe_out}")
    print(f"  ✅ Saved {out_grid_mp4} (ffprobe verified nb_read_frames == 124)")


def main() -> None:
    json_path = OUTPUT_DIR / "g4_sol_attn_ablation_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("🎨 Rendering Experiment 03 Visual Artifacts Locally on Mac (0.00 CU)...")
    render_pareto_frontier_chart(data)
    render_mechanisms_and_lifecycle_chart(data)
    render_frame_metrics_curves(data)
    render_keyframe_strip_and_grid_video(data)
    print("✅ All visual artifacts rendered successfully!")


if __name__ == "__main__":
    main()
