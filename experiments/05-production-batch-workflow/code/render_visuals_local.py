#!/usr/bin/env python3
"""
Local Zero-CU Visual & Multi-Grid Video Renderer (`render_visuals_local.py`) for Experiment 05.
Executes entirely on the local Mac (0.00 CU/hr) after Stage 2 Google Drive pull completes.

Generates:
  1. outputs/batch_amortization_and_lifecycle_cu.png
  2. outputs/per_job_latency_and_seed_quality.png
  3. outputs/keyframe_strip_6jobs.png
  4. outputs/comparison_grid_6jobs.mp4
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
EXP_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "code" else _SCRIPT_DIR
OUTPUT_DIR = EXP_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "exp05_batch_workflow_metrics.json"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SHOT_COLORS = {
    "shot_01_cyber_mouse_orbit": "#2563EB",
    "shot_02_neon_circuit_pulse": "#10B981",
    "shot_03_hero_reveal_finale": "#F59E0B",
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


def render_amortization_and_ablation_chart(metrics: Dict[str, Any]) -> None:
    curve = metrics["batch_amortization_curve"]
    cond_ab = metrics["conditioning_dedup_ablation"]
    p1_p2 = metrics["pipeline_p1_vs_p2_ablation"]
    lt = metrics["lifecycle_telemetry"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20.5, 6.6), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.set_axisbelow(True)

    # Panel A: Batch Amortization Curve (CU/video & Videos per 100 CU vs Batch Size M)
    ms = [row["batch_size_M"] for row in curve]
    cu_per_vid = [row["amortized_cu_per_video"] for row in curve]
    core_share = [row["core_render_share_pct"] for row in curve]
    marginal_cu = lt["mean_job_marginal_cu"]

    ax1.plot(ms, cu_per_vid, marker="o", linewidth=2.6, color="#2563EB", label="攒批单次提交: 摊薄后单条视频总成本 (CU/条)")
    ax1.axhline(y=cu_per_vid[0], color="#EF4444", linestyle=":", linewidth=1.8, label=f"交互式单跑 1 条 (M=1): {cu_per_vid[0]:.4f} CU/条")
    ax1.axhline(y=marginal_cu, color="#10B981", linestyle="--", linewidth=2.0, label=f"Run 12 纯边际成本渐近下界: {marginal_cu:.4f} CU/条")

    m6_row = next(r for r in curve if r["batch_size_M"] == 6)
    ax1.scatter([6], [m6_row["amortized_cu_per_video"]], color="#D97706", s=130, zorder=6, marker="D")
    ax1.annotate(
        f"本次实测批次 (M=6)\n摊薄至 {m6_row['amortized_cu_per_video']:.4f} CU/条\n(较 M=1 省 {m6_row['cu_saving_pct_vs_M1_interactive']:.1f}% CU!)",
        xy=(6, m6_row["amortized_cu_per_video"]),
        xytext=(9.0, m6_row["amortized_cu_per_video"] + 0.09),
        fontsize=8.5,
        fontweight="bold",
        color="#B45309",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#D97706", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.2),
    )

    ax1_r = ax1.twinx()
    ax1_r.plot(ms, core_share, marker="s", linestyle="-.", linewidth=1.8, color="#8B5CF6", alpha=0.85, label="核心渲染占会话总 CU 比例 (%)")
    ax1_r.set_ylabel("核心渲染占会话总时长/CU 比例 (%)", fontsize=9.8, color="#6D28D9")
    ax1_r.set_ylim(10, 102)

    ax1.set_xlabel("单次 G4 会话攒批任务数 M (条视频/会话)", fontsize=10.2, fontweight="bold")
    ax1.set_ylabel("摊薄后每条 124 帧视频总算力成本 (CU)", fontsize=10.2, fontweight="bold")
    ax1.set_title("A. 固定冷启动税摊薄曲线 (M=1..32, Run 12)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylim(0.06, max(cu_per_vid) * 1.15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.8, framealpha=0.95)

    # Panel B: Staged Conditioning Pre-Encode & SHA-256 Deduplication vs Naive
    cond_categories = ["朴素逐任务重复编码\n(6次 Qwen3-VL + VAE)", "本工作流: 阶段式去重预编码\n(3次唯一分镜 + 3次 0.0s 缓存命中)"]
    cond_vals = [cond_ab["naive_6job_encode_sum_sec"], cond_ab["staged_dedup_encode_sum_sec"]]
    bars_c = ax2.bar([0, 1], cond_vals, width=0.46, color=["#EF4444", "#10B981"], edgecolor=["#991B1B", "#065F46"], linewidth=1.2)
    for b, v in zip(bars_c, cond_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0, v + 0.6,
            f"{v:.2f}s\n({v * lt['mean_job_marginal_cu'] / lt['mean_job_gpu_e2e_sec']:.4f} CU)",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold"
        )
    ax2.annotate(
        f"SHA-256 条件缓存命中 3/6 任务 (0.000s!)\n净省 {cond_ab['encode_time_saved_sec']:.2f}s (-{cond_ab['encode_time_saving_pct']:.1f}%)\n"
        f"+ 编码完立即卸载 Qwen3-VL 释放 {cond_ab['vram_freed_by_vlm_unload_gib']:.2f} GiB 显存!",
        xy=(1, cond_vals[1]),
        xytext=(0.12, max(cond_vals) * 0.76),
        fontsize=8.3,
        fontweight="bold",
        color="#065F46",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#10B981", lw=1.0),
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(cond_categories, fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("6 任务总条件编码耗时 (秒)", fontsize=10.2, fontweight="bold")
    ax2.set_title("B. SHA-256 条件去重缓存与 Qwen3-VL 卸载收益", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylim(0, max(cond_vals) * 1.32)

    # Panel C: Pipeline Mode P1 (Sequential GPU + Async CPU) vs Mode P2 (Dual CUDA Stream Overlap)
    p_labels = ["模式 P1 (推荐):\nGPU单流顺序 DiT->VAE\n+ 后台 CPU 异步 ffmpeg", "模式 P2 (对照):\n双 CUDA Stream 并发\nVAE(K) || DiT(K+1)"]
    p_times = [p1_p2["p1_sequential_transition_sum_sec"], p1_p2["p2_overlapped_wall_sec"]]
    p_vrams = [p1_p2["p1_peak_vram_gib"], p1_p2["p2_peak_vram_gib"]]

    x_p = np.arange(2)
    w_p = 0.34
    b_t = ax3.bar(x_p - w_p / 2, p_times, width=w_p, color="#3B82F6", edgecolor="#1D4ED8", label="单次跨任务过渡耗时 (VAE_K + DiT_K+1, 秒)")
    ax3_r = ax3.twinx()
    b_v = ax3_r.bar(x_p + w_p / 2, p_vrams, width=w_p, color="#F59E0B", edgecolor="#B45309", hatch="//", alpha=0.88, label="峰值显存占用 Peak VRAM (GiB)")

    for b, v in zip(b_t, p_times):
        ax3.text(b.get_x() + b.get_width() / 2.0, v + 0.8, f"{v:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E40AF")
    for b, v in zip(b_v, p_vrams):
        ax3_r.text(b.get_x() + b.get_width() / 2.0, v + 0.8, f"{v:.2f} GiB", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#92400E")

    summary_p12 = (
        f"双流重叠实测剖析:\n"
        f"• SM 争用使 VAE 从 {p1_p2['p1_sequential_vae_sec']:.2f}s 拉长至 {p1_p2['p2_vae_stream_under_contention_sec']:.2f}s ({p1_p2['vae_slowdown_ratio_under_sm_contention']:.2f}x)\n"
        f"• 净时间差: {p1_p2['net_wall_saving_per_transition_sec']:+.2f}s ({p1_p2['net_wall_saving_pct_of_e2e']:+.2f}%) | 显存: {p1_p2['p1_peak_vram_gib']:.2f}->{p1_p2['p2_peak_vram_gib']:.2f} GiB\n"
        f"• 输出精度: PSNR={p1_p2['p1_vs_p2_output_psnr_db']:.1f} dB (max_diff={p1_p2['p1_vs_p2_output_max_abs_diff']:.5f})"
    )
    ax3.text(
        0.03, 0.77, summary_p12, transform=ax3.transAxes, fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.32", fc="#FFFBEB", ec="#D97706", lw=1.0)
    )

    ax3.set_xticks(x_p)
    ax3.set_xticklabels(p_labels, fontsize=8.8, fontweight="bold")
    ax3.set_ylabel("跨任务过渡耗时 (秒)", fontsize=10.0, fontweight="bold", color="#1E40AF")
    ax3_r.set_ylabel("峰值显存占用 Peak VRAM (GiB)", fontsize=10.0, fontweight="bold", color="#92400E")
    ax3.set_ylim(0, max(p_times) * 1.62)
    ax3_r.set_ylim(0, max(p_vrams) * 1.62)
    ax3.set_title("C. 流水线微消融: 模式 P1 vs. 模式 P2 (双流重叠)", fontsize=11.5, fontweight="bold", pad=10)
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3_r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=7.6, framealpha=0.95)

    fig.suptitle(
        "Experiment 05: MiniMax H3-Base-FL2VA 异构批处理生产工作流 — 冷启动摊薄、SHA-256 条件去重与 P1/P2 流水线消融 (Blackwell G4)",
        fontsize=13.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUTPUT_DIR / "batch_amortization_and_lifecycle_cu.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def render_per_job_latency_and_quality_chart(metrics: Dict[str, Any]) -> None:
    jobs = metrics["jobs"]
    cond_events = {e["job_id"]: e for e in metrics["conditioning_dedup_ablation"]["per_job_conditioning_events"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 6.6), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.set_axisbelow(True)

    x = np.arange(len(jobs))
    j_labels = [f"Job {j['index']:02d}\n({j['shot_id'][:7]}..)\nseed={j['seed']}" for j in jobs]
    cond_s = [cond_events[j["job_id"]]["encode_sec"] for j in jobs]
    den_s = [j["denoise_sec"] for j in jobs]
    vae_s = [j["vae_total_sec"] for j in jobs]
    ffmpeg_s = [j["async_ffmpeg_mux_sec"] for j in jobs]

    w = 0.36
    b1 = ax1.bar(x - w / 2, cond_s, width=w, color="#8B5CF6", edgecolor="#5B21B6", label="阶段1: 条件编码 (命中缓存 = 0.00s!)")
    b2 = ax1.bar(x - w / 2, den_s, bottom=cond_s, width=w, color="#2563EB", edgecolor="#1E40AF", label="阶段2: 常驻 Run 12 8步去噪 (0.00s 切换开销)")
    bottom_vae = [c + d for c, d in zip(cond_s, den_s)]
    b3 = ax1.bar(x - w / 2, vae_s, bottom=bottom_vae, width=w, color="#10B981", edgecolor="#065F46", label="阶段2: Video+Audio VAE 解码")
    b4 = ax1.bar(x + w / 2, ffmpeg_s, width=w, color="#94A3B8", edgecolor="#475569", hatch="//", label="后台 CPU 异步 ffmpeg 音画合成 (0.00s 阻塞 GPU!)")

    for i, j in enumerate(jobs):
        tot_gpu = cond_s[i] + den_s[i] + vae_s[i]
        hit_str = "HIT 0.0s" if cond_events[j["job_id"]]["cache_hit"] else f"Cond {cond_s[i]:.1f}s"
        ax1.text(
            x[i] - w / 2, tot_gpu + 0.7,
            f"GPU {tot_gpu:.1f}s\n({hit_str})",
            ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#1E3A8A"
        )
        ax1.text(
            x[i] + w / 2, ffmpeg_s[i] + 0.5,
            f"CPU {ffmpeg_s[i]:.1f}s\n(全重叠)",
            ha="center", va="bottom", fontsize=7.2, color="#475569"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(j_labels, fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("单任务阶段耗时 (秒)", fontsize=10.2, fontweight="bold")
    ax1.set_title("A. 6 任务连续常驻渲染耗时拆解 (GPU 主流 vs. 后台 CPU 异步封装)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylim(0, max(bottom_vae[i] + vae_s[i] for i in range(len(jobs))) * 1.22)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # Panel B: Multi-Seed Quality Audit & Best-Seed Recommendation per Storyboard
    fl2v_scores = [j["S_fl2v_pixel"] for j in jobs]
    prod_scores = [j["prod_quality_score"] for j in jobs]
    colors_b = [SHOT_COLORS.get(j["shot_id"], "#2563EB") for j in jobs]

    bars_q = ax2.bar(x - 0.18, prod_scores, width=0.34, color=colors_b, edgecolor="#1E293B", linewidth=1.2, label="生产综合质量评分 Prod_Score (按分镜着色)")
    bars_f = ax2.bar(x + 0.18, fl2v_scores, width=0.34, color="#64748B", edgecolor="#1E293B", alpha=0.82, label="首尾帧像素锚定分 S_fl2v_pixel")

    for i, j in enumerate(jobs):
        is_best = j.get("is_recommended_seed_for_shot", False)
        badge = f"★ 推荐 #{j['shot_seed_rank']}\n{prod_scores[i]:.2f}" if is_best else f"#{j['shot_seed_rank']}\n{prod_scores[i]:.2f}"
        ax2.text(
            x[i] - 0.18, prod_scores[i] + 0.4,
            badge,
            ha="center", va="bottom", fontsize=7.8,
            fontweight="bold" if is_best else "normal",
            color="#B45309" if is_best else "#1E293B",
        )
        ax2.text(
            x[i] + 0.18, fl2v_scores[i] + 0.4,
            f"{fl2v_scores[i]:.1f}",
            ha="center", va="bottom", fontsize=7.5, color="#334155"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(j_labels, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("质量评分 ∈ [0, 100] (越高越优)", fontsize=10.2, fontweight="bold")
    ax2.set_title("B. 多分镜多 Seed 抽卡自动质检评分与最佳 Seed 优选", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylim(75, 103)
    ax2.legend(loc="lower right", fontsize=8.2, framealpha=0.95)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "per_job_latency_and_seed_quality.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def render_keyframe_strip_and_grid_mp4(metrics: Dict[str, Any]) -> None:
    jobs = metrics["jobs"]
    all_frames = {j["job_id"]: load_frames_from_mp4(OUTPUT_DIR / j["mp4_file"]) for j in jobs}

    # 1. Keyframe Strip (6 jobs x 5 keyframes: 0, 30, 62, 93, 123)
    kf_indices = [0, 30, 62, 93, 123]
    fig, axes = plt.subplots(len(jobs), len(kf_indices), figsize=(18.5, 13.5), dpi=180)
    fig.patch.set_facecolor("#0F172A")

    for r_idx, j in enumerate(jobs):
        frames = all_frames[j["job_id"]]
        rec_tag = " [★ BEST SEED]" if j.get("is_recommended_seed_for_shot") else ""
        row_label = (
            f"Job {j['index']:02d}: {j['shot_id']}\n"
            f"seed={j['seed']}{rec_tag}\n"
            f"GPU={j['gpu_e2e_sec']:.2f}s ({j['cu_e2e']:.4f} CU)\n"
            f"Score={j['prod_quality_score']:.2f} | FL2V={j['S_fl2v_pixel']:.1f}"
        )
        for c_idx, f_idx in enumerate(kf_indices):
            ax = axes[r_idx, c_idx]
            idx_clamped = min(f_idx, frames.shape[0] - 1)
            ax.imshow(frames[idx_clamped])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#F59E0B" if j.get("is_recommended_seed_for_shot") else "#334155")
                spine.set_linewidth(2.0 if j.get("is_recommended_seed_for_shot") else 0.8)
            if r_idx == 0:
                ax.set_title(f"Frame {idx_clamped:03d} ({idx_clamped/24.0:.2f}s)", fontsize=10.5, fontweight="bold", color="#F8FAFC", pad=6)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=8.3, fontweight="bold", color="#F8FAFC", rotation=0, labelpad=88, va="center")

    fig.suptitle(
        "Experiment 05: 3 Storyboards × 6 Batch Jobs Keyframe Audit Strip (Locked ADR-03 Run 12 [S8-FullStack-Pareto])",
        fontsize=14.0, fontweight="bold", color="#F8FAFC", y=0.985
    )
    plt.tight_layout(rect=[0.11, 0, 1, 0.965])
    strip_path = OUTPUT_DIR / "keyframe_strip_6jobs.png"
    fig.savefig(strip_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved {strip_path}")

    # 2. 3x2 Synchronized Comparison Grid MP4 (0.00 CU on local Mac CPU)
    cell_w, cell_h = 640, 360
    cols, rows = 3, 2
    grid_w, grid_h = cell_w * cols, cell_h * rows
    num_frames = min(f.shape[0] for f in all_frames.values())
    tmp_grid = OUTPUT_DIR / "_tmp_grid_noaudio.mp4"
    out_grid = OUTPUT_DIR / "comparison_grid_6jobs.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_grid), fourcc, 24.0, (grid_w, grid_h))
    for t in range(num_frames):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, j in enumerate(jobs):
            r, c = divmod(idx, cols)
            y0, x0 = r * cell_h, c * cell_w
            frm_rgb = all_frames[j["job_id"]][t]
            frm_bgr = cv2.resize(cv2.cvtColor(frm_rgb, cv2.COLOR_RGB2BGR), (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            star = "*BEST* " if j.get("is_recommended_seed_for_shot") else ""
            header = f"Job {j['index']:02d} | {j['shot_id'][:18]} | seed={j['seed']} {star}"
            sub = f"Run 12: {j['gpu_e2e_sec']:.2f}s ({j['cu_e2e']:.4f} CU) | Score={j['prod_quality_score']:.1f} | F={t:03d}"
            cv2.rectangle(frm_bgr, (0, 0), (cell_w, 50), (15, 23, 42), -1)
            cv2.putText(frm_bgr, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1, cv2.LINE_AA)
            cv2.putText(frm_bgr, sub, (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = frm_bgr
        writer.write(canvas)
    writer.release()

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ref_audio_mp4 = OUTPUT_DIR / jobs[0]["mp4_file"]
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_grid), "-i", str(ref_audio_mp4),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-c:a", "copy",
            str(out_grid),
        ],
        check=True,
    )
    tmp_grid.unlink(missing_ok=True)
    print(f"  ✅ Saved {out_grid}")


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    render_amortization_and_ablation_chart(metrics)
    render_per_job_latency_and_quality_chart(metrics)
    render_keyframe_strip_and_grid_mp4(metrics)
    print("🎨 [Local 0-CU Visual Renderer] All Experiment 05 figures & comparison grid video generated!")


if __name__ == "__main__":
    main()
