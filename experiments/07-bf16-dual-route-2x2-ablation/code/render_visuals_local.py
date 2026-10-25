#!/usr/bin/env python3
"""
Local Zero-CU Visual Chart & 2x2 Quad-Grid Video Renderer for Experiment 07
(`experiments/07-bf16-dual-route-2x2-ablation/code/render_visuals_local.py`).
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

_SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = _SCRIPT_DIR.parent
OUTPUTS_DIR = EXP_ROOT / "outputs"


def _read_video_frames_ffmpeg(mp4_path: Path, width: int = 1344, height: int = 768) -> np.ndarray:
    ffmpeg_bin = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", str(mp4_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    frame_bytes = width * height * 3
    n_frames = len(raw) // frame_bytes
    return np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8).reshape((n_frames, height, width, 3))


def render_charts_and_quad_video(metrics_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = payload["runs"]

    labels = [f"{r['short']}\n({r['profile_id']})" for r in runs]
    denoise_s = [float(r["denoise_sec"]) for r in runs]
    e2e_s = [float(r["e2e_sec"]) for r in runs]
    cu_e2e = [float(r["cu_e2e"]) for r in runs]
    sec_per_step = [float(r["sec_per_step"]) for r in runs]

    # Figure 1: Speed & CU Comparison across 2x2 Matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    x = np.arange(len(runs))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, denoise_s, w, label="DiT Denoise (s)", color="#2563eb")
    b2 = axes[0].bar(x + w / 2, e2e_s, w, label="Total GPU E2E (s)", color="#0ea5e9")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Latency (seconds)")
    axes[0].set_title("Experiment 07: 2x2 BF16 Latency Comparison (1344x768, 124f)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}s", ha="center", va="bottom", fontsize=8)

    b3 = axes[1].bar(x - w / 2, cu_e2e, w, label="E2E Cost (CU)", color="#10b981")
    ax2_r = axes[1].twinx()
    b4 = ax2_r.bar(x + w / 2, sec_per_step, w, label="Per-Step Latency (s/step)", color="#f59e0b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Colab Compute Units (CU)")
    ax2_r.set_ylabel("Seconds / Step")
    axes[1].set_title("Experiment 07: CU Cost & Per-Step Efficiency")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b3:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f} CU", ha="center", va="bottom", fontsize=8)
    for bar in b4:
        h = bar.get_height()
        ax2_r.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}s", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig1_exp07_2x2_speed_and_cu.png")
    plt.close(fig)

    # Figure 2: 4D Quality & Production Quality Comparison
    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    m_names = ["S_ref (Intra-Route)", "S_vbench", "S_fl2v", "S_audio", "Q_total", "prod_quality"]
    colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b", "#10b981"]
    bw = 0.13
    for i, (m_label, col) in enumerate(zip(m_names, colors)):
        vals = []
        for r in runs:
            q = r["quality"]
            if i == 0:
                vals.append(float(q["ref_fidelity"]["S_ref"]))
            elif i == 1:
                vals.append(float(q["vbench_temporal"]["S_vbench"]))
            elif i == 2:
                vals.append(float(q["fl2v_alignment"]["S_fl2v"]))
            elif i == 3:
                vals.append(float(q["audio_quality"]["S_audio"]))
            elif i == 4:
                vals.append(float(q["Q_total"]))
            else:
                vals.append(float(q["production_metrics"]["prod_quality_score"]))
        offset = (i - 2.5) * bw
        bars = ax.bar(x + offset, vals, bw, label=m_label, color=col)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (0 - 100)")
    ax.set_title("Experiment 07: 4-Dimensional Quality & Production Quality across 2x2 BF16 Matrix")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig2_exp07_2x2_4d_quality_breakdown.png")
    plt.close(fig)

    # Read all 4 MP4 videos for keyframe strip & 2x2 quad-grid comparison video
    videos_u8: List[np.ndarray] = []
    for r in runs:
        mp4_p = OUTPUTS_DIR / Path(r["video_file"]).name
        videos_u8.append(_read_video_frames_ffmpeg(mp4_p, 1344, 768))

    # Figure 3: Keyframe Strip (Frames 0, 31, 62, 93, 123)
    kf_indices = [0, 31, 62, 93, 123]
    thumb_w, thumb_h = 336, 192
    strip_img = Image.new("RGB", (thumb_w * len(kf_indices), thumb_h * len(runs)), (15, 15, 20))
    draw = ImageDraw.Draw(strip_img)
    for r_idx, (r, v_arr) in enumerate(zip(runs, videos_u8)):
        for c_idx, f_no in enumerate(kf_indices):
            f_clamped = min(f_no, v_arr.shape[0] - 1)
            tile = Image.fromarray(v_arr[f_clamped]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            strip_img.paste(tile, (c_idx * thumb_w, r_idx * thumb_h))
            label_txt = f"{r['short']} | f={f_clamped} | E2E={r['e2e_sec']:.1f}s | Q={r['quality']['Q_total']:.1f}"
            draw.rectangle([c_idx * thumb_w + 4, r_idx * thumb_h + 4, c_idx * thumb_w + 328, r_idx * thumb_h + 24], fill=(0, 0, 0))
            draw.text((c_idx * thumb_w + 8, r_idx * thumb_h + 7), label_txt, fill=(255, 255, 255))
    strip_img.save(OUTPUTS_DIR / "fig3_exp07_keyframe_strip.png")

    # 2x2 Quad-Grid 124-Frame Comparison Video (1344x768)
    n_frames = min(v.shape[0] for v in videos_u8)
    tile_w, tile_h = 672, 384
    quad_frames = np.zeros((n_frames, 768, 1344, 3), dtype=np.uint8)
    for t in range(n_frames):
        for r_idx, (r, v_arr) in enumerate(zip(runs, videos_u8)):
            row, col = divmod(r_idx, 2)
            tile_pil = Image.fromarray(v_arr[t]).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            d_tile = ImageDraw.Draw(tile_pil)
            banner = (
                f"{r['short']} ({r['profile_id']}) | {r['steps']} steps ({r['scheduler']}) | "
                f"E2E: {r['e2e_sec']:.1f}s ({r['cu_e2e']:.3f} CU) | Q_total: {r['quality']['Q_total']:.2f}"
            )
            d_tile.rectangle([6, 6, tile_w - 6, 30], fill=(0, 0, 0))
            d_tile.text((12, 11), banner, fill=(255, 255, 255))
            quad_frames[t, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = np.asarray(tile_pil)

    ffmpeg_bin = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg") or "ffmpeg"
    tmp_quad = OUTPUTS_DIR / "_tmp_quad_video.mp4"
    out_quad = OUTPUTS_DIR / "exp07_2x2_quad_grid_comparison.mp4"
    run01_mp4 = OUTPUTS_DIR / Path(runs[0]["video_file"]).name

    proc = subprocess.Popen(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1344x768", "-r", "24",
            "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(tmp_quad),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(quad_frames.tobytes())
    proc.stdin.close()
    proc.wait()

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_quad), "-i", str(run01_mp4),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            str(out_quad),
        ],
        check=True,
    )
    tmp_quad.unlink(missing_ok=True)
    print(f"  ✅ Rendered fig1, fig2, fig3 and 2x2 comparison video: {out_quad}")


if __name__ == "__main__":
    render_charts_and_quad_video(OUTPUTS_DIR / "exp07_bf16_2x2_metrics.json")
