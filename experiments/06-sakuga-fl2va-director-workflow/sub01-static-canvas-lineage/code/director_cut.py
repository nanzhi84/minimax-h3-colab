#!/usr/bin/env python3
"""
Experiment 06: Reusable AI Short-Form / Sakuga FL2VA Director Workflow Toolkit
(`experiments/06-sakuga-fl2va-director-workflow/code/director_cut.py`).

Provides three zero-CU local stages around the Cloud G4 5x-Seed Batch Engine:
  1. `--align-keyframes`: Aligns `Frame A .. Frame F` to exact `1344x768` (multiple of 32),
     computes shared-lineage consistency metrics, and renders `outputs/keyframe_storyboard_sheet.png`.
  2. `--select-and-cut <batch_dir>`: Evaluates all 5 seeds per shot (20 videos total), renders a
     4x5 Seed Gacha comparison matrix, and performs frame-accurate seamless director editing:
       - Removes duplicate boundary Frame 0 on chained transitions (`A->B` to `B->C`, `D->E` to `E->F`)
       - Cuts on kinetic action peaks (eliminating AI tail deceleration)
       - Injects 2 Ufotable-style high-contrast black-and-white Impact Frames at climax transitions
       - Applies equal-power audio crossfades & J-Cut pre-rolls to prevent waveform clicks/pops
       - Outputs exact 360-frame (`15.00s @ 24fps`) `director_master_15s.mp4`.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:
    uv_bin = shutil.which("uv")
    if uv_bin and os.environ.get("_DIRECTOR_CUT_UV_REEXEC") != "1":
        env = dict(os.environ, _DIRECTOR_CUT_UV_REEXEC="1")
        os.execve(
            uv_bin,
            [
                uv_bin,
                "run",
                "--with",
                "pillow",
                "--with",
                "numpy",
                "--with",
                "opencv-python-headless",
                "--with",
                "soundfile",
                "--with",
                "imageio",
                "--with",
                "imageio[ffmpeg]",
                "python3",
                *sys.argv,
            ],
            env,
        )
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUTPUTS_DIR = EXP_DIR / "outputs"
MANIFEST_PATH = SCRIPT_DIR / "exp06_sakuga_15s_manifest.json"

TARGET_W = 1344
TARGET_H = 768
FPS = 24

KEYFRAME_SPECS: List[Dict[str, str]] = [
    {
        "key": "Frame A",
        "file": "frame_A_standoff.png",
        "role": "Shot 1 First Frame (Base Master)",
        "chain": "Chain 1 Start (A)",
        "desc": "崩塌神庙雷炎对峙全景 (银发雷剑士拔刀蓄力 vs 黑甲绯红炎武者)",
    },
    {
        "key": "Frame B",
        "file": "frame_B_thunder_dash.png",
        "role": "Shot 1 Last / Shot 2 First (Shared Node)",
        "chain": "A -> B (Seamless Bridge)",
        "desc": "同基底演化：雷光破空贴地疾刺逼近对手身前，炎武者举刀格挡",
    },
    {
        "key": "Frame C",
        "file": "frame_C_blade_clash.png",
        "role": "Shot 2 Last Frame",
        "chain": "B -> C (Chain 1 End)",
        "desc": "同基底演化：双刃十字死锁迸发白炽雷火与环形冲击波气浪",
    },
    {
        "key": "Frame D",
        "file": "frame_D_eye_closeup.png",
        "role": "Shot 3 First Frame (Montage Cut)",
        "chain": "C || D (Cross-Cam Cut)",
        "desc": "同源跨机位特写：剑士瞳孔苍蓝雷光骤缩，横刃蓄满白炽高压等离子",
    },
    {
        "key": "Frame E",
        "file": "frame_E_dragon_slash.png",
        "role": "Shot 3 Last / Shot 4 First (Shared Node)",
        "chain": "D -> E (Seamless Bridge)",
        "desc": "同基底演化：全力横斩挥出苍蓝雷弧，刀锋化作咆哮雷龙破空而出",
    },
    {
        "key": "Frame F",
        "file": "frame_F_dragon_finale.png",
        "role": "Shot 4 Last Frame (Climax Finale)",
        "chain": "E -> F (Chain 2 End)",
        "desc": "同基底终章：巨型苍蓝雷龙吞噬火柱贯穿神庙，碎石浮空直冲云霄",
    },
]

# Director trimming & transition blueprint for exact 360 frames = 15.00s @ 24fps
# Rule: At Shared-Node Bridges (Frame B between Shot 1 & 2, Frame E between Shot 3 & 4),
# Clip K MUST run to its final frame (N-1) and Clip K+1 MUST start at frame 1 (dropping duplicate Frame 0).
# All duration trimming occurs at outer boundaries (Shot 1 start, Shot 4 tail) and the Cross-Camera Cut (Shot 2 tail || Shot 3 start).
DIRECTOR_EDIT_BLUEPRINT: List[Dict[str, Any]] = [
    {
        "shot_id": "shot_01_thunder_dash",
        "chain_mode": "chain_start_to_bridge_B",
        "raw_frames": 90,
        "trim_start_frame": 6,   # Trim initial 6 static wind-up frames; run all the way to Frame 89 (Frame B!)
        "keep_frames": 84,      # Frames [6..89] = 84 frames (3.50s), ending at exact Frame B anchor
        "impact_flash_frames": 0,
        "impact_flash_offset": 0,
        "audio_fade_in_sec": 0.05,
        "crossfade_prev_sec": 0.0,
        "note": "Keep [6..89] (3.50s): Trim 6f static wind-up, end at exact Frame 89 (Frame B bridge)",
    },
    {
        "shot_id": "shot_02_blade_clash_shockwave",
        "chain_mode": "seamless_bridge_B_to_montage",
        "raw_frames": 124,
        "trim_start_frame": 1,   # Drop Frame 0 (Frame B duplicate) for 100% seamless A->B->C continuity!
        "keep_frames": 101,      # Frames [1..101] = 101 frames (4.21s), cut on peak shockwave before tail slowdown
        "impact_flash_frames": 0,
        "impact_flash_offset": 0,
        "audio_fade_in_sec": 0.0,
        "crossfade_prev_sec": 0.08,
        "note": "Drop Frame 0 (Frame B dup), Keep [1..101] (4.21s) + 80ms audio crossfade, cut on shockwave peak",
    },
    {
        "shot_id": "shot_03_eye_awakening_slash",
        "chain_mode": "montage_j_cut_to_bridge_E",
        "raw_frames": 73,
        "trim_start_frame": 13,  # Cut in on action at frame 13; run all the way to Frame 72 (Frame E!)
        "keep_frames": 60,       # Frames [13..72] = 60 frames (2.50s), ending at exact Frame E anchor
        "impact_flash_frames": 0,
        "impact_flash_offset": 0,
        "audio_fade_in_sec": 0.0,
        "crossfade_prev_sec": 0.12,  # 120ms J-Cut audio pre-roll
        "note": "Montage J-Cut: Keep [13..72] (2.50s), end at exact Frame 72 (Frame E bridge) + 120ms J-Cut",
    },
    {
        "shot_id": "shot_04_dragon_finale_blast",
        "chain_mode": "seamless_bridge_E_impact_finale",
        "raw_frames": 124,
        "trim_start_frame": 1,   # Drop Frame 0 (Frame E duplicate) for seamless D->E->F continuity!
        "keep_frames": 115,      # Frames [1..115] = 115 frames (4.79s), trim last 8 static tail frames
        "impact_flash_frames": 2,
        "impact_flash_offset": 2,  # Preserve f[0..1] color bridge across Frame E, flash B&W Impact Frames at f[2..3]
        "audio_fade_in_sec": 0.0,
        "crossfade_prev_sec": 0.08,
        "note": "Drop Frame 0 (Frame E dup), preserve 2f color bridge then inject 2 B&W Impact Frames, Keep [1..115] (4.79s)",
    },
]


def compute_hist_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Computes normalized RGB color histogram correlation between two [H, W, 3] float32 images."""
    scores = []
    for c in range(3):
        ha, _ = np.histogram(img_a[:, :, c], bins=64, range=(0.0, 1.0), density=True)
        hb, _ = np.histogram(img_b[:, :, c], bins=64, range=(0.0, 1.0), density=True)
        corr = float(np.corrcoef(ha, hb)[0, 1])
        scores.append(max(0.0, corr))
    return round(100.0 * float(np.mean(scores)), 2)


def align_and_verify_keyframes() -> Dict[str, Any]:
    """
    Ensures all 6 keyframes in `code/assets/` are exact `1344x768` RGB PNGs (LANCZOS cover-fit),
    computes pairwise shared-lineage consistency, and renders `outputs/keyframe_storyboard_sheet.png`.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    aligned_nps: Dict[str, np.ndarray] = {}
    aligned_imgs: Dict[str, Image.Image] = {}
    frame_reports: List[Dict[str, Any]] = []

    for spec in KEYFRAME_SPECS:
        fpath = ASSETS_DIR / spec["file"]
        if not fpath.exists():
            raise FileNotFoundError(f"Missing keyframe image: {fpath}")
        raw_img = Image.open(fpath).convert("RGB")
        orig_size = raw_img.size
        if orig_size != (TARGET_W, TARGET_H):
            aligned_img = ImageOps.fit(raw_img, (TARGET_W, TARGET_H), method=Image.Resampling.LANCZOS)
            aligned_img.save(fpath, format="PNG", optimize=True)
        else:
            aligned_img = raw_img

        arr = np.asarray(aligned_img, dtype=np.float32) / 255.0
        aligned_nps[spec["key"]] = arr
        aligned_imgs[spec["key"]] = aligned_img

        # Compute sharpness & contrast stats
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        edge_energy = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy))) * 255.0

        frame_reports.append({
            "key": spec["key"],
            "file": spec["file"],
            "role": spec["role"],
            "chain": spec["chain"],
            "desc": spec["desc"],
            "original_size": list(orig_size),
            "aligned_size": [TARGET_W, TARGET_H],
            "mean_luma": round(float(np.mean(gray) * 255.0), 2),
            "contrast_std": round(float(np.std(gray) * 255.0), 2),
            "edge_sharpness": round(edge_energy, 2),
        })

    # Compute lineage palette similarity along chains
    chain_pairs = [
        ("Frame A", "Frame B", "Shot 1 (A -> B): 拔刀蓄力 -> 雷光疾刺"),
        ("Frame B", "Frame C", "Shot 2 (B -> C): 雷光疾刺 -> 十字架刀"),
        ("Frame A", "Frame D", "Montage (A || D): 主角全身 -> 瞳孔特写"),
        ("Frame D", "Frame E", "Shot 3 (D -> E): 瞳孔特写 -> 横斩化龙"),
        ("Frame E", "Frame F", "Shot 4 (E -> F): 横斩化龙 -> 雷龙终章"),
    ]
    lineage_scores = []
    for ka, kb, label in chain_pairs:
        sim = compute_hist_similarity(aligned_nps[ka], aligned_nps[kb])
        lineage_scores.append({"pair": f"{ka} -> {kb}", "label": label, "palette_correlation_pct": sim})

    # Render 3x2 Storyboard Sheet (2040 x 1080)
    thumb_w, thumb_h = 640, 360
    pad = 24
    header_h = 92
    caption_h = 74
    sheet_w = pad * 4 + thumb_w * 3
    sheet_h = header_h + (thumb_h + caption_h + pad) * 2 + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 16, 24))
    draw = ImageDraw.Draw(sheet)

    def _load_cjk_font(size: int) -> Any:
        for fp in (
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ):
            if Path(fp).exists():
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    font_title = _load_cjk_font(25)
    font_sub = _load_cjk_font(16)
    font_cap = _load_cjk_font(14)

    draw.text(
        (pad, 18),
        "Experiment 06 — 飞碟社风格 15s 雷炎剑戟战：同源基底关键帧链 (Frame A -> B -> C || D -> E -> F)",
        fill=(240, 246, 255),
        font=font_title,
    )
    avg_sim = round(float(np.mean([x["palette_correlation_pct"] for x in lineage_scores])), 2)
    draw.text(
        (pad, 54),
        f"Resolution: 1344x768 (16:9 768p) | 4 FL2VA Storyboards x 5 Seeds = 20 Jobs | Avg Shared-Lineage Palette Consistency: {avg_sim}%",
        fill=(110, 215, 255),
        font=font_sub,
    )

    for idx, spec in enumerate(KEYFRAME_SPECS):
        row = idx // 3
        col = idx % 3
        x0 = pad + col * (thumb_w + pad)
        y0 = header_h + row * (thumb_h + caption_h + pad)

        thumb = aligned_imgs[spec["key"]].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x0, y0))
        border_color = (56, 189, 248) if row == 0 else (244, 114, 182)
        draw.rectangle([x0 - 2, y0 - 2, x0 + thumb_w + 1, y0 + thumb_h + 1], outline=border_color, width=2)

        # Badge overlay
        draw.rectangle([x0 + 8, y0 + 8, x0 + 240, y0 + 36], fill=(10, 14, 22))
        draw.text((x0 + 14, y0 + 12), f"{spec['key']}  [{spec['chain']}]", fill=border_color, font=font_cap)

        cy = y0 + thumb_h + 8
        draw.text((x0, cy), f"{spec['role']}  ({spec['file']})", fill=(226, 232, 240), font=font_cap)
        draw.text((x0, cy + 24), spec["desc"], fill=(148, 163, 184), font=font_cap)

    sheet_path = OUTPUTS_DIR / "keyframe_storyboard_sheet.png"
    sheet.save(sheet_path, format="PNG", optimize=True)

    summary = {
        "target_resolution": [TARGET_W, TARGET_H],
        "keyframes": frame_reports,
        "shared_lineage_consistency": lineage_scores,
        "avg_palette_consistency_pct": avg_sim,
        "storyboard_sheet_path": str(sheet_path),
    }
    report_json = OUTPUTS_DIR / "keyframe_verification.json"
    report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Aligned all 6 keyframes to {TARGET_W}x{TARGET_H} | Avg Palette Consistency: {avg_sim}%")
    print(f"🖼️  Saved Storyboard Contact Sheet: {sheet_path}")
    return summary


def make_ufotable_impact_frame(frame_rgb_u8: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Transforms a transition frame into a high-contrast black-and-white Ufotable Sakuga Impact Frame
    to accentuate violent energy release and mask cross-cut seams.
    """
    gray = (
        0.299 * frame_rgb_u8[:, :, 0].astype(np.float32)
        + 0.587 * frame_rgb_u8[:, :, 1].astype(np.float32)
        + 0.114 * frame_rgb_u8[:, :, 2].astype(np.float32)
    )
    thresh = np.percentile(gray, 58 if invert else 45)
    bw = np.where(gray >= thresh, 255.0, 0.0)
    if invert:
        bw = 255.0 - bw
    out = np.stack([bw, bw, bw], axis=-1).astype(np.uint8)
    return out


def select_best_takes_and_assemble_master(
    batch_dir: Path,
    override_seeds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the 20 rendered MP4 videos in `batch_dir`, selects the highest-scoring (or director-overridden)
    seed per storyboard shot, and executes the frame-accurate 15.00s (360-frame) Director Master Cut.
    """
    import imageio.v2 as imageio
    import soundfile as sf

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = batch_dir / "studio_batch_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing studio_batch_metrics.json in {batch_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    jobs = metrics.get("jobs", [])
    override_seeds = override_seeds or {}

    selected_jobs: Dict[str, Dict[str, Any]] = {}
    all_candidates_by_shot: Dict[str, List[Dict[str, Any]]] = {}

    for j in jobs:
        sid = j["shot_id"]
        q = j.get("quality_metrics") or j
        s_fl2v = float(q.get("S_fl2v_pixel") or 85.0)
        flicker = float(q.get("temporal_flickering_score") or 90.0)
        smooth = float(q.get("motion_smoothness_score") or 85.0)
        motion_mae = float(q.get("dynamic_motion_mae_255") or 10.0)
        av_corr = max(0.0, float(q.get("av_sync_corr") or 0.0)) * 100.0
        prod_q = float(q.get("prod_quality_score") or 80.0)
        # Sakuga Director Score combines prod_quality_score (S_fl2v + smoothness + flicker + sharpness) with AV sync & kinetic energy
        motion_bonus = min(100.0, 60.0 + motion_mae * 1.8)
        sakuga_score = round(
            0.55 * prod_q + 0.20 * smooth + 0.15 * av_corr + 0.10 * motion_bonus,
            2,
        )
        j["sakuga_director_score"] = sakuga_score
        all_candidates_by_shot.setdefault(sid, []).append(j)

    for bp in DIRECTOR_EDIT_BLUEPRINT:
        sid = bp["shot_id"]
        cands = sorted(all_candidates_by_shot.get(sid, []), key=lambda x: x["sakuga_director_score"], reverse=True)
        if not cands:
            raise RuntimeError(f"No rendered jobs found for {sid}")
        if sid in override_seeds:
            target_seed = override_seeds[sid]
            matched = [c for c in cands if int(c["seed"]) == target_seed]
            chosen = matched[0] if matched else cands[0]
        else:
            chosen = cands[0]
        selected_jobs[sid] = chosen

    # Assemble 360 frames (15.00s @ 24fps) and synchronized stereo 32kHz audio with crossfades
    master_frames: List[np.ndarray] = []
    audio_segments: List[ Tuple[np.ndarray, int, float] ] = []
    edit_decision_log: List[Dict[str, Any]] = []

    ffmpeg_bin = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg")
    sample_rate = 32000

    for idx, bp in enumerate(DIRECTOR_EDIT_BLUEPRINT):
        sid = bp["shot_id"]
        chosen = selected_jobs[sid]
        mp4_path = batch_dir / chosen["mp4_file"]
        if not mp4_path.exists():
            raise FileNotFoundError(f"Missing MP4 file: {mp4_path}")

        reader = imageio.get_reader(str(mp4_path))
        clip_frames = [np.asarray(f, dtype=np.uint8) for f in reader]
        reader.close()

        start_f = int(bp["trim_start_frame"])
        keep_f = int(bp["keep_frames"])
        sliced_frames = clip_frames[start_f : start_f + keep_f]
        while len(sliced_frames) < keep_f:
            sliced_frames.append(sliced_frames[-1].copy())

        flash_n = int(bp.get("impact_flash_frames", 0))
        flash_off = int(bp.get("impact_flash_offset", 0))
        if flash_n >= 1 and len(sliced_frames) > flash_off:
            sliced_frames[flash_off] = make_ufotable_impact_frame(sliced_frames[flash_off], invert=True)
        if flash_n >= 2 and len(sliced_frames) > flash_off + 1:
            sliced_frames[flash_off + 1] = make_ufotable_impact_frame(sliced_frames[flash_off + 1], invert=False)

        master_frames.extend(sliced_frames)

        # Extract audio track via ffmpeg to temporary wav
        tmp_wav = OUTPUTS_DIR / f"_tmp_{sid}.wav"
        subprocess.run(
            [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(mp4_path), "-vn", "-ac", "2", "-ar", str(sample_rate), str(tmp_wav)],
            check=True,
        )
        wav_data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
        tmp_wav.unlink(missing_ok=True)

        start_sample = int(round((start_f / FPS) * sr))
        keep_samples = int(round((keep_f / FPS) * sr))
        cf_sec = float(bp.get("crossfade_prev_sec", 0.0))
        cf_samples = int(round(cf_sec * sr))

        # Include extra pre-roll samples for crossfade/J-cut if available
        actual_start = max(0, start_sample - cf_samples)
        pre_roll_got = start_sample - actual_start
        seg_wav = wav_data[actual_start : start_sample + keep_samples]
        if seg_wav.shape[0] < pre_roll_got + keep_samples:
            pad_len = (pre_roll_got + keep_samples) - seg_wav.shape[0]
            seg_wav = np.pad(seg_wav, ((0, pad_len), (0, 0)), mode="constant")

        audio_segments.append((seg_wav, pre_roll_got, keep_samples))
        edit_decision_log.append({
            "order": idx + 1,
            "shot_id": sid,
            "selected_job_id": chosen["job_id"],
            "selected_seed": chosen["seed"],
            "sakuga_director_score": chosen["sakuga_director_score"],
            "prod_quality_score": (chosen.get("quality_metrics") or chosen).get("prod_quality_score"),
            "S_fl2v_pixel": (chosen.get("quality_metrics") or chosen).get("S_fl2v_pixel"),
            "motion_smoothness_score": (chosen.get("quality_metrics") or chosen).get("motion_smoothness_score"),
            "av_sync_corr": (chosen.get("quality_metrics") or chosen).get("av_sync_corr"),
            "chain_mode": bp["chain_mode"],
            "trim_start_frame": start_f,
            "kept_frames": keep_f,
            "duration_sec": round(keep_f / FPS, 2),
            "impact_flash_frames": flash_n,
            "crossfade_prev_sec": cf_sec,
            "note": bp["note"],
        })

    # Stitch audio with equal-power cosine crossfade
    total_frames = len(master_frames)
    total_samples = int(round((total_frames / FPS) * sample_rate))
    master_audio = np.zeros((total_samples, 2), dtype=np.float32)
    cursor = 0
    for idx, (seg_wav, pre_roll, keep_s) in enumerate(audio_segments):
        if idx == 0 or pre_roll == 0:
            body = seg_wav[pre_roll : pre_roll + keep_s]
            end_pos = min(total_samples, cursor + body.shape[0])
            master_audio[cursor:end_pos] = body[: end_pos - cursor]
            cursor += keep_s
        else:
            # Crossfade region [cursor - pre_roll : cursor]
            cf_start = max(0, cursor - pre_roll)
            cf_len = cursor - cf_start
            if cf_len > 0:
                t = np.linspace(0.0, 0.5 * math.pi, cf_len, dtype=np.float32)[:, None]
                fade_out = np.cos(t)
                fade_in = np.sin(t)
                master_audio[cf_start:cursor] = (
                    master_audio[cf_start:cursor] * fade_out + seg_wav[:cf_len] * fade_in
                )
            body = seg_wav[pre_roll : pre_roll + keep_s]
            end_pos = min(total_samples, cursor + body.shape[0])
            master_audio[cursor:end_pos] = body[: end_pos - cursor]
            cursor += keep_s

    out_mp4 = OUTPUTS_DIR / "director_master_15s.mp4"
    tmp_v = OUTPUTS_DIR / "_tmp_master_v.mp4"
    tmp_a = OUTPUTS_DIR / "_tmp_master_a.wav"

    writer = imageio.get_writer(
        str(tmp_v), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16
    )
    for f in master_frames:
        writer.append_data(f)
    writer.close()

    sf.write(str(tmp_a), np.clip(master_audio, -1.0, 1.0), sample_rate)
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp_v),
            "-i",
            str(tmp_a),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out_mp4),
        ],
        check=True,
    )
    tmp_v.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)

    edl_summary = {
        "master_video_path": str(out_mp4),
        "total_frames": total_frames,
        "fps": FPS,
        "exact_duration_sec": round(total_frames / FPS, 2),
        "edit_decision_list": edit_decision_log,
        "all_candidates_ranking": {
            sid: [
                {
                    "job_id": c["job_id"],
                    "seed": c["seed"],
                    "sakuga_director_score": c["sakuga_director_score"],
                    "prod_quality_score": (c.get("quality_metrics") or c).get("prod_quality_score"),
                    "S_fl2v_pixel": (c.get("quality_metrics") or c).get("S_fl2v_pixel"),
                    "motion_smoothness_score": (c.get("quality_metrics") or c).get("motion_smoothness_score"),
                    "dynamic_motion_mae_255": (c.get("quality_metrics") or c).get("dynamic_motion_mae_255"),
                    "av_sync_corr": (c.get("quality_metrics") or c).get("av_sync_corr"),
                }
                for c in sorted(cands, key=lambda x: x["sakuga_director_score"], reverse=True)
            ]
            for sid, cands in all_candidates_by_shot.items()
        },
    }
    edl_path = OUTPUTS_DIR / "director_edit_decision_list.json"
    edl_path.write_text(json.dumps(edl_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render 4x5 Seed Gacha Contact Sheet (all 20 takes at midpoint frame)
    def _load_cjk_font(size: int) -> Any:
        for fp in (
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ):
            if Path(fp).exists():
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    f_hdr = _load_cjk_font(22)
    f_sub = _load_cjk_font(14)
    f_cell = _load_cjk_font(12)

    cw, ch = 360, 202
    pad = 16
    row_lbl_h = 34
    cap_h = 46
    top_h = 84
    mat_w = pad + 5 * (cw + pad)
    mat_h = top_h + 4 * (row_lbl_h + ch + cap_h + pad) + pad
    mat_img = Image.new("RGB", (mat_w, mat_h), (12, 15, 23))
    mdraw = ImageDraw.Draw(mat_img)
    mdraw.text(
        (pad, 16),
        "Experiment 06 — 4 分镜 × 5 Seeds (20 条视频) 导演抽卡优选矩阵 (Midpoint Action Frame)",
        fill=(240, 246, 255),
        font=f_hdr,
    )
    mdraw.text(
        (pad, 48),
        "金色边框 = 导演选中最优卡 (Selected Winner) | 评分综合 S_fl2v 首尾帧锁死度 + 运动平滑度 + 音画同步相关系数 (AV Sync) + 动能幅度",
        fill=(250, 204, 21),
        font=f_sub,
    )

    for r_idx, bp in enumerate(DIRECTOR_EDIT_BLUEPRINT):
        sid = bp["shot_id"]
        chosen_jid = selected_jobs[sid]["job_id"]
        cands_by_seed = sorted(all_candidates_by_shot.get(sid, []), key=lambda x: int(x["seed"]))
        ry = top_h + r_idx * (row_lbl_h + ch + cap_h + pad)
        mdraw.text(
            (pad, ry + 4),
            f"Shot {r_idx+1}: {sid}  |  Selected Winner: {selected_jobs[sid]['job_id']} (seed={selected_jobs[sid]['seed']}, Score={selected_jobs[sid]['sakuga_director_score']})",
            fill=(125, 211, 252),
            font=f_sub,
        )
        for c_idx, cand in enumerate(cands_by_seed):
            cx = pad + c_idx * (cw + pad)
            cy = ry + row_lbl_h
            mp4_p = batch_dir / cand["mp4_file"]
            rdr = imageio.get_reader(str(mp4_p))
            n_f = int(cand.get("num_frames", 90))
            mid_idx = max(0, n_f // 2)
            try:
                mid_f = np.asarray(rdr.get_data(mid_idx), dtype=np.uint8)
            except Exception:
                mid_f = np.asarray(rdr.get_data(0), dtype=np.uint8)
            rdr.close()
            pil_cell = Image.fromarray(mid_f).resize((cw, ch), Image.Resampling.LANCZOS)
            mat_img.paste(pil_cell, (cx, cy))

            is_win = cand["job_id"] == chosen_jid
            b_col = (250, 204, 21) if is_win else (51, 65, 85)
            b_w = 3 if is_win else 1
            mdraw.rectangle([cx - b_w, cy - b_w, cx + cw + b_w - 1, cy + ch + b_w - 1], outline=b_col, width=b_w)
            if is_win:
                mdraw.rectangle([cx + 6, cy + 6, cx + 148, cy + 28], fill=(250, 204, 21))
                mdraw.text((cx + 12, cy + 9), f"★ WINNER seed={cand['seed']}", fill=(15, 23, 42), font=f_cell)
            else:
                mdraw.rectangle([cx + 6, cy + 6, cx + 104, cy + 28], fill=(15, 23, 42))
                mdraw.text((cx + 12, cy + 9), f"seed={cand['seed']}", fill=(203, 213, 225), font=f_cell)

            q = cand.get("quality_metrics") or cand
            mdraw.text(
                (cx, cy + ch + 5),
                f"Director={cand['sakuga_director_score']} | ProdQ={q.get('prod_quality_score')} | S_fl2v={q.get('S_fl2v_pixel')}",
                fill=(250, 204, 21) if is_win else (226, 232, 240),
                font=f_cell,
            )
            mdraw.text(
                (cx, cy + ch + 23),
                f"Smooth={q.get('motion_smoothness_score')} | AV_Sync={q.get('av_sync_corr')} | Motion={q.get('dynamic_motion_mae_255')}",
                fill=(148, 163, 184),
                font=f_cell,
            )

    gacha_sheet_path = OUTPUTS_DIR / "seed_gacha_4x5_matrix.png"
    mat_img.save(gacha_sheet_path, format="PNG", optimize=True)

    # Render Transition Seam & Impact Frame Audit Strip (8 key frames from the 360-frame master cut)
    audit_indices = [
        (0, "f000 (0.00s): Shot 1 Start (s1[6], Frame A Dash)"),
        (83, "f083 (3.46s): Shot 1 End (s1[89] == Frame B)"),
        (84, "f084 (3.50s): Shot 2 Start (s2[1], Dropped f0!)"),
        (184, "f184 (7.67s): Shot 2 Cut on Peak Shockwave"),
        (185, "f185 (7.71s): Shot 3 Montage Cut (s3[13] Eye)"),
        (244, "f244 (10.17s): Shot 3 End (s3[72] == Frame E)"),
        (245, "f245 (10.21s): Shot 4 Start (s4[1], Seamless E!)"),
        (247, "f247 (10.29s): Shot 4 Ufotable B&W Impact Flash"),
    ]
    aw, ah = 440, 248
    audit_w = pad + 4 * (aw + pad)
    audit_h = top_h + 2 * (ah + 42 + pad) + pad
    audit_img = Image.new("RGB", (audit_w, audit_h), (12, 15, 23))
    adraw = ImageDraw.Draw(audit_img)
    adraw.text(
        (pad, 16),
        "Experiment 06 — 15.00s (360 帧 @ 24fps) 导演精剪成片切镜接缝与黑白闪帧核验图",
        fill=(240, 246, 255),
        font=f_hdr,
    )
    adraw.text(
        (pad, 48),
        "验证点：f083->f084 链式去重帧无缝衔接 (B->C) | f184->f185 跨机位瞳孔快切 (J-Cut) | f244->f245 飞碟社黑白冲击闪帧 (Impact Frame)",
        fill=(56, 189, 248),
        font=f_sub,
    )
    for idx, (f_idx, label) in enumerate(audit_indices):
        r, c = idx // 4, idx % 4
        ax = pad + c * (aw + pad)
        ay = top_h + r * (ah + 42 + pad)
        f_pil = Image.fromarray(master_frames[f_idx]).resize((aw, ah), Image.Resampling.LANCZOS)
        audit_img.paste(f_pil, (ax, ay))
        adraw.rectangle([ax - 2, ay - 2, ax + aw + 1, ay + ah + 1], outline=(56, 189, 248), width=2)
        adraw.text((ax, ay + ah + 8), label, fill=(226, 232, 240), font=f_sub)

    audit_strip_path = OUTPUTS_DIR / "master_cut_transition_audit.png"
    audit_img.save(audit_strip_path, format="PNG", optimize=True)

    print(f"🖼️  [Seed Gacha Matrix] {gacha_sheet_path}")
    print(f"🖼️  [Transition Audit ] {audit_strip_path}")
    print(f"🎬 [Director Master Cut Complete] {out_mp4} ({total_frames} frames = {total_frames/FPS:.2f}s @ {FPS}fps)")
    return edl_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 06 Sakuga FL2VA Director Toolkit")
    parser.add_argument("--align-keyframes", action="store_true", help="Align Frame A..F to 1344x768 and render storyboard sheet")
    parser.add_argument("--select-and-cut", type=Path, default=None, help="Path to completed batch directory to select best seeds and cut 15s master")
    parser.add_argument("--override-seeds", type=str, default="", help="Optional manual seed overrides, e.g. 'shot_01_thunder_dash=108,shot_02_blade_clash_shockwave=777'")
    args = parser.parse_args()

    if args.align_keyframes:
        align_and_verify_keyframes()
    if args.select_and_cut:
        overrides: Dict[str, int] = {}
        if args.override_seeds.strip():
            for kv in args.override_seeds.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    overrides[k.strip()] = int(v.strip())
        select_best_takes_and_assemble_master(args.select_and_cut, override_seeds=overrides)


if __name__ == "__main__":
    main()
