#!/usr/bin/env python3
"""
Local 0-CU Planner, Asset Validator, Conditioning Deduplication Analyzer, and Google Drive Pre-Stager
(`studio/api/local_planner.py`) for the MiniMax H3-Base-FL2VA Production Studio Backend.
Supports flexible Aspect Ratio (16:9, 9:16, 1:1, 4:3, 3:4, 21:9), Quality (512p, 768p, 1080p),
and Duration (17k+5 frame grid @ 24fps), as well as FL2VA / I2VA / T2VA conditioning modes.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .manifest_schema import (
        LOCKED_RUN12_CONFIG,
        BatchManifest,
        compute_conditioning_hash,
        estimate_tokens_and_denoise_sec,
        resolve_dimensions,
        snap_frames_17k_plus_5,
    )
except ImportError:
    from manifest_schema import (
        LOCKED_RUN12_CONFIG,
        BatchManifest,
        compute_conditioning_hash,
        estimate_tokens_and_denoise_sec,
        resolve_dimensions,
        snap_frames_17k_plus_5,
    )

_SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_ROOT = _SCRIPT_DIR.parent
REPO_ROOT = STUDIO_ROOT.parent

TOKEN_PATH = Path.home() / ".config/colab-cli/token.json"
_QUOTA_PROJECT_FILE = Path.home() / ".config/colab-cli/quota_project"
DEFAULT_QUOTA_PROJECT = (
    os.environ.get("STUDIO_QUOTA_PROJECT")
    or os.environ.get("GCP_QUOTA_PROJECT")
    or (_QUOTA_PROJECT_FILE.read_text(encoding="utf-8").strip() if _QUOTA_PROJECT_FILE.exists() else "")
)
G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0
EST_COLD_START_SEC = 160.64
EST_VLM_ENCODE_PER_UNIQUE_SHOT_SEC = 3.58
EST_GDRIVE_EGRESS_STAGE1_SEC = 1.85


def resolve_or_seed_default_assets(manifest_dir: Path) -> None:
    """
    Ensures default test assets (`assets/shot01_first.png`, `assets/shot01_last.png`) exist
    by copying from Experiment 03/05's verified 1344x768 first/last frames if not yet present.
    """
    assets_dir = manifest_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    exp03_out = REPO_ROOT / "experiments" / "03-g4-sol-attn-pareto-ablation" / "outputs"
    f1 = assets_dir / "shot01_first.png"
    f2 = assets_dir / "shot01_last.png"
    if not f1.exists() and (exp03_out / "input_first_frame.png").exists():
        shutil.copy2(exp03_out / "input_first_frame.png", f1)
    if not f2.exists() and (exp03_out / "input_last_frame.png").exists():
        shutil.copy2(exp03_out / "input_last_frame.png", f2)


def resolve_shot_geometry(manifest: BatchManifest, shot: Any) -> Tuple[Tuple[int, int], int, float]:
    ar = shot.aspect_ratio or manifest.aspect_ratio
    qual = shot.quality or manifest.quality
    res = resolve_dimensions(aspect_ratio=ar, quality=qual, explicit_resolution=shot.resolution or (manifest.resolution if not (shot.aspect_ratio or shot.quality) else None))
    if shot.num_frames is not None:
        n_frames = snap_frames_17k_plus_5(int(shot.num_frames), fps=manifest.fps, is_seconds=False)
    elif shot.duration_sec is not None:
        n_frames = snap_frames_17k_plus_5(float(shot.duration_sec), fps=manifest.fps, is_seconds=True)
    else:
        n_frames = manifest.num_frames
    dur_sec = round(n_frames / manifest.fps, 2)
    return res, n_frames, dur_sec


def analyze_and_pack_batch(
    manifest_path: Path,
    output_tar_path: Path,
) -> Dict[str, Any]:
    manifest_dir = manifest_path.parent
    resolve_or_seed_default_assets(manifest_dir)
    manifest = BatchManifest.from_json_file(manifest_path)

    shot_map = {s.shot_id: s for s in manifest.storyboards}
    unique_hashes: Dict[str, List[str]] = {}
    shot_perf_map: Dict[str, Dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "input_jobs"
        (stage_dir / "assets").mkdir(parents=True, exist_ok=True)

        serialized_shots = []
        for s in manifest.storyboards:
            s_res, s_frames, s_dur = resolve_shot_geometry(manifest, s)
            b_first = b""
            b_last = b""
            rel_first = ""
            rel_last = ""

            if s.first_frame:
                p_first = (manifest_dir / s.first_frame).resolve() if not Path(s.first_frame).is_absolute() else Path(s.first_frame)
                if not p_first.exists():
                    raise FileNotFoundError(f"Missing first_frame for {s.shot_id}: {p_first}")
                b_first = p_first.read_bytes()
                rel_first = f"assets/{s.shot_id}_first.png"
                (stage_dir / rel_first).write_bytes(b_first)

            if s.last_frame:
                p_last = (manifest_dir / s.last_frame).resolve() if not Path(s.last_frame).is_absolute() else Path(s.last_frame)
                if not p_last.exists():
                    raise FileNotFoundError(f"Missing last_frame for {s.shot_id}: {p_last}")
                b_last = p_last.read_bytes()
                rel_last = f"assets/{s.shot_id}_last.png"
                (stage_dir / rel_last).write_bytes(b_last)

            c_hash = compute_conditioning_hash(
                prompt=s.prompt,
                negative_prompt=s.negative_prompt,
                first_frame_bytes=b_first,
                last_frame_bytes=b_last,
                resolution=s_res,
                num_frames=s_frames,
            )
            s.cond_hash = c_hash
            unique_hashes.setdefault(c_hash, []).append(s.shot_id)

            perf_est = estimate_tokens_and_denoise_sec(s_res[0], s_res[1], s_frames, steps=manifest.steps)
            shot_perf_map[s.shot_id] = {
                "resolution": list(s_res),
                "num_frames": s_frames,
                "duration_sec": s_dur,
                **perf_est,
            }

            mode_str = "FL2VA" if (rel_first and rel_last) else ("I2VA" if rel_first else "T2VA")
            serialized_shots.append({
                "shot_id": s.shot_id,
                "title": s.title,
                "mode": mode_str,
                "prompt": s.prompt,
                "negative_prompt": s.negative_prompt,
                "first_frame": rel_first,
                "last_frame": rel_last,
                "resolution": list(s_res),
                "num_frames": s_frames,
                "duration_sec": s_dur,
                "cond_hash": c_hash,
                "total_joint_tokens": perf_est["total_joint_tokens"],
            })

        job_breakdown = []
        total_denoise_sec = 0.0
        total_e2e_render_sec = 0.0
        for idx, j in enumerate(manifest.jobs, 1):
            s_obj = shot_map[j.shot_id]
            s_perf = shot_perf_map[j.shot_id]
            denoise_s = s_perf["est_denoise_sec"]
            e2e_s = s_perf["est_e2e_sec"]
            total_denoise_sec += denoise_s
            total_e2e_render_sec += e2e_s
            job_breakdown.append({
                "index": idx,
                "job_id": j.job_id,
                "shot_id": j.shot_id,
                "cond_hash": s_obj.cond_hash,
                "seed": j.seed,
                "resolution": s_perf["resolution"],
                "num_frames": s_perf["num_frames"],
                "duration_sec": s_perf["duration_sec"],
                "total_joint_tokens": s_perf["total_joint_tokens"],
                "locked_config": LOCKED_RUN12_CONFIG["config_id"],
                "est_denoise_sec": denoise_s,
                "est_e2e_sec": e2e_s,
                "est_marginal_cu": round(e2e_s * G4_CU_RATE_PER_SEC, 4),
                "note": j.note,
            })

        resolved_manifest = {
            "project": manifest.project,
            "batch_id": manifest.batch_id,
            "title": manifest.title,
            "locked_inference_config": LOCKED_RUN12_CONFIG,
            "aspect_ratio": manifest.aspect_ratio,
            "quality": manifest.quality,
            "resolution": list(manifest.resolution),
            "duration_sec": manifest.duration_sec,
            "num_frames": manifest.num_frames,
            "fps": manifest.fps,
            "steps": manifest.steps,
            "cfg": manifest.cfg,
            "resize_mode": manifest.resize_mode,
            "storyboards": serialized_shots,
            "jobs": job_breakdown,
        }
        resolved_json_text = json.dumps(resolved_manifest, indent=2, ensure_ascii=False)
        (stage_dir / "batch_manifest_resolved.json").write_text(resolved_json_text, encoding="utf-8")

        output_tar_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-cf", str(output_tar_path), "-C", str(stage_dir), "."], check=True)

    tar_bytes = output_tar_path.stat().st_size
    tar_sha256 = hashlib.sha256(output_tar_path.read_bytes()).hexdigest()

    num_jobs = len(manifest.jobs)
    num_unique_conds = len(unique_hashes)
    naive_cond_sec = num_jobs * 5.13
    staged_dedup_cond_sec = 3.20 + num_unique_conds * EST_VLM_ENCODE_PER_UNIQUE_SHOT_SEC
    cond_saved_sec = max(0.0, naive_cond_sec - staged_dedup_cond_sec)

    est_total_session_sec = (
        EST_COLD_START_SEC
        + staged_dedup_cond_sec
        + total_e2e_render_sec
        + EST_GDRIVE_EGRESS_STAGE1_SEC
    )
    est_total_session_cu = est_total_session_sec * G4_CU_RATE_PER_SEC
    est_amortized_cu_per_video = est_total_session_cu / max(1, num_jobs)
    est_core_render_share_pct = (
        (staged_dedup_cond_sec + total_e2e_render_sec) / est_total_session_sec * 100.0
    )

    return {
        "project": manifest.project,
        "batch_id": manifest.batch_id,
        "title": manifest.title,
        "locked_config": LOCKED_RUN12_CONFIG["config_id"],
        "aspect_ratio": manifest.aspect_ratio,
        "quality": manifest.quality,
        "resolution": list(manifest.resolution),
        "num_frames": manifest.num_frames,
        "duration_sec": manifest.duration_sec,
        "fps": manifest.fps,
        "num_storyboards": len(manifest.storyboards),
        "num_unique_conditionings": num_unique_conds,
        "num_jobs": num_jobs,
        "cond_dedup_hits": num_jobs - num_unique_conds,
        "naive_vlm_cond_sec": round(naive_cond_sec, 2),
        "staged_dedup_vlm_cond_sec": round(staged_dedup_cond_sec, 2),
        "cond_time_saved_sec": round(cond_saved_sec, 2),
        "est_total_render_sec": round(total_e2e_render_sec, 2),
        "est_total_session_sec": round(est_total_session_sec, 2),
        "est_total_session_cu": round(est_total_session_cu, 4),
        "est_amortized_cu_per_video": round(est_amortized_cu_per_video, 4),
        "est_core_render_share_pct": round(est_core_render_share_pct, 2),
        "input_tar_path": str(output_tar_path),
        "input_tar_bytes": tar_bytes,
        "input_tar_sha256": tar_sha256,
        "jobs": job_breakdown,
    }


def prestage_input_tar_to_gdrive_0cu(
    tar_path: Path,
    quota_project: str = DEFAULT_QUOTA_PROJECT,
) -> Dict[str, Any]:
    """
    Uploads `input_jobs.tar` to Google Drive v3 API from the local Mac BEFORE starting any Colab GPU VM (0.00 CU!).
    """
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    for attempt in range(5):
        try:
            creds.refresh(Request())
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.8 * (attempt + 1))
    TOKEN_PATH.write_text(creds.to_json())

    data = tar_path.read_bytes()
    sz = len(data)
    md5 = hashlib.md5(data).hexdigest()
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            r = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum",
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "X-Goog-User-Project": quota_project,
                    "Content-Type": "application/x-tar",
                    "Content-Length": str(sz),
                },
                data=data,
                timeout=120,
            )
            r.raise_for_status()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.8 * (attempt + 1))
    elapsed = time.perf_counter() - t0
    info = r.json()
    assert int(info["size"]) == sz and info.get("md5Checksum") == md5
    return {
        "input_file_id": info["id"],
        "bytes": sz,
        "md5": md5,
        "upload_sec": round(elapsed, 3),
        "gpu_cu_consumed": 0.0,
    }


def print_plan_summary(plan: Dict[str, Any], batch_dir: Path | None = None) -> None:
    print("=" * 96)
    print(f"📋 [Studio 0-CU Planner] Project: {plan.get('project', 'default')} | Batch: {plan['title']}")
    if batch_dir is not None:
        print(f"📂 Batch Output Directory: {batch_dir} (saved: {batch_dir / 'manifest.json'})")
    print(
        f"🔒 Locked Config: {plan['locked_config']} | Default Geometry: "
        f"{plan['resolution'][0]}×{plan['resolution'][1]} ({plan['aspect_ratio']}, {plan['quality']}) × "
        f"{plan['num_frames']}f ({plan['duration_sec']:.2f}s @ {plan['fps']}fps)"
    )
    print("=" * 96)
    print(
        f"  • Storyboards: {plan['num_storyboards']} | Unique Conditionings: {plan['num_unique_conditionings']} | "
        f"Total Render Jobs: {plan['num_jobs']} (Cache Hits: {plan['cond_dedup_hits']} jobs = 0.0s VLM encode!)"
    )
    print(
        f"  • Packed Input Bundle: {plan['input_tar_path']} ({plan['input_tar_bytes']/1e6:.2f} MB, "
        f"sha256={plan['input_tar_sha256'][:12]}...)"
    )
    print("-" * 96)
    for j in plan["jobs"]:
        res_str = f"{j['resolution'][0]}x{j['resolution'][1]}x{j['num_frames']}f"
        print(
            f"  [Job {j['index']:02d}] {j['job_id']:<22} | {res_str:<14} (N={j['total_joint_tokens']}) | "
            f"seed={j['seed']:<5} | est={j['est_e2e_sec']:.1f}s ({j['est_marginal_cu']:.4f} CU)"
        )
    print("-" * 96)
    print(
        f"  📊 Estimated G4 Session Total: {plan['est_total_session_sec']:.1f}s "
        f"({plan['est_total_session_sec']/60:.2f} min) = {plan['est_total_session_cu']:.4f} CU"
    )
    print(
        f"  💎 Amortized Cost per Video:   {plan['est_amortized_cu_per_video']:.4f} CU/video | "
        f"Core Render Share: {plan['est_core_render_share_pct']:.1f}%"
    )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio Local 0-CU Batch Planner & Pre-Stager")
    parser.add_argument("-m", "--manifest", type=Path, default=_SCRIPT_DIR / "sample_single_manifest.json")
    parser.add_argument("-o", "--output-tar", type=Path, default=STUDIO_ROOT / "outputs" / "input_jobs.tar")
    parser.add_argument("--prestage-gdrive", action="store_true")
    args = parser.parse_args()

    plan = analyze_and_pack_batch(args.manifest, args.output_tar)
    print_plan_summary(plan)
    if args.prestage_gdrive:
        res = prestage_input_tar_to_gdrive_0cu(args.output_tar)
        print(
            f"☁️  [0-CU Pre-Staged to Google Drive] input_file_id={res['input_file_id']} "
            f"({res['bytes']/1e6:.2f} MB in {res['upload_sec']}s, 0.0000 CU)"
        )


if __name__ == "__main__":
    main()
