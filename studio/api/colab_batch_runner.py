#!/usr/bin/env python3
"""
Production Colab Blackwell G4 Runner & Single/Batch Experiment CLI (`studio/api/colab_batch_runner.py`).

Outputs are organized by project and auto-incrementing batch number starting from 1:
  studio/outputs/<project_name>/<batch_seq>/
    ├── manifest.json                 # Complete snapshot of the input manifest & prompts
    ├── assets/                       # Input reference frames referenced by manifest.json
    ├── by_storyboard/                # Rendered MP4 videos grouped by shot_id (no duplicates in root)
    │   └── <shot_id>/
    │       └── job_01_...mp4
    ├── manifest_results.json         # Per-job quality & seed ranking summary
    └── studio_batch_metrics.json     # Full hardware, kernel & CU telemetry
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COLAB_PYTHON = Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"
try:
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    if COLAB_PYTHON.exists() and Path(sys.executable).resolve() != COLAB_PYTHON.resolve():
        os.execv(str(COLAB_PYTHON), [str(COLAB_PYTHON), *sys.argv])
    raise

try:
    from .local_planner import (
        DEFAULT_QUOTA_PROJECT,
        G4_CU_RATE_PER_HOUR,
        G4_CU_RATE_PER_SEC,
        STUDIO_ROOT,
        TOKEN_PATH,
        analyze_and_pack_batch,
        prestage_input_tar_to_gdrive_0cu,
        print_plan_summary,
        resolve_or_seed_default_assets,
    )
    from .manifest_schema import resolve_dimensions, snap_frames_17k_plus_5
except ImportError:
    from local_planner import (
        DEFAULT_QUOTA_PROJECT,
        G4_CU_RATE_PER_HOUR,
        G4_CU_RATE_PER_SEC,
        STUDIO_ROOT,
        TOKEN_PATH,
        analyze_and_pack_batch,
        prestage_input_tar_to_gdrive_0cu,
        print_plan_summary,
        resolve_or_seed_default_assets,
    )
    from manifest_schema import resolve_dimensions, snap_frames_17k_plus_5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = STUDIO_ROOT / "outputs"
COLAB_BIN = shutil.which("colab") or str(Path.home() / ".local/share/uv/tools/google-colab-cli/bin/colab")
SESSIONS_PATH = Path.home() / ".config/colab-cli/sessions.json"
DEFAULT_SESSION_NAME = "h3-studio-g4"


def get_fresh_credentials() -> Credentials:
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
    return creds


def fetch_ccu_info(creds: Credentials) -> Dict[str, Any]:
    for attempt in range(4):
        try:
            ts_local = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            r = requests.get(
                "https://colab.research.google.com/tun/m/ccu-info?authuser=0",
                headers={"Authorization": f"Bearer {creds.token}", "Accept": "application/json"},
                timeout=20,
            )
            r.raise_for_status()
            text = r.text
            if text.startswith(")]}'"):
                text = text.split("\n", 1)[1]
            return {
                "local_timestamp": ts_local,
                "utc_header_date": r.headers.get("Date", ""),
                "http_status": r.status_code,
                "response_body": json.loads(text),
            }
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("Unreachable")


def stage2_local_parallel_pull_0cu(
    file_id: str,
    local_dest_path: Path,
    expected_raw_bytes: int,
    expected_sha256: str,
    quota_project: str = DEFAULT_QUOTA_PROJECT,
    workers: int = 8,
    chunk_mib: int = 8,
) -> Dict[str, Any]:
    creds = get_fresh_credentials()
    chunk_size = chunk_mib * 1024 * 1024
    ranges: List[Tuple[int, int]] = [
        (s, min(s + chunk_size - 1, expected_raw_bytes - 1))
        for s in range(0, expected_raw_bytes, chunk_size)
    ]

    def fetch_range(start_end: Tuple[int, int]) -> Tuple[int, bytes]:
        start, end = start_end
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "X-Goog-User-Project": quota_project,
            "Range": f"bytes={start}-{end}",
        }
        for attempt in range(4):
            try:
                r = requests.get(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                    headers=headers,
                    timeout=90,
                )
                r.raise_for_status()
                return start, r.content
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(0.4 * (attempt + 1))
        raise RuntimeError("Unreachable")

    t0 = time.perf_counter()
    buf = bytearray(expected_raw_bytes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for s, data in ex.map(fetch_range, ranges):
            buf[s : s + len(data)] = data

    local_dest_path.parent.mkdir(parents=True, exist_ok=True)
    local_dest_path.write_bytes(buf)
    elapsed = time.perf_counter() - t0
    sha_actual = hashlib.sha256(buf).hexdigest()
    if sha_actual != expected_sha256:
        raise ValueError(f"Stage 2 SHA-256 mismatch: {sha_actual} != {expected_sha256}")
    return {
        "file_id": file_id,
        "local_dest": str(local_dest_path),
        "raw_bytes": expected_raw_bytes,
        "stage2_sec": round(elapsed, 4),
        "stage2_mb_s": round((expected_raw_bytes / 1e6) / max(elapsed, 1e-6), 2),
        "verified_sha256": True,
        "gpu_cu_consumed": 0.0,
    }


def delete_gdrive_file_quiet(file_id: str, quota_project: str = DEFAULT_QUOTA_PROJECT) -> bool:
    try:
        creds = get_fresh_credentials()
        r = requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": quota_project},
            timeout=20,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def _is_batch_dir_executed(batch_dir: Path) -> bool:
    """Returns True if this batch directory has already started or completed a Colab run."""
    if not batch_dir.exists():
        return False
    if (batch_dir / "manifest_results.json").exists() or (batch_dir / "studio_batch_raw.log").exists():
        return True
    if any(batch_dir.rglob("*.mp4")):
        return True
    return False


def _extract_pending_stage1_from_log(batch_dir: Path) -> Optional[Dict[str, Any]]:
    """
    If Stage 1 GDrive upload completed on Colab and G4 was already terminated,
    but local Stage 2 download was interrupted (e.g. transient local SSL/network blip),
    returns the parsed STUDIO_STAGE1_EGRESS_JSON dict so Step 4 can resume at 0.00 CU.
    """
    if not batch_dir.exists() or (batch_dir / "manifest_results.json").exists():
        return None
    raw_log = batch_dir / "studio_batch_raw.log"
    if not raw_log.exists():
        return None
    for line in raw_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("STUDIO_STAGE1_EGRESS_JSON:"):
            try:
                return json.loads(line.split("STUDIO_STAGE1_EGRESS_JSON:", 1)[1].strip())
            except Exception:
                return None
    return None


def allocate_project_batch_dir(
    outputs_root: Path,
    project_name: str,
    force_new: bool = False,
) -> Tuple[Path, int]:
    """
    Allocates `studio/outputs/<project_name>/<batch_seq>/` where `batch_seq` starts from 1 and increments.
    If the latest numbered directory was only created by a `--dry-run` (not yet executed on Colab)
    and `force_new=False`, reuses that sequence number so dry-runs do not skip sequence numbers.
    """
    safe_project = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in project_name.strip()) or "default"
    project_dir = outputs_root / safe_project
    project_dir.mkdir(parents=True, exist_ok=True)

    existing_seqs = sorted(
        int(d.name) for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit() and int(d.name) >= 1
    )
    if not existing_seqs:
        next_seq = 1
    else:
        latest_seq = existing_seqs[-1]
        latest_dir = project_dir / str(latest_seq)
        if not force_new and not _is_batch_dir_executed(latest_dir):
            next_seq = latest_seq
        else:
            next_seq = latest_seq + 1

    batch_dir = project_dir / str(next_seq)
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, next_seq


def snapshot_manifest_into_batch_dir(
    source_manifest_path: Path,
    batch_dir: Path,
    project_name: str,
    batch_seq: int,
) -> Path:
    """
    Saves the input manifest as `<batch_dir>/manifest.json` and copies any local reference frame images
    into `<batch_dir>/assets/` so `<batch_dir>/manifest.json` is 100% self-contained.
    """
    src_dir = source_manifest_path.parent
    resolve_or_seed_default_assets(src_dir)
    raw = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    raw["project"] = project_name
    raw["batch_seq"] = batch_seq
    raw["batch_id"] = f"{project_name}_batch_{batch_seq}"

    assets_dir = batch_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    used_asset_names: Dict[str, Path] = {}

    for s in raw.get("storyboards", []):
        sid = s.get("shot_id", "shot")
        for key in ("first_frame", "last_frame"):
            val = s.get(key, "")
            if val:
                src_img = Path(val) if Path(val).is_absolute() else (src_dir / val).resolve()
                if not src_img.exists():
                    for candidate in (Path.cwd() / val, _SCRIPT_DIR / val):
                        if candidate.resolve().exists():
                            src_img = candidate.resolve()
                            break
                if src_img.exists():
                    base_name = src_img.name
                    if base_name in used_asset_names and used_asset_names[base_name] != src_img.resolve():
                        base_name = f"{sid}_{key}_{src_img.name}"
                    used_asset_names[base_name] = src_img.resolve()
                    dst_rel = f"assets/{base_name}"
                    dst_img = batch_dir / dst_rel
                    if src_img.resolve() != dst_img.resolve():
                        shutil.copy2(src_img, dst_img)
                    s[key] = dst_rel

    dest_manifest_path = batch_dir / "manifest.json"
    dest_manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest_manifest_path


def build_quick_manifest_dict(
    project: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    quality: str = "768p",
    duration_sec: float = 5.0,
    num_frames: Optional[int] = None,
    first_frame: str = "",
    last_frame: str = "",
    seeds: List[int] = [42],
    resize_mode: str = "cover",
) -> Dict[str, Any]:
    resolve_or_seed_default_assets(_SCRIPT_DIR)
    w, h = resolve_dimensions(aspect_ratio=aspect_ratio, quality=quality)
    frames = (
        snap_frames_17k_plus_5(num_frames, is_seconds=False)
        if num_frames is not None
        else snap_frames_17k_plus_5(duration_sec, is_seconds=True)
    )
    shot_id = f"shot_01_{aspect_ratio.replace(':', 'x')}_{quality}_{frames}f"
    return {
        "project": project,
        "batch_id": f"{project}_quick",
        "title": f"Studio Run ({aspect_ratio} {quality} {w}x{h}, {frames}f / {frames/24:.2f}s, {len(seeds)} seed(s))",
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "resolution": [w, h],
        "duration_sec": round(frames / 24.0, 2),
        "num_frames": frames,
        "fps": 24,
        "steps": 8,
        "cfg": 1.0,
        "resize_mode": resize_mode,
        "storyboards": [
            {
                "shot_id": shot_id,
                "title": f"Shot 01 ({w}x{h} x {frames}f)",
                "prompt": prompt,
                "negative_prompt": "",
                "first_frame": first_frame,
                "last_frame": last_frame,
            }
        ],
        "jobs": [
            {
                "job_id": f"job_{idx:02d}_seed{s}",
                "shot_id": shot_id,
                "seed": int(s),
                "note": f"{aspect_ratio} {quality} ({w}x{h}) {frames}f seed={s}",
            }
            for idx, s in enumerate(seeds, 1)
        ],
    }


def run_colab_batch(
    manifest_path: Path,
    outputs_root: Path = DEFAULT_OUTPUT_ROOT,
    project_override: Optional[str] = None,
    session_name: str = DEFAULT_SESSION_NAME,
    workers: int = 8,
    dry_run: bool = False,
    force_new_batch: bool = False,
) -> Dict[str, Any]:
    raw_in = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_name = project_override or str(raw_in.get("project") or raw_in.get("project_name") or "default").strip()

    pending_stage1: Optional[Dict[str, Any]] = None
    # If manifest_path is already `<outputs_root>/<project>/<seq>/manifest.json` and either not yet executed
    # or has a pending Stage 1 GDrive egress bundle awaiting 0-CU Stage 2 pull, use it in-place
    if (
        manifest_path.name == "manifest.json"
        and manifest_path.parent.name.isdigit()
        and manifest_path.parent.parent.name == project_name
        and not force_new_batch
    ):
        pending_stage1 = _extract_pending_stage1_from_log(manifest_path.parent)
        if pending_stage1 is not None or not _is_batch_dir_executed(manifest_path.parent):
            batch_dir = manifest_path.parent
            batch_seq = int(batch_dir.name)
            saved_manifest_path = manifest_path
        else:
            batch_dir, batch_seq = allocate_project_batch_dir(outputs_root, project_name, force_new=force_new_batch)
            saved_manifest_path = snapshot_manifest_into_batch_dir(manifest_path, batch_dir, project_name, batch_seq)
    else:
        batch_dir, batch_seq = allocate_project_batch_dir(outputs_root, project_name, force_new=force_new_batch)
        saved_manifest_path = snapshot_manifest_into_batch_dir(manifest_path, batch_dir, project_name, batch_seq)

    input_tar_path = batch_dir / "input_jobs.tar"
    plan = analyze_and_pack_batch(saved_manifest_path, input_tar_path)
    plan["project"] = project_name
    plan["batch_seq"] = batch_seq
    plan["batch_dir"] = str(batch_dir)
    print_plan_summary(plan, batch_dir=batch_dir)

    if dry_run:
        input_tar_path.unlink(missing_ok=True)
        print(
            f"🟢 [Dry-Run Mode] 0.0000 CU consumed. Saved manifest snapshot to: {saved_manifest_path}\n"
            f"   Omit `--dry-run` to execute Batch #{batch_seq} of project '{project_name}' on Colab Blackwell G4."
        )
        return plan

    if pending_stage1 is not None:
        input_tar_path.unlink(missing_ok=True)
        stage1_egress_info = pending_stage1
        input_file_id = ""
        print(
            f"🔄 [0-CU Stage 2 Resume] Found completed G4 Stage 1 bundle in studio_batch_raw.log "
            f"(file_id={stage1_egress_info['file_id']}, {int(stage1_egress_info['raw_bytes'])/1e6:.2f} MB). "
            f"Skipping G4 allocation (0.0000 CU) and resuming Step 4!"
        )
    else:
        ingress_prestage = prestage_input_tar_to_gdrive_0cu(input_tar_path)
        input_file_id = ingress_prestage["input_file_id"]
        print(
            f"☁️  [Step 0: 0-CU Pre-Stage] Uploaded input_jobs.tar ({ingress_prestage['bytes']/1e6:.2f} MB) "
            f"to GDrive file_id={input_file_id} in {ingress_prestage['upload_sec']}s (0.0000 CU)"
        )

        creds = get_fresh_credentials()
        pre_ccu = fetch_ccu_info(creds)
        print(
            f"💰 [Pre-Session Ledger] Balance: {pre_ccu['response_body']['currentBalance']:.4f} CU | "
            f"Active Assignments: {pre_ccu['response_body'].get('assignmentsCount', 0)}"
        )

        session_connected = False
        t_assign_start = time.time()
        t_kernel_ready = t_assign_start
        t_unassign_done = t_assign_start
        endpoint = ""
        stage1_egress_info = {}
        with tempfile.NamedTemporaryFile(mode="w", suffix="_studio_engine.py", delete=False, encoding="utf-8") as tf_rt:
            runtime_script = Path(tf_rt.name)
        os.chmod(runtime_script, 0o600)
        raw_log_path = batch_dir / "studio_batch_raw.log"

        try:
            print(f"\n🚀 [Step 1] Allocating Blackwell G4 Session ({session_name}) for {project_name}/#{batch_seq}...")
            subprocess.run([COLAB_BIN, "new", "-s", session_name, "--gpu", "g4"], check=True)
            session_connected = True
            t_kernel_ready = time.time()
            if SESSIONS_PATH.exists():
                s_data = json.loads(SESSIONS_PATH.read_text())
                endpoint = s_data.get(session_name, {}).get("endpoint", "")
            print(f"✅ [Step 1] Connected to G4 endpoint={endpoint} in {t_kernel_ready - t_assign_start:.2f}s")

            creds = get_fresh_credentials()
            engine_template = (_SCRIPT_DIR / "remote_batch_engine.py").read_text(encoding="utf-8")
            engine_code = (
                engine_template
                .replace("__STUDIO_INPUT_FILE_ID__", input_file_id)
                .replace("__STUDIO_ACCESS_TOKEN__", creds.token)
                .replace("__STUDIO_EXPECTED_SHA256__", plan["input_tar_sha256"])
                .replace("__STUDIO_QUOTA_PROJECT__", DEFAULT_QUOTA_PROJECT)
            )
            runtime_script.write_text(engine_code, encoding="utf-8")

            print("\n🎬 [Step 2] Executing Staged Run 12 Production Engine on G4 (streaming live logs)...")
            with open(raw_log_path, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    [COLAB_BIN, "exec", "-s", session_name, "-f", str(runtime_script), "--timeout", "1800"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_fh.write(line)
                    log_fh.flush()
                    if line.startswith("STUDIO_STAGE1_EGRESS_JSON:"):
                        stage1_egress_info = json.loads(line.split("STUDIO_STAGE1_EGRESS_JSON:", 1)[1].strip())
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"remote_batch_engine.py exited with code {ret}")
        finally:
            runtime_script.unlink(missing_ok=True)
            input_tar_path.unlink(missing_ok=True)
            if session_connected:
                print(f"\n🛑 [Step 3] Immediately Stopping G4 Session ({session_name}) to Halt GPU Billing...")
                subprocess.run([COLAB_BIN, "stop", "-s", session_name], check=False)
                t_unassign_done = time.time()
                session_connected = False

        vm_wall_sec = round(t_unassign_done - t_assign_start, 3)
        post_ccu = fetch_ccu_info(get_fresh_credentials())
        print(
            f"💰 [Post-Session Ledger] Balance: {post_ccu['response_body']['currentBalance']:.4f} CU | "
            f"Total G4 Session Wall Time: {vm_wall_sec:.2f}s ({vm_wall_sec * G4_CU_RATE_PER_SEC:.4f} theoretical CU)"
        )

    if not stage1_egress_info:
        raise RuntimeError("Did not receive STUDIO_STAGE1_EGRESS_JSON from remote engine.")

    print("\n📥 [Step 4: 0-CU Stage 2 Egress Pull] Downloading output bundle from Google Drive (8 HTTP Range workers)...")
    output_tar_local = batch_dir / "studio_outputs_bundle.tar"
    stage2_info = stage2_local_parallel_pull_0cu(
        file_id=stage1_egress_info["file_id"],
        local_dest_path=output_tar_local,
        expected_raw_bytes=int(stage1_egress_info["raw_bytes"]),
        expected_sha256=str(stage1_egress_info["sha256"]),
        workers=workers,
    )
    print(
        f"  ✅ Pulled {stage2_info['raw_bytes']/1e6:.2f} MB in {stage2_info['stage2_sec']:.2f}s "
        f"({stage2_info['stage2_mb_s']:.2f} MB/s) at 0.0000 CU!"
    )

    if input_file_id:
        delete_gdrive_file_quiet(input_file_id)
    delete_gdrive_file_quiet(stage1_egress_info["file_id"])

    with tarfile.open(output_tar_local, "r") as tf:
        tf.extractall(batch_dir)
    output_tar_local.unlink(missing_ok=True)

    metrics_path = batch_dir / "studio_batch_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["project"] = project_name
    metrics["batch_seq"] = batch_seq
    metrics["batch_dir"] = str(batch_dir)

    by_sb_dir = batch_dir / "by_storyboard"
    by_sb_dir.mkdir(parents=True, exist_ok=True)
    for job in metrics["jobs"]:
        s_dir = by_sb_dir / job["shot_id"]
        s_dir.mkdir(parents=True, exist_ok=True)
        mp4_name = Path(job["mp4_file"]).name
        src_mp4 = batch_dir / mp4_name
        dst_mp4 = s_dir / mp4_name
        if src_mp4.exists():
            shutil.move(str(src_mp4), str(dst_mp4))
        job["mp4_file"] = f"by_storyboard/{job['shot_id']}/{mp4_name}"

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    results_path = batch_dir / "manifest_results.json"
    if results_path.exists():
        res_data = json.loads(results_path.read_text(encoding="utf-8"))
        res_data["jobs"] = metrics["jobs"]
        results_path.write_text(json.dumps(res_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # If the batch contains >= 2 storyboards, automatically stitch the best take of each shot into a single full video (0 CU)
    recs = metrics.get("storyboard_seed_recommendations", {})
    raw_manifest = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
    ordered_shot_ids = [s["shot_id"] for s in raw_manifest.get("storyboards", [])]
    if len(ordered_shot_ids) >= 2:
        job_by_id = {j["job_id"]: j for j in metrics["jobs"]}
        concat_mp4s: List[Path] = []
        for sid in ordered_shot_ids:
            rec_jid = recs.get(sid, {}).get("recommended_job_id")
            if rec_jid and rec_jid in job_by_id:
                p_mp4 = batch_dir / job_by_id[rec_jid]["mp4_file"]
                if p_mp4.exists():
                    concat_mp4s.append(p_mp4)
        if len(concat_mp4s) >= 2:
            concat_list_file = batch_dir / "_ffmpeg_concat_list.txt"
            concat_list_file.write_text(
                "\n".join(f"file '{p.resolve()}'" for p in concat_mp4s) + "\n",
                encoding="utf-8",
            )
            stitched_mp4 = batch_dir / "full_storyboard_concat.mp4"
            ffmpeg_bin = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg")
            subprocess.run(
                [ffmpeg_bin, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list_file), "-c", "copy", str(stitched_mp4)],
                check=False,
            )
            concat_list_file.unlink(missing_ok=True)
            if stitched_mp4.exists():
                print(f"🎬 [0-CU Auto-Stitch] Stitched {len(concat_mp4s)} shots into full video: {stitched_mp4}")

    print(f"\n🎉 [Studio Render Complete] Project '{project_name}' Batch #{batch_seq} saved to: {batch_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio Colab Blackwell G4 Runner (Project + Auto-Increment Batch)")
    parser.add_argument("-p", "--project", type=str, default=None, help="Project name (saved under studio/outputs/<project>/<batch_seq>/)")
    parser.add_argument("-m", "--manifest", type=Path, default=None, help="Path to input manifest.json")
    parser.add_argument("--prompt", type=str, default=None, help="Quick single-shot prompt (auto-generates manifest.json in the batch dir)")
    parser.add_argument("--aspect", type=str, default="16:9", help="Aspect ratio preset: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9")
    parser.add_argument("--quality", type=str, default="768p", help="Quality preset: 512p (fast), 768p (standard), 1080p (ultra)")
    parser.add_argument("--duration", type=float, default=5.0, help="Target duration in seconds (snapped to 17k+5 frame grid @ 24fps)")
    parser.add_argument("--frames", type=int, default=None, help="Explicit frame count (snapped to 17k+5 grid)")
    parser.add_argument("--first-frame", type=str, default="", help="Path to first_frame PNG/JPG (optional)")
    parser.add_argument("--last-frame", type=str, default="", help="Path to last_frame PNG/JPG (optional)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds, e.g. '42' or '42,108,2026'")
    parser.add_argument("--resize-mode", type=str, default="cover", choices=["cover", "stretch"], help="Aspect alignment mode for reference frames")
    parser.add_argument("-o", "--outputs-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root outputs directory (default: studio/outputs)")
    parser.add_argument("-s", "--session", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--new-batch", action="store_true", help="Force allocating a new batch sequence number even if previous was dry-run only")
    parser.add_argument("--dry-run", action="store_true", help="Run local 0-CU planning, save manifest.json into batch dir, and print CU estimate")
    args = parser.parse_args()

    if args.prompt:
        proj = args.project or "default"
        seeds_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        batch_dir, batch_seq = allocate_project_batch_dir(args.outputs_root, proj, force_new=args.new_batch)
        quick_dict = build_quick_manifest_dict(
            project=proj,
            prompt=args.prompt,
            aspect_ratio=args.aspect,
            quality=args.quality,
            duration_sec=args.duration,
            num_frames=args.frames,
            first_frame=args.first_frame,
            last_frame=args.last_frame,
            seeds=seeds_list,
            resize_mode=args.resize_mode,
        )
        tmp_manifest = batch_dir / "_quick_source_manifest.json"
        tmp_manifest.write_text(json.dumps(quick_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_path = snapshot_manifest_into_batch_dir(tmp_manifest, batch_dir, proj, batch_seq)
        tmp_manifest.unlink(missing_ok=True)
        project_override = proj
    else:
        manifest_path = args.manifest or (_SCRIPT_DIR / "sample_single_manifest.json")
        project_override = args.project

    run_colab_batch(
        manifest_path=manifest_path,
        outputs_root=args.outputs_root,
        project_override=project_override,
        session_name=args.session,
        workers=args.workers,
        dry_run=args.dry_run,
        force_new_batch=args.new_batch if not args.prompt else False,
    )


if __name__ == "__main__":
    main()
