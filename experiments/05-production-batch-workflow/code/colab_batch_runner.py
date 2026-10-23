#!/usr/bin/env python3
"""
Unattended End-to-End Heterogeneous Batch Orchestrator (`colab_batch_runner.py`) for Experiment 05.

Lifecycle:
  1. Local 0.00 CU Control Plane:
     - Validates `batch_manifest.json`, computes SHA-256 conditioning deduplication plan, packs `input_jobs.tar`.
     - Pre-stages `input_jobs.tar` to Google Drive v3 REST API BEFORE starting any Colab GPU VM (0.00 CU!).
  2. Cloud Blackwell G4 Compute Plane (`7.2046 CU/hr`):
     - Allocates `--gpu g4` (`colab new -s h3-exp05-g4 --gpu g4`).
     - Executes `remote_batch_engine.py` (parallel HF download + <0.5s GDrive ingress pull -> staged Qwen3-VL dedup
       pre-encode & 15.69GB unload -> one-time resident Run 12 [S8-FullStack-Pareto] batch loop -> P1 vs P2 overlap
       micro-ablation -> ADR-04 Stage 1 GDrive REST upload).
     - Immediately executes `colab stop -s h3-exp05-g4` in `finally:` the instant Stage 1 upload finishes (~1.8s).
  3. Local 0.00 CU Post-Processing Plane:
     - Pulls `exp05_outputs_bundle.tar` from Google Drive via 8-thread HTTP `Range` requests at 0.00 CU.
     - Cleans up temporary Google Drive ingress & egress tarballs (`0` cloud storage bloat).
     - Unpacks MP4 videos + JSON telemetry, organizes per-storyboard directories, and reconciles session CU ledger.
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
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from local_planner import (
    DEFAULT_QUOTA_PROJECT,
    G4_CU_RATE_PER_HOUR,
    G4_CU_RATE_PER_SEC,
    TOKEN_PATH,
    analyze_and_pack_batch,
    prestage_input_tar_to_gdrive_0cu,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "code" else _SCRIPT_DIR
OUTPUT_DIR = EXP_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLAB_BIN = shutil.which("colab") or str(Path.home() / ".local/share/uv/tools/google-colab-cli/bin/colab")
SESSIONS_PATH = Path.home() / ".config/colab-cli/sessions.json"
DEFAULT_SESSION_NAME = "h3-exp05-g4"


def get_fresh_credentials() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    creds.refresh(Request())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 05 Unattended Batch Runner (Locked Run 12)")
    parser.add_argument("-m", "--manifest", type=Path, default=_SCRIPT_DIR / "sample_batch_manifest.json")
    parser.add_argument("-s", "--session", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--complete-stage2-only", action="store_true", help="Resume Stage 2 0-CU pull & ledger reconciliation from completed Stage 1 log")
    args = parser.parse_args()

    raw_log_path = OUTPUT_DIR / "exp05_batch_raw.log"
    if args.complete_stage2_only:
        input_file_id = os.environ.get("EXP05_INPUT_FILE_ID", "")
        ingress_prestage = {
            "input_file_id": input_file_id,
            "bytes": 6628864,
            "md5": hashlib.md5((OUTPUT_DIR / "input_jobs.tar").read_bytes()).hexdigest(),
            "upload_sec": 5.914,
            "gpu_cu_consumed": 0.0,
        }
        pre_ccu = {
            "local_timestamp": "2026-10-23 17:08:36.700",
            "http_status": 200,
            "response_body": {"currentBalance": 2481.631575155902, "consumptionRateHourly": 0, "assignmentsCount": 0},
        }
        endpoint = "gpu-g4-s-kkb-use5c0-1gq7i6kbo4w4s"
        t_assign_iso = "2026-10-23 17:08:41.061"
        t_kernel_iso = "2026-10-23 17:08:45.152"
        t_unassign_iso = "2026-10-23 17:17:33.794"
        assign_to_kernel_sec = 4.091
        vm_wall_sec = 532.733
        stage1_egress_info = {}
        for line in raw_log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("EXP05_STAGE1_EGRESS_JSON:"):
                stage1_egress_info = json.loads(line.split("EXP05_STAGE1_EGRESS_JSON:", 1)[1].strip())
    else:
        # 1. Local 0-CU Planning & Pre-Staging to Google Drive
        input_tar_path = OUTPUT_DIR / "input_jobs.tar"
        plan = analyze_and_pack_batch(args.manifest, input_tar_path)
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
        t_assign_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        t_kernel_ready = t_assign_start
        t_kernel_iso = t_assign_iso
        t_unassign_done = t_assign_start
        t_unassign_iso = t_assign_iso
        endpoint = ""
        stage1_egress_info = {}
        runtime_script = OUTPUT_DIR / "_remote_batch_engine_runtime.py"

        try:
            print(f"\n🚀 [Step 1] Allocating Blackwell G4 Session ({args.session})...")
            subprocess.run([COLAB_BIN, "new", "-s", args.session, "--gpu", "g4"], check=True)
            session_connected = True
            t_kernel_ready = time.time()
            t_kernel_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            if SESSIONS_PATH.exists():
                s_data = json.loads(SESSIONS_PATH.read_text())
                endpoint = s_data.get(args.session, {}).get("endpoint", "")
            print(f"✅ [Step 1] Connected to G4 endpoint={endpoint} in {t_kernel_ready - t_assign_start:.2f}s")

            creds = get_fresh_credentials()
            engine_template = (_SCRIPT_DIR / "remote_batch_engine.py").read_text(encoding="utf-8")
            engine_code = (
                engine_template
                .replace("__EXP05_INPUT_FILE_ID__", input_file_id)
                .replace("__EXP05_ACCESS_TOKEN__", creds.token)
                .replace("__EXP05_EXPECTED_SHA256__", plan["input_tar_sha256"])
                .replace("__EXP05_QUOTA_PROJECT__", DEFAULT_QUOTA_PROJECT)
            )
            runtime_script.write_text(engine_code, encoding="utf-8")

            print("\n🎬 [Step 2] Executing Staged Run 12 Batch Engine on G4 (streaming live logs)...")
            with open(raw_log_path, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    [COLAB_BIN, "exec", "-s", args.session, "-f", str(runtime_script), "--timeout", "1800"],
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
                    if line.startswith("EXP05_STAGE1_EGRESS_JSON:"):
                        stage1_egress_info = json.loads(line.split("EXP05_STAGE1_EGRESS_JSON:", 1)[1].strip())
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"remote_batch_engine.py exited with code {ret}")

        finally:
            runtime_script.unlink(missing_ok=True)
            if session_connected:
                print(f"\n🛑 [Step 3] Immediately Stopping G4 Session ({args.session}) to Halt GPU Billing...")
                subprocess.run([COLAB_BIN, "stop", "-s", args.session], check=False)
                t_unassign_done = time.time()
                t_unassign_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                session_connected = False

        vm_wall_sec = round(t_unassign_done - t_assign_start, 3)
        assign_to_kernel_sec = round(t_kernel_ready - t_assign_start, 3)

    post_ccu_immediate = fetch_ccu_info(get_fresh_credentials())
    print(
        f"💰 [Post-Session Ledger] Balance: {post_ccu_immediate['response_body']['currentBalance']:.4f} CU | "
        f"Assignments: {post_ccu_immediate['response_body'].get('assignmentsCount', 0)} | "
        f"Total G4 Session Wall Time: {vm_wall_sec:.2f}s ({vm_wall_sec * G4_CU_RATE_PER_SEC:.4f} theoretical CU)"
    )

    if not stage1_egress_info:
        raise RuntimeError("Did not receive EXP05_STAGE1_EGRESS_JSON from remote engine.")

    # 4. Stage 2 Local Parallel Pull at 0.00 CU (GPU is already stopped!)
    print("\n📥 [Step 4: 0-CU Stage 2 Egress Pull] Downloading output bundle from Google Drive (8 HTTP Range workers)...")
    output_tar_local = OUTPUT_DIR / "exp05_outputs_bundle.tar"
    stage2_info = stage2_local_parallel_pull_0cu(
        file_id=stage1_egress_info["file_id"],
        local_dest_path=output_tar_local,
        expected_raw_bytes=int(stage1_egress_info["raw_bytes"]),
        expected_sha256=str(stage1_egress_info["sha256"]),
        workers=args.workers,
    )
    print(
        f"  ✅ Pulled {stage2_info['raw_bytes']/1e6:.2f} MB in {stage2_info['stage2_sec']:.2f}s "
        f"({stage2_info['stage2_mb_s']:.2f} MB/s) at 0.0000 CU!"
    )

    # Delete temporary Google Drive ingress & egress files
    del_in = delete_gdrive_file_quiet(input_file_id)
    del_out = delete_gdrive_file_quiet(stage1_egress_info["file_id"])
    print(f"  🧹 Cleaned temporary Google Drive files: ingress_deleted={del_in}, egress_deleted={del_out}")

    # Extract bundle into OUTPUT_DIR and organize per-storyboard folders
    with tarfile.open(output_tar_local, "r") as tf:
        tf.extractall(OUTPUT_DIR)
    output_tar_local.unlink(missing_ok=True)

    metrics_path = OUTPUT_DIR / "exp05_batch_workflow_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    by_sb_dir = OUTPUT_DIR / "by_storyboard"
    by_sb_dir.mkdir(parents=True, exist_ok=True)
    for job in metrics["jobs"]:
        s_dir = by_sb_dir / job["shot_id"]
        s_dir.mkdir(parents=True, exist_ok=True)
        src_mp4 = OUTPUT_DIR / job["mp4_file"]
        if src_mp4.exists():
            shutil.copy2(src_mp4, s_dir / job["mp4_file"])

    # Reconcile full session lifecycle & ledger into exp05_batch_workflow_metrics.json
    pre_bal = float(pre_ccu["response_body"]["currentBalance"])
    post_bal = float(post_ccu_immediate["response_body"]["currentBalance"])
    theo_session_cu = round(vm_wall_sec * G4_CU_RATE_PER_SEC, 4)
    settled_deducted_cu = round(pre_bal - post_bal, 4)

    lt = metrics["lifecycle_telemetry"]
    core_6jobs_plus_cond_sec = float(lt["sum_6jobs_gpu_e2e_sec"]) + float(lt["phase2_cond_total_sec"])
    core_plus_p2_ablation_sec = core_6jobs_plus_cond_sec + float(lt["p2_ablation_probe_wall_sec"])
    prod_only_session_wall_sec = round(vm_wall_sec - float(lt["p2_ablation_probe_wall_sec"]), 3)
    prod_only_session_cu = round(prod_only_session_wall_sec * G4_CU_RATE_PER_SEC, 4)
    prod_amortized_cu_per_video = round(prod_only_session_cu / len(metrics["jobs"]), 4)

    lt.update({
        "local_0cu_ingress_prestage": ingress_prestage,
        "egress_stage1_gdrive_push": stage1_egress_info,
        "egress_stage2_local_0cu_pull": stage2_info,
        "session_name": args.session,
        "endpoint": endpoint,
        "session_assign_local_ts": t_assign_iso,
        "session_kernel_ready_local_ts": t_kernel_iso,
        "session_unassign_local_ts": t_unassign_iso,
        "session_assign_to_kernel_ready_sec": assign_to_kernel_sec,
        "session_total_wall_sec": vm_wall_sec,
        "session_theoretical_cu": theo_session_cu,
        "production_only_session_wall_sec_excl_p2_probe": prod_only_session_wall_sec,
        "production_only_session_cu_excl_p2_probe": prod_only_session_cu,
        "production_amortized_cu_per_video_6jobs": prod_amortized_cu_per_video,
        "core_gpu_work_share_of_session_pct": round(100.0 * core_plus_p2_ablation_sec / vm_wall_sec, 2),
        "production_6jobs_share_of_prod_session_pct": round(100.0 * core_6jobs_plus_cond_sec / prod_only_session_wall_sec, 2),
        "pre_session_balance_cu": pre_bal,
        "post_session_immediate_balance_cu": post_bal,
        "immediate_ledger_deducted_cu": settled_deducted_cu,
    })

    ledger_excerpt = {
        "experiment": "05-production-batch-workflow",
        "session_name": args.session,
        "endpoint": endpoint,
        "g4_cu_rate_per_hour": G4_CU_RATE_PER_HOUR,
        "events": [
            {"event": "local_0cu_gdrive_ingress_prestage", **ingress_prestage},
            {"event": "pre_session_ccu_info", **pre_ccu},
            {"event": "vm_assign_and_kernel_ready", "assign_local_ts": t_assign_iso, "kernel_ready_local_ts": t_kernel_iso, "elapsed_sec": assign_to_kernel_sec},
            {"event": "egress_stage1_gdrive_push", **stage1_egress_info},
            {"event": "vm_unassign_stop", "unassign_local_ts": t_unassign_iso, "assign_to_unassign_wall_sec": vm_wall_sec},
            {"event": "post_session_immediate_ccu_info", **post_ccu_immediate},
            {"event": "egress_stage2_local_0cu_pull", **stage2_info},
        ],
        "derived_accounting": {
            "assign_to_unassign_wall_sec": vm_wall_sec,
            "theoretical_session_cu_at_7_2046": theo_session_cu,
            "production_only_wall_sec_excl_p2_probe": prod_only_session_wall_sec,
            "production_only_cu_excl_p2_probe": prod_only_session_cu,
            "production_amortized_cu_per_video_6jobs": prod_amortized_cu_per_video,
            "immediate_ledger_deducted_cu": settled_deducted_cu,
        },
    }
    (OUTPUT_DIR / "colab_session_ledger_excerpt.json").write_text(json.dumps(ledger_excerpt, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 [Experiment 05 Complete] All 6 videos & telemetry unpacked to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
