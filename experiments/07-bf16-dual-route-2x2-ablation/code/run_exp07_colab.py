#!/usr/bin/env python3
"""
Local Zero-Waste Colab G4 Orchestrator for Experiment 07
(`experiments/07-bf16-dual-route-2x2-ablation/code/run_exp07_colab.py`).

Strictly follows ADR-04 Two-Stage Google Drive REST API Egress:
  1. Records pre-session CCU balance & allocates `--gpu G4`
  2. Uploads `exp07_config.py`, `eval_metrics.py`, and `remote_exp07_engine.py`
  3. Runs `remote_exp07_engine.py` (streaming live logs to `outputs/exp07_raw.log`)
  4. Immediately terminates G4 (`colab stop`) inside `try...finally` the instant Stage 1 GDrive upload finishes
  5. Pulls `exp07_outputs_bundle.tar` from Google Drive at 0.00 CU using 8 parallel HTTP Range threads
  6. Reconciles session CU ledger and invokes `render_visuals_local.py` at 0.00 CU
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
from typing import Any, Dict, List, Tuple

COLAB_PYTHON = Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"
try:
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    if COLAB_PYTHON.exists() and Path(sys.executable).resolve() != COLAB_PYTHON.resolve():
        os.execv(str(COLAB_PYTHON), [str(COLAB_PYTHON), *sys.argv])
    raise

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = _SCRIPT_DIR.parent
OUTPUTS_DIR = EXP_ROOT / "outputs"
REPO_ROOT = EXP_ROOT.parent.parent
COLAB_BIN = shutil.which("colab") or str(Path.home() / ".local/share/uv/tools/google-colab-cli/bin/colab")
TOKEN_PATH = Path.home() / ".config/colab-cli/token.json"
SESSIONS_PATH = Path.home() / ".config/colab-cli/sessions.json"
_QUOTA_PROJECT_FILE = Path.home() / ".config/colab-cli/quota_project"
DEFAULT_QUOTA_PROJECT = (
    os.environ.get("STUDIO_QUOTA_PROJECT")
    or os.environ.get("GCP_QUOTA_PROJECT")
    or (_QUOTA_PROJECT_FILE.read_text(encoding="utf-8").strip() if _QUOTA_PROJECT_FILE.exists() else "")
)
DEFAULT_SESSION_NAME = "h3-exp07-g4"
G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0


def resolve_active_quota_project(access_token: str, preferred: str = "") -> str:
    cand = (
        preferred
        or os.environ.get("STUDIO_QUOTA_PROJECT")
        or os.environ.get("GCP_QUOTA_PROJECT")
        or (_QUOTA_PROJECT_FILE.read_text(encoding="utf-8").strip() if _QUOTA_PROJECT_FILE.exists() else "")
    )
    if cand:
        return cand
    for attempt in range(4):
        try:
            r = requests.get(
                "https://cloudresourcemanager.googleapis.com/v1/projects",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15,
            )
            r.raise_for_status()
            for proj in r.json().get("projects", []):
                if proj.get("lifecycleState") == "ACTIVE" and proj.get("projectId"):
                    pid = str(proj["projectId"])
                    _QUOTA_PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    _QUOTA_PROJECT_FILE.write_text(pid, encoding="utf-8")
                    return pid
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("Could not resolve active GCP quota project.")


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
            ts_local = datetime.datetime.now().isoformat(timespec="milliseconds")
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
    quota_project = resolve_active_quota_project(creds.token, quota_project)
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
        "local_dest": f"outputs/{local_dest_path.name}",
        "raw_bytes": expected_raw_bytes,
        "stage2_sec": round(elapsed, 4),
        "stage2_mb_s": round((expected_raw_bytes / 1e6) / max(elapsed, 1e-6), 2),
        "verified_sha256": True,
        "gpu_cu_consumed": 0.0,
    }


def delete_gdrive_file_quiet(file_id: str, quota_project: str = DEFAULT_QUOTA_PROJECT) -> bool:
    try:
        creds = get_fresh_credentials()
        quota_project = resolve_active_quota_project(creds.token, quota_project)
        r = requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": quota_project},
            timeout=20,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 07 Local Zero-Waste Colab G4 Orchestrator")
    parser.add_argument("-s", "--session", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_log_path = OUTPUTS_DIR / "exp07_bf16_2x2_raw.log"
    ledger_path = OUTPUTS_DIR / "colab_session_ledger_excerpt.json"

    creds = get_fresh_credentials()
    pre_ccu = fetch_ccu_info(creds)
    pre_bal = float(pre_ccu["response_body"]["currentBalance"])
    print(
        f"💰 [Pre-Session Ledger] Balance: {pre_bal:.4f} CU | "
        f"Active Assignments: {pre_ccu['response_body'].get('assignmentsCount', 0)}"
    )

    session_connected = False
    t_assign_start = time.time()
    t_kernel_ready = t_assign_start
    t_unassign_done = t_assign_start
    ts_assign_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
    ts_kernel_iso = ts_assign_iso
    ts_unassign_iso = ts_assign_iso
    stage1_egress_info: Dict[str, Any] = {}

    with tempfile.NamedTemporaryFile(mode="w", suffix="_remote_exp07.py", delete=False, encoding="utf-8") as tf_rt:
        runtime_script = Path(tf_rt.name)
    os.chmod(runtime_script, 0o600)

    try:
        print(f"\n🚀 [Step 1] Allocating Blackwell G4 Session ({args.session})...")
        ts_assign_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
        t_assign_start = time.time()
        subprocess.run([COLAB_BIN, "new", "-s", args.session, "--gpu", "g4"], check=True)
        session_connected = True
        t_kernel_ready = time.time()
        ts_kernel_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
        print(f"✅ [Step 1] Connected to G4 in {t_kernel_ready - t_assign_start:.2f}s")

        # Upload canonical first_frame.png and last_frame.png + helper modules to /content/
        first_src = REPO_ROOT / "studio/api/default_assets/first_frame.png"
        last_src = REPO_ROOT / "studio/api/default_assets/last_frame.png"
        if first_src.exists() and last_src.exists():
            subprocess.run([COLAB_BIN, "upload", "-s", args.session, str(first_src), "/content/input_first_frame.png"], check=True)
            subprocess.run([COLAB_BIN, "upload", "-s", args.session, str(last_src), "/content/input_last_frame.png"], check=True)

        for mod_name in ("exp07_config.py", "eval_metrics.py"):
            subprocess.run(
                [COLAB_BIN, "upload", "-s", args.session, str(_SCRIPT_DIR / mod_name), f"/content/{mod_name}"],
                check=True,
            )

        creds = get_fresh_credentials()
        active_qp = resolve_active_quota_project(creds.token, DEFAULT_QUOTA_PROJECT)
        engine_template = (_SCRIPT_DIR / "remote_exp07_engine.py").read_text(encoding="utf-8")
        engine_code = (
            engine_template
            .replace("__EXP07_ACCESS_TOKEN__", creds.token)
            .replace("__EXP07_QUOTA_PROJECT__", active_qp)
        )
        runtime_script.write_text(engine_code, encoding="utf-8")

        print("\n🎬 [Step 2] Executing Experiment 07 2x2 BF16 Dual-Route Ablation on G4 (streaming live logs)...")
        with open(raw_log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [COLAB_BIN, "exec", "-s", args.session, "-f", str(runtime_script), "--timeout", "2400"],
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
                if line.startswith("EXP07_STAGE1_EGRESS_JSON:"):
                    stage1_egress_info = json.loads(line.split("EXP07_STAGE1_EGRESS_JSON:", 1)[1].strip())
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"remote_exp07_engine.py exited with code {ret}")
    finally:
        runtime_script.unlink(missing_ok=True)
        if session_connected:
            print(f"\n🛑 [Step 3] Immediately Stopping G4 Session ({args.session}) to Halt GPU Billing...")
            subprocess.run([COLAB_BIN, "stop", "-s", args.session], check=False)
            t_unassign_done = time.time()
            ts_unassign_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
            session_connected = False

    vm_wall_sec = round(t_unassign_done - t_assign_start, 3)
    post_ccu = fetch_ccu_info(get_fresh_credentials())
    post_bal = float(post_ccu["response_body"]["currentBalance"])
    print(
        f"💰 [Post-Session Ledger] Balance: {post_bal:.4f} CU | "
        f"Total G4 Session Wall Time: {vm_wall_sec:.2f}s ({vm_wall_sec * G4_CU_RATE_PER_SEC:.4f} theoretical CU)"
    )

    if not stage1_egress_info:
        raise RuntimeError("Did not receive EXP07_STAGE1_EGRESS_JSON from remote engine.")

    print("\n📥 [Step 4: 0-CU Stage 2 Egress Pull] Downloading exp07_outputs_bundle.tar from Google Drive...")
    output_tar_local = OUTPUTS_DIR / "exp07_outputs_bundle.tar"
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
    delete_gdrive_file_quiet(stage1_egress_info["file_id"])

    with tarfile.open(output_tar_local, "r") as tf:
        tf.extractall(OUTPUTS_DIR)
    output_tar_local.unlink(missing_ok=True)

    # Reconcile session ledger into exp07_bf16_2x2_metrics.json
    settled_ccu = fetch_ccu_info(get_fresh_credentials())
    settled_bal = float(settled_ccu["response_body"]["currentBalance"])
    ledger_payload = {
        "experiment": "07-bf16-dual-route-2x2-ablation",
        "session_name": args.session,
        "gpu_type": "G4",
        "g4_cu_rate_per_hour": G4_CU_RATE_PER_HOUR,
        "events": {
            "pre_session_ccu_info": pre_ccu,
            "vm_assign_post_ts": ts_assign_iso,
            "kernel_ready_ts": ts_kernel_iso,
            "vm_unassign_post_ts": ts_unassign_iso,
            "post_session_immediate_ccu_info": post_ccu,
            "post_session_settled_ccu_info": settled_ccu,
            "stage1_gdrive_egress": stage1_egress_info,
            "stage2_local_0cu_pull": stage2_info,
        },
        "derived_accounting": {
            "assign_to_unassign_wall_sec": vm_wall_sec,
            "theoretical_session_cu": round(vm_wall_sec * G4_CU_RATE_PER_SEC, 4),
            "pre_balance_cu": pre_bal,
            "post_immediate_balance_cu": post_bal,
            "post_settled_balance_cu": settled_bal,
            "settled_deducted_cu": round(pre_bal - settled_bal, 4),
        },
    }
    ledger_path.write_text(json.dumps(ledger_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_path = OUTPUTS_DIR / "exp07_bf16_2x2_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["lifecycle_telemetry"].update({
        "session_assign_to_unassign_wall_sec": vm_wall_sec,
        "session_theoretical_cu": round(vm_wall_sec * G4_CU_RATE_PER_SEC, 4),
        "pre_session_balance_cu": pre_bal,
        "post_session_immediate_balance_cu": post_bal,
        "post_session_settled_balance_cu": settled_bal,
        "settled_ledger_deducted_cu": round(pre_bal - settled_bal, 4),
        "stage1_gdrive_egress": stage1_egress_info,
        "stage2_local_0cu_pull": stage2_info,
    })
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    render_script = _SCRIPT_DIR / "render_visuals_local.py"
    if render_script.exists():
        print("\n🎨 [Step 5: 0-CU Local Visual Rendering] Generating 2x2 comparison charts & 4-grid video...")
        uv_bin = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        if Path(uv_bin).exists():
            subprocess.run([uv_bin, "run", "--with", "matplotlib", "--with", "pillow", "--with", "numpy", "python3", str(render_script)], check=True)
        else:
            subprocess.run([sys.executable, str(render_script)], check=True)

    print(f"\n🎉 [Experiment 07 Complete] All outputs & metrics saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
