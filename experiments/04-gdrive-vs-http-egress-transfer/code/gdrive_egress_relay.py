#!/usr/bin/env python3
"""
Zero-Click Two-Stage Google Drive Egress Relay Utility (`gdrive_egress_relay.py`).

Replaces `colab download` (which base64-encodes files via Jupyter `/api/contents?content=1` with +36.84% wire
inflation while holding the GPU VM open) and replaces interactive `colab drivemount` FUSE mounts.

Workflow:
  1. Stage 1 (On Colab GPU VM, Datacenter -> Datacenter at 50~90 MB/s):
     Streams the raw binary file/tar bundle directly to Google Drive v3 API (`uploadType=media`),
     returning `file_id` and verifying remote `size` + `md5Checksum` upon HTTP 200 OK.
  2. Immediate GPU Teardown (`colab stop -s <session>`):
     Terminates GPU billing immediately after Stage 1 completes (~1.0s for single video, ~2.2s for 12-video bundle).
  3. Stage 2 (On Local Mac at 0.00 CU after GPU VM is stopped):
     Downloads the raw binary file from Google Drive v3 API (`/drive/v3/files/{fileId}?alt=media`)
     using 8-worker parallel HTTP `Range` requests, verifies integrity, and optionally deletes the temporary Drive file.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
from typing import Any, Dict, List, Tuple

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

TOKEN_PATH = pathlib.Path.home() / ".config/colab-cli/token.json"
_QUOTA_PROJECT_FILE = pathlib.Path.home() / ".config/colab-cli/quota_project"
COLAB_BIN = shutil.which("colab") or str(pathlib.Path.home() / ".local/share/uv/tools/google-colab-cli/bin/colab")
DEFAULT_QUOTA_PROJECT = (
    os.environ.get("GCP_QUOTA_PROJECT")
    or os.environ.get("STUDIO_QUOTA_PROJECT")
    or (_QUOTA_PROJECT_FILE.read_text(encoding="utf-8").strip() if _QUOTA_PROJECT_FILE.exists() else "")
)


def load_fresh_colab_credentials(token_path: pathlib.Path = TOKEN_PATH) -> Credentials:
    creds = Credentials.from_authorized_user_file(str(token_path))
    creds.refresh(Request())
    token_path.write_text(creds.to_json())
    return creds


def stage1_remote_push_and_stop(
    session_name: str,
    remote_file_path: str,
    quota_project: str = DEFAULT_QUOTA_PROJECT,
    stop_immediately: bool = True,
) -> Dict[str, Any]:
    """
    Executes Stage 1 on the remote Colab VM: pushes `remote_file_path` to Google Drive v3 API
    over Google's internal datacenter network, then immediately stops the Colab GPU session.
    """
    creds = load_fresh_colab_credentials()
    remote_snippet = f"""
import hashlib, json, pathlib, time, requests
p = pathlib.Path({json.dumps(remote_file_path)})
data = p.read_bytes()
sz = len(data)
md5 = hashlib.md5(data).hexdigest()
sha256 = hashlib.sha256(data).hexdigest()
t0 = time.perf_counter()
r = requests.post(
    "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum",
    headers={{
        "Authorization": "Bearer " + {json.dumps(creds.token)},
        "X-Goog-User-Project": {json.dumps(quota_project)},
        "Content-Type": "application/octet-stream",
        "Content-Length": str(sz),
    }},
    data=data,
    timeout=180,
)
elapsed = time.perf_counter() - t0
r.raise_for_status()
info = r.json()
assert int(info["size"]) == sz and info.get("md5Checksum") == md5
print("RELAY_STAGE1_JSON:" + json.dumps({{
    "file_id": info["id"],
    "raw_bytes": sz,
    "md5": md5,
    "sha256": sha256,
    "stage1_sec": round(elapsed, 4),
    "stage1_mb_s": round((sz / 1e6) / elapsed, 2),
}}))
"""
    tmp_script = pathlib.Path(f"/tmp/_relay_stage1_{session_name}.py")
    tmp_script.write_text(remote_snippet)
    try:
        proc = subprocess.run(
            [COLAB_BIN, "exec", "-s", session_name, "-f", str(tmp_script), "--timeout", "240"],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        tmp_script.unlink(missing_ok=True)
        if stop_immediately:
            subprocess.run([COLAB_BIN, "stop", "-s", session_name], check=False)

    for line in proc.stdout.splitlines():
        if line.startswith("RELAY_STAGE1_JSON:"):
            return json.loads(line.split("RELAY_STAGE1_JSON:", 1)[1])
    raise RuntimeError(f"Failed to parse Stage 1 output: {proc.stdout}\n{proc.stderr}")


def stage2_local_parallel_pull(
    file_id: str,
    local_dest_path: pathlib.Path,
    expected_raw_bytes: int,
    expected_sha256: str,
    quota_project: str = DEFAULT_QUOTA_PROJECT,
    workers: int = 8,
    chunk_mib: int = 8,
    delete_from_gdrive_after_pull: bool = True,
) -> Dict[str, Any]:
    """
    Executes Stage 2 on the local machine at 0.00 GPU CU (after the Colab VM is stopped):
    Downloads the file from Google Drive v3 API using parallel HTTP Range requests.
    """
    creds = load_fresh_colab_credentials()
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
        raise ValueError(f"SHA256 mismatch: {sha_actual} != {expected_sha256}")

    if delete_from_gdrive_after_pull:
        requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": quota_project},
            timeout=20,
        )

    return {
        "file_id": file_id,
        "local_dest": str(local_dest_path),
        "raw_bytes": expected_raw_bytes,
        "stage2_sec": round(elapsed, 4),
        "stage2_mb_s": round((expected_raw_bytes / 1e6) / elapsed, 2),
        "verified_sha256": True,
        "deleted_from_gdrive": delete_from_gdrive_after_pull,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Stage Zero-Click Google Drive Egress Relay")
    parser.add_argument("-s", "--session", required=True, help="Colab session name")
    parser.add_argument("--remote-path", required=True, help="Remote file path on Colab VM")
    parser.add_argument("--local-path", required=True, type=pathlib.Path, help="Local destination path")
    parser.add_argument("--workers", type=int, default=8, help="Parallel HTTP Range workers for Stage 2")
    parser.add_argument("--keep-session", action="store_true", help="Do not stop Colab session after Stage 1")
    parser.add_argument("--keep-gdrive", action="store_true", help="Keep file in Google Drive after local pull")
    args = parser.parse_args()

    s1 = stage1_remote_push_and_stop(
        session_name=args.session,
        remote_file_path=args.remote_path,
        stop_immediately=not args.keep_session,
    )
    print(f"[Stage 1 Complete] Uploaded {s1['raw_bytes']:,} B to GDrive in {s1['stage1_sec']}s ({s1['stage1_mb_s']} MB/s)")
    s2 = stage2_local_parallel_pull(
        file_id=s1["file_id"],
        local_dest_path=args.local_path,
        expected_raw_bytes=s1["raw_bytes"],
        expected_sha256=s1["sha256"],
        workers=args.workers,
        delete_from_gdrive_after_pull=not args.keep_gdrive,
    )
    print(f"[Stage 2 Complete] Pulled to {s2['local_dest']} at 0 CU in {s2['stage2_sec']}s ({s2['stage2_mb_s']} MB/s)")


if __name__ == "__main__":
    main()
