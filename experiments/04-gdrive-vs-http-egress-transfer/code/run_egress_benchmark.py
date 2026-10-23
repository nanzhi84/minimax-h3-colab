#!/usr/bin/env python3
"""
Experiment 04: Colab-to-Local HTTP Egress (`colab download` / Jupyter `/api/contents`)
vs. Colab-to-Google-Drive Egress Relay (`Colab G4 VM -> Google Drive v3 API -> Stop GPU -> 0-CU Local Pull`)
and Traditional Colab FUSE Mount (`/content/drive` + `drive.flush_and_unmount()`).

Benchmarks across 4 realistic H.264 MP4 + JSON output bundle tiers on NVIDIA RTX PRO 6000 Blackwell (G4):
  - T1_Single_Run (~10.63 MB raw tar -> ~14.55 MB /api/contents base64 JSON wire) : 1x Run 12 [S8-FullStack-Pareto] video + JSON
  - T2_4Run_Batch (~41.10 MB raw tar -> ~56.25 MB /api/contents base64 JSON wire) : 4x H3 videos + JSON
  - T3_12Run_Exp03 (122.42 MB raw tar -> 167.52 MB /api/contents base64 JSON wire): Exact Experiment 03 12-video bundle
  - T4_36Run_Sweep (367.26 MB raw tar -> 502.56 MB /api/contents base64 JSON wire): 36x H3 videos large grid-search bundle
"""

import base64
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "code" else _SCRIPT_DIR
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATH = pathlib.Path.home() / ".config/colab-cli/token.json"
_QUOTA_PROJECT_FILE = pathlib.Path.home() / ".config/colab-cli/quota_project"
SESSIONS_PATH = pathlib.Path.home() / ".config/colab-cli/sessions.json"
COLAB_BIN = shutil.which("colab") or str(pathlib.Path.home() / ".local/share/uv/tools/google-colab-cli/bin/colab")
QUOTA_PROJECT = (
    os.environ.get("GCP_QUOTA_PROJECT")
    or os.environ.get("STUDIO_QUOTA_PROJECT")
    or (_QUOTA_PROJECT_FILE.read_text(encoding="utf-8").strip() if _QUOTA_PROJECT_FILE.exists() else "")
)
SESSION_NAME = "h3-exp04-g4"
SEED_FILE_ID = os.environ.get("EXP04_SEED_FILE_ID", "")
SEED_EXPECTED_SIZE = 122419200

# Settled Blackwell G4 hourly rate from Experiment 02 & 03 /tun/m/ccu-info audits
G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0
EXP03_EFFECTIVE_BILLED_RATE_PER_HOUR = 6.9488


def get_fresh_credentials() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    # Force refresh if < 20 min remaining
    creds.refresh(Request())
    TOKEN_PATH.write_text(creds.to_json())
    return creds


def fetch_ccu_info(creds: Credentials) -> Dict[str, Any]:
    ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    r = requests.get(
        "https://colab.research.google.com/tun/m/ccu-info?authuser=0",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Accept": "application/json",
        },
        timeout=20,
    )
    r.raise_for_status()
    text = r.text
    if text.startswith(")]}'"):
        text = text.split("\n", 1)[1]
    body = json.loads(text)
    return {
        "local_timestamp": ts_local,
        "utc_header_date": r.headers.get("Date", ""),
        "http_status": r.status_code,
        "response_body": body,
    }


REMOTE_PROBE_TEMPLATE = r'''#!/usr/bin/env python3
import base64
import hashlib
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ACCESS_TOKEN = __ACCESS_TOKEN__
QUOTA_PROJECT = __QUOTA_PROJECT__
SEED_FILE_ID = __SEED_FILE_ID__

WORK_DIR = pathlib.Path("/content/exp04_work")
EXTRACT_DIR = WORK_DIR / "extracted"
WORK_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

headers_auth = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "X-Goog-User-Project": QUOTA_PROJECT,
}

# 1. Inspect Hardware & Google Colab FUSE drive.py implementation
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
cpu_count = os.cpu_count()
drive_py_info = {}
try:
    from google.colab import drive as colab_drive
    src = inspect.getsource(colab_drive)
    drive_py_info = {
        "module_path": inspect.getfile(colab_drive),
        "has_flush_and_unmount": hasattr(colab_drive, "flush_and_unmount"),
        "flush_timeout_mentions": [
            line.strip() for line in src.splitlines()
            if "timeout" in line or "flush" in line or "drive-fs" in line
        ][:15],
    }
except Exception as e:
    drive_py_info = {"error": str(e)}

print(f"[Remote] GPU: {gpu_name} | vCPUs: {cpu_count}")

# 2. Pull pre-staged 122.42 MB Exp03 bundle from Google Drive (Datacenter -> Datacenter)
t3_path = WORK_DIR / "T3_12Run_Exp03_122MB.tar"
t0 = time.perf_counter()
with requests.get(
    f"https://www.googleapis.com/drive/v3/files/{SEED_FILE_ID}?alt=media",
    headers=headers_auth,
    stream=True,
    timeout=120,
) as r:
    r.raise_for_status()
    with open(t3_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
seed_pull_sec = time.perf_counter() - t0
t3_size = t3_path.stat().st_size
print(f"[Remote] Pulled seed T3 ({t3_size:,} B) from GDrive in {seed_pull_sec:.3f}s ({t3_size/1e6/seed_pull_sec:.2f} MB/s)")

# Extract T3 to construct T1, T2, and T4 from real H.264 MP4s + JSON
with tarfile.open(t3_path, "r") as tf:
    tf.extractall(EXTRACT_DIR)

# Build T1 (Single Run 12 video + JSON)
t1_stage = WORK_DIR / "stage_t1"
t1_stage.mkdir(exist_ok=True)
shutil.copy2(EXTRACT_DIR / "g4_run12_Run12_FullStack_CKSolDirect_Mod.mp4", t1_stage / "g4_run12_Run12_FullStack_CKSolDirect_Mod.mp4")
shutil.copy2(EXTRACT_DIR / "g4_sol_attn_ablation_metrics.json", t1_stage / "g4_sol_attn_ablation_metrics.json")
t1_path = WORK_DIR / "T1_Single_Run_10MB.tar"
subprocess.run(["tar", "-cf", str(t1_path), "-C", str(t1_stage), "."], check=True)

# Build T2 (4 Runs: Run 01..04 MP4s + JSON)
t2_stage = WORK_DIR / "stage_t2"
t2_stage.mkdir(exist_ok=True)
for fn in sorted(os.listdir(EXTRACT_DIR)):
    if fn.startswith(("g4_run01_", "g4_run02_", "g4_run03_", "g4_run04_")) or fn.endswith(".json"):
        shutil.copy2(EXTRACT_DIR / fn, t2_stage / fn)
t2_path = WORK_DIR / "T2_4Run_Batch_41MB.tar"
subprocess.run(["tar", "-cf", str(t2_path), "-C", str(t2_stage), "."], check=True)

# Build T4 (36 Runs: 3x replication of the 12-video Exp03 bundle)
t4_stage = WORK_DIR / "stage_t4"
t4_stage.mkdir(exist_ok=True)
for batch_idx in range(1, 4):
    for fn in sorted(os.listdir(EXTRACT_DIR)):
        shutil.copy2(EXTRACT_DIR / fn, t4_stage / f"b{batch_idx:02d}_{fn}")
t4_path = WORK_DIR / "T4_36Run_Sweep_367MB.tar"
subprocess.run(["tar", "-cf", str(t4_path), "-C", str(t4_stage), "."], check=True)

tiers = [
    ("T1_Single_Run", "1x Run 12 [S8-FullStack-Pareto] MP4 + JSON", t1_path),
    ("T2_4Run_Batch", "4x H3 MP4s (Runs 01-04) + JSON", t2_path),
    ("T3_12Run_Exp03", "12x H3 MP4s + PNGs + JSON (Exact Exp 03 Bundle)", t3_path),
    ("T4_36Run_Sweep", "36x H3 MP4s + JSONs (Large Grid Sweep)", t4_path),
]

tier_meta = {}
for code, desc, path in tiers:
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    sz = len(data)
    b64_raw = ((sz + 2) // 3) * 4
    newlines = (b64_raw + 75) // 76
    est_wire = b64_raw + 2 * newlines + 335
    tier_meta[code] = {
        "code": code,
        "desc": desc,
        "remote_path": str(path),
        "raw_bytes": sz,
        "raw_mb": round(sz / 1e6, 3),
        "raw_mib": round(sz / (1024 ** 2), 3),
        "expected_http_contents_wire_bytes": est_wire,
        "expected_http_contents_wire_mb": round(est_wire / 1e6, 3),
        "sha256": sha256,
        "md5": md5,
    }
    print(f"[Remote] Prepared {code}: raw={sz:,} B ({sz/1e6:.2f} MB), sha256={sha256[:12]}")


def upload_media_single_post(path: pathlib.Path, remote_name: str, expected_md5: str) -> dict:
    sz = path.stat().st_size
    session = requests.Session()
    t0 = time.perf_counter()
    with open(path, "rb") as fh:
        r = session.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum",
            headers={
                **headers_auth,
                "Content-Type": "application/x-tar",
                "Content-Length": str(sz),
            },
            data=fh,
            timeout=180,
        )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    info = r.json()
    assert int(info["size"]) == sz, f"Size mismatch: {info}"
    assert info.get("md5Checksum") == expected_md5, f"MD5 mismatch: {info}"
    return {
        "method": "GDrive-REST-Media",
        "file_id": info["id"],
        "elapsed_sec": round(elapsed, 4),
        "throughput_mb_s": round((sz / 1e6) / elapsed, 2),
        "throughput_mib_s": round((sz / (1024 ** 2)) / elapsed, 2),
        "verified_md5": info.get("md5Checksum") == expected_md5,
        "verified_size": int(info["size"]) == sz,
    }


def upload_resumable_put(path: pathlib.Path, remote_name: str, expected_md5: str) -> dict:
    sz = path.stat().st_size
    session = requests.Session()
    t0 = time.perf_counter()
    r_init = session.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum",
        headers={
            **headers_auth,
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/x-tar",
            "X-Upload-Content-Length": str(sz),
        },
        json={"name": remote_name},
        timeout=30,
    )
    r_init.raise_for_status()
    init_sec = time.perf_counter() - t0
    upload_url = r_init.headers["Location"]
    t1 = time.perf_counter()
    with open(path, "rb") as fh:
        r_put = session.put(
            upload_url,
            headers={
                "Content-Length": str(sz),
                "Content-Type": "application/x-tar",
            },
            data=fh,
            timeout=180,
        )
    put_sec = time.perf_counter() - t1
    elapsed = time.perf_counter() - t0
    r_put.raise_for_status()
    info = r_put.json()
    assert int(info["size"]) == sz, f"Size mismatch: {info}"
    assert info.get("md5Checksum") == expected_md5, f"MD5 mismatch: {info}"
    return {
        "method": "GDrive-REST-Resumable",
        "file_id": info["id"],
        "init_sec": round(init_sec, 4),
        "put_sec": round(put_sec, 4),
        "elapsed_sec": round(elapsed, 4),
        "throughput_mb_s": round((sz / 1e6) / elapsed, 2),
        "throughput_mib_s": round((sz / (1024 ** 2)) / elapsed, 2),
        "verified_md5": info.get("md5Checksum") == expected_md5,
        "verified_size": int(info["size"]) == sz,
    }


# Warmup 1 small 1MB upload to establish TLS session pool diagnostics
warmup_res = upload_media_single_post(t1_path, "warmup_t1.tar", tier_meta["T1_Single_Run"]["md5"])
requests.delete(f"https://www.googleapis.com/drive/v3/files/{warmup_res['file_id']}", headers=headers_auth, timeout=20)
print(f"[Remote] Warmup GDrive API complete ({warmup_res['elapsed_sec']:.3f}s)")

gdrive_uploads = {}
cleanup_file_ids = []

for code, desc, path in tiers:
    md5 = tier_meta[code]["md5"]
    # Run 3 trials for T3_12Run_Exp03 (exact Exp 03 bundle) and 2 trials for T1, T2, T4
    num_trials = 3 if code == "T3_12Run_Exp03" else 2
    media_trials = []
    resumable_trials = []
    for t_idx in range(1, num_trials + 1):
        res_m = upload_media_single_post(path, f"exp04_{code}_media_r{t_idx}.tar", md5)
        media_trials.append(res_m)
        if t_idx > 1:
            cleanup_file_ids.append(res_m["file_id"])
        print(f"  [{code}] Media Trial {t_idx}/{num_trials}: {res_m['elapsed_sec']:.3f}s ({res_m['throughput_mb_s']:.2f} MB/s)")

        res_r = upload_resumable_put(path, f"exp04_{code}_resumable_r{t_idx}.tar", md5)
        resumable_trials.append(res_r)
        cleanup_file_ids.append(res_r["file_id"])
        print(f"  [{code}] Resumable Trial {t_idx}/{num_trials}: {res_r['elapsed_sec']:.3f}s (init={res_r['init_sec']:.3f}s, put={res_r['put_sec']:.3f}s, {res_r['throughput_mb_s']:.2f} MB/s)")

    # Delete extra trial files on remote immediately, keep trial 1 of media for Local Stage-2 pull
    for fid in cleanup_file_ids:
        try:
            requests.delete(f"https://www.googleapis.com/drive/v3/files/{fid}", headers=headers_auth, timeout=20)
        except Exception:
            pass
    cleanup_file_ids.clear()

    m_times = [x["elapsed_sec"] for x in media_trials]
    r_times = [x["elapsed_sec"] for x in resumable_trials]
    m_mean = sum(m_times) / len(m_times)
    r_mean = sum(r_times) / len(r_times)
    sz = tier_meta[code]["raw_bytes"]

    gdrive_uploads[code] = {
        "stage2_file_id": media_trials[0]["file_id"],
        "media_trials": media_trials,
        "resumable_trials": resumable_trials,
        "media_mean_sec": round(m_mean, 4),
        "media_min_sec": round(min(m_times), 4),
        "media_max_sec": round(max(m_times), 4),
        "media_mean_mb_s": round((sz / 1e6) / m_mean, 2),
        "media_mean_mib_s": round((sz / (1024 ** 2)) / m_mean, 2),
        "resumable_mean_sec": round(r_mean, 4),
        "resumable_min_sec": round(min(r_times), 4),
        "resumable_max_sec": round(max(r_times), 4),
        "resumable_mean_mb_s": round((sz / 1e6) / r_mean, 2),
        "resumable_mean_mib_s": round((sz / (1024 ** 2)) / r_mean, 2),
    }

# Also benchmark local VM base64+json encoding CPU overhead in isolation on T3 (what Jupyter /api/contents does before sending bytes!)
t_b64_start = time.perf_counter()
raw_t3 = t3_path.read_bytes()
b64_str = base64.encodebytes(raw_t3).decode("ascii")
json_str = json.dumps({"name": "T3.tar", "content": b64_str, "format": "base64", "type": "file"})
vm_b64_json_serialize_sec = time.perf_counter() - t_b64_start
vm_b64_json_wire_bytes = len(json_str)
del raw_t3, b64_str, json_str
print(f"[Remote] Isolated Jupyter Base64+JSON serialization on T3 (122.42 MB): {vm_b64_json_serialize_sec:.3f}s -> wire={vm_b64_json_wire_bytes:,} B")

out_summary = {
    "gpu_name": gpu_name,
    "cpu_count": cpu_count,
    "drive_py_info": drive_py_info,
    "seed_pull_from_gdrive_sec": round(seed_pull_sec, 4),
    "seed_pull_from_gdrive_mb_s": round((t3_size / 1e6) / seed_pull_sec, 2),
    "vm_t3_isolated_b64_json_serialize_sec": round(vm_b64_json_serialize_sec, 4),
    "vm_t3_isolated_b64_json_wire_bytes": vm_b64_json_wire_bytes,
    "tiers": tier_meta,
    "gdrive_uploads": gdrive_uploads,
}

out_json_path = WORK_DIR / "remote_probe_results.json"
out_json_path.write_text(json.dumps(out_summary, indent=2))
print("REMOTE_PROBE_SUCCESS")
'''


def benchmark_http_jupyter_contents_download(
    proxy_url: str,
    proxy_token: str,
    creds: Credentials,
    remote_rel_path: str,
    expected_sha256: str,
    expected_raw_bytes: int,
) -> Dict[str, Any]:
    """
    Replicates `colab download` (`colab_cli/contents.py:74-91`) with fine-grained phase timers:
      1. TTFB (`ttfb_sec`): VM reads file, runs `base64.encodebytes()`, runs `json.dumps()`, and sends HTTP headers.
      2. Body Transfer (`body_transfer_sec`): Streaming the +36.84% inflated base64 JSON over the Runtime Proxy to local Mac.
      3. Decode & Disk Write (`decode_write_sec`): `json.loads()` + `base64.b64decode()` + local disk write.
    """
    url = f"{proxy_url.rstrip('/')}/api/contents/{remote_rel_path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "Accept": "application/json",
    }
    params = {
        "authuser": "0",
        "colab-runtime-proxy-token": proxy_token,
        "content": "1",
    }
    t0 = time.perf_counter()
    with requests.get(url, headers=headers, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        t_headers = time.perf_counter()
        ttfb_sec = t_headers - t0
        content_length_hdr = int(r.headers.get("Content-Length", "0"))
        chunks = []
        wire_bytes = 0
        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                chunks.append(chunk)
                wire_bytes += len(chunk)
        t_body = time.perf_counter()
        body_transfer_sec = t_body - t_headers

    raw_json_bytes = b"".join(chunks)
    payload = json.loads(raw_json_bytes)
    decoded_bytes = base64.b64decode(payload["content"])
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(decoded_bytes)
        tmp.flush()
    t_end = time.perf_counter()
    decode_write_sec = t_end - t_body
    total_sec = t_end - t0

    sha_actual = hashlib.sha256(decoded_bytes).hexdigest()
    assert len(decoded_bytes) == expected_raw_bytes, f"Raw size mismatch: {len(decoded_bytes)} vs {expected_raw_bytes}"
    assert sha_actual == expected_sha256, f"SHA256 mismatch: {sha_actual} vs {expected_sha256}"

    return {
        "method": "HTTP-Jupyter-Direct (/api/contents?content=1)",
        "raw_bytes": expected_raw_bytes,
        "wire_bytes": wire_bytes,
        "content_length_header": content_length_hdr,
        "wire_inflation_ratio": round(wire_bytes / expected_raw_bytes, 6),
        "wire_inflation_pct": round((wire_bytes / expected_raw_bytes - 1.0) * 100.0, 2),
        "ttfb_sec": round(ttfb_sec, 4),
        "body_transfer_sec": round(body_transfer_sec, 4),
        "decode_write_sec": round(decode_write_sec, 4),
        "total_sec": round(total_sec, 4),
        "effective_payload_mb_s": round((expected_raw_bytes / 1e6) / total_sec, 2),
        "effective_payload_mib_s": round((expected_raw_bytes / (1024 ** 2)) / total_sec, 2),
        "wire_mb_s": round((wire_bytes / 1e6) / total_sec, 2),
        "verified_sha256": sha_actual == expected_sha256,
    }


def benchmark_gdrive_to_local_stage2(
    creds: Credentials,
    file_id: str,
    expected_raw_bytes: int,
    expected_sha256: str,
    workers: int = 8,
    chunk_mib: int = 8,
) -> Dict[str, Any]:
    """
    Stage 2 of Method B2 (Executed AFTER `colab stop` -> 0.00 GPU CU!):
    Downloads raw binary from Google Drive v3 API (`/drive/v3/files/{fileId}?alt=media`)
    using HTTP Range requests (`workers=1` sequential or `workers=8` parallel).
    """
    chunk_size = chunk_mib * 1024 * 1024
    ranges: List[Tuple[int, int]] = []
    for s in range(0, expected_raw_bytes, chunk_size):
        ranges.append((s, min(s + chunk_size - 1, expected_raw_bytes - 1)))

    def fetch_range(start_end: Tuple[int, int]) -> Tuple[int, bytes]:
        start, end = start_end
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "X-Goog-User-Project": QUOTA_PROJECT,
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
    t_net = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(buf)
        tmp.flush()
    t_end = time.perf_counter()

    total_sec = t_end - t0
    sha_actual = hashlib.sha256(buf).hexdigest()
    assert sha_actual == expected_sha256, f"Stage 2 SHA256 mismatch for {file_id}"

    return {
        "workers": workers,
        "chunk_mib": chunk_mib,
        "num_chunks": len(ranges),
        "raw_bytes": expected_raw_bytes,
        "wire_bytes": expected_raw_bytes,
        "wire_inflation_ratio": 1.0,
        "net_transfer_sec": round(t_net - t0, 4),
        "disk_write_sec": round(t_end - t_net, 4),
        "total_sec": round(total_sec, 4),
        "effective_payload_mb_s": round((expected_raw_bytes / 1e6) / total_sec, 2),
        "effective_payload_mib_s": round((expected_raw_bytes / (1024 ** 2)) / total_sec, 2),
        "gpu_cu_consumed": 0.0,
        "verified_sha256": sha_actual == expected_sha256,
    }


def main() -> None:
    print("=" * 95)
    print("🚀 Experiment 04: Colab-to-Local HTTP Egress vs. Colab-to-Google-Drive Egress Benchmark")
    print("=" * 95)

    creds = get_fresh_credentials()
    pre_ccu = fetch_ccu_info(creds)
    print(
        f"[Ledger Pre-Check] Balance: {pre_ccu['response_body']['currentBalance']:.4f} CU | "
        f"Active Assignments: {pre_ccu['response_body']['assignmentsCount']}"
    )

    session_connected = False
    t_assign_start = time.time()
    t_unassign_done = time.time()
    remote_results: Dict[str, Any] = {}
    http_direct_results: Dict[str, Any] = {}

    try:
        print(f"\n[Step 1] Creating G4 GPU Session ({SESSION_NAME})...")
        t_assign_start = time.time()
        subprocess.run(
            [COLAB_BIN, "new", "-s", SESSION_NAME, "--gpu", "g4"],
            check=True,
        )
        session_connected = True

        sessions_data = json.loads(SESSIONS_PATH.read_text())
        s_info = sessions_data[SESSION_NAME]
        endpoint = s_info["endpoint"]
        proxy_url = s_info["url"]
        proxy_token = s_info["token"]
        print(f"[Step 1] Connected endpoint={endpoint}, proxy_url={proxy_url}")

        # Write remote probe script with fresh access token
        remote_code = (
            REMOTE_PROBE_TEMPLATE
            .replace("__ACCESS_TOKEN__", json.dumps(creds.token))
            .replace("__QUOTA_PROJECT__", json.dumps(QUOTA_PROJECT))
            .replace("__SEED_FILE_ID__", json.dumps(SEED_FILE_ID))
        )
        local_probe_script = OUTPUT_DIR / "_remote_egress_probe_runtime.py"
        local_probe_script.write_text(remote_code)

        print("\n[Step 2] Running Remote G4 -> Google Drive v3 API Benchmark across T1..T4...")
        subprocess.run(
            [COLAB_BIN, "exec", "-s", SESSION_NAME, "-f", str(local_probe_script), "--timeout", "300"],
            check=True,
        )
        local_probe_script.unlink(missing_ok=True)

        # Pull small remote_probe_results.json
        local_remote_json = OUTPUT_DIR / "remote_probe_results.json"
        subprocess.run(
            [
                COLAB_BIN,
                "download",
                "-s",
                SESSION_NAME,
                "/content/exp04_work/remote_probe_results.json",
                str(local_remote_json),
            ],
            check=True,
        )
        remote_results = json.loads(local_remote_json.read_text())

        print("\n[Step 3] Running Colab G4 -> Local Mac Direct HTTP (/api/contents) Benchmark on T1, T2, T3...")
        # Refresh session proxy token from sessions.json in case refreshed
        sessions_data = json.loads(SESSIONS_PATH.read_text())
        s_info = sessions_data[SESSION_NAME]
        proxy_url = s_info["url"]
        proxy_token = s_info["token"]

        for code in ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03"]:
            t_meta = remote_results["tiers"][code]
            r_http = benchmark_http_jupyter_contents_download(
                proxy_url=proxy_url,
                proxy_token=proxy_token,
                creds=creds,
                remote_rel_path=t_meta["remote_path"],
                expected_sha256=t_meta["sha256"],
                expected_raw_bytes=t_meta["raw_bytes"],
            )
            http_direct_results[code] = r_http
            print(
                f"  [{code}] HTTP Direct: total={r_http['total_sec']:.3f}s "
                f"(ttfb={r_http['ttfb_sec']:.3f}s, body={r_http['body_transfer_sec']:.3f}s, decode={r_http['decode_write_sec']:.3f}s) | "
                f"raw={r_http['raw_bytes']/1e6:.2f}MB -> wire={r_http['wire_bytes']/1e6:.2f}MB (+{r_http['wire_inflation_pct']:.2f}%) | "
                f"eff={r_http['effective_payload_mb_s']:.2f} MB/s"
            )

    finally:
        if session_connected:
            print(f"\n[Step 4] Immediately Stopping G4 Session ({SESSION_NAME}) to Halt GPU Billing...")
            subprocess.run([COLAB_BIN, "stop", "-s", SESSION_NAME], check=False)
            t_unassign_done = time.time()
            session_connected = False

    post_ccu_immediate = fetch_ccu_info(creds)
    vm_wall_sec = round(t_unassign_done - t_assign_start, 3)
    print(
        f"[Ledger Post-Check] Balance: {post_ccu_immediate['response_body']['currentBalance']:.4f} CU | "
        f"Active Assignments: {post_ccu_immediate['response_body']['assignmentsCount']} | "
        f"VM Wall Span: {vm_wall_sec:.2f}s"
    )

    print("\n[Step 5] GPU VM is STOPPED (0.00 CU/hr)! Running Stage 2 (Google Drive -> Local Mac) across T1..T4...")
    creds = get_fresh_credentials()
    stage2_results: Dict[str, Any] = {}
    for code in ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03", "T4_36Run_Sweep"]:
        t_meta = remote_results["tiers"][code]
        fid = remote_results["gdrive_uploads"][code]["stage2_file_id"]

        # Test parallel 8-worker HTTP Range pull (recommended zero-CU pull)
        r_par8 = benchmark_gdrive_to_local_stage2(
            creds=creds,
            file_id=fid,
            expected_raw_bytes=t_meta["raw_bytes"],
            expected_sha256=t_meta["sha256"],
            workers=8,
            chunk_mib=8,
        )
        print(
            f"  [{code}] Stage 2 GDrive->Local (8 workers, 0 CU): {r_par8['total_sec']:.3f}s "
            f"({r_par8['effective_payload_mb_s']:.2f} MB/s) | sha256_ok={r_par8['verified_sha256']}"
        )

        # Also test single-worker sequential pull on T1..T3 for direct 1-stream comparison
        r_seq1 = None
        if code in ("T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03"):
            r_seq1 = benchmark_gdrive_to_local_stage2(
                creds=creds,
                file_id=fid,
                expected_raw_bytes=t_meta["raw_bytes"],
                expected_sha256=t_meta["sha256"],
                workers=1,
                chunk_mib=16,
            )
            print(
                f"  [{code}] Stage 2 GDrive->Local (1 worker, 0 CU):  {r_seq1['total_sec']:.3f}s "
                f"({r_seq1['effective_payload_mb_s']:.2f} MB/s) | sha256_ok={r_seq1['verified_sha256']}"
            )

        stage2_results[code] = {
            "parallel_8w": r_par8,
            "sequential_1w": r_seq1,
        }

        # Delete temporary stage2 file from Google Drive
        try:
            requests.delete(
                f"https://www.googleapis.com/drive/v3/files/{fid}",
                headers={"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": QUOTA_PROJECT},
                timeout=20,
            )
        except Exception as e:
            print(f"  [Warn] Cleanup failed for {fid}: {e}")

    # Also delete the initial seed file from Google Drive
    try:
        requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{SEED_FILE_ID}",
            headers={"Authorization": f"Bearer {creds.token}", "X-Goog-User-Project": QUOTA_PROJECT},
            timeout=20,
        )
        print(f"[Cleanup] Deleted seed file {SEED_FILE_ID} and all temporary benchmark files from Google Drive.")
    except Exception:
        pass

    # Fit linear model for HTTP-Jupyter-Direct to project T4 (367.26 MB raw / 502.56 MB wire)
    # T_http(MB_raw) = alpha * MB_raw + beta
    xs = [remote_results["tiers"][c]["raw_mb"] for c in ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03"]]
    ys = [http_direct_results[c]["total_sec"] for c in ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03"]]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)
    intercept = y_mean - slope * x_mean
    t4_raw_mb = remote_results["tiers"]["T4_36Run_Sweep"]["raw_mb"]
    t4_raw_bytes = remote_results["tiers"]["T4_36Run_Sweep"]["raw_bytes"]
    t4_wire_bytes = remote_results["tiers"]["T4_36Run_Sweep"]["expected_http_contents_wire_bytes"]
    t4_http_proj_sec = round(slope * t4_raw_mb + intercept, 4)

    http_direct_results["T4_36Run_Sweep"] = {
        "method": "HTTP-Jupyter-Direct (Linear Fit Projection + Exp03 Rate)",
        "is_projected": True,
        "raw_bytes": t4_raw_bytes,
        "wire_bytes": t4_wire_bytes,
        "wire_inflation_ratio": round(t4_wire_bytes / t4_raw_bytes, 6),
        "wire_inflation_pct": round((t4_wire_bytes / t4_raw_bytes - 1.0) * 100.0, 2),
        "total_sec": t4_http_proj_sec,
        "effective_payload_mb_s": round((t4_raw_bytes / 1e6) / t4_http_proj_sec, 2),
        "wire_mb_s": round((t4_wire_bytes / 1e6) / t4_http_proj_sec, 2),
    }

    # Build comprehensive tier-by-tier comparison table
    comparison_by_tier = {}
    for code in ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03", "T4_36Run_Sweep"]:
        t_meta = remote_results["tiers"][code]
        gd_up = remote_results["gdrive_uploads"][code]
        http_d = http_direct_results[code]
        s2_par8 = stage2_results[code]["parallel_8w"]
        s2_seq1 = stage2_results[code]["sequential_1w"]

        # Method A: HTTP Direct (`colab download`)
        # GPU must stay alive for the entire HTTP download!
        a_gpu_sec = http_d["total_sec"]
        a_wall_sec = http_d["total_sec"]
        a_cu = a_gpu_sec * G4_CU_RATE_PER_SEC

        # Method B2: Google Drive REST Relay (Stage 1: G4->GDrive Media Upload + Immediate Stop; Stage 2: 0-CU Local Pull)
        b2_gpu_sec = gd_up["media_mean_sec"]
        b2_stage2_par8_sec = s2_par8["total_sec"]
        b2_wall_par8_sec = round(b2_gpu_sec + b2_stage2_par8_sec, 4)
        b2_cu = b2_gpu_sec * G4_CU_RATE_PER_SEC

        b2_resum_gpu_sec = gd_up["resumable_mean_sec"]
        b2_resum_cu = b2_resum_gpu_sec * G4_CU_RATE_PER_SEC

        # Method B1: Traditional FUSE `/content/drive` (`drive.mount` + `shutil.copy2` + `drive.flush_and_unmount`)
        # Even if already mounted earlier (0s mount at tail), FUSE daemon startup earlier burned 13.41s mean (11.60s-15.22s)
        # plus interactive OAuth wait (91.23s-131.11s in Exp01).
        fuse_daemon_init_sec = 13.41  # Mean of Exp 01 pure drive-fs mount post-auth (11.60s, 15.22s)
        fuse_unmount_overhead_sec = 2.50  # FUSE unmount + sqlite metadata sync overhead
        b1_already_mounted_tail_gpu_sec = round(b2_resum_gpu_sec + fuse_unmount_overhead_sec, 4)
        b1_with_daemon_init_gpu_sec = round(fuse_daemon_init_sec + b1_already_mounted_tail_gpu_sec, 4)

        comparison_by_tier[code] = {
            "tier_meta": t_meta,
            "method_A_http_direct": {
                "gpu_billable_sec": round(a_gpu_sec, 4),
                "wall_clock_sec": round(a_wall_sec, 4),
                "gpu_cu_at_7_2046": round(a_cu, 6),
                "effective_payload_mb_s": http_d["effective_payload_mb_s"],
                "wire_bytes": http_d["wire_bytes"],
                "is_projected": http_d.get("is_projected", False),
            },
            "method_B2_gdrive_rest_relay": {
                "stage1_gpu_billable_sec": round(b2_gpu_sec, 4),
                "stage1_resumable_gpu_sec": round(b2_resum_gpu_sec, 4),
                "stage1_backbone_mb_s": gd_up["media_mean_mb_s"],
                "stage2_off_gpu_par8_sec": round(b2_stage2_par8_sec, 4),
                "stage2_off_gpu_par8_mb_s": s2_par8["effective_payload_mb_s"],
                "stage2_off_gpu_seq1_sec": round(s2_seq1["total_sec"], 4) if s2_seq1 else None,
                "total_wall_clock_par8_sec": round(b2_wall_par8_sec, 4),
                "total_wall_clock_seq1_sec": round(b2_gpu_sec + s2_seq1["total_sec"], 4) if s2_seq1 else None,
                "gpu_cu_at_7_2046": round(b2_cu, 6),
                "gpu_time_speedup_vs_http": round(a_gpu_sec / b2_gpu_sec, 2),
                "gpu_cu_saving_pct_vs_http": round((1.0 - b2_cu / a_cu) * 100.0, 2),
                "gpu_cu_saved_abs": round(a_cu - b2_cu, 6),
                "wall_clock_speedup_par8_vs_http": round(a_wall_sec / b2_wall_par8_sec, 2),
            },
            "method_B1_gdrive_fuse_mount_ref": {
                "exp01_interactive_mount_total_sec_range": [91.23, 131.11],
                "exp01_pure_fuse_daemon_init_mean_sec": fuse_daemon_init_sec,
                "tail_flush_and_unmount_est_sec": b1_already_mounted_tail_gpu_sec,
                "total_gpu_sec_incl_daemon_init": b1_with_daemon_init_gpu_sec,
                "total_gpu_cu_incl_daemon_init": round(b1_with_daemon_init_gpu_sec * G4_CU_RATE_PER_SEC, 6),
            },
        }

    # Also compute analytical sensitivity across local network bandwidths (0.5 MB/s to 50 MB/s) for T3 (122.42 MB raw / 167.52 MB wire)
    t3_raw_mb = remote_results["tiers"]["T3_12Run_Exp03"]["raw_mb"]
    t3_wire_mb = remote_results["tiers"]["T3_12Run_Exp03"]["expected_http_contents_wire_mb"]
    t3_vm_serialize_sec = remote_results["vm_t3_isolated_b64_json_serialize_sec"]
    t3_b2_gpu_sec = remote_results["gdrive_uploads"]["T3_12Run_Exp03"]["media_mean_sec"]
    t3_b2_cu = t3_b2_gpu_sec * G4_CU_RATE_PER_SEC

    bandwidth_sensitivity_t3 = []
    for bw_mb_s in [0.5, 1.0, 2.0, 3.0, 5.0, 6.30, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        # HTTP /api/contents transmits t3_wire_mb (167.52 MB) over single stream after VM base64 serialization
        http_sec = t3_vm_serialize_sec + (t3_wire_mb / bw_mb_s)
        http_cu = http_sec * G4_CU_RATE_PER_SEC
        # GDrive Relay Stage 1 is fixed inside Google backbone (t3_b2_gpu_sec), independent of local bandwidth!
        # Stage 2 transmits raw binary t3_raw_mb (122.42 MB) at 0 CU!
        gdrive_stage2_sec = t3_raw_mb / bw_mb_s
        gdrive_wall_sec = t3_b2_gpu_sec + gdrive_stage2_sec
        bandwidth_sensitivity_t3.append({
            "local_wire_bandwidth_mb_s": bw_mb_s,
            "http_direct_gpu_sec": round(http_sec, 2),
            "http_direct_cu": round(http_cu, 5),
            "gdrive_relay_gpu_sec": round(t3_b2_gpu_sec, 2),
            "gdrive_relay_cu": round(t3_b2_cu, 5),
            "gdrive_relay_wall_1stream_sec": round(gdrive_wall_sec, 2),
            "cu_saved_abs": round(http_cu - t3_b2_cu, 5),
            "cu_saving_pct": round((1.0 - t3_b2_cu / http_cu) * 100.0, 2),
            "wall_clock_delta_sec": round(gdrive_wall_sec - http_sec, 2),
            "is_gdrive_faster_in_wall_clock_too": gdrive_wall_sec < http_sec,
        })

    master_output = {
        "experiment": "Experiment 04: Colab-to-Local HTTP Egress vs. Colab-to-Google-Drive Egress Relay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "session_name": SESSION_NAME,
            "endpoint": endpoint,
            "gpu_name": remote_results.get("gpu_name"),
            "cpu_count": remote_results.get("cpu_count"),
            "g4_cu_rate_per_hour_ref": G4_CU_RATE_PER_HOUR,
            "exp03_effective_rate_per_hour_ref": EXP03_EFFECTIVE_BILLED_RATE_PER_HOUR,
        },
        "session_ledger": {
            "pre_session_ccu_info": pre_ccu,
            "post_session_immediate_ccu_info": post_ccu_immediate,
            "assign_to_unassign_wall_sec": vm_wall_sec,
            "theoretical_session_cu_at_7_2046": round(vm_wall_sec * G4_CU_RATE_PER_SEC, 4),
            "immediate_ledger_deducted_cu": round(
                pre_ccu["response_body"]["currentBalance"] - post_ccu_immediate["response_body"]["currentBalance"],
                4,
            ),
        },
        "exp03_historical_baseline": {
            "tar_raw_bytes": 122419200,
            "http_contents_wire_bytes": 167521347,
            "wire_inflation_ratio": round(167521347 / 122419200, 6),
            "wire_inflation_pct": 36.84,
            "ttfb_sec": 2.114,
            "body_transfer_sec": 24.484,
            "total_download_sec": 26.598,
            "effective_payload_mb_s": 4.60,
            "wire_mb_s": 6.30,
            "gpu_cu_at_7_2046": 0.05323,
        },
        "remote_probe": remote_results,
        "http_direct_benchmarks": http_direct_results,
        "stage2_local_pulls": stage2_results,
        "comparison_by_tier": comparison_by_tier,
        "bandwidth_sensitivity_t3": bandwidth_sensitivity_t3,
    }

    metrics_path = OUTPUT_DIR / "exp04_egress_transfer_metrics.json"
    metrics_path.write_text(json.dumps(master_output, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved complete Experiment 04 metrics to {metrics_path}")


if __name__ == "__main__":
    main()
