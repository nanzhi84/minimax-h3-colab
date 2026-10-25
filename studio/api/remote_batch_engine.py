#!/usr/bin/env python3
"""
Cloud Blackwell G4 Production Batch & Single-Shot Rendering Engine (`studio/api/remote_batch_engine.py`).
Supports the 2x2 BF16 model-routing & operator-tier matrix (`model_route` in {"fl2va", "ref2va"} x
`profile` in {"quality", "fast"}) with staged Qwen3-VL-32B BF16 pre-encode + immediate VRAM unload.
"""
import argparse, concurrent.futures, gc, hashlib, json, logging, math, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

COMFY_ROOT = Path("/content/ComfyUI")
CUSTOM_SOL_DIR = COMFY_ROOT / "custom_nodes" / "comfyui_sol_attn"
COMFYUI_COMMIT = "f5ea88af1c46c50a39792d00af80e5af941f99f6"
COMMUNITY_SOL_COMMIT = "69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3"
PINNED_PROD_PACKAGES: List[str] = [
    "comfy-kitchen==0.2.40",
    "sageattention==1.0.6",
    "triton==3.6.0",
    "hf_transfer",
    "huggingface_hub",
    "opencv-python-headless",
    "scipy",
    "soundfile",
    "imageio[ffmpeg]",
    "psutil",
    "requests",
]
G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0
REPO_ID = "Comfy-Org/MiniMax-H3"
DEFAULT_BF16_BUNDLE_FILES: List[Dict[str, Any]] = [
    {"role": "FL2VA DiT (Pruned BF16)", "repo_id": REPO_ID, "rel_path": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", "target_rel_path": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", "size": 40225724176},
    {"role": "Text Encoder (Qwen3-VL-32B BF16)", "repo_id": REPO_ID, "rel_path": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", "target_rel_path": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", "size": 51506295256},
    {"role": "Video VAE (FP16)", "repo_id": REPO_ID, "rel_path": "vae/minimax_h3_video_vae_fp16.safetensors", "target_rel_path": "vae/minimax_h3_video_vae_fp16.safetensors", "size": 5207808496},
    {"role": "Audio VAE (FP32)", "repo_id": REPO_ID, "rel_path": "vae/minimax_h3_audio_vae_fp32.safetensors", "target_rel_path": "vae/minimax_h3_audio_vae_fp32.safetensors", "size": 605254808},
    {"role": "FL2V 8-Step Turbo LoRA (BF16)", "repo_id": REPO_ID, "rel_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "target_rel_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "size": 1956193000},
]


class FallbackAlertDetector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.fallback_errors: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if (
            "Error running sage attention" in msg
            or "falling back to pytorch attention" in msg
            or "[MiniMax H3 fusion] eager fallback" in msg
        ):
            self.fallback_errors.append(msg)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def pull_input_tar_from_gdrive(
    input_file_id: str, access_token: str, quota_project: str, work_dir: Path, expected_sha256: str = ""
) -> Dict[str, Any]:
    import requests
    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path, extract_dir = work_dir / "input_jobs.tar", work_dir / "input_jobs"
    extract_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{input_file_id}?alt=media",
        headers={"Authorization": f"Bearer {access_token}", "X-Goog-User-Project": quota_project},
        timeout=120,
    )
    r.raise_for_status()
    data = r.content
    tar_path.write_bytes(data)
    pull_sec = time.perf_counter() - t0
    sha_actual = hashlib.sha256(data).hexdigest()
    if expected_sha256 and sha_actual != expected_sha256:
        raise ValueError(f"input_jobs.tar SHA256 mismatch: {sha_actual} != {expected_sha256}")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(extract_dir)
    total_sec = time.perf_counter() - t0
    mb_s = (len(data) / 1e6) / max(pull_sec, 1e-6)
    print(f"  ⚡ [0-CU Ingress Pull] Pulled input_jobs.tar ({len(data)/1e6:.2f} MB) from GDrive in {pull_sec:.3f}s ({mb_s:.1f} MB/s)")
    return {
        "input_file_id": input_file_id,
        "input_tar_bytes": len(data),
        "input_tar_sha256": sha_actual,
        "datacenter_pull_sec": round(pull_sec, 4),
        "datacenter_pull_and_extract_sec": round(total_sec, 4),
        "datacenter_pull_mb_s": round(mb_s, 2),
    }


def bootstrap_and_parallel_ingress(
    input_file_id: str, access_token: str, quota_project: str, work_dir: Path, expected_sha256: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t_phase0 = time.perf_counter()
    print("=" * 96 + "\n🛠️ [Phase 1/4] Lean Production Bootstrap + <0.5s GDrive Job Pull + Route-Specific BF16 HF Download\n" + "=" * 96)
    t_env0 = time.perf_counter()
    if not COMFY_ROOT.exists():
        run_cmd(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)])
        run_cmd(["git", "checkout", COMFYUI_COMMIT], cwd=COMFY_ROOT)
    CUSTOM_SOL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_SOL_DIR.exists():
        run_cmd(["git", "clone", "https://github.com/r-vage/ComfyUI-sol-attn.git", str(CUSTOM_SOL_DIR)])
        run_cmd(["git", "checkout", COMMUNITY_SOL_COMMIT], cwd=CUSTOM_SOL_DIR)
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_ROOT / "requirements.txt")])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *PINNED_PROD_PACKAGES])
    env_bootstrap_sec = time.perf_counter() - t_env0

    ingress_stats = pull_input_tar_from_gdrive(input_file_id, access_token, quota_project, work_dir, expected_sha256)
    manifest = json.loads((work_dir / "input_jobs" / "batch_manifest_resolved.json").read_text(encoding="utf-8"))
    locked_cfg = manifest.get("locked_inference_config", {})
    bundle_files: List[Dict[str, Any]] = locked_cfg.get("weights_bundle") or DEFAULT_BF16_BUNDLE_FILES

    from huggingface_hub import hf_hub_download
    models_dir = COMFY_ROOT / "models"
    n_files = len(bundle_files)

    def dl_one_weight(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[str, int, float, bool]:
        idx, item = idx_item
        repo_id = item.get("repo_id", REPO_ID)
        rel_path = item["rel_path"]
        target_rel_path = item.get("target_rel_path", rel_path)
        fsize = int(item["size"])
        target_file = models_dir / target_rel_path
        if target_file.exists() and target_file.stat().st_size == fsize:
            print(f"  [{idx}/{n_files}] ✅ Cached {item['role']} ({fsize / 1e9:.2f} GB)")
            return item["role"], fsize, 0.0, True
        t0 = time.perf_counter()
        dl_str = hf_hub_download(repo_id=repo_id, revision="main", filename=rel_path, local_dir=str(models_dir))
        dl_path = Path(dl_str)
        if target_file.resolve() != dl_path.resolve():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dl_path), str(target_file))
        dt = time.perf_counter() - t0
        print(f"  [{idx}/{n_files}] ⬇️ {item['role']} ({fsize / 1e9:.2f} GB) in {dt:.1f}s ({(fsize / (1024**2)) / max(dt, 1e-6):.0f} MiB/s)")
        return item["role"], fsize, dt, False

    t_dl0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        weight_futures = [pool.submit(dl_one_weight, (i, item)) for i, item in enumerate(bundle_files, 1)]
        weight_results = [f.result() for f in weight_futures]
    shutil.rmtree(models_dir / ".cache", ignore_errors=True)
    wall_download_sec = time.perf_counter() - t_dl0
    phase1_total_sec = time.perf_counter() - t_phase0
    print(f"  ✅ Phase 1 completed in {phase1_total_sec:.2f}s (env={env_bootstrap_sec:.2f}s, dl_wall={wall_download_sec:.2f}s)")
    return {
        "env_bootstrap_sec": round(env_bootstrap_sec, 3),
        "wall_download_sec": round(wall_download_sec, 3),
        "sum_individual_dl_sec": round(sum(r[2] for r in weight_results), 3),
        "phase1_total_sec": round(phase1_total_sec, 3),
        "total_weight_bytes": sum(r[1] for r in weight_results),
        "ingress_gdrive_pull": ingress_stats,
    }, manifest


def clone_conditioning(cond: list) -> list:
    cloned = []
    for item in cond:
        dict_cond = {}
        for k, v in item[1].items():
            if k == "minimax_keyframes" and isinstance(v, list):
                dict_cond[k] = [
                    {"resolved_frame_index": kf["resolved_frame_index"], "latent": kf["latent"].clone() if kf.get("latent") is not None else None}
                    for kf in v
                ]
            elif k == "minimax_refs" and isinstance(v, list):
                dict_cond[k] = [
                    {rk: (rv.clone() if hasattr(rv, "clone") else rv) for rk, rv in ref_item.items()}
                    for ref_item in v
                ]
            elif hasattr(v, "clone"):
                dict_cond[k] = v.clone()
            else:
                dict_cond[k] = v
        cloned.append([item[0].clone(), dict_cond])
    return cloned


def clone_av_latent(base_latent: Dict[str, Any]) -> Dict[str, Any]:
    import comfy.nested_tensor
    v_lat, a_lat = base_latent["samples"].unbind()
    return {"samples": comfy.nested_tensor.NestedTensor((v_lat.clone(), a_lat.clone()))}


def compute_ssim_frame(img1: Any, img2: Any) -> float:
    import cv2, numpy as np
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    mu1, mu2 = cv2.GaussianBlur(img1, (11, 11), 1.5), cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq, sigma2_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq, cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    return float(np.mean(((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)))


def load_aligned_image_np(path: Path, width: int, height: int, resize_mode: str = "cover") -> Any:
    import numpy as np
    from PIL import Image, ImageOps
    img = Image.open(path).convert("RGB")
    if resize_mode == "cover":
        img = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)
    else:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_raw_image_tensor(path: Path) -> Any:
    import numpy as np, torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def mux_and_score_job_cpu(
    frames_np: Any,
    waveform_np: Any,
    sample_rate: int,
    first_np: Optional[Any],
    last_np: Optional[Any],
    out_mp4: Path,
    fps: int = 24,
) -> Dict[str, Any]:
    import cv2, imageio.v2 as imageio, numpy as np, soundfile as sf
    from scipy import ndimage
    t_mux0 = time.perf_counter()
    tmp_vid, tmp_wav = out_mp4.with_suffix(".tmp_video.mp4"), out_mp4.with_suffix(".tmp_audio.wav")
    frames_u8 = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(str(tmp_vid), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    for frame in frames_u8:
        writer.append_data(frame)
    writer.close()
    sf.write(str(tmp_wav), np.clip(waveform_np.T if waveform_np.ndim == 2 else waveform_np, -1.0, 1.0), sample_rate)
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(tmp_vid), "-i", str(tmp_wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_mp4)],
        check=True,
    )
    tmp_vid.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)
    mux_sec = time.perf_counter() - t_mux0

    t_q0 = time.perf_counter()
    first_ssim = round(compute_ssim_frame(frames_np[0], first_np), 4) if first_np is not None else None
    last_ssim = round(compute_ssim_frame(frames_np[-1], last_np), 4) if last_np is not None else None
    if first_ssim is not None and last_ssim is not None:
        s_fl2v_pixel: Optional[float] = round(100.0 * (0.40 * first_ssim + 0.60 * last_ssim), 2)
    elif first_ssim is not None:
        s_fl2v_pixel = round(100.0 * first_ssim, 2)
    elif last_ssim is not None:
        s_fl2v_pixel = round(100.0 * last_ssim, 2)
    else:
        s_fl2v_pixel = None

    frame_mae = np.mean(np.abs(frames_np[1:] - frames_np[:-1]), axis=(1, 2, 3))
    temporal_flickering_score = float(100.0 * (1.0 - np.mean(frame_mae)))
    dynamic_motion_px = float(np.mean(frame_mae) * 255.0)
    motion_smoothness = float(100.0 / (1.0 + np.mean(np.abs(np.diff(frame_mae))) / (np.mean(frame_mae) + 1e-4)))
    sample_idx = np.linspace(0, frames_u8.shape[0] - 1, min(9, frames_u8.shape[0]), dtype=int)
    sharpness_lap = float(np.mean([float(cv2.Laplacian(cv2.cvtColor(frames_u8[i], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()) for i in sample_idx]))
    sharpness_norm = min(100.0, 100.0 * (sharpness_lap / 260.0))

    L, T = waveform_np.shape[-1], frames_np.shape[0]
    spf = max(1, L // T)
    a_env = np.array([np.sqrt(np.mean(waveform_np[:, t * spf : (t + 1) * spf] ** 2) + 1e-12) for t in range(T)], dtype=np.float64)
    m_env = np.concatenate([[frame_mae[0]], frame_mae]).astype(np.float64)
    a_s, m_s = ndimage.uniform_filter1d(a_env, size=3), ndimage.uniform_filter1d(m_env, size=3)
    av_corr = float(np.corrcoef(a_s, m_s)[0, 1]) if (np.std(a_s) > 1e-6 and np.std(m_s) > 1e-6) else 0.0
    audio_rms_db = float(20.0 * math.log10(math.sqrt(float(np.mean(waveform_np ** 2)) + 1e-12)))
    if s_fl2v_pixel is not None:
        prod_quality_score = float(0.40 * s_fl2v_pixel + 0.25 * temporal_flickering_score + 0.20 * motion_smoothness + 0.15 * sharpness_norm)
    else:
        prod_quality_score = float(0.45 * temporal_flickering_score + 0.30 * motion_smoothness + 0.25 * sharpness_norm)
    q_sec = time.perf_counter() - t_q0
    mp4_bytes = out_mp4.stat().st_size
    return {
        "async_ffmpeg_mux_sec": round(mux_sec, 4),
        "async_cpu_quality_sec": round(q_sec, 4),
        "mp4_bytes": mp4_bytes,
        "mp4_mb": round(mp4_bytes / 1e6, 3),
        "first_frame_ssim": first_ssim,
        "last_frame_ssim": last_ssim,
        "S_fl2v_pixel": s_fl2v_pixel,
        "temporal_flickering_score": round(temporal_flickering_score, 2),
        "motion_smoothness_score": round(motion_smoothness, 2),
        "dynamic_motion_mae_255": round(dynamic_motion_px, 3),
        "sharpness_laplacian_var": round(sharpness_lap, 2),
        "audio_rms_db": round(audio_rms_db, 2),
        "av_sync_corr": round(av_corr, 4),
        "prod_quality_score": round(prod_quality_score, 2),
    }


DEFAULT_INPUT_FILE_ID = os.environ.get("STUDIO_INPUT_FILE_ID", "__STUDIO_INPUT_FILE_ID__")
DEFAULT_ACCESS_TOKEN = os.environ.get("STUDIO_ACCESS_TOKEN", "__STUDIO_ACCESS_TOKEN__")
DEFAULT_EXPECTED_SHA256 = os.environ.get("STUDIO_EXPECTED_SHA256", "__STUDIO_EXPECTED_SHA256__")
DEFAULT_QUOTA_PROJECT = os.environ.get("STUDIO_QUOTA_PROJECT", "__STUDIO_QUOTA_PROJECT__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio Cloud G4 Production Engine (2x2 BF16 Route/Profile Matrix)")
    parser.add_argument("--input-file-id", default=DEFAULT_INPUT_FILE_ID)
    parser.add_argument("--access-token", default=DEFAULT_ACCESS_TOKEN)
    parser.add_argument("--quota-project", default=DEFAULT_QUOTA_PROJECT)
    parser.add_argument("--expected-input-sha256", default=DEFAULT_EXPECTED_SHA256)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/studio_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/studio_outputs"))
    parser.add_argument("--output-tar", type=Path, default=Path("/content/studio_outputs_bundle.tar"))
    args, _ = parser.parse_known_args()

    t_script_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase1_stats, manifest = bootstrap_and_parallel_ingress(
        args.input_file_id, args.access_token, args.quota_project, args.work_dir, args.expected_input_sha256
    )

    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))
    import comfy_kitchen as ck, psutil, requests, torch
    import comfy.ldm.minimax.model as minimax_model_mod
    import comfy.ldm.modules.attention as comfy_attn_mod
    import comfy.model_management as mm
    import comfy.ops, nodes
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    import comfyui_sol_attn.minimax as community_sol_mod
    from comfyui_sol_attn.minimax import MiniMaxH3FusedModulation
    from sageattention import sageattn

    props = torch.cuda.get_device_properties(0)
    detector = FallbackAlertDetector()
    logging.getLogger().addHandler(detector)
    locked_cfg = manifest.get("locked_inference_config", {})
    model_route = str(locked_cfg.get("model_route", manifest.get("model_route", "fl2va")))
    profile_id = str(locked_cfg.get("profile_id", f"{model_route}_bf16_{manifest.get('profile', 'fast')}"))
    fps, steps = int(manifest["fps"]), int(manifest["steps"])
    scheduler_name = str(manifest.get("scheduler", locked_cfg.get("scheduler", "simple")))
    resize_mode = manifest.get("resize_mode", "cover")
    clip_file = Path(locked_cfg.get("clip_name", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors")).name
    video_vae_file = Path(locked_cfg.get("video_vae_name", "vae/minimax_h3_video_vae_fp16.safetensors")).name
    audio_vae_file = Path(locked_cfg.get("audio_vae_name", "vae/minimax_h3_audio_vae_fp32.safetensors")).name

    print("\n" + "=" * 96 + f"\n🧠 [Phase 2/4] Staged Conditioning Pre-Encode ({clip_file} + {video_vae_file}, route={model_route})\n" + "=" * 96)
    t_p2_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    t_vlm_load0 = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip(clip_file, type="minimax")[0]
    vae_loader = nodes.VAELoader()
    video_vae = vae_loader.load_vae(video_vae_file)[0]
    audio_vae = vae_loader.load_vae(audio_vae_file)[0]
    vlm_vae_load_sec = time.perf_counter() - t_vlm_load0

    shots_by_id = {s["shot_id"]: s for s in manifest["storyboards"]}
    input_stage_dir = args.work_dir / "input_jobs"
    cond_cache: Dict[str, Dict[str, Any]] = {}
    unique_encode_secs: List[float] = []

    for job in manifest["jobs"]:
        jid, sid = job["job_id"], job["shot_id"]
        shot = shots_by_id[sid]
        c_hash = shot["cond_hash"]
        s_w, s_h = int(shot["resolution"][0]), int(shot["resolution"][1])
        s_len = int(shot["num_frames"])

        if c_hash not in cond_cache:
            t_c0 = time.perf_counter()
            first_np, last_np = None, None
            first_t, last_t = None, None
            if shot.get("first_frame"):
                p_first = input_stage_dir / shot["first_frame"]
                first_np = load_aligned_image_np(p_first, s_w, s_h, resize_mode=resize_mode)
                first_t = torch.from_numpy(first_np).unsqueeze(0)
            if shot.get("last_frame"):
                p_last = input_stage_dir / shot["last_frame"]
                last_np = load_aligned_image_np(p_last, s_w, s_h, resize_mode=resize_mode)
                last_t = torch.from_numpy(last_np).unsqueeze(0)

            with torch.inference_mode():
                if model_route == "ref2va":
                    ref_imgs_dict: Dict[str, Any] = {}
                    for r_idx, rel_p in enumerate(shot.get("ref_images", [])):
                        p_ref = input_stage_dir / rel_p
                        ref_imgs_dict[f"ref_image_{r_idx}"] = load_raw_image_tensor(p_ref)
                        if r_idx == 0 and first_np is None:
                            first_np = load_aligned_image_np(p_ref, s_w, s_h, resize_mode=resize_mode)
                        elif r_idx == 1 and last_np is None:
                            last_np = load_aligned_image_np(p_ref, s_w, s_h, resize_mode=resize_mode)
                    b_cond, b_lat = MiniMaxH3ReferenceToVideo.execute(
                        clip=clip,
                        prompt=shot["prompt"],
                        width=s_w,
                        height=s_h,
                        length=s_len,
                        ref_image_size=str(shot.get("ref_image_size", "match")),
                        vae=video_vae,
                        audio_vae=audio_vae,
                        ref_images=ref_imgs_dict,
                    )
                else:
                    b_cond, b_lat = MiniMaxH3ImageToVideo.execute(
                        clip=clip,
                        vae=video_vae,
                        prompt=shot["prompt"],
                        width=s_w,
                        height=s_h,
                        length=s_len,
                        first_frame=first_t,
                        last_frame=last_t,
                    )
            torch.cuda.synchronize()
            enc_s = time.perf_counter() - t_c0
            unique_encode_secs.append(enc_s)
            cond_cache[c_hash] = {
                "shot_id": sid,
                "cond_hash": c_hash,
                "width": s_w,
                "height": s_h,
                "length": s_len,
                "base_cond": b_cond,
                "base_latent": b_lat,
                "first_np": first_np,
                "last_np": last_np,
                "encode_sec": enc_s,
            }
            print(f"  [Cond MISS] {jid} ({sid}, {s_w}x{s_h}x{s_len}f, hash={c_hash[:8]}): encoded in {enc_s:.2f}s")
        else:
            cached_entry = cond_cache[c_hash]
            print(f"  [Cond HIT ] {jid} ({sid}, hash={c_hash[:8]}): 0.0000s reuse (saved {cached_entry['encode_sec']:.2f}s!)")

    vram_before_vlm_unload_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    t_un0 = time.perf_counter()
    del clip
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vlm_unload_sec = time.perf_counter() - t_un0
    vram_after_vlm_unload_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    phase2_total_sec = time.perf_counter() - t_p2_start
    print(f"  ✅ Phase 2 complete in {phase2_total_sec:.2f}s | Freed Qwen3-VL VRAM: {vram_before_vlm_unload_gib - vram_after_vlm_unload_gib:.2f} GiB")

    unet_file = Path(locked_cfg.get("unet_name", "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors")).name
    lora_rel = locked_cfg.get("lora_name")
    dense_backend = str(locked_cfg.get("dense_backend", "sdpa"))
    sparse_mode = str(locked_cfg.get("sparse_mode", "none"))
    fused_mod = bool(locked_cfg.get("fused_mod", True))

    print("\n" + "=" * 96 + f"\n🔒 [Phase 3/4] Resident [{profile_id}] Injection ({unet_file}) & 1-Step Warmup\n" + "=" * 96)
    t_dit0 = time.perf_counter()
    base_model = nodes.UNETLoader().load_unet(unet_file, "default")[0]
    if lora_rel:
        lora_file = Path(lora_rel).name
        resident_model = nodes.LoraLoaderModelOnly().load_lora_model_only(base_model, lora_file, float(locked_cfg.get("lora_strength", 1.0)))[0]
    else:
        resident_model = base_model.clone()

    kernel_counters = {"sdpa": 0, "sage": 0, "ck_int8": 0, "ck_sol_chunked": 0, "ck_sol_direct": 0, "triton_sol": 0, "fused_mod_blocks": 0}

    def wrap_counter(key: str, fn):
        def _wrapped(*a, **kw):
            kernel_counters[key] += 1
            return fn(*a, **kw)
        return _wrapped

    comfy_attn_mod.sageattn = wrap_counter("sage", sageattn)
    comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True
    comfy.ops.scaled_dot_product_attention = wrap_counter("sdpa", comfy.ops.scaled_dot_product_attention)
    ck.int8_attention = wrap_counter("ck_int8", ck.int8_attention)
    ck.int8_attention_from_prequantized = wrap_counter("ck_int8", ck.int8_attention_from_prequantized)
    ck.sol_attn_chunked = wrap_counter("ck_sol_chunked", ck.sol_attn_chunked)
    ck.sol_attn = wrap_counter("ck_sol_direct", ck.sol_attn)
    community_sol_mod.sol_attn = wrap_counter("triton_sol", community_sol_mod.sol_attn)
    community_sol_mod._FusionLog.hit = wrap_counter("fused_mod_blocks", community_sol_mod._FusionLog.hit)

    if dense_backend == "sage":
        comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_sage
        minimax_model_mod.optimized_attention = comfy_attn_mod.attention_sage
    elif dense_backend == "ck_int8":
        comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_comfy_kitchen_int8
        minimax_model_mod.optimized_attention = comfy_attn_mod.attention_comfy_kitchen_int8
    else:
        comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_pytorch
        minimax_model_mod.optimized_attention = comfy_attn_mod.attention_pytorch

    if sparse_mode in ("ck_sol_direct", "ck_sol_chunked"):
        resident_model = BlockSparseAttention.execute(
            resident_model.clone(),
            selection={"selection": "sol-attn", "tau": float(locked_cfg.get("tau", 1.3))},
            start_percent=float(locked_cfg.get("start_percent", 0.20)),
            end_percent=float(locked_cfg.get("end_percent", 1.0)),
            dense_blocks=str(locked_cfg.get("dense_blocks", "")),
            min_tokens=int(locked_cfg.get("min_tokens", 4096)),
            extra_tokens=int(locked_cfg.get("extra_tokens", 256)),
            sink_conditioning=str(locked_cfg.get("sink_conditioning", "exact_kv_and_rows")),
            verbose=False,
        )[0]
        if sparse_mode == "ck_sol_direct":
            resident_model.model_options["transformer_options"].get("patches_replace", {}).pop("dit", None)

    if fused_mod:
        resident_model = MiniMaxH3FusedModulation().patch(model=resident_model, enabled=True)[0]

    mm.load_models_gpu([resident_model])
    torch.cuda.synchronize()
    dit_load_and_patch_sec = time.perf_counter() - t_dit0

    sampler_obj = KSamplerSelect.execute("res_multistep")[0]
    vae_decoder = nodes.VAEDecode()
    t_warm0 = time.perf_counter()
    first_cache_entry = next(iter(cond_cache.values()))
    warm_cond, warm_lat = clone_conditioning(first_cache_entry["base_cond"]), clone_av_latent(first_cache_entry["base_latent"])
    with torch.inference_mode():
        warm_sigmas = BasicScheduler.execute(resident_model, scheduler_name, 1, 1.0)[0]
        warm_guider = BasicGuider.execute(resident_model, warm_cond)[0]
        _, warm_den = SamplerCustomAdvanced.execute(RandomNoise.execute(42)[0], warm_guider, sampler_obj, warm_sigmas, warm_lat)
        _ = vae_decoder.decode(video_vae, warm_den)[0]
        _ = VAEDecodeAudio.execute(audio_vae, warm_den)[0]
    torch.cuda.synchronize()
    del warm_cond, warm_lat, warm_guider, warm_den
    gc.collect()
    torch.cuda.empty_cache()
    if detector.fallback_errors:
        raise RuntimeError(f"Fallback detected during 1-step warmup: {detector.fallback_errors}")
    warmup_1step_sec = time.perf_counter() - t_warm0
    print(f"  🔥 1-Step [{profile_id}] + Video/Audio VAE warmup finished in {warmup_1step_sec:.2f}s")

    print("\n" + "=" * 96 + f"\n🚀 [Phase 4/4] Mode P1 Continuous Rendering ({len(manifest['jobs'])} Jobs: Resident [{profile_id}] + Async CPU ffmpeg)\n" + "=" * 96)
    mp4_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    mp4_futures: Dict[str, concurrent.futures.Future] = {}
    job_metrics_list: List[Dict[str, Any]] = []
    t_batch_loop0 = time.perf_counter()

    for idx, job in enumerate(manifest["jobs"], 1):
        jid, sid, seed = job["job_id"], job["shot_id"], int(job["seed"])
        c_entry = cond_cache[job["cond_hash"]]
        print(f"\n▶️ [Job {idx:02d}/{len(manifest['jobs']):02d}] {jid} | shot={sid} ({c_entry['width']}x{c_entry['height']}x{c_entry['length']}f) | seed={seed}")
        t_sw0 = time.perf_counter()
        cond_copy, latent_copy = clone_conditioning(c_entry["base_cond"]), clone_av_latent(c_entry["base_latent"])
        noise_obj = RandomNoise.execute(seed)[0]
        detector.fallback_errors.clear()
        for k in kernel_counters:
            kernel_counters[k] = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        switch_prep_sec = time.perf_counter() - t_sw0

        t_j0 = time.perf_counter()
        with torch.inference_mode():
            sigmas = BasicScheduler.execute(resident_model, scheduler_name, steps, 1.0)[0]
            guider = BasicGuider.execute(resident_model, cond_copy)[0]
            _, out_denoised = SamplerCustomAdvanced.execute(noise_obj, guider, sampler_obj, sigmas, latent_copy)
            torch.cuda.synchronize()
            t_denoise = time.perf_counter() - t_j0
            dit_calls = dict(kernel_counters)

            t_v0 = time.perf_counter()
            video_frames_t = vae_decoder.decode(video_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_video = time.perf_counter() - t_v0

            t_a0 = time.perf_counter()
            audio_dict = VAEDecodeAudio.execute(audio_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_audio = time.perf_counter() - t_a0

        t_gpu_e2e = time.perf_counter() - t_j0
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if detector.fallback_errors:
            raise RuntimeError(f"Fallback detected in {jid}: {detector.fallback_errors}")

        t_cpu0 = time.perf_counter()
        frames_np = video_frames_t.detach().cpu().float().numpy()
        if frames_np.ndim == 5:
            frames_np = frames_np[0]
        waveform_np = audio_dict["waveform"].detach().cpu().float().numpy()
        if waveform_np.ndim == 3:
            waveform_np = waveform_np[0]
        sample_rate = int(audio_dict["sample_rate"])
        del guider, out_denoised, video_frames_t, audio_dict

        mp4_path = args.output_dir / f"{jid}.mp4"
        mp4_futures[jid] = mp4_pool.submit(
            mux_and_score_job_cpu, frames_np, waveform_np, sample_rate, c_entry["first_np"], c_entry["last_np"], mp4_path, fps
        )
        cpu_handoff_sec = time.perf_counter() - t_cpu0
        cu_den, cu_e2e = t_denoise * G4_CU_RATE_PER_SEC, t_gpu_e2e * G4_CU_RATE_PER_SEC
        print(
            f"   ✅ Denoise: {t_denoise:.2f}s ({cu_den:.4f} CU) | VideoVAE: {t_vae_video:.2f}s | AudioVAE: {t_vae_audio:.2f}s | "
            f"GPU E2E: {t_gpu_e2e:.2f}s ({cu_e2e:.4f} CU) | PeakVRAM: {peak_vram_gib:.2f} GiB"
        )
        job_metrics_list.append({
            "index": idx,
            "job_id": jid,
            "shot_id": sid,
            "model_route": model_route,
            "profile_id": profile_id,
            "resolution": [c_entry["width"], c_entry["height"]],
            "num_frames": c_entry["length"],
            "cond_hash": job["cond_hash"],
            "seed": seed,
            "note": job.get("note", ""),
            "locked_config": profile_id,
            "switch_prep_sec": round(switch_prep_sec, 5),
            "denoise_sec": round(t_denoise, 4),
            "vae_video_sec": round(t_vae_video, 4),
            "vae_audio_sec": round(t_vae_audio, 4),
            "vae_total_sec": round(t_vae_video + t_vae_audio, 4),
            "gpu_e2e_sec": round(t_gpu_e2e, 4),
            "cpu_handoff_sec": round(cpu_handoff_sec, 4),
            "cu_denoise": round(cu_den, 6),
            "cu_e2e": round(cu_e2e, 6),
            "peak_vram_gib": round(peak_vram_gib, 3),
            "kernel_calls": dit_calls,
            "mp4_file": mp4_path.name,
        })

    batch_loop_wall_sec = time.perf_counter() - t_batch_loop0
    for j in job_metrics_list:
        j.update(mp4_futures[j["job_id"]].result())
    mp4_pool.shutdown(wait=True)

    by_shot: Dict[str, List[Dict[str, Any]]] = {}
    for j in job_metrics_list:
        by_shot.setdefault(j["shot_id"], []).append(j)
    shot_best_seeds: Dict[str, Any] = {}
    for sid, jlist in by_shot.items():
        ranked = sorted(jlist, key=lambda x: x["prod_quality_score"], reverse=True)
        for rank_idx, item in enumerate(ranked, 1):
            item["shot_seed_rank"] = rank_idx
            item["is_recommended_seed_for_shot"] = rank_idx == 1
        shot_best_seeds[sid] = {
            "recommended_job_id": ranked[0]["job_id"],
            "recommended_seed": ranked[0]["seed"],
            "prod_quality_score": ranked[0]["prod_quality_score"],
            "S_fl2v_pixel": ranked[0]["S_fl2v_pixel"],
            "num_candidates": len(ranked),
        }

    sum_jobs_gpu_e2e_sec = sum(j["gpu_e2e_sec"] for j in job_metrics_list)
    cold_start_fixed_sec = phase1_stats["phase1_total_sec"] + vlm_vae_load_sec + vlm_unload_sec + dit_load_and_patch_sec + warmup_1step_sec
    payload = {
        "batch_id": manifest["batch_id"],
        "title": manifest["title"],
        "model_route": model_route,
        "profile_id": profile_id,
        "locked_inference_config": manifest["locked_inference_config"],
        "hardware": {
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_gib": round(props.total_memory / (1024 ** 3), 2),
            "ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "g4_cu_rate_per_hour": G4_CU_RATE_PER_HOUR,
        },
        "lifecycle_telemetry": {
            **phase1_stats,
            "phase2_cond_total_sec": round(phase2_total_sec, 3),
            "dit_load_and_patch_sec": round(dit_load_and_patch_sec, 3),
            "warmup_1step_sec": round(warmup_1step_sec, 3),
            "cold_start_fixed_total_sec": round(cold_start_fixed_sec, 3),
            "cold_start_fixed_cu": round(cold_start_fixed_sec * G4_CU_RATE_PER_SEC, 4),
            "batch_loop_wall_sec": round(batch_loop_wall_sec, 3),
            "sum_jobs_gpu_e2e_sec": round(sum_jobs_gpu_e2e_sec, 3),
            "script_pre_egress_wall_sec": round(time.perf_counter() - t_script_start, 3),
        },
        "storyboard_seed_recommendations": shot_best_seeds,
        "jobs": job_metrics_list,
    }
    (args.output_dir / "studio_batch_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "manifest_results.json").write_text(
        json.dumps({"batch_id": manifest["batch_id"], "model_route": model_route, "profile_id": profile_id, "status": "completed", "jobs": job_metrics_list, "recommendations": shot_best_seeds}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    t_tar0 = time.perf_counter()
    subprocess.run(["tar", "-cf", str(args.output_tar), "-C", str(args.output_dir), "."], check=True)
    tar_pack_sec = time.perf_counter() - t_tar0
    tar_data = args.output_tar.read_bytes()
    sz, md5, sha256 = len(tar_data), hashlib.md5(tar_data).hexdigest(), hashlib.sha256(tar_data).hexdigest()
    t_up0 = time.perf_counter()
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum",
        headers={"Authorization": f"Bearer {args.access_token}", "X-Goog-User-Project": args.quota_project, "Content-Type": "application/x-tar", "Content-Length": str(sz)},
        data=tar_data,
        timeout=180,
    )
    stage1_sec = time.perf_counter() - t_up0
    r.raise_for_status()
    info = r.json()
    assert int(info["size"]) == sz and info.get("md5Checksum") == md5
    stage1_mb_s = (sz / 1e6) / max(stage1_sec, 1e-6)
    print(f"  ⚡ [ADR-04 Stage 1 Complete] Uploaded {sz/1e6:.2f} MB to GDrive in {stage1_sec:.3f}s ({stage1_mb_s:.1f} MB/s)")
    print("STUDIO_STAGE1_EGRESS_JSON:" + json.dumps({"file_id": info["id"], "raw_bytes": sz, "md5": md5, "sha256": sha256, "tar_pack_sec": round(tar_pack_sec, 4), "stage1_sec": round(stage1_sec, 4), "stage1_mb_s": round(stage1_mb_s, 2)}))


if __name__ == "__main__":
    main()
