#!/usr/bin/env python3
"""
Experiment 07 Configuration, 2x2 BF16 Dual-Route Matrix & Parallel 7-Weight Bootstrap
(`experiments/07-bf16-dual-route-2x2-ablation/code/exp07_config.py`).

Defines the 2x2 native BF16 ablation matrix on Google Colab Blackwell G4 (--gpu G4):
  - Run 01 [R1-FL2VA-BF16-Quality]: fl2va_pruned_bf16 + 20 steps (simple) + cuDNN BF16 SDPA + FusedMod
  - Run 02 [R2-FL2VA-BF16-Fast]   : fl2va_pruned_bf16 + 8-step FL2V Turbo LoRA (simple) + Run 12 Sol-Attn (tau=1.3) + FusedMod
  - Run 03 [R3-Ref2VA-BF16-Quality]: ref2va_pruned_bf16 + 20 steps (beta) + cuDNN BF16 SDPA + FusedMod
  - Run 04 [R4-Ref2VA-BF16-Fast]   : ref2va_pruned_bf16 + 8-step Ref2V v1.1 Turbo LoRA (simple) + Run 09 HiFi Sol-Attn (tau=1.0, prot="0,1,48,49") + FusedMod
"""

import concurrent.futures
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

COMFY_ROOT = Path("/content/ComfyUI")
CUSTOM_SOL_DIR = COMFY_ROOT / "custom_nodes" / "comfyui_sol_attn"
OUTPUT_DIR = Path("/content/exp07_outputs")

COMFYUI_COMMIT = "f5ea88af1c46c50a39792d00af80e5af941f99f6"
COMMUNITY_SOL_COMMIT = "69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3"

PINNED_PACKAGES: List[str] = [
    "comfy-kitchen==0.2.40",
    "sageattention==1.0.6",
    "triton==3.6.0",
    "hf_transfer",
    "huggingface_hub",
    "lpips",
    "open_clip_torch",
    "pyiqa",
    "scikit-image",
    "scipy",
    "soundfile",
    "imageio[ffmpeg]",
    "matplotlib",
    "psutil",
    "requests",
]

G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0
EXP03_INT8_SDPA_8STEP_E2E_SEC = 82.1030
EXP03_INT8_RUN12_8STEP_E2E_SEC = 44.1972

REPO_ID = "Comfy-Org/MiniMax-H3"
LIGHTX2V_TURBO_REPO_ID = "lightx2v/Minimax-h3-Turbo"

# 7 unique BF16 / FP16 / FP32 weight files across the 2x2 matrix (141.70 GB / 131.97 GiB total)
EXP07_BF16_WEIGHT_FILES: List[Dict[str, Any]] = [
    {
        "role": "FL2VA DiT (Pruned BF16)",
        "repo_id": REPO_ID,
        "rel_path": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "target_rel_path": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "size": 40225724176,
    },
    {
        "role": "Ref2VA DiT (Pruned BF16)",
        "repo_id": REPO_ID,
        "rel_path": "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
        "target_rel_path": "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
        "size": 40225724176,
    },
    {
        "role": "Text Encoder (Qwen3-VL-32B BF16)",
        "repo_id": REPO_ID,
        "rel_path": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "target_rel_path": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "size": 51506295256,
    },
    {
        "role": "Video VAE (FP16)",
        "repo_id": REPO_ID,
        "rel_path": "vae/minimax_h3_video_vae_fp16.safetensors",
        "target_rel_path": "vae/minimax_h3_video_vae_fp16.safetensors",
        "size": 5207808496,
    },
    {
        "role": "Audio VAE (FP32)",
        "repo_id": REPO_ID,
        "rel_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "target_rel_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "size": 605254808,
    },
    {
        "role": "FL2V 8-Step Turbo LoRA (BF16)",
        "repo_id": REPO_ID,
        "rel_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "target_rel_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "size": 1956193000,
    },
    {
        "role": "Ref2V 8-Step Turbo v1.1 768p LoRA (BF16)",
        "repo_id": LIGHTX2V_TURBO_REPO_ID,
        "rel_path": "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
        "target_rel_path": "loras/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
        "size": 1956193000,
    },
]

# Prompt for FL2VA route: strictly conforms to `.agents/skills/h3-prompt-writing/references/base-en.txt`
OFFICIAL_FL2VA_PROMPT = (
    "<integrated_multimodal_description>"
    "0-5 seconds: A medium tracking shot glides smoothly forward along a rain-slicked cyberpunk alley at night "
    "as a young street guitarist in a dark leather jacket strums an acoustic guitar beneath glowing cyan and magenta neon signs; "
    "warm amber sparks drift upward around him while crisp acoustic guitar chords and gentle fingerpicking resonate over the soft pattering "
    "of rain on wet pavement, and the camera gradually pushes in to a close-up of his expressive face illuminated by golden neon reflections."
    "</integrated_multimodal_description>\n"
    "<overall_soundscape>"
    "Steady light rain pattering on wet asphalt and distant city ambiance blending naturally with the warm room resonance of an acoustic guitar."
    "</overall_soundscape>\n"
    "<non_diegetic_music>None.</non_diegetic_music>"
)

# Prompt for Ref2VA route: strictly conforms to `.agents/skills/h3-prompt-writing/references/ref-en.txt` (6-section format)
OFFICIAL_REF2VA_PROMPT = (
    "[Visual Reference Description]\n"
    "<Picture 1>: Medium shot of a young street guitarist in a dark leather jacket playing an acoustic guitar in a rain-slicked cyberpunk alley at night under glowing cyan and magenta neon signs.\n"
    "<Picture 2>: Close-up portrait of the same street guitarist's expressive face and acoustic guitar illuminated by warm golden amber neon reflections and drifting sparks.\n\n"
    "[Audio Reference Description]\n"
    "None.\n\n"
    "[Integrated Video-Audio Description]\n"
    "0-5 seconds: Starting from the medium alley framing of the street guitarist in <Picture 1>, the camera glides smoothly forward along the rain-slicked cyberpunk street as he strums his acoustic guitar beneath glowing cyan and magenta neon signs; warm amber sparks drift upward around him while crisp acoustic guitar chords and gentle fingerpicking resonate over the soft pattering of rain on wet pavement, naturally transitioning into the warm golden close-up composition of <Picture 2>.\n\n"
    "[Overall Soundscape]\n"
    "Steady light rain pattering on wet asphalt and distant city ambiance blending naturally with the warm resonance of an acoustic guitar.\n\n"
    "[Background Music]\n"
    "None.\n\n"
    "[Video Style]\n"
    "Cinematic cyberpunk night photography, photorealistic lighting, shallow depth of field, rich neon reflections, 24fps film look."
)

# 2x2 BF16 Ablation Matrix (4 canonical profiles)
EXP07_2X2_MATRIX: List[Dict[str, Any]] = [
    {
        "run_idx": 1,
        "id": "Run01_FL2VA_BF16_Quality_20Step_SDPA",
        "short": "[R1-FL2VA-BF16-Quality]",
        "profile_id": "fl2va_bf16_quality",
        "model_route": "fl2va",
        "profile": "quality",
        "unet_file": "minimax_h3_fl2va_pruned_bf16.safetensors",
        "lora_file": None,
        "steps": 20,
        "scheduler": "simple",
        "dense_backend": "sdpa",
        "sparse_mode": "none",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "tau": 0.0,
        "extra_tokens": 0,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": True,
        "route_baseline_id": "Run01_FL2VA_BF16_Quality_20Step_SDPA",
        "notes": "FL2VA BF16 Quality ceiling: Pruned BF16 DiT + Qwen3-VL BF16 + 20 steps (simple, no LoRA) + cuDNN BF16 SDPA + FusedMod",
    },
    {
        "run_idx": 2,
        "id": "Run02_FL2VA_BF16_Fast_8Step_Run12",
        "short": "[R2-FL2VA-BF16-Fast]",
        "profile_id": "fl2va_bf16_fast",
        "model_route": "fl2va",
        "profile": "fast",
        "unet_file": "minimax_h3_fl2va_pruned_bf16.safetensors",
        "lora_file": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "steps": 8,
        "scheduler": "simple",
        "dense_backend": "sage",
        "sparse_mode": "ck_sol_direct",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 256,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": True,
        "route_baseline_id": "Run01_FL2VA_BF16_Quality_20Step_SDPA",
        "notes": "FL2VA BF16 Fast Pareto: Pruned BF16 DiT + 8-step FL2V Turbo LoRA + Run 12 (20% Sage warmup + 80% direct ck.sol_attn tau=1.3 tok=256) + FusedMod",
    },
    {
        "run_idx": 3,
        "id": "Run03_Ref2VA_BF16_Quality_20Step_SDPA",
        "short": "[R3-Ref2VA-BF16-Quality]",
        "profile_id": "ref2va_bf16_quality",
        "model_route": "ref2va",
        "profile": "quality",
        "unet_file": "minimax_h3_ref2va_pruned_bf16.safetensors",
        "lora_file": None,
        "steps": 20,
        "scheduler": "beta",
        "dense_backend": "sdpa",
        "sparse_mode": "none",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "tau": 0.0,
        "extra_tokens": 0,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": True,
        "route_baseline_id": "Run03_Ref2VA_BF16_Quality_20Step_SDPA",
        "notes": "Ref2VA BF16 Quality ceiling: Pruned BF16 DiT + Qwen3-VL BF16 + 20 steps (beta, no LoRA) + cuDNN BF16 SDPA + FusedMod",
    },
    {
        "run_idx": 4,
        "id": "Run04_Ref2VA_BF16_Fast_8Step_Run09HiFi",
        "short": "[R4-Ref2VA-BF16-Fast]",
        "profile_id": "ref2va_bf16_fast",
        "model_route": "ref2va",
        "profile": "fast",
        "unet_file": "minimax_h3_ref2va_pruned_bf16.safetensors",
        "lora_file": "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
        "steps": 8,
        "scheduler": "simple",
        "dense_backend": "sage",
        "sparse_mode": "ck_sol_direct",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.0,
        "extra_tokens": 256,
        "dense_blocks": "0,1,48,49",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": True,
        "route_baseline_id": "Run03_Ref2VA_BF16_Quality_20Step_SDPA",
        "notes": "Ref2VA BF16 Fast HiFi Pareto: Pruned BF16 DiT + 8-step Ref2V v1.1 768p Turbo LoRA + Run 09 HiFi (20% Sage + 80% direct ck.sol_attn tau=1.0 tok=256 prot=0,1,48,49) + FusedMod",
    },
]


class FallbackAlertDetector(logging.Handler):
    """Detects silent attention or Triton modulation fallbacks during generation."""

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


def _prefetch_eval_weights_bg() -> float:
    t0 = time.perf_counter()
    import lpips
    import open_clip
    import pyiqa
    import torch
    import torchvision.models.optical_flow as of_models

    _ = lpips.LPIPS(net="alex", verbose=False)
    _ = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    _ = of_models.raft_small(weights=of_models.Raft_Small_Weights.DEFAULT)
    _ = pyiqa.create_metric("musiq", as_loss=False)
    return time.perf_counter() - t0


def bootstrap_and_download_bf16_parallel() -> Dict[str, Any]:
    """Clones pinned ComfyUI + ComfyUI-sol-attn and downloads all 7 BF16 weight files in parallel."""
    t_phase0 = time.perf_counter()
    print("=" * 96)
    print("🛠️ [Phase 1/5] Environment Bootstrap & 8-Thread Parallel BF16 Weight Download (141.70 GB)")
    print("=" * 96)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_env0 = time.perf_counter()
    if not COMFY_ROOT.exists():
        run_cmd(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)])
        run_cmd(["git", "checkout", COMFYUI_COMMIT], cwd=COMFY_ROOT)
    CUSTOM_SOL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_SOL_DIR.exists():
        run_cmd(["git", "clone", "https://github.com/r-vage/ComfyUI-sol-attn.git", str(CUSTOM_SOL_DIR)])
        run_cmd(["git", "checkout", COMMUNITY_SOL_COMMIT], cwd=CUSTOM_SOL_DIR)

    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_ROOT / "requirements.txt")])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES])
    env_bootstrap_sec = time.perf_counter() - t_env0

    from huggingface_hub import hf_hub_download
    models_dir = COMFY_ROOT / "models"
    n_files = len(EXP07_BF16_WEIGHT_FILES)

    def _download_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[str, int, float, bool]:
        idx, item = idx_item
        repo_id = item["repo_id"]
        rel_path = item["rel_path"]
        target_rel_path = item["target_rel_path"]
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
        speed_mbs = (fsize / (1024 ** 2)) / max(dt, 1e-6)
        print(f"  [{idx}/{n_files}] ⬇️ Downloaded {item['role']} ({fsize / 1e9:.2f} GB) in {dt:.1f}s ({speed_mbs:.1f} MiB/s)")
        return item["role"], fsize, dt, False

    t_dl0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        eval_fut = pool.submit(_prefetch_eval_weights_bg)
        weight_futs = [pool.submit(_download_one, (i, item)) for i, item in enumerate(EXP07_BF16_WEIGHT_FILES, 1)]
        dl_results = [f.result() for f in weight_futs]
        eval_prefetch_sec = eval_fut.result()

    shutil.rmtree(models_dir / ".cache", ignore_errors=True)
    wall_download_sec = time.perf_counter() - t_dl0
    sum_single_dl_sec = sum(r[2] for r in dl_results)
    phase1_total_sec = time.perf_counter() - t_phase0
    print(
        f"  ✅ Phase 1 completed in {phase1_total_sec:.2f}s "
        f"(env={env_bootstrap_sec:.2f}s, parallel_dl_wall={wall_download_sec:.2f}s vs sequential_sum={sum_single_dl_sec:.2f}s)"
    )
    return {
        "env_bootstrap_sec": round(env_bootstrap_sec, 3),
        "wall_download_sec": round(wall_download_sec, 3),
        "sum_individual_dl_sec": round(sum_single_dl_sec, 3),
        "eval_backbone_prefetch_sec": round(eval_prefetch_sec, 3),
        "phase1_total_sec": round(phase1_total_sec, 3),
        "total_weight_bytes": sum(r[1] for r in dl_results),
    }


def prepare_first_and_last_frames(width: int = 1344, height: int = 768) -> Tuple[Any, Any, Any, Any]:
    """Loads or synthesizes the exact deterministic first_frame (<Picture 1>) and last_frame (<Picture 2>) used in Exp 02-06."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    first_candidates = [Path("/content/input_first_frame.png"), COMFY_ROOT / "input" / "first_frame.png"]
    last_candidates = [Path("/content/input_last_frame.png"), COMFY_ROOT / "input" / "last_frame.png"]
    first_path = next((p for p in first_candidates if p.exists()), first_candidates[0])
    last_path = next((p for p in last_candidates if p.exists()), last_candidates[0])

    if first_path.exists():
        img1 = Image.open(first_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    else:
        img1 = Image.new("RGB", (width, height), (14, 22, 42))
        draw1 = ImageDraw.Draw(img1)
        for y in range(height):
            r = int(14 + 28 * (y / height))
            g = int(22 + 55 * (y / height))
            b = int(42 + 95 * (y / height))
            draw1.line([(0, y), (width, y)], fill=(r, g, b))
        draw1.rectangle([width // 3, height // 4, 2 * width // 3, 3 * height // 4], outline=(0, 220, 255), width=6)

    if last_path.exists():
        img2 = Image.open(last_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    else:
        img2 = Image.new("RGB", (width, height), (28, 18, 48))
        draw2 = ImageDraw.Draw(img2)
        for y in range(height):
            r = int(48 + 110 * (y / height))
            g = int(28 + 65 * (y / height))
            b = int(38 + 45 * (y / height))
            draw2.line([(0, y), (width, y)], fill=(r, g, b))
        draw2.ellipse([width // 3, height // 5, 2 * width // 3, 4 * height // 5], outline=(255, 190, 60), width=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img1.save(OUTPUT_DIR / "input_first_frame.png")
    img2.save(OUTPUT_DIR / "input_last_frame.png")

    arr1 = np.asarray(img1, dtype=np.float32) / 255.0
    arr2 = np.asarray(img2, dtype=np.float32) / 255.0
    return torch.from_numpy(arr1).unsqueeze(0), torch.from_numpy(arr2).unsqueeze(0), arr1, arr2


def clone_conditioning(cond: list) -> list:
    """Deep-clones conditioning tensors for both FL2VA (minimax_keyframes) and Ref2VA (minimax_refs)."""
    cloned = []
    for item in cond:
        dict_cond = {}
        for k, v in item[1].items():
            if k == "minimax_keyframes" and isinstance(v, list):
                dict_cond[k] = [
                    {
                        "resolved_frame_index": kf["resolved_frame_index"],
                        "latent": kf["latent"].clone() if kf.get("latent") is not None else None,
                    }
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


def assert_expected_kernel_calls(cfg: Dict[str, Any], calls: Dict[str, int]) -> None:
    """Hard verification of attention & FusedModulation kernel call counts for 20-step and 8-step runs."""
    steps = int(cfg["steps"])
    expected_total_attn = 50 * steps + 2
    expected_fused_mod = 50 * steps if cfg.get("fused_mod") else 0

    total_attn = (
        calls["sdpa"]
        + calls["sage"]
        + calls["ck_int8"]
        + calls["ck_sol_chunked"]
        + calls["ck_sol_direct"]
        + calls["triton_sol"]
    )
    if total_attn != expected_total_attn:
        raise RuntimeError(f"[{cfg['id']}] Expected {expected_total_attn} total attention calls, got {total_attn}: {calls}")
    if calls["fused_mod_blocks"] != expected_fused_mod:
        raise RuntimeError(f"[{cfg['id']}] Expected fused_mod_blocks={expected_fused_mod}, got {calls['fused_mod_blocks']}")

    sm = cfg["sparse_mode"]
    if sm == "none":
        b_key = cfg["dense_backend"]
        if calls[b_key] != expected_total_attn:
            raise RuntimeError(f"[{cfg['id']}] Pure dense run expected {b_key}=={expected_total_attn}, got {calls}")
    elif sm == "ck_sol_direct":
        n_prot = len([x for x in str(cfg.get("dense_blocks", "")).split(",") if x.strip()])
        sparse_steps = steps - int(round(steps * float(cfg.get("start_percent", 0.20))))
        expected_sparse = sparse_steps * (50 - n_prot)
        if calls["ck_sol_direct"] != expected_sparse or calls["sage"] != expected_total_attn - expected_sparse:
            raise RuntimeError(
                f"[{cfg['id']}] Expected ck_sol_direct={expected_sparse} & sage={expected_total_attn - expected_sparse}, got {calls}"
            )
