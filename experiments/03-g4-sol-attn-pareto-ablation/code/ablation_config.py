#!/usr/bin/env python3
"""
Configuration, Pinned Dependencies, Parallel Weight Bootstrap, and Helper Utilities
for Experiment 03 (MiniMax H3-Base-FL2VA Sol-Attn & Community Acceleration Pareto Ablation on Blackwell G4).
"""

import concurrent.futures
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

COMFY_ROOT = Path("/content/ComfyUI")
CUSTOM_SOL_DIR = COMFY_ROOT / "custom_nodes" / "comfyui_sol_attn"
OUTPUT_DIR = Path("/content/exp03_g4_outputs")

# Pinned upstream repository commits & package versions used in Experiment 03
COMFYUI_COMMIT = "f5ea88af1c46c50a39792d00af80e5af941f99f6"
COMMUNITY_SOL_COMMIT = "69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3"
PINNED_PIP_PACKAGES: List[str] = [
    "comfy-kitchen==0.2.40", "sageattention==1.0.6", "triton==3.6.0",
    "hf_transfer", "huggingface_hub", "lpips", "pyiqa", "transformers",
    "opencv-python-headless", "matplotlib", "scipy", "soundfile", "imageio[ffmpeg]",
]

# Empirical Blackwell G4 hourly rate derived from Experiment 02 /tun/m/ccu-info ledger audit
G4_CU_RATE_PER_HOUR = 7.2046
G4_CU_RATE_PER_SEC = G4_CU_RATE_PER_HOUR / 3600.0
# Experiment 02 A100 baseline table E2E latency (149.22s; raw 149.2165s -> 0.280617 CU at 6.77 CU/hr)
A100_BASELINE_E2E_SEC = 149.22
A100_BASELINE_CU = 0.280617

REPO_ID = "Comfy-Org/MiniMax-H3"
REPO_REVISION = "main"
GOLD_BUNDLE_FILES: List[Dict[str, Any]] = [
    {
        "role": "FL2VA DiT (Pruned INT8 ConvRot)",
        "rel_path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "size": 20970379616,
    },
    {
        "role": "Text Encoder (Qwen3-VL-32B NVFP4 AWQ)",
        "rel_path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "size": 15687142551,
    },
    {
        "role": "Video VAE (INT8 ConvRot)",
        "rel_path": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "size": 2811065184,
    },
    {
        "role": "8-Step Turbo LoRA (BF16)",
        "rel_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "size": 1956193000,
    },
    {
        "role": "Audio VAE (FP32)",
        "rel_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "size": 605254808,
    },
]

OFFICIAL_IMAGE_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
    "refs/heads/main/input/transparent_rgb_gaming_mouse.png"
)

OFFICIAL_PROMPT = (
    "Editorial tech product film. The transparent gaming mouse from <Picture 1> in its original scene: "
    "a pitch-black studio void with a dark, subtle reflective surface, lit by dramatic duotone vibrant blue "
    "and warm neon orange rim lighting, deep soft shadow falloff into pure black. Monochromatic dark palette "
    "with electric blue and amber accents. Material motif: glowing internal metallic micro-components and "
    "glossy acrylic refractions. The environment is constant throughout.\n"
    "SHOT 1: The scene opens exactly on image 1, the mouse resting confidently on the dark surface; the blue "
    "and orange lights slowly pulse brighter, refracting deeply through the transparent acrylic shell as the "
    "camera executes a slow, deliberate push-in to reveal the intricate circuitry.\n"
    "SHOT 2: Cut to an extreme macro profile of the ridged scroll wheel and layered internal micro-components; "
    "the camera glides slowly along the side as a sharp beam of warm orange light sweeps across the metallic "
    "textures, contrasting perfectly against the deep blue ambient glow.\n"
    "SHOT 3: Cut to a low-angle beauty shot: the mouse levitates weightlessly a few centimeters above the dark "
    "reflective surface, rotating in a slow, precise orbit; the duotone lighting flares gently along the glassy "
    "transparent edges before fading slowly into a sleek silhouette.\n"
    "Audio: deep pulsing sub-bass room tone, sharp tactile mechanical clicks, a sweeping glassy whoosh on cuts, "
    "and a rising electronic swell that resolves to near-silence on the final fade."
)

G4_ABLATION_MATRIX: List[Dict[str, Any]] = [
    {
        "run_id": 1,
        "code": "[D0-SDPA]",
        "label": "Run01_Baseline_SDPA",
        "family": "Dense Baseline",
        "dense_backend": "sdpa",
        "sparse_mode": "none",
        "fused_mod": False,
        "desc": "PyTorch 2.11 cuDNN SDPA (BF16, 100% Dense) + Eager AdaLN",
    },
    {
        "run_id": 2,
        "code": "[D1-Sage]",
        "label": "Run02_Dense_Sage_Triton",
        "family": "Dense Accelerated",
        "dense_backend": "sage",
        "sparse_mode": "none",
        "fused_mod": False,
        "desc": "ADR-02 Baseline: SageAttention 1.0.6 (Triton INT8-QK, BF16-PV, 100% Dense)",
    },
    {
        "run_id": 3,
        "code": "[D2-CKInt8]",
        "label": "Run03_Dense_CK_Int8",
        "family": "Dense Accelerated",
        "dense_backend": "ck_int8",
        "sparse_mode": "none",
        "fused_mod": False,
        "desc": "Native comfy-kitchen Pure-INT8 Dense Attention (CUDA qk_int_sv_i8 INT8-QK+PV)",
    },
    {
        "run_id": 4,
        "code": "[D3-Sage+Mod]",
        "label": "Run04_Dense_Sage_FusedMod",
        "family": "Dense + Operator Fusion",
        "dense_backend": "sage",
        "sparse_mode": "none",
        "fused_mod": True,
        "desc": "SageAttention 1.0.6 + Community Triton MiniMaxH3FusedModulation (Bit-Exact AdaLN/Gate)",
    },
    {
        "run_id": 5,
        "code": "[S1-CKChunk-Full]",
        "label": "Run05_CK_SolChunked_Full_t13",
        "family": "Native CK Sol-Attn (4K Chunked)",
        "dense_backend": "sage",
        "sparse_mode": "ck_chunked",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 0,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Stock H3 4K-Chunked Producer (ck.sol_attn_chunked): 0-100% steps, tau=1.3, tok=0",
    },
    {
        "run_id": 6,
        "code": "[S2-CKChunk-W20-Tok256]",
        "label": "Run06_CK_SolChunked_W20_t13_Tok256",
        "family": "Native CK Sol-Attn (4K Chunked)",
        "dense_backend": "sage",
        "sparse_mode": "ck_chunked",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 256,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Stock Node Schema Default (ck.sol_attn_chunked): 20% Sage warmup, tau=1.3, tok=256",
    },
    {
        "run_id": 7,
        "code": "[S3-CKDirect-Full]",
        "label": "Run07_CK_SolDirect_Full_t13",
        "family": "Native CK Sol-Attn (Direct Zero-Copy)",
        "dense_backend": "sage",
        "sparse_mode": "ck_direct",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 0,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Discovery 1 Unchunked Direct Zero-Copy (ck.sol_attn): 0-100% steps, tau=1.3, tok=0",
    },
    {
        "run_id": 8,
        "code": "[S4-CKDirect-W20-Tok256]",
        "label": "Run08_CK_SolDirect_W20_t13_Tok256",
        "family": "Native CK Sol-Attn (Direct Zero-Copy)",
        "dense_backend": "sage",
        "sparse_mode": "ck_direct",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 256,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Direct Zero-Copy (ck.sol_attn): 20% Sage warmup, tau=1.3, extra_tokens=256",
    },
    {
        "run_id": 9,
        "code": "[S5-CKDirect-HiFi-t10]",
        "label": "Run09_CK_SolDirect_HiFi_t10_Prot",
        "family": "Native CK Sol-Attn (Direct Zero-Copy)",
        "dense_backend": "sage",
        "sparse_mode": "ck_direct",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.0,
        "extra_tokens": 256,
        "dense_blocks": "0,1,48,49",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "High-Fidelity Direct Sol-Attn: 20% Sage warmup, tau=1.0, tok=256, dense_blocks='0,1,48,49'",
    },
    {
        "run_id": 10,
        "code": "[S6-Triton-Int8QK]",
        "label": "Run10_Community_Triton_Sol_Int8QK",
        "family": "Community Triton Sol-Attn",
        "dense_backend": "sage",
        "sparse_mode": "triton_mem_eff",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "tau": 1.3,
        "int8_qk": True,
        "int8_pv": False,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Community r-vage Triton MemoryEfficientSol: 0-100% steps, tau=1.3, int8_qk=True, int8_pv=False",
    },
    {
        "run_id": 11,
        "code": "[S7-Triton-Sched]",
        "label": "Run11_Community_Scheduled_Sol_Fixed",
        "family": "Community Triton Sol-Attn",
        "dense_backend": "sage",
        "sparse_mode": "triton_scheduled_fixed",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau_start": 1.5,
        "tau_end": 0.8,
        "curve": "cosine",
        "int8_qk": True,
        "int8_pv": True,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": False,
        "desc": "Discovery 2 Flow-Shift-Fixed ScheduledSol: 20% Sage warmup, cosine tau 1.5->0.8, int8_qk+pv",
    },
    {
        "run_id": 12,
        "code": "[S8-FullStack-Pareto]",
        "label": "Run12_FullStack_CKSolDirect_Mod",
        "family": "Full-Stack Hybrid",
        "dense_backend": "sage",
        "sparse_mode": "ck_direct",
        "start_percent": 0.20,
        "end_percent": 1.0,
        "tau": 1.3,
        "extra_tokens": 256,
        "dense_blocks": "",
        "sink_conditioning": "exact_kv_and_rows",
        "fused_mod": True,
        "desc": "Full-Stack Hybrid: Direct Zero-Copy ck.sol_attn (W20, tau=1.3, tok=256) + Triton H3FusedModulation",
    },
]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def prefetch_eval_backbones() -> float:
    """Single background worker that sequentially pre-caches the 5 evaluation models on CPU."""
    t0 = time.perf_counter()
    import lpips
    import pyiqa
    import torchvision.models.optical_flow as tv_flow
    from transformers import AutoModel, CLIPVisionModelWithProjection

    _ = lpips.LPIPS(net="alex", verbose=False)
    _ = AutoModel.from_pretrained("facebook/dinov2-small")
    _ = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    _ = tv_flow.raft_small(weights=tv_flow.Raft_Small_Weights.DEFAULT, progress=False)
    _ = pyiqa.create_metric("musiq", device="cpu")
    return time.perf_counter() - t0


def bootstrap_and_download_parallel() -> Dict[str, Any]:
    """
    Phase 1: Parallel Bootstrap & Multi-Stream Weight Prefetch.
    Checks out pinned ComfyUI and ComfyUI-sol-attn commits, installs dependencies,
    and runs 5 H3 weight download workers concurrently with 1 evaluation model prefetch worker.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_phase0 = time.perf_counter()
    print("=" * 96)
    print("🛠️ [Phase 1/5] Zero-Waste Parallel Bootstrap & 5-Stream Weight Prefetch on Blackwell G4")
    print("=" * 96)

    t_env0 = time.perf_counter()
    if not COMFY_ROOT.exists():
        run_cmd(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)])
        run_cmd(["git", "checkout", COMFYUI_COMMIT], cwd=COMFY_ROOT)
    else:
        print(f"  ✅ ComfyUI already cloned at {COMFY_ROOT}")

    CUSTOM_SOL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_SOL_DIR.exists():
        run_cmd(["git", "clone", "https://github.com/r-vage/ComfyUI-sol-attn.git", str(CUSTOM_SOL_DIR)])
        run_cmd(["git", "checkout", COMMUNITY_SOL_COMMIT], cwd=CUSTOM_SOL_DIR)
    else:
        print(f"  ✅ r-vage/ComfyUI-sol-attn already cloned at {CUSTOM_SOL_DIR}")

    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_ROOT / "requirements.txt")])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *PINNED_PIP_PACKAGES])
    env_bootstrap_sec = time.perf_counter() - t_env0

    from huggingface_hub import hf_hub_download

    models_dir = COMFY_ROOT / "models"
    input_dir = COMFY_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    first_path = input_dir / "transparent_rgb_gaming_mouse.png"

    def dl_one_weight(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[str, int, float, bool]:
        idx, item = idx_item
        rel_path, fsize = item["rel_path"], item["size"]
        target_file = models_dir / rel_path
        if target_file.exists() and target_file.stat().st_size == fsize:
            print(f"  [{idx}/5] ✅ Cached {item['role']} ({fsize / 1e9:.2f} GB)")
            return item["role"], fsize, 0.0, True
        t0 = time.perf_counter()
        hf_hub_download(repo_id=REPO_ID, revision=REPO_REVISION, filename=rel_path, local_dir=str(models_dir))
        dt = time.perf_counter() - t0
        speed_mbs = (fsize / (1024 ** 2)) / max(dt, 1e-6)
        print(f"  [{idx}/5] ⬇️ {item['role']} ({fsize / 1e9:.2f} GB) in {dt:.1f}s ({speed_mbs:.0f} MiB/s)")
        return item["role"], fsize, dt, False

    def dl_input_image() -> float:
        t0 = time.perf_counter()
        if not first_path.exists() or first_path.stat().st_size < 1000:
            urllib.request.urlretrieve(OFFICIAL_IMAGE_URL, str(first_path))
        return time.perf_counter() - t0

    t_dl0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fut_img = pool.submit(dl_input_image)
        fut_eval = pool.submit(prefetch_eval_backbones)
        weight_futures = [pool.submit(dl_one_weight, (i, item)) for i, item in enumerate(GOLD_BUNDLE_FILES, 1)]
        weight_results = [f.result() for f in weight_futures]
        _ = fut_img.result()
        eval_prefetch_sec = fut_eval.result()

    shutil.rmtree(models_dir / ".cache", ignore_errors=True)
    wall_download_sec = time.perf_counter() - t_dl0
    total_bytes = sum(r[1] for r in weight_results)
    sum_individual_dl_sec = sum(r[2] for r in weight_results)
    phase1_total_sec = time.perf_counter() - t_phase0
    print(
        f"  ✅ Phase 1 finished in {phase1_total_sec:.2f}s "
        f"(env={env_bootstrap_sec:.2f}s, parallel_dl_wall={wall_download_sec:.2f}s, "
        f"eval_prefetch_overlapped={eval_prefetch_sec:.2f}s)"
    )
    return {
        "env_bootstrap_sec": round(env_bootstrap_sec, 3),
        "wall_download_sec": round(wall_download_sec, 3),
        "sum_individual_dl_sec": round(sum_individual_dl_sec, 3),
        "eval_prefetch_overlapped_sec": round(eval_prefetch_sec, 3),
        "phase1_total_sec": round(phase1_total_sec, 3),
        "total_bytes": total_bytes,
    }


def prepare_first_and_last_frames(width: int = 1344, height: int = 768) -> Tuple[Any, Any, Any, Any]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    first_path = COMFY_ROOT / "input" / "transparent_rgb_gaming_mouse.png"
    if not first_path.exists() or first_path.stat().st_size < 1000:
        urllib.request.urlretrieve(OFFICIAL_IMAGE_URL, str(first_path))

    pil_first = Image.open(first_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    first_np = np.asarray(pil_first, dtype=np.float32) / 255.0

    src_u8 = (first_np * 255.0).clip(0, 255).astype(np.uint8)
    rot_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle=-6.0, scale=1.18)
    rot_mat[0, 2] += width * 0.03
    rot_mat[1, 2] -= height * 0.02
    warped_u8 = cv2.warpAffine(
        src_u8, rot_mat, (width, height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
    )
    last_np = warped_u8.astype(np.float32) / 255.0
    last_np[..., 0] = np.clip(last_np[..., 0] * 1.08, 0.0, 1.0)
    last_np[..., 2] = np.clip(last_np[..., 2] * 1.12, 0.0, 1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray((first_np * 255.0).astype(np.uint8)).save(OUTPUT_DIR / "input_first_frame.png")
    Image.fromarray((last_np * 255.0).astype(np.uint8)).save(OUTPUT_DIR / "input_last_frame.png")

    return torch.from_numpy(first_np).unsqueeze(0), torch.from_numpy(last_np).unsqueeze(0), first_np, last_np


def clone_conditioning(cond: list) -> list:
    cloned = []
    for item in cond:
        tensor_cond = item[0].clone()
        dict_cond = {}
        for k, v in item[1].items():
            if k == "minimax_keyframes" and isinstance(v, list):
                dict_cond[k] = [
                    {
                        "resolved_frame_index": kf["resolved_frame_index"],
                        "latent": kf["latent"].clone() if "latent" in kf and kf["latent"] is not None else None,
                    }
                    for kf in v
                ]
            elif hasattr(v, "clone"):
                dict_cond[k] = v.clone()
            else:
                dict_cond[k] = v
        cloned.append([tensor_cond, dict_cond])
    return cloned


def clone_av_latent(base_latent: Dict[str, Any]) -> Dict[str, Any]:
    import comfy.nested_tensor

    v_lat, a_lat = base_latent["samples"].unbind()
    return {"samples": comfy.nested_tensor.NestedTensor((v_lat.clone(), a_lat.clone()))}


class FallbackAlertDetector(logging.Handler):
    """Detects any unexpected fallback in SageAttention, BlockSparseAttention, or Community Sol-Attn."""

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


def assert_expected_kernel_calls(cfg: Dict[str, Any], actual: Dict[str, int]) -> None:
    """Strictly verifies that every single DiT & Refiner call executed on the intended attention entry."""
    sparse_mode = cfg["sparse_mode"]
    dense_backend = cfg["dense_backend"]
    start_pct = float(cfg.get("start_percent", 0.0))
    dense_blocks_str = str(cfg.get("dense_blocks", ""))
    n_dense_blocks = len([x for x in dense_blocks_str.split(",") if x.strip()]) if dense_blocks_str else 0

    expected = {
        "sdpa": 0,
        "sage": 0,
        "ck_int8": 0,
        "ck_sol_chunked": 0,
        "ck_sol_direct": 0,
        "triton_sol": 0,
        "fused_mod_blocks": 400 if cfg.get("fused_mod", False) else 0,
    }
    if sparse_mode == "none":
        expected[dense_backend] = 402
    else:
        dense_steps = 2 if start_pct > 0.0 else 0
        sparse_steps = 8 - dense_steps
        sparse_calls = sparse_steps * (50 - n_dense_blocks)
        expected[dense_backend] = 402 - sparse_calls
        if sparse_mode == "ck_chunked":
            expected["ck_sol_chunked"] = sparse_calls
        elif sparse_mode == "ck_direct":
            expected["ck_sol_direct"] = sparse_calls
        elif sparse_mode in ("triton_mem_eff", "triton_scheduled_fixed"):
            expected["triton_sol"] = sparse_calls

    if actual != expected:
        raise RuntimeError(
            f"Attention entry telemetry mismatch in {cfg['label']} ({cfg['code']})!\n"
            f"  Expected: {expected}\n  Actual:   {actual}"
        )
