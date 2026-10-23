#!/usr/bin/env python3
"""
Experiment 02: MiniMax H3-Base-FL2VA 2^3=8 Full-Factorial Acceleration Ablation on Colab A100-80GB
Evaluates the 3 most acclaimed yet controversial ComfyUI acceleration toggles on top of the
ADR-01 Golden 5-Pack (42.03 GB):
  - S (SageAttention) : attention_sage vs default attention_pytorch (SDPA / FlashAttention-2)
  - C (EasyCache)     : ComfyUI native step-skipping residual cache (EasyCacheNode) vs disabled
  - T (torch.compile) : ComfyUI native TorchCompileModel(backend="inductor") vs Eager mode
                        (measuring both Cold Compile T_cold and Warm Steady-State T_warm)

Computes a 4-Dimensional Benchmark Quality Score (Q_total in [0, 100]):
  1. S_ref    : DiT Acceleration Reference Fidelity (PSNR, SSIM, LPIPS vs Run 1 Baseline)
                [EasyCache arXiv:2507.02860 / TeaCache CVPR 2025 / SageAttention2 ICLR 2025]
  2. S_vbench : VBench Video Intrinsic Quality & Temporal Consistency
                (DINO Subject Consistency, CLIP Background Consistency, VBench Temporal Flickering,
                 RAFT Optical Flow Motion Smoothness, MUSIQ Imaging Quality)
  3. S_fl2v   : VBench-I2V First & Last Frame Condition Consistency
                (DINO+CLIP Deep Feature Similarity + SSIM vs Input First/Last Anchors)
  4. S_audio  : Joint Audio-Video Quality & Sync
                (Multi-Res STFT-SNR, 80-Band Log-Mel Cosine Similarity, AV-Sync Energy Envelope Correlation)

Outputs:
  - /content/exp02_outputs/ablation_metrics.json
  - /content/exp02_outputs/speed_quality_pareto.png
  - /content/exp02_outputs/keyframe_strip_8runs.png
  - /content/exp02_outputs/frame_metrics_curves.png
  - /content/exp02_outputs/comparison_grid_8runs.mp4
"""

import os
import sys

# Enable high-performance Rust Xet transfer and disable Colab secret polling before any HF import
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import logging
import math
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

COMFY_ROOT = Path("/content/ComfyUI")
OUTPUT_DIR = Path("/content/exp02_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "Comfy-Org/MiniMax-H3"
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

# 8 Full-Factorial Configurations: (run_id, code, label, S_sage, C_cache, T_compile)
ABLATION_MATRIX = [
    (1, "[0,0,0]", "Run1_Baseline",      False, False, False),
    (2, "[1,0,0]", "Run2_Only_Sage",     True,  False, False),
    (3, "[0,1,0]", "Run3_Only_Cache",    False, True,  False),
    (4, "[0,0,1]", "Run4_Only_Compile",  False, False, True),
    (5, "[1,1,0]", "Run5_Sage_Cache",    True,  True,  False),
    (6, "[1,0,1]", "Run6_Sage_Compile",  True,  False, True),
    (7, "[0,1,1]", "Run7_Cache_Compile", False, True,  True),
    (8, "[1,1,1]", "Run8_All_Three",     True,  True,  True),
]


def run_cmd(cmd: List[str], cwd: Path = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def bootstrap_environment() -> None:
    print("=" * 90)
    print("🛠️ [Phase 1/5] Bootstrapping ComfyUI + SageAttention + Quality Evaluation Dependencies")
    print("=" * 90)

    if not COMFY_ROOT.exists():
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)])
    else:
        print(f"  ✅ ComfyUI already cloned at {COMFY_ROOT}")

    run_cmd([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", str(COMFY_ROOT / "requirements.txt"),
    ])
    run_cmd([
        sys.executable, "-m", "pip", "install", "-q",
        "hf_transfer", "huggingface_hub", "sageattention", "lpips", "pyiqa",
        "transformers", "opencv-python-headless", "matplotlib", "scipy", "soundfile", "imageio[ffmpeg]",
    ])


def download_golden_5pack() -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    print("\n" + "=" * 90)
    print("📥 [Phase 2/5] Downloading ADR-01 Golden 5-Pack (42.03 GB) to /content/ComfyUI/models")
    print("=" * 90)

    models_dir = COMFY_ROOT / "models"
    total_bytes = 0
    total_dl_sec = 0.0

    for idx, item in enumerate(GOLD_BUNDLE_FILES, 1):
        rel_path = item["rel_path"]
        fsize = item["size"]
        target_file = models_dir / rel_path
        if target_file.exists() and target_file.stat().st_size == fsize:
            print(f"  [{idx}/5] ✅ Cached {item['role']} ({fsize / 1e9:.2f} GB)")
            total_bytes += fsize
            continue

        t0 = time.perf_counter()
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            local_dir=str(models_dir),
        )
        dl_sec = time.perf_counter() - t0
        shutil.rmtree(models_dir / ".cache", ignore_errors=True)
        total_bytes += fsize
        total_dl_sec += dl_sec
        speed_mbs = (fsize / (1024 ** 2)) / max(dl_sec, 1e-6)
        print(f"  [{idx}/5] ⬇️ {item['role']} ({fsize / 1e9:.2f} GB) in {dl_sec:.1f}s ({speed_mbs:.0f} MiB/s)")

    return {"total_bytes": total_bytes, "download_sec": total_dl_sec}


def prepare_first_and_last_frames(width: int = 1344, height: int = 768) -> Tuple[Any, Any, Any, Any]:
    """
    Download official transparent_rgb_gaming_mouse.png as first_frame, and construct a deterministic
    high-contrast zoomed/rotated target last_frame so FL2VA exercises both first-frame and last-frame
    anchoring across the 124-frame sequence.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    input_dir = COMFY_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    first_path = input_dir / "transparent_rgb_gaming_mouse.png"

    if not first_path.exists() or first_path.stat().st_size < 1000:
        print(f"  ⬇️ Downloading official input image: {OFFICIAL_IMAGE_URL}")
        urllib.request.urlretrieve(OFFICIAL_IMAGE_URL, str(first_path))

    pil_first = Image.open(first_path).convert("RGB")
    pil_first_resized = pil_first.resize((width, height), Image.Resampling.LANCZOS)
    first_np = np.asarray(pil_first_resized, dtype=np.float32) / 255.0

    # Deterministic target last_frame (SHOT 3 beauty orbit + 1.18x zoom + rim-light shift)
    src_u8 = (first_np * 255.0).clip(0, 255).astype(np.uint8)
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle=-6.0, scale=1.18)
    rot_mat[0, 2] += width * 0.03
    rot_mat[1, 2] -= height * 0.02
    warped_u8 = cv2.warpAffine(
        src_u8, rot_mat, (width, height),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
    )
    last_np = warped_u8.astype(np.float32) / 255.0
    last_np[..., 0] = np.clip(last_np[..., 0] * 1.08, 0.0, 1.0)
    last_np[..., 2] = np.clip(last_np[..., 2] * 1.12, 0.0, 1.0)

    Image.fromarray((first_np * 255.0).astype(np.uint8)).save(OUTPUT_DIR / "input_first_frame.png")
    Image.fromarray((last_np * 255.0).astype(np.uint8)).save(OUTPUT_DIR / "input_last_frame.png")

    first_tensor = torch.from_numpy(first_np).unsqueeze(0)
    last_tensor = torch.from_numpy(last_np).unsqueeze(0)
    return first_tensor, last_tensor, first_np, last_np


def clone_conditioning(cond: list) -> list:
    """Deep-clone ComfyUI conditioning structure so no run mutates shared tensors/dicts."""
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
    nested = base_latent["samples"]
    v_lat, a_lat = nested.unbind()
    return {
        "samples": comfy.nested_tensor.NestedTensor((v_lat.clone(), a_lat.clone()))
    }


class SageFallbackDetector(logging.Handler):
    """Detects if ComfyUI's attention_sage ever logs a fallback to attention_pytorch."""

    def __init__(self) -> None:
        super().__init__()
        self.fallback_errors: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Error running sage attention" in msg or "falling back to pytorch attention" in msg:
            self.fallback_errors.append(msg)


def reset_inductor_and_dynamo_caches(run_id: int, sub_tag: str) -> None:
    """
    Ensure Cold-Compile runs start without reusing previous runs' Inductor/Triton artifacts.
    """
    import torch._dynamo
    import torch._inductor.codecache

    torch._dynamo.reset()
    try:
        torch._inductor.codecache.PyCodeCache.clear()
    except Exception:
        pass
    cache_dir = Path(f"/tmp/inductor_exp02_run{run_id}_{sub_tag}")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)


class QualityEvaluator:
    """
    Implements the 4-Dimensional Benchmark Quality Evaluation System aligned with academic benchmarks:
      1. S_ref    : DiT Acceleration Reference Fidelity (PSNR, SSIM, LPIPS vs Run 1 GT)
                    Formula matches README.md: 0.35 * min(1, PSNR/40) + 0.35 * SSIM + 0.30 * max(0, 1 - LPIPS/0.30)
      2. S_vbench : VBench Video Quality & Temporal Consistency (equal weights 0.25 x 4):
                    - Subject & Background Consistency (DINOv2 + CLIP ViT-B/32 frame cosine similarity)
                    - Temporal Flickering Score (VBench official: 1.0 - MAE(V[t], V[t-1]) / 255.0)
                    - Motion Smoothness & Dynamic Degree (Torchvision RAFT optical flow continuity)
                    - Imaging Quality (MUSIQ multi-scale perceptual image quality via pyiqa)
      3. S_fl2v   : VBench-I2V First & Last Frame Anchoring Consistency
                    (0.40 * Sim_first + 0.60 * Sim_last using DINOv2 + CLIP + SSIM)
      4. S_audio  : MovieGen / VABench Stereo Audio Quality & AV-Sync
                    (80-Band Log-Mel Spectrogram Cosine Sim, Multi-Res STFT-SNR, AV-Sync Energy Envelope Correlation)
    """

    def __init__(self, device: str = "cuda"):
        import lpips
        import torch
        import torchvision.models.optical_flow as tv_flow
        from transformers import AutoModel, CLIPVisionModelWithProjection

        self.device = torch.device(device)
        print("  🔬 Loading LPIPS (AlexNet), DINOv2 (facebook/dinov2-small), CLIP (openai/clip-vit-base-patch32), RAFT, and MUSIQ...")
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

        # Official VBench Subject Consistency backbone: DINO
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()
        # Official VBench Background / Image Consistency backbone: CLIP ViT-B/32
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()

        # Official VBench Dynamic Degree / Motion Smoothness optical flow backbone: RAFT
        raft_weights = tv_flow.Raft_Small_Weights.DEFAULT
        self.raft_model = tv_flow.raft_small(weights=raft_weights, progress=False).to(self.device).eval()

        # Official VBench Imaging Quality backbone: MUSIQ (via pyiqa, normalized to [0, 1] by /100)
        self.musiq_metric = None
        try:
            import pyiqa
            self.musiq_metric = pyiqa.create_metric("musiq", device=self.device)
        except Exception as e:
            print(f"  ⚠️ pyiqa MUSIQ fallback notice: {e}")

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def extract_dino_and_clip_features(self, frames_np: Any, batch_size: int = 16) -> Tuple[Any, Any]:
        """
        frames_np: [T, H, W, 3] float32 in [0, 1]
        Returns (dino_embeds [T, D1], clip_embeds [T, D2]) L2-normalized on CPU.
        """
        import torch
        import torch.nn.functional as F

        dino_list = []
        clip_list = []
        n = frames_np.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.from_numpy(frames_np[i : i + batch_size]).to(self.device, dtype=torch.float32)
                batch = batch.permute(0, 3, 1, 2)
                batch_224 = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

                dino_in = (batch_224 - self.imagenet_mean) / self.imagenet_std
                dino_out = self.dino_model(pixel_values=dino_in).last_hidden_state[:, 0, :]
                dino_list.append(F.normalize(dino_out, dim=-1).cpu())

                clip_in = (batch_224 - self.clip_mean) / self.clip_std
                clip_out = self.clip_model(pixel_values=clip_in).image_embeds
                clip_list.append(F.normalize(clip_out, dim=-1).cpu())

        return torch.cat(dino_list, dim=0), torch.cat(clip_list, dim=0)

    @staticmethod
    def compute_ssim_frame(img1: Any, img2: Any) -> float:
        import cv2
        import numpy as np

        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
        )
        return float(np.mean(ssim_map))

    def compute_reference_metrics(
        self, frames_test: Any, frames_gt: Any, is_baseline: bool = False
    ) -> Dict[str, Any]:
        """
        Dimension 1: PSNR, SSIM, LPIPS against Run 1 (Baseline) across all 124 frames.
        Strictly uses the README.md formula:
          S_ref = 100 * (0.35 * min(1, PSNR/40.0) + 0.35 * SSIM + 0.30 * max(0, 1 - LPIPS/0.30))
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        T = frames_test.shape[0]
        if is_baseline:
            return {
                "psnr_mean": 100.0,
                "psnr_min": 100.0,
                "ssim_mean": 1.0,
                "ssim_min": 1.0,
                "lpips_mean": 0.0,
                "lpips_max": 0.0,
                "psnr_per_frame": [100.0] * T,
                "ssim_per_frame": [1.0] * T,
                "lpips_per_frame": [0.0] * T,
                "S_ref": 100.0,
            }

        psnr_list = []
        ssim_list = []
        lpips_list = []

        for t in range(T):
            f_test = frames_test[t]
            f_gt = frames_gt[t]
            mse = float(np.mean((f_test - f_gt) ** 2))
            psnr = 100.0 if mse < 1e-10 else float(10.0 * math.log10(1.0 / mse))
            ssim = self.compute_ssim_frame(f_test, f_gt)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        with torch.no_grad():
            for i in range(0, T, 8):
                b_test = torch.from_numpy(frames_test[i : i + 8]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b_gt = torch.from_numpy(frames_gt[i : i + 8]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b_test = F.interpolate(b_test, size=(384, 672), mode="bilinear", align_corners=False) * 2.0 - 1.0
                b_gt = F.interpolate(b_gt, size=(384, 672), mode="bilinear", align_corners=False) * 2.0 - 1.0
                lp = self.lpips_fn(b_test, b_gt).view(-1).cpu().tolist()
                lpips_list.extend(lp)

        psnr_mean = float(np.mean(psnr_list))
        psnr_min = float(np.min(psnr_list))
        ssim_mean = float(np.mean(ssim_list))
        ssim_min = float(np.min(ssim_list))
        lpips_mean = float(np.mean(lpips_list))
        lpips_max = float(np.max(lpips_list))

        # Exact README.md formula (40.0 dB saturation)
        psnr_norm = min(1.0, max(0.0, psnr_mean / 40.0))
        ssim_norm = min(1.0, max(0.0, ssim_mean))
        lpips_norm = min(1.0, max(0.0, 1.0 - lpips_mean / 0.30))
        s_ref = 100.0 * (0.35 * psnr_norm + 0.35 * ssim_norm + 0.30 * lpips_norm)

        return {
            "psnr_mean": psnr_mean,
            "psnr_min": psnr_min,
            "ssim_mean": ssim_mean,
            "ssim_min": ssim_min,
            "lpips_mean": lpips_mean,
            "lpips_max": lpips_max,
            "psnr_per_frame": psnr_list,
            "ssim_per_frame": ssim_list,
            "lpips_per_frame": lpips_list,
            "S_ref": float(s_ref),
        }

    def compute_vbench_and_fl2v_metrics(
        self,
        frames_test: Any,
        first_anchor_np: Any,
        last_anchor_np: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        """
        Dimension 2 (VBench Video Quality & Temporal Consistency, equal weights 0.25 x 4) &
        Dimension 3 (VBench-I2V First/Last Frame Anchoring Consistency, 0.40 * first + 0.60 * last).
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        T = frames_test.shape[0]
        dino_feats, clip_feats = self.extract_dino_and_clip_features(frames_test)
        anc_dino, anc_clip = self.extract_dino_and_clip_features(np.stack([first_anchor_np, last_anchor_np], axis=0))

        # 1. VBench Subject Consistency (DINO) & Background Consistency (CLIP)
        dino_adj = torch.sum(dino_feats[:-1] * dino_feats[1:], dim=-1).numpy()
        dino_first = torch.sum(dino_feats[1:] * dino_feats[0:1], dim=-1).numpy()
        subject_consistency = float(0.5 * np.mean(dino_adj) + 0.5 * np.mean(dino_first))

        clip_adj = torch.sum(clip_feats[:-1] * clip_feats[1:], dim=-1).numpy()
        clip_first = torch.sum(clip_feats[1:] * clip_feats[0:1], dim=-1).numpy()
        background_consistency = float(0.5 * np.mean(clip_adj) + 0.5 * np.mean(clip_first))

        sub_bg_consistency = 0.5 * subject_consistency + 0.5 * background_consistency

        # 2. VBench Temporal Flickering Score: 1.0 - MAE(V[t], V[t-1]) / 255.0 (on [0, 1] scale)
        diffs = np.abs(frames_test[1:] - frames_test[:-1])
        frame_mae = np.mean(diffs, axis=(1, 2, 3))
        temporal_flickering_score = float(1.0 - np.mean(frame_mae))

        # 3. VBench Optical Flow Motion Smoothness & Dynamic Degree via Torchvision RAFT
        flow_mags = []
        with torch.no_grad():
            for i in range(0, T - 1, 8):
                end_i = min(T - 1, i + 8)
                b1 = torch.from_numpy(frames_test[i:end_i]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b2 = torch.from_numpy(frames_test[i + 1 : end_i + 1]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b1 = F.interpolate(b1, size=(256, 448), mode="bilinear", align_corners=False) * 2.0 - 1.0
                b2 = F.interpolate(b2, size=(256, 448), mode="bilinear", align_corners=False) * 2.0 - 1.0
                flow_pred = self.raft_model(b1, b2)[-1]  # [B, 2, H, W]
                mag = torch.sqrt(torch.sum(flow_pred ** 2, dim=1)).mean(dim=(1, 2)).cpu().tolist()
                flow_mags.extend(mag)

        flow_mags_np = np.array(flow_mags, dtype=np.float32)
        flow_accel = np.abs(np.diff(flow_mags_np))
        motion_smoothness = float(1.0 / (1.0 + np.mean(flow_accel) / (np.mean(flow_mags_np) + 1e-4)))

        # 4. VBench Imaging Quality (MUSIQ sampled on 9 evenly spaced frames across the 124 frames)
        sample_indices = np.linspace(0, T - 1, 9, dtype=int)
        if self.musiq_metric is not None:
            musiq_scores = []
            with torch.no_grad():
                for idx in sample_indices:
                    img_t = torch.from_numpy(frames_test[idx]).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    img_t = F.interpolate(img_t, size=(384, 672), mode="bilinear", align_corners=False)
                    score = float(self.musiq_metric(img_t).item()) / 100.0
                    musiq_scores.append(min(1.0, max(0.0, score)))
            imaging_quality = float(np.mean(musiq_scores))
        else:
            imaging_quality = 0.75

        # Equal-weighted synthesis of the 4 VBench sub-dimensions (matching README.md Section 4.2)
        s_vbench = 100.0 * 0.25 * (
            sub_bg_consistency + temporal_flickering_score + motion_smoothness + imaging_quality
        )

        vbench_dict = {
            "subject_consistency_dino": subject_consistency,
            "background_consistency_clip": background_consistency,
            "sub_bg_consistency": sub_bg_consistency,
            "temporal_flickering_score": temporal_flickering_score,
            "motion_smoothness_raft": motion_smoothness,
            "dynamic_degree_raft_px": float(np.mean(flow_mags_np)),
            "imaging_quality_musiq": imaging_quality,
            "S_vbench": float(s_vbench),
        }

        # Dimension 3: VBench-I2V First & Last Frame Condition Consistency
        first_dino_sim = float(torch.sum(dino_feats[0] * anc_dino[0]).item())
        first_clip_sim = float(torch.sum(clip_feats[0] * anc_clip[0]).item())
        last_dino_sim = float(torch.sum(dino_feats[-1] * anc_dino[1]).item())
        last_clip_sim = float(torch.sum(clip_feats[-1] * anc_clip[1]).item())

        first_feat_sim = 0.5 * first_dino_sim + 0.5 * first_clip_sim
        last_feat_sim = 0.5 * last_dino_sim + 0.5 * last_clip_sim
        first_ssim = self.compute_ssim_frame(frames_test[0], first_anchor_np)
        last_ssim = self.compute_ssim_frame(frames_test[-1], last_anchor_np)

        first_combined = 0.5 * first_feat_sim + 0.5 * first_ssim
        last_combined = 0.5 * last_feat_sim + 0.5 * last_ssim
        s_fl2v = 100.0 * (0.40 * first_combined + 0.60 * last_combined)

        fl2v_dict = {
            "first_frame_dino_sim": first_dino_sim,
            "first_frame_clip_sim": first_clip_sim,
            "first_frame_ssim": first_ssim,
            "last_frame_dino_sim": last_dino_sim,
            "last_frame_clip_sim": last_clip_sim,
            "last_frame_ssim": last_ssim,
            "S_fl2v": float(s_fl2v),
        }

        motion_energy_t = np.concatenate([[flow_mags_np[0]], flow_mags_np])
        return vbench_dict, fl2v_dict, motion_energy_t

    @staticmethod
    def _hz_to_mel(hz: Any) -> Any:
        import numpy as np
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: Any) -> Any:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _build_mel_filterbank(cls, sr: int, n_fft: int, n_mels: int = 80) -> Any:
        import numpy as np

        fmin, fmax = 20.0, sr / 2.0
        mel_pts = np.linspace(cls._hz_to_mel(fmin), cls._hz_to_mel(fmax), n_mels + 2)
        hz_pts = cls._mel_to_hz(mel_pts)
        bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
        for m in range(1, n_mels + 1):
            f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
            for k in range(f_left, f_center):
                if f_center > f_left:
                    fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        return fbank

    @classmethod
    def compute_audio_and_avsync_metrics(
        cls,
        audio_test: Any,
        audio_gt: Any,
        motion_energy_t: Any,
        baseline_av_corr: float = None,
        is_baseline: bool = False,
        sample_rate: int = 32000,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Dimension 4: Stereo Audio Quality (Multi-Res STFT-SNR, True 80-Band Log-Mel Cosine Similarity)
        and Audio-Visual Temporal Synchronization (AV-Sync Envelope Correlation with nearest-edge filtering).
        """
        import numpy as np
        from scipy import ndimage, signal

        L = min(audio_test.shape[-1], audio_gt.shape[-1])
        a_test = audio_test[:, :L].astype(np.float64)
        a_gt = audio_gt[:, :L].astype(np.float64)

        T = len(motion_energy_t)
        samples_per_frame = max(1, L // T)
        audio_env = np.zeros(T, dtype=np.float64)
        for t in range(T):
            seg = a_test[:, t * samples_per_frame : (t + 1) * samples_per_frame]
            audio_env[t] = np.sqrt(np.mean(seg ** 2) + 1e-12)

        # Use nearest-boundary smoothing (never zero-padded) so constant signals never get artificial edge correlation
        a_env_s = ndimage.uniform_filter1d(audio_env, size=3, mode="nearest")
        m_env_s = ndimage.uniform_filter1d(motion_energy_t.astype(np.float64), size=3, mode="nearest")
        if np.std(a_env_s) > 1e-6 and np.std(m_env_s) > 1e-6:
            av_corr = float(np.corrcoef(a_env_s, m_env_s)[0, 1])
        else:
            av_corr = 0.0

        if is_baseline or baseline_av_corr is None:
            baseline_av_corr = av_corr

        av_sync_retention = float(max(0.0, min(1.0, 1.0 - abs(av_corr - baseline_av_corr))))

        if is_baseline:
            return {
                "stft_snr_db": 100.0,
                "log_mel_cosine_sim": 1.0,
                "rms_diff_db": 0.0,
                "av_sync_corr": av_corr,
                "av_sync_retention": 1.0,
                "S_audio": 100.0,
            }, baseline_av_corr

        # 1. Multi-resolution STFT log-magnitude SNR
        snr_list = []
        for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
            for ch in range(min(2, a_test.shape[0])):
                _, _, Z_t = signal.stft(a_test[ch], fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
                _, _, Z_g = signal.stft(a_gt[ch], fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
                mag_t = np.log10(np.abs(Z_t) + 1e-7)
                mag_g = np.log10(np.abs(Z_g) + 1e-7)
                noise_pow = np.mean((mag_t - mag_g) ** 2)
                sig_pow = np.mean(mag_g ** 2) + 1e-10
                snr = 100.0 if noise_pow < 1e-12 else float(10.0 * math.log10(sig_pow / noise_pow))
                snr_list.append(snr)

        # 2. True 80-Band Log-Mel Spectrogram Cosine Similarity (n_fft=1024, hop=256)
        mel_fbank = cls._build_mel_filterbank(sr=sample_rate, n_fft=1024, n_mels=80)
        mel_cos_list = []
        for ch in range(min(2, a_test.shape[0])):
            _, _, Z_t = signal.stft(a_test[ch], fs=sample_rate, nperseg=1024, noverlap=1024 - 256)
            _, _, Z_g = signal.stft(a_gt[ch], fs=sample_rate, nperseg=1024, noverlap=1024 - 256)
            pow_t = np.abs(Z_t) ** 2
            pow_g = np.abs(Z_g) ** 2
            log_mel_t = np.log10(np.matmul(mel_fbank, pow_t) + 1e-7)
            log_mel_g = np.log10(np.matmul(mel_fbank, pow_g) + 1e-7)
            v1 = log_mel_t.reshape(-1)
            v2 = log_mel_g.reshape(-1)
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            mel_cos_list.append(cos)

        stft_snr_db = float(np.mean(snr_list))
        log_mel_cos = float(np.mean(mel_cos_list))

        rms_t = 20.0 * math.log10(math.sqrt(float(np.mean(a_test ** 2)) + 1e-12))
        rms_g = 20.0 * math.log10(math.sqrt(float(np.mean(a_gt ** 2)) + 1e-12))
        rms_diff_db = float(abs(rms_t - rms_g))

        snr_norm = min(1.0, max(0.0, stft_snr_db / 30.0))
        cos_norm = min(1.0, max(0.0, log_mel_cos))
        s_audio = 100.0 * (0.40 * cos_norm + 0.35 * snr_norm + 0.25 * av_sync_retention)

        return {
            "stft_snr_db": stft_snr_db,
            "log_mel_cosine_sim": log_mel_cos,
            "rms_diff_db": rms_diff_db,
            "av_sync_corr": av_corr,
            "av_sync_retention": av_sync_retention,
            "S_audio": float(s_audio),
        }, baseline_av_corr


def compute_pareto_frontier(points: List[Dict[str, Any]], time_key: str, quality_key: str = "Q_total") -> List[int]:
    pareto_ids = []
    for p in points:
        t_p = p[time_key]
        q_p = p[quality_key]
        dominated = False
        for other in points:
            if other["run_id"] == p["run_id"]:
                continue
            t_o = other[time_key]
            q_o = other[quality_key]
            if (t_o <= t_p and q_o >= q_p) and (t_o < t_p or q_o > q_p):
                dominated = True
                break
        if not dominated:
            pareto_ids.append(p["run_id"])

    pareto_points = [p for p in points if p["run_id"] in pareto_ids]
    pareto_points.sort(key=lambda x: x[time_key])
    return [p["run_id"] for p in pareto_points]


def save_video_with_audio(frames_np: Any, audio_np: Any, sample_rate: int, out_mp4: Path, fps: int = 24) -> None:
    import imageio.v2 as imageio
    import numpy as np
    import soundfile as sf

    tmp_vid = out_mp4.with_suffix(".tmp_video.mp4")
    tmp_wav = out_mp4.with_suffix(".tmp_audio.wav")

    writer = imageio.get_writer(str(tmp_vid), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    for t in range(frames_np.shape[0]):
        frame_u8 = (np.clip(frames_np[t], 0.0, 1.0) * 255.0).astype(np.uint8)
        writer.append_data(frame_u8)
    writer.close()

    sf.write(str(tmp_wav), np.clip(audio_np.T, -1.0, 1.0), sample_rate)

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp_vid),
            "-i", str(tmp_wav),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_mp4),
        ],
        check=True,
    )
    tmp_vid.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)


def generate_visual_artifacts(
    results: List[Dict[str, Any]],
    all_frames: Dict[int, Any],
    baseline_audio: Any,
    audio_sr: int,
) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. speed_quality_pareto.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    for ax_idx, (time_key, speedup_key, title_suffix, pareto_flag) in enumerate([
        ("cold_e2e_sec", "cold_speedup", "Single-Shot / Cold-Start (T_cold)", "is_pareto_cold"),
        ("warm_e2e_sec", "warm_speedup", "Steady-State / Warm Batch (T_warm)", "is_pareto_warm"),
    ]):
        ax = axes[ax_idx]
        pareto_pts = [r for r in results if r[pareto_flag]]
        pareto_pts.sort(key=lambda x: x[time_key])

        if len(pareto_pts) >= 2:
            px = [p[time_key] for p in pareto_pts]
            py = [p["Q_total"] for p in pareto_pts]
            ax.plot(px, py, linestyle="--", linewidth=2.4, color="#d62728", alpha=0.85, label="Pareto Frontier")

        for idx, r in enumerate(results):
            x = r[time_key]
            y = r["Q_total"]
            is_p = r[pareto_flag]
            marker = "*" if is_p else "o"
            size = 240 if is_p else 120
            edge = "#000000" if is_p else "none"
            ax.scatter(
                x, y, s=size, c=colors[idx], marker=marker,
                edgecolors=edge, linewidths=1.5 if is_p else 0.0, zorder=5
            )
            label_txt = f"Run {r['run_id']} {r['code']}\n({r[speedup_key]:.2f}x, Q={y:.1f})"
            offset_y = 6 if (idx % 2 == 0) else -16
            ax.annotate(
                label_txt, (x, y),
                textcoords="offset points", xytext=(8, offset_y),
                fontsize=8.5, fontweight="bold" if is_p else "normal",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[idx], alpha=0.8),
            )

        ax.set_title(f"Speed vs. 4D Quality Score — {title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xlabel("End-to-End Generation Latency (seconds, Lower = Faster ←)", fontsize=10.5)
        ax.set_ylabel("4D Composite Quality Score Q_total ∈ [0, 100] (Higher = Better ↑)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speed_quality_pareto.png", dpi=180)
    plt.close(fig)

    # 2. frame_metrics_curves.png
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=160, sharex=True)
    frame_indices = np.arange(124)
    for idx, r in enumerate(results):
        if r["run_id"] == 1:
            continue
        lbl = f"Run {r['run_id']} {r['code']} ({r['label']})"
        axes[0].plot(frame_indices, r["ref_metrics"]["psnr_per_frame"], label=lbl, color=colors[idx], linewidth=1.5)
        axes[1].plot(frame_indices, r["ref_metrics"]["ssim_per_frame"], label=lbl, color=colors[idx], linewidth=1.5)
        axes[2].plot(frame_indices, r["ref_metrics"]["lpips_per_frame"], label=lbl, color=colors[idx], linewidth=1.5)

    axes[0].set_ylabel("PSNR (dB) ↑")
    axes[0].set_title("Per-Frame Fidelity Trajectory Across 124 Frames (vs. Run 1 Baseline)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", ncol=2, fontsize=8)

    axes[1].set_ylabel("SSIM ↑")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].set_ylabel("LPIPS ↓")
    axes[2].set_xlabel("Frame Index (0 .. 123 @ 24 FPS)")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "frame_metrics_curves.png", dpi=160)
    plt.close(fig)

    # 3. keyframe_strip_8runs.png (8 runs x 5 keyframes: 0, 30, 61, 92, 123)
    kf_indices = [0, 30, 61, 92, 123]
    thumb_w, thumb_h = 336, 192
    pad_left = 230
    pad_top = 44
    canvas = np.ones((pad_top + 8 * thumb_h, pad_left + len(kf_indices) * thumb_w, 3), dtype=np.uint8) * 24

    for col_i, f_idx in enumerate(kf_indices):
        x0 = pad_left + col_i * thumb_w
        cv2.putText(
            canvas, f"Frame {f_idx:03d} ({f_idx*100//123}%)",
            (x0 + 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA
        )

    for row_i, r in enumerate(results):
        run_id = r["run_id"]
        y0 = pad_top + row_i * thumb_h
        frames = all_frames[run_id]
        txt1 = f"Run {run_id} {r['code']}"
        txt2 = f"Warm: {r['warm_e2e_sec']:.2f}s ({r['warm_speedup']:.2f}x)"
        txt3 = f"Q_total: {r['Q_total']:.1f} | PSNR: {r['ref_metrics']['psnr_mean']:.1f}dB"
        cv2.putText(canvas, txt1, (10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 215, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, txt2, (10, y0 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt3, (10, y0 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 255, 180), 1, cv2.LINE_AA)

        for col_i, f_idx in enumerate(kf_indices):
            x0 = pad_left + col_i * thumb_w
            img_u8 = (np.clip(frames[f_idx], 0.0, 1.0) * 255.0).astype(np.uint8)
            thumb = cv2.resize(img_u8, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb

    cv2.imwrite(str(OUTPUT_DIR / "keyframe_strip_8runs.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # 4. comparison_grid_8runs.mp4
    cell_w, cell_h = 480, 270
    T = 124
    grid_frames = np.zeros((T, cell_h * 2, cell_w * 4, 3), dtype=np.float32)
    for idx, r in enumerate(results):
        run_id = r["run_id"]
        row_i = idx // 4
        col_i = idx % 4
        y0, x0 = row_i * cell_h, col_i * cell_w
        frames = all_frames[run_id]
        overlay = f"R{run_id} {r['code']} | {r['warm_e2e_sec']:.1f}s | Q={r['Q_total']:.1f}"
        for t in range(T):
            cell_u8 = (np.clip(frames[t], 0.0, 1.0) * 255.0).astype(np.uint8)
            cell_resized = cv2.resize(cell_u8, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(cell_resized, (6, 6), (370, 34), (0, 0, 0), -1)
            cv2.putText(
                cell_resized, overlay, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 100), 1, cv2.LINE_AA
            )
            grid_frames[t, y0 : y0 + cell_h, x0 : x0 + cell_w] = cell_resized.astype(np.float32) / 255.0

    save_video_with_audio(grid_frames, baseline_audio, audio_sr, OUTPUT_DIR / "comparison_grid_8runs.mp4", fps=24)


def main() -> None:
    bootstrap_environment()
    dl_stats = download_golden_5pack()

    sys.path.insert(0, str(COMFY_ROOT))
    import numpy as np
    import psutil
    import torch
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 1024
    torch._dynamo.config.suppress_errors = False

    import comfy.ldm.minimax.model as minimax_model_mod
    import comfy.ldm.modules.attention as comfy_attn_mod
    import comfy.model_management as mm
    import comfy.model_prefetch
    import comfy.ops
    import nodes
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_easycache import EasyCacheHolder, EasyCacheNode
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from comfy_extras.nodes_torch_compile import TorchCompileModel

    # 1. Verify SageAttention kernel directly on H3 head dimensions [1, 40, 256, 128] before any run
    from sageattention import sageattn
    comfy_attn_mod.sageattn = sageattn
    comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True

    q_test = torch.randn(1, 40, 256, 128, device="cuda", dtype=torch.bfloat16)
    out_test = sageattn(q_test, q_test, q_test, is_causal=False, tensor_layout="HND")
    assert out_test.shape == q_test.shape, "SageAttention smoke test failed!"
    del q_test, out_test

    # Mark opaque C++/CUDA custom kernels (comfy_kitchen INT8 fused linear_input_act, sageattn, prefetch_queue_pop)
    # with @torch.compiler.disable so TorchCompileModel(backend="inductor") does not crash on FakeTensor.data_ptr()
    comfy.ops._try_quantized_linear_input_act = torch.compiler.disable(comfy.ops._try_quantized_linear_input_act)
    comfy_attn_mod.attention_sage = torch.compiler.disable(comfy_attn_mod.attention_sage)
    comfy.model_prefetch.prefetch_queue_pop = torch.compiler.disable(comfy.model_prefetch.prefetch_queue_pop)

    # Attach log detector to guarantee attention_sage never silently falls back to attention_pytorch
    sage_detector = SageFallbackDetector()
    logging.getLogger().addHandler(sage_detector)

    # Pure function pointer swap (zero Python wrapper mutation inside compiled region!)
    def set_sage_attention(enabled: bool) -> None:
        target_fn = comfy_attn_mod.attention_sage if enabled else comfy_attn_mod.attention_pytorch
        comfy_attn_mod.optimized_attention = target_fn
        minimax_model_mod.optimized_attention = target_fn

    # Hook EasyCacheHolder.reset (which runs in OUTER_SAMPLE finally block, outside the compiled DiT forward)
    last_easycache_stats: Dict[str, Any] = {
        "total_steps_skipped": 0,
        "output_change_rates": [],
        "approx_output_change_rates": [],
    }
    orig_ec_reset = EasyCacheHolder.reset

    def hooked_ec_reset(self):
        last_easycache_stats["total_steps_skipped"] = int(self.total_steps_skipped)
        last_easycache_stats["output_change_rates"] = [float(x) for x in self.output_change_rates]
        last_easycache_stats["approx_output_change_rates"] = [float(x) for x in self.approx_output_change_rates]
        return orig_ec_reset(self)

    EasyCacheHolder.reset = hooked_ec_reset

    props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 90)
    print(f"🖥️ [Phase 3/5] Hardware & Pre-Encoding Setup on {props.name} ({props.total_memory / (1024**3):.1f} GiB VRAM)")
    print("=" * 90)

    width, height, length, steps, seed = 1344, 768, 124, 8, 42
    first_tensor, last_tensor, first_np, last_np = prepare_first_and_last_frames(width, height)

    t_enc0 = time.perf_counter()
    clip_loader = nodes.CLIPLoader()
    clip = clip_loader.load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]

    vae_loader = nodes.VAELoader()
    video_vae = vae_loader.load_vae("minimax_h3_video_vae_int8_convrot.safetensors")[0]
    audio_vae = vae_loader.load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]

    with torch.inference_mode():
        base_cond, base_latent = MiniMaxH3ImageToVideo.execute(
            clip=clip,
            vae=video_vae,
            prompt=OFFICIAL_PROMPT,
            width=width,
            height=height,
            length=length,
            first_frame=first_tensor,
            last_frame=last_tensor,
        )
    torch.cuda.synchronize()
    cond_encode_sec = time.perf_counter() - t_enc0
    print(f"  ✅ Shared Text + First/Last Keyframe Conditioning encoded in {cond_encode_sec:.2f}s")

    # Free Qwen3-VL-32B from VRAM
    del clip
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    t_load0 = time.perf_counter()
    unet_loader = nodes.UNETLoader()
    base_model = unet_loader.load_unet("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default")[0]
    lora_loader = nodes.LoraLoaderModelOnly()
    lora_model = lora_loader.load_lora_model_only(
        base_model, "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.0
    )[0]
    mm.load_models_gpu([lora_model])
    torch.cuda.synchronize()
    dit_load_sec = time.perf_counter() - t_load0
    print(f"  ✅ FL2VA DiT + 8-Step Turbo LoRA loaded and resident in A100 VRAM in {dit_load_sec:.2f}s")

    print("\n" + "=" * 90)
    print("🚀 [Phase 4/5] Executing 2^3 = 8 Full-Factorial Acceleration Ablation Matrix")
    print("=" * 90)

    sampler_obj = KSamplerSelect.execute("res_multistep")[0]
    vae_decoder = nodes.VAEDecode()

    def execute_single_pass(
        work_model: Any,
        s_flag: bool,
        c_flag: bool,
        pass_tag: str,
    ) -> Dict[str, Any]:
        last_easycache_stats["total_steps_skipped"] = 0
        last_easycache_stats["output_change_rates"] = []
        last_easycache_stats["approx_output_change_rates"] = []
        sage_detector.fallback_errors.clear()

        set_sage_attention(s_flag)
        cond_copy = clone_conditioning(base_cond)
        latent_copy = clone_av_latent(base_latent)
        noise_obj = RandomNoise.execute(seed)[0]

        frames_before = int(torch._dynamo.utils.counters["frames"]["total"])
        ok_before = int(torch._dynamo.utils.counters["frames"]["ok"])

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        with torch.inference_mode():
            sigmas = BasicScheduler.execute(work_model, "simple", steps, 1.0)[0]
            guider = BasicGuider.execute(work_model, cond_copy)[0]
            out_latent, out_denoised = SamplerCustomAdvanced.execute(
                noise_obj, guider, sampler_obj, sigmas, latent_copy
            )
            torch.cuda.synchronize()
            t_denoise = time.perf_counter() - t0

            t_v0 = time.perf_counter()
            video_frames_t = vae_decoder.decode(video_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_video = time.perf_counter() - t_v0

            t_a0 = time.perf_counter()
            audio_dict = VAEDecodeAudio.execute(audio_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_audio = time.perf_counter() - t_a0

        t_e2e = time.perf_counter() - t0
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if s_flag and sage_detector.fallback_errors:
            raise RuntimeError(f"SageAttention fallback detected during {pass_tag}: {sage_detector.fallback_errors}")

        frames_compiled_delta = int(torch._dynamo.utils.counters["frames"]["total"]) - frames_before
        ok_compiled_delta = int(torch._dynamo.utils.counters["frames"]["ok"]) - ok_before

        frames_np = video_frames_t.detach().cpu().float().numpy()
        if frames_np.ndim == 5:
            frames_np = frames_np[0]
        waveform_np = audio_dict["waveform"].detach().cpu().float().numpy()
        if waveform_np.ndim == 3:
            waveform_np = waveform_np[0]
        sample_rate = int(audio_dict["sample_rate"])

        del guider, out_latent, out_denoised, video_frames_t, audio_dict

        steps_skipped = last_easycache_stats["total_steps_skipped"] if c_flag else 0
        attn_backend_str = "sageattn" if s_flag else "sdpa_pytorch"
        print(
            f"    [{pass_tag}] Denoise: {t_denoise:.2f}s | VideoVAE: {t_vae_video:.2f}s | "
            f"AudioVAE: {t_vae_audio:.2f}s | E2E: {t_e2e:.2f}s | PeakVRAM: {peak_vram_gib:.2f} GiB | "
            f"Skipped: {steps_skipped}/{steps} | Attn: {attn_backend_str} | DynamoFrames(+{frames_compiled_delta}/ok+{ok_compiled_delta})"
        )

        return {
            "denoise_sec": float(t_denoise),
            "vae_video_sec": float(t_vae_video),
            "vae_audio_sec": float(t_vae_audio),
            "e2e_sec": float(t_e2e),
            "peak_vram_gib": float(peak_vram_gib),
            "steps_skipped": int(steps_skipped),
            "output_change_rates": list(last_easycache_stats["output_change_rates"]),
            "approx_output_change_rates": list(last_easycache_stats["approx_output_change_rates"]),
            "attention_backend": attn_backend_str,
            "dynamo_frames_delta": frames_compiled_delta,
            "dynamo_ok_delta": ok_compiled_delta,
            "frames_np": frames_np,
            "waveform_np": waveform_np,
            "sample_rate": sample_rate,
        }

    # Exact EasyCache parameters matching README.md (thresh=0.20, start=0.15, end=0.95)
    easycache_threshold = 0.20
    easycache_start_pct = 0.15
    easycache_end_pct = 0.95

    raw_run_outputs: Dict[int, Dict[str, Any]] = {}
    all_frames_dict: Dict[int, Any] = {}

    print("\n  🔥 Running 1 Warmup Pass ([0,0,0]) to initialize CUDA/cuBLAS workspaces...")
    warmup_model = lora_model.clone()
    _ = execute_single_pass(warmup_model, False, False, "WARMUP")
    del warmup_model

    for run_id, code, label, s_flag, c_flag, t_flag in ABLATION_MATRIX:
        print(f"\n▶️ Run {run_id}/8 {code} — {label} (Sage={s_flag}, EasyCache={c_flag}, Compile={t_flag})")
        ckpt_json = OUTPUT_DIR / f"ckpt_run{run_id}.json"
        ckpt_npz = OUTPUT_DIR / f"ckpt_run{run_id}.npz"
        if ckpt_json.exists() and ckpt_npz.exists():
            with open(ckpt_json, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            npz_data = np.load(ckpt_npz)
            all_frames_dict[run_id] = npz_data["frames_np"]
            cached_meta["waveform_np"] = npz_data["waveform_np"]
            raw_run_outputs[run_id] = cached_meta
            print(
                f"    [CACHED] Cold: {cached_meta['cold_e2e_sec']:.2f}s | "
                f"Warm: {cached_meta['warm_e2e_sec']:.2f}s | "
                f"PeakVRAM: {cached_meta['peak_vram_gib']:.2f} GiB | "
                f"Skipped: {cached_meta['steps_skipped']}/{steps}"
            )
            continue

        work_model = lora_model.clone()

        if c_flag:
            work_model = EasyCacheNode.execute(
                work_model,
                reuse_threshold=easycache_threshold,
                start_percent=easycache_start_pct,
                end_percent=easycache_end_pct,
                verbose=True,
            )[0]

        if t_flag:
            reset_inductor_and_dynamo_caches(run_id, "cold")
            work_model = TorchCompileModel.execute(work_model, backend="inductor")[0]
            pass_cold = execute_single_pass(work_model, s_flag, c_flag, f"Run{run_id}a-ColdCompile")
            pass_warm = execute_single_pass(work_model, s_flag, c_flag, f"Run{run_id}b-WarmSteady")
        else:
            pass_single = execute_single_pass(work_model, s_flag, c_flag, f"Run{run_id}-Eager")
            pass_cold = pass_single
            pass_warm = pass_single

        mp4_path = OUTPUT_DIR / f"run{run_id}_{label}.mp4"
        save_video_with_audio(
            pass_warm["frames_np"], pass_warm["waveform_np"], pass_warm["sample_rate"], mp4_path, fps=24
        )

        all_frames_dict[run_id] = pass_warm["frames_np"]
        run_meta = {
            "run_id": run_id,
            "code": code,
            "label": label,
            "S_sage": s_flag,
            "C_cache": c_flag,
            "T_compile": t_flag,
            "easycache_threshold": easycache_threshold if c_flag else None,
            "cold_denoise_sec": pass_cold["denoise_sec"],
            "warm_denoise_sec": pass_warm["denoise_sec"],
            "vae_video_sec": pass_warm["vae_video_sec"],
            "vae_audio_sec": pass_warm["vae_audio_sec"],
            "cold_e2e_sec": pass_cold["e2e_sec"],
            "warm_e2e_sec": pass_warm["e2e_sec"],
            "compile_overhead_sec": max(0.0, pass_cold["e2e_sec"] - pass_warm["e2e_sec"]),
            "peak_vram_gib": max(pass_cold["peak_vram_gib"], pass_warm["peak_vram_gib"]),
            "steps_skipped": pass_warm["steps_skipped"],
            "output_change_rates": pass_warm["output_change_rates"],
            "approx_output_change_rates": pass_warm["approx_output_change_rates"],
            "attention_backend": pass_warm["attention_backend"],
            "cold_dynamo_frames": pass_cold["dynamo_frames_delta"],
            "warm_dynamo_frames": pass_warm["dynamo_frames_delta"],
            "sample_rate": pass_warm["sample_rate"],
        }
        with open(ckpt_json, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        np.savez(ckpt_npz, frames_np=pass_warm["frames_np"], waveform_np=pass_warm["waveform_np"])

        run_meta["waveform_np"] = pass_warm["waveform_np"]
        raw_run_outputs[run_id] = run_meta

        del work_model
        gc.collect()

    # Completely unload and free all ComfyUI models before initializing QualityEvaluator
    del lora_model, base_model, video_vae, audio_vae, base_cond, base_latent
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("📊 [Phase 5/5] Computing 4-Dimensional Benchmark Quality Scores & Pareto Frontier")
    print("=" * 90)

    evaluator = QualityEvaluator(device="cuda")
    gt_frames = all_frames_dict[1]
    gt_audio = raw_run_outputs[1]["waveform_np"]
    gt_sr = raw_run_outputs[1]["sample_rate"]
    t_base_cold = raw_run_outputs[1]["cold_e2e_sec"]
    t_base_warm = raw_run_outputs[1]["warm_e2e_sec"]

    baseline_av_corr = None
    final_results: List[Dict[str, Any]] = []

    for run_id in range(1, 9):
        r = raw_run_outputs[run_id]
        frames_t = all_frames_dict[run_id]
        audio_t = r.pop("waveform_np")
        sr_t = r.pop("sample_rate")
        is_base = (run_id == 1)

        ref_m = evaluator.compute_reference_metrics(frames_t, gt_frames, is_baseline=is_base)
        vbench_m, fl2v_m, motion_energy_t = evaluator.compute_vbench_and_fl2v_metrics(
            frames_t, first_np, last_np
        )
        audio_m, baseline_av_corr = evaluator.compute_audio_and_avsync_metrics(
            audio_t, gt_audio, motion_energy_t,
            baseline_av_corr=baseline_av_corr, is_baseline=is_base, sample_rate=sr_t
        )

        q_total = (
            0.35 * ref_m["S_ref"]
            + 0.25 * vbench_m["S_vbench"]
            + 0.25 * fl2v_m["S_fl2v"]
            + 0.15 * audio_m["S_audio"]
        )

        r["cold_speedup"] = float(t_base_cold / r["cold_e2e_sec"])
        r["warm_speedup"] = float(t_base_warm / r["warm_e2e_sec"])
        r["S_ref"] = ref_m["S_ref"]
        r["S_vbench"] = vbench_m["S_vbench"]
        r["S_fl2v"] = fl2v_m["S_fl2v"]
        r["S_audio"] = audio_m["S_audio"]
        r["Q_total"] = float(q_total)
        r["ref_metrics"] = ref_m
        r["vbench_metrics"] = vbench_m
        r["fl2v_metrics"] = fl2v_m
        r["audio_metrics"] = audio_m
        final_results.append(r)

        print(
            f"  [Run {run_id} {r['code']}] Cold: {r['cold_e2e_sec']:.2f}s ({r['cold_speedup']:.2f}x) | "
            f"Warm: {r['warm_e2e_sec']:.2f}s ({r['warm_speedup']:.2f}x) | "
            f"Q_total={r['Q_total']:.2f} [S_ref={r['S_ref']:.1f}, S_vb={r['S_vbench']:.1f}, "
            f"S_fl2v={r['S_fl2v']:.1f}, S_aud={r['S_audio']:.1f}] | "
            f"PSNR={ref_m['psnr_mean']:.2f}dB (min {ref_m['psnr_min']:.2f}) | "
            f"SSIM={ref_m['ssim_mean']:.4f} | LPIPS={ref_m['lpips_mean']:.4f}"
        )

    pareto_cold_ids = compute_pareto_frontier(final_results, time_key="cold_e2e_sec", quality_key="Q_total")
    pareto_warm_ids = compute_pareto_frontier(final_results, time_key="warm_e2e_sec", quality_key="Q_total")

    eager_pair_map = {4: 1, 6: 2, 7: 3, 8: 5}
    for r in final_results:
        r["is_pareto_cold"] = r["run_id"] in pareto_cold_ids
        r["is_pareto_warm"] = r["run_id"] in pareto_warm_ids
        if r["run_id"] in eager_pair_map:
            eager_r = raw_run_outputs[eager_pair_map[r["run_id"]]]
            t_eager = eager_r["warm_e2e_sec"]
            t_warm = r["warm_e2e_sec"]
            t_cold = r["cold_e2e_sec"]
            if t_warm < t_eager - 1e-3:
                overhead = max(0.0, t_cold - t_warm)
                r["breakeven_videos_vs_eager"] = max(1, int(math.ceil(overhead / (t_eager - t_warm))))
            else:
                r["breakeven_videos_vs_eager"] = None
        else:
            r["breakeven_videos_vs_eager"] = 1

    generate_visual_artifacts(final_results, all_frames_dict, gt_audio, gt_sr)

    summary_payload = {
        "hardware": {
            "gpu_name": props.name,
            "vram_gib": round(props.total_memory / (1024 ** 3), 2),
            "ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "torch_version": torch.__version__,
        },
        "config": {
            "resolution": f"{width}x{height}",
            "frames": length,
            "fps": 24,
            "steps": steps,
            "seed": seed,
            "cond_encode_sec": round(cond_encode_sec, 3),
            "dit_load_sec": round(dit_load_sec, 3),
            "download_sec": round(dl_stats["download_sec"], 3),
            "easycache_threshold": easycache_threshold,
            "easycache_start_percent": easycache_start_pct,
            "easycache_end_percent": easycache_end_pct,
        },
        "pareto_frontier_cold_run_ids": pareto_cold_ids,
        "pareto_frontier_warm_run_ids": pareto_warm_ids,
        "runs": final_results,
    }

    json_path = OUTPUT_DIR / "ablation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"✅ Experiment 02 Completed! All artifacts saved to {OUTPUT_DIR}")
    print(f"   • Cold-Start Pareto Frontier Runs : {pareto_cold_ids}")
    print(f"   • Warm-Batch Pareto Frontier Runs : {pareto_warm_ids}")
    print("=" * 90)


if __name__ == "__main__":
    main()
