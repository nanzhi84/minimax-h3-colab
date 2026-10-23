#!/usr/bin/env python3
"""
Experiment 02 (Supplemental): H100-80GB + SageAttention Cross-Architecture Ablation
Evaluates whether SageAttention is a positive or negative optimization on NVIDIA H100 (Hopper SM90)
compared to NVIDIA A100 (Ampere SM80) on MiniMax H3-Base-FL2VA (1344x768, 124 frames, 8 steps, seed=42).

Includes:
  1. Operator-Level Microbenchmark at H3-FL2VA's exact Joint AV Attention shape (B=1, H=40, N=42058, D=128, BF16):
     - PyTorch SDPA (FlashAttention-2 / cuDNN SDPA on Hopper SM90)
     - ComfyUI attention_pytorch(skip_reshape=True)
     - ComfyUI attention_sage(skip_reshape=True)
     - Raw sageattn(tensor_layout="HND") vs sageattn(tensor_layout="NHD")
  2. End-to-End 2x2 Ablation Matrix on H100 ({SDPA, SageAttention} x {NoCache, EasyCache}):
     - Run 1 [0,0,0] : H100_Baseline (SDPA + Eager)
     - Run 2 [1,0,0] : H100_Only_Sage (SageAttention + Eager)
     - Run 3 [0,1,0] : H100_Only_Cache (SDPA + EasyCache)
     - Run 5 [1,1,0] : H100_Sage_Cache (SageAttention + EasyCache)
  3. Full 4-Dimensional Benchmark Quality Evaluation (S_ref, S_vbench, S_fl2v, S_audio, Q_total) on H100.

Outputs:
  - /content/exp02_h100_outputs/h100_sage_ablation_metrics.json
  - /content/exp02_h100_outputs/h100_run1_Baseline.mp4
  - /content/exp02_h100_outputs/h100_run2_Only_Sage.mp4
  - /content/exp02_h100_outputs/h100_run3_Only_Cache.mp4
  - /content/exp02_h100_outputs/h100_run5_Sage_Cache.mp4
"""

import os
import sys

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import importlib.metadata
import inspect
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
OUTPUT_DIR = Path("/content/exp02_h100_outputs")
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

# 2x2 SageAttention x EasyCache Ablation Matrix on H100: (run_id, code, label, S_sage, C_cache)
H100_ABLATION_MATRIX = [
    (1, "[0,0,0]", "H100_Run1_Baseline",  False, False),
    (2, "[1,0,0]", "H100_Run2_Only_Sage", True,  False),
    (3, "[0,1,0]", "H100_Run3_Only_Cache", False, True),
    (5, "[1,1,0]", "H100_Run5_Sage_Cache", True,  True),
]


def run_cmd(cmd: List[str], cwd: Path = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def bootstrap_environment() -> None:
    print("=" * 90)
    print("🛠️ [Phase 1/5] Bootstrapping ComfyUI + SageAttention + Quality Evaluation Dependencies on H100")
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

    first_tensor = torch.from_numpy(first_np).unsqueeze(0)
    last_tensor = torch.from_numpy(last_np).unsqueeze(0)
    return first_tensor, last_tensor, first_np, last_np


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
    nested = base_latent["samples"]
    v_lat, a_lat = nested.unbind()
    return {
        "samples": comfy.nested_tensor.NestedTensor((v_lat.clone(), a_lat.clone()))
    }


class SageFallbackDetector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.fallback_errors: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Error running sage attention" in msg or "falling back to pytorch attention" in msg:
            self.fallback_errors.append(msg)


class QualityEvaluator:
    def __init__(self, device: str = "cuda"):
        import lpips
        import torch
        import torchvision.models.optical_flow as tv_flow
        from transformers import AutoModel, CLIPVisionModelWithProjection

        self.device = torch.device(device)
        print("  🔬 Loading LPIPS (AlexNet), DINOv2 (facebook/dinov2-small), CLIP (openai/clip-vit-base-patch32), RAFT, and MUSIQ...")
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        raft_weights = tv_flow.Raft_Small_Weights.DEFAULT
        self.raft_model = tv_flow.raft_small(weights=raft_weights, progress=False).to(self.device).eval()

        import pyiqa
        self.musiq_metric = pyiqa.create_metric("musiq", device=self.device)

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def extract_dino_and_clip_features(self, frames_np: Any, batch_size: int = 16) -> Tuple[Any, Any]:
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
            "S_ref": float(s_ref),
        }

    def compute_vbench_and_fl2v_metrics(
        self,
        frames_test: Any,
        first_anchor_np: Any,
        last_anchor_np: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        import numpy as np
        import torch
        import torch.nn.functional as F

        T = frames_test.shape[0]
        dino_feats, clip_feats = self.extract_dino_and_clip_features(frames_test)
        anc_dino, anc_clip = self.extract_dino_and_clip_features(np.stack([first_anchor_np, last_anchor_np], axis=0))

        dino_adj = torch.sum(dino_feats[:-1] * dino_feats[1:], dim=-1).numpy()
        dino_first = torch.sum(dino_feats[1:] * dino_feats[0:1], dim=-1).numpy()
        subject_consistency = float(0.5 * np.mean(dino_adj) + 0.5 * np.mean(dino_first))

        clip_adj = torch.sum(clip_feats[:-1] * clip_feats[1:], dim=-1).numpy()
        clip_first = torch.sum(clip_feats[1:] * clip_feats[0:1], dim=-1).numpy()
        background_consistency = float(0.5 * np.mean(clip_adj) + 0.5 * np.mean(clip_first))

        sub_bg_consistency = 0.5 * subject_consistency + 0.5 * background_consistency

        diffs = np.abs(frames_test[1:] - frames_test[:-1])
        frame_mae = np.mean(diffs, axis=(1, 2, 3))
        temporal_flickering_score = float(1.0 - np.mean(frame_mae))

        flow_mags = []
        with torch.no_grad():
            for i in range(0, T - 1, 8):
                end_i = min(T - 1, i + 8)
                b1 = torch.from_numpy(frames_test[i:end_i]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b2 = torch.from_numpy(frames_test[i + 1 : end_i + 1]).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
                b1 = F.interpolate(b1, size=(256, 448), mode="bilinear", align_corners=False) * 2.0 - 1.0
                b2 = F.interpolate(b2, size=(256, 448), mode="bilinear", align_corners=False) * 2.0 - 1.0
                flow_pred = self.raft_model(b1, b2)[-1]
                mag = torch.sqrt(torch.sum(flow_pred ** 2, dim=1)).mean(dim=(1, 2)).cpu().tolist()
                flow_mags.extend(mag)

        flow_mags_np = np.array(flow_mags, dtype=np.float32)
        flow_accel = np.abs(np.diff(flow_mags_np))
        motion_smoothness = float(1.0 / (1.0 + np.mean(flow_accel) / (np.mean(flow_mags_np) + 1e-4)))

        sample_indices = np.linspace(0, T - 1, 9, dtype=int)
        musiq_scores = []
        with torch.no_grad():
            for idx in sample_indices:
                img_t = torch.from_numpy(frames_test[idx]).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                img_t = F.interpolate(img_t, size=(384, 672), mode="bilinear", align_corners=False)
                score = float(self.musiq_metric(img_t).item()) / 100.0
                musiq_scores.append(min(1.0, max(0.0, score)))
        imaging_quality = float(np.mean(musiq_scores))

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

        stft_snrs = []
        for nperseg in (512, 1024, 2048):
            _, _, Z_t = signal.stft(a_test, fs=sample_rate, nperseg=nperseg)
            _, _, Z_g = signal.stft(a_gt, fs=sample_rate, nperseg=nperseg)
            mag_t = np.abs(Z_t)
            mag_g = np.abs(Z_g)
            err_power = np.mean((mag_t - mag_g) ** 2) + 1e-12
            sig_power = np.mean(mag_g ** 2) + 1e-12
            stft_snrs.append(float(10.0 * math.log10(sig_power / err_power)))

        stft_snr_db = float(np.mean(stft_snrs))

        n_fft = 2048
        _, _, Z_mel_t = signal.stft(a_test, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - 512)
        _, _, Z_mel_g = signal.stft(a_gt, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - 512)
        pow_t = np.mean(np.abs(Z_mel_t) ** 2, axis=0)
        pow_g = np.mean(np.abs(Z_mel_g) ** 2, axis=0)
        fbank = cls._build_mel_filterbank(sr=sample_rate, n_fft=n_fft, n_mels=80)
        mel_spec_t = np.log10(np.matmul(fbank, pow_t) + 1e-8)
        mel_spec_g = np.log10(np.matmul(fbank, pow_g) + 1e-8)

        vec_t = mel_spec_t.reshape(-1)
        vec_g = mel_spec_g.reshape(-1)
        log_mel_cos = float(
            np.dot(vec_t, vec_g) / (np.linalg.norm(vec_t) * np.linalg.norm(vec_g) + 1e-12)
        )

        rms_t = math.sqrt(float(np.mean(a_test ** 2)) + 1e-12)
        rms_g = math.sqrt(float(np.mean(a_gt ** 2)) + 1e-12)
        rms_diff_db = abs(20.0 * math.log10(rms_t / rms_g))

        snr_norm = min(1.0, max(0.0, (stft_snr_db + 5.0) / 35.0))
        mel_norm = min(1.0, max(0.0, log_mel_cos))
        s_audio = 100.0 * (0.45 * mel_norm + 0.35 * snr_norm + 0.20 * av_sync_retention)

        return {
            "stft_snr_db": stft_snr_db,
            "log_mel_cosine_sim": log_mel_cos,
            "rms_diff_db": rms_diff_db,
            "av_sync_corr": av_corr,
            "av_sync_retention": av_sync_retention,
            "S_audio": float(s_audio),
        }, baseline_av_corr


def save_video_with_audio(
    frames_np: Any,
    audio_np: Any,
    sample_rate: int,
    out_mp4: Path,
    fps: int = 24,
) -> None:
    import imageio.v2 as imageio
    import numpy as np
    import soundfile as sf

    tmp_vid = out_mp4.with_suffix(".tmp_video.mp4")
    tmp_wav = out_mp4.with_suffix(".tmp_audio.wav")

    frames_u8 = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(str(tmp_vid), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    for frame in frames_u8:
        writer.append_data(frame)
    writer.close()

    wav_data = audio_np.T if audio_np.ndim == 2 else audio_np
    sf.write(str(tmp_wav), wav_data, sample_rate)

    ffmpeg_bin = "ffmpeg"
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            tmp_vid.replace(out_mp4)
            tmp_wav.unlink(missing_ok=True)
            return

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
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


def run_attention_microbenchmark(
    comfy_attn_mod: Any, sageattn_fn: Any, sage_detector: SageFallbackDetector
) -> Dict[str, Any]:
    """
    Benchmarks exact single-call attention latency at H3-FL2VA's Joint AV Sequence Shape:
      B=1, H=40, N=42058, D=128, dtype=bfloat16
    and inspects how ComfyUI's attention_sage vs attention_pytorch handles tensor layouts.
    """
    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 90)
    print("🔬 [Phase 3A] Operator-Level Attention Microbenchmark at H3-FL2VA Shape (B=1, H=40, N=42058, D=128)")
    print("=" * 90)

    sage_version = importlib.metadata.version("sageattention")
    attn_sage_src = inspect.getsource(comfy_attn_mod.attention_sage)
    print(f"  • Installed sageattention package version: {sage_version}")
    print("  • ComfyUI attention_sage source code:")
    for line in attn_sage_src.strip().splitlines():
        print(f"      {line}")

    B, H, N, D = 1, 40, 42058, 128
    # Total FLOPs for forward self-attention = 4 * B * H * N^2 * D
    flops = 4.0 * B * H * (N ** 2) * D

    q_hnd = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k_hnd = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v_hnd = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    q_nhd = q_hnd.transpose(1, 2).contiguous()
    k_nhd = k_hnd.transpose(1, 2).contiguous()
    v_nhd = v_hnd.transpose(1, 2).contiguous()

    def bench_op(
        name: str, fn, warmup: int = 4, reps: int = 12, compute_tflops: bool = True, check_sage: bool = False
    ) -> Dict[str, Any]:
        sage_detector.fallback_errors.clear()
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = fn()
        torch.cuda.synchronize()
        if check_sage and sage_detector.fallback_errors:
            raise RuntimeError(f"SageAttention fallback detected during microbenchmark {name}: {sage_detector.fallback_errors}")
        avg_sec = (time.perf_counter() - t0) / reps
        avg_ms = avg_sec * 1000.0
        tflops = round((flops / avg_sec) / 1e12, 2) if compute_tflops else None
        # 40 blocks x 8 steps = 320 calls per video
        est_320_calls_sec = avg_sec * 320.0
        tflops_str = f"{tflops:6.1f} TFLOPS" if tflops is not None else "    N/A TFLOPS"
        print(
            f"  [{name:<28}] {avg_ms:7.2f} ms/call | {tflops_str} | "
            f"320 calls (40 blk x 8 step) = {est_320_calls_sec:6.2f}s"
        )
        return {
            "avg_ms": round(avg_ms, 3),
            "tflops": tflops,
            "est_320_calls_sec": round(est_320_calls_sec, 2),
        }

    results = {
        "sageattention_version": sage_version,
        "shape_B_H_N_D": [B, H, N, D],
        "comfy_attention_sage_source": attn_sage_src,
    }

    with torch.inference_mode():
        results["sdpa_default_hnd"] = bench_op(
            "PyTorch SDPA (HND)",
            lambda: F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, is_causal=False),
        )
        results["comfy_attention_pytorch"] = bench_op(
            "Comfy attention_pytorch",
            lambda: comfy_attn_mod.attention_pytorch(q_hnd, k_hnd, v_hnd, heads=H, skip_reshape=True),
        )
        results["comfy_attention_sage"] = bench_op(
            "Comfy attention_sage",
            lambda: comfy_attn_mod.attention_sage(q_hnd, k_hnd, v_hnd, heads=H, skip_reshape=True),
            check_sage=True,
        )
        results["raw_sageattn_hnd"] = bench_op(
            "Raw sageattn (HND)",
            lambda: sageattn_fn(q_hnd, k_hnd, v_hnd, is_causal=False, tensor_layout="HND"),
            check_sage=True,
        )
        results["raw_sageattn_nhd"] = bench_op(
            "Raw sageattn (NHD)",
            lambda: sageattn_fn(q_nhd, k_nhd, v_nhd, is_causal=False, tensor_layout="NHD"),
            check_sage=True,
        )
        try:
            from sageattention.triton.quant_per_block import per_block_int8
            results["sage_quant_per_block_int8"] = bench_op(
                "Sage INT8 Q/K Quant Only",
                lambda: per_block_int8(q_hnd, k_hnd, sm_scale=1.0 / (D ** 0.5), tensor_layout="HND"),
                compute_tflops=False,
            )
        except Exception as e:
            results["sage_quant_per_block_int8"] = {"error": str(e)}

        # Measure numeric agreement of raw_sageattn_hnd vs sdpa_default_hnd on a single layer
        out_sdpa = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, is_causal=False)
        out_sage = sageattn_fn(q_hnd, k_hnd, v_hnd, is_causal=False, tensor_layout="HND")
        cos_sim = float(
            F.cosine_similarity(out_sdpa.float().reshape(1, -1), out_sage.float().reshape(1, -1)).item()
        )
        rel_l1 = float(
            (torch.abs(out_sdpa.float() - out_sage.float()).mean() / (torch.abs(out_sdpa.float()).mean() + 1e-8)).item()
        )
        results["single_layer_cosine_sim"] = round(cos_sim, 6)
        results["single_layer_rel_l1"] = round(rel_l1, 6)
        print(f"  • Single-layer Output Fidelity (Sage vs SDPA): CosineSim={cos_sim:.6f}, Rel-L1={rel_l1:.5f}")

    del q_hnd, k_hnd, v_hnd, q_nhd, k_nhd, v_nhd, out_sdpa, out_sage
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    bootstrap_environment()
    dl_stats = download_golden_5pack()

    sys.path.insert(0, str(COMFY_ROOT))
    import numpy as np
    import psutil
    import torch

    import comfy.ldm.minimax.model as minimax_model_mod
    import comfy.ldm.modules.attention as comfy_attn_mod
    import comfy.model_management as mm
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

    from sageattention import sageattn
    comfy_attn_mod.sageattn = sageattn
    comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True

    props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 90)
    print(
        f"🖥️ [Phase 3/5] Hardware & Pre-Encoding Setup on {props.name} "
        f"(SM {props.major}.{props.minor}, {props.total_memory / (1024**3):.1f} GiB VRAM)"
    )
    print("=" * 90)

    sage_detector = SageFallbackDetector()
    logging.getLogger().addHandler(sage_detector)

    microbench_results = run_attention_microbenchmark(comfy_attn_mod, sageattn, sage_detector)

    attn_call_counters = {"sageattn": 0, "sdpa_pytorch": 0}
    orig_attn_sage = comfy_attn_mod.attention_sage
    orig_attn_pytorch = comfy_attn_mod.attention_pytorch

    def counted_attn_sage(*args, **kwargs):
        attn_call_counters["sageattn"] += 1
        return orig_attn_sage(*args, **kwargs)

    def counted_attn_pytorch(*args, **kwargs):
        attn_call_counters["sdpa_pytorch"] += 1
        return orig_attn_pytorch(*args, **kwargs)

    def set_sage_attention(enabled: bool) -> None:
        target_fn = counted_attn_sage if enabled else counted_attn_pytorch
        comfy_attn_mod.optimized_attention = target_fn
        minimax_model_mod.optimized_attention = target_fn

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
    print(f"  ✅ FL2VA DiT + 8-Step Turbo LoRA loaded and resident in H100 VRAM in {dit_load_sec:.2f}s")

    print("\n" + "=" * 90)
    print("🚀 [Phase 4/5] Executing H100 2x2 SageAttention x EasyCache Ablation Matrix")
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
        attn_call_counters["sageattn"] = 0
        attn_call_counters["sdpa_pytorch"] = 0

        set_sage_attention(s_flag)
        cond_copy = clone_conditioning(base_cond)
        latent_copy = clone_av_latent(base_latent)
        noise_obj = RandomNoise.execute(seed)[0]

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

        frames_np = video_frames_t.detach().cpu().float().numpy()
        if frames_np.ndim == 5:
            frames_np = frames_np[0]
        waveform_np = audio_dict["waveform"].detach().cpu().float().numpy()
        if waveform_np.ndim == 3:
            waveform_np = waveform_np[0]
        sample_rate = int(audio_dict["sample_rate"])

        assert np.all(np.isfinite(frames_np)), f"Non-finite values in video frames during {pass_tag}!"
        assert np.all(np.isfinite(waveform_np)), f"Non-finite values in audio waveform during {pass_tag}!"

        del guider, out_latent, out_denoised, video_frames_t, audio_dict

        steps_skipped = last_easycache_stats["total_steps_skipped"] if c_flag else 0
        attn_backend_str = "sageattn" if s_flag else "sdpa_pytorch"
        sage_calls = attn_call_counters["sageattn"]
        sdpa_calls = attn_call_counters["sdpa_pytorch"]
        print(
            f"    [{pass_tag}] Denoise: {t_denoise:.2f}s | VideoVAE: {t_vae_video:.2f}s | "
            f"AudioVAE: {t_vae_audio:.2f}s | E2E: {t_e2e:.2f}s | PeakVRAM: {peak_vram_gib:.2f} GiB | "
            f"Skipped: {steps_skipped}/{steps} | Attn: {attn_backend_str} (sage_calls={sage_calls}, sdpa_calls={sdpa_calls})"
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
            "sage_attn_calls": int(sage_calls),
            "sdpa_attn_calls": int(sdpa_calls),
            "frames_np": frames_np,
            "waveform_np": waveform_np,
            "sample_rate": sample_rate,
        }

    easycache_threshold = 0.20
    easycache_start_pct = 0.15
    easycache_end_pct = 0.95

    raw_run_outputs: Dict[int, Dict[str, Any]] = {}
    all_frames_dict: Dict[int, Any] = {}

    print("\n  🔥 Running Warmup Passes ([0,0,0] SDPA and [1,0,0] SageAttention) to initialize CUDA/cuBLAS/Triton workspaces on H100...")
    warmup_sdpa = lora_model.clone()
    _ = execute_single_pass(warmup_sdpa, False, False, "H100-WARMUP-SDPA")
    del warmup_sdpa
    warmup_sage = lora_model.clone()
    _ = execute_single_pass(warmup_sage, True, False, "H100-WARMUP-SAGE")
    del warmup_sage

    for run_id, code, label, s_flag, c_flag in H100_ABLATION_MATRIX:
        print(f"\n▶️ H100 Run {run_id} {code} — {label} (Sage={s_flag}, EasyCache={c_flag})")
        work_model = lora_model.clone()
        if c_flag:
            work_model = EasyCacheNode.execute(
                work_model,
                reuse_threshold=easycache_threshold,
                start_percent=easycache_start_pct,
                end_percent=easycache_end_pct,
                verbose=True,
            )[0]

        pass_out = execute_single_pass(work_model, s_flag, c_flag, label)
        mp4_path = OUTPUT_DIR / f"h100_run{run_id}_{label}.mp4"
        save_video_with_audio(
            pass_out["frames_np"], pass_out["waveform_np"], pass_out["sample_rate"], mp4_path, fps=24
        )

        all_frames_dict[run_id] = pass_out["frames_np"]
        raw_run_outputs[run_id] = {
            "run_id": run_id,
            "code": code,
            "label": label,
            "S_sage": s_flag,
            "C_cache": c_flag,
            "denoise_sec": pass_out["denoise_sec"],
            "vae_video_sec": pass_out["vae_video_sec"],
            "vae_audio_sec": pass_out["vae_audio_sec"],
            "e2e_sec": pass_out["e2e_sec"],
            "peak_vram_gib": pass_out["peak_vram_gib"],
            "steps_skipped": pass_out["steps_skipped"],
            "output_change_rates": pass_out["output_change_rates"],
            "approx_output_change_rates": pass_out["approx_output_change_rates"],
            "attention_backend": pass_out["attention_backend"],
            "sage_attn_calls": pass_out["sage_attn_calls"],
            "sdpa_attn_calls": pass_out["sdpa_attn_calls"],
            "waveform_np": pass_out["waveform_np"],
            "sample_rate": pass_out["sample_rate"],
        }

        del work_model
        gc.collect()

    del lora_model, base_model, video_vae, audio_vae, base_cond, base_latent
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("📊 [Phase 5/5] Computing 4-Dimensional Quality Scores on H100")
    print("=" * 90)

    evaluator = QualityEvaluator(device="cuda")
    gt_frames = all_frames_dict[1]
    gt_audio = raw_run_outputs[1]["waveform_np"]
    t_base = raw_run_outputs[1]["e2e_sec"]
    t_base_denoise = raw_run_outputs[1]["denoise_sec"]

    baseline_av_corr = None
    final_results: List[Dict[str, Any]] = []

    for run_id, _, _, _, _ in H100_ABLATION_MATRIX:
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

        r["e2e_speedup_vs_h100_base"] = float(t_base / r["e2e_sec"])
        r["denoise_speedup_vs_h100_base"] = float(t_base_denoise / r["denoise_sec"])
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
            f"  [H100 Run {run_id} {r['code']}] Denoise: {r['denoise_sec']:.2f}s ({r['denoise_speedup_vs_h100_base']:.2f}x) | "
            f"E2E: {r['e2e_sec']:.2f}s ({r['e2e_speedup_vs_h100_base']:.2f}x) | "
            f"Q_total={r['Q_total']:.2f} [S_ref={r['S_ref']:.1f}, S_vb={r['S_vbench']:.1f}, "
            f"S_fl2v={r['S_fl2v']:.1f}, S_aud={r['S_audio']:.1f}] | "
            f"PSNR={ref_m['psnr_mean']:.2f}dB | SSIM={ref_m['ssim_mean']:.4f} | LPIPS={ref_m['lpips_mean']:.4f}"
        )

    comfy_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(COMFY_ROOT), text=True
    ).strip()

    payload = {
        "hardware": {
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_gib": round(props.total_memory / (1024 ** 3), 2),
            "ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": importlib.metadata.version("triton"),
            "sageattention_version": importlib.metadata.version("sageattention"),
            "comfyui_commit": comfy_commit,
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
        },
        "microbenchmark": microbench_results,
        "runs": final_results,
    }

    json_path = OUTPUT_DIR / "h100_sage_ablation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"✅ H100 Supplemental SageAttention Ablation Completed! Saved to {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
