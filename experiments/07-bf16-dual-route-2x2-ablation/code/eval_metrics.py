#!/usr/bin/env python3
"""
Experiment 07 4-Dimensional Orthogonal Quality Evaluator + Production Metrics
(`experiments/07-bf16-dual-route-2x2-ablation/code/eval_metrics.py`).

Strictly preserves the exact 4D metric formulas and weights from Experiments 02 & 03:
  Q_total = 0.35 * S_ref + 0.25 * S_vbench + 0.25 * S_fl2v + 0.15 * S_audio
Plus the production pipeline metrics (S_fl2v_pixel, prod_quality_score) from Experiments 04-06.
"""

import concurrent.futures
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import ndimage, signal


def _ssim_single_frame_fast(pair: Tuple[np.ndarray, np.ndarray]) -> float:
    img1, img2 = pair
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return float(np.mean(ssim_map))


def save_video_with_audio(
    frames_np: np.ndarray,
    waveform_np: np.ndarray,
    sample_rate: int,
    out_mp4: Path,
    fps: int = 24,
) -> None:
    """Encodes all 124 video frames and audio without `-shortest` truncation."""
    tmp_vid = out_mp4.with_suffix(".tmp_video.mp4")
    tmp_wav = out_mp4.with_suffix(".tmp_audio.wav")
    frames_u8 = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(
        str(tmp_vid), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16
    )
    for frame in frames_u8:
        writer.append_data(frame)
    writer.close()

    wav_data = waveform_np.T if waveform_np.ndim == 2 else waveform_np
    sf.write(str(tmp_wav), np.clip(wav_data, -1.0, 1.0), sample_rate)

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_vid), "-i", str(tmp_wav),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(out_mp4),
        ],
        check=True,
    )
    tmp_vid.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)


def compute_production_quality_metrics(
    frames_np: np.ndarray,
    waveform_np: np.ndarray,
    first_np: np.ndarray,
    last_np: np.ndarray,
) -> Dict[str, Any]:
    """Computes the fast production metrics (S_fl2v_pixel, prod_quality_score) from studio/api/remote_batch_engine.py."""
    first_ssim = round(_ssim_single_frame_fast((frames_np[0], first_np)), 4)
    last_ssim = round(_ssim_single_frame_fast((frames_np[-1], last_np)), 4)
    s_fl2v_pixel = round(100.0 * (0.40 * first_ssim + 0.60 * last_ssim), 2)

    frames_u8 = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
    frame_mae = np.mean(np.abs(frames_np[1:] - frames_np[:-1]), axis=(1, 2, 3))
    temporal_flickering_score = float(100.0 * (1.0 - np.mean(frame_mae)))
    dynamic_motion_px = float(np.mean(frame_mae) * 255.0)
    motion_smoothness = float(100.0 / (1.0 + np.mean(np.abs(np.diff(frame_mae))) / (np.mean(frame_mae) + 1e-4)))
    sample_idx = np.linspace(0, frames_u8.shape[0] - 1, min(9, frames_u8.shape[0]), dtype=int)
    sharpness_lap = float(
        np.mean([float(cv2.Laplacian(cv2.cvtColor(frames_u8[i], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()) for i in sample_idx])
    )
    sharpness_norm = min(100.0, 100.0 * (sharpness_lap / 260.0))

    L, T = waveform_np.shape[-1], frames_np.shape[0]
    spf = max(1, L // T)
    a_env = np.array(
        [np.sqrt(np.mean(waveform_np[:, t * spf : (t + 1) * spf] ** 2) + 1e-12) for t in range(T)], dtype=np.float64
    )
    m_env = np.concatenate([[frame_mae[0]], frame_mae]).astype(np.float64)
    a_s, m_s = ndimage.uniform_filter1d(a_env, size=3), ndimage.uniform_filter1d(m_env, size=3)
    av_corr = float(np.corrcoef(a_s, m_s)[0, 1]) if (np.std(a_s) > 1e-6 and np.std(m_s) > 1e-6) else 0.0
    audio_rms_db = float(20.0 * math.log10(math.sqrt(float(np.mean(waveform_np ** 2)) + 1e-12)))
    prod_quality_score = float(
        0.40 * s_fl2v_pixel + 0.25 * temporal_flickering_score + 0.20 * motion_smoothness + 0.15 * sharpness_norm
    )
    return {
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


class QualityEvaluator:
    """Executes the exact 4-Dimensional quality evaluation pipeline from Experiments 02 & 03."""

    def __init__(self, device: str = "cuda") -> None:
        import lpips
        import open_clip
        import pyiqa
        import torchvision.models.optical_flow as of_models

        self.device = device
        print("  📦 Initializing 4D QualityEvaluator models on GPU (LPIPS, DINOv2, CLIP, RAFT, MUSIQ)...")
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(device).eval()
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip_model = self.clip_model.to(device).eval()
        self.raft = of_models.raft_small(weights=of_models.Raft_Small_Weights.DEFAULT).to(device).eval()
        self.musiq = pyiqa.create_metric("musiq", device=device, as_loss=False)
        self._ssim_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def _dino_embed(self, imgs_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dinov2(x)
        return F.normalize(feats, dim=-1)

    @torch.inference_mode()
    def _clip_embed(self, imgs_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        feats = self.clip_model.encode_image(x)
        return F.normalize(feats.float(), dim=-1)

    @torch.inference_mode()
    def compute_ref_fidelity(
        self, frames: np.ndarray, ref_frames: np.ndarray, is_baseline: bool = False
    ) -> Dict[str, Any]:
        T = frames.shape[0]
        if is_baseline:
            return {
                "psnr_mean": 100.0,
                "ssim_mean": 1.0,
                "lpips_mean": 0.0,
                "S_ref": 100.0,
                "per_frame_psnr": [100.0] * T,
                "per_frame_ssim": [1.0] * T,
                "per_frame_lpips": [0.0] * T,
            }

        mse_per_frame = np.mean((frames.astype(np.float64) - ref_frames.astype(np.float64)) ** 2, axis=(1, 2, 3))
        psnr_list = [100.0 if m < 1e-10 else float(10.0 * math.log10(1.0 / m)) for m in mse_per_frame]
        ssim_list = list(self._ssim_pool.map(_ssim_single_frame_fast, [(frames[t], ref_frames[t]) for t in range(T)]))

        lpips_list: List[float] = []
        bs = 8
        for i in range(0, T, bs):
            b1 = torch.from_numpy(frames[i : i + bs]).permute(0, 3, 1, 2).to(self.device) * 2.0 - 1.0
            b2 = torch.from_numpy(ref_frames[i : i + bs]).permute(0, 3, 1, 2).to(self.device) * 2.0 - 1.0
            b1_half = F.interpolate(b1, scale_factor=0.5, mode="bilinear", align_corners=False)
            b2_half = F.interpolate(b2, scale_factor=0.5, mode="bilinear", align_corners=False)
            d = self.lpips_fn(b1_half, b2_half).view(-1).cpu().tolist()
            lpips_list.extend([float(v) for v in d])

        psnr_mean = float(np.mean(psnr_list))
        ssim_mean = float(np.mean(ssim_list))
        lpips_mean = float(np.mean(lpips_list))
        psnr_norm = min(100.0, max(0.0, (psnr_mean - 20.0) / 30.0 * 100.0))
        ssim_norm = min(100.0, max(0.0, ssim_mean * 100.0))
        lpips_norm = min(100.0, max(0.0, (1.0 - lpips_mean * 2.0) * 100.0))
        s_ref = 0.40 * psnr_norm + 0.35 * ssim_norm + 0.25 * lpips_norm

        return {
            "psnr_mean": round(psnr_mean, 4),
            "ssim_mean": round(ssim_mean, 4),
            "lpips_mean": round(lpips_mean, 5),
            "S_ref": round(s_ref, 2),
            "per_frame_psnr": [round(x, 3) for x in psnr_list],
            "per_frame_ssim": [round(x, 4) for x in ssim_list],
            "per_frame_lpips": [round(x, 5) for x in lpips_list],
        }

    @torch.inference_mode()
    def compute_vbench_temporal(self, frames: np.ndarray) -> Dict[str, Any]:
        T = frames.shape[0]
        idx_sub = np.linspace(0, T - 1, min(32, T), dtype=int)
        sub_frames = frames[idx_sub]

        dino_f = self._dino_embed(sub_frames)
        cos_adj_dino = (dino_f[:-1] * dino_f[1:]).sum(dim=-1)
        cos_anchor_dino = (dino_f[1:] * dino_f[0:1]).sum(dim=-1)
        subject_consistency = float((0.6 * cos_adj_dino.mean() + 0.4 * cos_anchor_dino.mean()).item() * 100.0)

        clip_f = self._clip_embed(sub_frames)
        cos_adj_clip = (clip_f[:-1] * clip_f[1:]).sum(dim=-1)
        cos_anchor_clip = (clip_f[1:] * clip_f[0:1]).sum(dim=-1)
        background_consistency = float((0.6 * cos_adj_clip.mean() + 0.4 * cos_anchor_clip.mean()).item() * 100.0)

        l1_diff = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
        temporal_flickering = float((1.0 - float(np.mean(l1_diff))) * 100.0)

        idx_flow = np.linspace(0, T - 1, min(16, T), dtype=int)
        flow_frames = torch.from_numpy(frames[idx_flow]).permute(0, 3, 1, 2).to(self.device)
        flow_frames = F.interpolate(flow_frames, size=(384, 672), mode="bilinear", align_corners=False) * 2.0 - 1.0
        flows = []
        for i in range(len(idx_flow) - 1):
            f_pred = self.raft(flow_frames[i : i + 1], flow_frames[i + 1 : i + 2])[-1]
            flows.append(f_pred)
        flows_t = torch.cat(flows, dim=0)
        flow_mag = torch.norm(flows_t, dim=1).mean(dim=(1, 2))
        dynamic_degree = float(flow_mag.mean().item())
        flow_jerk = torch.norm(flows_t[1:] - flows_t[:-1], dim=1).mean()
        motion_smoothness = float((100.0 / (1.0 + (flow_jerk / (flow_mag.mean() + 1.0)).item())))

        idx_aes = np.linspace(0, T - 1, min(8, T), dtype=int)
        aes_batch = torch.from_numpy(frames[idx_aes]).permute(0, 3, 1, 2).to(self.device)
        aes_batch = F.interpolate(aes_batch, size=(384, 672), mode="bilinear", align_corners=False)
        aesthetic_score = float(self.musiq(aes_batch).mean().item())

        s_vbench = (
            0.25 * subject_consistency
            + 0.20 * background_consistency
            + 0.20 * temporal_flickering
            + 0.20 * motion_smoothness
            + 0.15 * aesthetic_score
        )
        return {
            "subject_consistency": round(subject_consistency, 2),
            "background_consistency": round(background_consistency, 2),
            "temporal_flickering": round(temporal_flickering, 2),
            "motion_smoothness": round(motion_smoothness, 2),
            "dynamic_degree_px": round(dynamic_degree, 2),
            "aesthetic_quality": round(aesthetic_score, 2),
            "S_vbench": round(s_vbench, 2),
        }

    @torch.inference_mode()
    def compute_fl2v_alignment(
        self, frames: np.ndarray, first_ref: np.ndarray, last_ref: np.ndarray
    ) -> Dict[str, Any]:
        gen_first, gen_last = frames[0], frames[-1]
        ssim_first = _ssim_single_frame_fast((gen_first, first_ref))
        ssim_last = _ssim_single_frame_fast((gen_last, last_ref))

        pack = np.stack([gen_first, first_ref, gen_last, last_ref], axis=0)
        dino_e = self._dino_embed(pack)
        clip_e = self._clip_embed(pack)

        dino_first = float((dino_e[0] * dino_e[1]).sum().item())
        dino_last = float((dino_e[2] * dino_e[3]).sum().item())
        clip_first = float((clip_e[0] * clip_e[1]).sum().item())
        clip_last = float((clip_e[2] * clip_e[3]).sum().item())

        score_first = (0.4 * dino_first + 0.3 * clip_first + 0.3 * ssim_first) * 100.0
        score_last = (0.4 * dino_last + 0.3 * clip_last + 0.3 * ssim_last) * 100.0
        s_fl2v = 0.40 * score_first + 0.60 * score_last

        return {
            "first_frame_ssim": round(ssim_first, 4),
            "first_frame_dino": round(dino_first, 4),
            "first_frame_clip": round(clip_first, 4),
            "first_frame_score": round(score_first, 2),
            "last_frame_ssim": round(ssim_last, 4),
            "last_frame_dino": round(dino_last, 4),
            "last_frame_clip": round(clip_last, 4),
            "last_frame_score": round(score_last, 2),
            "S_fl2v": round(s_fl2v, 2),
        }

    def compute_audio_metrics(
        self,
        waveform: np.ndarray,
        ref_waveform: np.ndarray,
        frames: np.ndarray,
        sr: int,
        is_baseline: bool = False,
    ) -> Dict[str, Any]:
        w = np.mean(waveform, axis=0) if waveform.ndim == 2 else waveform
        rw = np.mean(ref_waveform, axis=0) if ref_waveform.ndim == 2 else ref_waveform
        min_len = min(len(w), len(rw))
        w, rw = w[:min_len], rw[:min_len]
        rms_db = float(20.0 * math.log10(float(np.sqrt(np.mean(w ** 2)) + 1e-8)))

        if is_baseline:
            stft_log_sim = 100.0
            stft_linear_sim = 100.0
            mel_corr = 1.0
        else:
            _, _, Z_w = signal.stft(w, fs=sr, nperseg=1024, noverlap=768)
            _, _, Z_rw = signal.stft(rw, fs=sr, nperseg=1024, noverlap=768)
            mag_w_log = np.log1p(np.abs(Z_w))
            mag_rw_log = np.log1p(np.abs(Z_rw))
            stft_l1_rel = float(np.mean(np.abs(mag_w_log - mag_rw_log)) / (np.mean(mag_rw_log) + 1e-6))
            stft_log_sim = max(0.0, min(100.0, (1.0 - stft_l1_rel) * 100.0))

            mag_w_lin = np.abs(Z_w) + 1e-7
            mag_rw_lin = np.abs(Z_rw) + 1e-7
            l2_rel_lin = float(np.linalg.norm(mag_w_lin - mag_rw_lin) / (np.linalg.norm(mag_rw_lin) + 1e-7))
            stft_linear_sim = max(0.0, min(100.0, (1.0 - 0.5 * l2_rel_lin) * 100.0))

            v1, v2 = mag_w_log.flatten(), mag_rw_log.flatten()
            mel_corr = (
                float(np.corrcoef(v1, v2)[0, 1]) if (np.std(v1) > 1e-8 and np.std(v2) > 1e-8) else 0.0
            )

        T = frames.shape[0]
        samples_per_frame = max(1, len(w) // T)
        audio_env = np.array(
            [np.sqrt(np.mean(w[t * samples_per_frame : (t + 1) * samples_per_frame] ** 2) + 1e-8) for t in range(T)]
        )
        motion_env = np.zeros(T, dtype=np.float64)
        motion_env[1:] = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
        motion_env[0] = motion_env[1]
        a_norm = (audio_env - np.mean(audio_env)) / (np.std(audio_env) + 1e-6)
        m_norm = (motion_env - np.mean(motion_env)) / (np.std(motion_env) + 1e-6)
        av_sync = max(0.0, min(100.0, (float(np.mean(a_norm * m_norm)) + 1.0) * 50.0))
        mel_score = max(0.0, min(100.0, mel_corr * 100.0))

        s_audio = 0.45 * stft_log_sim + 0.35 * mel_score + 0.20 * av_sync
        s_audio_linear_supp = 0.50 * stft_linear_sim + 0.30 * mel_score + 0.20 * av_sync

        return {
            "audio_rms_db": round(rms_db, 2),
            "stft_log_similarity": round(stft_log_sim, 2),
            "stft_linear_similarity_supp": round(stft_linear_sim, 2),
            "spectral_corr": round(mel_corr, 4),
            "av_sync_score": round(av_sync, 2),
            "S_audio": round(s_audio, 2),
            "S_audio_linear_supp": round(s_audio_linear_supp, 2),
        }

    def evaluate_run_dual_ref(
        self,
        frames: np.ndarray,
        waveform: np.ndarray,
        sr: int,
        route_ref_frames: np.ndarray,
        route_ref_waveform: np.ndarray,
        run01_ref_frames: np.ndarray,
        run01_ref_waveform: np.ndarray,
        first_ref: np.ndarray,
        last_ref: np.ndarray,
        is_route_baseline: bool = False,
        is_run01_baseline: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates 4D quality relative to both the route's own 20-step Quality baseline AND Run 01 globally."""
        ref_route = self.compute_ref_fidelity(frames, route_ref_frames, is_baseline=is_route_baseline)
        ref_run01 = self.compute_ref_fidelity(frames, run01_ref_frames, is_baseline=is_run01_baseline)
        vb_m = self.compute_vbench_temporal(frames)
        fl_m = self.compute_fl2v_alignment(frames, first_ref, last_ref)
        au_route = self.compute_audio_metrics(waveform, route_ref_waveform, frames, sr, is_baseline=is_route_baseline)
        au_run01 = self.compute_audio_metrics(waveform, run01_ref_waveform, frames, sr, is_baseline=is_run01_baseline)
        prod_m = compute_production_quality_metrics(frames, waveform, first_ref, last_ref)

        q_total = (
            0.35 * ref_route["S_ref"]
            + 0.25 * vb_m["S_vbench"]
            + 0.25 * fl_m["S_fl2v"]
            + 0.15 * au_route["S_audio"]
        )
        q_total_linear_supp = (
            0.35 * ref_route["S_ref"]
            + 0.25 * vb_m["S_vbench"]
            + 0.25 * fl_m["S_fl2v"]
            + 0.15 * au_route["S_audio_linear_supp"]
        )
        q_total_vs_run01 = (
            0.35 * ref_run01["S_ref"]
            + 0.25 * vb_m["S_vbench"]
            + 0.25 * fl_m["S_fl2v"]
            + 0.15 * au_run01["S_audio"]
        )
        q_total_linear_supp_vs_run01 = (
            0.35 * ref_run01["S_ref"]
            + 0.25 * vb_m["S_vbench"]
            + 0.25 * fl_m["S_fl2v"]
            + 0.15 * au_run01["S_audio_linear_supp"]
        )
        return {
            "ref_fidelity": ref_route,
            "ref_fidelity_vs_run01": {
                "psnr_mean": ref_run01["psnr_mean"],
                "ssim_mean": ref_run01["ssim_mean"],
                "lpips_mean": ref_run01["lpips_mean"],
                "S_ref_vs_run01": ref_run01["S_ref"],
            },
            "vbench_temporal": vb_m,
            "fl2v_alignment": fl_m,
            "audio_quality": au_route,
            "audio_quality_vs_run01": {
                "stft_log_similarity": au_run01["stft_log_similarity"],
                "stft_linear_similarity_supp": au_run01["stft_linear_similarity_supp"],
                "spectral_corr": au_run01["spectral_corr"],
                "S_audio_vs_run01": au_run01["S_audio"],
                "S_audio_linear_supp_vs_run01": au_run01["S_audio_linear_supp"],
            },
            "production_metrics": prod_m,
            "Q_total": round(q_total, 2),
            "Q_total_linear_supp": round(q_total_linear_supp, 2),
            "Q_total_vs_run01": round(q_total_vs_run01, 2),
            "Q_total_linear_supp_vs_run01": round(q_total_linear_supp_vs_run01, 2),
        }
