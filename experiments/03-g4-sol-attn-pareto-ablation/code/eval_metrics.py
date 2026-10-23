#!/usr/bin/env python3
"""
4-Dimensional Benchmark Quality Evaluation System (S_ref, S_vbench, S_fl2v, S_audio),
Pareto Frontier Solver, and Lossless-Frame-Count MP4 Muxing for Experiment 03.
"""

import concurrent.futures
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QualityEvaluator:
    """
    Exact 4-Dimensional Benchmark Quality Evaluation System from Experiment 02 (run_ablation.py):
      Q_total = 0.35 * S_ref + 0.25 * S_vbench + 0.25 * S_fl2v + 0.15 * S_audio
    Also computes the supplemental linear-STFT audio formulation (S_audio_linear_supp / Q_total_linear_supp)
    from Experiment 02's run_h100_sage_ablation.py for cross-script comparability.
    """

    def __init__(self, device: str = "cuda"):
        import lpips
        import pyiqa
        import torch
        import torchvision.models.optical_flow as tv_flow
        from transformers import AutoModel, CLIPVisionModelWithProjection

        self.device = torch.device(device)
        print(
            "  🔬 Loading LPIPS (AlexNet), DINOv2 (facebook/dinov2-small), "
            "CLIP (openai/clip-vit-base-patch32), RAFT, and pyiqa MUSIQ..."
        )
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        raft_weights = tv_flow.Raft_Small_Weights.DEFAULT
        self.raft_model = tv_flow.raft_small(weights=raft_weights, progress=False).to(self.device).eval()

        # Fail fast if pyiqa MUSIQ cannot be initialized rather than falling back to a constant score
        self.musiq_metric = pyiqa.create_metric("musiq", device=self.device)
        if self.musiq_metric is None:
            raise RuntimeError("Failed to initialize pyiqa MUSIQ metric.")

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self._cached_anchors: Optional[Tuple[Any, Any]] = None
        self._ssim_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def close(self) -> None:
        self._ssim_pool.shutdown(wait=True)

    def extract_dino_and_clip_features(self, frames_np: Any, batch_size: int = 16) -> Tuple[Any, Any]:
        import torch
        import torch.nn.functional as F

        dino_list, clip_list = [], []
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
                "psnr_per_frame": [100.0] * T,
                "ssim_per_frame": [1.0] * T,
                "lpips_per_frame": [0.0] * T,
                "S_ref": 100.0,
            }

        def eval_frame_pair(t: int) -> Tuple[float, float]:
            f_test, f_gt = frames_test[t], frames_gt[t]
            mse = float(np.mean((f_test - f_gt) ** 2))
            psnr = 100.0 if mse < 1e-10 else float(10.0 * math.log10(1.0 / mse))
            ssim = self.compute_ssim_frame(f_test, f_gt)
            return psnr, ssim

        pairs = list(self._ssim_pool.map(eval_frame_pair, range(T)))
        psnr_list = [p[0] for p in pairs]
        ssim_list = [p[1] for p in pairs]

        lpips_list = []
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
        import numpy as np
        import torch
        import torch.nn.functional as F

        T = frames_test.shape[0]
        dino_feats, clip_feats = self.extract_dino_and_clip_features(frames_test)
        if self._cached_anchors is None:
            self._cached_anchors = self.extract_dino_and_clip_features(
                np.stack([first_anchor_np, last_anchor_np], axis=0)
            )
        anc_dino, anc_clip = self._cached_anchors

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
        baseline_av_corr: Optional[float] = None,
        is_baseline: bool = False,
        sample_rate: int = 32000,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Computes both:
          1. Primary S_audio (Experiment 02 run_ablation.py):
             - STFT-SNR on log10(|Z| + 1e-7), hop = n_fft / 4, computed per-channel & per-resolution (6 dB values averaged), normalized via snr_db / 30.0
             - Log-Mel Cosine on 1024-FFT per-channel then averaged across channels
             - Weights: 0.40 * cos + 0.35 * snr_norm + 0.25 * av_sync_retention
          2. Supplemental S_audio_linear_supp (Experiment 02 run_h100_sage_ablation.py):
             - STFT-SNR on linear |Z|, SciPy default hop = n_fft / 2, channel-pooled signal/error power before 10*log10 (3 dB values averaged), normalized via (snr_lin_db + 5.0) / 35.0
             - Log-Mel Cosine on 2048-FFT channel-averaged power spectrum
             - Weights: 0.45 * cos_supp + 0.35 * snr_lin_norm + 0.20 * av_sync_retention
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
                "stft_snr_linear_db": 100.0,
                "log_mel_cosine_sim": 1.0,
                "log_mel_cosine_sim_linear_supp": 1.0,
                "rms_diff_db": 0.0,
                "av_sync_corr": av_corr,
                "av_sync_retention": 1.0,
                "S_audio": 100.0,
                "S_audio_linear_supp": 100.0,
            }, baseline_av_corr

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

        linear_snrs = []
        for nperseg in (512, 1024, 2048):
            _, _, Z_t = signal.stft(a_test, fs=sample_rate, nperseg=nperseg)
            _, _, Z_g = signal.stft(a_gt, fs=sample_rate, nperseg=nperseg)
            mag_t, mag_g = np.abs(Z_t), np.abs(Z_g)
            err_power = np.mean((mag_t - mag_g) ** 2) + 1e-12
            sig_power = np.mean(mag_g ** 2) + 1e-12
            linear_snrs.append(float(10.0 * math.log10(sig_power / err_power)))

        n_fft_supp = 2048
        _, _, Z_mel_t = signal.stft(a_test, fs=sample_rate, nperseg=n_fft_supp, noverlap=n_fft_supp - 512)
        _, _, Z_mel_g = signal.stft(a_gt, fs=sample_rate, nperseg=n_fft_supp, noverlap=n_fft_supp - 512)
        pow_supp_t = np.mean(np.abs(Z_mel_t) ** 2, axis=0)
        pow_supp_g = np.mean(np.abs(Z_mel_g) ** 2, axis=0)
        fbank_supp = cls._build_mel_filterbank(sr=sample_rate, n_fft=n_fft_supp, n_mels=80)
        mel_spec_t = np.log10(np.matmul(fbank_supp, pow_supp_t) + 1e-8)
        mel_spec_g = np.log10(np.matmul(fbank_supp, pow_supp_g) + 1e-8)
        vec_t, vec_g = mel_spec_t.reshape(-1), mel_spec_g.reshape(-1)
        log_mel_cos_supp = float(np.dot(vec_t, vec_g) / (np.linalg.norm(vec_t) * np.linalg.norm(vec_g) + 1e-12))

        mel_fbank = cls._build_mel_filterbank(sr=sample_rate, n_fft=1024, n_mels=80)
        mel_cos_list = []
        for ch in range(min(2, a_test.shape[0])):
            _, _, Z_t = signal.stft(a_test[ch], fs=sample_rate, nperseg=1024, noverlap=1024 - 256)
            _, _, Z_g = signal.stft(a_gt[ch], fs=sample_rate, nperseg=1024, noverlap=1024 - 256)
            pow_t, pow_g = np.abs(Z_t) ** 2, np.abs(Z_g) ** 2
            log_mel_t = np.log10(np.matmul(mel_fbank, pow_t) + 1e-7)
            log_mel_g = np.log10(np.matmul(mel_fbank, pow_g) + 1e-7)
            v1, v2 = log_mel_t.reshape(-1), log_mel_g.reshape(-1)
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            mel_cos_list.append(cos)

        stft_snr_db = float(np.mean(snr_list))
        stft_snr_linear_db = float(np.mean(linear_snrs))
        log_mel_cos = float(np.mean(mel_cos_list))

        rms_t = 20.0 * math.log10(math.sqrt(float(np.mean(a_test ** 2)) + 1e-12))
        rms_g = 20.0 * math.log10(math.sqrt(float(np.mean(a_gt ** 2)) + 1e-12))
        rms_diff_db = float(abs(rms_t - rms_g))

        snr_norm = min(1.0, max(0.0, stft_snr_db / 30.0))
        cos_norm = min(1.0, max(0.0, log_mel_cos))
        s_audio = 100.0 * (0.40 * cos_norm + 0.35 * snr_norm + 0.25 * av_sync_retention)

        snr_lin_norm = min(1.0, max(0.0, (stft_snr_linear_db + 5.0) / 35.0))
        mel_supp_norm = min(1.0, max(0.0, log_mel_cos_supp))
        s_audio_linear_supp = 100.0 * (0.45 * mel_supp_norm + 0.35 * snr_lin_norm + 0.20 * av_sync_retention)

        return {
            "stft_snr_db": stft_snr_db,
            "stft_snr_linear_db": stft_snr_linear_db,
            "log_mel_cosine_sim": log_mel_cos,
            "log_mel_cosine_sim_linear_supp": round(log_mel_cos_supp, 6),
            "rms_diff_db": rms_diff_db,
            "av_sync_corr": av_corr,
            "av_sync_retention": av_sync_retention,
            "S_audio": float(s_audio),
            "S_audio_linear_supp": float(s_audio_linear_supp),
        }, baseline_av_corr


def compute_pareto_frontier(points: List[Dict[str, Any]], time_key: str, quality_key: str = "Q_total") -> List[int]:
    pareto_ids = []
    for p in points:
        t_p, q_p = p[time_key], p[quality_key]
        dominated = False
        for other in points:
            if other["run_id"] == p["run_id"]:
                continue
            t_o, q_o = other[time_key], other[quality_key]
            if (t_o <= t_p and q_o >= q_p) and (t_o < t_p or q_o > q_p):
                dominated = True
                break
        if not dominated:
            pareto_ids.append(p["run_id"])

    pareto_points = [p for p in points if p["run_id"] in pareto_ids]
    pareto_points.sort(key=lambda x: x[time_key])
    return [p["run_id"] for p in pareto_points]


def save_video_with_audio(
    frames_np: Any,
    audio_np: Any,
    sample_rate: int,
    out_mp4: Path,
    fps: int = 24,
) -> float:
    """Encodes all 124 video frames and muxes AAC audio without -shortest truncation."""
    t0 = time.perf_counter()
    import imageio.v2 as imageio
    import numpy as np
    import soundfile as sf

    tmp_vid = out_mp4.with_suffix(".tmp_video.mp4")
    tmp_wav = out_mp4.with_suffix(".tmp_audio.wav")

    frames_u8 = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(
        str(tmp_vid), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16
    )
    for frame in frames_u8:
        writer.append_data(frame)
    writer.close()

    wav_data = np.clip(audio_np.T if audio_np.ndim == 2 else audio_np, -1.0, 1.0)
    sf.write(str(tmp_wav), wav_data, sample_rate)

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(tmp_vid),
            "-i", str(tmp_wav),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(out_mp4),
        ],
        check=True,
    )
    tmp_vid.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)
    return time.perf_counter() - t0
