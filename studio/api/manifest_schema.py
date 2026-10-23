#!/usr/bin/env python3
"""
Production Schema, Aspect Ratio / Resolution / Duration (17k+5) Resolver,
and Locked ADR-03 Run 12 [S8-FullStack-Pareto] Configuration for Studio API (`studio/api/manifest_schema.py`).
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Strictly locked to ADR-03 Default Production Configuration: Run 12 [S8-FullStack-Pareto]
LOCKED_RUN12_CONFIG: Dict[str, Any] = {
    "config_id": "Run 12 [S8-FullStack-Pareto]",
    "adr_ref": "ADR-03",
    "dense_backend": "sage",
    "sparse_mode": "ck_sol_direct",
    "start_percent": 0.20,
    "end_percent": 1.0,
    "tau": 1.3,
    "extra_tokens": 256,
    "dense_blocks": "",
    "sink_conditioning": "exact_kv_and_rows",
    "fused_mod": True,
    "baseline_resolution": [1344, 768],
    "baseline_frames": 124,
    "est_denoise_sec": 42.37,
    "est_vae_decode_sec": 4.35,
    "est_e2e_sec": 46.72,
    "est_marginal_cu": 0.0935,
    "desc": "前 2/8 步 SageAttention 稠密预热 + 后 6/8 步零拷贝直通 ck.sol_attn (tau=1.3, extra_tokens=256) + MiniMaxH3FusedModulation",
}

# All dimensions are strict multiples of 32 (required by MiniMax H3 Video VAE & 2x2 patchify)
ASPECT_RATIO_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "512p": {
        "16:9": (896, 512),
        "9:16": (512, 896),
        "1:1": (640, 640),
        "4:3": (768, 576),
        "3:4": (576, 768),
        "21:9": (1024, 448),
    },
    "768p": {
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "21:9": (1536, 640),
    },
    "1080p": {
        "16:9": (1920, 1088),
        "9:16": (1088, 1920),
        "1:1": (1408, 1408),
        "4:3": (1600, 1216),
        "3:4": (1216, 1600),
        "21:9": (2176, 928),
    },
}

# Standard duration presets mapped to MiniMax H3's native 24fps (17k + 5) temporal grid
DURATION_PRESETS_SEC: Dict[int, int] = {
    2: 39,   # k=2: 17*2 + 5 = 39 frames  (1.62s @ 24fps)
    3: 73,   # k=4: 17*4 + 5 = 73 frames  (3.04s @ 24fps)
    4: 90,   # k=5: 17*5 + 5 = 90 frames  (3.75s @ 24fps)
    5: 124,  # k=7: 17*7 + 5 = 124 frames (5.17s @ 24fps, ADR-01~05 Standard)
    6: 141,  # k=8: 17*8 + 5 = 141 frames (5.88s @ 24fps)
    8: 192,  # k=11: 17*11 + 5 = 192 frames (8.00s @ 24fps)
    10: 243, # k=14: 17*14 + 5 = 243 frames (10.12s @ 24fps)
}


def snap_to_multiple_of_32(val: int) -> int:
    return max(256, int(round(val / 32.0)) * 32)


def snap_frames_17k_plus_5(frames_or_sec: float, fps: int = 24, is_seconds: bool = False) -> int:
    """
    Snaps target duration (seconds) or frame count to MiniMax H3's native `17k + 5` frame grid.
    Example: 5s @ 24fps -> 124 frames (k=7).
    """
    if is_seconds:
        sec_int = int(round(frames_or_sec))
        if sec_int in DURATION_PRESETS_SEC and abs(frames_or_sec - sec_int) < 1e-3:
            return DURATION_PRESETS_SEC[sec_int]
        raw_frames = max(22, int(round(frames_or_sec * fps)))
    else:
        raw_frames = max(22, int(round(frames_or_sec)))
    k = max(1, int(round((raw_frames - 5) / 17.0)))
    return 17 * k + 5


def resolve_dimensions(
    aspect_ratio: str = "16:9",
    quality: str = "768p",
    explicit_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    if explicit_resolution is not None:
        return (
            snap_to_multiple_of_32(explicit_resolution[0]),
            snap_to_multiple_of_32(explicit_resolution[1]),
        )
    q_key = quality.lower().strip()
    if q_key in ("480p", "draft", "fast"):
        q_key = "512p"
    elif q_key in ("720p", "standard", "hd"):
        q_key = "768p"
    elif q_key in ("fhd", "ultra"):
        q_key = "1080p"
    if q_key not in ASPECT_RATIO_PRESETS:
        raise ValueError(f"Unsupported quality tier '{quality}'. Choose from: {list(ASPECT_RATIO_PRESETS.keys())}")
    ar_map = ASPECT_RATIO_PRESETS[q_key]
    if aspect_ratio not in ar_map:
        raise ValueError(f"Unsupported aspect_ratio '{aspect_ratio}'. Choose from: {list(ar_map.keys())}")
    return ar_map[aspect_ratio]


def estimate_tokens_and_denoise_sec(
    width: int,
    height: int,
    num_frames: int,
    steps: int = 8,
) -> Dict[str, Any]:
    """
    Estimates MiniMax H3 joint multimodal sequence length N and Blackwell G4 Run 12 latency.
    Baseline: 1344x768 x 124 frames -> 1008 spatial tokens/frame, N = 42,058 tokens,
    42.37s denoise + 4.35s VAE = 46.72s E2E.
    """
    spatial_tokens = (width // 32) * (height // 32)
    latent_t = ((num_frames - 5) // 17) * 5 + 2
    video_tokens = spatial_tokens * latent_t
    prefix_tokens = 4762
    total_tokens = prefix_tokens + video_tokens

    base_tokens = 42058.0
    token_ratio = total_tokens / base_tokens
    step_ratio = steps / 8.0
    # Run 12 uses hybrid dense (25%) + block-sparse Sol-Attn (75%), scaling ~ O(N^1.45)
    denoise_scale = (token_ratio ** 1.45) * step_ratio
    vae_scale = (width * height * num_frames) / (1344.0 * 768.0 * 124.0)

    est_denoise_sec = round(42.37 * denoise_scale, 2)
    est_vae_sec = round(4.35 * vae_scale, 2)
    est_e2e_sec = round(est_denoise_sec + est_vae_sec, 2)
    return {
        "spatial_tokens_per_frame": spatial_tokens,
        "video_tokens": video_tokens,
        "total_joint_tokens": total_tokens,
        "token_ratio_vs_baseline": round(token_ratio, 3),
        "est_denoise_sec": est_denoise_sec,
        "est_vae_decode_sec": est_vae_sec,
        "est_e2e_sec": est_e2e_sec,
    }


@dataclass
class StoryboardShot:
    shot_id: str
    title: str
    prompt: str
    negative_prompt: str = ""
    first_frame: str = ""
    last_frame: str = ""
    aspect_ratio: Optional[str] = None
    quality: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    duration_sec: Optional[float] = None
    num_frames: Optional[int] = None
    cond_hash: str = ""


@dataclass
class RenderJob:
    job_id: str
    shot_id: str
    seed: int
    note: str = ""


@dataclass
class BatchManifest:
    batch_id: str
    title: str
    project: str = "default"
    aspect_ratio: str = "16:9"
    quality: str = "768p"
    resolution: Tuple[int, int] = (1344, 768)
    duration_sec: float = 5.0
    num_frames: int = 124
    fps: int = 24
    steps: int = 8
    cfg: float = 1.0
    resize_mode: str = "cover"  # "cover" (center-crop to aspect) or "stretch"
    storyboards: List[StoryboardShot] = field(default_factory=list)
    jobs: List[RenderJob] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: Path) -> "BatchManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fps = int(raw.get("fps", 24))
        if fps != 24:
            raise ValueError(f"MiniMax H3 native audio-video sync requires fps=24 (got {fps})")
        steps = int(raw.get("steps", 8))
        if steps < 1 or steps > 50:
            raise ValueError(f"Invalid steps={steps}; must be in [1, 50] (default 8 for Turbo LoRA)")
        cfg = float(raw.get("cfg", 1.0))
        resize_mode = str(raw.get("resize_mode", "cover")).strip()
        if resize_mode not in ("cover", "stretch"):
            raise ValueError(f"Invalid resize_mode '{resize_mode}'; choose 'cover' or 'stretch'")

        project = str(raw.get("project") or raw.get("project_name") or "default").strip()
        if not project or project in (".", "..") or "/" in project or "\\" in project:
            raise ValueError(f"Invalid project name: '{project}'")

        aspect_ratio = raw.get("aspect_ratio", "16:9")
        quality = raw.get("quality", "768p")
        explicit_res = tuple(raw["resolution"]) if "resolution" in raw else None
        res = resolve_dimensions(aspect_ratio=aspect_ratio, quality=quality, explicit_resolution=explicit_res)

        if "num_frames" in raw:
            num_frames = snap_frames_17k_plus_5(int(raw["num_frames"]), fps=fps, is_seconds=False)
        elif "duration_sec" in raw:
            num_frames = snap_frames_17k_plus_5(float(raw["duration_sec"]), fps=fps, is_seconds=True)
        else:
            num_frames = 124
        duration_sec = round(num_frames / fps, 2)

        shots: List[StoryboardShot] = []
        seen_shot_ids = set()
        for s in raw.get("storyboards", []):
            sid = str(s["shot_id"]).strip()
            if not sid or sid in (".", "..") or "/" in sid or "\\" in sid:
                raise ValueError(f"Invalid shot_id: '{sid}'")
            if sid in seen_shot_ids:
                raise ValueError(f"Duplicate shot_id '{sid}' in manifest")
            seen_shot_ids.add(sid)
            s_res = tuple(s["resolution"]) if s.get("resolution") else None
            shots.append(
                StoryboardShot(
                    shot_id=sid,
                    title=s.get("title", sid),
                    prompt=s["prompt"],
                    negative_prompt=s.get("negative_prompt", ""),
                    first_frame=s.get("first_frame", ""),
                    last_frame=s.get("last_frame", ""),
                    aspect_ratio=s.get("aspect_ratio"),
                    quality=s.get("quality"),
                    resolution=s_res,
                    duration_sec=s.get("duration_sec"),
                    num_frames=s.get("num_frames"),
                )
            )
        if not shots:
            raise ValueError("Manifest must contain at least one storyboard shot")

        jobs: List[RenderJob] = []
        seen_job_ids = set()
        for j in raw.get("jobs", []):
            jid = str(j["job_id"]).strip()
            if not jid or jid in (".", "..") or "/" in jid or "\\" in jid:
                raise ValueError(f"Invalid job_id: '{jid}'")
            if jid in seen_job_ids:
                raise ValueError(f"Duplicate job_id '{jid}' in manifest")
            seen_job_ids.add(jid)
            jobs.append(
                RenderJob(
                    job_id=jid,
                    shot_id=j["shot_id"],
                    seed=int(j["seed"]),
                    note=j.get("note", ""),
                )
            )
        if not jobs:
            raise ValueError("Manifest must contain at least one render job")

        for job in jobs:
            if job.shot_id not in seen_shot_ids:
                raise ValueError(f"Job {job.job_id} references unknown shot_id '{job.shot_id}'")

        return cls(
            batch_id=raw.get("batch_id", f"{project}_batch"),
            title=raw.get("title", raw.get("batch_id", f"{project}_batch")),
            project=project,
            aspect_ratio=aspect_ratio,
            quality=quality,
            resolution=(int(res[0]), int(res[1])),
            duration_sec=duration_sec,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            resize_mode=resize_mode,
            storyboards=shots,
            jobs=jobs,
        )


def compute_conditioning_hash(
    prompt: str,
    negative_prompt: str,
    first_frame_bytes: bytes,
    last_frame_bytes: bytes,
    resolution: Tuple[int, int] = (1344, 768),
    num_frames: int = 124,
) -> str:
    """
    Computes a deterministic SHA-256 digest over all inputs to Qwen3-VL + Video VAE Encode.
    Jobs sharing the same `cond_hash` reuse the pre-encoded conditioning tensors in 0.00s.
    """
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(negative_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{resolution[0]}x{resolution[1]}x{num_frames}".encode("ascii"))
    h.update(b"\x00")
    h.update(hashlib.sha256(first_frame_bytes).digest())
    h.update(hashlib.sha256(last_frame_bytes).digest())
    return h.hexdigest()[:16]
