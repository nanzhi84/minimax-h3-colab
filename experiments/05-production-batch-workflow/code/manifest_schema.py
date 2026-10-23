#!/usr/bin/env python3
"""
Standard Schema, Locked ADR-03 Run 12 [S8-FullStack-Pareto] Configuration,
and Conditioning Deduplication Hash Calculator for the MiniMax H3-Base-FL2VA
Production Batch Workflow (Experiment 05).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "est_denoise_sec": 42.37,
    "est_vae_decode_sec": 4.35,
    "est_e2e_sec": 46.72,
    "est_marginal_cu": 0.0935,
    "desc": "前 2/8 步 SageAttention 稠密预热 + 后 6/8 步零拷贝直通 ck.sol_attn (tau=1.3, extra_tokens=256) + MiniMaxH3FusedModulation",
}


@dataclass
class StoryboardShot:
    shot_id: str
    title: str
    prompt: str
    negative_prompt: str = ""
    first_frame: str = ""
    last_frame: str = ""
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
    resolution: Tuple[int, int] = (1344, 768)
    num_frames: int = 124
    fps: int = 24
    steps: int = 8
    cfg: float = 1.0
    storyboards: List[StoryboardShot] = field(default_factory=list)
    jobs: List[RenderJob] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: Path) -> "BatchManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        shots = [StoryboardShot(**s) for s in raw.get("storyboards", [])]
        jobs = [
            RenderJob(
                job_id=j["job_id"],
                shot_id=j["shot_id"],
                seed=int(j["seed"]),
                note=j.get("note", ""),
            )
            for j in raw.get("jobs", [])
        ]
        shot_ids = {s.shot_id for s in shots}
        for job in jobs:
            if job.shot_id not in shot_ids:
                raise ValueError(f"Job {job.job_id} references unknown shot_id '{job.shot_id}'")
        res = tuple(raw.get("resolution", [1344, 768]))
        return cls(
            batch_id=raw["batch_id"],
            title=raw.get("title", raw["batch_id"]),
            resolution=(int(res[0]), int(res[1])),
            num_frames=int(raw.get("num_frames", 124)),
            fps=int(raw.get("fps", 24)),
            steps=int(raw.get("steps", 8)),
            cfg=float(raw.get("cfg", 1.0)),
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
