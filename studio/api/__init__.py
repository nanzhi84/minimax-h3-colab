"""
Studio API package for MiniMax H3-Base-FL2VA on Google Colab Blackwell G4.
"""

from .manifest_schema import (
    ASPECT_RATIO_PRESETS,
    DURATION_PRESETS_SEC,
    LOCKED_RUN12_CONFIG,
     BatchManifest,
    RenderJob,
    StoryboardShot,
    compute_conditioning_hash,
    estimate_tokens_and_denoise_sec,
    resolve_dimensions,
    snap_frames_17k_plus_5,
)

__all__ = [
    "ASPECT_RATIO_PRESETS",
    "DURATION_PRESETS_SEC",
    "LOCKED_RUN12_CONFIG",
    "BatchManifest",
    "RenderJob",
    "StoryboardShot",
    "compute_conditioning_hash",
    "estimate_tokens_and_denoise_sec",
    "resolve_dimensions",
    "snap_frames_17k_plus_5",
]
