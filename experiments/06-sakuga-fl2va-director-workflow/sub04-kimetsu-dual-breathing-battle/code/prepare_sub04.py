# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Validate and align 1344x768 keyframes and compile FL2VA / I2VA batch manifests using h3-prompt-writing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

EXP = Path(__file__).resolve().parents[1]
RAW_DIR = EXP / "outputs" / "keyframes_raw"
ALIGNED_DIR = EXP / "outputs" / "keyframes_aligned"
ROUNDS_DIR = EXP / "outputs" / "rounds"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def format_h3_prompt(mode: str, num_frames: int, description: str, soundscape: str, music: str = "N/A") -> str:
    """Format prompt strictly following .agents/skills/h3-prompt-writing/references/base-en.txt."""
    end_sec = num_frames / 24.0
    if mode == "fl2va":
        header = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) "
            f"aligns with the {end_sec:.2f}-second mark of the target video.\n\n"
        )
    elif mode == "i2va":
        header = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        )
    elif mode == "t2va":
        header = ""
    else:
        raise ValueError(f"Unsupported H3 prompt mode: {mode}")
    return (
        f"{header}"
        f"integrated_multimodal_description: [Shot 1] {description.strip()}\n\n"
        f"overall_soundscape: {soundscape.strip()}\n\n"
        f"non_diegetic_music: {music.strip()}"
    )


def prepare(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text())
    name = spec["round"]
    if not name.isidentifier() or not spec.get("shots"):
        raise ValueError("Invalid round identifier or empty shot list")
    out = ROUNDS_DIR / name
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Round '{name}' manifest already exists; use a new round name")
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    needed_keys: list[str] = []
    for shot in spec["shots"]:
        for key in ("first", "last"):
            val = shot.get(key, "")
            if val and val not in needed_keys:
                needed_keys.append(val)

    frame_map: dict[str, dict] = {}
    thumbs: dict[str, Image.Image] = {}
    for kf_name in needed_keys:
        raw_path = RAW_DIR / f"{kf_name}.png"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw keyframe: {raw_path}")
        image = Image.open(raw_path).convert("RGB")
        gray_mean = float(np.asarray(image.convert("L"), dtype=np.float32).mean())
        if gray_mean < 3.0:
            raise ValueError(f"Keyframe {kf_name} is near-black (mean={gray_mean:.2f})")
        aligned = ImageOps.fit(image, (1344, 768), method=Image.Resampling.LANCZOS)
        aligned_global = ALIGNED_DIR / f"{kf_name}.png"
        aligned.save(aligned_global, optimize=True, compress_level=9)
        target = assets / f"{kf_name}.png"
        aligned.save(target, optimize=True, compress_level=9)
        frame_map[kf_name] = {
            "keyframe_id": kf_name,
            "raw_path": str(raw_path.relative_to(EXP)),
            "raw_sha256": digest(raw_path),
            "condition_path": f"assets/{target.name}",
            "condition_sha256": digest(target),
            "raw_size": list(image.size),
            "condition_size": list(aligned.size),
            "mean_luma": round(gray_mean, 2),
        }
        thumbs[kf_name] = aligned.resize((320, 183), Image.Resampling.LANCZOS)

    shots, jobs = [], []
    for shot in spec["shots"]:
        n = int(shot["num_frames"])
        if n < 22 or (n - 5) % 17 != 0:
            raise ValueError(f"Invalid H3 17k+5 frame count {n} for {shot['id']}")
        mode = shot.get("mode", "fl2va").lower()
        first_kf = shot.get("first", "")
        last_kf = shot.get("last", "")
        if mode == "fl2va" and (not first_kf or not last_kf):
            raise ValueError(f"FL2VA shot {shot['id']} requires both 'first' and 'last' keyframes")
        if mode == "i2va" and not first_kf:
            raise ValueError(f"I2VA shot {shot['id']} requires 'first' keyframe")
        prompt = format_h3_prompt(
            mode=mode,
            num_frames=n,
            description=shot["description"],
            soundscape=shot["soundscape"],
            music=shot.get("music", "N/A"),
        )
        shots.append(
            {
                "shot_id": shot["id"],
                "title": shot["title"],
                "prompt": prompt,
                "num_frames": n,
                "first_frame": frame_map[first_kf]["condition_path"] if first_kf else "",
                "last_frame": frame_map[last_kf]["condition_path"] if (mode == "fl2va" and last_kf) else "",
            }
        )
        for seed in shot["seeds"]:
            jobs.append(
                {
                    "job_id": f"job_{len(jobs) + 1:02d}_{shot['id']}_seed{seed}",
                    "shot_id": shot["id"],
                    "seed": int(seed),
                    "note": f"{mode.upper()} {n}f | {spec['hypothesis']}",
                }
            )

    manifest = {
        "project": f"exp06_sub04_{name}",
        "title": spec["title"],
        "batch_id": f"exp06_sub04_{name}",
        "resolution": [1344, 768],
        "fps": 24,
        "steps": 8,
        "cfg": 1.0,
        "num_frames": 39,
        "resize_mode": "cover",
        "storyboards": shots,
        "jobs": jobs,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "round": name,
        "spec_sha256": digest(spec_path),
        "hypothesis": spec["hypothesis"],
        "conditions": frame_map,
        "manifest_sha256": digest(manifest_path),
    }
    (out / "conditions.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")

    cols = 5
    rows = max(1, (len(needed_keys) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * 320, 50 + rows * 215), "#101824")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), spec["title"], fill="white", font=load_font(22))
    for idx, kf_name in enumerate(needed_keys):
        x, y = (idx % cols) * 320, 50 + (idx // cols) * 215
        sheet.paste(thumbs[kf_name], (x, y))
        draw.text(
            (x + 8, y + 186),
            f"{kf_name} (luma={frame_map[kf_name]['mean_luma']})",
            fill="#94e0ed",
            font=load_font(16),
        )
    sheet.save(out / "conditions.jpg", quality=92)
    print(
        json.dumps(
            {
                "round": name,
                "shots": len(shots),
                "jobs": len(jobs),
                "manifest": str(manifest_path.relative_to(EXP)),
            },
            ensure_ascii=False,
        )
    )
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    prepare(parser.parse_args().spec)
