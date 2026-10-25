# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Extract actual reference conditions and compile explicit FL2VA batch manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from analyze_motion import font, probe
from PIL import Image, ImageDraw, ImageOps

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "outputs" / "reproduction"
SOURCE = EXP / "outputs" / "reference" / "wuy0uer_7664781155287239972.mp4"
SOURCE_SHA = "ffde3a774c44e3a0f1e5e58dafd32148c3966e4ee2ad76d3b122882ccac42208"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text())
    name = spec["round"]
    if not name.isidentifier() or not spec["shots"]:
        raise ValueError("Invalid round identifier or empty shot list")
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("Reference source hash mismatch")
    info = probe(SOURCE)
    if info["fps"] != 30 or info["declared_frames"] != 895:
        raise ValueError("Unexpected source frame count or FPS")
    out = ROOT / name
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new round name; existing round manifests are immutable")
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    needed = sorted({s[k] for s in spec["shots"] for k in ["first", "last"]})
    if not all(isinstance(n, int) and 0 <= n < 895 for n in needed):
        raise ValueError("Invalid reference frame index")
    frame_map, thumbs = {}, {}
    cap = cv2.VideoCapture(str(SOURCE))
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in needed:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 3:
                    raise ValueError(f"Black condition rejected at reference frame {count}")
                aligned = ImageOps.fit(image, (1344, 768), method=Image.Resampling.LANCZOS)
                target = assets / f"reference_f{count:06d}.png"
                aligned.save(target, optimize=True, compress_level=9)
                frame_map[str(count)] = {
                    "source_frame": count, "source_time_sec": round(count / 30, 6),
                    "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "condition_path": f"assets/{target.name}", "condition_sha256": digest(target),
                    "source_size": list(image.size), "condition_size": list(aligned.size),
                    "resize": "Pillow ImageOps.fit LANCZOS center cover; 16:9 source to 1344x768 canvas",
                }
                thumbs[count] = aligned.resize((320, 183))
            count += 1
    finally:
        cap.release()
    if count != 895 or len(frame_map) != len(needed):
        raise ValueError("Reference decode is incomplete")
    shots, jobs = [], []
    for shot in spec["shots"]:
        n = shot["num_frames"]
        if not isinstance(n, int) or n < 22 or (n - 5) % 17:
            raise ValueError(f"Invalid H3 temporal grid for {shot['id']}")
        end = n / 24
        prompt = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) "
            f"aligns with the {end:.2f}-second mark of the target video.\n\n"
            f"integrated_multimodal_description: [Shot 1] {shot['description']}\n"
            f"overall_soundscape: {shot['soundscape']}\nnon_diegetic_music: N/A"
        )
        shots.append({
            "shot_id": shot["id"], "title": shot["title"], "prompt": prompt,
            "num_frames": n, "first_frame": frame_map[str(shot["first"])]["condition_path"],
            "last_frame": frame_map[str(shot["last"])]["condition_path"],
        })
        for seed in shot["seeds"]:
            jobs.append({"job_id": f"job_{len(jobs) + 1:02d}_{shot['id']}_seed{seed}",
                         "shot_id": shot["id"], "seed": seed, "note": spec["hypothesis"]})
    manifest = {
        "project": f"exp06_sub03_{name}", "title": spec["title"],
        "batch_id": f"exp06_sub03_{name}", "resolution": [1344, 768],
        "fps": 24, "steps": 8, "cfg": 1.0, "num_frames": 124,
        "resize_mode": "cover", "storyboards": shots, "jobs": jobs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    provenance = {"reference_sha256": SOURCE_SHA, "decoded_frames": count,
                  "spec_sha256": digest(spec_path), "hypothesis": spec["hypothesis"],
                  "conditions": frame_map, "manifest_sha256": digest(out / "manifest.json")}
    (out / "conditions.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    sheet = Image.new("RGB", (1600, 50 + ((len(needed) + 4) // 5) * 215), "#101824")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), spec["title"], fill="white", font=font(22))
    for index, frame in enumerate(needed):
        x, y = index % 5 * 320, 50 + index // 5 * 215
        sheet.paste(thumbs[frame], (x, y))
        draw.text((x + 8, y + 186), f"f{frame:03d} · {frame / 30:.3f}s", fill="#94e0ed", font=font(18))
    sheet.save(out / "conditions.jpg")
    print(json.dumps({"round": name, "shots": len(shots), "jobs": len(jobs),
                      "manifest": str((out / "manifest.json").relative_to(EXP))}, ensure_ascii=False))
    return out / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    prepare(parser.parse_args().spec)
