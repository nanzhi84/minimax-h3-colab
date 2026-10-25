# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Verify generated H3 candidate videos and render per-round contact sheets."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare_sub04 import load_font

EXP = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    data = json.loads(raw)
    streams = data.get("streams", [])
    v_stream = next(s for s in streams if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    num, den = (int(x) for x in v_stream["r_frame_rate"].split("/"))
    fps = round(num / den, 4)
    return {
        "width": int(v_stream["width"]),
        "height": int(v_stream["height"]),
        "fps": fps,
        "declared_frames": int(v_stream.get("nb_frames", 0)),
        "duration_sec": float(data["format"].get("duration", 0.0)),
        "size_bytes": int(data["format"].get("size", 0)),
        "has_audio": has_audio,
    }


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()

    round_dir = EXP / "outputs" / "rounds" / args.round
    manifest = json.loads((round_dir / "manifest.json").read_text())
    metrics = json.loads((args.batch_dir / "studio_batch_metrics.json").read_text())
    out_dir = EXP / "outputs" / "rounds" / f"{args.round}_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    shot_by_id = {s["shot_id"]: s for s in manifest["storyboards"]}
    verified_jobs = []
    rows_thumbs: list[tuple[str, list[Image.Image]]] = []

    for job_metric in metrics["jobs"]:
        jid = job_metric["job_id"]
        sid = job_metric["shot_id"]
        shot = shot_by_id[sid]
        mp4 = (args.batch_dir / job_metric["mp4_file"]).resolve()
        info = probe(mp4)
        frames = decode(mp4)
        expected_n = int(shot["num_frames"])
        if (
            len(frames) != expected_n
            or info["declared_frames"] != expected_n
            or info["fps"] != 24
            or (info["width"], info["height"]) != (1344, 768)
            or not info["has_audio"]
        ):
            raise ValueError(f"Job {jid} failed technical contract: {info}, decoded={len(frames)}")

        small = [cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
        diffs = [float(np.mean(np.abs(small[i] - small[i - 1]))) for i in range(1, len(small))]
        startup_mot = float(np.mean(diffs[:3])) if len(diffs) >= 3 else float(np.mean(diffs))
        mid_mot = float(np.mean(diffs[3:-3])) if len(diffs) > 6 else float(np.mean(diffs))
        tail_mot = float(np.mean(diffs[-3:])) if len(diffs) >= 3 else float(np.mean(diffs))

        sample_idx = np.linspace(0, len(frames) - 1, 6).round().astype(int).tolist()
        thumbs = [
            Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).resize(
                (256, 144), Image.Resampling.LANCZOS
            )
            for i in sample_idx
        ]
        mode_tag = "FL2VA" if shot.get("last_frame") else "I2VA"
        label = (
            f"{jid} [{mode_tag} {expected_n}f] | Q={job_metric.get('prod_quality_score', 0):.1f} "
            f"mid_mot={mid_mot:.1f} start={startup_mot:.1f} tail={tail_mot:.1f}"
        )
        rows_thumbs.append((label, thumbs))
        verified_jobs.append(
            {
                "job_id": jid,
                "shot_id": sid,
                "seed": int(job_metric["seed"]),
                "mode": mode_tag,
                "frames": len(frames),
                "fps": info["fps"],
                "resolution": [info["width"], info["height"]],
                "has_audio": info["has_audio"],
                "mp4_relpath": str(mp4.relative_to(EXP)),
                "sha256": digest(mp4),
                "prod_quality_score": float(job_metric.get("prod_quality_score", 0.0)),
                "S_fl2v_pixel": float(job_metric.get("S_fl2v_pixel", 0.0)),
                "step_mae_mean": round(float(np.mean(diffs)), 2),
                "startup_motion_0_3": round(startup_mot, 2),
                "core_motion_mid": round(mid_mot, 2),
                "tail_motion_last_3": round(tail_mot, 2),
            }
        )

    sheet_h = 44 + len(rows_thumbs) * 172
    sheet = Image.new("RGB", (6 * 256, sheet_h), "#0e1520")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 10),
        f"Sub04 {args.round} Candidate Review ({len(verified_jobs)} jobs)",
        fill="white",
        font=load_font(20),
    )
    for r_idx, (label, thumbs) in enumerate(rows_thumbs):
        y0 = 44 + r_idx * 172
        draw.text((10, y0), label, fill="#9be2f2", font=load_font(15))
        for c_idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, (c_idx * 256, y0 + 22))
    sheet_path = out_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=90)

    receipt = {
        "status": "PASS",
        "round": args.round,
        "batch_dir": str(args.batch_dir.resolve().relative_to(EXP)),
        "total_jobs": len(verified_jobs),
        "contact_sheet": str(sheet_path.relative_to(EXP)),
        "jobs": verified_jobs,
    }
    (out_dir / "technical_acceptance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "round": args.round,
                "jobs": len(verified_jobs),
                "contact_sheet": str(sheet_path.relative_to(EXP)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
