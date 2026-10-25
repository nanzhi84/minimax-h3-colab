# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Assemble declared generated takes at native 24fps with an auditable frame map."""

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from analyze_motion import decode, probe

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "outputs" / "reference" / "wuy0uer_7664781155287239972.mp4"
RATE = 32000


def call(command: list) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace")[-4000:]) from error


def frame_select_expression(indices: list) -> str:
    """Bound expression-tree depth; a linear 358-term sum fails FFmpeg's parser."""
    if len(indices) <= 16:
        return "+".join(f"eq(n,{index})" for index in indices)
    middle = len(indices) // 2
    return f"({frame_select_expression(indices[:middle])})+({frame_select_expression(indices[middle:])})"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    shutil.copyfile(source, temporary)
    if digest(source) != digest(temporary):
        raise ValueError("Director entry copy failed its hash check")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edl", type=Path, required=True)
    parser.add_argument("--director", action="store_true", help="Also save the selected director deliverables at outputs/")
    args = parser.parse_args()
    edl = json.loads(args.edl.read_text())
    if not edl["name"].isidentifier():
        raise ValueError("Invalid master identifier")
    out = EXP / "outputs" / "reproduction" / "masters" / edl["name"]
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "acceptance.json"
    receipt_path.write_text(json.dumps({"status": "RUNNING", "edl_sha256": digest(args.edl)}) + "\n")
    frames_total = sum(part["target_frames"] for part in edl["parts"])
    if frames_total != int(edl.get("target_total_frames", 716)) or not 24 <= frames_total <= 716:
        raise ValueError("EDL total must match its declared duration within the reference length")
    duration_tag = f"{frames_total / 24:.2f}".replace(".", "s")
    audio_parts, part_files, provenance = [], [], []
    for index, part in enumerate(edl["parts"]):
        count = int(part["target_frames"])
        if count < 1:
            raise ValueError("Empty EDL part")
        video = out / f"part_{index:02d}.mp4"
        samples = round(count * RATE / 24)
        if part.get("black") is True:
            call(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                  "color=c=black:s=1344x768:r=24", "-frames:v", str(count), "-an",
                  "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(video)])
            audio_parts.append(np.zeros((samples, 2), dtype=np.int16))
            provenance.append({"type": "explicit_black", "frames": count})
        else:
            batch = EXP / part["batch"]
            metrics = json.loads((batch / "studio_batch_metrics.json").read_text())
            job = next(j for j in metrics["jobs"] if j["job_id"] == part["job_id"])
            source = (batch / job["mp4_file"]).resolve()
            if source == SOURCE.resolve() or "by_storyboard" not in source.parts:
                raise ValueError("Only actual generated batch takes are allowed")
            info = probe(source)
            if "selected_frames" in part:
                selected = [int(x) for x in part["selected_frames"]]
                if (len(selected) != count or selected != sorted(set(selected))
                        or selected[0] < 0 or selected[-1] >= info["declared_frames"]):
                    raise ValueError("Invalid explicit selected_frames list")
                start, end = selected[0], selected[-1] + 1
            else:
                start = int(part.get("start_frame", 0))
                end = int(part.get("end_frame_exclusive", info["declared_frames"]))
                if not 0 <= start < end <= info["declared_frames"] or count > end - start:
                    raise ValueError("Invalid range or implicit frame duplication requested")
                selected = np.linspace(start, end - 1, count).round().astype(int).tolist()
                if len(set(selected)) != count:
                    raise ValueError("Repeated-frame resampling is forbidden")
            expression = frame_select_expression(selected)
            fade_frames = int(part.get("fade_out_frames", 0))
            if fade_frames and not 2 <= fade_frames <= count:
                raise ValueError("Fade must cover 2..target_frames frames")
            filters = f"select='{expression}',setpts=N/(24*TB)"
            if fade_frames:
                filters += f",fade=t=out:start_frame={count - fade_frames}:nb_frames={fade_frames - 1}"
            call(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-an", "-vf",
                  filters, "-frames:v", str(count),
                  "-r", "24", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                  "-pix_fmt", "yuv420p", str(video)])
            speed = (end - start) / count
            duration = count / 24
            fade_out = max(.006, fade_frames / 24)
            rem_speed = speed
            atempo_steps = []
            while rem_speed > 2.0:
                atempo_steps.append("atempo=2.0")
                rem_speed /= 2.0
            while rem_speed < 0.5:
                atempo_steps.append("atempo=0.5")
                rem_speed /= 0.5
            atempo_steps.append(f"atempo={rem_speed:.9f}")
            atempo_chain = ",".join(atempo_steps)
            audio_filter = (f"atrim=start={start / 24:.9f}:end={end / 24:.9f},asetpts=PTS-STARTPTS,"
                            f"{atempo_chain},aresample={RATE},apad,atrim=duration={duration:.9f},"
                            f"afade=t=in:d=0.006,afade=t=out:st={max(0, duration - fade_out):.9f}:d={fade_out:.9f}")
            raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", str(source),
                                           "-vn", "-af", audio_filter, "-ac", "2", "-ar", str(RATE),
                                           "-f", "s16le", "pipe:1"], timeout=90)
            audio = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
            missing = max(0, samples - len(audio))
            if missing:
                audio = np.pad(audio, ((0, missing), (0, 0)))
            audio_parts.append(audio[:samples])
            provenance.append({"type": "h3_generated", "batch": part["batch"], "job_id": job["job_id"],
                               "video_sha256": digest(source), "selected_native_frame_indices": selected,
                               "output_frames": count, "audio_speed_factor": speed,
                               "audio_tail_padding_samples": missing, "audio_fade_in_sec": .006,
                               "audio_fade_out_sec": fade_out, "video_fade_out_frames": fade_frames,
                               "audio_alignment": "atempo, apad and atrim to the declared target duration"})
        part_files.append(video)
    waveform = np.concatenate(audio_parts)
    wav = out / "_generated_audio.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(waveform.astype("<i2").tobytes())
    master = out / f"reproduction_{duration_tag}.mp4"
    # Decode each part independently. H.264 stream-copy concatenation lost the
    # two-frame black segment in the real tight cut and repeated its predecessor.
    inputs = [item for part_file in part_files for item in ["-threads", "1", "-i", str(part_file)]]
    normalize = "".join(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];" for i in range(len(part_files)))
    join = "".join(f"[v{i}]" for i in range(len(part_files)))
    video_filter = normalize + join + f"concat=n={len(part_files)}:v=1:a=0,setpts=N/(24*TB)[v]"
    call(["ffmpeg", "-y", "-loglevel", "error", "-filter_complex_threads", "1", *inputs,
          "-i", str(wav), "-filter_complex", video_filter,
          "-map", "[v]", "-map", f"{len(part_files)}:a:0", "-r", "24",
          "-frames:v", str(frames_total), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
          "-t", f"{frames_total / 24:.9f}", "-movflags", "+faststart", str(master)])
    info = probe(master)
    master_frames = decode(master)
    if (len(master_frames) != frames_total or info["declared_frames"] != frames_total or info["fps"] != 24
            or not info["has_audio"] or (info["width"], info["height"]) != (1344, 768)):
        raise ValueError("Reconstructed master failed full-decode contract")
    cursor = 0
    for part in edl["parts"]:
        if part.get("black") and any(float(master_frames[i].mean()) >= 2
                                     for i in range(cursor, cursor + part["target_frames"])):
            raise ValueError("Declared black segment was lost or moved during video concatenation")
        cursor += part["target_frames"]
    raw_pts = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                      "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(master)], timeout=60)
    pts = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw_pts)["frames"]]
    if len(pts) != frames_total or not np.allclose(pts, np.arange(frames_total) / 24, atol=2e-6, rtol=0):
        raise ValueError("Reconstructed master has irregular frame timestamps")
    comparison = out / "reference_vs_h3.mp4"
    font = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    reference_label = "原片 reference" if frames_total == 716 else "原片 全片等比变速参考"
    filters = (
        f"[0:v]setpts=PTS*{frames_total / 716:.12f},fps=24,scale=640:360:force_original_aspect_ratio=decrease,pad=640:392:(ow-iw)/2:32,"
        f"drawtext=fontfile='{font}':text='{reference_label}':fontcolor=white:fontsize=20:x=12:y=5[a];"
        f"[1:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:392:(ow-iw)/2:32,"
        f"drawtext=fontfile='{font}':text='H3-FL2VA 生成与剪辑':fontcolor=white:fontsize=20:x=12:y=5[b];"
        "[a][b]hstack=inputs=2[v]"
    )
    call(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE), "-i", str(master),
          "-filter_complex", filters, "-map", "[v]", "-map", "1:a:0", "-frames:v", str(frames_total),
          "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "aac",
          "-movflags", "+faststart", str(comparison)])
    receipt = {"status": "PASS", "scope": "完整复现视频技术验收，视觉质量另行评审",
               "edl_sha256": digest(args.edl), "reference_sha256": digest(SOURCE),
               "master": master.name, "master_sha256": digest(master), "frames": frames_total,
               "fps": 24, "duration_sec": frames_total / 24, "comparison_sha256": digest(comparison),
               "comparison_reference_time_ratio": 716 / frames_total,
               "pts_cadence_verified": True, "declared_black_frames_verified": True,
               "master_pts_policy": "independently decode each selected part, concatenate frames, rebuild PTS at 24fps",
               "parts": provenance, "original_audio_used": False, "optical_flow_interpolation_used": False}
    if args.director:
        variant = "" if frames_total == 716 else "_tight"
        director = EXP / f"outputs/sub03_director_master{variant}_{duration_tag}.mp4"
        side_by_side = EXP / f"outputs/sub03_reference_vs_director{variant}.mp4"
        decisions = EXP / f"outputs/sub03_director_edit_decision_list{variant}.json"
        publish_file(master, director)
        publish_file(comparison, side_by_side)
        if digest(director) != receipt["master_sha256"] or digest(side_by_side) != receipt["comparison_sha256"]:
            raise ValueError("Director entry files differ from the verified master")
        temporary_decisions = decisions.with_suffix(".tmp.json")
        temporary_decisions.write_text(json.dumps({**edl, "master_source_directory": str(out.relative_to(EXP)),
                                                  "acceptance": receipt}, ensure_ascii=False, indent=2) + "\n")
        temporary_decisions.replace(decisions)
        receipt["director_outputs"] = {"master": str(director.relative_to(EXP)),
                                       "comparison": str(side_by_side.relative_to(EXP)),
                                       "decisions": str(decisions.relative_to(EXP))}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    (out / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=2) + "\n")
    wav.unlink()
    print(json.dumps({"status": "PASS", "master": str(master.relative_to(EXP)), "frames": frames_total}))


if __name__ == "__main__":
    main()
