# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Local, native-cadence evidence builder; motion proxies never decide acceptance."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_review import render_review

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SIZE = (320, 180)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    raw = json.loads(result.stdout)
    video = next(s for s in raw["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(video["r_frame_rate"]))
    if fps <= 0 or video["r_frame_rate"] != video["avg_frame_rate"]:
        raise ValueError(f"{path.name}: variable or invalid frame rate")
    return {
        "fps": fps, "declared_frames": int(video["nb_frames"]),
        "width": video["width"], "height": video["height"], "codec": video["codec_name"],
        "container_duration_sec": float(raw["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in raw["streams"]),
    }


def decode(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
            if len(frames) > 5000:
                raise ValueError("Input exceeds the bounded 5000-frame study limit")
    finally:
        cap.release()
    if len(frames) < 3:
        raise ValueError(f"{path.name}: empty or truncated decode")
    return frames


def font(size: int):
    for path in [Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def sheet(frames: list, groups: list, fps: float, title: str, path: Path) -> None:
    cols = max(map(len, groups))
    canvas = Image.new("RGB", (cols * 320, 56 + len(groups) * 216), "#0d1420")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="white", font=font(22))
    for row, indices in enumerate(groups):
        for col, index in enumerate(indices):
            index = min(max(0, index), len(frames) - 1)
            x, y = col * 320, 56 + row * 216
            canvas.paste(Image.fromarray(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)), (x, y))
            draw.text((x + 8, y + 186), f"f{index:03d}  {index / fps:.3f}s",
                      fill="#b9e4ff", font=font(18))
    canvas.save(path)


def runs(indices: list, fps: float, min_frames: int = 3) -> list:
    groups = []
    for index in indices:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [{"first_frame": g[0], "last_frame": g[-1], "count": len(g),
             "start_sec": round(g[0] / fps, 4), "end_sec_exclusive": round((g[-1] + 1) / fps, 4)}
            for g in groups if len(g) >= min_frames]


def frame_diagnostics(frames: list, fps: float) -> list:
    rows = []
    previous = frames[0]
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    for index, current in enumerate(frames[1:], 1):
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        initial_flow = np.empty((*gray.shape, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, initial_flow, .5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2)
        rows.append({
            "frame": index, "time_sec": round(index / fps, 6),
            "rgb_step_mae_255": round(float(np.abs(current.astype(float) - previous).mean()), 6),
            "gray_mean": round(float(gray.mean()), 6),
            "flow_median_screenwidth_per_sec": round(float(np.median(magnitude) * fps / SIZE[0]), 6),
            "flow_p90_screenwidth_per_sec": round(float(np.percentile(magnitude, 90) * fps / SIZE[0]), 6),
            "flow_median_x_px_per_frame": round(float(np.median(flow[:, :, 0])), 6),
            "flow_median_y_px_per_frame": round(float(np.median(flow[:, :, 1])), 6),
        })
        previous, previous_gray = current, gray
    return rows


def boundaries(spec: dict, count: int) -> tuple:
    if "edl" not in spec:
        return spec.get("boundaries", []), None
    edl_path = EXP / spec["edl"]
    edl = json.loads(edl_path.read_text())
    edits = edl["edit_decision_list"]
    if sum(s["kept_frames"] for s in edits) != count:
        raise ValueError(f"{spec['id']}: EDL and decoded frames differ")
    cumulative = 0
    result = []
    for before, after in pairwise(edits):
        cumulative += before["kept_frames"]
        result.append({"frame": cumulative, "kind": after["chain_mode"]})
    return result, {"path": spec["edl"], "sha256": sha256(edl_path)}


def analyze(spec: dict, out: Path) -> dict:
    path = EXP / spec["path"]
    digest = sha256(path)
    if spec.get("sha256", digest) != digest:
        raise ValueError(f"{spec['id']}: source hash mismatch")
    info = probe(path)
    if info["fps"] != spec["expected_fps"]:
        raise ValueError(f"{spec['id']}: unexpected FPS")
    frames = decode(path)
    count = len(frames)
    if count != info["declared_frames"] or count != spec["expected_frames"]:
        raise ValueError(f"{spec['id']}: decoded / probed / expected frame counts differ")
    fps = info["fps"]
    # Check every frame PTS; equal average/nominal rates alone do not prove CFR.
    pts_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    pts = [float(f["best_effort_timestamp_time"]) for f in json.loads(pts_result.stdout)["frames"]]
    if len(pts) != count or not np.allclose(np.diff(pts), 1 / fps, atol=2e-6, rtol=0):
        raise ValueError(f"{spec['id']}: irregular PTS cadence")
    rows = frame_diagnostics(frames, fps)
    with (out / f"{spec['id']}_per_frame.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    step = np.array([r["rgb_step_mae_255"] for r in rows]) / 255
    legacy = 100 / (1 + float(np.abs(np.diff(step)).mean()) / (float(step.mean()) + 1e-4))
    nodes, edl = boundaries(spec, count)
    windows, groups = [], []
    for node in nodes:
        b = int(node["frame"])
        if not 6 <= b < count - 6:
            raise ValueError(f"{spec['id']}: invalid boundary window")
        pre = rows[b - 7:b - 1]
        post = rows[b:b + 6]
        windows.append({**node, "time_sec": round(b / fps, 6),
                        "boundary_step_mae_255": rows[b - 1]["rgb_step_mae_255"],
                        "pre_6_step_mae_mean": round(float(np.mean([r["rgb_step_mae_255"] for r in pre])), 4),
                        "post_6_step_mae_mean": round(float(np.mean([r["rgb_step_mae_255"] for r in post])), 4),
                        "interpretation": "外观变化诊断；不等同于角色速度、加速度或剪辑合格判定"})
        groups.append([b + offset for offset in [-6, -3, -1, 0, 1, 3, 6]])
    if groups:
        sheet(frames, groups, fps, f"{spec['title']} | 每行一个接缝，时间使用原生 FPS", out / f"{spec['id']}_seams.jpg")
    indices = [round(t * fps) for t in range(math.ceil(count / fps))]
    sheet(frames, [indices[i:i + 5] for i in range(0, len(indices), 5)], fps,
          spec["title"] + " | 1 秒取样：不能替代正常速度观看", out / f"{spec['id']}_overview.jpg")
    if spec["id"] == "reference":
        for start in [0.5, 8.5, 18.5, 23.0]:
            indices = [round((start + i / 8) * fps) for i in range(16)]
            sheet(frames, [indices[i:i + 4] for i in range(0, 16, 4)], fps,
                  f"参考动作窗口 {start:.1f}–{start + 2:.1f}s | 8 Hz 取样", out / f"reference_motion_{start:.1f}s.jpg")
    black = [i for i, frame in enumerate(frames) if float(frame.mean()) < 2]
    return {"id": spec["id"], "path": spec["path"], "sha256": digest, **info,
            "decoded_frames": count, "pts_cadence_verified": True,
            "duration_sec": round(count / fps, 6), "edl": edl,
            "near_duplicate_step_count_mae_le_0_2": int(np.sum(step * 255 <= .2)),
            "black_runs_rgb_mean_lt_2": runs(black, fps),
            "legacy_smoothness_proxy_at_320x180": round(legacy, 2),
            "legacy_score_warning": "复算帧差代理，不用于跨内容或跨 FPS 排名，也不代表人体动作自然",
            "boundary_windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXP / "outputs" / "analysis")
    parser.add_argument("--receipt", type=Path, default=REPO / "artifacts" / "exp06-sub03-motion-review" / "acceptance.json")
    args = parser.parse_args()
    if not shutil.which("ffprobe"):
        raise SystemExit("ffprobe is required")
    cv2.setNumThreads(1)
    spec_path = EXP / "code" / "study.json"
    study = json.loads(spec_path.read_text())
    out = args.output.resolve()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a previous success receipt as its result.
    args.receipt.write_text(json.dumps({"status": "RUNNING", "study_sha256": sha256(spec_path)}))
    out.mkdir(parents=True, exist_ok=True)
    results = [analyze(spec, out) for spec in study["clips"]]
    evidence = {
        "status": "PASS", "scope": "本地参考分析工件；不代表新 H3 生成或主观流畅度验收通过",
        "source": study["source"], "study_sha256": sha256(spec_path),
        "code_sha256": {p.name: sha256(p) for p in [Path(__file__), EXP / "code" / "render_review.py"]},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "libraries": {p: importlib.metadata.version(p) for p in ["numpy", "pillow", "opencv-python-headless"]},
                        "ffprobe": subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0]},
        "analysis_size": list(SIZE), "clips": results,
        "assertions": ["参考片哈希匹配", "三个视频完整解码", "原生 FPS 和逐帧 PTS 核对通过",
                       "两个 EDL 与成片帧数匹配", "CSV 和原生帧时间联系表已生成"],
        "unverified": ["作者具体模型、提示词及后期流程", "新 H3 对照生成", "yoryon 主观丝滑度验收", "音轨听感与节拍人工验收"],
    }
    render_review(study, out, EXP)
    evidence["artifacts"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in [".csv", ".jpg", ".html"]}
    serialized = json.dumps(evidence, ensure_ascii=False, indent=2)
    (out / "acceptance.json").write_text(serialized + "\n")
    args.receipt.write_text(serialized + "\n")
    print(json.dumps({"status": "PASS", "clips": len(results),
                      "decoded_frames": sum(c["decoded_frames"] for c in results),
                      "scope": evidence["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
