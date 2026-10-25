# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Decode real H3 candidates and build source-aligned evidence for visual review."""

import argparse
import hashlib
import html
import json
import os
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from analyze_motion import decode, font, probe
from PIL import Image, ImageDraw

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "outputs" / "reference" / "wuy0uer_7664781155287239972.mp4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_sheet(source: list, generated: list, spec: dict, job: dict, path: Path) -> None:
    canvas = Image.new("RGB", (1920, 970), "#111927")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{job['job_id']} | 原片与 H3 同阶段取样（各自原生时间）", fill="white", font=font(23))
    src_idx = np.linspace(spec["first"], spec["last"], 12).round().astype(int)
    gen_idx = np.linspace(0, len(generated) - 1, 12).round().astype(int)
    for block in range(2):
        for kind, frames, indices, fps in [(0, source, src_idx, 30), (1, generated, gen_idx, 24)]:
            row = block * 2 + kind
            y = 58 + row * 224
            for col in range(6):
                index = int(indices[block * 6 + col])
                x = col * 320
                canvas.paste(Image.fromarray(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)), (x, y))
                label = "原片" if kind == 0 else "H3"
                draw.text((x + 8, y + 185), f"{label} f{index} · {index / fps:.3f}s",
                          fill="#8ed7ed" if kind == 0 else "#f5c88b", font=font(18))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    spec = json.loads(args.spec.read_text())
    specifications = {s["id"]: s for s in spec["shots"]}
    manifest = json.loads((batch / "manifest.json").read_text())
    manifests = {s["shot_id"]: s for s in manifest["storyboards"]}
    metrics = json.loads((batch / "studio_batch_metrics.json").read_text())
    if len(metrics["jobs"]) != len(manifest["jobs"]):
        raise ValueError("Missing jobs in real batch results")
    source = decode(SOURCE)
    out = EXP / "outputs" / "reproduction" / f"{spec['round']}_review"
    out.mkdir(parents=True, exist_ok=True)
    results, cards = [], []
    for job in metrics["jobs"]:
        path = batch / job["mp4_file"]
        info = probe(path)
        frames = decode(path)
        expected = manifests[job["shot_id"]]["num_frames"]
        if (info["fps"] != 24 or len(frames) != expected or info["declared_frames"] != expected
                or (info["width"], info["height"]) != (1344, 768) or not info["has_audio"]):
            raise ValueError(f"Invalid decoded video contract: {job['job_id']}")
        image = out / f"{job['job_id']}.jpg"
        comparison_sheet(source, frames, specifications[job["shot_id"]], job, image)
        differences = np.array([float(np.abs(b.astype(float) - a).mean())
                                for a, b in pairwise(frames)])
        results.append({
            "job_id": job["job_id"], "shot_id": job["shot_id"], "seed": job["seed"],
            "video": str(path.relative_to(EXP)), "sha256": sha(path),
            "fps": info["fps"], "frames": len(frames), "duration_sec": len(frames) / 24,
            "resolution": [info["width"], info["height"]], "has_audio": info["has_audio"],
            "source_first": specifications[job["shot_id"]]["first"],
            "source_last": specifications[job["shot_id"]]["last"],
            "endpoint_ssim": {k: job.get(k) for k in ["first_frame_ssim", "last_frame_ssim"]},
            "image_step_mae_mean_320x180": round(float(differences.mean()), 4),
            "metric_warning": "图像差分含特效与相机；不代表人体动作质量，不自动排名",
            "comparison_sheet": image.name, "visual_review": "PENDING",
        })
        url = html.escape(os.path.relpath(path, out), quote=True)
        title = html.escape(job["job_id"])
        cards.append(f'''<article><h2>{title}</h2><video src="{url}" controls muted playsinline preload="metadata"></video>
<p>{expected} 帧 · {expected / 24:.3f}s · 24fps · 真实 H3 输出</p>
<label>速度 <select aria-label="{title}播放速度"><option value="1">1×</option><option value="0.5">0.5×</option></select></label>
<p><a href="{image.name}">原片 / H3 动作阶段联系表</a></p></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3-FL2VA 复现候选审片</title><style>
body{font:16px/1.6 system-ui;background:#0c1420;color:#e6edf8;max-width:1500px;margin:auto;padding:28px}
h1{font-size:28px}h2{font-size:17px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}
article{background:#172334;border:1px solid #32465f;padding:18px;border-radius:12px}
video{width:100%;aspect-ratio:16/9;background:#000}p{color:#b9cadd}a{color:#85dced}
select{font:inherit;background:#25374c;color:white;padding:8px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><h1>H3-FL2VA 真实生成候选：''' + html.escape(spec["round"]) + '''</h1>
<p>先静音 1× 看完整动作，再半速检查身体、握剑手、刀锋和镜头路径。联系表按动作阶段等比例取样，两种视频时间分别标注；不同长度不代表严格逐时刻匹配。所有候选保留，不按综合分自动定胜者。</p>
<div class="grid">''' + "".join(cards) + '''</div><script>
document.querySelectorAll('article').forEach(card=>{const v=card.querySelector('video');
card.querySelector('select').addEventListener('change',e=>v.playbackRate=Number(e.target.value));
v.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(v!==other)other.pause()}));
});</script></html>'''
    (out / "review.html").write_text(document)
    receipt = {"status": "PASS", "scope": "真实视频技术检查，不代表视觉复现通过",
               "source_sha256": sha(SOURCE), "spec_sha256": sha(args.spec),
               "manifest_sha256": sha(batch / "manifest.json"), "jobs": results,
               "assertions": ["全部预期候选存在且完整解码", "输出原生24fps与目标帧数一致",
                              "全部1344x768且存在音轨", "每条输出已记录哈希和原片帧范围"]}
    (out / "technical_acceptance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "jobs": len(results), "review": str(out.relative_to(EXP))}))


if __name__ == "__main__":
    main()
