# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Verify rendered EDLs and make a synchronized, local reference review page."""

import argparse
import csv
import html
import json
import os
import platform
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
from analyze_motion import decode, frame_diagnostics, probe, runs, sha256, sheet

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "outputs/reference/wuy0uer_7664781155287239972.mp4"
LABELS = {
    "round01_long_baseline": "首批长镜头基线（场景漂移）",
    "round02_short_chain": "短窗口链（动作路线对照）",
    "round02_long_window_retimed": "两个长窗口压缩版（同长对照）",
    "round03_rewritten_long": "第三批长镜头（场景和运镜修订）",
    "round03_clear_end": "第三批长镜头（清晰尾帧再接爆发）",
    "selected_reproduction": "最终选片版",
    "selected_reproduction_tight": "节奏强化导演版（逐段修剪与变速）",
}


def audit(edl_path: Path, out: Path) -> dict:
    edl = json.loads(edl_path.read_text())
    folder = EXP / "outputs/reproduction/masters" / edl["name"]
    receipt = json.loads((folder / "acceptance.json").read_text())
    video = folder / receipt["master"]
    expected = int(edl.get("target_total_frames", 716))
    if (receipt["status"] != "PASS" or receipt["edl_sha256"] != sha256(edl_path)
            or receipt["master_sha256"] != sha256(video) or receipt["reference_sha256"] != sha256(SOURCE)):
        raise ValueError("EDL, source or master no longer matches its acceptance receipt")
    info, frames = probe(video), decode(video)
    if info["fps"] != 24 or len(frames) != expected or info["declared_frames"] != expected or not info["has_audio"]:
        raise ValueError("Invalid complete master")
    comparison = folder / "reference_vs_h3.mp4"
    comparison_info = probe(comparison)
    if (sha256(comparison) != receipt["comparison_sha256"] or len(decode(comparison)) != expected
            or comparison_info["fps"] != 24 or not comparison_info["has_audio"]):
        raise ValueError("Comparison video no longer matches its receipt or cannot fully decode")
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)], timeout=60)
    pts = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw)["frames"]]
    if len(pts) != expected or not np.allclose(pts, np.arange(expected) / 24, atol=2e-6, rtol=0):
        raise ValueError("Master is not CFR at every frame")
    rows = frame_diagnostics(frames, 24)
    with (out / f"{edl['name']}_per_frame.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cursor, cuts, groups, black_expected = 0, [], [], []
    for part in edl["parts"]:
        if part.get("black"):
            black_expected.extend(range(cursor, cursor + part["target_frames"]))
        cursor += part["target_frames"]
    if cursor != expected or any(float(frames[i].mean()) >= 2 for i in black_expected):
        raise ValueError("Explicit black frames do not match the EDL")
    cursor = 0
    for before, after in pairwise(edl["parts"]):
        cursor += before["target_frames"]
        if before.get("black") or after.get("black"):
            continue
        b = cursor
        cuts.append({"frame": b, "time_sec": b / 24, "before": before["job_id"], "after": after["job_id"],
                     "step_mae_255": rows[b - 1]["rgb_step_mae_255"],
                     "pre_six_step_mae_mean": float(np.mean([r["rgb_step_mae_255"] for r in rows[b - 7:b - 1]])),
                     "post_six_step_mae_mean": float(np.mean([r["rgb_step_mae_255"] for r in rows[b:b + 6]])),
                     "interpretation": "外观变化和光流包含相机、衣物、特效，不能判定人体速度连续或主观丝滑"})
        groups.append([b + i for i in [-6, -3, -1, 0, 1, 3, 6]])
    if groups:
        sheet(frames, groups, 24, edl["name"] + " | 接缝前后原生帧", out / f"{edl['name']}_seams.jpg")
    samples = np.linspace(0, expected - 18, 30).round().astype(int).tolist()
    sheet(frames, [samples[i:i + 5] for i in range(0, 30, 5)], 24,
          edl["name"] + " | 完整动作阶段取样", out / f"{edl['name']}_overview.jpg")
    return {"name": edl["name"], "title": LABELS.get(edl["name"], edl["name"]),
            "description": edl["description"], "master_sha256": sha256(video), "comparison_sha256": sha256(comparison),
            "edl_sha256": sha256(edl_path), "url": os.path.relpath(video, out),
            "comparison_url": os.path.relpath(folder / "reference_vs_h3.mp4", out),
            "frames": len(frames), "fps": 24, "pts_cadence_verified": True,
            "reference_time_ratio": 716 / expected,
            "declared_black_frames": len(black_expected),
            "observed_black_runs": runs([i for i, f in enumerate(frames) if float(f.mean()) < 2], 24),
            "joins": cuts, "visual_quality": "另行评审，技术PASS不等于复现成功"}


def render(results: list, selected: str, out: Path) -> None:
    options = "".join(f'<option value="{html.escape(r["name"])}" {"selected" if r["name"] == selected else ""}>'
                      f'{html.escape(r["title"])}</option>' for r in results)
    source = html.escape(os.path.relpath(SOURCE, out), quote=True)
    data = json.dumps(results, ensure_ascii=False).replace("</", "<\\/")
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>子实验3 H3参考片复现审片</title><style>
body{font:16px/1.55 system-ui;background:#0d1522;color:#e8eef8;max-width:1440px;padding:24px;margin:auto}
h1{font-size:27px;margin:0 0 12px}p{color:#bacce0}button,select{font:inherit;background:#23384f;color:white;border:1px solid #4b6681;border-radius:6px;padding:8px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #65d7e8}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
video{width:100%;background:black;aspect-ratio:16/9}h2{font-size:17px}.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}
input[type=range]{flex:1;min-width:220px}a{color:#85deef}.note{background:#172738;padding:12px;border-radius:7px}
@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>子实验3 · MiniMax H3-FL2VA 参考片复现</h1>
<p>原片 29.833秒 / 30fps；生成母版24fps。等长版按成片时间同步；节奏版对照中的原片按全片比例加速，仅供大阶段参考。H3中间动作由模型生成，修剪、变速和收尾见EDL。</p>
<div class="bar"><label for="version">比较版本</label><select id="version">''' + options + '''</select><a id="download">打开生成视频</a><a id="compare">并排对照视频</a></div>
<p id="description" class="note"></p>
<div class="pair"><section><h2>原片 · wuy0uer 参考</h2><video id="source" src="''' + source + '''" muted playsinline preload="metadata"></video></section>
<section><h2>H3-FL2VA · 生成与剪辑</h2><video id="master" muted playsinline preload="metadata"></video></section></div>
<div class="bar"><button id="play">同步播放</button><button id="pause">暂停</button><button id="previous">前一帧</button><button id="next">后一帧</button>
<label for="speed">速度</label><select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option></select><label><input id="sound" type="checkbox">H3原生音效</label></div>
<div class="bar"><label for="seek">时间</label><input id="seek" type="range" min="0" max="715" step="1" value="0" aria-valuetext="0.000秒"><output id="time">0.000s</output></div>
<div id="joins" class="bar"></div><p id="status" role="status">加载视频元数据…</p>
<p>先用1×看整体节奏，再半速检查接缝、握剑手、身体形变和运镜。共享首尾图只能约束边界外观，不能保证速度连续。图像差分与光流仅供定位异常，不作自动质量排名。</p>
<p><a id="sheet">接缝联系表</a> · <a id="overview">动作阶段联系表</a> · <a href="acceptance.json">技术核验回执</a> · <a href="../../../generation-report.md">完整工作流报告</a></p>
<script>
const takes=''' + data + ''';
const $=id=>document.getElementById(id), a=$('source'),b=$('master');
const stop=()=>{a.pause();b.pause()};
const ratio=()=>takes.find(t=>t.name===$('version').value).reference_time_ratio;
const settings=()=>{const rate=Number($('speed').value);b.defaultPlaybackRate=b.playbackRate=rate;a.defaultPlaybackRate=a.playbackRate=rate*ratio();b.muted=!$('sound').checked};
const seek=t=>{stop();a.currentTime=t*ratio();b.currentTime=t;$('seek').value=Math.round(t*24);$('seek').setAttribute('aria-valuetext',t.toFixed(3)+'秒');$('time').textContent=t.toFixed(3)+'s · H3 f'+Math.round(t*24)};
function load(){stop();const item=takes.find(t=>t.name===$('version').value);b.src=item.url;a.currentTime=0;$('seek').max=item.frames-1;settings();
$('description').textContent=item.description;$('download').href=item.url;$('compare').href=item.comparison_url;
$('sheet').href=item.name+'_seams.jpg';$('sheet').hidden=!item.joins.length;$('overview').href=item.name+'_overview.jpg';
$('joins').replaceChildren();item.joins.forEach(j=>{const el=document.createElement('button');el.textContent='接缝 '+j.time_sec.toFixed(2)+'s';
el.addEventListener('click',()=>seek(Math.max(0,j.time_sec-.5)));$('joins').appendChild(el)});$('seek').value=0;$('time').textContent='0.000s';}
$('version').addEventListener('change',load);$('pause').addEventListener('click',stop);
$('play').addEventListener('click',async()=>{if(b.ended)seek(0);settings();a.currentTime=b.currentTime*ratio();try{await Promise.all([a.play(),b.play()])}catch(e){$('status').textContent='播放失败：'+e.message}});
$('seek').addEventListener('input',e=>seek(Number(e.target.value)/24));
$('previous').addEventListener('click',()=>seek(Math.max(0,Math.round(b.currentTime*24)-1)/24));
$('next').addEventListener('click',()=>seek(Math.min(Number($('seek').max),Math.round(b.currentTime*24)+1)/24));
$('speed').addEventListener('change',settings);
$('sound').addEventListener('change',e=>b.muted=!e.target.checked);
b.addEventListener('timeupdate',()=>{$('seek').value=Math.min(Number($('seek').max),Math.round(b.currentTime*24));$('seek').setAttribute('aria-valuetext',b.currentTime.toFixed(3)+'秒');$('time').textContent=b.currentTime.toFixed(3)+'s · H3 f'+Math.min(Number($('seek').max),Math.round(b.currentTime*24));
if(!b.paused&&Math.abs(a.currentTime-b.currentTime*ratio())>.08)a.currentTime=b.currentTime*ratio()});b.addEventListener('ended',stop);
[a,b].forEach(v=>{v.addEventListener('loadedmetadata',()=>{settings();$('status').textContent='已加载 · 原片 '+a.videoWidth+'×'+a.videoHeight+' / H3 '+b.videoWidth+'×'+b.videoHeight});
v.addEventListener('error',()=>{$('status').textContent='视频加载失败，请检查文件与HTTP Range支持'})});load();
</script></html>'''
    (out / "index.html").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edls", type=Path, nargs="+", required=True)
    parser.add_argument("--preferred", required=True)
    args = parser.parse_args()
    out = EXP / "outputs/reproduction/final_review"
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "acceptance.json"
    receipt.write_text('{"status":"RUNNING"}\n')
    results = [audit(path.resolve(), out) for path in args.edls]
    if args.preferred not in {r["name"] for r in results}:
        raise ValueError("Preferred version is not in the reviewed EDLs")
    render(results, args.preferred, out)
    evidence = {"status": "PASS", "scope": "母版完整解码、逐帧时间戳、EDL来源与黑场；视觉质量另行评审",
                "reference_sha256": sha256(SOURCE), "code_sha256": sha256(Path(__file__)),
                "environment": {"python": platform.python_version(), "system": platform.platform(),
                                "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
                "masters": results, "preferred": args.preferred,
                "assertions": ["母版与EDL/参考哈希匹配", "每条母版及对照视频完整解码到各EDL声明总帧数且24fps",
                               "每帧PTS符合n/24", "声明黑场为黑帧", "全部母版及对照视频有音轨"]}
    receipt.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "masters": len(results), "page": str(out.relative_to(EXP))}))


if __name__ == "__main__":
    main()
