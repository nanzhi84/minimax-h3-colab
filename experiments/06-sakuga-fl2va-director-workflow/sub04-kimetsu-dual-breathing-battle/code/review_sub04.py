# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Audit Sub04 masters, generate side-by-side workflow comparisons, and build the Interactive Gacha & NLE Workbench."""

import argparse
import csv
import hashlib
import html
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare_sub04 import load_font
from review_candidates import decode, probe

EXP = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace")[-4000:]) from error


def render_grid(
    title: str,
    frames: list[np.ndarray],
    indices: list[int],
    cols: int,
    out_path: Path,
) -> None:
    w, h = 384, 216
    rows = max(1, (len(indices) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * w, 46 + rows * (h + 30)), "#0d131c")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 10), title, fill="white", font=load_font(22))
    for k, f_idx in enumerate(indices):
        idx_clamped = max(0, min(len(frames) - 1, f_idx))
        rgb = cv2.cvtColor(frames[idx_clamped], cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS)
        r, c = divmod(k, cols)
        x, y = c * w, 46 + r * (h + 30)
        sheet.paste(im, (x, y))
        draw.text(
            (x + 8, y + h + 4),
            f"f{idx_clamped:03d}  {idx_clamped / 24:.3f}s",
            fill="#a9d4e8",
            font=load_font(16),
        )
    sheet.save(out_path, quality=90)


def build_side_by_side(
    left_mp4: Path,
    left_label: str,
    right_mp4: Path,
    right_label: str,
    frames_total: int,
    out_mp4: Path,
) -> None:
    font = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    left_safe = left_label.replace(":", "·")
    right_safe = right_label.replace(":", "·")
    filters = (
        f"[0:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:392:(ow-iw)/2:32,"
        f"drawtext=fontfile='{font}':text='{left_safe}':fontcolor=white:fontsize=19:x=12:y=6[a];"
        f"[1:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:392:(ow-iw)/2:32,"
        f"drawtext=fontfile='{font}':text='{right_safe}':fontcolor=white:fontsize=19:x=12:y=6[b];"
        "[a][b]hstack=inputs=2[v]"
    )
    call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(left_mp4),
            "-i",
            str(right_mp4),
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-frames:v",
            str(frames_total),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )


def audit_master(edl_path: Path, review_dir: Path) -> dict:
    edl = json.loads(edl_path.read_text())
    name = edl["name"]
    master_dir = EXP / "outputs" / "masters" / name
    acceptance = json.loads((master_dir / "acceptance.json").read_text())
    mp4 = master_dir / acceptance["master"]
    info = probe(mp4)
    frames = decode(mp4)
    total = int(edl["target_total_frames"])
    if len(frames) != total or info["declared_frames"] != total or info["fps"] != 24:
        raise ValueError(f"Master {name} failed full-decode check")

    small = [cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
    step_maes = [0.0] + [float(np.mean(np.abs(small[i] - small[i - 1]))) for i in range(1, len(small))]

    # Write per-frame CSV
    csv_path = review_dir / f"{name}_per_frame.csv"
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame", "time_sec", "luma_mean", "step_mae_255"])
        for idx, frm in enumerate(frames):
            writer.writerow([idx, f"{idx / 24:.6f}", f"{float(frm.mean()):.3f}", f"{step_maes[idx]:.4f}"])

    # Render 30-frame overview grid
    overview_idx = np.linspace(0, total - 1, 30).round().astype(int).tolist()
    render_grid(f"{name} | 全片30点动作阶段取样", frames, overview_idx, 5, review_dir / f"{name}_overview.jpg")

    # Compute joins & seam contact sheet
    joins = []
    seam_rows = []
    cursor = 0
    for idx, part in enumerate(edl["parts"]):
        cnt = int(part["target_frames"])
        if idx > 0:
            join_f = cursor
            before_job = edl["parts"][idx - 1].get("job_id", "black")
            after_job = part.get("job_id", "black")
            pre_slice = step_maes[max(1, join_f - 6) : join_f]
            post_slice = step_maes[join_f + 1 : min(total, join_f + 7)]
            joins.append(
                {
                    "frame": join_f,
                    "time_sec": round(join_f / 24, 4),
                    "slot": part.get("slot", f"part_{idx:02d}"),
                    "mode": part.get("mode", "FL2VA"),
                    "before": before_job,
                    "after": after_job,
                    "step_mae_255": round(step_maes[join_f], 3),
                    "pre_six_step_mae_mean": round(float(np.mean(pre_slice)), 3) if pre_slice else 0.0,
                    "post_six_step_mae_mean": round(float(np.mean(post_slice)), 3) if post_slice else 0.0,
                }
            )
            row_offsets = [-6, -3, -1, 0, 1, 3, 6]
            seam_rows.extend([max(0, min(total - 1, join_f + off)) for off in row_offsets])
        cursor += cnt

    if seam_rows:
        render_grid(f"{name} | 跨镜接缝前后原生帧 (-6,-3,-1,0,+1,+3,+6)", frames, seam_rows, 7, review_dir / f"{name}_seams.jpg")

    join_vals = [j["step_mae_255"] for j in joins]
    return {
        "name": name,
        "description": edl["description"],
        "master_sha256": digest(mp4),
        "edl_sha256": digest(edl_path),
        "url": f"../masters/{name}/{mp4.name}",
        "frames": total,
        "fps": 24,
        "duration_sec": round(total / 24, 3),
        "mean_seam_step_mae_255": round(float(np.mean(join_vals)), 2) if join_vals else 0.0,
        "max_seam_step_mae_255": round(float(np.max(join_vals)), 2) if join_vals else 0.0,
        "mean_global_step_mae_255": round(float(np.mean(step_maes[1:])), 2),
        "joins": joins,
    }


def render_workbench_html(results: list[dict], gacha_report: dict, out_dir: Path) -> None:
    """Render an interactive Gacha & NLE Clip-Trimming Workbench HTML page."""
    slots_json = json.dumps(gacha_report.get("director_30s00_slots", []), ensure_ascii=False)
    masters_json = json.dumps(results, ensure_ascii=False)
    comp_json = json.dumps(gacha_report.get("workflow_comparison", {}), ensure_ascii=False)

    cards_html = []
    for m in results:
        cards_html.append(
            f"""
            <div class="master-card">
              <h3>{html.escape(m['name'])} <span class="badge">{m['frames']}f / {m['duration_sec']}s</span></h3>
              <p class="desc">{html.escape(m['description'])}</p>
              <div class="metrics">
                <span>平均接缝步进 MAE: <strong>{m['mean_seam_step_mae_255']}</strong></span>
                <span>最大接缝步进 MAE: <strong>{m['max_seam_step_mae_255']}</strong></span>
                <span>全片平均运动幅度: <strong>{m['mean_global_step_mae_255']}</strong></span>
              </div>
              <video controls preload="metadata" src="{html.escape(m['url'])}"></video>
              <div class="links">
                <a href="{html.escape(m['name'])}_overview.jpg" target="_blank">30点概览图</a> ·
                <a href="{html.escape(m['name'])}_seams.jpg" target="_blank">接缝多帧检视表</a> ·
                <a href="{html.escape(m['name'])}_per_frame.csv" target="_blank">逐帧CSV</a>
              </div>
            </div>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Sub04 鬼灭之刃风格双呼吸死斗 — 抽卡工作流对比与导演精剪工作台</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b1017; color: #e5eef8; }}
    header {{ padding: 20px 28px; background: #111926; border-bottom: 1px solid #223248; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    main {{ padding: 20px 28px; max-width: 1640px; margin: 0 auto; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; margin-bottom: 28px; }}
    .master-card, .slot-card {{ background: #131c29; border: 1px solid #24344d; border-radius: 10px; padding: 16px; }}
    .master-card h3, .slot-card h4 {{ margin: 0 0 8px; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #1e3a5f; color: #8fd3ff; padding: 2px 8px; border-radius: 6px; font-size: 13px; }}
    .desc {{ color: #a7b8cc; font-size: 13.5px; margin: 0 0 10px; line-height: 1.45; }}
    .metrics {{ display: flex; gap: 16px; font-size: 13px; color: #9ec5e6; margin-bottom: 10px; flex-wrap: wrap; }}
    video {{ width: 100%; border-radius: 8px; background: #000; }}
    .links {{ margin-top: 8px; font-size: 13px; }}
    a {{ color: #67bfff; text-decoration: none; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; background: #152233; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
    button, select, input {{ background: #1d2d44; color: #fff; border: 1px solid #355075; border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
    button:hover {{ background: #294366; }}
    .slots-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #223248; padding: 5px 6px; text-align: left; }}
    .selected-row {{ background: rgba(46, 160, 67, 0.18); }}
    pre {{ background: #0d141e; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; max-height: 300px; }}
  </style>
</head>
<body>
  <header>
    <h1>Sub04：《鬼灭之刃》风格水火双呼吸死斗（30s）— 通用抽卡工作流对比与交互式精剪台</h1>
    <p class="desc">涵盖 20 个逐拍同源关键帧分镜（KF00..KF20），支持对比 <strong>Workflow A（未修剪基线）</strong>、<strong>Workflow B（I2VA + 73帧跨拍混合消融）</strong>、<strong>30.00s 单调DP最优导演版</strong> 与 <strong>26.00s 节奏强化导演版</strong>，并支持逐镜挑卡与调整入出点导出自定义 EDL。</p>
  </header>
  <main>
    <div class="toolbar">
      <strong>同屏对比视频：</strong>
      <a href="../../sub04_workflow_a_vs_director.mp4" target="_blank">▶ 未修剪基线 vs 30s DP导演版 (同屏对比)</a>
      <span>|</span>
      <a href="../../sub04_workflow_b_vs_director.mp4" target="_blank">▶ I2VA/73f混合版 vs 30s DP导演版 (同屏对比)</a>
      <span>|</span>
      <a href="workflow_comparison_30s.jpg" target="_blank">🖼 三大工作流逐秒对比大图</a>
      <button id="export-edl-btn" style="margin-left:auto;">导出当前自定义挑卡 EDL JSON</button>
    </div>

    <h2>一、四版完整成片对比</h2>
    <div class="grid-2">
      {"".join(cards_html)}
    </div>

    <h2>二、20 镜抽卡池（Seeds &amp; FL2VA / I2VA / 73f）与入出帧精剪台</h2>
    <div id="slots-container" class="slots-grid"></div>

    <h2>三、导出的自定义 EDL JSON 预览</h2>
    <pre id="edl-output">点击上方“导出当前自定义挑卡 EDL JSON”生成可交由 assemble_sub04.py 直接渲染的 EDL。</pre>
  </main>
  <script>
    const SLOTS = {slots_json};
    const MASTERS = {masters_json};
    const COMP = {comp_json};
    const container = document.getElementById('slots-container');
    const state = SLOTS.map(s => ({{
      slot_id: s.slot_id,
      selected_job: s.selected_job,
      in_frame: parseInt(s.schedule_span.split('..')[0], 10) || 0,
      out_frame: parseInt(s.schedule_span.split('..')[1], 10) || 38,
      target_frames: s.target_frames
    }}));

    SLOTS.forEach((slot, idx) => {{
      const card = document.createElement('div');
      card.className = 'slot-card';
      const rows = slot.candidates.map(c => {{
        const isSel = c.job_id === slot.selected_job ? 'selected-row' : '';
        return `<tr class="${{isSel}}">
          <td><code>${{c.job_id}}</code></td>
          <td>${{c.mode}}</td>
          <td>${{c.seed}}</td>
          <td>${{c.schedule_span}}</td>
          <td>${{c.seam_mae_255}}</td>
          <td>${{c.active_motion_255}}</td>
          <td>${{c.out_anchor_mae_255}}</td>
          <td><strong>${{c.combined_cost}}</strong></td>
        </tr>`;
      }}).join('');
      const opts = slot.candidates.map(c =>
        `<option value="${{c.job_id}}" ${{c.job_id === slot.selected_job ? 'selected' : ''}}>${{c.job_id}} (${{c.mode}}, seam=${{c.seam_mae_255}})</option>`
      ).join('');
      card.innerHTML = `
        <h4>${{slot.slot_id}} · ${{slot.title}} <span class="badge">${{slot.keyframes}} (${{slot.target_frames}}f)</span></h4>
        <div class="metrics">
          <span>DP最优卡: <code>${{slot.selected_job}}</code></span>
          <span>选帧区间: <code>${{slot.schedule_span}}</code></span>
          <span>接缝MAE: <strong>${{slot.seam_mae_255}}</strong></span>
        </div>
        <div style="display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap;">
          <label>挑卡: <select data-idx="${{idx}}" class="sel-job">${{opts}}</select></label>
          <label>入点帧: <input type="number" min="0" max="70" value="${{state[idx].in_frame}}" data-idx="${{idx}}" class="in-f" style="width:52px"/></label>
          <label>出点帧: <input type="number" min="1" max="72" value="${{state[idx].out_frame}}" data-idx="${{idx}}" class="out-f" style="width:52px"/></label>
        </div>
        <table>
          <thead><tr><th>候选 Job</th><th>模式</th><th>Seed</th><th>DP帧段</th><th>接缝MAE</th><th>有效运动</th><th>尾锚点差</th><th>综合代价</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
      container.appendChild(card);
    }});

    document.querySelectorAll('.sel-job').forEach(el => {{
      el.addEventListener('change', e => {{ state[e.target.dataset.idx].selected_job = e.target.value; }});
    }});
    document.querySelectorAll('.in-f').forEach(el => {{
      el.addEventListener('change', e => {{ state[e.target.dataset.idx].in_frame = parseInt(e.target.value, 10); }});
    }});
    document.querySelectorAll('.out-f').forEach(el => {{
      el.addEventListener('change', e => {{ state[e.target.dataset.idx].out_frame = parseInt(e.target.value, 10); }});
    }});
    document.getElementById('export-edl-btn').addEventListener('click', () => {{
      const out = {{
        name: "custom_director_cut",
        target_total_frames: state.reduce((a, b) => a + b.target_frames, 0),
        parts: state.map(s => ({{
          slot: s.slot_id,
          job_id: s.selected_job,
          start_frame: s.in_frame,
          end_frame_exclusive: s.out_frame + 1,
          target_frames: s.target_frames
        }}))
      }};
      document.getElementById('edl-output').textContent = JSON.stringify(out, null, 2);
    }});
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edls", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    out_dir = EXP / "outputs" / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [audit_master(p.resolve(), out_dir) for p in args.edls]

    by_name = {r["name"]: r for r in results}
    if "workflow_a_fl2va39_raw" in by_name and "director_master_30s00" in by_name:
        m_a = EXP / "outputs" / "masters" / "workflow_a_fl2va39_raw" / "master_30s00.mp4"
        m_d = EXP / "outputs" / "masters" / "director_master_30s00" / "master_30s00.mp4"
        sbs_a = EXP / "outputs" / "sub04_workflow_a_vs_director.mp4"
        build_side_by_side(
            m_a,
            "Workflow A: Seed42 未修剪均匀抽帧",
            m_d,
            "Director Master: 多Seed + 单调DP精剪",
            720,
            sbs_a,
        )
    if "workflow_b_hybrid_i2va_73f" in by_name and "director_master_30s00" in by_name:
        m_b = EXP / "outputs" / "masters" / "workflow_b_hybrid_i2va_73f" / "master_30s00.mp4"
        m_d = EXP / "outputs" / "masters" / "director_master_30s00" / "master_30s00.mp4"
        sbs_b = EXP / "outputs" / "sub04_workflow_b_vs_director.mp4"
        build_side_by_side(
            m_b,
            "Workflow B: I2VA + 73f跨拍长窗混合",
            m_d,
            "Director Master: 39f FL2VA + 单调DP精剪",
            720,
            sbs_b,
        )

    # Render a 15-point 3-way comparison image (Workflow A vs Workflow B vs Director Master)
    if all(k in by_name for k in ("workflow_a_fl2va39_raw", "workflow_b_hybrid_i2va_73f", "director_master_30s00")):
        fa = decode(EXP / "outputs" / "masters" / "workflow_a_fl2va39_raw" / "master_30s00.mp4")
        fb = decode(EXP / "outputs" / "masters" / "workflow_b_hybrid_i2va_73f" / "master_30s00.mp4")
        fd = decode(EXP / "outputs" / "masters" / "director_master_30s00" / "master_30s00.mp4")
        sample_pts = np.linspace(18, 702, 15).round().astype(int).tolist()
        w, h = 420, 236
        canvas = Image.new("RGB", (3 * w, 50 + len(sample_pts) * (h + 26)), "#0c121a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14, 12),
            "Sub04 三大工作流逐拍横向对比 (左: Workflow A 未修剪 | 中: Workflow B I2VA+73f混合 | 右: Director 39f FL2VA+DP精剪)",
            fill="white",
            font=load_font(20),
        )
        for r_idx, f_idx in enumerate(sample_pts):
            y0 = 50 + r_idx * (h + 26)
            for c_idx, (tag, seq) in enumerate(
                [("Workflow A (Raw)", fa), ("Workflow B (I2VA+73f)", fb), ("Director (FL2VA+DP)", fd)]
            ):
                im = Image.fromarray(cv2.cvtColor(seq[f_idx], cv2.COLOR_BGR2RGB)).resize(
                    (w, h), Image.Resampling.LANCZOS
                )
                x0 = c_idx * w
                canvas.paste(im, (x0, y0))
                draw.text(
                    (x0 + 8, y0 + h + 3),
                    f"{tag} | f{f_idx:03d} ({f_idx / 24:.2f}s)",
                    fill="#9ad1eb",
                    font=load_font(15),
                )
        canvas.save(out_dir / "workflow_comparison_30s.jpg", quality=90)

    gacha_path = out_dir / "gacha_and_dp_report.json"
    gacha_report = json.loads(gacha_path.read_text()) if gacha_path.exists() else {}
    render_workbench_html(results, gacha_report, out_dir)

    receipt = {
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
        "masters": results,
    }
    (out_dir / "acceptance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "masters": len(results), "workbench": "outputs/review/index.html"}))


if __name__ == "__main__":
    main()
