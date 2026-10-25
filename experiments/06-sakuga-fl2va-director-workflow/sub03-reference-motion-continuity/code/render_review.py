"""Render a local video observation desk without external requests or uploads."""

import html
import os
from pathlib import Path


def render_review(study: dict, out: Path, base: Path) -> None:
    cards = []
    for spec in study["clips"]:
        source = html.escape(os.path.relpath(base / spec["path"], out), quote=True)
        chapters = "".join(
            f'<button data-seek="{float(time)}" disabled>{time:g}s · {html.escape(label)}</button>'
            for time, label in spec["chapters"]
        )
        cards.append(f'''<article>
<h2>{html.escape(spec['title'])}</h2>
<video controls muted playsinline preload="metadata" src="{source}"></video>
<p class="media-state" role="status" aria-live="polite">正在读取本地视频…</p>
<button class="retry" hidden>重新读取视频</button>
<p>原生 {spec['expected_fps']}fps · 以主体动作与方向判断；三片内容不同。</p>
<div class="chapters">{chapters}</div>
<label>播放速度 <select aria-label="{html.escape(spec['title'])}播放速度">
<option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label>
<p><a href="{spec['id']}_overview.jpg">逐秒画面</a> · <a href="{spec['id']}_seams.jpg">接缝窗口</a></p>
</article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 06 · 子实验3动作观察台</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b111b;color:#e5edf8}
body{max-width:1500px;margin:0 auto;padding:32px;line-height:1.65}
h1{font-size:30px;margin:8px 0}h2{font-size:20px}p{color:#b5c4d8}
.label{color:#70d2df;font-size:13px;letter-spacing:2px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
article,.note{background:#131e2e;border:1px solid #26394f;border-radius:12px;padding:20px}
video{width:100%;aspect-ratio:16/9;background:#000}.chapters{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
button,select{font:inherit;background:#22374b;border:1px solid #45617c;color:#e6f6ff;padding:8px;border-radius:6px;cursor:pointer}
button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #71e4d5;outline-offset:3px}a{color:#81d9ed}
.note{margin:24px 0}.terms{display:grid;grid-template-columns:repeat(2,1fr);gap:0 30px}li{margin:12px 0}
@media(max-width:1000px){.grid,.terms{grid-template-columns:1fr}body{padding:16px}}
</style>
<div class="label">EXPERIMENT 06 / SUB03 / LOCAL REFERENCE STUDY</div>
<h1>丝滑来自动作路径与节奏，也来自镜头怎样跟随它</h1>
<p>先在静音、1×速度看动作是否接得上，再半速查看身体、剑、衣摆和背景。每段单独播放，避免把不同内容当成严格 A/B。</p>
<div class="grid">''' + "".join(cards) + '''</div>
<div class="note"><strong>已观察到的差异</strong>
<p>参考片以单人身体转动、刀锋弧线、衣摆和叶片的延迟运动维持连续性；约15秒处有明确换段。前两版有较长的目标姿势停留，特效继续变化。更大光流和共享关键帧均不能直接证明动作顺畅。</p>
<p>作者只公开说“两张图”，并在可见回复中提到抽卡。具体模型、两图用途、延长及插帧流程尚未证实。</p></div>
<div class="terms">
<section><h2>先学动作，再学运镜</h2><ul>
<li><strong>Blocking / Staging · 调度与动作可读性：</strong>人往哪里走、剑划哪条线、观众先看哪里。</li>
<li><strong>Anticipation · 预备动作：</strong>重心先压低或肩髋先蓄力，再发力。</li>
<li><strong>Timing / Spacing · 节奏与位移分配：</strong>不要从起到落匀速滑动，也不要每到关键帧就重启。</li>
<li><strong>Arcs / Follow-through · 弧线与拖尾：</strong>手、剑、发梢、衣摆按不同时间经过连续轨迹。</li>
</ul></section>
<section><h2>让镜头服务同一个动作</h2><ul>
<li><strong>Tracking / Arc · 跟拍与环绕：</strong>先试一种主运镜；镜头的速度和主体变化要有理由。</li>
<li><strong>Screen direction · 银幕方向：</strong>跨镜头保留动作朝向和视线落点，必要时用可见转身解释改变。</li>
<li><strong>Match on action · 动作匹配剪辑：</strong>匹配动作阶段、方向和速度，不能只对上同一张图片。</li>
<li><strong>Occlusion · 遮挡：</strong>柱子、衣摆和近景叶片可组织景深与转场；遮挡不能自动修复形体。</li>
</ul></section></div>
<p><a href="acceptance.json">环境、输入哈希与断言回执</a> · 所有媒体均在本地读取；本页面不调用外部服务。</p>
<script>
document.querySelectorAll('article').forEach(card=>{
 const video=card.querySelector('video');
 const state=card.querySelector('.media-state'),retry=card.querySelector('.retry');
 const ready=()=>{state.textContent='可播放 · '+video.duration.toFixed(3)+' 秒';retry.hidden=true;
  card.querySelectorAll('[data-seek]').forEach(button=>button.disabled=false)};
 video.addEventListener('loadedmetadata',ready);
 if(video.readyState>=1)ready();
 video.addEventListener('waiting',()=>state.textContent='正在缓冲…');
 video.addEventListener('canplay',ready);
 video.addEventListener('seeked',()=>state.textContent='已定位 '+video.currentTime.toFixed(3)+' 秒');
 video.addEventListener('error',()=>{
  state.textContent='视频读取失败，请确认本地服务和输入文件后重试。';retry.hidden=false;
  card.querySelectorAll('[data-seek]').forEach(button=>button.disabled=true);
 });
 retry.addEventListener('click',()=>{state.textContent='正在重新读取…';video.load()});
 card.querySelectorAll('[data-seek]').forEach(button=>button.addEventListener('click',()=>{
  video.pause(); video.currentTime=Number(button.dataset.seek);
 }));
 card.querySelector('select').addEventListener('change',event=>{
  video.playbackRate=Number(event.target.value);state.textContent='播放速度 '+video.playbackRate+'×';
 });
 video.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause()}));
});
</script></html>'''
    (out / "review.html").write_text(document, encoding="utf-8")
