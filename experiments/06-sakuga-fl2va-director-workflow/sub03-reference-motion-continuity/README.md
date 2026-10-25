# 子实验 3：从 wuy0uer 参考片研究动作连续性

状态：参考视频下载、本地拆解与前两版复核完成；30 CU授权下的七批（round01–round03、round04a/b、round05、round06）共126条H3候选和两条36镜逐拍对齐完整导演版已完成，GPU已停止。生成结果、选片及费用集中保存在[生成实验报告](generation-report.md)，整个Experiment 06的结论见[工作流总结](../workflow-summary.md)。

直接查看[24.88秒节奏强化导演版](outputs/sub03_director_master_tight_24s88.mp4)、[29.83秒等长导演版](outputs/sub03_director_master_29s83.mp4)、[原片对照](outputs/sub03_reference_vs_director.mp4)或[同步审片台](outputs/reproduction/final_review/index.html)。基于38个原片关键帧与亚拍锚点（0.20s–1.00s间隔）、39帧（`k=2`）原生短窗口多Seed抽卡（使用 `h3-prompt-writing` 技能撰写提示词）与参考片引导的单调动态规划（Monotonic DP）逐帧选帧，全片36镜与原片逐拍高度对齐（全片平均参考MAE `24.90/255`，34处跨镜接缝平均步进MAE仅 `2.50/255`）。

## 参考研究阶段的目标与边界

回答“为什么运动更多了，还是不丝滑”，把动作编排、速度变化、镜头调度和接缝连续性转成下一轮可检验的设计。参考研究阶段输入为用户指定的[抖音作品](https://www.douyin.com/video/7664781155287239972)及子实验 1、2 的实际成片与 EDL，仅进行0 CU本地分析。随后用户另行授权30 CU实际复现，生成数据流、预算和命令见[生成执行计划](reproduction-plan.md)及[生成实验报告](generation-report.md)。

分析工具只负责媒体校验、按原始帧率逐帧读取、生成联系表及诊断数值。摄影语言和动作可读性由逐段画面复核判断；帧差和光流不能自动证明三维几何、受力真实性或主观流畅度。实验数据与衍生图片全部写入本目录的 `outputs/`，运行回执写入仓库 `artifacts/`。

## 编码前列明的失败方式与边界

1. 输入缺失、参考片哈希不符、解码截断：明确失败，禁止用其他视频或补帧代替。
2. 可变帧率、探测与实际解码帧数不符：停止数值分析，不能用平均帧率伪造时间轴。
3. 把闪光、烟雾、镜头运动的光流当成人物动作：报告只称“图像运动代理”，保留人工动作复核。
4. 两段共享一张图片，但出入速度、方向或动作阶段不同：检查接缝前后窗口，不能只删除一帧后宣布无缝。
5. 将剪辑、插帧或生成模型猜成作者的实际工具：明确区分页面公开信息、可见画面和待验证推测。
6. 参考片 30fps，前两版 24fps，内容和风格也不同：分别使用原生帧率；不输出跨片“流畅度排行榜”。
7. 来源页面包含登录态和带签名媒体地址：只留作品公开链接、作者公开说明、媒体哈希和必要画面，不保存 Cookie、签名直链或账号凭据。
8. 黑场、近似重复帧、特效闪帧会影响统计：标注其时间与用途，不能简单判成播放故障。

## 参考研究提出的单人条件对照

先做约 6 秒的单人转身挥剑，以同一人物、场景、时长和成对 Seed 对照 I2VA（仅首帧）与 FL2VA（首尾帧）。每镜只保留一个主要动作和一种主运镜；首尾帧留在可解释的同一空间。先验证蓄力、发力、刀锋弧线、衣摆拖尾和收势，再增加第二个人物和碰撞。

后续逐项增加“大跨度机位”“第二名角色”“跨段剪辑”，每次只增加一个因素。原片作为运动和构图参考，不把“两张图”解释为已经证实的首尾帧控制，也不假定参考作者使用 H3。通过人工盲看后才扩展到 15 秒多镜头成片。

这组I2VA/FL2VA条件消融仍是独立的待验证方案；本次追加授权执行的是指定原片的FL2VA复现，没有把该方案算作已运行。

## 参考研究阶段的验收范围

- 本轮完成条件：原片可解码且来源可追溯；两版真实成片均已复核；输出可重建的联系表、逐帧数据、接缝窗口、环境与断言回执；报告写清术语、问题和下一轮对照变量。
- 后续完成条件：实际生成 I2VA/FL2VA 对照、静音正常速度与半速盲看，核对形体、动作阶段、方向和接缝，再由 yoryon 判断“丝滑”。本轮不把这些尚未执行的项算作通过。

## 重建参考研究工件与打开观察台

前提：在仓库根目录运行，安装 `uv`、Python 3.14、`ffprobe`；保留 `code/study.json` 指定的三个视频和两份 EDL。参考片需要从原作品下载且匹配已记录的 SHA-256；脚本不自动使用浏览器凭据。首次运行需从包仓库获取脚本头部锁定的 NumPy、OpenCV、Pillow 版本。

```bash
harness run exp06-sub03-motion-review -- \
  uv run "$PWD/experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/analyze_motion.py"
```

该命令读取真实输入，逐帧核对 PTS 和解码帧数，生成 `outputs/analysis/`，并写入 `artifacts/exp06-sub03-motion-review/acceptance.json`；Harness 另在当次 `artifacts/<run-id>/` 留下 stdout、stderr 与执行回执。输入缺失或不符会以非零退出，运行开始先把旧成功回执置为 RUNNING，避免失败后误用上次 PASS。

通过支持 HTTP Range 的静态服务打开观察台（Node.js/npm 为此步骤的额外前提）：

```bash
npx --yes http-server@14.1.1 experiments/06-sakuga-fl2va-director-workflow \
  -a 127.0.0.1 -p 8767 -c-1
```

然后访问[本地动作观察台](http://127.0.0.1:8767/sub03-reference-motion-continuity/outputs/analysis/review.html)。服务只绑定本机；按 Ctrl-C 停止。通过 HTTP Range 支持验证章节跳转，不能只以页面或视频元数据加载成功作为验收。原始视频帧率保留，半速播放不生成插帧媒体。

```bash
harness doctor
uvx ruff check experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code
uv run --with pyright --with numpy==2.5.4 --with opencv-python-headless==5.0.0.93 \
  --with pillow==13.0.0 pyright \
  experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code
```

本轮环境与断言以生成的验收回执为准。`outputs/`、`work/` 和 `artifacts/` 均忽略入库；仅清理这些目录会移除媒体和证据，重建需要上述原始输入。
