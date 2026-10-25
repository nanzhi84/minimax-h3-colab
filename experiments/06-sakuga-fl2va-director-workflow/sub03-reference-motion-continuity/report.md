<!-- harness-ignore LNK001: outputs/ 和 artifacts/ 是忽略入库、由本实验重建的本地验收工件 -->
# 子实验 3：wuy0uer 参考片与前两版的动作连续性复盘

日期：2026-10-24。结论范围：本报告记录0 CU本地参考拆解阶段。随后获得30 CU授权的实际H3-FL2VA复现、候选与成片结论另见[生成实验报告](generation-report.md)，不回写为本阶段的执行结果。

## 1. 结论

前两版解决了一部分人物外观和构图变化问题，但没有建立持续可读的动作路径。人物较早到达目标姿势后，画面仍可通过雷电、烟雾和相机变化获得很大的帧差与光流，观众看到的身体动作却已经停下。继续增加焦段变化、镜头翻转和特效，不能直接解决这个问题。

参考片主要展示单人连续动作、刀锋弧线、头发与衣摆的延迟跟随，镜头通过跟拍、环绕和前景遮挡组织这些运动。值得学习的是动作与镜头怎样配合，以及速度怎样变化。它与当前双人交锋、刚体接触、2D 线稿和多段 FL2VA 拼接的任务不同，不能拿两种成片的总分证明模型能力高低。

## 2. 下载来源与已知边界

- [原作品](https://www.douyin.com/video/7664781155287239972)，作者 `wuy0uer`；[下载原片](outputs/reference/wuy0uer_7664781155287239972.mp4)由内置浏览器当前可见的 video 元素保存，没有再压制。
- 媒体：HEVC + AAC，1024×576，30fps，895 帧，29.833333 秒，3,067,658 字节。
- SHA-256：`ffde3a774c44e3a0f1e5e58dafd32148c3966e4ee2ad76d3b122882ccac42208`。
- 作者标题提到“两张图”；可见作者回复提到抽卡。[页面截图](outputs/reference/page_author_evidence.png)和[公开信息摘录](code/study.json)保留了证据。
- 这不足以确认两图是一组首尾帧，也可能是两个片段各用一张起始图；具体生成模型、提示词、续生成、重抽次数、变速和插帧均未得到作者确认。

从画面可以概括为“两个约 15 秒的动作段落”，不能据此断言是两个 15 秒生成任务或一镜到底。可测黑场是 `14.9333–15.0667s`（4 帧）和 `29.1333–29.8333s`（21 帧）；第二段第一帧为 `f452 / 15.0667s`。切换处先冲击、白亮变化，再黑场、换景，允许动作重新开始；它没有要求两段在同一姿势上无缝接力。

## 3. 按可见动作拆参考片

先看[逐秒联系表](outputs/analysis/reference_overview.jpg)，再在[动作观察台](outputs/analysis/review.html)正常速度和半速查看；时间范围是分析窗口，不是声称存在这些剪辑点。

| 窗口 | 可见画面与注意力路径 | 应学习的具体做法 |
| --- | --- | --- |
| 0–2.5s | 月亮与人物先建立轮廓；人物横向掠行，经过立柱附近的水光，再转身划出刀光圆弧 | 先建立方向，随后加速；身体、剑尖、拖尾有共同的运动意图。[8Hz 窗口](outputs/analysis/reference_motion_0.5s.jpg) |
| 2.5–8.5s | 人物腾跃，画幅随之俯仰、旋转，柱体与长条光轨提供纵深线索 | 镜头运动跟随动作结果，保留可追踪的主体；不等同于真实三维几何已经被验证 |
| 8.5–10.5s | 人物接近镜头、回身，发梢与衣摆延迟摆动，随后出现大范围亮光 | 用轮廓、弧线、拖尾和明暗变化引导视线。[8Hz 窗口](outputs/analysis/reference_motion_8.5s.jpg) |
| 10.5–15.0667s | 横向推进、近景衣摆掠过，最后冲击亮光与黑场进入下一段 | 大幅模糊、遮挡和黑场可掩盖难以直接匹配的状态；不能推断具体后期做法 |
| 15.0667–20.5s | 暖色红叶场景，剑尖、身体转向与相机距离持续变化 | 单人剑舞比双人接触更容易把注意力集中在一条动作线上。[8Hz 窗口](outputs/analysis/reference_motion_18.5s.jpg) |
| 20.5–29.1333s | 转身、衣摆、红叶回旋逐渐形成大的环形构图，最终近景叶片占据画面 | 大动作与衣摆、环境叶片以不同时间展开；从局部转动扩展为环境中的圆弧。[8Hz 窗口](outputs/analysis/reference_motion_23.0s.jpg) |

片中仍有快速遮挡、模糊和不容易精读的身体/兵器形变。“整体观感顺”不等于每一帧都物理正确；本轮也未完成音轨听感与节拍人工验收。

## 4. 为什么前两个子实验仍然不丝滑

### 4.1 目标姿势到得早，动作却没有继续发展

[子实验 1 逐秒图](outputs/analysis/sub01_overview.jpg)中，前段冲刺后较长时间维持碰撞姿态，眼部特写也主要依靠电光变化。[子实验 2 逐秒图](outputs/analysis/sub02_overview.jpg)约 `1.5–4.5s` 的身体与双刃相对姿态变化仍较小，虽然构图和雷电继续动。这是画面复核，不是人物骨骼测量。

这支持“动作阶段分配不足、关键姿势停留过长”的诊断。是否由首尾约束、动作数量、采样或模型偏好造成，尚需控制变量生成；不能从一批成片直接认定唯一原因。

### 4.2 相同图片只处理位置，无法规定入场与离场动势

两版 EDL 都在 `f84 / 3.5000s`、`f185 / 7.7083s`、`f245 / 10.2083s` 切换片段。[子实验 2 接缝图](outputs/analysis/sub02_seams.jpg)显示：共享节点附近的画面接近，但动作阶段仍可能重复；特写转回全身后也需要重新建立动势。

`A→B` 与 `B→C` 分别生成，即使两端都接近 B，也没有共享上一段的运动历史。位置近似连续，不保证速度、方向和加速度连续。删除重复首帧可以缩短停顿，不能保证连续性。人工精剪又在末段第 2、3 个局部帧注入黑白闪帧，其绝对位置是 `f247–248 / 10.2917–10.3333s`；这会产生很大帧差，也应在动作真正命中时才使用，不能作为通用接缝修复。

### 4.3 首尾差异过大，并不自动产生合理运镜

子实验 2 的设计同时要求跨景别、俯仰变化、焦段变化、角色动作、受击与特效。它扩大了状态跨度，却没有保证中间是一条可解释的路径。画面透视改变也可能包含形变，不能仅用边缘 MAE 或稠密光流证明真实视差、镜头角度或动量守恒。

对前序总览中的机制表述作如下更正：背景起止位置相同，只限制净位移，不意味着中途光流必须为零；镜头可以移动后返回。起止背景不同，也不强制模型合成正确三维运动。“尾帧必须大跨度重构”“受击尾帧会自动反推出正确冲击过程”均不应当作已证明的物理规律。

### 4.4 现有评分衡量的内容不足以验收动作

[云端评分实现](../../../studio/api/remote_batch_engine.py)的 `motion_smoothness_score` 使用相邻帧平均绝对差的变化率，没有追踪身体、剑尖或相机轨迹。稳定不动也可获得高分，故它只是帧差稳定性代理。

[现有选卡实现](../sub02-dynamic-perspective-parallax/code/director_cut.py)给运动幅度奖励，并按每个分镜独立排名，没有评估候选组合的接缝动势。各镜头分别得分最高，不意味着拼起来最好。

本轮在同一 320×180 分析尺寸复算，三片代理值为：参考 `54.90`、子实验 1 `62.47`、子实验 2 `61.87`。这些内容、帧率、剪辑均不相同，**不能作为流畅度排名**；它们只是直观说明，这个公式无法回答用户的主观问题。也不能把此次完整成片复算值与前序单镜头原分辨率分数混用。

## 5. 需要掌握哪些术语

需要学少量术语来明确表达意图，尤其是动画动作原则。术语的作用是约束“谁在何时怎样动”，不是把许多镜头词叠在提示词里。

| 中文 / 术语 | 在本实验里如何落实 |
| --- | --- |
| 调度、动作可读性 / Blocking, Staging | 先明确脚和重心去哪里、剑走哪条线、画面主要看谁；角色轮廓不要长时间埋在粒子里 |
| 预备动作 / Anticipation | 发力前有小幅重心下压、肩髋蓄力，让后续加速有起点 |
| 节奏、位移分配 / Timing, Spacing | 交代蓄力、加速、最快点、收势；连续不等于全程匀速，也不等于全程很快 |
| 弧线 / Arcs | 手、剑尖和衣摆沿可读弧线移动，不在关键姿势之间直接融化或直线滑动 |
| 跟随与重叠动作 / Follow-through, Overlapping action | 躯干先转，剑、发梢、衣摆有不同拖尾；停止之后余势逐步消失 |
| 跟拍、环绕 / Tracking shot, Arc shot | 镜头跟主体位移或绕主体运动，先选一种主运镜；焦段数字只能表达倾向，不能据此验出真实毫米数 |
| 银幕方向、轴线 / Screen direction, 180° rule | 跨镜头保留左右关系和运动方向；改变方向要通过可见转身、跨轴运动或明确切换解释 |
| 动作匹配剪辑 / Match on action | 在同一动作进程里匹配姿态、方向和速度，再切镜；不是仅对齐同一张图片 |
| 遮挡、运动模糊 / Occlusion, Motion blur | 用柱、衣摆、叶片和速度产生的模糊组织转场；它们不能恢复缺失的动作或正确人体 |

动作原则参考 [Adobe 动画十二原则](https://www.adobe.com/creativecloud/animation/discover/principles-of-animation.html)；空间连续性参考 [Adobe 轴线说明](https://www.adobe.com/creativecloud/video/discover/what-is-the-180-degree-rule.html)；动作匹配参考 [Adobe match cut](https://www.adobe.com/creativecloud/video/discover/match-cut.html)。这些是解释和设计工具，不是参考作者制作过程的证据。

## 6. 下一轮工作流：先让一个动作成立

1. **先画动作路线。** 一个角色、同一场景、可辨认的持剑手。写出“压低重心 → 滑步转身横扫 → 身体减速 → 发梢/衣摆继续 → 收势”，而非先列四个大招。
2. **制作起始帧 S0 和相近机位的结束帧 S1。** S0 保留腿部与持剑手、预留挥剑空间；S1 只完成同一动作，不突然变成眼部微距或高空鸟瞰。两张均为下一轮待制作的独立资产，原参考片只用于观察。
3. **约 6 秒的小对照。** 固定 `1344×768 / 24fps / 141f = 5.875s / 8 steps / Run 12`，Seed `42, 777`；I2VA 用 S0，FL2VA 用 S0+S1。两组使用同一个动作正文与主运镜，仅调整模式必需的对齐说明和结尾约束，共 4 条。结果差异仍需同时观察两个 Seed，不能凭一次抽卡下结论。
4. **写清时序，但不把它当精确控制器。** 大致预备 `0–0.6s`、滑步与横扫 `0.6–3.4s`、减速与拖尾 `3.4–4.8s`、收势 `4.8–5.875s`；这些是导演目标，输出必须另行核验。摄影只用小幅跟拍，先固定画幅与主运动方向。
5. **按动作盲选。** 静音 1× 先看是否自然，然后 0.5× 查穿插、握剑手漂移、滑步、意外停顿；标记失败时间和理由。保留完整候选，暂不用合成总分自动定胜者。
6. **再加入复杂度。** 单人动作通过后，分别增加机位跨度、第二个人物、接触点、跨段剪辑。拼接时在多帧动作窗口选接点，并联合选择相邻 take；不强求每段都经过同一张静止 B 图。

本仓库已支持通过 H3-Base-FL2VA checkpoint 做仅首帧的 I2VA，见[条件传入代码](../../../studio/api/remote_batch_engine.py)，本轮无需更换模型或改生产调度架构。模型实际能力与两种条件的结果差异仍待这组运行验证。

### H3 提示词也需要校正

[MiniMax 官方 base-mode 指南](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)要求模式对应的图像对齐说明，再组织 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music`；运镜写在自然语言动作中。前两版的 `[Push in]`、`[Arc shot]` 不能视为拥有额外强制控制能力的特殊指令。

下一轮三字段正文可从以下简短版本开始；生成前还需根据模式补上官方对应的图像对齐说明，FL2VA 结尾显式到达 S1。它是待验证的提示词设计，不是原作者提示词：

```text
integrated_multimodal_description: [Shot 1] In one continuous medium-wide shot, the swordswoman lowers her center of gravity, takes one sliding step to screen right, and turns her hips and shoulders into a single horizontal sword sweep. Her sword tip follows one clear arc. The camera tracks right with small amplitude, keeping her full body readable. Her torso slows first; her hair and robe continue along the turn before settling into a balanced stance.
overall_soundscape: One sliding footstep, a sword swish at the sweep, and cloth rustling that fades after the body settles. Quiet courtyard ambience.
non_diegetic_music: N/A
```

此外，当前生产 schema 虽接受 `negative_prompt`，但[实际条件编码](../../../studio/api/remote_batch_engine.py)没有传入该字段，运行使用 BasicGuider；前两版写入的负向词不能当作已生效的约束。此处只记录诊断，本轮不修改生产接口。新的对照应优先用正面动作描述，也不能把“加负向词后不动了”当成有效消融。

## 7. 本轮实际验收

三片共 `895 + 360 + 360 = 1615` 帧完整解码；逐帧 PTS 确认参考 30fps、前两版 24fps；两份 EDL 均核对到 360 帧。固定依赖版本，输出原生帧时间的联系表、接缝窗口、CSV、来源哈希和环境回执。

- [本地观察台](outputs/analysis/review.html)：三片独立播放、章节定位、1×/0.5×；默认通过本地 HTTP 服务打开。
- [浏览器交互回执](outputs/analysis/browser_acceptance.json)与[观察台截图](outputs/analysis/observation_desk.png)：已实际核对三片定位到 `1.50 / 3.25 / 7.45s`，参考片 0.5× 播放进度正常前进，三片媒体均无错误。
- [可重复运行的分析脚本](code/analyze_motion.py)和[输入/取样定义](code/study.json)。
- [验收回执](outputs/analysis/acceptance.json)，同内容写入 `artifacts/exp06-sub03-motion-review/acceptance.json`；命令与前提见 [README](README.md)。
- `harness doctor`、新增代码 Ruff 与 Pyright 检查通过；没有补写实现镜像式单元测试。

本轮没有分配 GPU，未生成新的 H3 视频。上述 PASS 只覆盖本地分析工件；不覆盖新方案的成片质量、参考作者真实流程、音乐听感或 yoryon 对“丝滑”的最终认可。
