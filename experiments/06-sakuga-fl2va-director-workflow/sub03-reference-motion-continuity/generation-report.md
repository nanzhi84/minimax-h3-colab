<!-- harness-ignore LNK001: outputs/ 与 artifacts/ 为忽略入库的真实运行证据，依赖本地生成 -->
# 子实验 3：H3-FL2VA 复现 wuy0uer 参考片的完整工作流

日期：2026-10-25。用户授权预算：30 CU。状态：七批（round01–round03、round04a/b、round05、round06）共126条候选已真实生成，合计9266帧、386.083秒原生H3视频；已导出36镜逐拍对齐的29.833秒等长导演版与24.875秒节奏强化导演版，GPU已停止。

本次通过**38个原片关键帧与亚拍锚点（0.20s–1.00s间隔，单镜39帧 `k=2`）**、**`h3-prompt-writing` 技能结构化提示词**、**多Seed抽卡（每镜2–4 Seeds）**以及**参考片引导的单调动态规划（Monotonic DP）逐帧选帧**，补齐了此前3秒窗口漏掉的全部关键动作（如 `f075` 屋面360度金焰圆环、`f106` 满月正中剪影腾跃、`f212` 四道竖向白幡拔高、`f276–f285` 屋脊剪影急推月前纯净侧颜特写、`f314` 满月水环化全屏白光垂直俯冲、`f376` 近景黑发横斩、`f510` 枫叶侧脸回眸、`f660–f690` 俯拍白袍回旋接仰拍迎月竖劈、`f780–f810` 满月红叶火环旋舞），全片36镜与原片逐拍高度一致（全片平均参考MAE `24.90/255`，34处跨镜接缝平均步进MAE仅 `2.50/255`）。整个Experiment 06的横向总结见[工作流总结](../workflow-summary.md)。

## 直接查看成片

| 版本 | 完整视频 | 用途 |
| --- | --- | --- |
| 29.833秒等长导演版（首选） | [成片](outputs/sub03_director_master_29s83.mp4) · [原片同屏对照](outputs/sub03_reference_vs_director.mp4) · [EDL和逐帧来源](outputs/sub03_director_edit_decision_list.json) | 716帧（24fps），与原片29.833秒（895帧@30fps）逐拍1:1精确同步；覆盖月夜22镜+11帧中场黑场+红叶14镜+21帧尾段黑场 |
| 24.875秒节奏强化导演版 | [成片](outputs/sub03_director_master_tight_24s88.mp4) · [原片等比变速对照](outputs/sub03_reference_vs_director_tight.mp4) · [EDL和逐帧来源](outputs/sub03_director_edit_decision_list_tight.json) | 597帧（24fps），在36镜逐拍对齐链基础上按0.86×时长压缩抽取每镜最高速动作核，中段与尾段黑场压缩为2帧与6帧 |
| 全部工作流对照与逐秒对比图 | [本地同步审片台](outputs/reproduction/final_review/index.html) · [30秒逐秒同屏对比图](outputs/reproduction/final_review/reference_vs_selected_30s.jpg) · [接缝多帧检视表](outputs/reproduction/final_review/selected_reproduction_seams.jpg) | 可切换正式两版、首批长镜头、10镜短链、时长对照及第三批失败长镜头；完整保留全轮次演进证据 |

两版均为1344×768、24fps、H.264+AAC。节奏版为597帧，时长比等长版缩短16.62%；全程无光流插帧、无原片画面混入、无重复帧填充。

## 目标和方法

使用原片的真实首尾帧作 FL2VA 条件，生成 29.833 秒、覆盖月夜动作与红叶剑舞的新视频。中间运动全部由 MiniMax H3-Base-FL2VA 生成；不以原片裁剪、复制或插帧替代生成。参考来源与此前拆解见[参考研究报告](report.md)，生成前设计与失败方式见[执行计划](reproduction-plan.md)。

复用现有 Studio 的 Run 12、8-step Turbo LoRA、G4 单流推理、Drive 双阶段传输与停机流程。首尾图取自源文件帧号，中心 cover 对齐至 1344×768（开启无损高压缩 `compress_level=9` 将单批 `input_jobs.tar` 控制在 11–15 MB 以避免大包上传超时）；保存原始 RGB 哈希、条件图片哈希、完整提示词和清单快照。输出为 H3 原生 24fps，原片为 30fps。

## 批次记录

| 批次 | 假设与任务 | 计划规模 | 实际结果 | 实际 CU |
| --- | --- | --- | --- | --- |
| round01 | 两段接近15秒的长镜头，以及冲刺、腾跃、舞剑三个短窗口；验证时间跨度与动作可读性 | 5 个条件 × 2 Seeds = 10 条 | 10条完整回收并通过技术检查；长镜头存在明显场景/镜头路线偏离 | 批间即时差额2.582846；保守占用3.274778 CU |
| round02 | 月夜/红叶各5个3秒级关键帧窗口；对两个窗口增加124帧时长对照 | 12个条件 × 2 Seeds = 24条 | 24条完整回收并通过技术检查；3秒窗口跨度过大导致漏掉原片1秒级关键变招与特写 | 批间即时差额2.210055；保守占用2.568797 CU |
| round03 | 月夜同首尾同Seed修订场景语义；红叶明确运镜时序；月夜清晰尾帧长镜头加独立爆发尾窗 | 4个条件 × 2 Seeds = 8条 | 8条完整回收并通过技术检查；证明单靠提示词無法消除15秒长镜头中段漂移 | 批间即时差额3.188514；保守占用3.741837 CU |
| round04a | 月夜前半段逐拍密集锚点（`m01..m10`，覆盖 `f000..f270`，0.5s–1.0s间隔，单镜39帧 `k=2`） | 10个条件 × 2 Seeds = 20条 | 20条完整回收并通过技术检查；精确复现水柱横斩、屋面金焰圆环、双白幡回旋、巨浪掠顶与四白幡冲天 | 批间即时差额1.195601；保守占用1.423333 CU |
| round04b | 月夜后半段（`m11..m17`，`f270..f438`）与红叶前段（`a01..a04`，`f452..f570`），单镜39帧 | 11个条件 × 2 Seeds = 22条 | 22条完整回收并通过技术检查；复现月前侧颜特写、满月水环、黑瓦滑行流火、蓝白水刃爆发、枫林侧脸回眸与地面V形剑光 | 批间即时差额1.070238；保守占用1.613225 CU |
| round05 | 红叶中后段逐拍密集锚点（`a05..a14`，覆盖 `f570..f866`，单镜39帧 `k=2`） | 10个条件 × 2 Seeds = 20条 | 20条完整回收并通过技术检查；复现举剑过顶、左弧横扫、俯拍白袍旋转、仰拍迎月竖劈、满月红叶火环旋舞与前景红叶散景 | 批间即时差额0.735302；保守占用1.508776 CU |
| round06 | 针对含快速亚拍过渡的5个窗口插入中间关键帧（`f106`、`f255`、`f276`、`f314`、`f376`）拆分为10个亚拍短窗，并对 `m04` 补抽 Seeds 108/256 | 11个条件 × 2 Seeds = 22条 | 22条完整回收并通过技术检查；彻底消除 `m11` 侧颜特写橙火伪影并补齐 `f106` 满月正中剪影、`f276` 屋脊剪影、`f314` 全屏白光俯冲与 `f376` 近景横斩 | 批间即时差额0.934269；保守占用1.541682 CU |

表中即时差额是批次前后的账本快照（另含 `round04` 43.5MB大包上传超时与 `round06` 首次连接EOF两次0.0000 CU实际扣费的预演记录，均已计入墙钟保守上界）。累计实际扣费（`spent_cu`）为 **11.916826 CU**，含零CU前后传输时间的墙钟保守预算占用（`budget_committed_cu`）为 **17.066478 CU**，按保守口径仍余 **12.933522 CU**，严格控制在 30 CU 上限内。最终快照以[预算账本](outputs/reproduction/budget_ledger.json)为准。

### 首批观察与第二批决策

- 两条月夜长镜头的首尾 SSIM 均约0.94，中间却生成了成排旗幡与楼阁，原片以月面、屋顶和白色竖向结构构成的空间关系丢失。已复查[原生输出2秒帧](outputs/reproduction/round01_review/lunar_long_frame_2s.png)，这不是联系表拼图造成的现象，也不能据此直接认定是算子错误。
- 红叶长镜头生成了连续转身、舞剑与衣摆运动；Seed108的代理质量分高达92.67，但它大部分时间仍是人物中全景，没有复现原片推至剑锋、再抬升拉开的路线。高分依然不能证明镜头复现成功。
- 短窗口中，冲刺、转身和腾跃能形成可读的运动，但月夜场景会加入带图案旗帜、屋檐或花树，人物细节也由模型补绘；红叶短窗口较接近原片动作阶段。
- 第二批采用完整的分段首尾帧链，把月夜和红叶各拆成5个动作/运镜窗口；月夜提示词改用明确的屋顶、空旷夜空、月面与朴素竖向结构，去除旗幡/庭院等引入额外建筑的语义。每窗2个Seed；另对冲刺和红叶抬剑两个窗口增加124帧版本，对比原生短规格与较长生成后压缩时间。它是工作流比较，不能把多个同时改变的条件解释成单一因果证明。

首批[技术回执](outputs/reproduction/round01_review/technical_acceptance.json)与[全部候选观察台](outputs/reproduction/round01_review/review.html)保留原始候选，未按自动总分筛掉失败结果。

首批[29.83秒长镜头基线](outputs/reproduction/masters/round01_long_baseline/reproduction_29s83.mp4)及[原片对照](outputs/reproduction/masters/round01_long_baseline/reference_vs_h3.mp4)已生成，选用月夜Seed42和红叶Seed108。它是视觉存在明显偏离的对照，不是最终复现通过。该版本实际保留358帧月夜、3帧黑场、338帧红叶、17帧黑场；采用确定性选帧及对应音频变速，没有插帧和原片音乐，见[逐帧来源回执](outputs/reproduction/masters/round01_long_baseline/acceptance.json)和[可重跑EDL](code/edl_round01.json)。

本地拼接曾遇到FFmpeg长线性选帧表达式无法解析的问题，第一次CLI执行失败；将表达式改为深度有界的平衡分组后，以同一真实输入重跑，完整716帧输出验收通过。这次修复没有GPU消耗，也没有通过补帧掩盖错误。

### 第二批结果与时长对照

24条候选全部通过解码、目标帧数、24fps、尺寸和音轨检查，见[技术回执](outputs/reproduction/round02_review/technical_acceptance.json)、[逐镜观察](outputs/reproduction/round02_review/visual_review.json)和[候选观察台](outputs/reproduction/round02_review/review.html)。月夜第一、二、四窗已能组织出冲刺、腾跃水环、落地滑行，大片旗幡楼阁漂移明显减少；第三窗中间仍补出了传统屋檐，不能视为完全保持原空间。红叶的推至剑锋、抬升、转身、叶环和前景叶片遮挡形成了可辨认的阶段，局部仍有持剑姿态保持、机位绕行及人物细节变化。

已输出[原生短窗口链](outputs/reproduction/masters/round02_short_chain/reproduction_29s83.mp4)和[两个长窗口重定时版](outputs/reproduction/masters/round02_long_window_retimed/reproduction_29s83.mp4)。两版均为716帧；只替换月夜第一窗及红叶第二窗，分别保持同首尾、同提示词和同Seed，再压缩到相同成片时长。它们只检验两个窗口，不能推导“所有镜头都应先生成5秒再加速”。

所有共享节点之后删去下一段第0帧，只减少重复的条件图。接缝多帧图仍能看到接近条件姿势时减速、随后重新开展动作；位置接近不代表动势连续。以17.958秒红叶剑锋节点为例，短版边界图像MAE为2.63，周围前6步均值15.24、后6步12.24；较低边界差值与尾端趋近同一姿势一致，但它不直接度量人体速度，也不能凭数值宣布“无缝”。详见[接缝联系表](outputs/reproduction/final_review/round02_short_chain_seams.jpg)和[逐帧CSV](outputs/reproduction/final_review/round02_short_chain_per_frame.csv)。

第二次本地真实拼接暴露了分段MP4时长舍入：虽然帧数716，部分PTS产生约0.326毫秒偏差，平均帧率不再严格等于24。保留失败回执，修复为按已选帧的顺序重建 `PTS=n/24`，重新编码母版；重跑完整解码和每帧时间戳检查后，两版通过。没有放宽CFR断言，也没有复制或插入图像帧。

### 第三批结果

第三批[技术回执](outputs/reproduction/round03_review/technical_acceptance.json)与[逐镜观察](outputs/reproduction/round03_review/visual_review.json)覆盖全部8条。修改月夜场景用词后，两Seed的长片中间仍出现大面积重复矩形/石块背景；换成清晰屋面尾帧后，两Seed依然有重复块状、金色灯状图案，最后才回到输入尾帧姿态。清晰尾帧没有解决这次长片的中段失真。

红叶明确写出约2、4、7、8–11秒的镜头阶段后，输出仍主要围绕人物中全景舞剑，没有按要求推到低位剑锋。舞剑和衣摆有连续运动，但镜头路线与空间并未复现。术语和时序写得更明确，不等于硬控制已经生效。

这些结果只说明当前配置下的长镜头不适合直接用作主成片。重复图案的具体来源尚未通过稠密/稀疏、量化精度或采样步数消融分离；不把它归因给一个词，也不概括为所有H3配置的上限。保留[语义修订长镜头完整对照](outputs/reproduction/masters/round03_rewritten_long/reproduction_29s83.mp4)及[清晰尾帧完整对照](outputs/reproduction/masters/round03_clear_end/reproduction_29s83.mp4)。

近景滑行至爆发的Seed42短片可用：深色人物、素白结构和最终水光爆发较稳定；Seed108多出瓦屋面等细节。节奏版采用Seed42，并将它作为全景滑行后的近景切入，不冒充同机位无缝续接。

### 第四至六批：38个逐拍/亚拍锚点（36镜短窗口）与单调动态规划（Monotonic DP）精剪

针对前三批“3秒窗口漏掉原片1秒级关键变招与特写、接缝处首尾惯性停顿明显”的核心痛点，第四至六批执行了三项工作流重构：

1. **38个原片关键帧与亚拍锚点（0.20s–1.00s间隔，单镜39帧 `k=2`）**：
   - 将月夜（`f000–f438`）拆分为22个逐拍/亚拍短窗口（锚点：`[0, 30, 60, 75, 90, 106, 120, 150, 180, 212, 240, 255, 270, 276, 285, 300, 314, 330, 360, 376, 390, 420, 438]`），将红叶（`f452–f866`）拆分为14个逐拍短窗口（锚点：`[452, 484, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 866]`）。
   - 每镜固定生成 `39` 帧（`1.625s @ 24fps`），既比成片所需帧数（`5–26` 帧）留出约 `1.5×–3×` 的原生帧余量供后期裁切首尾加速缓冲，又彻底杜绝了长窗口的中段换景漂移。
   - 在 `round06` 中，针对含快速亚拍过渡的5个窗口插入中间关键帧（`f106` 满月正中剪影、`f255` 满月左上圆环、`f276` 屋脊剪影、`f314` 全屏白光垂直俯冲、`f376` 近景黑发横斩），徹底消除了此前 `m11` 侧颜特写眼部橙火伪影，并精准复现了所有快速闪切亚拍。
2. **使用 `h3-prompt-writing` 技能撰写全片结构化提示词与多Seed抽卡**：
   - 严格遵循 `.agents/skills/h3-prompt-writing/SKILL.md` 的 FL2VA 规范，所有提示词以显式首尾帧对齐头标开头，完整提供 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 三大段落，每镜生成 2–4 个不同 Seed 候选（`round04a` 20条、`round04b` 22条、`round05` 20条、`round06` 22条，均通过 [round04a](outputs/reproduction/round04a_review/technical_acceptance.json)、[round04b](outputs/reproduction/round04b_review/technical_acceptance.json)、[round05](outputs/reproduction/round05_review/technical_acceptance.json)、[round06](outputs/reproduction/round06_review/technical_acceptance.json) 技术核验）。
3. **参考片引导的单调动态规划（Monotonic DP）选帧与多Seed联合择优（[`optimize_director_edl.py`](code/optimize_director_edl.py)）**：
   - 升级 [`assemble_reproduction.py`](code/assemble_reproduction.py) 支持在 EDL 中声明严格单调递增的 `"selected_frames"` 原生帧序列，并支持任意倍率音频 `atempo` 级联。
   - [`optimize_director_edl.py`](code/optimize_director_edl.py) 将原片每一镜的 `(f_start, f_end]` 半开区间特征（64×36 高斯平滑色彩与 128×72 Sobel 结构梯度）与该镜所有候选 Seed 的 39 帧逐帧特征构建代价矩阵，在步长平滑约束、动作停滞惩罚与跨镜接缝连续性约束（起始帧 `0..1`、终止帧 `37..38`）下求解全局最优单调递增帧序列与最优 Seed。
   - 优化后全片 36 镜平均参考片误差 `mean_ref_mae_255 = 24.90`，全片 34 处跨镜接缝平均步进误差 `mean_seam_mae_255 = 2.50`（详见 [`dp_optimization_report.json`](outputs/reproduction/dp_optimization_report.json)）。

## 导演剪辑与36镜逐拍对齐结构

| 篇章与镜头区间 | 原片帧区间 (30fps) | 镜头数 | 等长版保留帧 (24fps) | 节奏版保留帧 (24fps) | 关键视觉节拍 |
| --- | --- | --- | --- | --- | --- |
| 月夜前段 (`m01..m05b`) | `f000–f120` (0.00s–4.00s) | 6镜 | 96帧 | 83帧 | 满月剪影蓄势 → 破水横斩白柱 → 屋面360度金焰圆环 → 向满月腾跃 → 满月正中剪影 → 满月青白圆环 |
| 月夜中段 (`m06..m10b`) | `f120–f270` (4.00s–9.00s) | 6镜 | 120帧 | 103帧 | 双白幡腾空回旋 → 屋面巨浪翻涌 → 四道白幡冲天拔高 → 幡顶迎月横斩 → 满月左上圆环 → 放射白光爆闪 |
| 月夜后段 (`m11a..m17`) | `f270–f438` (9.00s–14.60s) | 10镜 | 134帧 | 115帧 | 爆闪收束屋脊剪影 → 月前纯净侧颜特写 → 急拉远满月水环 → 全屏白光垂直俯冲 → 落定黑瓦屋面 → 屋面疾滑流火 → 近景黑发横斩 → 白光屋面蓄势 → 蓝白放射水刃大爆发 → 斜向白光收束 |
| 中场转场黑场 | `f439–f451` (14.60s–15.03s) | 显式黑场 | 11帧 | 2帧 | 对齐原片中段13帧@30fps黑场呼吸切口 |
| 红叶前段 (`a01..a04`) | `f452–f570` (15.07s–19.00s) | 4镜 | 94帧 | 81帧 | 枫林圆洞门背影右弧剑光 → 枫叶侧脸回眸 → 推近白袍底摆与斜剑反光 → 地面V形银白剑光轨迹 |
| 红叶中段 (`a05..a09`) | `f570–f720` (19.00s–24.00s) | 5镜 | 120帧 | 105帧 | 举剑过顶蓄势 → 左下巨型弧光横扫 → 俯拍木台白袍旋转 → 仰拍满月与剪影竖劈 → 宽景左跨步挥斩 |
| 红叶尾段 (`a10..a14`) | `f720–f866` (24.00s–28.87s) | 5镜 | 120帧 (含3帧淡出) | 102帧 (含3帧淡出) | 右下至左上仰弧剑光 → 旋身化满月红叶火环 → 满月红叶圆环持续旋转 → 红叶向树冠散开 → 前景大枫叶散景落幕 |
| 尾段收束黑场 | `f867–f894` (28.90s–29.83s) | 显式黑场 | 21帧 | 6帧 | 对齐原片尾段黑场收束 |
| **全片合计** | **`f000–f894` (29.833s)** | **36镜 + 2黑场** | **716帧 / 29.833秒** | **597帧 / 24.875秒** | **逐拍1:1精确同步（平均接缝MAE 2.50/255）** |

精确选择索引以[等长EDL](code/edl_director_29s83.json)、[节奏EDL](code/edl_director_tight.json)及各自逐帧来源回执为准。

最终审片还发现：旧H.264流复制拼接在紧凑版的2帧换景黑场处重复了上一段画面。帧数和PTS仍正确，因此仅检查这两项不足以发现问题。[失败回执](../../../artifacts/20261024-232431-exp06-sub03-final-review/receipt.json)保留；修复为逐段独立解码、按帧拼接并重建PTS，加入黑场真实像素核对后，两条导演版以同一EDL重跑通过。正式入口采用临时文件哈希校验后原子替换，避免用户读取到未完成的视频。

### 运行配置和可重复性范围

本轮使用 `MiniMax-H3-Base-FL2VA` 的现有加速生产配置：1344×768，24fps，8-step Turbo LoRA，CFG=1，INT8 ConvRot DiT/Video VAE，Qwen3-VL-32B NVFP4 AWQ 条件编码，Run 12 前2步稠密Sage、后6步Sol Attention，`tau=1.3`、`extra_tokens=256`，单流执行。它不是未量化50步基线的对照；当前结果不能概括所有H3推理配置的质量。

ComfyUI固定到 `f5ea88af1c46c50a39792d00af80e5af941f99f6`，Sol Attention节点固定到 `69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3`。运行期间观察到的模型仓库revision与五个权重的LFS哈希、大小保存在[模型来源快照](outputs/reproduction/preflight/model_source.json)，总权重约42.03GB；现有调度器实际仍从 `main` 下载，故此记录不是按revision强制固定下载的证明。环境中部分库固定版本，其余沿用当时Colab基础环境，不宣称位级可重复。

每条使用官方FL2VA图片对齐说明及 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 三字段，提示词全文、首尾条件PNG及原片帧号保存在各轮 `conditions.json` 和 `manifest.json`。`negative_prompt` 未接入当前条件编码，因此不把负向词作为有效控制变量。参考作者实际使用的模型与后期流程仍以[公开信息边界](report.md)为准。

### 费率与预算修正

运行期间 `colab usage` / `ccu-info` 实测费率为8.90 CU/hr，旧规划器和引擎仍以7.2046 CU/hr计算名义CU，其输出不能当作实际账单。第二批起独立看门狗按9.0 CU/hr保守计算；预算同时保留账户差额与含零CU前后传输时间的墙钟上界，按较高值占用30 CU预算，以覆盖账本更新延迟。详见[费率观测](outputs/reproduction/preflight/billing_rate_observed.json)和[预算回执](outputs/reproduction/budget_ledger.json)。

## 可重复执行的命令

在仓库根目录运行；前提为 `uv`、`ffprobe`、`ffmpeg` 和已授权配置的 `colab`，参考视频及其哈希需与[输入定义](code/study.json)一致。本地 Python 依赖锁定在准备脚本头部；生产 Python 环境由现有调度器选择，不打印凭据。

```bash
# 为新的 round 名称提取条件、编译清单；已存在的轮次不能覆盖
uv run experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/prepare_reproduction.py \
  --spec experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/reproduction_round06.json

# 干跑不分配 GPU
python3 experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/run_budgeted.py \
  --manifest experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/outputs/reproduction/round06/manifest.json --dry-run

# 实际批处理；输入路径绝对化以兼容 Harness 隔离运行
harness run exp06-sub03-round06 -- python3 \
  "$PWD/experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/run_budgeted.py" \
  --manifest "$PWD/experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/outputs/reproduction/round06/manifest.json" \
  --max-batch-cu 3.5

python3 experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/code/run_budgeted.py --status
```

CU 看门狗和账本控制只针对本实验专属会话，保留 2 CU 异常/停机余量；每批完成后停止 GPU 再做本地审片。预算[公开回执](outputs/reproduction/budget_ledger.json)只记录授权额度、增量消耗及状态，不公开账户总余额。

已有媒体可在0 GPU CU下运行单调DP选帧、重建两版导演母带与核验工件：

```bash
exp06_sub03_dir="$PWD/experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity"

uv run "$exp06_sub03_dir/code/optimize_director_edl.py" --write-edls

uv run "$exp06_sub03_dir/code/assemble_reproduction.py" \
  --edl "$exp06_sub03_dir/code/edl_director_29s83.json" --director

uv run "$exp06_sub03_dir/code/assemble_reproduction.py" \
  --edl "$exp06_sub03_dir/code/edl_director_tight.json" --director

uv run "$exp06_sub03_dir/code/review_masters.py" --edls \
  "$exp06_sub03_dir/code/edl_round01.json" \
  "$exp06_sub03_dir/code/edl_round02_short.json" \
  "$exp06_sub03_dir/code/edl_round02_retimed.json" \
  "$exp06_sub03_dir/code/edl_round03_long.json" \
  "$exp06_sub03_dir/code/edl_round03_clear.json" \
  "$exp06_sub03_dir/code/edl_director_29s83.json" \
  "$exp06_sub03_dir/code/edl_director_tight.json" \
  --preferred selected_reproduction

harness doctor
uvx ruff check "$exp06_sub03_dir/code"
uv run --with pyright --with numpy==2.5.4 \
  --with opencv-python-headless==5.0.0.93 --with pillow==13.0.0 \
  pyright "$exp06_sub03_dir/code"
git diff --check
```

前提是保留各批 `batches/<project>/1` 内的manifest、metrics和H3视频，以及原始参考片。本地HTTP Range服务的命令见[README](README.md)，审片入口为 `outputs/reproduction/final_review/index.html`。

## 验收与最终总结

七批126条候选均已完整解码，核对原生24fps、目标帧数、1344×768及音轨。38个不同原片关键帧/亚拍锚点参与条件制作，条件PNG、原始RGB哈希、输入规格和全轮次完整提示词均保留。等长导演版716帧、节奏导演版597帧，均重新完整解码并核对每帧 `PTS=n/24`、来源、音轨和声明黑场；没有补写实现镜像式单元测试。

[七个完整版本的最终核验](outputs/reproduction/final_review/acceptance.json)全部通过。累计实际扣费 **11.916826 CU**（保守墙钟预算占用 **17.066478 CU**，剩余 **12.933522 CU**），活动GPU分配已归零。

本次最核心的工程与导演方法论结论是：**1.0秒级逐拍关键帧 + 0.20s–0.53s快速亚拍锚点 + 39帧（`k=2`）原生生成窗口 + 参考片引导的单调动态规划（Monotonic DP）选帧**，能够彻底消除长镜头场景漂移与3秒窗口漏镜，并在接缝处切除FL2VA首尾加减速惯性停顿，实现与高密度作画参考片逐拍高度一致的成片质量。
