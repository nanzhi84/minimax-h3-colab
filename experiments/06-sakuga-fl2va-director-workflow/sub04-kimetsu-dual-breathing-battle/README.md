<!-- harness-ignore LNK001: outputs/ 为本机忽略入库的真实视频及验收工件 -->
# 子实验 04：无参考原创《鬼灭之刃》风格水火双呼吸死斗（30s）

本子实验在没有外部参考视频的条件下，独立完成一部 **30.00 秒（720 帧 @ 24fps，1344×768）** 的 Ufotable《鬼灭之刃》风格双人遭遇死斗与“水之呼吸 → 火之神神乐（日晕之龙·头舞）”呼吸法连招 AI 动画短片，并首次通过控制变量实验对比了 **未修剪固定 Seed 基线（Workflow A）**、**I2VA 仅首帧 + 73 帧跨拍长窗口混合消融（Workflow B）** 与 **20 镜同源关键帧 39 帧 FL2VA 多 Seed 抽卡 + 一拍二（On-Twos）去重弧长单调动态规划精剪（Director Master）** 三种工作流。

完整实验设计与量化证据见 [实验报告](report.md)。

## 核心交付入口

- **30.00 秒导演母带（主交付，720 帧 @ 24fps）**：[outputs/sub04_director_master_30s00.mp4](outputs/sub04_director_master_30s00.mp4)（EDL 审计回执：[outputs/sub04_director_edit_decision_list.json](outputs/sub04_director_edit_decision_list.json)）
- **25.83 秒高密度节奏强化导演版（620 帧 @ 24fps）**：[outputs/sub04_director_master_tight_25s83.mp4](outputs/sub04_director_master_tight_25s83.mp4)（EDL 审计回执：[outputs/sub04_director_edit_decision_list_tight.json](outputs/sub04_director_edit_decision_list_tight.json)）
- **同屏对比视频（Workflow A 未修剪基线 vs 30s 导演母带）**：[outputs/sub04_workflow_a_vs_director.mp4](outputs/sub04_workflow_a_vs_director.mp4)
- **同屏对比视频（Workflow B I2VA+73f 混合 vs 30s 导演母带）**：[outputs/sub04_workflow_b_vs_director.mp4](outputs/sub04_workflow_b_vs_director.mp4)
- **交互式抽卡与出入点精剪工作台（Gacha & NLE Workbench）**：[outputs/review/index.html](outputs/review/index.html)
- **全片 30 点概览与 19 处跨镜接缝检视表**：[director_master_30s00_overview.jpg](outputs/review/director_master_30s00_overview.jpg)、[director_master_30s00_seams.jpg](outputs/review/director_master_30s00_seams.jpg)、[workflow_comparison_30s.jpg](outputs/review/workflow_comparison_30s.jpg)
- **独立 30.0 CU 预算账本**：[outputs/budget_ledger.json](outputs/budget_ledger.json)（6 批共生成 76 条 H3 视频，实际扣费 **`3.255638 CU`**，含计费延迟保守占用 `8.560879 CU`，剩余 `21.439121 CU`，GPU 会话已全部归零释放）

## 三大工作流与四版成片核心指标对比

| 方案名称 | 时长 / 帧数 | 跨镜锚点 MAE (`seam_mae`) | 成片切镜步进 MAE (`step_mae`) | 切镜重复静止数 / 跳切数 | 全片平均运动幅度 (`global_step_mae`) | 关键结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **Workflow A（固定 Seed42 未修剪基线）** | 30.00s / 720f | `1.38` (max `1.72`) | `3.46` (max `4.84`) | **19 处重复帧顿挫** / 0 跳切 | `14.03` | `0..38` 保留了切镜两端共享关键帧，导致每处切镜速度骤降 75%（出现 2 帧静止卡顿） |
| **Workflow B（I2VA + 73f 跨拍混合）** | 30.00s / 720f | `23.53` (max `75.94`) | `31.97` (max `87.32`) | 8 处静止 / **8 处剧烈跳切** | `15.82` | 仅首帧约束（I2VA）与 73 帧跨拍长窗在尾帧/中段严重失锁（平均尾锚漂移 `24.09`），引发断崖跳切 |
| **Director Master（30.00s 主交付）** | 30.00s / 720f | **`1.47` (max `2.08`)** | **`5.51` (max `16.24`)** | **0 处重复帧 / 0 跳切** | **`14.74` (+5.1%)** | 锁定 `frame 38` 尾锚点（`2.36`）并切除 `frame 0`（及一拍二起势驻留帧），彻底消除切镜卡顿与跳切 |
| **Director Tight（25.83s 节奏版）** | 25.83s / 620f | **`1.47` (max `2.08`)** | **`6.93` (max `20.72`)** | **0 处重复帧 / 0 跳切** | **`16.93` (+20.7%)** | 进一步消融段内“一拍二（On-Twos）”保持帧，浓缩为高密度“一拍一（On-Ones）”作画 |

## 代码与可复现流水线

- [prepare_sub04.py](code/prepare_sub04.py)：校验并对齐 21 张同源关键帧（`kf00..kf20`）至 `1344×768`，严格遵循 [SKILL.md](../../.agents/skills/h3-prompt-writing/SKILL.md) 组装 `FL2VA` 与 `I2VA` 批次 `manifest.json`。
- [run_budgeted.py](code/run_budgeted.py)：独立 `30.0 CU` 预算看门狗与 Colab Blackwell G4 生命周期执行器。
- [review_candidates.py](code/review_candidates.py)：逐条核验 76 条生成视频的帧数、帧率、分辨率、音轨与首/中/尾动态指标。
- [optimize_sub04_edl.py](code/optimize_sub04_edl.py)：多 Seed 抽卡评分器与基于累积运动弧长（Arc-Length）及“一拍二（On-Twos）”保持帧消融的单调动态规划选帧器。
- [assemble_sub04.py](code/assemble_sub04.py)：24fps 恒定帧率精确选帧拼接与 `atempo` 级联音画同步合成器。
- [review_sub04.py](code/review_sub04.py)：生成四套母带逐帧 CSV、30 点概览图、19 处接缝检视表、同屏对比视频与交互式 HTML 精剪工作台。
