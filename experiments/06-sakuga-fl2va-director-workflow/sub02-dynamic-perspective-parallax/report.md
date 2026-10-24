<!-- harness-ignore LNK001: outputs/ and studio/outputs/ are gitignored runtime artifacts -->
# Experiment 06 子实验 2（`sub02-dynamic-perspective-parallax`）实测报告：大跨度三维透视重构、背景视差解放与刚体受击反馈

> **实验代号**：`exp06_sub02_dynamic_parallax_15s`（Batch `#1`）  
> **执行日期**：2026-10-24  
> **算力底座**：Colab Blackwell G4（96GB VRAM）· `Run 12 [S8-FullStack-Pareto]`（[ADR-03](../../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md) / [ADR-04](../../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) / [ADR-05](../../../docs/adr/0005-staged-batch-production-workflow.md)）  
> **对照基线**：[子实验 1（`sub01-static-canvas-lineage/report.md`）](../sub01-static-canvas-lineage/report.md)

---

## 1. 核心假设验证结论（Executive Summary）

针对子实验 1（同画布局部微调锁死背景）暴露出的**“空间调度停滞（零视差原地摆姿）”**与**“兵器图层错位叠加（缺乏刚体受击反馈）”**两大核心缺陷，子实验 2 彻底放弃在原图画布上微调姿态，改为**“身份特征锁定 + 尾帧强制重构三维机位/焦段/俯仰角 + 显式绘制刚体受击破坏结果”**。在相同的 Colab Blackwell G4 8-step 生产管线下完成 4 分镜 × 5 Seeds = 20 条视频抽卡与 15.00s（360 帧）成片精剪后，通过 Farneback 稠密光流（`cv2.calcOpticalFlowFarneback`）与生产质量评分对决，核心假设得到全面证实：

1. **打破 `Video VAE` 背景坐标锁死，解放 3D 摄像机运镜与视差**：
   - 在子实验 1 空间停滞最严重的 **Shot 1 (`A -> B`)** 中，子实验 2 将首尾帧从“同机位平视微移”重构为 **“24mm 高角越肩俯拍 (`+35°`) $\rightarrow$ 14mm 贴地虫视广角仰拍 (`-30°`)”**，首尾帧边缘背景视差强度（`Border Parallax MAE`）从 `28.26` 跃升至 **`46.42`（`1.64x`，`+64.3%`）**。
   - 渲染出的视频在 Farneback 稠密光流实测中，**背景平均运镜光流从 `1.17 px/frame` 暴涨至 `2.68 px/frame`（`2.29x`，`+129.1%`）**，**P90 峰值运镜光流从 `2.60 px/frame` 飙升至 `7.42 px/frame`（`2.85x`，`+185.4%`）**，**前景人物位移光流从 `0.92 px/frame` 提升至 `2.23 px/frame`（`2.43x`，`+142.4%`）**！
2. **大跨度透视轨迹不仅没有降低稳定性，反而消除了中间帧无事可做的抖动（运动平滑度 +17.60 pts）**：
   - 在 Shot 1 (`90f = 3.75s`) 中，子实验 1 因首尾帧过于接近，模型在第 2~3 秒无事可做而产生微幅拉扯（`motion_smoothness_score = 61.45`）；子实验 2 给出了明确的俯冲切入仰拍三维空间轨迹，模型在全部 90 帧内匀速分配推镜与透视旋转，**运动平滑度反升至 `79.05`（`+17.60 pts`）**，综合生产质量分 `prod_quality_score` 从 `82.01` 升至 **`86.00`（`+3.99 pts`）**！
3. **显式受击形变锚点成功逼迫扩散模型反推刚体动量传递**：
   - 在 **Shot 2 (`B -> C`)** 中，通过在尾帧 `Frame C` 显式绘制**“120° 环绕拉丝背景 + 雷刃劈裂黑铁胸甲凹陷 + 躯干 C 形后仰折弯 + 双膝弯曲 + 铁靴在石板犁出深槽”**，模型不再生成浮空图层叠加的假打，而是在 124 帧内合成了从双刃死锁咬合（`Frame B`）到破防轰退（`Frame C`）的连续刚体受击过程，首尾帧像素还原度 `S_fl2v_pixel` 从 `77.83` 提升至 **`83.50`（`+5.67 pts`）**，生产质量分达 **`85.99`**（`seed=777` 高达 **`87.63`**）。

---

## 2. 子实验 1 vs. 子实验 2 稠密光流与质量指标全景对比

| 分镜对 (时长) | 指标维度 | 子实验 1 (`sub01` 锁画布微调) | 子实验 2 (`sub02` 三维透视重构 + 刚体受击) | 相对增益 / 演进结论 |
| :--- | :--- | :---: | :---: | :--- |
| **Shot 1 (`A -> B`)**<br>`90f` (`3.75s`) | 首尾帧边缘视差 (`Border MAE`)<br>背景平均运镜光流 (`BG Flow`)<br>背景 P90 峰值光流 (`P90 BG`)<br>主体平均位移光流 (`FG Flow`)<br>运动平滑度 (`Smoothness`)<br>生产质量分 (`ProdQ`) | `28.26`<br>`1.17 px/f`<br>`2.60 px/f`<br>`0.92 px/f`<br>`61.45`<br>`82.01` | **`46.42`**<br>**`2.68 px/f`**<br>**`7.42 px/f`**<br>**`2.23 px/f`**<br>**`79.05`**<br>**`86.00`** | **`1.64x` (`+64.3%`)** 首尾背景透视差<br>**`2.29x` (`+129.1%`)** 真实 3D 摄像机推进<br>**`2.85x` (`+185.4%`)** 峰值俯冲推镜速度<br>**`2.43x` (`+142.4%`)** 角色跨空间冲刺位移<br>**`+17.60 pts`** 消除原地停滞抖动<br>**`+3.99 pts`** 画质与动态双优 |
| **Shot 2 (`B -> C`)**<br>`124f` (`5.17s`) | 首尾帧边缘视差 (`Border MAE`)<br>背景平均运镜光流 (`BG Flow`)<br>背景 P90 峰值光流 (`P90 BG`)<br>首尾帧锚定分 (`S_fl2v`)<br>运动平滑度 (`Smoothness`)<br>生产质量分 (`ProdQ`) | `35.98`<br>`1.77 px/f`<br>`4.48 px/f`<br>`77.83`<br>`64.52`<br>`82.37` | **`51.30`**<br>**`2.23 px/f`**<br>**`5.91 px/f`**<br>**`83.50`**<br>**`70.87`**<br>**`85.99`** | **`1.43x` (`+42.6%`)** 120° 环绕背景重构<br>**`1.26x` (`+26.0%`)** 环绕摇镜背景视差<br>**`1.32x` (`+31.9%`)** 峰值横向拉丝视差<br>**`+5.67 pts`** 胸甲凹陷与 C 形后仰精准落位<br>**`+6.35 pts`**（`seed=777` 达 `79.76`）<br>**`+3.62 pts`**（`seed=777` 达 `87.63`） |
| **Shot 3 (`D -> E`)**<br>`73f` (`3.04s`) | 首尾帧边缘视差 (`Border MAE`)<br>首尾帧锚定分 (`S_fl2v`)<br>声画同步相关 (`AV_Sync`) | `68.84`<br>`80.22`<br>`0.2874` | **`74.04`**<br>**`81.70`**<br>**`0.6040`** | **`1.08x`** 85mm 荷兰角微距 $\rightarrow$ 12mm 对角俯冲<br>**`+1.48 pts`** 咆哮雷龙扑向镜头前景<br>**`2.10x`** 蓄力蜂鸣与拔刀雷鸣精准对齐 |
| **Shot 4 (`E -> F`)**<br>`124f` (`5.17s`) | 背景平均运镜光流 (`BG Flow`)<br>背景 P90 峰值光流 (`P90 BG`)<br>首尾帧锚定分 (`S_fl2v`)<br>生产质量分 (`ProdQ`) | `2.28 px/f`<br>`4.48 px/f`<br>`76.92`<br>`82.72` | **`2.51 px/f`**<br>**`7.63 px/f`**<br>**`80.13`**<br>**`82.94`** | **`1.11x`** 30m 高空鸟瞰全景急拉<br>**`1.70x` (`+70.3%`)** 峰值气浪掀飞屋顶瓦片<br>**`+3.21 pts`** 黑甲武士轰飞撞穿前景石墙<br>**`+0.22 pts`** 终局大场面稳定收束 |

---

## 3. Colab Blackwell G4 算力账本审计（Batch `#1`）

| 审计阶段 | 耗时 / 速率 | 算力消耗 (`CU`) | 备注 |
| :--- | :---: | :---: | :--- |
| **Step 0: 本地 0-CU 排期与打包上传** | `95.46s` (`14.24 MB`) | `0.0000 CU` | 仅打包 6 张 `frame_A..F.png`（修复默认资产冗余拷贝 Bug） |
| **Step 1~2: G4 冷启动 + 20 任务渲染 + Stage 1 卸载** | `957.47s` (`15.96 min`) | `1.9162 CU` (理论)<br>`2.1594 CU` (账本净差) | 16 次跨 Seed `cond_hash` 缓存命中（`0.0s` 重复 VLM 编码）；Stage 1 机房内网卸载 `337.21 MB` 仅耗时 **`3.626s` (`93.0 MB/s`)** |
| **Step 3: G4 实例销毁止血** | `< 2.0s` | `0.0000 CU/hr` | `Active Assignments: 0`（账本余额：`2474.9738` $\rightarrow$ `2472.8144 CU`） |
| **Step 4: 本地 0-CU Stage 2 多线程拉取与精剪** | `48.16s` (`7.00 MB/s`) | `0.0000 CU` | 本地执行 `director_cut.py` 产出 15.00s 成片与抽卡矩阵 |

---

## 4. 20 任务抽卡优选与 15.00s（360 帧）导演精剪决策表（EDL）

| 顺序 | 分镜 ID | 胜出任务 (`Seed`) | 导演总分 (`Director`) | 生产质量 (`ProdQ`) | 首尾锚定 (`S_fl2v`) | 原始帧数 | 裁剪区间 (`Keep`) | 成片帧数 (`时长`) | 接缝桥接策略 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `shot_01_thunder_dash` | `job_01` (`seed=42`) | **74.15** | **86.00** | 79.95 | `90f` | `[6..89]` | **84f (`3.50s`)** | 裁去首部 6 帧静止蓄力；**尾帧严格停在第 89 帧（`Frame B` 14mm 仰拍双刃咬合）** |
| **2** | `shot_02_blade_clash_shockwave` | `job_06` (`seed=42`) | **78.67** | **85.99** | **83.50** | `124f` | `[1..101]` | **101f (`4.21s`)** | **剔除第 0 帧（消除 `Frame B` 重复）**；120° 环绕跟拍 + 胸甲碎裂轰退，在第 101 帧冲击波峰值硬切 |
| **3** | `shot_03_eye_awakening_slash` | `job_11` (`seed=42`) | **71.61** | 80.31 | 81.70 | `73f` | `[13..72]` | **60f (`2.50s`)** | 跨机位蒙太奇 J-Cut（`120ms` 音频先行）；**尾帧严格停在第 72 帧（`Frame E` 12mm 腾空化龙）** |
| **4** | `shot_04_dragon_finale_blast` | `job_18` (`seed=777`) | **73.13** | **82.94** | 80.13 | `124f` | `[1..115]` | **115f (`4.79s`)** | **剔除第 0 帧（消除 `Frame E` 重复）**；保留 2 帧彩色桥接后注入 2 帧黑白闪烁帧（Impact Frames），30m 鸟瞰急拉收尾 |
| **合计** | **4 镜无缝成片** | — | **74.39** | **83.81** | **81.32** | `411f` (`17.13s`) | 精确裁剪 `51f` | **360f (`15.00s`)** | 共享节点 `B`、`E` 零跳帧桥接 + 跨机位 `C || D` J-Cut |

---

## 5. 产物清单与归档路径

- **成片与对比视频**：
  - 导演精剪 15.00s 成片（360 帧）：[`outputs/sub02_director_master_15s.mp4`](./outputs/sub02_director_master_15s.mp4)
  - 原始未裁剪拼接基线（411 帧 / 17.13s）：[`studio/outputs/exp06_sub02_dynamic_parallax_15s/1/full_storyboard_concat.mp4`](../../../studio/outputs/exp06_sub02_dynamic_parallax_15s/1/full_storyboard_concat.mp4)
- **可视化审计图谱**：
  - 6 关键帧三维透视故事板：[`outputs/sub02_keyframe_storyboard_sheet.png`](./outputs/sub02_keyframe_storyboard_sheet.png)
  - 4×5（20 条视频）导演抽卡优选矩阵：[`outputs/sub02_seed_gacha_4x5_matrix.png`](./outputs/sub02_seed_gacha_4x5_matrix.png)
  - 15.00s 成片切镜接缝与透视重构核验图：[`outputs/sub02_master_cut_transition_audit.png`](./outputs/sub02_master_cut_transition_audit.png)
- **结构化决策日志**：
  - 导演剪辑决策表（EDL JSON）：[`outputs/sub02_director_edit_decision_list.json`](./outputs/sub02_director_edit_decision_list.json)
