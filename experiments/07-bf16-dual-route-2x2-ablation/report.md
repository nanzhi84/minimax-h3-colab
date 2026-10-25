<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# 实验 07 报告：全浮点 BF16 底座下 `FL2VA` × `Ref2VA` 两模型两档位（2×2 矩阵）速度与四维质量深度评测

- **实验目录**: [`experiments/07-bf16-dual-route-2x2-ablation/`](./README.md)
- **实测指标 JSON**: [`outputs/exp07_bf16_2x2_metrics.json`](./outputs/exp07_bf16_2x2_metrics.json)
- **会话账本摘录**: [`outputs/colab_session_ledger_excerpt.json`](./outputs/colab_session_ledger_excerpt.json)
- **可视化图表与对比视频**:
  - 速度与 CU 成本对比图：[`outputs/fig1_exp07_2x2_speed_and_cu.png`](./outputs/fig1_exp07_2x2_speed_and_cu.png)
  - 四维质量与生产质量分解图：[`outputs/fig2_exp07_2x2_4d_quality_breakdown.png`](./outputs/fig2_exp07_2x2_4d_quality_breakdown.png)
  - 4 档位关键帧长图（`f=0, 31, 62, 93, 123`）：[`outputs/fig3_exp07_keyframe_strip.png`](./outputs/fig3_exp07_keyframe_strip.png)
  - 2×2 四宫格 124 帧同步对比视频：[`outputs/exp07_2x2_quad_grid_comparison.mp4`](./outputs/exp07_2x2_quad_grid_comparison.mp4)

---

## 1. 升级背景与 2×2 BF16 路由架构设计

在实验 02–06 中，我们基于 `INT8 ConvRot` / `NVFP4 AWQ` 量化权重验证了 Blackwell G4（`NVIDIA RTX PRO 6000 Blackwell Server Edition`, `94.97 GiB` GDDR7）上的加速算子栈与零浪费传输管线。针对本次生产架构升级：
1. **彻底移除模型权重量化，全线切换为原生浮点权重（`BF16 / FP16 / FP32`）**：
   - DiT 扩散主干：采用无损裁剪掉了约 `13B` 仅训练期冗余 AdaLN 参数的 **`minimax_h3_fl2va_pruned_bf16.safetensors` (`40.23 GB`)** 与 **`minimax_h3_ref2va_pruned_bf16.safetensors` (`40.23 GB`)**（与 `66.28 GB` 未裁剪版在推理路径上逐位等价，且在 `95 GiB` 显存与 `/content` 磁盘上留出充足余量）。
   - 多模态文本编码器：**`qwen3vl_32b_minimax_h3_bf16.safetensors` (`51.51 GB`)**。
   - 视频与音频 VAE：**`minimax_h3_video_vae_fp16.safetensors` (`5.21 GB`)** + **`minimax_h3_audio_vae_fp32.safetensors` (`0.61 GB`)**。
   - 8 步蒸馏加速 LoRA：**`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (`1.96 GB`)**（用于 `fl2va` Fast 档）与 **`minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` (`1.96 GB`)**（来自 `lightx2v/Minimax-h3-Turbo`，用于 `ref2va` Fast 档）。
2. **本地按需选路（`model_route`）× 双算子档位（`profile`）的 2×2 生产矩阵**：
   - 生产调用 Colab 后端前，由本地 [`manifest_schema.py`](../../studio/api/manifest_schema.py) 与 [`local_planner.py`](../../studio/api/local_planner.py) 锁定路由与档位（`fl2va_bf16_quality`、`fl2va_bf16_fast`、`ref2va_bf16_quality`、`ref2va_bf16_fast`），远端 [`remote_batch_engine.py`](../../studio/api/remote_batch_engine.py) 仅下载当前批次所需的单一路由权重包（`97.55 GB` 或 `99.51 GB`）。
   - **两阶段显存生命周期保障 95 GiB 不 OOM**：Phase 2 加载 `Qwen3-VL-32B BF16` (`47.97 GiB`) + `Video VAE FP16` (`4.85 GiB`) 完成全批次条件预编码（峰值显存 **`54.14 GiB`**）后，**立即彻底卸载 `Qwen3-VL-32B BF16`（实测释放 `52.82 GiB` 显存，静态显存归零至 `0.01 GiB`）**；随后 Phase 3 加载 `Pruned BF16 DiT` (`37.55 GiB`) 常驻去噪，使全流程峰值显存严格控制在 **`43.07 ~ 48.25 GiB`**（仅占 G4 显存的 `50.8%`）。

---

## 2. 统一测试用例与提示词双规范对齐

为在相同锚定条件下公平对比 `FL2VA` 与 `Ref2VA` 两条模型路由及 `Quality`（20 步原生无 LoRA）与 `Fast`（8 步 Turbo LoRA + Sol-Attn 稀疏加速）两个档位：
- **统一几何与随机种子**：分辨率 **`1344×768`**（16:9 标准 768p）、帧数 **`124` 帧**（`5.17s @ 24 FPS`，满足 $17k+5$ 网格）、采样器 `res_multistep`、`cfg = 1.0`、`seed = 42`。
- **统一视觉锚点图像**：沿用实验 02–06 的赛博朋克雨夜街头吉他手标准锚点图（[`input_first_frame.png`](./outputs/input_first_frame.png) 与 [`input_last_frame.png`](./outputs/input_last_frame.png)）：
  - 在 **`FL2VA` 路由**（`Run 01`, `Run 02`）中，作为 `MiniMaxH3ImageToVideo` 的 `first_frame` 与 `last_frame` 传入；提示词严格遵循 [`h3-prompt-writing/references/base-en.txt`](../../.agents/skills/h3-prompt-writing/references/base-en.txt) 的三标签 XML 结构（`<integrated_multimodal_description>` + `<overall_soundscape>` + `<non_diegetic_music>`）。
  - 在 **`Ref2VA` 路由**（`Run 03`, `Run 04`）中，作为 `MiniMaxH3ReferenceToVideo` 的 `ref_image_0`（对应 `<Picture 1>`）与 `ref_image_1`（对应 `<Picture 2>`，`ref_image_size="match"`）传入；提示词严格遵循 [`h3-prompt-writing/references/ref-en.txt`](../../.agents/skills/h3-prompt-writing/references/ref-en.txt) 的六段式结构（`[Visual Reference Description]` 显式绑定 `<Picture 1>` 与 `<Picture 2>`）。

---

## 3. 速度、显存与算子调用计数器实测对比

### 3.1 2×2 四档位性能与算子遥测全表

| 指标项 | **Run 01**<br>`fl2va_bf16_quality` | **Run 02**<br>`fl2va_bf16_fast` | **Run 03**<br>`ref2va_bf16_quality` | **Run 04**<br>`ref2va_bf16_fast` |
| :--- | :---: | :---: | :---: | :---: |
| **模型路由 (`model_route`)** | `fl2va` (`ImageToVideo`) | `fl2va` (`ImageToVideo`) | `ref2va` (`ReferenceToVideo`) | `ref2va` (`ReferenceToVideo`) |
| **DiT 权重 (`BF16`)** | `fl2va_pruned_bf16` | `fl2va_pruned_bf16` | `ref2va_pruned_bf16` | `ref2va_pruned_bf16` |
| **8-Step Turbo LoRA (`BF16`)** | 无 | `fl2v_turbo_8step_v1.0` | 无 | `ref2v_turbo_8step_v1.1_768p` |
| **采样步数 / 调度器** | `20` 步 / `simple` | `8` 步 / `simple` | `20` 步 / `beta` | `8` 步 / `simple` |
| **稠密注意力后端** | `cuDNN BF16 SDPA` (`100%`) | `SageAttention` (`start=0.20`) | `cuDNN BF16 SDPA` (`100%`) | `SageAttention` (`start=0.20` + 首尾保护) |
| **稀疏注意力 (`Sol-Attn`)** | 关闭 | 直通 `ck.sol_attn` (`tau=1.3, tok=256`) | 关闭 | 直通 `ck.sol_attn` (`tau=1.0, tok=256, prot=0,1,48,49`) |
| **算子调用计数 (`kernel_calls`)** | `sdpa=1002`<br>`fused_mod=1000` | `sage=102, ck_sol_direct=300`<br>`fused_mod=400` | `sdpa=1002`<br>`fused_mod=1000` | `sage=126, ck_sol_direct=276`<br>`fused_mod=400` |
| **DiT 去噪耗时 (`denoise_sec`)** | `234.41s` | **`59.19s`** | `235.69s` | **`62.87s`** |
| **单步平均耗时 (`sec_per_step`)** | `11.72s/step` | **`7.40s/step`** (**`1.58×`** 单步提速) | `11.78s/step` | **`7.86s/step`** (**`1.50×`** 单步提速) |
| **Video VAE (`FP16`) 解码耗时** | `7.03s` | `7.13s` | `7.39s` | `7.07s` |
| **Audio VAE (`FP32`) 解码耗时** | `0.036s` | `0.036s` | `0.104s` | `0.036s` |
| **端到端 GPU 总耗时 (`e2e_sec`)** | `241.47s` | **`66.35s`** | `243.18s` | **`69.98s`** |
| **单条视频 GPU 成本 (`cu_e2e`)** | `0.4833 CU` | **`0.1328 CU`** | `0.4867 CU` | **`0.1400 CU`** |
| **相对同路由 Quality 档加速比** | `1.00×` (基准) | 去噪 **`3.96×`** / E2E **`3.64×`** | `1.00×` (基准) | 去噪 **`3.75×`** / E2E **`3.48×`** |
| **常驻显存 / 峰值显存 (`GiB`)** | `42.97` / `48.24 GiB` | `42.97` / `48.24 GiB` | `42.97` / `43.07 GiB` | `42.97` / `48.25 GiB` |

### 3.2 原生浮点 `BF16`（实验 07）对比量化 `INT8 ConvRot`（实验 03）的速度开销分析
- **8 步 `Run 12` 帕累托档位对比**：
  - 实验 03 `INT8 ConvRot`（`20.97 GB` DiT + `2.81 GB` INT8 Video VAE）：去噪 `39.96s`（`5.00s/step`）+ Video/Audio VAE `4.23s` = 端到端 **`44.20s` (`0.0885 CU`)**。
  - 实验 07 原生 **`BF16` (`fl2va_bf16_fast`)**（`40.23 GB` BF16 DiT + `5.21 GB` FP16 Video VAE）：去噪 **`59.19s`**（`7.40s/step`）+ Video/Audio VAE `7.16s` = 端到端 **`66.35s` (`0.1328 CU`)**。
  - **核心发现**：即便完全不使用 INT8 权重量化，8 步 `fl2va_bf16_fast`（`66.35s`）与 `ref2va_bf16_fast`（`69.98s`）依然比实验 03 的 **8 步 INT8 量化稠密 SDPA 基线 (`82.10s`)** 快 **`1.24×`** 与 **`1.17×`**！这得益于 `Run 12` / `Run 09` 的直通零拷贝 `ck.sol_attn` 稀疏注意力将单步耗时从 BF16 SDPA 的 `11.72s/step` 压降至 `7.40s/step`（单步提速 `1.58×`），抵消了从 INT8 线性层切回 BF16 线性层带来的带宽增量。

---

## 4. 四维正交质量评估与生产质量对比

沿用实验 02/03 完全相同的四维评估器 `QualityEvaluator`（`S_ref` 35% + `S_vbench` 25% + `S_fl2v` 25% + `S_audio` 15%）及实验 04–06 生产指标（`S_fl2v_pixel`、`prod_quality_score`）：

### 4.1 四维质量与生产指标全表

| 评估维度 / 细项指标 | **Run 01**<br>`fl2va_bf16_quality`<br>(20步 无LoRA SDPA) | **Run 02**<br>`fl2va_bf16_fast`<br>(8步 LoRA + Run12) | **Run 03**<br>`ref2va_bf16_quality`<br>(20步 无LoRA SDPA) | **Run 04**<br>`ref2va_bf16_fast`<br>(8步 LoRA + Run09HiFi) |
| :--- | :---: | :---: | :---: | :---: |
| **1. 同路由参考保真度 (`S_ref`, 35%)** | **`100.00`** (FL2VA 基准) | `23.29` (跨LoRA轨迹) | **`100.00`** (Ref2VA 基准) | `17.05` (跨LoRA轨迹) |
| ↳ `PSNR` (dB) / `SSIM` / `LPIPS` (同路由) | `100.0` / `1.0000` / `0.0000` | `15.49` / `0.5979` / `0.4526` | `100.0` / `1.0000` / `0.0000` | `13.03` / `0.4334` / `0.4623` |
| ↳ 全局相对 `Run 01` 保真度 (`S_ref_vs_run01`) | `100.00` | `23.29` | `16.62` | `14.40` |
| **2. VBench 时序与画质 (`S_vbench`, 25%)** | `66.40` | `65.86` (`-0.54`) | **`84.98`** (`+18.58`) | **`87.51`** (`+21.11`) |
| ↳ 主体一致性 (`DINOv2`) / 背景一致性 (`CLIP`) | `57.52` / `71.43` | `55.44` / `69.61` | **`81.97`** / **`92.91`** | **`88.74`** / **`95.35`** |
| ↳ 时序抗闪烁度 / 光流平滑度 (`RAFT`) | `98.02` / `47.36` | `97.04` / `50.86` | `97.70` / **`84.04`** | **`98.36`** / **`86.62`** |
| ↳ 光流动态幅度 (`px`) / 美学评分 (`MUSIQ`) | `27.19` / `57.73` | `29.76` / `56.66` | `10.07` / **`63.71`** | `5.88` / `61.69` |
| **3. 首尾帧多特征对齐 (`S_fl2v`, 25%)** | **`93.42`** | **`91.70`** (`-1.72`) | `22.01` | `19.79` |
| ↳ 首帧综合分 (`SSIM` / `DINO` / `CLIP`) | **`98.34`** (`0.9875`/`0.9823`/`0.9805`) | **`97.03`** (`0.9875`/`0.9567`/`0.9713`) | `22.99` (`0.4491`/`-0.010`/`0.3307`) | `21.43` (`0.3915`/`-0.025`/`0.3558`) |
| ↳ 尾帧综合分 (`SSIM` / `DINO` / `CLIP`) | **`90.15`** (`0.9824`/`0.8231`/`0.9250`) | **`88.15`** (`0.9815`/`0.7722`/`0.9273`) | `21.35` (`0.5003`/`0.0207`/`0.1838`) | `18.70` (`0.3864`/`-0.028`/`0.2746`) |
| **4. 音频质量与音画同步 (`S_audio`, 15%)** | `93.43` | `57.78` (线性口径 `71.82`) | **`94.96`** | `63.23` (线性口径 `75.12`) |
| ↳ 频谱相关系数 (`spectral_corr`) / `av_sync` | `1.0000` / `67.16` | **`0.8719`** / `58.78` | `1.0000` / **`74.80`** | **`0.8836`** / `58.95` |
| **综合四维得分 `Q_total` (同路由口径)** | **`88.97`** | `56.21` (线性口径 `58.31`) | **`75.99`** | `42.28` (线性口径 `44.06`) |
| **综合四维得分 `Q_total_vs_run01` (全局口径)** | **`88.97`** | `56.21` (线性口径 `58.31`) | `42.39` (线性口径 `44.19`) | `40.26` (线性口径 `42.31`) |
| **5. 生产管线无参考/锚点评分 (`prod_quality`)** | `86.55` | **`91.04`** (`+4.49`) | `68.86` | `66.05` |
| ↳ 首尾帧像素锚定分 (`S_fl2v_pixel`) | **`98.44`** | **`98.39`** (`-0.05`) | `47.98` | `38.84` |
| ↳ 拉普拉斯清晰度方差 (`sharpness_lap`) | `78.20` | **`161.57`** (`2.07×` 锐度) | `110.04` | **`140.74`** (`1.28×` 锐度) |

---

## 5. 核心实验洞察与生产选档指南

1. **`Quality`（20 步无 LoRA）与 `Fast`（8 步 Turbo LoRA + Sol-Attn）的质量-速度权衡**：
   - 在 `FL2VA` 路由下，`fl2va_bf16_fast`（`Run 02`）与 `fl2va_bf16_quality`（`Run 01`）的首尾帧像素锚定度完全一致（`S_fl2v_pixel = 98.39` vs `98.44`），VBench 综合分几乎无损（`65.86` vs `66.40`），且 8 步蒸馏 LoRA 带来了更高的画面拉普拉斯锐度（`161.57` vs `78.20`，使 `prod_quality_score` 达到 **`91.04`** vs `86.55`），同时端到端耗时从 `241.47s` 骤降至 **`66.35s`（提速 `3.64×`，成本从 `0.4833 CU` 降至 `0.1328 CU`）**。
   - 在 `Ref2VA` 路由下，`ref2va_bf16_fast`（`Run 04`）在 8 步 `Ref2V v1.1 768p Turbo LoRA` + `Run 09 HiFi`（$\tau=1.0$ + 首尾敏感层 `0,1,48,49` 稠密保护）加持下，**`S_vbench` 达到全场最高的 `87.51`**（超越 20 步 `Run 03` 的 `84.98`），主体一致性达 `88.74`、背景一致性达 `95.35`、时序抗闪烁达 `98.36`，同时端到端耗时从 `243.18s` 降至 **`69.98s`（提速 `3.48×`）**。
   - **关于跨权重 `S_ref` 的物理解释**：由于 `Quality` 档使用的是**无 LoRA 原生 20 步基础模型 ($W$)**，而 `Fast` 档加载了 **8 步流匹配蒸馏 LoRA ($W + \Delta W_{\text{LoRA}}$)**，二者在相同随机种子 `seed=42` 下会演化出两条不同的中间帧运动轨迹（尽管音频频谱相关系数仍高达 `0.8719 ~ 0.8836`）。因此逐像素全参考指标 `S_ref`（`23.29` / `17.05`）衡量的是“加/不加蒸馏 LoRA 的轨迹偏移”，而无参考时序指标 `S_vbench`、多特征锚定指标 `S_fl2v` 与 `prod_quality_score` 则直接印证了 `Fast` 档的画质与稳定性。
2. **`FL2VA` 与 `Ref2VA` 两大模型路由的本质互补性**：
   - **`FL2VA`（首尾帧硬约束路由）**：通过 `denoise_mask` 将首帧（`t=0`）与尾帧（`t=123`）硬锁在目标潜变量序列两端，因此首尾帧对齐得分极高（`S_fl2v = 91.70 ~ 93.42`，`S_fl2v_pixel = 98.39 ~ 98.44`），适合**分镜首尾画面已精确绘制、需要严格桥接起止构图**的镜头；但当首尾两张图景别跨度较大（中景街道 → 面部特写）时，硬插值会导致中间帧相机推进幅度较大（`dynamic_degree_px = 27.19 ~ 29.76`），`S_vbench` 落在 `65.86 ~ 66.40`。
   - **`Ref2VA`（多参考主体/风格自由运镜路由）**：将 `<Picture 1>` 与 `<Picture 2>` 编码为前置 `ref_img` 上下文参考 Token（`minimax_refs`），不强制锁定第 0 帧与第 123 帧的像素网格（因此 `S_fl2v` 为 `19.79 ~ 22.01`），从而赋予 DiT 完全自然的 3D 运镜与主体动作生成自由度，使 **VBench 主体一致性（`81.97 ~ 88.74`）、背景一致性（`92.91 ~ 95.35`）、RAFT 运动平滑度（`84.04 ~ 86.62`）与 MUSIQ 美学质量（`61.69 ~ 63.71`）全面跃升，`S_vbench` 高达 `84.98 ~ 87.51`（领先 `FL2VA` 达 `+18.6 ~ +21.7` 分）**，适合**给定角色定妆照/场景设定图（1~7 张）生成高一致性自由运镜镜头**。
3. **生产选档决策表**：
   - **日常多 Seed 抽卡与批量分镜生产**：优先选择 **`fl2va_bf16_fast`**（首尾帧受控，`66.35s / 0.1328 CU`）或 **`ref2va_bf16_fast`**（多图主体参考，`69.98s / 0.1400 CU`），兼具原生 BF16 精度与 `~3.5-3.6×` 端到端加速。
   - **无蒸馏伪影、长镜头复杂物理交互或定版精修**：选择 **`fl2va_bf16_quality`**（`241.47s / 0.4833 CU`）或 **`ref2va_bf16_quality`**（`243.18s / 0.4867 CU`，`beta` 调度器 + 20 步原生 `cuDNN BF16 SDPA`）。
