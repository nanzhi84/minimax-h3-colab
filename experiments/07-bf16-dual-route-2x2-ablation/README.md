<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# 实验 07：全浮点 BF16 底座下 `FL2VA` × `Ref2VA` 两模型两档位（2×2 矩阵）速度与四维质量消融

- **实验编号**: `07-bf16-dual-route-2x2-ablation`
- **完整实验报告**: [`report.md`](./report.md)（实测指标 JSON：[`outputs/exp07_bf16_2x2_metrics.json`](./outputs/exp07_bf16_2x2_metrics.json)，会话账本摘录：[`outputs/colab_session_ledger_excerpt.json`](./outputs/colab_session_ledger_excerpt.json)，四宫格 124 帧对比视频：[`outputs/exp07_2x2_quad_grid_comparison.mp4`](./outputs/exp07_2x2_quad_grid_comparison.mp4)）
- **关联架构文档**: [系统架构总览 (`docs/architecture/overview.md`)](../../docs/architecture/overview.md), [ADR-03: Blackwell G4 Sol-Attn 帕累托前沿](../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md), [ADR-04: Google Drive 二阶段零浪费传输](../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md), [ADR-05: 显存常驻与分阶段管线](../../docs/adr/0005-staged-batch-production-workflow.md)
- **前置实验**: [实验 03: Blackwell G4 Sol-Attn 帕累托前沿消融](../03-g4-sol-attn-pareto-ablation/report.md), [实验 05: 异构分阶段批量生产工作流](../05-production-batch-workflow/report.md)
- **目标硬件**: Google Colab **Blackwell G4 (`--gpu G4`, `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 94.97 GiB GDDR7, SM 12.0)**，账本实测费率 **`7.2046 CU/hr`** (`0.00200128 CU/s`)
- **测试底座**: 全套**非量化原生浮点权重**（`minimax_h3_fl2va_pruned_bf16` `40.23 GB`、`minimax_h3_ref2va_pruned_bf16` `40.23 GB`、`qwen3vl_32b_minimax_h3_bf16` `51.51 GB`、`minimax_h3_video_vae_fp16` `5.21 GB`、`minimax_h3_audio_vae_fp32` `0.61 GB`，以及两套 BF16 8-Step Turbo LoRA，7 个权重文件合计 `141.68 GB`）
- **实测消耗**: 单次 G4 会话分配至释放总墙钟 `1611.50s`（理论折算 **`3.2251 CU`**，账本净扣减 **`3.7114 CU`**），其中 4 组 2×2 BF16 端到端核心推理总耗时 `620.98s`（**`1.2428 CU`**），Stage 2 本地 8 线程拉取 `49.90 MB` 产物包耗时 `24.52s`（消耗 **`0.0000 CU`**）

---

## 1. 实验目标与 2×2 矩阵设计

本实验验证生产管线（`studio/api/`）升级为**全浮点 BF16 底座 + 本地按需路由两模型 × 两档位（2×2 共 4 个 Profile）**后的真实生成速度、显存水位、算子调用命中率与四维质量指标：

| Run ID | 档位 ID (`profile_id`) | 模型路由 (`model_route`) | DiT 底座 (`BF16`) + LoRA | 采样步数与调度器 | 注意力与算子融合配置 | 单条生产下载体积 |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| **Run 01** | `fl2va_bf16_quality` | `fl2va` (`MiniMaxH3ImageToVideo`) | `fl2va_pruned_bf16` (`40.23 GB`)，无 LoRA | `20` 步, `simple` | PyTorch 2.11 `cuDNN BF16 SDPA` (`100%` 稠密) + `MiniMaxH3FusedModulation` | **`97.55 GB`** (4 文件) |
| **Run 02** | `fl2va_bf16_fast` | `fl2va` (`MiniMaxH3ImageToVideo`) | `fl2va_pruned_bf16` + `fl2v_turbo_8step_v1.0_bf16` (`1.96 GB`) | `8` 步, `simple` | **`Run 12`**: 前 2 步 (`start=0.20`) `SageAttention` + 后 6 步直通 `ck.sol_attn` (`tau=1.3, tok=256`) + `FusedMod` | **`99.51 GB`** (5 文件) |
| **Run 03** | `ref2va_bf16_quality` | `ref2va` (`MiniMaxH3ReferenceToVideo`) | `ref2va_pruned_bf16` (`40.23 GB`)，无 LoRA | `20` 步, `beta` | PyTorch 2.11 `cuDNN BF16 SDPA` (`100%` 稠密) + `MiniMaxH3FusedModulation` | **`97.55 GB`** (4 文件) |
| **Run 04** | `ref2va_bf16_fast` | `ref2va` (`MiniMaxH3ReferenceToVideo`) | `ref2va_pruned_bf16` + `ref2v_turbo_8step_v1.1_768p_bf16` (`1.96 GB`) | `8` 步, `simple` | **`Run 09 HiFi`**: 前 2 步 `SageAttention` + 后 6 步直通 `ck.sol_attn` (`tau=1.0, tok=256, dense_blocks="0,1,48,49"`) + `FusedMod` | **`99.51 GB`** (5 文件) |

---

## 2. 核心实测结果速览（`1344×768`, 124 帧, 24 FPS, `seed=42`）

| Run ID | `profile_id` | 步数 | 去噪耗时 (`s`) | 单步耗时 (`s/step`) | 端到端 GPU 耗时 (`s`) | 端到端成本 (`CU`) | 同路由加速比 | 峰值显存 (`GiB`) | `S_ref` (同路由) | `S_vbench` | `S_fl2v` | `S_audio` | `Q_total` (同路由) | `prod_quality` (`S_fl2v_pixel`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 01** | `fl2va_bf16_quality` | 20 | `234.41s` | `11.72s` | **`241.47s`** | **`0.4833 CU`** | `1.00×` | `48.24` | `100.00` | `66.40` | **`93.42`** | `93.43` | **`88.97`** | `86.55` (**`98.44`**) |
| **Run 02** | `fl2va_bf16_fast` | 8 | **`59.19s`** | **`7.40s`** | **`66.35s`** | **`0.1328 CU`** | **`3.64×`** (去噪 **`3.96×`**) | `48.24` | `23.29` | `65.86` | **`91.70`** | `57.78` | `56.21` | **`91.04`** (**`98.39`**) |
| **Run 03** | `ref2va_bf16_quality` | 20 | `235.69s` | `11.78s` | **`243.18s`** | **`0.4867 CU`** | `1.00×` | `43.07` | `100.00` | **`84.98`** | `22.01` | **`94.96`** | **`75.99`** | `68.86` (`47.98`) |
| **Run 04** | `ref2va_bf16_fast` | 8 | **`62.87s`** | **`7.86s`** | **`69.98s`** | **`0.1400 CU`** | **`3.48×`** (去噪 **`3.75×`**) | `48.25` | `17.05` | **`87.51`** | `19.79` | `63.23` | `42.28` | `66.05` (`38.84`) |

---

## 3. 代码结构与复现指南

### 3.1 实验代码结构（全部位于 `code/` 目录且单文件 `< 500` 行）
- [`exp07_config.py`](./code/exp07_config.py)：7 个 BF16 权重文件清单、符合 [`h3-prompt-writing`](../../.agents/skills/h3-prompt-writing/SKILL.md) 规范的 `FL2VA`（XML 三标签）与 `Ref2VA`（六段式 `<Picture 1>` / `<Picture 2>`）提示词、2×2 矩阵定义与算子调用计数器强校验。
- [`eval_metrics.py`](./code/eval_metrics.py)：与实验 02/03 完全一致的四维评估器 `QualityEvaluator`（`S_ref`, `S_vbench`, `S_fl2v`, `S_audio`, `Q_total`）及生产快评指标 `compute_production_quality_metrics`（`S_fl2v_pixel`, `prod_quality_score`）。
- [`remote_exp07_engine.py`](./code/remote_exp07_engine.py)：远端 Colab G4 执行引擎（8 线程并发拉取 + Phase 2 双路由预编码与 `Qwen3-VL-32B BF16` 即时卸载释放 `52.82 GiB` 显存 + 分桶执行 `FL2VA` / `Ref2VA` + Stage 1 Google Drive 极速回传）。
- [`run_exp07_colab.py`](./code/run_exp07_colab.py)：本地零浪费编排器（自动发现活跃 GCP 配额项目、`try...finally` 瞬时 `colab stop`、0-CU Stage 2 多线程拉取与账本对账）。
- [`render_visuals_local.py`](./code/render_visuals_local.py)：本地 Mac 零 CU 图表与 2×2 四宫格 124 帧对比视频渲染器。

### 3.2 一键复现与生产 CLI 路由调用
```bash
# 1. 运行实验 07 端到端 2x2 BF16 矩阵对比（自动关停 G4 并在本地生成图表与四宫格对比视频）
python3 experiments/07-bf16-dual-route-2x2-ablation/code/run_exp07_colab.py

# 2. 生产管线本地路由调用示例（按需只下载对应路由的 97.5~99.5 GB BF16 权重）：
# (a) FL2VA + Fast 档位 (8步 Run 12)
python3 studio/api/colab_batch_runner.py -p demo --route fl2va --profile fast \
  --first-frame studio/api/default_assets/first_frame.png \
  --last-frame studio/api/default_assets/last_frame.png \
  --prompt "<integrated_multimodal_description>...</integrated_multimodal_description>..." --dry-run

# (b) Ref2VA + Quality 档位 (20步 cuDNN BF16 SDPA + beta 调度器)
python3 studio/api/colab_batch_runner.py -p demo --route ref2va --profile quality \
  --ref-images studio/api/default_assets/first_frame.png,studio/api/default_assets/last_frame.png \
  --prompt "[Visual Reference Description]\n<Picture 1>: ..." --dry-run
```
