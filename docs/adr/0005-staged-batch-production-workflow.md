<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# ADR-05：锁定“本地 0-CU 策划攒批预投料 + 云端阶段式 `Qwen3-VL` 去重预编码并卸载 + 常驻 `Run 12` 模式 P1 单流渲染 + 2 秒内网卸载停机”为标准批处理视频生产工作流

- **状态（Status）**：已采纳（Accepted，正式锁定为日常自媒体多镜头与多 Seed 抽卡视频生产的**唯一标准批处理执行协议**）
- **日期（Date）**：2026-10-23
- **关联实验**：[`experiments/05-production-batch-workflow`](../../experiments/05-production-batch-workflow/README.md)（完整实验报告：[`report.md`](../../experiments/05-production-batch-workflow/report.md)；原始实测指标：[`exp05_batch_workflow_metrics.json`](../../experiments/05-production-batch-workflow/outputs/exp05_batch_workflow_metrics.json)；会话账本摘录：[`colab_session_ledger_excerpt.json`](../../experiments/05-production-batch-workflow/outputs/colab_session_ledger_excerpt.json)；一键生产调度器：[`colab_batch_runner.py`](../../experiments/05-production-batch-workflow/code/colab_batch_runner.py)）
- **前置依赖**：
  - [ADR-01：锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）与 HF 直连加载链路](./0001-model-weights-loading-pipeline.md)
  - [ADR-02：锁定 Google Colab Blackwell 96GB（`--gpu G4`）为标准推理机型](./0002-blackwell-gpu-and-sageattention-acceleration.md)
  - [ADR-03：在 Blackwell G4 上正式锁定 `Run 12 [S8-FullStack-Pareto]` 为默认标准推理配置，并确立零浪费 GPU 生命周期工程规范](./0003-sol-attn-and-g4-pareto-frontier.md)
  - [ADR-04：锁定两段式零交互 Google Drive v3 REST API 云端内网中转卸载为唯一默认标准尾端回传方案](./0004-two-stage-gdrive-rest-egress-relay.md)

---

## 1. 背景与问题（Context）

在 [ADR-01](./0001-model-weights-loading-pipeline.md) ~ [ADR-04](./0004-two-stage-gdrive-rest-egress-relay.md) 将单条 `1344×768 × 124 帧` 声画同源视频的核心渲染边际成本压降至 **`45.89s ~ 46.72s`（`0.0918 ~ 0.0935 CU`）**、尾端归档回传压降至 **`1.54s ~ 2.27s`（`0.0031 ~ 0.0045 CU`）** 后，真实自媒体视频生产仍面临三个批处理架构问题：

1. **单任务交互式会话的固定冷启动税占比过高（`~76%`）**：
   - 每次分配一台全新 Colab Blackwell G4 实例，必须支付容器环境初始化（`~22.3s`）、8 线程拉取 `42.03 GB` 黄金五件套（`~115.8s`）、模型加载与 1 步算子 JIT 预热（`~22.4s`）的固定冷启动成本（实测 $T_{\text{cold}} = \mathbf{160.64\text{ s}}$，折合 **`0.3215 CU`**）。若每次仅交互式跑 1 条视频（总计 `212.81s` / `0.4259 CU`），高达 `76%` 的算力消耗在冷启动等待上。
2. **多 Seed 抽卡场景下的冗余 `Qwen3-VL-32B` 编码与显存常驻浪费**：
   - 创作者常对同一分镜（相同 `prompt` 与首尾参考帧 `first_frame` / `last_frame`）提交多个随机种子（`seed`）抽卡。若逐任务重复调用 `Qwen3-VL-32B` + `Video VAE Encode`，既浪费 GPU 时间，又迫使 `15.69 GB` 的 `Qwen3-VL-32B` 与 `14.53 GB` 的 `DiT` 全程同时占用显存。
3. **跨任务 `VAE Decode(K)` 与 `DiT Denoise(K+1)` 应当顺序执行（模式 P1）还是双 CUDA Stream 并发重叠（模式 P2）？**
   - 表面上看，在独立 `torch.cuda.Stream()` 上并发执行上一任务的 `VAE Decode`（`4.33s`）与下一任务的 `DiT Denoise`（`41.56s`）似乎能隐藏 `4.33s` 的解码耗时；但在 `Run 12` 已经将 Blackwell G4 的 188 个 SM 与 GDDR7 带宽压榨至 `~98%` 满载的前提下，双流并发是否会引发严重的 SM 争用与负优化，需要实测裁决。

---

## 2. 实验证据（Evidence）

在实验 05（[`experiments/05-production-batch-workflow`](../../experiments/05-production-batch-workflow/report.md)，G4 节点 `gpu-g4-s-kkb-use5c0-1gq7i6kbo4w4s`）中，我们对 3 个分镜 × 6 个任务（`exp05_run12_batch_6jobs`）进行了全链路实测与对账：

1. **冷启动摊薄实证（$M=6$ 单视频综合 CU 直降 `62.2% ~ 64.9%`）**：
   - 整场会话（含 `50.64s` 的 P2 微消融探针）总挂载 `532.733s`，账本实扣 **`1.0229 CU`**（`2481.6316 -> 2480.6086 CU`）。
   - 扣除 P2 消融探针后的**纯 6 视频生产周期仅 `482.093s`（理论 `0.9648 CU` / 实扣 `0.9257 CU`）**，摊薄后每条 124 帧声画同源视频仅需 **`0.1543` 实扣 CU（拟合稳态 `0.1496 CU`）**，相比单次跑 1 条视频（`0.4259 CU`）**节省 `62.2% ~ 64.9%` 算力**；当批次规模增至 $M=12 \sim 24$ 条时，单条视频总成本进一步降至 **`0.1085 ~ 0.1222 CU`**。
2. **首尾双端 0-CU Google Drive REST 中转实证**：
   - **0-CU 预投料（Ingress）**：G4 启动前，本地以 `0.0000 CU` 将 `input_jobs.tar`（`6.63 MB`）推入 Google Drive（`5.91s`）；G4 启动后在拉取 HF 权重的并发间隙，仅用 **`1.89s`** 从机房内网拉回解压（GPU 关键路径零增量耗时）。
   - **1.54 秒极速卸载（Egress）**：6 条视频归档包 `exp05_outputs_bundle.tar`（`56.43 MB`）在 G4 上仅用 **`1.543s`（`36.58 MB/s`，`0.0031 CU`）** 直传 Google Drive 并立即 `colab stop`；随后本地在晚高峰慢网（`0.97 MB/s`）下拉取耗时 `57.93s` 消耗 **`0.0000 CU`**，避免了传统 `colab download` 挂机阻塞 GPU `~81.5s`（`0.1631 CU`），实测节省 **`98.1%`** 尾端算力。
3. **阶段式条件去重缓存与 `Qwen3-VL-32B` 卸载实证**：
   - 6 个任务（3 个唯一分镜 + 3 个多 Seed 变体）通过 `cond_hash` 去重后，仅编码 3 次（`5.134s + 2.816s + 2.800s = 10.750s`），3 个变体任务 **`0.0000s` 命中缓存**，比朴素 6 次编码（`19.174s`）**省时 `8.424s`（`-43.94%`）**。
   - 编码完毕立即卸载 `Qwen3-VL-32B`（`0.954s`），**当场释放 `17.213 GiB` 显存（`17.222 -> 0.009 GiB`）**，使后续 6 任务连续去噪峰值显存稳定在 **`27.41 GiB`**。
4. **模式 P1 vs. 模式 P2（双 CUDA Stream 并发重叠）微消融裁决**：
   - **模式 P1（顺序 `VAE(K) -> DiT(K+1)` + 后台 CPU `ThreadPoolExecutor` 异步 `ffmpeg`）**：单次跨任务过渡耗时 `4.3351s + 41.0038s = 45.3389s`，峰值显存 `27.413 GiB`，后台 `10.613s` 的 CPU `ffmpeg` 合成 100% 重叠（6 任务总交接间隙仅 `0.545s` = `90.9 ms/任务`）。
   - **模式 P2（双流并发 `VAE(K) || DiT(K+1)`）**：因 `Run 12` 已占满 G4 SM 与显存带宽，并发争用导致 `VAE Decode` 耗时暴涨 **`10.51x`**（`4.335s -> 45.553s`）、`DiT Denoise` 耗时拉长 `1.085x`（`41.004s -> 44.503s`），双流重叠总墙钟耗时 **`45.5819s`**，比模式 P1 **反而慢了 `0.243s`（`-0.54%` 负优化）**，同时峰值显存暴涨 `+0.591 GiB`（至 `28.004 GiB`）且丧失比特级确定性（`PSNR = 32.08 dB`）。

---

## 3. 决策内容（Decision）

自本 ADR 起，所有基于 Colab Blackwell G4 的批量视频生成与多镜头/多种子生产任务，正式锁定采用 **三阶段异构批处理生产工作流（Three-Stage Heterogeneous Batch Production Workflow）**：

### 3.1 第一阶段：本地 Mac 零成本策划、校验与 0-CU 预投料（`0.00 CU/hr`）
1. 所有分镜提示词、首尾参考帧（统一裁剪/对齐为 `1344×768` PNG）与任务队列（仅需指定 `job_id`、`shot_id`、`seed`，推理配置全局唯一锁定 [ADR-03](./0003-sol-attn-and-g4-pareto-frontier.md) 的 `Run 12 [S8-FullStack-Pareto]`）均在本地编写为 `batch_manifest.json`。
2. 在启动任何 Colab GPU 会话之前，必须先运行 [`local_planner.py`](../../experiments/05-production-batch-workflow/code/local_planner.py) 完成静态 Schema 校验、`cond_hash` 去重预分析、打包未压缩 `input_jobs.tar`，并以 **`0.00 CU`** 直传至 Google Drive v3 REST API 获取 `input_file_id`。

### 3.2 第二阶段：云端 Blackwell G4 三步阶段式常驻批处理（`7.2046 CU/hr`）
1. **并发入向拉取（Overlapped Ingress）**：G4 启动后，在 8 线程并发下载 `42.03 GB` 黄金五件套的同时，后台线程从 Google Drive 机房内网（`< 2s`）拉回 `input_jobs.tar` 并解压至 `/content/job_inputs`。
2. **阶段 1 — 批量条件去重预编码与 `Qwen3-VL-32B` 立即卸载**：
   - 先仅加载 `Qwen3-VL-32B` 与 `Video/Audio VAE`，按 `cond_hash = sha256(prompt, neg_prompt, first_frame_sha256, last_frame_sha256)` 仅对唯一分镜执行一次条件编码，同分镜多 Seed 任务 `0.00s` 直接引用缓存字典。
   - **强制规范**：全部唯一分镜条件编码完毕后，**必须立即销毁 `Qwen3-VL-32B` 并调用 `torch.cuda.empty_cache()` 释放 `17.21 GiB` 显存**，严禁携载 `Qwen3-VL-32B` 进入 `DiT` 去噪阶段。
3. **阶段 2 — 常驻 `Run 12` 连续单流渲染 + 后台 CPU 异步封装（锁定模式 P1，禁用模式 P2）**：
   - 加载 `DiT` 并一次性注入 `Run 12 [S8-FullStack-Pareto]` 算子栈（前 2/8 步 `SageAttention` + 后 6/8 步直通 `ck.sol_attn` `tau=1.3, extra_tokens=256` + `MiniMaxH3FusedModulation`）常驻显存，任务切换严禁重打补丁（`model_repatch_sec = 0.000s`）。
   - **锁定模式 P1**：GPU 默认流严格按顺序执行 `DiT_Denoise(K) -> VAE_Decode(K)`，严禁开启双 CUDA Stream 跨任务并发 `VAE(K) || DiT(K+1)`（模式 P2）。
   - 每个任务的 `VAE Decode` 完成后，立即将主机内存张量投入后台 `ThreadPoolExecutor(max_workers=3)` 异步执行 `ffmpeg` H.264+AAC 音画合成、质量评分与 `manifest_results.json` 增量落盘，GPU 主流零等待切入任务 $K+1$。
4. **阶段 3 — 2 秒机房内网卸载与毫秒级 `colab stop`（[ADR-04](./0004-two-stage-gdrive-rest-egress-relay.md) Stage 1）**：
   - 等待后台 `ffmpeg` 线程池收尾后，使用未压缩 `tar -cf` 打包全部 MP4 与 JSON，直传 Google Drive v3 REST API（`~1.5s`），并在 `finally` 块中**立即触发 `colab stop` 销毁 GPU 实例**。

### 3.3 第三阶段：停机后本地 0-CU 并发拉取、按分镜归档与宫格审片（`0.00 CU/hr`）
1. 确认 G4 已停机（`0.00 CU/hr`）后，本地通过 8 线程 `HTTP Range` 并发拉回归档包并自动清理 Google Drive 临时文件（`ingress_file_id` 与 `egress_file_id` 双清）。
2. 自动按 `outputs/by_storyboard/{shot_id}/` 分类归档 MP4，并在本地 CPU 免费合成多镜头关键帧审片矩阵图与同步对比宫格视频。

---

## 4. 影响与后果（Consequences）

- **单条视频生产综合算力成本降低 `62.2% ~ 75.3%`**：在 $M=6 \sim 32$ 的日常攒批规模下，每 `100 CU` 可稳定产出 **`668 ~ 951` 条** `1344×768 × 124 帧` 高清声画同源视频（相比单跑模式的 `235` 条提升 **`2.85x ~ 4.05x`**）。
- **显存安全裕度高达 `67.5 GiB`**：由于在去噪前卸载了 `17.21 GiB` 的 `Qwen3-VL-32B` 且禁用了双流并发，峰值显存恒定在 `27.41 GiB`，彻底杜绝长批次运行时的显存碎片化或 OOM 风险。
- **形成 ADR-01 ~ ADR-05 完整工程闭环**：从入向权重加载（ADR-01）、硬件与注意力底座（ADR-02/03）、出向极速中转（ADR-04）到全周期批处理编排（ADR-05），全部固化为单文件 `< 500` 行的自动化 CLI 生产管线。
