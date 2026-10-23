<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# 实验 05 报告：基于 `Run 12 [S8-FullStack-Pareto]` 的异构阶段式批处理视频生产工作流 — 0-CU 双向云盘中转、SHA-256 条件去重缓存与 P1/P2 流水线微消融实测

> **实验日期**：2026-10-23  
> **测试平台**：Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (`--gpu g4`, Compute Capability `12.0`, `94.97 GiB` VRAM, 48 vCPU, `176.88 GiB` RAM, 节点 `gpu-g4-s-kkb-use5c0-1gq7i6kbo4w4s`) + 本地 macOS 控制平面  
> **软件栈**：`PyTorch 2.11.0+cu130`, `CUDA 13.0`, `Triton 3.6.0`, `SageAttention 1.0.6`, `comfy-kitchen 0.2.40` (`cuda` 后端原生支持 `sol_attn`)  
> **全局锁定推理配置（[ADR-03](../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md)）**：`Run 12 [S8-FullStack-Pareto]`（前 2/8 步 `SageAttention` 稠密预热 `start_percent=0.20` + 后 6/8 步零拷贝直通 `ck.sol_attn` `tau=1.3, extra_tokens=256, sink_conditioning="exact_kv_and_rows"` + `MiniMaxH3FusedModulation`，**全局统一不分档**）  
> **会话账本核对**：会话前余额 **`2481.6316 CU`** $\rightarrow$ 会话停机后余额 **`2480.6086 CU`**，整场 6 条 124 帧声画同源视频生产 + P1/P2 双流微消融实验**账本实际净扣减仅 `1.0229 CU`**（`532.733s` 总挂载时长；若扣除 `50.640s` 的 P2 微消融探针，纯 6 视频生产周期仅 **`482.093s` / `0.9257` 实扣 CU**，折合每条视频仅 **`0.1543` 实扣 CU / `0.1608` 理论 CU**！）

---

## 1. 核心结论速览（Executive Summary）

在前序实验 01 ~ 04 完成了**权重加载链路（[ADR-01](../../docs/adr/0001-model-weights-loading-pipeline.md)）**、**Blackwell 硬件选型（[ADR-02](../../docs/adr/0002-blackwell-gpu-and-sageattention-acceleration.md)）**、**`Run 12 [S8-FullStack-Pareto]` 帕累托最优推理栈（[ADR-03](../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md)）**与**两段式 Google Drive REST 尾端卸载（[ADR-04](../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md)）**的基础上，实验 05 构建并实测验证了面向自媒体多镜头与多 Seed 抽卡场景的 **三阶段异构批处理生产工作流（Three-Stage Heterogeneous Batch Production Workflow）**。

通过真实生成 **3 个分镜脚本 × 6 个渲染任务（共 6 条 `1344×768 × 124 帧` 24fps H.264+AAC 声画同源视频）** 并同场完成阶段式消融，我们得出四大核心工程结论：

1. **冷启动固定税大幅摊薄，单条视频综合 CU 成本直降 `62.2% ~ 64.9%`**：
   - 每次新建 Colab G4 会话存在固定冷启动成本 $T_{\text{cold}} = \mathbf{160.64\text{ s}}$（折合 **`0.3215 CU`**，含容器环境就绪 `22.29s`、8 线程并发拉取 `42.03 GB` 黄金五件套 `115.81s`、`DiT` 加载与 `Run 12` 补丁注入 `2.33s`、1 步算子 JIT 预热 `16.86s`）。
   - 若交互式单次仅跑 1 条视频（$M=1$，总耗时 `212.81s` / **`0.4259 CU`**），固定冷启动占比高达 **`76.02%`**；而采用本工作流单次攒批 $M=6$ 条任务连续执行，纯生产会话耗时 **`482.09s`**，每条 124 帧视频的摊薄总成本从 `0.4259 CU` 降至 **`0.1496 ~ 0.1608 CU`（账本实扣折合 `0.1543 CU/条`）**，**单条视频综合算力成本直降 `62.2% ~ 64.9%`**！当批次规模扩展至 $M=12 \sim 24$ 条时，单条综合成本进一步逼近 **`0.1085 ~ 0.1222 CU`**（核心渲染有效占比达 `77.8% ~ 87.4%`）。
2. **首尾双端 0-CU Google Drive REST 内网中转：消除跨境慢网对昂贵 GPU 的双向阻塞**：
   - **入向投料（0-CU Ingress）**：在启动 G4 之前，本地编排器以 **`0.0000 CU`** 将首尾帧 PNG 与任务清单打包为 `input_jobs.tar`（`6.63 MB`）预推至 Google Drive（耗时 `5.91s`）；G4 启动后在 8 线程拉取 HF 权重的并发间隙，通过 Google 机房内网仅用 **`1.89s`** 拉回并解压，对 GPU 关键路径的增量耗时严格为 **`0.000s`**。
   - **出向卸载（[ADR-04](../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) Egress）**：6 条视频与评测日志在远端打包为未压缩 `exp05_outputs_bundle.tar`（`56.43 MB` 仅需 `0.024s`），通过机房内网直传 Google Drive 仅耗时 **`1.543s`（`36.58 MB/s`，消耗 `0.0031 CU`）**，随后立即触发 `colab stop` 切断 GPU 计费！
   - **真实弱网防线实证**：本场实验恰逢本地跨境网络处于晚高峰慢速窗口（阶段 2 本地拉取 `56.43 MB` 耗时 `57.93s`，有效吞吐 `0.97 MB/s`）。因为此时 G4 实例早已销毁，这 `57.93s` 消耗了 **`0.0000 CU`**；若仍采用传统 `colab download`（含 `+36.84%` Base64 膨胀即 `77.22 MB`）挂着 GPU 硬拉，将额外阻塞 G4 长达 **`~81.5s`（白白烧掉 `0.1631 CU`，相当于多付 1.8 条视频的算力费）**，实测节省 **`98.1%`** 的尾端 GPU 算力！
3. **阶段式批量条件预编码 + SHA-256 去重缓存 + 立即卸载 `Qwen3-VL-32B`（省时 `43.94%`、释放 `17.21 GiB` 显存）**：
   - 针对自媒体常用的“同分镜多 Seed 抽卡”场景（6 个任务包含 3 个唯一分镜），引擎通过计算 `cond_hash = sha256(prompt, neg_prompt, first_frame_sha256, last_frame_sha256)` 仅对 3 个唯一分镜执行 `Qwen3-VL-32B` + `Video VAE` 编码（耗时 `5.134s + 2.816s + 2.800s = 10.750s`），其余 3 个多 Seed 变体任务（`job_02`、`job_03`、`job_05`）**`0.0000s` 直接命中显存缓存**，相比朴素逐任务重复编码（`19.174s`）**净省 `8.424s`（`-43.94%`）**。
   - 3 个唯一分镜编码完毕的瞬间，引擎立即彻底卸载 `Qwen3-VL-32B`（耗时 `0.954s`），**当场释放 `17.213 GiB` 显存（从 `17.222 GiB` 降至 `0.009 GiB`）**，使得后续 `DiT` 常驻基线显存仅 `19.54 GiB`、6 任务去噪全程峰值显存稳定在 **`27.41 GiB`**（不到 G4 `94.97 GiB` 显存的 `29%`）。
4. **常驻 `Run 12` 流水线微消融（模式 P1 vs. 模式 P2）：证伪双 CUDA Stream 并发重叠，锁定“单流 GPU + 异步 CPU `ffmpeg`”（模式 P1）**：
   - 在连续消费 6 个任务期间，由于全局唯一锁定 `Run 12 [S8-FullStack-Pareto]`，常驻 `DiT` 模型的**重打补丁开销严格为 `0.000s`**，任务间状态重置仅需 **`1.8 ~ 3.6 ms`（均值 `3.1 ms`）**；6 任务平均去噪耗时 **`41.561s`**（甚至比实验 03 单跑基准 `42.37s` 还快 `0.81s`）、平均 `Video+Audio VAE` 解码耗时 **`4.330s`**、平均单视频 GPU 端到端耗时 **`45.892s`（边际成本仅 `0.0918 CU`）**。
   - **模式 P1（GPU 单流顺序 `DiT -> VAE` + 后台 CPU `ThreadPoolExecutor` 异步 `ffmpeg`）**：6 条视频总计 `10.613s` 的 CPU H.264+AAC 音画合成与质量评分被 **100% 隐藏在下一任务的 GPU 去噪阴影下**，6 个任务总循环墙钟时间（`275.897s`）与纯 GPU 算子总耗时（`275.351s`）之间仅相差 **`0.545s`（平均每任务 CPU 交接间隙仅 `90.9 ms`）**。
   - **模式 P2（主/副双 CUDA Stream 并发重叠 `VAE_Decode(K) || DiT_Denoise(K+1)`）实测证伪**：由于 `Run 12` 在 Blackwell G4 上已将 188 个 SM 与 GDDR7 显存带宽压榨至 `~98%` 满载，当在副流 `vae_stream` 上并发启动任务 $K$ 的 `VAE Decode` 时，严重的 SM 与 L2/GDDR7 带宽争用导致 **`VAE Decode` 耗时从 `4.335s` 暴涨 `10.51x` 至 `45.553s`**，同时主流 `DiT Denoise` 也被拖慢 `1.085x`（从 `41.004s` 增至 `44.503s`），最终双流并发总墙钟耗时为 **`45.582s`**，不仅没有节省时间，反而比模式 P1 顺序执行（`45.339s`）**倒亏 `-0.243s`（`-0.54%` 负优化）**，且导致瞬时显存峰值暴涨 **`+0.591 GiB`**（`27.413 -> 28.004 GiB`）并引入非确定性浮点微扰（`PSNR = 32.08 dB`）。因此，生产环境**正式锁定模式 P1，严禁开启双 CUDA Stream 跨模型并发**。

![Experiment 05 冷启动摊薄曲线、SHA-256 条件去重收益与 P1/P2 流水线微消融](outputs/batch_amortization_and_lifecycle_cu.png)

---

## 2. 全生命周期阶段耗时与账本精确对账（Session Lifecycle & Ledger Reconciliation）

本实验严格遵循 [ADR-03](../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md) 与 [ADR-04](../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) 的零浪费生命周期纪律，从本地 0-CU 预投料到云端执行、再到 1.54 秒极速卸载停机与本地 0-CU 拉取，完整时间轴与账本核对如下：

| 阶段编号与名称 | 执行位置 | 是否消耗 GPU CU | 实测墙钟耗时 (s) | 折合理论算力 (CU @ `7.2046/hr`) | 占纯生产会话比例 (`482.09s`) | 关键工程动作与实测指标 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 0: 本地清单校验与 0-CU 预投料** | 本地 Mac | **否 (`0.00 CU`)** | `5.914s` | `0.0000 CU` | `0.00%` | 校验 3 分镜 × 6 任务，打包 `input_jobs.tar` (`6.63 MB`) 直传 Google Drive (`file_id=1FAp...`) |
| **Phase 0: G4 分配与内核握手 + 环境就绪** | 云端 G4 | 是 | `26.379s`<br>(`4.09s + 22.29s`) | `0.0528 CU` | `5.47%` | `colab new -s h3-exp05-g4 --gpu g4` 分配节点 `use5c0` + 挂载自定义节点与声画补丁 |
| **Phase 1: 8 线程拉取 42.03GB 权重 + 内网拉取任务包** | 云端 G4 | 是 | `115.807s`<br>*(任务包 `1.89s` 全重叠)* | `0.2318 CU` | `24.02%` | 8 线程并发下载黄金五件套（聚合吞吐 `362.9 MB/s`，串行和 `271.90s`）；并发内网拉回 `input_jobs.tar` 仅 `1.89s` |
| **Phase 2: 阶段式批量条件编码 + 卸载 `Qwen3-VL`** | 云端 G4 | 是 | `13.948s` | `0.0279 CU` | `2.89%` | 加载 `Qwen3-VL + VAE` (`2.24s`) $\rightarrow$ 编码 3 个唯一分镜 (`10.75s`) + 3 个缓存命中 (`0.00s`) $\rightarrow$ 卸载 `Qwen3-VL` (`0.95s`，释放 `17.21 GiB` VRAM) |
| **Phase 3: 常驻 `DiT` 加载 + `Run 12` 注入与 1 步预热** | 云端 G4 | 是 | `19.185s`<br>(`2.33s + 16.86s`) | `0.0384 CU` | `3.98%` | 加载 `14.53 GB` `nvfp4_awq` DiT 并一次性挂载 `Run 12` 算子栈 (`2.33s`) + 1 步 JIT 预热 (`16.86s`) |
| **Phase 4A: 连续消费 6 条 124 帧视频 (`Mode P1`)** | 云端 G4 + 后台 CPU | 是 | **`275.897s`**<br>*(单条均值 `45.98s`)* | **`0.5521 CU`**<br>*(单条 `0.0920 CU`)* | **`57.23%`** | 6 条 `1344×768×124f` 连续 `Run 12` 推理（GPU 算子累计 `275.35s`，后台 CPU 异步 `ffmpeg` `10.61s` 100% 重叠） |
| *Phase 4B (仅实验附加): `Mode P2` 双流重叠消融探针* | *云端 G4* | *是 (实验附加)* | *`50.640s`* | *`0.1013 CU`* | *(不计入常规生产)* | *实测双 CUDA Stream `VAE(Job06) || DiT(Job01)` SM 争用与精度对比* |
| **Phase 5 (Stage 1): 远端 `.tar` 归档 + 机房内网直传 GDrive + `colab stop`** | 云端 G4 | 是 | **`1.567s`**<br>(`0.024s + 1.543s`) | **`0.0031 CU`** | **`0.33%`** | 打包 `56.43 MB` `exp05_outputs_bundle.tar` (`0.024s`) + 直传 GDrive (`1.543s`, `36.58 MB/s`) + 立即销毁 G4 |
| **Stage 2: 停机后本地 8 线程拉取 + 解包 + 渲染对比图** | 本地 Mac | **否 (`0.00 CU`)** | `57.929s` (拉取)<br>+ `6.8s` (渲染) | `0.0000 CU` | `0.00%` | G4 已停机 (`0.00 CU/hr`)；本地拉回 `56.43 MB` 归档包并通过 SHA-256 校验，自动清理云盘临时文件 |
| **纯 6 视频生产周期合计（扣除 Phase 4B 探针）** | **云端 G4** | **是** | **`482.093s`** (`8.03 min`) | **`0.9648 CU` (理论)**<br>**`0.9257 CU` (实扣)** | **`100.00%`** | **摊薄后单条视频仅 `80.35s` / `0.1608` 理论 CU (`0.1543` 实扣 CU)，核心工作占比达 `60.12%`！** |
| **含 Phase 4B 消融探针的完整会话账本合计** | **云端 G4** | **是** | **`532.733s`** (`8.88 min`) | **`1.0661 CU` (理论)**<br>**`1.0229 CU` (实扣)** | — | **账本精确核对：`2481.6316 CU -> 2480.6086 CU`（等效结算费率 `6.9125 CU/hr`）** |

> **账本核对溯源文件**：完整会话分配与释放记录保存在 [`outputs/colab_session_ledger_excerpt.json`](outputs/colab_session_ledger_excerpt.json)，云端原始标准输出日志保存在 [`outputs/exp05_batch_raw.log`](outputs/exp05_batch_raw.log)，全量结构化指标保存在 [`outputs/exp05_batch_workflow_metrics.json`](outputs/exp05_batch_workflow_metrics.json)。

---

## 3. 三项核心工程优化与微消融实测分析（Detailed Ablations）

### 3.1 消融一：阶段式批量条件预编码 + SHA-256 哈希去重缓存 vs. 朴素逐任务重复编码

在自媒体视频创作中，创作者通常会对同一个分镜（相同的首尾锚定帧 `first_frame` / `last_frame` 与提示词 `prompt`）设置 2~4 个不同的随机种子（`seed`）进行“抽卡优选”。
- **传统朴素流水线的问题**：每个任务在进入去噪前都要独立调用一次 `Qwen3-VL-32B` 文本编码器与 `Video VAE.encode`，不仅对相同分镜做冗余计算，而且要求 `15.69 GB` 的 `Qwen3-VL-32B` 与 `14.53 GB` 的 `DiT` 全程同时常驻在显存中。
- **本工作流的阶段式去重预编码（Phase 2）**：
  1. 在 [`manifest_schema.py`](code/manifest_schema.py#L61-L82) 中，通过 `compute_conditioning_hash()` 对 `(prompt, negative_prompt, first_frame_sha256, last_frame_sha256, width, height, num_frames)` 计算唯一 SHA-256 签名 `cond_hash`。
  2. 在 G4 启动后的 **Phase 2**，先仅加载 `Qwen3-VL-32B` 与 `Video/Audio VAE`（耗时 `2.242s`），遍历批次内的 6 个任务：

| 任务序号与 `job_id` | 所属分镜 `shot_id` | 随机种子 `seed` | 条件签名 `cond_hash` | 缓存状态 (`cache_hit`) | 阶段式去重编码耗时 (s) | 朴素逐任务编码耗时 (s) | 单任务节省耗时 (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`job_01_shot01_seed42`** | `shot_01_cyber_mouse_orbit` | `42` | `d9854f808d9dbcec` | 首次编码 (`MISS`) | `5.1337s` *(含首次算子初始化)* | `5.1337s` | `0.0000s` |
| **`job_02_shot01_seed108`** | `shot_01_cyber_mouse_orbit` | `108` | `d9854f808d9dbcec` | **命中缓存 (`HIT`)** | **`0.0000s`** | `2.8080s` | **`+2.8080s`** |
| **`job_03_shot01_seed2026`** | `shot_01_cyber_mouse_orbit` | `2026` | `d9854f808d9dbcec` | **命中缓存 (`HIT`)** | **`0.0000s`** | `2.8080s` | **`+2.8080s`** |
| **`job_04_shot02_seed77`** | `shot_02_neon_circuit_pulse` | `77` | `35cbd493250c68da` | 首次编码 (`MISS`) | `2.8158s` | `2.8158s` | `0.0000s` |
| **`job_05_shot02_seed888`** | `shot_02_neon_circuit_pulse` | `888` | `35cbd493250c68da` | **命中缓存 (`HIT`)** | **`0.0000s`** | `2.8080s` | **`+2.8080s`** |
| **`job_06_shot03_seed99`** | `shot_03_hero_reveal_finale` | `99` | `9c41b2e51c9d1330` | 首次编码 (`MISS`) | `2.8002s` | `2.8002s` | `0.0000s` |
| **6 任务条件编码总计** | **3 个唯一分镜** | — | **命中率 `3/6 = 50.0%`** | **3 次 `0.00s` 复用** | **`10.7497s`** (`0.0215 CU`) | **`19.1737s`** (`0.0384 CU`) | **净省 `8.4240s` (`-43.94%`)** |

  3. **立即卸载 `Qwen3-VL-32B` 释放显存**：在 3 个唯一分镜的条件张量（仅占 `~9 MiB` 显存）生成完毕后，引擎立即销毁 `Qwen3-VL-32B` 实例并执行 `torch.cuda.empty_cache()`（仅耗时 `0.954s`），将活跃显存从 **`17.222 GiB` 瞬间压降至 `0.009 GiB`（净释放 `17.213 GiB` 显存！）**。

---

### 3.2 消融二：连续 `Run 12` 渲染流水线模式 P1（单流 GPU + 异步 CPU `ffmpeg`）vs. 模式 P2（双 CUDA Stream 重叠 `VAE(K) || DiT(K+1)`）

在设计阶段，一个非常自然的工程设想是：**能否利用副 CUDA Stream (`vae_stream`) 在执行任务 $K$ 的 `VAE Decode`（`~4.33s`）的同时，让主 CUDA Stream (`default_stream`) 并发启动任务 $K+1$ 的 `DiT [Run 12]` 去噪（`~41.00s`），从而白赚 `4.33s` 的跨任务过渡时间？**

我们在 Blackwell G4 上使用完全相同的任务对（`Job 06` 的潜变量执行 `VAE Decode` + `Job 01` 的条件与种子 `seed=42` 执行 `DiT [Run 12]` 8 步去噪）进行了严密的背靠背实测（代码位于 [`remote_batch_engine.py`](code/remote_batch_engine.py#L317-L386)）：

| 评测维度 | **流水线模式 P1（生产锁定）**：<br>GPU 单流顺序 `VAE(K) -> DiT(K+1)` + 后台 CPU 异步 `ffmpeg` | **流水线模式 P2（双流并发对照）**：<br>副流 `VAE(K)` $\parallel$ 主流 `DiT(K+1)` 双 CUDA Stream 重叠 | 模式 P2 相对模式 P1 的实测差异与物理机制解释 |
| :--- | :---: | :---: | :--- |
| **任务 $K$ `Video+Audio VAE` 解码流耗时** | **`4.3351s`** | **`45.5527s`** | **拉长 `10.51x`（慢了 `+41.22s`）！**<br>`Run 12` 的 `SageAttention` 与 `ck.sol_attn` 占满 G4 188 个 SM，副流 3D 卷积内核被严重饥饿抢占 |
| **任务 $K+1$ `DiT [Run 12]` 8 步去噪流耗时** | **`41.0038s`** | **`44.5033s`** | **拉长 `1.085x`（慢了 `+3.50s`）！**<br>副流 3D VAE 的高带宽特征图读写争抢 L2 Cache 与 GDDR7 带宽，拖慢主流 DiT |
| **单次跨任务过渡总墙钟耗时 (`Wall Time`)** | **`45.3389s`**<br>(`4.3351s + 41.0038s`) | **`45.5819s`**<br>(`max(45.5527s, 44.5033s)`) | **净收益 `-0.2430s`（`-0.54%` 负优化）！**<br>不仅未能省下 `4.33s`，反而因为双流互抢 SM/带宽导致总耗时更长 |
| **GPU 显存常驻基线 / 瞬时峰值 (`Peak VRAM`)** | 常驻 `19.54 GiB` / 峰值 **`27.413 GiB`** | 双模同驻 `22.704 GiB` / 峰值 **`28.004 GiB`** | **峰值显存增加 `+0.591 GiB`（常驻基线增加 `+3.16 GiB`）**<br>需同时在 VRAM 驻留 3D VAE 激活图与 DiT 去噪激活图 |
| **输出像素一致性 (`P1 vs. P2`)** | 基准真值 | `PSNR = 32.08 dB`<br>(`bit_exact = False`) | 并发流改变了 CUDA 原子归约与块调度时序，引发 BF16/FP4 细微浮点偏移 |
| **CPU `ffmpeg` 音画封装阻塞耗时** | **`0.000s`（后台 3 线程 100% 重叠）** | `0.000s` | 模式 P1 仅靠 CPU 异步线程池即已消除全部 I/O 与编码停顿 |

> **微消融物理定律总结**：当主计算负载（`Run 12` 下的 `MiniMax H3 DiT`）已经是经过高度算子融合（`FusedMod` + `Sage` + `ck.sol_attn`）、SM 与显存带宽利用率接近 `98%` 的重度计算密集型内核时，在同卡另一条 CUDA Stream 上并发塞入同样渴求显存带宽的 3D 因果卷积 `Video VAE Decode`，只会引发破坏性的 **SM 调度争用与 L2 缓存抖动（Cache Thrashing）**。相反，**模式 P1（GPU 单流顺序执行 + 后台 CPU `ThreadPoolExecutor` 异步消费 `ffmpeg` 编码与质检）** 实现了 **GPU 与 CPU 的正交异构并行**，零显存膨胀、100% 确定性输出、跨任务平均交接间隙仅 **`90.9 ms`**！

---

### 3.3 消融三：固定冷启动税摊薄模型（Batch Amortization Curve, $M = 1 \dots 32$）

基于本场实测标定的各阶段精确参数：
- 固定冷启动与卸载常数：$T_{\text{fixed}} = T_{\text{env+dl}} (138.26\text{ s}) + T_{\text{vlm\_load\_unload}} (3.20\text{ s}) + T_{\text{dit\_load+warmup}} (19.19\text{ s}) + T_{\text{egress\_stage1}} (1.57\text{ s}) = \mathbf{162.21\text{ s}}$（折合 **`0.3246 CU`**）；
- 每个唯一分镜的条件编码边际耗时：$T_{\text{cond\_unique}} = \mathbf{2.81\text{ s}}$（首个含冷初始化 `5.13s`），命中 SHA-256 缓存的分镜变体耗时 **`0.00s`**；
- 每个任务的常驻 `Run 12` GPU 端到端边际耗时：$T_{\text{job\_gpu}} = 41.561\text{ s (DiT)} + 4.330\text{ s (VAE)} + 0.091\text{ s (交接)} = \mathbf{45.982\text{ s}}$（折合边际算力 **`0.0920 CU`**）：

| 单次会话攒批任务数 $M$ | 唯一分镜数 $U$ | 会话总墙钟时长 (s) | 会话总消耗算力 (理论 CU) | **摊薄后单条 124 帧视频总耗时 (s/条)** | **摊薄后单条 124 帧视频总成本 (CU/条)** | **每 `100 CU` 可产出视频条数** | 核心渲染有效算力占比 (%) | **相对单跑 1 条 ($M=1$) 的单视频 CU 节省率** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M = 1$** *(交互式单跑)* | `1` | `212.81s` | `0.4259 CU` | `212.81s` | **`0.4259 CU`** | `234.8 条` | `23.98%` | 基准 (`0.00%`) |
| **$M = 2$** | `1` | `258.80s` | `0.5179 CU` | `129.40s` | **`0.2590 CU`** | `386.1 条` | `37.45%` | **省 `39.19%`** |
| **$M = 4$** | `2` | `353.60s` | `0.7076 CU` | `88.40s` | **`0.1769 CU`** | `565.3 条` | `54.16%` | **省 `58.46%`** |
| **$M = 6$ (本次实测)** | **`3`** | **`448.39s` 拟合 / `482.09s` 实测** | **`0.8974 CU` 拟合 / `0.9257 CU` 实扣** | **`74.73s` ~ `80.35s`** | **`0.1496 ~ 0.1543 CU`** | **`648 ~ 669 条`** | **`60.0% ~ 63.8%`** | **省 `62.2% ~ 64.9%`** |
| **$M = 8$** | `4` | `543.19s` (`9.05m`) | `1.0871 CU` | `67.90s` | **`0.1359 CU`** | `735.9 条` | `70.08%` | **省 `68.09%`** |
| **$M = 12$** *(推荐日常批次)* | `6` | `732.78s` (`12.21m`) | `1.4665 CU` | `61.07s` | **`0.1222 CU`** | `818.3 条` | `77.77%` | **省 `71.30%`** |
| **$M = 16$** | `8` | `922.37s` (`15.37m`) | `1.8459 CU` | `57.65s` | **`0.1154 CU`** | `866.8 条` | `82.29%` | **省 `72.91%`** |
| **$M = 24$** | `12` | `1301.56s` (`21.69m`) | `2.6048 CU` | `54.23s` | **`0.1085 CU`** | `921.4 条` | `87.39%` | **省 `74.52%`** |
| **$M = 32$** | `16` | `1680.74s` (`28.01m`) | `3.3636 CU` | `52.52s` | **`0.1051 CU`** | **`951.4 条`** | **`90.19%`** | **省 `75.32%`** |

---

## 4. 6 个批处理渲染任务逐条性能与多 Seed 自动质检选片报告（Per-Job Telemetry & Quality Audit）

在连续执行 6 个任务的过程中，每个任务均严格验证了算子调用计数器（100% 精确执行 **`102 次 SageAttention + 300 次 Direct ck.sol_attn + 400 次 MiniMaxH3FusedModulation`**，零回退），并在后台 CPU 线程中自动完成了多维质量审计与同分镜最佳 Seed 优选：

![6 任务常驻渲染耗时拆解与多 Seed 自动质检选片评分](outputs/per_job_latency_and_seed_quality.png)

| 任务序号 & `job_id` | 分镜 `shot_id` | `seed` | 条件编码耗时 (s) | 模型重打补丁 / 切换准备 | **8步去噪耗时 (`denoise_sec`)** | **Video + Audio VAE 解码耗时** | **GPU 端到端耗时 & 边际 CU** | 后台 CPU 异步 `ffmpeg` (全重叠) | 首/尾帧 SSIM | 首尾帧锚定分 `S_fl2v_pixel` | 时间一致性 / 运动平滑度 | **综合质量分 `Prod_Score` & 分镜内推荐排名** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Job 01`**<br>[`job_01_shot01_seed42.mp4`](outputs/job_01_shot01_seed42.mp4) | `shot_01_cyber_mouse_orbit` | `42` | `5.134s` *(首次)* | `0.000s` / `1.82 ms` | **`41.004s`** | `4.277s + 0.036s`<br>= **`4.313s`** | **`45.317s`**<br>(`0.0907 CU`) | `2.106s`<br>(`6.37 MB`) | `0.9238` / `0.9046` | `91.23` | `99.06` / `74.19` | `83.23` (#3) |
| **`Job 02`**<br>[`job_02_shot01_seed108.mp4`](outputs/job_02_shot01_seed108.mp4) | `shot_01_cyber_mouse_orbit` | `108` | **`0.000s` (HIT!)** | `0.000s` / `3.61 ms` | **`41.386s`** | `4.292s + 0.036s`<br>= **`4.328s`** | **`45.715s`**<br>(`0.0915 CU`) | `1.206s`<br>(`5.69 MB`) | `0.9231` / `0.9058` | `91.27` | `99.35` / `85.20` | `86.05` (#2) |
| **`Job 03`**<br>[`job_03_shot01_seed2026.mp4`](outputs/job_03_shot01_seed2026.mp4) | `shot_01_cyber_mouse_orbit` | `2026` | **`0.000s` (HIT!)** | `0.000s` / `3.08 ms` | **`41.701s`** | `4.297s + 0.036s`<br>= **`4.333s`** | **`46.034s`**<br>(`0.0921 CU`) | `1.198s`<br>(`7.96 MB`) | `0.9228` / `0.9027` | `91.07` | `98.71` / `89.84` | **`86.27` (★ #1 推荐)** |
| **`Job 04`**<br>[`job_04_shot02_seed77.mp4`](outputs/job_04_shot02_seed77.mp4) | `shot_02_neon_circuit_pulse` | `77` | `2.816s` *(首次)* | `0.000s` / `3.20 ms` | **`41.663s`** | `4.297s + 0.036s`<br>= **`4.333s`** | **`45.997s`**<br>(`0.0921 CU`) | `1.400s`<br>(`10.13 MB`) | `0.9210` / `0.9055` | `91.17` | `97.45` / `80.03` | **`86.32` (★ #1 推荐)** |
| **`Job 05`**<br>[`job_05_shot02_seed888.mp4`](outputs/job_05_shot02_seed888.mp4) | `shot_02_neon_circuit_pulse` | `888` | **`0.000s` (HIT!)** | `0.000s` / `3.26 ms` | **`41.978s`** | `4.303s + 0.037s`<br>= **`4.340s`** | **`46.318s`**<br>(`0.0927 CU`) | `1.363s`<br>(`13.35 MB`) | `0.9185` / `0.8990` | `90.68` | `93.67` / `81.72` | `84.56` (#2) |
| **`Job 06`**<br>[`job_06_shot03_seed99.mp4`](outputs/job_06_shot03_seed99.mp4) | `shot_03_hero_reveal_finale` | `99` | `2.800s` *(首次)* | `0.000s` / `3.45 ms` | **`41.636s`** | `4.299s + 0.036s`<br>= **`4.335s`** | **`45.971s`**<br>(`0.0920 CU`) | `3.340s`<br>(`6.28 MB`) | `0.9228` / `0.9062` | `91.29` | `98.97` / `87.92` | **`87.80` (★ #1 推荐)** |
| **6 任务均值 / 合计** | **3 分镜 × 6 任务** | — | **合计 `10.750s`** | **`0.000s` / `3.07 ms`** | **均值 `41.561s`**<br>(合计 `249.37s`) | **均值 `4.330s`**<br>(合计 `25.98s`) | **均值 `45.892s`**<br>(`0.0918 CU/条`) | **合计 `10.613s`**<br>(**`0.00s` 阻塞 GPU**) | `0.9220` / `0.9040` | **`91.12`** | `97.87` / `83.15` | **每分镜最优均分 `86.80`** |

### 4.1 6 任务关键帧视觉审片长图与 3×2 同步对比视频

本地 `0-CU` 后处理渲染器（[`render_visuals_local.py`](code/render_visuals_local.py)）已将全部 6 条视频按分镜自动归档至 [`outputs/by_storyboard/`](outputs/by_storyboard)，并生成了关键帧审片矩阵图与 3×2 同步对比 MP4 视频（[`outputs/comparison_grid_6jobs.mp4`](outputs/comparison_grid_6jobs.mp4)）：

![3 分镜 × 6 任务关键帧视觉审片矩阵](outputs/keyframe_strip_6jobs.png)

- **分镜 1（`shot_01_cyber_mouse_orbit`）3 Seed 抽卡对比**：
  - `seed=42`（`Job 01`）、`seed=108`（`Job 02`）、`seed=2026`（`Job 03`）共享完全相同的首尾帧与文本条件（`Job 02` 与 `Job 03` 条件编码耗时为 `0.000s`）。
  - 自动质检评分器精准识别出 **`Job 03 (seed=2026)`** 兼具丰富的蓝橙流光动态幅度（`dynamic_motion_mae = 3.292` vs `Job 02` 的 `1.648`）与高运动平滑度（`89.84` vs `Job 01` 的 `74.19`），综合质量分 **`86.27`** 位列第一，自动打上 `★ BEST SEED` 标记。
- **分镜 2（`shot_02_neon_circuit_pulse`）2 Seed 抽卡对比**：
  - **`Job 04 (seed=77)`** 清晰度（拉普拉斯方差 `164.36`）与首尾帧锚定度（`91.17`）均优于 `Job 05 (seed=888)`，以 **`86.32`** 分自动当选分镜 2 推荐镜头。
- **分镜 3（`shot_03_hero_reveal_finale`）终章镜头**：
  - **`Job 06 (seed=99)`** 取得全场最高的首尾帧锚定分（`91.29`）与综合质量分（**`87.80`**），完美收尾。

---

## 5. 生产级一键调用命令与产物索引（Production CLI Usage & Artifact Index）

实验 05 的全套模块化代码（全部严格遵守 `< 500` 行/文件规范）已固化为日常自媒体视频批处理生产工具链：

### 5.1 日常一键批处理生产命令

```bash
# 阶段 0: (可选) 本地 0-CU 静态校验清单、预估 CU 成本并查看条件去重率
~/.local/share/uv/tools/google-colab-cli/bin/python \
  experiments/05-production-batch-workflow/code/local_planner.py \
  --manifest experiments/05-production-batch-workflow/code/sample_batch_manifest.json \
  --dry-run

# 全自动一键执行: 本地 0-CU 预投料 -> G4 启动与 Run 12 批处理渲染 -> 1.5s 推回 GDrive 立即停机 -> 本地 0-CU 拉回解包与出图
~/.local/share/uv/tools/google-colab-cli/bin/python \
  experiments/05-production-batch-workflow/code/colab_batch_runner.py \
  --manifest experiments/05-production-batch-workflow/code/sample_batch_manifest.json \
  --session h3-prod-batch-g4 \
  --gpu g4
```

### 5.2 代码与产物文件索引

- **核心工程源码（`code/`，单文件均 `< 500` 行）**：
  - [`code/manifest_schema.py`](code/manifest_schema.py)（`118` 行）：清单 Schema 校验、全局锁定 `ADR-03 Run 12` 常量与 `SHA-256` 条件去重哈希计算。
  - [`code/local_planner.py`](code/local_planner.py)（`295` 行）：本地 `0-CU` 资产对齐、`input_jobs.tar` 打包与 Google Drive v3 REST 预投料。
  - [`code/remote_batch_engine.py`](code/remote_batch_engine.py)（`478` 行）：云端 G4 阶段式批处理引擎（含条件去重缓存、`Qwen3-VL` 卸载、常驻 `Run 12` 连续去噪、后台 CPU 异步 `ffmpeg` 及 P1/P2 微消融）。
  - [`code/colab_batch_runner.py`](code/colab_batch_runner.py)（`395` 行）：无人值守全生命周期编排器（集成 [ADR-04](../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) 两段式卸载与 `try...finally` 零泄漏停机保护）。
  - [`code/render_visuals_local.py`](code/render_visuals_local.py)（`359` 行）：本地 `0-CU` 可视化图表与 3×2 宫格对比视频生成器。
- **实测数据、视频与图表产物（`outputs/`）**：
  - [`outputs/exp05_batch_workflow_metrics.json`](outputs/exp05_batch_workflow_metrics.json)：完整遥测指标、消融对比、成本摊薄曲线与账本对账 JSON。
  - [`outputs/manifest_results.json`](outputs/manifest_results.json)：每个任务完成瞬间增量持久化的断点状态记录。
  - [`outputs/colab_session_ledger_excerpt.json`](outputs/colab_session_ledger_excerpt.json)：G4 会话分配/释放时间轴与 `1.0229 CU` 账本扣费核对摘录。
  - [`outputs/batch_amortization_and_lifecycle_cu.png`](outputs/batch_amortization_and_lifecycle_cu.png)：冷启动摊薄曲线、SHA-256 条件去重与 P1/P2 微消融三联图。
  - [`outputs/per_job_latency_and_seed_quality.png`](outputs/per_job_latency_and_seed_quality.png)：6 任务耗时拆解与多 Seed 自动质检选片图。
  - [`outputs/keyframe_strip_6jobs.png`](outputs/keyframe_strip_6jobs.png) & [`outputs/comparison_grid_6jobs.mp4`](outputs/comparison_grid_6jobs.mp4)：6 任务关键帧矩阵与 3×2 同步对比视频。
  - 按分镜自动归档的生产成片目录：[`outputs/by_storyboard/shot_01_cyber_mouse_orbit/`](outputs/by_storyboard/shot_01_cyber_mouse_orbit)、[`outputs/by_storyboard/shot_02_neon_circuit_pulse/`](outputs/by_storyboard/shot_02_neon_circuit_pulse)、[`outputs/by_storyboard/shot_03_hero_reveal_finale/`](outputs/by_storyboard/shot_03_hero_reveal_finale)。
