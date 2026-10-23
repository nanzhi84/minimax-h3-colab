<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# ADR-03：在 Blackwell G4 上正式锁定 `Run 12 [S8-FullStack-Pareto]`（零拷贝直通 `ck.sol_attn` + 前 2/8 步 `SageAttention` 预热 + `MiniMaxH3FusedModulation`，`46.72s` / `0.0935 CU`）为默认标准推理配置，并确立零浪费 GPU 生命周期工程规范

- **状态（Status）**：已采纳（Accepted，正式锁定 **`Run 12 [S8-FullStack-Pareto]`** 为后续实验与生产的唯一默认标准加速配置，升级替代 [ADR-02](./0002-blackwell-gpu-and-sageattention-acceleration.md) 中的 `Only-Sage` 策略）
- **日期（Date）**：2026-10-23
- **关联实验**：
  - [`experiments/03-g4-sol-attn-pareto-ablation`](../../experiments/03-g4-sol-attn-pareto-ablation/README.md)（完整实验报告：[`report.md`](../../experiments/03-g4-sol-attn-pareto-ablation/report.md)；原始实测指标：[`g4_sol_attn_ablation_metrics.json`](../../experiments/03-g4-sol-attn-pareto-ablation/outputs/g4_sol_attn_ablation_metrics.json)；会话账本摘录：[`colab_session_ledger_excerpt.json`](../../experiments/03-g4-sol-attn-pareto-ablation/outputs/colab_session_ledger_excerpt.json)）
  - [`experiments/04-gdrive-vs-http-egress-transfer`](../../experiments/04-gdrive-vs-http-egress-transfer/README.md)（尾端回传链路消融实验报告：[`report.md`](../../experiments/04-gdrive-vs-http-egress-transfer/report.md)；生产级两段式 Google Drive 回传工具：[`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py)）
- **前置依赖**：
  - [ADR-01：锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）与 HF 直连加载链路](./0001-model-weights-loading-pipeline.md)
  - [ADR-02：锁定 Google Colab Blackwell 96GB（`--gpu G4`）为标准推理机型](./0002-blackwell-gpu-and-sageattention-acceleration.md)

---

## 1. 背景与问题（Context）

在 [ADR-02](./0002-blackwell-gpu-and-sageattention-acceleration.md) 锁定 Google Colab Blackwell 96GB（`--gpu G4`，`NVIDIA RTX PRO 6000 Blackwell Server Edition`，SM 12.0，实验 02 账本反推实测等效费率 `7.2046 CU/hr`）与 `Only-Sage`（`sageattention==1.0.6`，实验 02 原测去噪 `68.01s` / 端到端 `72.34s` / `0.1448 CU`；实验 03 同会话复测 `Run 02` 去噪 `68.69s` / 端到端 `73.03s` / `0.1461 CU`）后，仍存在三项亟待突破的核心工程与算法瓶颈：

1. **全序列三维多模态注意力的二次方算力瓶颈（$N = 42,058$）**：
   - 在 `1344×768 × 124 帧` 声画同源生成负载下，50 层 DiT 每步需对 $N = 42,058$ 个联合 token（$4,562$ 音频与首尾参考帧条件 + $200$ 文本条件 + $37,296$ 视频潜变量 tokens，56 个注意力头，单次理论计算量 $50.717\text{ TFLOPs}$）执行自注意力，8 步共 `400` 次 `DiTBlock` 注意力调用占去噪耗时的 `75%` 以上。`EasyCache` 虽能跳步但导致首尾帧像素 SSIM 从 `0.925 / 0.903` 崩塌至 `0.557 / 0.590`。能否在不跳步、不损失首尾帧像素锚定（`S_fl2v > 94`）的前提下，通过 **Sol-Attn（Sparsifying Online Attention，`arXiv:2607.24027`）** 及原生全 INT8 算子进一步大幅压缩成本与延迟？
2. **上游 `comfy-kitchen` 与社区 `r-vage/ComfyUI-sol-attn` 加速方案的架构缺陷与取舍**：
   - 上游 ComfyUI `BlockSparseAttention` 节点默认对 H3 注入 `h3_sparse_attention`（走 4,096-token 分块的 `ck.sol_attn_chunked`），而社区 `r-vage/ComfyUI-sol-attn` 提供了 Triton 版 `sol_attn`、动态 `_TauSchedule` 调度器及 `MiniMaxH3FusedModulation` 调制融合算子。如何通过源码审计与正交消融厘清各方案的真实收益与潜在缺陷？
3. **Colab 会话全生命周期的非核心 CU 浪费问题**：
   - 在实验 02 中，A100 会话（`7.3077 CU`）仅有 `46.58%`（`3.4037 CU`）用于核心消融推理，Blackwell G4 会话（`2.2015 CU`）更仅有 `24.69%`（`0.5436 CU`）用于核心推理，其余 CU 消耗在独立进程环境探针（`287.98s`）、权重拉取、2×8 步冗余预热（`154.47s`）及远端同步 CPU 视频编码与绘图回传（`245.87s`）上。

---

## 2. 实验证据（Evidence）

在实验 03（[`experiments/03-g4-sol-attn-pareto-ablation`](../../experiments/03-g4-sol-attn-pareto-ablation/report.md)）中，我们在单次 G4 会话（`h3-exp03-g4`）内完成了 9 算子微基准、两项源码缺陷实证检验以及 12 组端到端消融实验：

### 2.1 Blackwell G4 十二组消融与帕累托前沿核心数据表

| 档位定位 | Run 编号与简码 | 核心加速技术栈 | 去噪耗时 $T_{\text{den}}$ (vs SDPA / Sage) | 端到端 $T_{\text{e2e}}$ (vs A100 基线) | 单条视频 CU (`条 / 100 CU`) | 相对 G4 SDPA / A100 基线省 CU | PSNR (dB) / SSIM | 首尾帧锚定 $S_{\text{fl2v}}$ | VBench $S_{\text{vbench}}$ | 综合质量 $Q_{\text{total}}$ ($Q_{\text{lin}}$) | 帕累托最优 (`Q_total` / `Q_lin`) |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **极速探索档** 🏆 | **Run 07** `[S3-CKDirect-Full]` | 零拷贝直通 `ck.sol_attn` (0–100%, $\tau=1.3$, `tok=0`) | **`32.76s`** (`2.38x` / `2.10x`) | **`37.11s`** (`4.02x`) | **`0.0743 CU`** (`1346.6 条`) | **`-54.98%` / `-73.54%`** | `14.94` / `0.6281` | **`94.57`** | `74.03` | **`66.45`** (`65.91`) | ⭐ **是 / 是** |
| *(对照：分块默认)* | Run 05 `[S1-CKChunk-Full]` | 上游默认 4K 分块 `ck.sol_attn_chunked` (0–100%, $\tau=1.3$) | `33.28s` (`2.35x` / `2.06x`) | `37.64s` (`3.96x`) | `0.0753 CU` (`1327.6 条`) | `-54.34%` / `-73.16%` | `14.90` / `0.6234` | `94.66` | `74.24` | `66.25` (`65.63`) | 否 *(被 R07 支配)* |
| **主力生产档** 🏆 | **Run 12** `[S8-FullStack-Pareto]` | **零拷贝直通 `ck.sol_attn` (W20, $\tau=1.3$, `tok=256`) + Triton `H3FusedModulation`** | **`42.37s`** (`1.84x` / `1.62x`) | **`46.72s`** (`3.19x`) | **`0.0935 CU`** (`1069.5 条`) | **`-43.31%` / `-66.68%`** | **`16.98` / `0.6918`** | **`94.24`** | **`73.99`** | **`72.25`** (`70.90`) | ⭐ **是 / 是** |
| **纯原生稀疏档** 🏆 | **Run 08** `[S4-CKDirect-W20-Tok256]` | 零拷贝直通 `ck.sol_attn` (W20, $\tau=1.3$, `tok=256`) | **`43.01s`** (`1.81x` / `1.60x`) | **`47.36s`** (`3.15x`) | **`0.0948 CU`** (`1055.0 条`) | **`-42.53%` / `-66.22%`** | `16.85` / `0.6876` | `94.30` | `73.83` | **`72.27`** (`70.84`) | ⭐ **是 / 否** *(Q_lin 被 R12 支配)* |
| *(对照：分块 W20)* | Run 06 `[S2-CKChunk-W20-Tok256]` | 上游节点默认分块 `ck.sol_attn_chunked` (W20, $\tau=1.3$, `tok=256`) | `43.64s` (`1.79x` / `1.57x`) | `48.00s` (`3.11x`) | `0.0961 CU` (`1041.0 条`) | `-41.76%` / `-65.77%` | `16.88` / `0.6881` | `94.25` | `73.61` | `72.17` (`70.79`) | 否 *(被 R12/R08 支配)* |
| **高保真稀疏档** 🏆 | **Run 09** `[S5-CKDirect-HiFi-t10]` | **零拷贝直通 `ck.sol_attn` (W20, $\tau=1.0$, `tok=256`, `dense_blocks="0,1,48,49"`)** | **`46.36s`** (`1.68x` / `1.48x`) | **`50.72s`** (`2.94x`) | **`0.1015 CU`** (`985.2 条`) | **`-38.47%` / `-63.83%`** | **`17.41` / `0.7019`** | **`94.41`** | **`74.44`** | **`73.30`** (`71.93`) | ⭐ **是 / 是** |
| *(对照：社区 Triton)* | Run 10 `[S6-Triton-Int8QK]` | 社区 `r-vage` Triton Sol (0–100%, $\tau=1.3$, `INT8-QK`) | `49.82s` (`1.57x` / `1.38x`) | `54.17s` (`2.75x`) | `0.1084 CU` (`922.4 条`) | `-34.28%` / `-61.37%` | `14.78` / `0.6163` | `94.81` | `73.26` | `67.03` (`65.77`) | 否 *(S_fl2v 前沿)* |
| *(对照：社区调度)* | Run 11 `[S7-Triton-Sched]` | 社区 `_TauSchedule` 修复版 (W20, $\tau: 1.5\rightarrow 0.8$, `INT8-QK+PV`) | `55.04s` (`1.42x` / `1.25x`) | `59.39s` (`2.51x`) | `0.1189 CU` (`841.4 条`) | `-27.95%` / `-57.65%` | `16.86` / `0.6879` | `94.74` | `73.86` | `73.11` (`72.02`) | 否 *(被 R03/R09 支配)* |
| **高保真稠密档** 🏆 | **Run 03** `[D2-CKInt8]` | **`comfy-kitchen` 原生 CUDA 全 INT8 稠密 (`attention_comfy_kitchen_int8`)** | **`54.06s`** (`1.44x` / **`1.27x`**) | **`58.41s`** (`2.55x`) | **`0.1169 CU`** (`855.4 条`) | **`-29.13%` / `-58.34%`** | **`22.75` / `0.8257`** | **`94.11`** | `72.72` | **`80.98`** (`79.15`) | ⭐ **是 / 是 (取代 Sage)** |
| *(旧 ADR-02 基准)* | Run 02 / 04 `[D1-Sage]` / `[+Mod]` | `SageAttention 1.0.6` (无 / 有 `H3FusedModulation`) | `68.69s` / `67.96s` | `73.03s` / `72.31s` | `0.1461` / `0.1447 CU` (`684.2` / `691.0 条`) | `-11.40% (-12.27%)` / `-47.92% (-48.43%)` | `20.19` / `0.7711` | `94.25` | `73.30` | `78.00` (`75.98`) | 否 *(E2E–Q 被 R03 支配)* |
| **绝对基准锚点** 🏆 | **Run 01** `[D0-SDPA]` | 原生 PyTorch 2.11 cuDNN BF16 SDPA | `78.06s` (`1.00x`) | `82.42s` (`1.81x`) | `0.1649 CU` (`606.2 条`) | `0.00%` / `-41.22%` | `100.0` / `1.0000` | `94.28` | `73.50` | **`91.95`** (`91.95`) | ⭐ **是 / 是** |

### 2.2 关键实测依据

1. **为什么在全稠密档位用 `comfy-kitchen` 全 INT8（`Run 03`）全面取代 ADR-02 的 `SageAttention 1.0.6`（`Run 02`）？**
   - `comfy-kitchen` v0.2.40 的 `attention_comfy_kitchen_int8` 对 50 层 `DiTBlock` 的 `AttentionTensorContainer` 调用 `prequantize_int8_attention → sage_sdpa_quantize`（执行 $Q/K$ Hadamard 正交旋转与代表性 key 向量中心化，并提前释放浮点容器张量引用），随后由 C++/CUDA `ck.int8_attention_from_prequantized`（400 次 `DiTBlock` 调用 + 2 次精炼器 `ck.int8_attention` 调用）将 $Q \cdot K^T$ 与 $P \cdot V$ 均置于 SM 12.0 INT8 Tensor Core 执行。实测单算子吞吐达 **`618.76 TFLOPS`（`81.97 ms` vs Sage `118.35 ms`）**；相比同会话复测的 `Run 02 [D1-Sage]`（去噪 `68.69s` / 端到端 `73.03s`），**去噪净省 `14.62s`（提速 `1.27x`）、端到端净省 `14.61s`（提速 `1.25x`，节省 `20.01%` CU）**（相对实验 02 G4 `Run 2` 原测 `72.34s` 则端到端净省 `13.93s`），同时参考保真度与音频保真度显著更高（**`PSNR = 22.75 dB` vs `20.19 dB`，`LPIPS = 0.0672` vs `0.1075`，$S_{\text{ref}} = 72.08$ vs `63.90`，$S_{\text{audio}} = 93.63$ vs `91.64`，$Q_{\text{total}} = 80.98$ vs `78.00`**，仅 $S_{\text{vbench}}$ `72.72 vs 73.30` 与 $S_{\text{fl2v}}$ `94.11 vs 94.25` 略低）。
2. **为什么在稀疏加速档位必须绕过上游 `h3_sparse_attention` 4K 分块生产者，改走零拷贝直通 `ck.sol_attn`（Discovery 1）？**
   - 源码与微基准实证表明，上游 `h3_sparse_attention`（`ck.sol_attn_chunked`）在首个稀疏步因 `kmean=None` 会将 11 个 4K 分片生成器迭代**两遍**，单块触发 **22 次** `qkv_proj`（单步 `1,100` 次 INT8 GEMM，单块耗时 `67.633 ms`），后续 warm 步触发 11 次 `qkv_proj`（`52.365 ms/blk`）。在 96 GB 显存的 G4 上，移除 `patches_replace["dit"]` 改走零拷贝直通 `ck.sol_attn` 后，每块仅执行 **1 次**全矩阵 `qkv_proj`（单步 `50` 次 GEMM，恒定 `53.303 ms/blk`）：虽然单次 warm 分块略快于直通，但直通路径通过消除首步双遍投影启动惩罚，在 8 步端到端生成中稳定净省 **`0.52s ~ 0.63s` 去噪耗时（`0.53s ~ 0.64s` 端到端耗时）** 且综合质量 $Q_{\text{total}}$ 更优。
3. **为什么混合稀疏生产档（`Run 12` / `Run 08` / `Run 09`）采用 `start_percent = 0.20`（8 步中前 2/8 步 = 25% 步数稠密预热）+ `extra_tokens = 256` 组合配置？**
   - 对比 0% 预热组（`Run 05, 07`，`start_percent = 0.0, extra_tokens = 0`），配置阈值 `start_percent = 0.20`（在 8 步采样中对应前 2/8 步 = 25% 步数 `SageAttention` 稠密预热，后 6/8 步 = 75% 步数稀疏注意力）与 `extra_tokens = 256` 后，第 53 帧与第 76 帧的切镜偏差被显著消除，`Run 08 / 09 / 12` 子集的最低帧 PSNR 从 `6.37 dB` 跃升至 **`11.39 ~ 11.58 dB`**（全部 5 组 `start_percent = 0.20` 预热配置 `Run 06, 08, 09, 11, 12` 为 `11.22 ~ 11.70 dB`），平均 PSNR 提升至 **`16.85 ~ 17.41 dB`**，$Q_{\text{total}}$ 从 `66.45` 跃升至 **`72.25 ~ 73.30`**，而端到端耗时仅需 **`46.72s ~ 50.72s`（`0.0935 ~ 0.1015 CU`）**。
4. **为什么采纳社区 `MiniMaxH3FusedModulation`，而仅将社区 Triton `sol_attn` 作为备选？**
   - 社区 `MiniMaxH3FusedModulation` 将每层 4 次分段 AdaLN 调制与门控残差替换为 4 次融合 Triton 算子调用，实测与 Eager **100% 位级等价（`max_diff = 0.0`）**，单视频去噪净省 `0.64s ~ 0.73s`（端到端净省 `0.64s ~ 0.72s`）；而社区 Triton `sol_attn` 在 SM 12.0 上耗时（`47.77 ~ 62.64 ms/call`）慢于 `comfy-kitchen` 原生 CUDA `ck.sol_attn`（`23.95 ~ 27.31 ms/call`），且其 `_TauSchedule` 存在未逆变换 $\alpha=12.0$ 流匹配时间偏移的缺陷（Discovery 2，`if progress < dense_percent` 导致未修复前 7/8 步退化为稠密）。
5. **为什么不采用外部社区 [`soren-labs/minimax-h3-colab`](https://github.com/soren-labs/minimax-h3-colab) 的 20 步 `Spectrum` / `TE-Speed` 路线？**
   - 审计该仓库源码与日志（详见实验 03 [`report.md` 第 8 节](../../experiments/03-g4-sol-attn-pareto-ablation/report.md#8-外部社区基准对比与-soren-labsminimax-h3-colab-同卡blackwell-g4方案的深度对标与根因拆解)）发现：其一，该仓库基于 `PyTorch 2.11.0+cu128` + `comfy-kitchen 0.2.26`，导致 `comfy-kitchen` 的 `cuda` 与 `triton` 后端全部被禁用（`disabled: True`），所有 `int8_convrot` 投影退化为慢速 `eager` 反量化（单步耗时达 `14.66s ~ 17.22s/step`）；其二，该仓库未加载 [ADR-01](./0001-model-weights-loading-pipeline.md) 的 8 步蒸馏 Turbo LoRA，被迫跑 20 步（开启 `Spectrum` 14 实 + 6 测后在 `1344×768 × 124 帧` 下仍需 `226.09s ~ 234.20s`，开启 `TE-Speed` 8 实 + 12 缓在 192 帧下需 `288.09s`，且二者互斥不可叠加）。相比之下，本 ADR 锁定的 **`Run 12 [S8-FullStack-Pareto]`（`46.72s` / `0.0935 CU`）在不跳步、不跳层的前提下比其同分辨率 `Spectrum` 方案快 `5.01 倍`（省 `80.1%` CU）**。

---

## 3. 决策内容（Decision）

自本 ADR 起，在 Google Colab Blackwell 96GB（`--gpu G4`）上运行 `MiniMax H3-Base-FL2VA` **正式锁定 `Run 12 [S8-FullStack-Pareto]` 为默认标准推理配置**，并确立零浪费 GPU 生命周期工程规范：

### 3.1 锁定默认标准配置：`Run 12 [S8-FullStack-Pareto]`（`42.37s` 去噪 / `46.72s` 端到端 / `0.0935 CU`）

后续所有实验与生产任务默认统一采用 **`Run 12 [S8-FullStack-Pareto]`** 全栈混合加速配置：
- **运行时环境要求**：`PyTorch >= 2.11.0+cu130`、`comfy-kitchen == 0.2.40`（确保 `cuda` 与 `triton` 后端均为 `enabled`）、`sageattention == 1.0.6`、`triton == 3.6.0`，以及社区扩展 `r-vage/ComfyUI-sol-attn`（Commit `69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3`）。
- **权重基座**：严格遵循 [ADR-01](./0001-model-weights-loading-pipeline.md) 黄金五件套（含第 5 件 `minimax_h3_fl2va_2505_ntk32_8steps_lora_fp16.safetensors` 8 步蒸馏 LoRA，`steps=8, sampler="res_multistep", scheduler="simple"`）。
- **三层核心加速栈参数（对应 8 步共 `102 Sage + 300 Direct Sol-Attn` 注意力调用 + `400` 次融合调制 Block）**：
  1. **前 2/8 步稠密预热后端（`start_percent = 0.20`，对应 Step 0~1 共 25% 步数、`102` 次调用）**：将 `comfy.ldm.modules.attention.optimized_attention` 绑定为 `attention_sage`（`SageAttention 1.0.6`），在噪声最高的初始两步锚定全局多镜头切镜结构与首尾帧色彩一致性。
  2. **后 6/8 步零拷贝直通稀疏注意力（Step 2~7 共 75% 步数、`300` 次调用）**：应用 `BlockSparseAttention`（`selection="sol-attn", tau=1.3, start_percent=0.20, end_percent=1.0, extra_tokens=256, dense_blocks="", sink_conditioning="exact_kv_and_rows"`），随后**立即弹出 `patches_replace["dit"]` 中的 `h3_sparse_attention` 4K 分块钩子**，使 50 层 `DiTBlock` 通过 `AttentionTensorContainer` 零拷贝直通 `ck.sol_attn`（消除首步 22 次 `qkv_proj` 双遍投影惩罚）。
  3. **全步位级无损调制融合（8 步 × 50 层 = `400` 个 Block）**：挂载社区 `MiniMaxH3FusedModulation().patch(model=work_model, enabled=True)[0]`，将每层 4 次分段 AdaLN 调制与门控残差融合为 4 次 Triton 算子调用（`Bit-Exact = True`）。
- **标准 Python 挂载代码模板**：
  ```python
  import comfy.ldm.minimax.model as minimax_model_mod
  import comfy.ldm.modules.attention as comfy_attn_mod
  from comfy_extras.nodes_sparse_attention import BlockSparseAttention
  from comfyui_sol_attn.minimax import MiniMaxH3FusedModulation
  from sageattention import sageattn

  # 1. 绑定前 2/8 步 (start_percent=0.20) 稠密预热后端为 SageAttention
  comfy_attn_mod.sageattn = sageattn
  comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True
  comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_sage
  minimax_model_mod.optimized_attention = comfy_attn_mod.attention_sage

  # 2. 注入原生 Sol-Attn 并移除 4K 分块钩子，启用零拷贝直通 ck.sol_attn
  work_model = BlockSparseAttention.execute(
      model=base_model.clone(),
      selection={"selection": "sol-attn", "tau": 1.3},
      start_percent=0.20,
      end_percent=1.0,
      extra_tokens=256,
      dense_blocks="",
      sink_conditioning="exact_kv_and_rows",
  )[0]
  work_model.model_options.get("transformer_options", {}).get("patches_replace", {}).pop("dit", None)

  # 3. 叠加位级无损 Triton 融合调制 MiniMaxH3FusedModulation
  work_model = MiniMaxH3FusedModulation().patch(model=work_model, enabled=True)[0]
  ```

### 3.2 特定场景备选帕累托档位（仅在偏离默认需求时按需显式指定）

- **高保真全稠密备选档 —— `Run 03 [D2-CKInt8]`（`58.41s` / `0.1169 CU`，`PSNR = 22.75 dB`，$Q_{\text{total}} = 80.98$）**：当评测或定稿要求 100% 稠密注意力（禁用块稀疏剪枝）时，绑定 `comfy_attn_mod.attention_comfy_kitchen_int8`（全面替代原 ADR-02 的 `Only-Sage`）。
- **高保真稀疏备选档 —— `Run 09 [S5-CKDirect-HiFi-t10]`（`50.72s` / `0.1015 CU`，$Q_{\text{total}} = 73.30$）**：在直通 `ck.sol_attn` 下收紧 $\tau=1.0$ 并保护首尾层 `dense_blocks="0,1,48,49"`。
- **极限低成本探索备选档 —— `Run 07 [S3-CKDirect-Full]`（`37.11s` / `0.0743 CU`，$Q_{\text{total}} = 66.45$）**：0–100% 全步直通 `ck.sol_attn`（`start_percent=0.0, tau=1.3, extra_tokens=0`），用于超大规模提示词/种子初筛。

### 3.3 零浪费 GPU 生命周期工程规范（Zero-Waste GPU Lifecycle Standard）

所有云端 GPU 实验与批处理任务必须遵循以下五条生命周期准则，确保核心推理占会话总 CU 比例 $\ge 65\%$（本次实验实测占会话全生命周期 **`69.64%`**，占打包前脚本计时范围 **`72.30%`**）：
1. **离线静态验证与 Codex 门禁前置（`0.00 CU`）**：严禁开启昂贵 GPU 实例进行交互式代码试错；所有脚本必须在本地完成语法、张量维度与 Codex（`gpt-6-astra xhigh`）审查后再上云单次跑通，且单个手写代码文件严格控制在 `< 500` 行。
2. **多线程并发预取与 I/O 重叠**：使用 `ThreadPoolExecutor(max_workers=8)` 并发拉取多模态模型权重与后台预加载评测模型权重。
3. **单次 1 步轻量预热**：完成算子级微基准预热后，全模型流水线预热严格限制为 `steps=1`（耗时 `13.83s`，较实验 02 的 `154.47s` 节省 `140.64s`），严禁执行冗余的多轮完整 8 步预热。
4. **CPU 异步编码与后处理卸载**：视频帧与音频波形一旦拷回主机内存，立即交由后台 `ThreadPoolExecutor` 异步执行无 `-shortest` 截尾的 `ffmpeg` 封装，使 12 组循环间隙压缩至 `4.56s`。
5. **两段式 Google Drive REST 云端内网极速卸载 + 毫秒级 `unassign` + 本地 0 CU 并发拉取与渲染（详见 [ADR-04](./0004-two-stage-gdrive-rest-egress-relay.md) 强制标准）**：
   - 后续所有实验与生产任务的尾端回传**一律强制采用 [ADR-04](./0004-two-stage-gdrive-rest-egress-relay.md) 锁定的 [`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py) 两段式 Google Drive REST 中转形式**（不再使用 `colab download` 直接拉取大归档包）：
     1. **阶段 1（GPU 唯一计费期，Google 机房内网骨干网 `54 ~ 80 MB/s`）**：远端将打包好的单一未压缩 `.tar` 归档包直接通过 Google Drive v3 REST API（复用 `~/.config/colab-cli/token.json` 的 `drive.file` 权限，零人工交互、免 `colab drivemount` FUSE 挂载开销）推入 Google Drive（单视频 `10.63 MB` 仅需 **`1.04s` / `0.0021 CU`**；12 视频全量包 `122.42 MB` 仅需 **`2.16s ~ 2.27s` / `0.0045 CU`**），HTTP `200 OK` 返回并校验 `md5Checksum` 的瞬间**立即执行 `colab stop` 销毁 GPU 实例**（较 `colab download` 节省 **`72.3% ~ 91.5%`** 尾端 GPU CU，弱网下节省达 **`97% ~ 99%`**）；
     2. **阶段 2（GPU 已停机 = `0.00 CU/hr`）**：在本地 Mac 使用 8 线程 HTTP `Range` 并发流从 Google Drive 拉回原始二进制 `.tar` 包（`122.42 MB` 仅需 `10.94s`，端到端总墙钟耗时 `13.21s`，甚至比 `colab download` 直拉的 `17.69s ~ 26.60s` 更快），校验 `SHA-256` 后自动清理云端临时文件，并在本地免费完成所有图表渲染与视频合成（`0.00 CU`）。

---

## 4. 影响与后果（Consequences）

### 正向收益（Pros）
- **单视频成本再降 `19.2% ~ 48.7%`（相对 A100 基线降幅达 `58.3% ~ 73.5%`）**：相比 ADR-02 的 `Only-Sage`（实验 02 `72.34s` / `0.1448 CU`；实验 03 `Run 02` `73.03s` / `0.1461 CU`），新的主力生产档 `Run 12` 将耗时压至 **`46.72s`（`0.0935 CU`）**，每 `100 CU` 视频产量跃升至 **`1,069.5 条`（较同会话 `Run 02` 提升 `+56.3%`）**；高保真稠密档 `Run 03` 将耗时压至 **`58.41s`（`0.1169 CU`）**，产量提升至 **`855.4 条`（提升 `+25.0%`）** 且参考保真度更优。
- **彻底解决首尾帧崩塌与冷启动浪费**：在不跳步的前提下，所采纳的四个加速档位（`Run 03, 09, 12, 07`）实现 **`1.41x ~ 2.22x`** 端到端加速（稀疏三档为 `1.63x ~ 2.22x`），所有档位首尾帧锚定分均稳定在 `S_fl2v > 94.1`；同时单次 12 组消融实验全会话理论折算仅耗费 **`1.9175 CU`**（已结算账本净扣 **`1.8494 CU`**，仅占实验 02 两次会话总消耗 `9.5092 CU` 的 **`20.16%` / `19.45%`**）。

### 约束与注意事项（Cons & Mitigations）
- **稀疏与稠密路径的确定性差异**：在本次实验中，稠密 `SageAttention` 叠加 `MiniMaxH3FusedModulation`（`Run 02` vs `Run 04`）经验证输出 100% 位级相同（MP4 SHA-256 完全一致），而稀疏 `ck.sol_attn` 叠加 `MiniMaxH3FusedModulation`（`Run 08` vs `Run 12`）存在 `0.0215` 分的 $Q_{\text{total}}$ 差异（`PSNR 16.85 vs 16.98 dB`）；若业务要求不启用块稀疏剪枝的稠密生成，应选用 `Run 03 [D2-CKInt8]` 或 `Run 01 [D0-SDPA]`。
- **社区 `_TauSchedule` 补丁要求**：若在实验中使用社区 `MiniMaxH3ScheduledSolAttentionPatch` 节点，必须注入流匹配时间偏移逆变换 $\hat{t} = \frac{\sigma}{12 - 11\sigma}$，否则前 87.5% 的步骤会因 `progress < dense_percent` 被回退为稠密计算。
