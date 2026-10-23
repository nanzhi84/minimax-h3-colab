<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# 实验 03：Blackwell G4 (`RTX PRO 6000`) 上 Sol-Attn 稀疏注意力与社区算子融合的成本-速度-质量帕累托前沿消融

- **实验编号**: `03-g4-sol-attn-pareto-ablation`
- **完整实验报告**: [`report.md`](./report.md)（实测指标 JSON：[`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json)，会话账本摘录：[`outputs/colab_session_ledger_excerpt.json`](./outputs/colab_session_ledger_excerpt.json)）
- **关联架构决策**: [ADR-01: 模型权重加载管线](../../docs/adr/0001-model-weights-loading-pipeline.md), [ADR-02: Blackwell GPU 选型与 SageAttention 无损加速基线](../../docs/adr/0002-blackwell-gpu-and-sageattention-acceleration.md), [ADR-03: Blackwell G4 Sol-Attn 帕累托前沿与零浪费生命周期规范](../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md)
- **前置实验**: [实验 02: A100/Blackwell 推理加速三维正交消融](../02-a100-acceleration-ablation/report.md)
- **目标硬件**: Google Colab **Blackwell G4 (`--gpu G4`, `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96GB GDDR7, SM 12.0)**，实验 02 账本反推实测等效费率 **`7.2046 CU/hr`** (`0.00200128 CU/s`)
- **测试模型**: MiniMax `H3-Base-FL2VA` (`1344×768`, 124 帧, 24 FPS, 8-Step Turbo LoRA, 首尾帧受控音视频同生, 联合序列长度 `N = 42,058`, `50` 层 `DiTBlock`, `56` 头 × `128` 维)
- **预算上限与实测消耗**: 预算上限 **$\le 30\text{ CU}$**；单次会话客户端分配至释放总时长 `958.12s`（理论折算 **`1.9175 CU`**，Colab 账本最终结算净扣减 **`1.8494 CU`**），其中 12 次核心消融推理消耗 **`1.3354 CU`**（占打包前脚本计时总时长 **`72.30%`**、占会话全生命周期 **`69.64%`**，较实验 02 G4 的 `24.69%` 提升 `2.82 倍`）

---

## 1. 实验背景与核心问题

### 1.1 实验 02 的结论与遗留挑战
在[实验 02](../02-a100-acceleration-ablation/report.md) 与 [ADR-02](../../docs/adr/0002-blackwell-gpu-and-sageattention-acceleration.md) 中，我们确立了 **`Blackwell G4 (--gpu G4)` + `SageAttention v1 (Triton INT8-QK + BF16-PV)`** 作为基线配置：
- 8 步 DiT 去噪耗时从 `77.37s` 降至 `68.01s`（**1.14x**，相对 A100 基线 `139.95s` 加速 **2.06x**），端到端耗时从 `81.69s` 降至 `72.34s`（单条成本从 `0.1635 CU` 降至 `0.1448 CU`），四维综合质量得分 `Q_total = 75.98`（补充线性 STFT 口径，其中首尾帧锚定得分 `S_fl2v = 94.25`、VBench 代理得分 `S_vbench = 73.30` 与基线持平）。
- 同时排除了在 8 步蒸馏流匹配与双流 `NestedTensor` 下导致首尾帧结构崩塌的 `EasyCache`（`S_fl2v` 暴跌至 `75.1`）以及存在冷编译税且稳态负收益的 `TorchCompile`。

然而，要进一步压降 H3-FL2VA 在 `N = 42,058` 超长序列下的注意力开销与单条视频 CU 成本，实验 02 留下了两个核心问题：
1. **算法层（Sol-Attn 稀疏注意力与社区算子融合）**：
   - 在 `N = 42,058`（`658` 个 64-token 块）下，稠密自注意力的 $O(N^2)$ 复杂度依然占据去噪计算的主导。最新训练免调优稀疏注意力方案 **Sol-Attn (Sparsifying Online Attention, `arXiv:2607.24027`)** 通过在线阈值选择对高分键块走精确注意力分支（Exact Branch）、对剩余低分键块走块均值池化尾部分支（Approximate Pooled Tail，保留 Softmax 分母质量），在长序列视频生成中展现出显著潜力。
   - 目前社区与上游存在两套截然不同的 H3 `Sol-Attn` 实现及配套融合算子：
     - **上游原生 `comfy-kitchen` (`v0.2.40`, `sm_120` 编译) C++/CUDA 实现**：`BlockSparseAttention`（包含 4K 分块流式 QKV 生产者 `ck.sol_attn_chunked` 与直通零拷贝 `ck.sol_attn`、全 INT8 MMA 精确分支、离群单 Token 补充路由 `extra_tokens`、以及纯 INT8 稠密注意力 `ck.int8_attention` / `ck.int8_attention_from_prequantized`）。
     - **社区 `r-vage/ComfyUI-sol-attn` (`v0.6.7`) Triton 实现**：包含零拷贝跨步 `MiniMaxH3MemoryEfficientSolAttentionPatch` (`_forward_int8_ptr` + `INLINE_Q`)、动态 $\tau$ 退火调度 `MiniMaxH3ScheduledSolAttentionPatch`、以及与注意力后端正交的 **逐段 AdaLN/Gate 比特级精确融合算子 `MiniMaxH3FusedModulation`** 和 **FFN 峰值显存分块算子 `MiniMaxH3ChunkFeedForward`**。
2. **工程层（GPU 核心消融推理 CU 聚焦度问题）**：
   - 在实验 02 中，A100 会话（`64.77 分钟`, `7.3077 CU`）真正用于 12 次核心消融推理的消耗仅 `3.4037 CU`（**46.58%**），其余 `3.9040 CU`（**53.42%**）耗费在拉取 42GB 权重、在线修复编译异常、冗余预热与后处理上。
   - Blackwell G4 会话（`18.33 分钟`, `2.2015 CU`）真正用于 4 次核心消融推理的消耗仅 `0.5436 CU`（**24.69%**），其余 `1.6579 CU`（**75.31%**）耗费在独立环境探针进程、微基准脚本、两次完整 8 步+VAE 解码预热（`154.47s`）及同步评测/编码封装上。

---

## 2. 源码级预审计发现（0 CU 离线静态分析）

在启动 G4 实例前，我们在本地（Mac，消耗 `0.00 CU`）完整克隆并审计了 `ComfyUI`、`comfy-kitchen (v0.2.40)`、`r-vage/ComfyUI-sol-attn (v0.6.7)` 与 `kijai/ComfyUI-SolAttn_triton` 的 C++/CUDA/Triton 源码，发现了两项关键的底层实现机制缺陷：

### 2.1 发现一：上游 `BlockSparseAttention` 在 H3 上的 4K 分块生产者 (`ck.sol_attn_chunked`) 双遍 `qkv_proj` 冷启动惩罚
- 在 `comfy_extras/nodes_sparse_attention.py` 的 `h3_sparse_attention` 中，上游为节省约 `1.81 GB`（`1.68 GiB`，`42058 × 21504 × 2 B`）的完整 Q/K/V BF16 激活显存，将 `N = 42,058` 切分为 `11` 个 `PRODUCER_CHUNK = 4096` 的切片流式调用 `attn.qkv_proj(x[i:i+4096])`，并以**生成器函数 (`callable`)** 形式传给 `ck.sol_attn_chunked`。
- 在 `comfy_kitchen/backends/cuda/__init__.py` 第 3178–3187 行中，当首个稀疏步 `kmean is None` 时，`sol_attn_chunked` 会调用 `produce()` **两次**（第一遍统计 `kmean` 与 `vscale`，第二遍执行真正量化），导致首个稀疏步的 50 个 `DiTBlock` 将 `attn.qkv_proj`（`[42058, 5376] × [5376, 21504]` INT8 GEMM）重复执行了 **2 遍（单块 22 次 = 单步 `50 × 22 = 1,100` 次分块 GEMM）**！
- 而在拥有 **95 GiB 可用显存** 的 Blackwell G4 上（H3 总峰值分配显存仅 `27.04 ~ 27.46 GiB`，不到显存容量的 `29%`），若绕过 `make_h3_block_patch` 的 4K 分块器，直接使用 `BlockSparseAttention` 的 **`make_attention_override` -> 直通零拷贝 `ck.sol_attn` (`launch_sol_attn`)**：
  1. 每层 `attn.qkv_proj(x)` 保持为 **1 次完整 `[42058, 5376]` 大矩阵 INT8 GEMM**；
  2. `ck.rms_rope_split_half_` 原地融合执行；
  3. `ck.sol_attn` 直接在零拷贝跨步视图 `(1, 42058, 56, 128)` 上单遍完成预处理、路由与全 INT8 精确注意力，**彻底消除首步双遍 `qkv_proj` 启动惩罚**！

### 2.2 发现二：社区 `ComfyUI-sol-attn` 动态 $\tau$ 调度 (`_TauSchedule`) 的 Flow-Shift ($\alpha=12.0$) 进度畸变 Bug
- 在 `r-vage/ComfyUI-sol-attn/minimax.py` 第 492–498 行中，`_TauSchedule.progress(sigma)` 直接按线性公式 `(sigma_hi - sigma) / span = 1.0 - sigma` 计算采样进度，并在第 398 行按 `if progress < dense_percent:` 判断是否回退为稠密注意力。
- 然而 `MiniMaxH3Model` 的视频流采用了高达 **`sigma_shift_video = 12.0`** 的流匹配信噪比偏移：
  $$\sigma(p) = \frac{\alpha(1 - p)}{1 + (\alpha - 1)(1 - p)}, \quad \alpha = 12.0$$
- 在 8 步采样中，前 **7 步（Step 0 至 Step 6，占总步数 87.5%）** 的偏移后 $\sigma$ 均落在 `[1.0000, 0.8000]` 区间（第 6 步 float32 `sigma = 0.8000000119`，即 `progress = 1.0 - sigma = 0.199999988 < 0.20`），直到最后一步 Step 7 才降至 `0.6316`。
- 这导致社区原版 `MiniMaxH3ScheduledSolAttentionPatch` 在设置 `dense_percent = 0.20` 时，误将 **前 7 步（87.5% 的步数）全部回退为稠密注意力**，且 $\tau$ 退火曲线被压缩在 `[0.0, 0.368]` 区间内无法到达 `tau_end`！
- 通过对偏移公式精确求逆：
  $$t_{\text{unshifted}} = \frac{\sigma}{\alpha - (\alpha - 1)\sigma}, \quad p_{\text{true}}(\sigma) = 1 - t_{\text{unshifted}}$$
  即可还原 `[0.000, 0.125, 0.250, ..., 0.875]` 的线性步进进度（前 2 步 `25%` 稠密预热 + 后 6 步 `75%` 余弦退火稀疏）。

---

## 3. 12 组系统消融矩阵设计

为全面刻画 Blackwell G4 上的 **成本 (`CU`) × 速度 (`Denoise s` / `E2E s`) × 质量 (`Q_total`)** 帕累托前沿，我们设计了覆盖四大层级的 **12 组正交与递进消融矩阵**（统一固定输入图像、受控尾帧、Prompt、分辨率 `1344×768`、124 帧、8 步、`seed=42`）：

| Run ID | 方案代号 | 方案标签 | 稠密注意力后端 | 稀疏注意力 (`Sol-Attn`) 引擎与调度策略 | DiT 调制融合 | 核心消融目的 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `[D0-SDPA]` | `Run01_Baseline_SDPA` | PyTorch 2.11 SDPA (cuDNN BF16) | 无（100% 稠密） | 原生 Eager | **绝对质量黄金基准** (`S_ref=100`, `S_audio=100`) |
| **2** | `[D1-Sage]` | `Run02_Dense_Sage_Triton` | `SageAttention` 1.0.6 (Triton INT8-QK, BF16-PV) | 无（100% 稠密） | 原生 Eager | **ADR-02 稠密基线复测** |
| **3** | `[D2-CKInt8]` | `Run03_Dense_CK_Int8` | `comfy_kitchen` Pure-INT8 (`qk_int_sv_i8` CUDA INT8-QK+PV) | 无（100% 稠密） | 原生 Eager | 验证 `comfy-kitchen` 原生 CUDA 全 INT8 注意力 (`QK+PV`) 对比 Triton Sage v1 |
| **4** | `[D3-Sage+Mod]` | `Run04_Dense_Sage_FusedMod` | `SageAttention` 1.0.6 | 无（100% 稠密） | **Triton `H3FusedModulation`** | 验证社区 Triton AdaLN/Gate 比特级融合在稠密路径上的纯增益 |
| **5** | `[S1-CKChunk-Full]` | `Run05_CK_SolChunked_Full_t13` | `SageAttention` (仅 Refiner) | **原版分块 `ck.sol_attn_chunked`**: `0-100%` 步, `tau=1.3`, `tok=0`, `exact_kv_and_rows` | 原生 Eager | 测试上游默认 4K 分块流式生产者全步长稀疏的显存与双遍冷启动开销 |
| **6** | `[S2-CKChunk-W20-Tok256]` | `Run06_CK_SolChunked_W20_t13_Tok256` | `SageAttention` (`start=0.20`，前 2/8 步 = 25% 步数) | **原版分块 `ck.sol_attn_chunked`**: `start=0.20` (后 6/8 步 = 75% 步数), `tau=1.3`, `tok=256`, `exact_kv_and_rows` | 原生 Eager | 测试 ComfyUI 节点 Schema 默认参数 (`start=0.20, tau=1.3, extra_tokens=256`) |
| **7** | `[S3-CKDirect-Full]` | `Run07_CK_SolDirect_Full_t13` | `SageAttention` (仅 Refiner) | **直通零拷贝 `ck.sol_attn`**: `0-100%` 步, `tau=1.3`, `tok=0`, `exact_kv_and_rows` | 原生 Eager | **验证发现一**：绕过 4K 分块器，单遍全矩阵 `qkv_proj` + `ck.sol_attn` 极限提速 |
| **8** | `[S4-CKDirect-W20-Tok256]` | `Run08_CK_SolDirect_W20_t13_Tok256` | `SageAttention` (`start=0.20`，前 2/8 步 = 25% 步数) | **直通零拷贝 `ck.sol_attn`**: `start=0.20` (后 6/8 步 = 75% 步数), `tau=1.3`, `tok=256`, `exact_kv_and_rows` | 原生 Eager | 直通零拷贝 `ck.sol_attn` + 前 2 步 (`start=0.20`) Sage 稠密锚定 + 256 离群 Token 精确路由 |
| **9** | `[S5-CKDirect-HiFi-t10]` | `Run09_CK_SolDirect_HiFi_t10_Prot` | `SageAttention` (`start=0.20` 前 2/8 步 + 首尾层 `0,1,48,49`) | **直通零拷贝 `ck.sol_attn` 高保真**: `start=0.20` (后 6/8 步), `tau=1.0`, `tok=256`, `dense_blocks="0,1,48,49"`, `exact_kv_and_rows` | 原生 Eager | **高保真帕累托点**：$\tau=1.0$ + 256 额外 Token + 首尾敏感层稠密保护 |
| **10** | `[S6-Triton-Int8QK]` | `Run10_Community_Triton_Sol_Int8QK` | `SageAttention` (仅 Refiner) | **社区 Triton `MemoryEfficientSol`**: `0-100%` 步, `tau=1.3`, `int8_qk=True`, `int8_pv=False`, `exact_kv_and_rows` | 原生 Eager | 社区 Triton SM120 指针内核 (`_forward_int8_ptr` + `INLINE_Q`) 对比原生 CUDA `ck.sol_attn` |
| **11** | `[S7-Triton-Sched]` | `Run11_Community_Scheduled_Sol_Fixed` | `SageAttention` (`dense_percent=0.20`，前 2/8 步 = 25% 步数) | **社区 Triton `ScheduledSol` (偏移修正)**: `dense_percent=0.20` (后 6/8 步 = 75% 步数), `tau: 1.5->0.8` cosine, `int8_qk=True`, `int8_pv=True`, `exact_kv_and_rows` | 原生 Eager | **验证发现二**：修正 Flow-Shift 后的余弦动态 $\tau$ 退火调度 (`1.5 -> 0.8`) + `int8_pv` |
| **12** | `[S8-FullStack-Pareto]` | `Run12_FullStack_CKSolDirect_Mod` | `SageAttention` (`start=0.20`，前 2/8 步 = 25% 步数) | **直通零拷贝 `ck.sol_attn`**: `start=0.20` (后 6/8 步 = 75% 步数), `tau=1.3`, `tok=256`, `exact_kv_and_rows` | **Triton `H3FusedModulation`** | **全栈混合加速生产甜点**：直通 `ck.sol_attn` + `Sage` 预热 + `H3FusedModulation` |

---

## 4. GPU 核心推理聚焦工程优化方案（Zero-Waste GPU Lifecycle Pipeline）

针对实验 02 中非核心操作占用过高 CU 的问题，实验 03 实施了 6 项系统级工程优化：
1. **离线零成本预审计（0 CU Pre-Flight Inspection & Codex Gate）**：所有源码剖析、算子接口对齐与 Codex (`gpt-6-astra` `xhigh`) 代码静态审查均在本地 Mac 完成，杜绝在 G4 实例上进行交互式探针调试。
2. **并发环境引导与多线程重叠预取（Parallel Bootstrap & Overlapped Prefetch）**：使用线程池并发拉取 42.03 GB Golden 5-Pack 的 5 个权重文件，同时在后台线程预加载 5 个轻量评测模型（AlexNet、DINOv2、CLIP、RAFT、MUSIQ）。
3. **单次预编码与显存常驻（Single-Pass Conditioning & DiT Pinning）**：文本与首尾帧潜变量仅编码 1 次并卸载 Qwen3-VL；50 层 DiT 模型一次性加载常驻 G4 显存。
4. **轻量级 1 步算子预热替代双遍 8 步全量预热（1-Step JIT Warmup）**：利用算子微基准测试与单次 1 步去噪干跑完成 cuBLAS/CUDA/Triton JIT 编译缓存预热，将预热耗时从实验 02 的 `154.47s` 压缩至 `13.83s`（节省 `140.64s`）。
5. **CPU 异步多线程 MP4 封装与 SSIM 并行化（Async CPU Offloading）**：将 CPU 密集型的 `libx264` 视频编码、`ffmpeg` 音视频混流以及逐帧高斯模糊 SSIM 计算交由多线程池并行执行，不在主 GPU 消融循环内阻塞 96GB G4 显卡。
6. **图表与 12 宫格对比视频本地零成本渲染（Local Mac Post-Processing at 0 CU）**：G4 实例完成推理与指标导出后**立即单包拉取并关停实例 (`colab stop`)**；所有高分辨率帕累托前沿图、逐帧曲线图、关键帧长图及 12 宫格 124 帧对比视频均在本地 Mac 渲染。

---

## 5. 四维正交质量评估体系（与实验 02 严格一致）

沿用实验 02 的四维评估框架与权重公式（以 `Run 01 [D0-SDPA]` 为无损参考锚点）：
$$Q_{\text{total}} = 0.35 \cdot S_{\text{ref}} + 0.25 \cdot S_{\text{vbench}} + 0.25 \cdot S_{\text{fl2v}} + 0.15 \cdot S_{\text{audio}}$$
- **$S_{\text{ref}}$ (35%)**：相对 `Run 01` 的全参考保真度（`PSNR` / `SSIM` / `LPIPS`）。
- **$S_{\text{vbench}}$ (25%)**：VBench 视频时序与画质（`DINOv2` 主体一致性、`CLIP` 背景一致性、时序抗闪烁度、`RAFT` 光流运动平滑度、`MUSIQ` 图像美学质量）。
- **$S_{\text{fl2v}}$ (25%)**：首尾帧受控对齐一致性（首帧 40% + 尾帧 60% 的 DINOv2 / CLIP / SSIM 综合锚定得分）。
- **$S_{\text{audio}}$ (15%)**：生成音频保真度与音画同步（同时输出对齐 `run_ablation.py` 的对数幅度谱主口径 `S_audio` / `Q_total` 与对齐 `run_h100_sage_ablation.py` 的线性幅度谱补充口径 `S_audio_linear_supp` / `Q_total_linear_supp`）。

---

## 6. 模块化代码结构与端到端复现指南（Reproducibility Guide）

### 6.1 目录代码结构（所有 Python 文件均 `< 500` 行）
- [`ablation_config.py`](./code/ablation_config.py)（492 行）：实验常量、锁定 Commit（`ComfyUI` @ `f5ea88af...`、`ComfyUI-sol-attn` @ `69d2e400...`）与锁定 Pip 包版本（`comfy-kitchen==0.2.40`、`sageattention==1.0.6`、`triton==3.6.0`）、12 组消融矩阵、8 线程并发权重拉取。
- [`eval_metrics.py`](./code/eval_metrics.py)（490 行）：四维质量评测器 `QualityEvaluator`、帕累托前沿求解器 `compute_pareto_frontier`、不截断末帧的无损帧数 MP4 封装器 `save_video_with_audio`。
- [`microbench_verifications.py`](./code/microbench_verifications.py)（497 行）：9 算子微基准、`_C.sol_attn_plan` 块路由稀疏度遥测、Discovery 1/2 与 `MiniMaxH3FusedModulation` 位级等价验证。
- [`run_g4_sol_attn_ablation.py`](./code/run_g4_sol_attn_ablation.py)（495 行）：G4 端到端 12 组消融主控入口，以及本地会话账本对账入口（`--reconcile-ledger-only`）。
- [`render_visuals_local.py`](./code/render_visuals_local.py)（485 行）：本地 Mac 零 CU 图表与 12 宫格 124 帧对比视频渲染脚本（内置 `ffprobe` `nb_read_frames == 124` 强校验）。

### 6.2 远端 G4 单次会话执行与本地零 CU 渲染命令
```bash
# 1. 记录会话前 CU 余额并分配 Blackwell G4 实例
colab usage
colab new --name h3-exp03-g4 --gpu G4

# 2. 上传 4 个远端执行模块并启动单次全自动消融（日志实时落盘）
for f in ablation_config.py eval_metrics.py microbench_verifications.py run_g4_sol_attn_ablation.py; do
  colab upload -s h3-exp03-g4 "$f" "/content/$f"
done
colab exec -s h3-exp03-g4 "python3 -u /content/run_g4_sol_attn_ablation.py 2>&1 | tee /content/exp03_g4_outputs/g4_sol_attn_ablation_raw.log"

# 3. 单包拉取全部产物并立即销毁 G4 实例（杜绝闲置计费）
colab download -s h3-exp03-g4 /content/exp03_g4_outputs_bundle.tar ./exp03_g4_outputs_bundle.tar
colab stop -s h3-exp03-g4
colab usage

# 4. 解压产物、对账会话级 CU 账本并在本地 Mac 零 CU 渲染全部图表与 12 宫格 124 帧视频
tar -xf ./exp03_g4_outputs_bundle.tar -C ./outputs/
python3 run_g4_sol_attn_ablation.py --reconcile-ledger-only
python3 code/render_visuals_local.py
```
