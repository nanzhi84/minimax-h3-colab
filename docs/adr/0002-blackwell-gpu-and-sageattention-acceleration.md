<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# ADR-02：锁定 Google Colab Blackwell 96GB（`--gpu G4`）为标准推理机型，加速策略锁定为单开 `SageAttention`（`[S=1, C=0, T=0]`），禁用 `EasyCache` 与 `torch.compile`

- **状态（Status）**：已采纳（Accepted）
- **日期（Date）**：2026-10-23
- **关联实验**：[`experiments/02-a100-acceleration-ablation`](../../experiments/02-a100-acceleration-ablation/README.md)（完整实验报告：[`report.md`](../../experiments/02-a100-acceleration-ablation/report.md)；原始实测指标：[`ablation_metrics.json`](../../experiments/02-a100-acceleration-ablation/outputs/ablation_metrics.json)、[`h100_sage_ablation_metrics.json`](../../experiments/02-a100-acceleration-ablation/outputs/h100_sage_ablation_metrics.json)）
- **前置依赖**：[ADR-01：锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）与 HF 直连加载链路](./0001-model-weights-loading-pipeline.md)

---

## 1. 背景与问题（Context）

在基于 [ADR-01](./0001-model-weights-loading-pipeline.md) 锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）与本地 NVMe 直连加载链路后，运行 **MiniMax `H3-Base-FL2VA`**（`1344×768`、124 帧、24 FPS、8 步 Turbo LoRA 声画同源生成）还需做出两项核心工程决策：

1. **Google Colab GPU 实例选型问题**：
   - Colab 提供 `A100`（80GB Ampere SM 8.0）、`H100`（80GB Hopper SM 9.0）与 `G4`（`NVIDIA RTX PRO 6000 Blackwell Server Edition`，96GB GDDR7 SM 12.0）三类大显存实例。如何兼顾**单条视频推理耗时、每条视频消耗的计算单元（CU）成本、显存余量以及资源池分配成功率**？
2. **推理加速算子与编译策略取舍问题**：
   - 围绕 ComfyUI 原生推理管线中的三项主流无训练加速技术——**`SageAttention`（`S`）**、**`EasyCache`（`C`）** 与 **`torch.compile`（`T`，`TorchCompileModel(backend="inductor")`）**，如何在保证**首尾帧高像素锚定精度（`FL2VA` 核心价值）**的前提下获得最优的端到端加速与最低单视频 CU 成本？

---

## 2. 实验证据（Evidence）

在实验 02 中，我们固定输入首尾帧、Prompt 与随机种子（`seed=42`），分别在 **`NVIDIA A100-SXM4-80GB High-RAM`（SM 8.0）** 上完成了 $2^3=8$ 全因子正交消融，并在 **`NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB`（`--gpu G4`，SM 12.0）** 上完成了算子级微基准与端到端消融，同时结合 `colab-cli` 底层账本（`/tun/m/ccu-info`）审计了全量 CU 消耗：

### 2.1 跨代 GPU 实例与核心加速配置实测对比表

| 硬件平台 (Colab 实测费率) | 加速配置 `[S,C,T]` | 方案名称 | 去噪耗时 $T_{\text{denoise}}$ | 端到端耗时 $T_{\text{e2e}}$ (相对 A100 基线加速比) | 单条视频 CU 成本 (`条 / 100 CU`) | 首帧 / 尾帧像素 SSIM | 首尾帧锚定分 $S_{\text{fl2v}}$ | VBench 代理分 $S_{\text{vbench}}$ | 综合质量 $Q_{\text{total}}$ |
| :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Blackwell 96GB (`G4`)**<br>(`7.2046 CU/hr`) | **`[1,0,0]`** 🏆 | **`Only-Sage`（采纳标准）** | **`68.01s`** | **`72.34s` (`2.06x` 🚀)** | **`0.1448 CU`** (`690.8 条`) | **`0.9248` / `0.9031`** | **`94.25`** | **`73.30`** | **`75.98`** |
| **Blackwell 96GB (`G4`)**<br>(`7.2046 CU/hr`) | `[0,0,0]` | `Baseline (SDPA)` | `77.37s` | `81.69s` (`1.83x`) | `0.1635 CU` (`611.7 条`) | `0.9255` / `0.9031` | `94.28` | `73.50` | `91.95` |
| **Blackwell 96GB (`G4`)**<br>(`7.2046 CU/hr`) | `[0,1,0]` | `Only-Cache` (跳 2/8 步) | `57.92s` | `62.25s` (`2.40x`) | `0.1246 CU` (`802.7 条`) | `0.5583` / `0.5892` 🔻 | `74.88` 🔻 | `70.93` | `68.09` |
| **Blackwell 96GB (`G4`)**<br>(`7.2046 CU/hr`) | `[1,1,0]` | `Sage + Cache` (跳 2/8 步) | `51.01s` | `55.36s` (`2.70x`) | `0.1108 CU` (`902.7 条`) | `0.5566` / `0.5899` 🔻 | `75.13` 🔻 | `71.40` | `65.02` |
| **A100 80GB (`A100`)**<br>(`6.7700 CU/hr`) | `[0,0,0]` | `Baseline (SDPA)` | `139.95s` | `149.22s` (`1.00x`) | `0.2806 CU` (`356.3 条`) | `0.9247` / `0.9029` | `94.09` | `72.94` | `91.76` |
| **A100 80GB (`A100`)**<br>(`6.7700 CU/hr`) | `[1,0,0]` | `Only-Sage` (A100 负优化) | `157.09s` 🔻 | `166.36s` (`0.90x` 🔻) | `0.3129 CU` (`319.6 条`) | `0.9253` / `0.9028` | `94.11` | `73.30` | `80.98` |
| **A100 80GB (`A100`)**<br>(`6.7700 CU/hr`) | `[0,1,0]` | `Only-Cache` (跳 2/8 步) | `104.91s` | `114.16s` (`1.31x`) | `0.2147 CU` (`465.8 条`) | `0.5593` / `0.5917` 🔻 | `75.10` 🔻 | `71.88` | `69.20` |
| **A100 80GB (`A100`)**<br>(`6.7700 CU/hr`) | `[0,0,1]` | `Only-Compile` (Cold / Warm) | `187.27s` / `141.09s` | `196.54s` / `150.37s` (`0.99x` 🔻) | `0.3696` / `0.2828 CU` | `0.9250` / `0.9040` | `94.14` | `73.27` | `78.55` |

### 2.2 关键实测依据

1. **为什么锁定 Blackwell 96GB（`--gpu G4`）作为唯一主力机型？**
   - **单视频 CU 成本降低 `41.7% ~ 48.4%`**：尽管 `G4` 的小时费率（`7.2046 CU/hr`）比 `A100 High-RAM`（`6.77 CU/hr`）略高 `+6.42%`，但由于推理速度达 A100 的 **`1.83x ~ 2.06x`**，生成单条视频的 CU 消耗从 A100 基线的 `0.2806 CU` 降至 **`0.1448 CU`（`Only-Sage`，净省 `48.41%` CU，折合仅约 ¥0.10/条）**，每 `100 CU` 可产出 **`690.8 条`** 高保真锚定视频（是 A100 基线 `356.3 条` 的 **`1.94 倍`**）。
   - **默认即享 `95 GiB 显存 + 177 GiB 内存`**：`--gpu G4` 在标准形态（`machineShape: 0`，无需额外加 `--high-mem`）下即提供 **`94.97 GiB` GDDR7 显存与 `176.88 GiB` 系统内存**，比 A100/H100 多出 `16 GB` 显存余量；同时预编码耗时从 `16.83s` 缩短至 `7.24s`，视频 VAE 解码耗时从 `9.19s` 缩短至 `4.31s`。
   - **极高的资源池分配成功率**：实验期间 `H100` 连续 8 次请求均返回 `HTTP 503 Service Unavailable`，而 `--gpu G4` 仅耗时 **`3.5 秒`** 即一次分配成功。

2. **为什么只开启 `SageAttention`（`S=1`）？**
   - **算子级与端到端双重正优化**：在 Blackwell (SM 12.0) 上，针对 H3-FL2VA 真实序列维度（`[B=1, H=40, N=42058, D=128]`，BF16），`Comfy attention_sage` 单次调用耗时从 `98.914 ms`（`366.24 TFLOPS`）降至 **`83.653 ms`（`433.06 TFLOPS`，吞吐提升 `+18.2%`）**；端到端 `402` 次注意力调用累计将去噪耗时从 `77.37s` 压降至 **`68.01s`（加速 `1.14x`）**，总耗时降至 **`72.34s`（净省 `9.35s`）**。
   - **无损保持首尾帧像素锚定与内生画质**：在 Blackwell 上单开 `SageAttention` 时，**首尾帧锚定指标（`S_fl2v = 94.25` vs 基线 `94.28`；首/尾帧像素 SSIM `0.9248 / 0.9031` vs 基线 `0.9255 / 0.9031`）与 VBench 代理得分（`S_vbench = 73.30` vs 基线 `73.50`）与原生 SDPA 基线完全持平**，且零编译等待开销。

3. **为什么禁用 `EasyCache`（`C=0`）？**
   - **破坏首尾帧像素级结构锚定（`FL2VA` 核心任务退化）**：在 8 步 Turbo 模型中，`EasyCache`（跳 `2/8` 步）虽然能提速 `1.31x`，但无论在 A100 还是 Blackwell 上，都会导致**生成视频的首帧像素 SSIM 从 `0.925` 暴跌至 `0.556 ~ 0.559`、尾帧像素 SSIM 从 `0.903` 暴跌至 `0.589 ~ 0.592`**（`S_fl2v` 从 `94.2` 跌至 `75.0`）。对于以“精准首尾帧控制”为核心卖点的 `H3-Base-FL2VA` 而言，牺牲首尾帧像素还原度换取跳步得不偿失。

4. **为什么禁用 `torch.compile`（`T=0`）？**
   - **存在冷启动编译税且稳态负收益**：由于 INT8/FP8 自定义量化算子（`_try_quantized_linear_input_act`）、`attention_sage`、`prefetch_queue_pop` 以及 `comfy/ldm/minimax/model.py:675` 中 `Tensor.item()` 导致的计算图切分（Graph Breaks），开启 `TorchCompileModel(backend="inductor")` 首次运行需额外付出 **`23.76s ~ 46.17s`（`0.0447 ~ 0.0868 CU`）** 的编译耗时，而第二次热运行耗时反而比对应 Eager 模式慢 **`+0.58% ~ +2.53%`**（`breakeven_videos_vs_eager = null`，永远无法回本）。

---

## 3. 决策内容（Decision）

自本 ADR 起，后续所有 `MiniMax H3-Base-FL2VA` 的云端推理、评测与生产脚本统一执行以下硬件与加速标准：

1. **标准云端硬件锁定为 Google Colab Blackwell 96GB（`--gpu G4`）**：
   - 使用 `colab-cli` 创建会话时统一指定 `--gpu G4`（对应 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，SM 12.0，`94.97 GiB` VRAM / `176.88 GiB` RAM）：
     ```bash
     colab create --name <session-name> --gpu G4
     ```
2. **标准加速开关组合锁定为 `[S=1, C=0, T=0]`（仅启用 `SageAttention`）**：
   - ✅ **启用 `SageAttention`（`S=1`）**：在环境中安装 `sageattention`（`>=1.0.6`），并将 ComfyUI 注意力分发器绑定至 `comfy.ldm.modules.attention.attention_sage`（配合 `skip_reshape=True` 直接走 `tensor_layout="HND"` 零拷贝路径）。
   - ❌ **禁用 `EasyCache`（`C=0`）**：标准生产与评测流程中**不注入 `EasyCache` 跳步节点**，严格执行完整 8 步去噪采样，确保首尾帧像素级锚定 SSIM 稳定在 `> 0.90`（`S_fl2v > 94`）。
   - ❌ **禁用 `torch.compile`（`T=0`）**：**不挂载 `TorchCompileModel` 节点**，全程采用原生 PyTorch Eager 模式执行，消除 `24s ~ 46s` 的首次编译阻塞与 Graph Break 调度开销。
3. **跨架构降级兜底规则（Fallback Policy）**：
   - 若极端时段 `--gpu G4` 资源池满载返回 `HTTP 503` 而被迫回退至 **`--gpu A100 --high-mem`（Ampere SM 8.0）**，脚本必须根据 `torch.cuda.get_device_capability()[0] < 9` **自动关闭 `SageAttention`，切换为 `[S=0, C=0, T=0]`（原生 `PyTorch SDPA + Eager`）**，避免触发 A100 上 `+11.49%` 的耗时倒挂：
     ```python
     major, minor = torch.cuda.get_device_capability()
     # 仅在 Hopper (SM 9.0) / Blackwell (SM 10.0 / 12.0) 及更新架构上启用 SageAttention
     use_sage_attention = (major >= 9)
     ```

---

## 4. 影响与后果（Consequences）

### 正向收益（Pros）
- **兼具高画质锚定与极速低成本**：在 Blackwell 96GB + `Only-Sage [1,0,0]` 标准配置下，单条 `1344×768 × 124 帧` 有声视频仅需 **`72.34 秒`**（约 **1 分 12 秒**）、消耗 **`0.1448 CU`**（折合人民币约 **0.10 元/条**），比 A100 原生基线快 **`2.06 倍`**、节省 **`48.41%` CU**，同时完整保留首尾帧像素锚定精度（`S_fl2v = 94.25`）。
- **推理管线极简稳定**：剔除 `EasyCache` 与 `torch.compile` 后，彻底消除了首尾帧结构崩塌风险、Dynamo/Inductor FakeTensor 兼容性报错以及首次冷编译等待，做到“模型入显存即满速出片”。
- **96 GB 显存提供充足扩展空间**：当前峰值分配显存仅约 `27.2 GiB`，剩余近 `68 GiB` 显存余量可直接支撑后续更高分辨率或更长帧数（如 161 帧 / 200 帧）实验。

### 约束与注意事项（Cons & Mitigations）
- **相对同卡 SDPA 基线存在轻微轨迹偏移**：`Only-Sage` 虽与基线具有同等级别的 `S_fl2v`（`94.25` vs `94.28`）与 `S_vbench`（`73.30` vs `73.50`），但因量化注意力核数值舍入差异，中段运动轨迹相对同卡 `SDPA` 基线存在一定偏移（`PSNR = 20.19 dB`）。若特定回归测试要求与原生 PyTorch `F.scaled_dot_product_attention` 保持严格逐像素一致，可临时切回 `[S=0, C=0, T=0]`（耗时 `81.69s`，`0.1635 CU`）。
- **会话冷启动权重拉取摊销**：新建 `G4` 实例首次拉取 `42.03 GB` 权重需约 `115.6s`（`0.2313 CU`），因此在单次 Colab 会话中应集中批量执行生成任务，跑完后立即执行 `colab stop` 释放实例。
