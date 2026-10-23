<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# 实验 03 报告：MiniMax H3-Base-FL2VA 在 Blackwell G4 上的 Sol-Attn 与社区融合算子帕累托前沿消融及零浪费 GPU 生命周期工程

- **实验目录**：[`./`](./)（`experiments/03-g4-sol-attn-pareto-ablation/`）
- **实验设计文档**：[`README.md`](./README.md)
- **模块化执行与渲染代码（全部单文件 `< 500` 行）**：
  - G4 端到端消融主控与账本对账入口：[`run_g4_sol_attn_ablation.py`](./code/run_g4_sol_attn_ablation.py)
  - 实验配置、版本锁定与 8 线程并发预取：[`ablation_config.py`](./code/ablation_config.py)
  - 四维质量评测器（`QualityEvaluator`）与帕累托求解器：[`eval_metrics.py`](./code/eval_metrics.py)
  - 9 算子微基准、工作区稀疏遥测与 Discovery 1/2 检验：[`microbench_verifications.py`](./code/microbench_verifications.py)
  - 本地 Mac 零 CU 图表与 12 宫格 124 帧对比视频渲染器：[`render_visuals_local.py`](./code/render_visuals_local.py)
- **原始日志、结构化指标、脱敏账本与可视化产物**：
  - G4 标准输出完整日志：[`outputs/g4_sol_attn_ablation_raw.log`](./outputs/g4_sol_attn_ablation_raw.log)
  - 12 组全量指标与微基准 JSON：[`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json)
  - Colab 会话级 HTTP 分配/回传/停机与 CU 账本脱敏摘录：[`outputs/colab_session_ledger_excerpt.json`](./outputs/colab_session_ledger_excerpt.json)
  - 成本–速度–质量三维帕累托前沿图：[`outputs/pareto_frontier_cost_speed_quality.png`](./outputs/pareto_frontier_cost_speed_quality.png)
  - 算子微基准、源码缺陷验证与生命周期 CU 聚焦对比图：[`outputs/sol_attn_mechanisms_and_lifecycle_cu.png`](./outputs/sol_attn_mechanisms_and_lifecycle_cu.png)
  - 12 组实验五关键帧（`f=0, 31, 62, 93, 123`，对应 `0%, 25%, 50%, 76%, 100%`）对比长图：[`outputs/keyframe_strip_12runs.png`](./outputs/keyframe_strip_12runs.png)
  - 124 帧逐帧保真度（PSNR / SSIM / LPIPS）轨迹曲线：[`outputs/frame_metrics_curves_12runs.png`](./outputs/frame_metrics_curves_12runs.png)
  - 12 组同屏对比有声视频（3×4 网格，完整 124 帧 / `5.166667s`）：[`outputs/comparison_grid_12runs.mp4`](./outputs/comparison_grid_12runs.mp4)

---

## 1. 核心结论速览（Executive Summary）

本实验在 **ADR-01 黄金五件套（`42.03 GB`）** 与 **ADR-02 固定基准负载（`1344×768`、124 帧、24 FPS、8 步 Turbo LoRA、`seed=42`，联合音视频+文本序列长 $N = 42,058$）** 下，选用 **Google Colab `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`--gpu G4`，96 GB GDDR7，SM 12.0，实验 02 账本反推实测等效费率 `7.2046 CU/hr`）**，围绕 **Sol-Attn（Sparsifying Online Attention，`arXiv:2607.24027`）** 及社区加速方案开展了 **12 组正交与递进消融实验**，并同步重构了零浪费 GPU 生命周期工程流水线：

1. **Blackwell G4 上的成本–速度–质量帕累托前沿（Pareto Frontier）：在主口径 `E2E vs Q_total` 下由 `Run 07`、`Run 12`、`Run 08`、`Run 09`、`Run 03`、`Run 01` 六个点构成（在补充线性 STFT 口径 `E2E vs Q_total_linear_supp` 下 `Run 12` 严格支配 `Run 08`，收缩为 `[7, 12, 9, 3, 1]` 五个点），覆盖 `37.11s ~ 82.42s`（`0.0743 ~ 0.1649 CU / 视频`）全谱系**：
   - **极限低成本/极速档 —— `Run 07 [S3-CKDirect-Full]`（零拷贝直通 `ck.sol_attn`，0–100% 步，$\tau=1.3$，`tok=0`）**：8 步去噪耗时从 G4 SDPA 基线的 `78.06s` 暴降至 **`32.76s`（去噪加速 `2.38x`，较 G4 Sage 加速 `2.10x`）**，端到端耗时降至 **`37.11s`（较 G4 SDPA 加速 `2.22x`，较实验 02 A100 基线 `149.22s` 加速 `4.02x`）**；单视频边际成本仅 **`0.0743 CU`**（每 `100 CU` 可生成 **`1,346.55 条`**视频，比 G4 SDPA 省 **`54.98%` CU**、比实验 02 A100 基线 `0.2806 CU` 省 **`73.54%` CU**），且首尾帧锚定得分保持 **`S_fl2v = 94.57`**（$Q_{\text{total}} = 66.45$）。
   - **均衡甜点生产档 —— `Run 12 [S8-FullStack-Pareto]`（零拷贝直通 `ck.sol_attn` + 20% Sage 预热 + $\tau=1.3$ + `extra_tokens=256` + 社区 Triton `MiniMaxH3FusedModulation`）**：去噪耗时 **`42.37s`（去噪加速 `1.84x` vs SDPA / `1.62x` vs Sage）**，端到端耗时 **`46.72s`（`0.0935 CU / 视频`，每 `100 CU` 生成 `1,069.50 条`）**，$Q_{\text{total}} = 72.25$（`PSNR = 16.98 dB`，`S_vbench = 73.99`，`S_fl2v = 94.24`）。相比官方节点默认分块方案 `Run 06 [S2-CKChunk-W20-Tok256]`（`48.00s`，$Q_{\text{total}} = 72.17$），**去噪与端到端均净省 `1.27s ~ 1.28s` 延迟，同时提升了综合质量得分（`+0.09` $Q_{\text{total}}$，`72.2533 vs 72.1676`；`+0.10 dB` PSNR），在 `(延迟/CU, Q_total)` 二维目标下实现严格帕累托支配（Strict Dominance）**。
   - **高保真稀疏档 —— `Run 09 [S5-CKDirect-HiFi-t10]`（20% Sage 预热 + $\tau=1.0$ + `tok=256` + 首尾层保护 `dense_blocks="0,1,48,49"`）**：端到端耗时 **`50.72s`（`0.1015 CU / 视频`，`985.22 条 / 100 CU`）**，将稀疏注意力的 PSNR 推升至 **`17.41 dB`**（`SSIM = 0.7019`，`LPIPS = 0.1543`），并取得全场最高的 VBench 代理得分 **`S_vbench = 74.44`**（$Q_{\text{total}} = 73.30$）。
   - **稠密注意力重大突破 —— `Run 03 [D2-CKInt8]`（`comfy-kitchen` 原生 CUDA 全 INT8 稠密注意力 `qk_int_sv_i8`）在 `(延迟/CU, Q_total)` 上严格支配 `SageAttention 1.0.6`（`Run 02` 与 `Run 04`）**：去噪耗时 **`54.06s`（端到端 `58.41s`，`0.1169 CU / 视频`）**，相比本实验同会话复测的 `Run 02 [D1-Sage]`（去噪 `68.69s`、端到端 `73.03s`）**去噪净省 `14.62s`（提速 `1.27x`）、端到端净省 `14.61s`（提速 `1.25x`，节省 `20.01%` CU）**（若对照实验 02 G4 的 `Run 2` 原测值 `68.01s / 72.34s`，则去噪净省 `13.95s`、端到端净省 `13.93s`）；同时参考像素保真度与音频保真度显著领先（**`PSNR = 22.75 dB` vs Sage `20.19 dB`，`SSIM = 0.8257` vs `0.7711`，`LPIPS = 0.0672` vs `0.1075`，$S_{\text{ref}} = 72.08$ vs `63.90`，$S_{\text{audio}} = 93.63$ vs `91.64`，$Q_{\text{total}} = 80.98$ vs `78.00`**，仅 $S_{\text{vbench}}$ `72.72 vs 73.30` 与 $S_{\text{fl2v}}$ `94.11 vs 94.25` 略低），将 `SageAttention` 从 `(E2E, Q_total)` 稠密帕累托前沿完全取代！
2. **解决了实验 02 中 `EasyCache` 跳步导致的首尾帧锚定崩塌问题**：
   - 在实验 02 中，`EasyCache`（跳过 2/8 步）虽带来 `1.31x ~ 1.48x` 加速（Blackwell `55.36s`），但导致首尾帧像素级 SSIM 从 `0.925 / 0.903` 暴跌至 `0.557 / 0.590`（`S_fl2v` 跌至 `75.1`）。
   - 本实验中，**全部 8 组 Sol-Attn 稀疏配置（`Run 05 ~ Run 12`，端到端耗时范围 `37.11s ~ 59.39s`）在每个去噪步均保持完整计算但稀疏路由 $K/V$ 块，首尾帧锚定得分全部稳定在 `S_fl2v = 94.24 ~ 94.81`（首帧 SSIM `0.9235 ~ 0.9272`、尾帧 SSIM `0.9029 ~ 0.9100`）**；其中全部 6 组原生 `ck.sol_attn` 配置（`Run 05 ~ 09, 12`，`37.11s ~ 50.72s`）及社区 `Run 10`（`54.17s`）均比 `Sage + EasyCache`（`55.36s`）更快（社区动态退火 `Run 11` 为 `59.39s`），彻底消除了首尾帧结构崩塌。
3. **两项源码级架构缺陷发现与实证修复（Discovery 1 & Discovery 2）**：
   - **发现 1（`BlockSparseAttention` 4K 分块生产者在首步重复执行 22 次 `qkv_proj` 的开销 vs. 零拷贝直通 `ck.sol_attn`）**：上游 `nodes_sparse_attention.py` 为 H3 注册的 `h3_sparse_attention` 走 `ck.sol_attn_chunked`（每次切片 4,096 tokens），在首个稀疏步因 `kmean is None` 会将 `chunks()` 生成器迭代**两遍**（单块调用 **22 次** `qkv_proj` = 单步 **1,100 次** INT8 GEMM，微基准单块耗时 `67.633 ms`），后续 warm 步需 11 次切片调用（单块耗时 `52.365 ms`）并复用前一步缓存的 `kmean/vscale`。移除 `patches_replace["dit"]` 改走零拷贝直通 `ck.sol_attn` 后，每步每块仅执行 **1 次**全矩阵 `qkv_proj`（单步 50 次 GEMM，单块恒定耗时 `53.303 ms`）：虽然单次 warm 分块（`52.365 ms`）略快于直通（`53.303 ms`），但直通路径通过消除首步 `67.633 ms/blk` 的双倍投影启动惩罚，在 8 步端到端对比中稳定净省 **`0.52s ~ 0.64s`**（`Run 05→07` 去噪省 `0.52s` / E2E 省 `0.53s`；`Run 06→08` 去噪省 `0.63s` / E2E 省 `0.64s`），且综合质量得分 $Q_{\text{total}}$ 同步提升（`66.25 → 66.45`、`72.17 → 72.27`）。
   - **发现 2（社区 `r-vage/ComfyUI-sol-attn` 在 Flow-Matching $\alpha=12.0$ 偏移下的 `_TauSchedule` 进度失真）**：MiniMax H3 的视频噪声调度采用 $\alpha=12.0$ 的流匹配时间偏移，使前 7 步（Step 0~6）的 $\sigma \in [1.0000, 0.8000]$（Step 6 的 float32 $\sigma = 0.8000000119$）。社区 `minimax.py:398` 直接用 `progress = 1.0 - sigma` 并按 `progress < dense_percent` 判定是否回退稠密，导致设置 `dense_percent=0.20` 时 **8 步中有 7 步（Step 0~6，占总步数 `87.5%`）满足 `progress < 0.20` 而被强制回退为稠密注意力**！通过逆变换 $\hat{t} = \frac{\sigma}{\alpha - (\alpha - 1)\sigma}$ 还原真实线性步进度后（`Run 11`），恢复了 **前 2 步（占 8 步的 `25%`，`progress = 0.000, 0.125 < 0.20`）稠密预热 + 后 6 步（`75%`）余弦退火稀疏（$\tau: 1.3975 \rightarrow 0.8266$）**，修复后端到端耗时为 `59.39s`（$Q_{\text{total}} = 73.11$，$S_{\text{audio}} = 83.90$）。
4. **零浪费 GPU 生命周期工程（Zero-Waste GPU Lifecycle Engineering）：核心推理 CU 占比从实验 02 G4 的 `24.69%` 跃升至 `69.64%`（占打包前脚本计时范围 `72.30%`）**：
   - 本次实验仅启动 **1 次** G4 会话（`h3-exp03-g4`，客户端 HTTP 分配至释放总时长 **`958.12s` = `15.97 分钟`**），按实测等效费率 `7.2046 CU/hr` 折算理论会话消耗 **`1.9175 CU`**，Colab 账本最终结算净扣减 **`1.8494 CU`**（停机 49 分钟后复查余额仍为 `2481.6316 CU`，确认无任何后续异步扣费；远低于 `30 CU` 预算上限的 **1/15**）。
   - 其中 **12 次正式核心消融推理耗时 `667.28s`（`1.3354 CU`）**，占打包前脚本计时范围（`922.94s`）的 **`72.30%`**、占实例从分配到释放全生命周期（`958.12s`）的 **`69.64%`**——相比实验 02 A100 会话（`46.58%`）提升 **`+23.06` 个百分点**，相比实验 02 G4 会话（`24.69%`）提升 **`+44.95` 个百分点（核心推理时间占比提升 `2.82 倍`）**！

---

## 2. 实验环境、零浪费工程优化与会话级 CU 审计

### 2.1 软硬件环境与版本锁定

数据来源：[`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json)（`hardware` 与 `config` 字段）及 [`ablation_config.py`](./code/ablation_config.py)。

| 维度 | 实验 03 实测环境配置 |
| :--- | :--- |
| **GPU 型号与架构** | `NVIDIA RTX PRO 6000 Blackwell Server Edition`（Compute Capability `SM 12.0`，96 GB GDDR7，实测可用 `94.97 GiB`） |
| **主机 CPU 与内存** | `48` vCPUs / `176.88 GiB` System RAM |
| **基础软件栈** | Python `3.13`, PyTorch `2.11.0+cu130`, CUDA `13.0`, Triton `3.6.0`, `sageattention 1.0.6` |
| **稀疏与量化内核库** | `comfy-kitchen 0.2.40`（原生 CUDA `sol_attn` / `sol_attn_chunked` / `int8_attention` SM 12.0 预编译二进制） |
| **ComfyUI 主仓库版本** | Commit `f5ea88af1c46c50a39792d00af80e5af941f99f6`（内置 `BlockSparseAttention` 节点与 `AttentionTensorContainer`） |
| **社区扩展仓库版本** | `r-vage/ComfyUI-sol-attn` Commit `69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3`（`v0.6.7`，含 Triton `sol_attn`、`_TauSchedule` 与 `MiniMaxH3FusedModulation`） |
| **固定生成负载与序列划分** | `1344×768`（768p 宽屏）、124 帧、24 FPS、8 步、`res_multistep`、`seed=42`。联合序列总长 **$N = 42,058$** 由三段组成（对应 `mod_segments = [(0, 4562, 1), (4562, 4762, 2), (4762, 42058, 0)]`）：前缀音频与首尾参考帧条件 tokens **$4,562$** + 文本条件 tokens **$200$**（精炼器前缀合计 $4,762$ tokens，占前 $75$ 个 64-token 块）+ 目标视频潜空间 tokens **$37,296$**。单层完整 BF16 QKV 激活张量（$42058 \times 21504 \times 2\text{ B}$）为 **`1.81 GB`（`1.68 GiB`）**。 |

### 2.2 零浪费 GPU 生命周期工程优化 vs. 实验 02 会话级 CU 拆解对比

在实验 02 中，A100 会话（`64.77 min`，`7.3077 CU`）有 `53.42%` 的时间消耗在非核心推理操作上；Blackwell G4 会话（`18.33 min`，`2.2015 CU`）虽已采用 HF 并发拉取权重，但仍有 `75.31%` 的时间消耗在独立环境探针进程（`287.98s`）、两次完整 8 步预热（`154.47s`）及远端同步 MP4 封装与绘图回传（`245.87s`）上。

在实验 03 中，我们在 [`run_g4_sol_attn_ablation.py`](./code/run_g4_sol_attn_ablation.py) 及其子模块中实施了五项工程优化：
1. **离线静态源码审计与首轮 Codex 门禁（`0.00 CU`）**：在启动 G4 实例前，在本地 Mac 完成对 `comfy-kitchen`、`ComfyUI`、`r-vage/ComfyUI-sol-attn` 及模型 safetensors 头部的源码审查与 Codex（`gpt-6-astra xhigh`）静态验证，提前捕获并修复了 `adaln_t_table` 8 维曲线插值等潜在异常，实现 **G4 零交互调试、一次运行成功**。
2. **多线程并发拉取 42.03 GB 黄金五件套 + 评测权重后台重叠预取（Phase 1）**：使用 `ThreadPoolExecutor(max_workers=8)` 同时提交 **5 个 H3 模型权重下载任务**（5 个文件各自下载耗时之和为 `310.19s`，其中最慢的单文件 20.97 GB DiT 耗时 `123.3s`）、1 个输入首帧图片下载任务，以及 **1 个串行预加载 5 个评测模型（LPIPS / DINOv2 / CLIP / RAFT / MUSIQ）的后台任务**（耗时 `136.24s`）。整个并发下载阶段挂钟耗时为 **`136.245s`（日志打印 `136.24s`）**——后台评测预取与 5 路 H3 权重下载重叠了 `123.3s`，仅将下载阶段挂钟尾端延长约 `12.9s`，使 Phase 5 评测模型加载几乎无网络等待。
3. **进程内微基准 + 1 步轻量预热替代 2×8 步冗余预热（Phase 3）**：在已常驻显存的 DiT 上用 `24.00s` 完成 9 组算子微基准与 Discovery 1/2 实证检验（同时完成全部 CUDA/Triton 内核 JIT 预热），随后仅用 **1 步 DiT + 音视频 VAE 解码（`13.829s`）** 预热完整采样与解码流水线，相比实验 02 G4 的两次 8 步预热（`154.47s`）直接节省 **`140.64s`（`0.2815 CU`）**。
4. **CPU 异步线程池重叠 12 路 H.264+AAC MP4 编码（Phase 4）**：每次 GPU 解码出 `frames_np` 与 `waveform_np` 后，立即投入后台 `ThreadPoolExecutor(max_workers=3)` 异步调用 `ffmpeg` 封装 MP4（12 路后台编码任务累计挂钟耗时 `17.713s`），使 12 组循环总挂钟耗时（`671.839s`）与 12 次纯 GPU 推理耗时之和（`667.279s`）仅相差 **`4.560s`（`0.0091 CU` 的模型克隆与显存清理间隙）**。
5. **单包 Tar 归档极速回传 + 本地 Mac 零 CU 渲染（Phase 5）**：在远端将 12 路 MP4 与指标 JSON 打包为单个未压缩 `/content/exp03_g4_outputs_bundle.tar`（HTTP `Content-Length = 167,521,347` 字节 = `167.52 MB` / `159.76 MiB`），单次流式下载（`26.598s`）后立即触发 `colab stop` 释放实例；所有 Matplotlib 高清图表与 `comparison_grid_12runs.mp4` 均在本地 Mac 上渲染（消耗 `0.00 CU`）。

#### 三次实验会话（Session-Level）全生命周期 CU 账本对比表

数据来源：[`outputs/colab_session_ledger_excerpt.json`](./outputs/colab_session_ledger_excerpt.json)（摘自 `~/.config/colab-cli/colab.log` 的 `/tun/m/assign`、`/tun/m/unassign`、`/tun/m/ccu-info` 及 tar 下载响应）与 [`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json)（`lifecycle_telemetry` 字段）。

| 会话标识与硬件 | 生命周期阶段 | 挂钟耗时 (秒 / 分钟) | 折算 CU 消耗 | 占会话总时长比例 | 工程实现对比说明 |
| :--- | :--- | :-: | :-: | :-: | :--- |
| **实验 02：`h3-exp02`**<br>`A100-SXM4-80GB High-RAM`<br>费率：`6.77 CU/hr`<br>*(12 次正式运行)* | 非核心开销（下载/排障/预热/评测/回传） | `2,075.97s` (`34.60m`) | `3.9040 CU` | `53.42%` | 含编译异常排障、150.28s 预热、同步评测绘图与逐文件回传。 |
| ↳ | **核心消融推理（12 次运行）** | **`1,809.96s` (`30.17m`)** | **`3.4037 CU`** | **`46.58%`** | 8 次 Cold/Eager + 4 次编译 Warm 运行。 |
| ↳ | **实验 02 A100 会话合计** | **`3,885.93s` (`64.77m`)** | **`7.3077 CU`** | **`100.00%`** | 账本差值 `7.6351 CU`（含实验 01 停机未结算尾差 `0.3274 CU`）。 |
| **实验 02：`h3-exp02-gpu`**<br>`RTX PRO 6000 Blackwell (G4)`<br>费率：`7.2046 CU/hr`<br>*(4 次正式运行)* | 非核心开销（独立探针/下载/2×8步预热/评测/回传） | `828.43s` (`13.81m`) | `1.6579 CU` | `75.31%` | 独立进程探针 (`287.98s`) + 两次 8 步预热 (`154.47s`) + 同步封装回传 (`245.87s`)。 |
| ↳ | **核心消融推理（4 次运行）** | **`271.64s` (`4.53m`)** | **`0.5436 CU`** | **`24.69%`** | 仅完成 4 次正式推理 (`Run 1, 2, 3, 5`)。 |
| ↳ | **实验 02 G4 会话合计** | **`1,100.07s` (`18.33m`)** | **`2.2015 CU`** | **`100.00%`** | 前后账本完全结算净差值 `2.2015 CU`。 |
| **实验 03：`h3-exp03-g4` (本次)**<br>`RTX PRO 6000 Blackwell (G4)`<br>(`gpu-g4-s-kkb-use5a0-1svx8j9jrgagq`)<br>实测等效费率：**`7.2046 CU/hr`**<br>*(12 次正式运行)* | **1. 环境构建 + 5 路 H3 权重并发下载 & 评测模型后台预取** | `165.07s` (`2.75m`) | `0.3304 CU` | `17.23%` | `git+pip` (`28.70s`) + 并发下载挂钟 `136.245s`（5 个 H3 文件独立耗时和 `310.19s`，最慢单文件 `123.3s`；评测预取线程 `136.24s`）。 |
| ↳ | **2. 一次性共享预编码 (`Qwen3-VL` + 首尾帧) 与 DiT 加载** | `9.46s` (`0.16m`) | `0.0189 CU` | `0.99%` | 预编码 `7.131s`（随后卸载 32B 文本编码器）+ DiT/LoRA 加载 `2.328s`。 |
| ↳ | **3. 进程内算子微基准 + 源码验证 + 1 步轻量预热** | `37.83s` (`0.63m`) | `0.0757 CU` | `3.95%` | 9 算子微基准与 Discovery 1/2 验证 (`23.996s`) + **1 步轻量预热 (`13.829s`，较实验 02 省 `140.64s`)**。 |
| ↳ | **4. 十二组配置核心消融推理（12 次正式运行）** | **`667.28s` (`11.12m`)** | **`1.3354 CU`** | **`69.64%`**<br>*(占脚本计时 `72.30%`)* | **去噪合计 `615.057s` (`1.2309 CU`) + 视频 VAE `51.785s` + 音频 VAE `0.437s`**；后台异步 MP4 编码任务累计耗时 `17.713s` 与推理重叠（循环间隙 `4.560s` = `0.0091 CU`）。 |
| ↳ | **5. 四维质量评测 (`32.06s`)、阶段过渡及脚本外连接/Tar回传/停机** | `78.48s` (`1.31m`) | `0.1571 CU` | `8.19%` | 含脚本计时内残余 `43.308s`（循环间隙 `4.560s` + 四维评测 `32.060s` + 阶段间卸载/导入 `6.688s`，与前四项合计构成打包前脚本计时 `922.943s` = `1.8471 CU`），以及脚本计时范围外的 `35.176s`（`0.0704 CU`：**含脚本启动前实例分配至内核就绪 `4.381s` = `0.0088 CU`**，以及内核就绪后脚本进程启停、JSON/Tar 归档、单包下载 `26.598s` 与 `unassign` 停机共 **`30.795s` = `0.0616 CU`**）。 |
| ↳ | **实验 03 G4 会话合计（`05:07:57.958` $\rightarrow$ `05:23:56.077`）** | **`958.12s` (`15.97m`)** | **`1.9175 CU`**<br>*(账本净扣 `1.8494 CU`)* | **`100.00%`** | 会话前余额 `2483.4810 CU` (`05:06:51`) $\rightarrow$ 停机后余额 `2481.6316 CU` (`05:23:57` 及 49 分钟后 `06:13:09` 复核完全不变，**最终已结算账本净扣减 `1.8494 CU`**；与按客户端 HTTP `assign → unassign` 挂钟时长 `958.119s` 线性折算的 `1.9175 CU` 相差 `0.0681 CU`，按 `7.2046 CU/hr` 折算**等效于约 `34.02s` 的时长差**——因客户端日志仅记录 HTTP 请求时刻与 `ccu-info` 余额、不记录服务端内部计费表起止时刻，本报告同时披露理论折算值与账本净扣值）。**完成的正式运行数是 `h3-exp02-gpu` 的 3 倍（12 vs 4），总消耗反而降低 `12.9%`（仅占实验 02 两会话总耗 `9.5092 CU` 的 `20.16%`）！** |

---

## 3. 底层算子微基准与两项源码级发现实证（Phase 3A/3B）

### 3.1 H3 真实联合序列形状（`B=1, H=56, N=42058, D=128, BF16`）算子级微基准

在实验 02 中，微基准使用的是理想连续张量（`H=40`）；而在实验 03 中，我们严格按照 `MiniMaxH3` 真实注意力头数（**`H=56, D=128`**，单次前向理论计算量 $4 B H N^2 D = 50.717194\text{ TFLOPs}$）以及 `qkv_proj` 切分后的**非连续跨步视图（Strided View，`stride(0)=21504`）**，对 9 种稠密与稀疏注意力算子进行了微基准评测（3 次预热 + 8 次计时同步，数据来自 [`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json) `microbenchmark.operators`）：

| 算子实现与配置 | 传入包装 / 布局 | 单次调用耗时 (ms) | 等效算力 (eff-TFLOPS) | 400 次调用预计总耗时 | 相对 `SDPA` 加速比 | 单层输出 CosineSim / Rel-L1 (vs SDPA) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| `PyTorch SDPA` (`attention_pytorch`, BF16 Dense) | `Container (Strided HND)` | `139.432 ms` | `363.74 TFLOPS` | `55.77s` | `1.000x` (基线) | `1.000000` / `0.000000` |
| `SageAttention 1.0.6` (`attention_sage`, Triton INT8-QK + BF16-PV) | `Container (Strided HND)` | `118.353 ms` | `428.52 TFLOPS` | `47.34s` | `1.178x` | `0.999908` / `0.013300` |
| **`CK Pure-INT8 Dense` (`attention_comfy_kitchen_int8`, CUDA INT8-QK+PV)** | **`Container (Prequant HND)`** | **`81.966 ms`** | **`618.76 TFLOPS`** | **`32.79s`** | **`1.701x` (较 Sage 快 `1.444x` 🚀)** | `0.999855` / `0.016778` |
| `CK Sol-Attn Direct` ($\tau=1.0$, `tok=256`, `sink_q=[67,75]`) | `Strided NHD` | `32.385 ms` | `1566.05 TFLOPS` | `12.95s` | `4.305x` | `0.711718` / `0.709745` |
| **`CK Sol-Attn Direct` ($\tau=1.3$, `tok=256`, `sink_q=[67,75]`)** | **`Strided NHD`** | **`27.308 ms`** | **`1857.25 TFLOPS`** | **`10.92s`** | **`5.106x` 🚀** | `0.686713` / `0.731919` |
| **`CK Sol-Attn Direct` ($\tau=1.3$, `tok=0`, `sink_q=[67,75]`)** | **`Strided NHD`** | **`23.948 ms`** | **`2117.85 TFLOPS`** | **`9.58s`** | **`5.822x` 🚀** | `0.684522` / `0.734167` |
| `Community Triton Sol` ($\tau=1.3$, `INT8-QK, BF16-PV`, `sink_q=(0,75)`) | `Strided NHD` | `62.637 ms` | `809.71 TFLOPS` | `25.05s` | `2.226x` | `0.725218` / `0.655437` |
| `Community Triton Sol` ($\tau=1.3$, `INT8-QK, BF16-PV`, 对齐 `sink_q=(67,75)`) | `Strided NHD` | `47.772 ms` | `1061.66 TFLOPS` | `19.11s` | `2.919x` | `0.688212` / `0.729989` |
| `Community Triton Sol` ($\tau=1.3$, `INT8-QK+PV`, `sink_q=(0,75)`) | `Strided NHD` | `60.085 ms` | `844.10 TFLOPS` | `24.03s` | `2.321x` | `0.725152` / `0.656447` |

*(注：稀疏算子在独立随机正态输入下的单层 CosineSim 约 `0.68 ~ 0.73`，是因为随机高斯张量各块内积近似同分布，**Sparsifying Online Attention** 的在线阈值 $\tau$ 仅将约 20%~30% 的块路由至精确注意力分支，其余 70%~80% 的块通过由块池化/均值摘要与在线 Softmax 统计状态 (`lse`) 构造的近似尾项（approximate pooled tail）做低秩补偿保留——注意源码中的 `kmean/vscale` 分别用于 $K$ 块均值中心化与 $V$ 量化缩放，而非直接等同于近似尾项；在真实 DiT 尖锐的时空注意力分布下，精确路由分支能够捕获主要能量峰。)*

#### `comfy-kitchen` 原生 `_C.sol_attn` 合成输入块路由稀疏度遥测（`N=42,058`，总块数 $N_{tb} = \lceil 42058 / 64 \rceil = 658$）

通过调用 `comfy_kitchen.backends.cuda._C.sol_attn_plan` 与 `_C.sol_attn`（在合成正态分布张量 `q_nhd_strided, k_nhd_strided, v_nhd_strided` 上显式传入工作区缓冲区 `ws`，配置 `sink_blocks=[0,75], sink_q=[67,75]`），并解析工作区中的块路由计数表 `cnt`（维度 `BH=56 × NQ=658`），我们获得了合成高斯输入下不同 $\tau$ 与 `extra_tokens` 的块路由遥测数据：

| 温度参数 $\tau$ | `extra_tokens` (`token_aug`) | 视频查询块平均路由块数 (`/ 658 块`) | 全查询块平均路由块数 (`/ 658 块`) | 视频查询精确块保留率 (`Kept %`) | **合成输入块路由稀疏率 (`Sparsity %`)** | `tokCnt` 采样切片均值 *(注)* | 路由表工作区显存 (`Workspace`) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `0.8` | `0` | `200.42 块` | `205.75 块` | `30.46%` | **`69.54%`** | `0.00` | `1,003.87 MiB` |
| **`1.0`** | **`256`** | **`169.53 块`** | `175.22 块` | **`25.76%`** | **`74.24%`** | `38.08` *(缓冲区前 4,648 项切片)* | `1,363.30 MiB` |
| **`1.3`** | **`0`** | **`133.73 块`** | `139.83 块` | **`20.32%`** | **`79.68%`** | `0.00` | `1,003.87 MiB` |
| **`1.3`** | **`256`** | **`133.73 块`** | `139.83 块` | **`20.32%`** | **`79.68%`** | `32.88` *(缓冲区前 4,648 项切片)* | `1,363.30 MiB` |
| `1.5` | `0` | `116.38 块` | `122.67 块` | `17.69%` | **`82.31%`** | `0.00` | `1,003.87 MiB` |

> [!NOTE]
> **遥测数据适用范围与 `tokCnt` 维度审计说明**：
> 1. 上表为**合成随机正态张量（Synthetic Gaussian Input）**下的块路由统计，用于量化在线阈值 $\tau$ 对工作区与路由比例的基准影响，并非在真实去噪步激活上采集的逐层动态稀疏率。
> 2. 工作区总大小（`1,003.87 MiB` / `1,363.30 MiB`）与块级路由计数 `cnt`（`56 × 658 = 36,848` 项）均为全量精确统计。对于额外单 token 路由计数缓冲区 `tokCnt`，`comfy-kitchen` v0.2.40 源码（`sol_layout.cuh:46`）定义每 2 个查询块为一组（`TOK_GROUP = 2`，每头共 `ng = (658 + 1) // 2 = 329` 组，全量 `56 × 329 = 18,424` 项）；本次 VM 运行时旧脚本按 `ng = (658 + 7) // 8 = 83` 切片读取了缓冲区前 `4,648` 项（得出 `38.08 / 32.88`），因此该列仅代表 `tokCnt` 缓冲区前 `4,648` 个组条目的切片均值，不能外推为全体 `329` 组的精确均值。模块化代码 [`microbench_verifications.py`](./code/microbench_verifications.py) 已修正为 `ng = (nq + 1) // 2`。

### 3.2 两项源码级缺陷发现与实证验证（Discovery 1 & Discovery 2 + FusedModulation）

1. **Discovery 1：上游 `BlockSparseAttention` 4K 分块生产者的首步双倍 `qkv_proj` 开销 vs. 零拷贝直通 `ck.sol_attn`**
   - **源码机制**：ComfyUI 上游 `comfy_extras/nodes_sparse_attention.py` 在执行 `BlockSparseAttention` 时同时注册了两个钩子：一是通过 `optimized_attention_override` 注册的零拷贝直通 `ck.sol_attn`，二是通过 `patches_replace["dit"]` 为 MiniMax H3 专门注册的 `h3_sparse_attention`（调用 `ck.sol_attn_chunked`，将 $N=42,058$ 切成 $\lceil 42058 / 4096 \rceil = 11$ 个 chunk 流式产生 QKV）。
   - **实测证据（[`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json) `discovery1_block_benchmark`）**：在真实 H3 `blocks[0]` 上对 `blk0.qkv_proj` 挂载调用计数器实测：
     - **分块路径首个稀疏步（Cold Bootstrap，`kmean=None`）**：`ck.sol_attn_chunked` 先遍历一遍 `chunks()` 计算全局 `kmean/vscale`，再遍历第二遍 `chunks()` 执行稀疏注意力，导致**单个 DiTBlock 调用 `qkv_proj` 高达 `22` 次（单步 50 层共 `1,100` 次 INT8 矩阵乘！），单块耗时高达 `67.633 ms`**。
     - **分块路径后续稀疏步（Warm Pooled）**：复用上一步缓存的 `kmean/vscale`，单块调用 **`11` 次** `qkv_proj`（单步 `550` 次小矩阵乘，单块耗时 `52.365 ms`，与直通路径同输入输出余弦相似度 `0.999984`、`Rel-L1 = 0.008976`）。
     - **零拷贝直通路径（Direct Zero-Copy `ck.sol_attn`）**：仅需移除 `patches_replace["dit"]`，H3 的 `AttentionTensorContainer` 即通过 `skip_reshape=True` 将跨步视图零拷贝传入 `ck.sol_attn`，**每块仅调用 `1` 次全矩阵 `qkv_proj`（单步 `50` 次大矩阵乘，单块恒定耗时 `53.303 ms`）**。
     - **收益归因**：注意在单块微基准中，warm 分块（`52.365 ms`）略快于直通（`53.303 ms`），但因为分块路径在首个稀疏步必须付出 `67.633 ms/blk`（多出 `+14.33 ms/blk × 50 块 ≈ 0.72s`）的双遍投影启动惩罚，6 个稀疏步（300 块）合算下来直通路径（`15.99s`）反而比分块路径（`16.47s`）净省 `0.48s`。在 8 步端到端实测中，直通路径稳定净省 **`0.52s ~ 0.64s`**（`Run 05 → 07`：去噪 `33.28s → 32.76s` 省 **`0.52s`**，E2E `37.64s → 37.11s` 省 **`0.53s`**；`Run 06 → 08`：去噪 `43.64s → 43.01s` 省 **`0.63s`**，E2E `48.00s → 47.36s` 省 **`0.64s`**），且综合质量得分 $Q_{\text{total}}$ 同步提升（`Run 05→07`: `66.25 → 66.45`；`Run 06→08`: `72.17 → 72.27`，其中 `Run 08` 的 $S_{\text{vbench}}$ 与 $S_{\text{audio}}$ 提升，参考 PSNR 为 `16.85 vs 16.88 dB`）。
2. **Discovery 2：社区 `r-vage/ComfyUI-sol-attn` 在 Flow-Matching $\alpha=12.0$ 时间偏移下的 `_TauSchedule` 进度失真与修复**
   - **源码机制**：MiniMax H3 的视频流匹配调度器使用 `sigma_shift_video = 12.0`，8 步采样的 $\sigma$ 序列为 `[1.0000, 0.9882, 0.9730, 0.9524, 0.9231, 0.8780, 0.8000, 0.6316]`（其中第 6 步 float32 精确值为 `0.8000000119`）。社区 `minimax.py` 中 `_TauSchedule.progress(sigma)` 直接按线性公式 `1.0 - sigma` 计算进度，得到 `[0.0000, 0.0118, 0.0270, 0.0476, 0.0769, 0.1220, 0.2000, 0.3684]`（第 6 步 float32 `progress = 0.199999988`）。
   - **实测证据（`discovery2_schedule_verification`）**：由于 `minimax.py:398` 的稠密回退判定条件为 **`if progress < dense_percent:`**，当按推荐配置设置 `dense_percent = 0.20` 时，**前 7 步（Step 0 ~ Step 6，占 8 步的 `87.5%`）全部满足 `progress < 0.20` 而被回退为稠密注意力**，仅最后 1 步（Step 7）启用了稀疏注意力！
   - **修复效果**：通过流匹配逆映射 $\hat{t} = \frac{\sigma}{\alpha - (\alpha - 1)\sigma}$（$\alpha=12.0$）还原未偏移时间后，8 步进度恢复为等距线性序列 `[0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]`，在 `dense_percent = 0.20` 阈值下实现 **前 2 步（Step 0~1，占 8 步的 `25%`）稠密预热 + 后 6 步（Step 2~7，占 8 步的 `75%`）余弦退火稀疏（$\tau$ 从 Step 2 的 `1.3975` 平滑退火至 Step 7 的 `0.8266`）**。修复后的 `Run 11` 端到端耗时为 `59.39s`（未单独执行未修复版完整 8 步生成，因其在 7/8 步等价于 `Run 02` 稠密 `SageAttention`）。
3. **社区 Triton `MiniMaxH3FusedModulation` 位级等价性（Bit-Exactness）实证**：
   - 社区 `h3_fusion.py` 将每个 `DiTBlock` 中 4 次基于 `mod_segments` 的分段 Python 切片调制操作替换为 **4 次融合 Triton 算子调用**（2 次 `fused_rmsnorm_modulate` / `fused_modulate_` + 2 次 `fused_gate_residual` / `fused_gate_add_`）。
   - 实测在 `N=42,058` 真实 `blk0` 上，Triton 融合调制与原生 Eager 实现的**最大绝对误差 `mod_max_abs_diff = 0.0`、`gate_max_abs_diff = 0.0`（100% 位级等价 `Bit-Exact = True`）**，单块 4 次调制耗时从 `4.567 ms` 降至 `3.086 ms`（400 个 block 预计节省 `0.59s`；在端到端实测中，`Run 02→04` 去噪净省 **`0.73s`** / E2E 净省 **`0.72s`**，`Run 08→12` 去噪净省 **`0.64s`** / E2E 净省 **`0.64s`**）。

![算子微基准、源码缺陷验证与生命周期 CU 聚焦对比图](outputs/sol_attn_mechanisms_and_lifecycle_cu.png)

---

## 4. Blackwell G4 十二组消融实验结果与帕累托前沿分析

### 4.1 十二组消融配置速度、CU 成本、注意力入口计数与四维质量总表

数据来源：[`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json) 与 [`outputs/g4_sol_attn_ablation_raw.log`](./outputs/g4_sol_attn_ablation_raw.log)。
- **计数口径说明**：表中「DiT 注意力入口与融合 Block 计数」记录的是 Python 级注意力入口函数调用次数（全模型 8 步共 **`402` 次**：其中 `SingleTokenRefinerBlock` 文本精炼器在 8 步中固定调用 `2` 次稠密注意力，50 层主干 `DiTBlock` 在 8 步中调用 **`400` 次**稠密或稀疏注意力）以及启用融合调制的 `DiTBlock` 命中次数（`400 Mod`，每块内部执行 4 次融合 Triton 调制操作），而非底层 CUDA kernel launch 总数。对于 `Run 03 [D2-CKInt8]`，`ck_int8` 计数器合计 `402` 次（对应 400 次 `DiTBlock` 容器调用 `ck.int8_attention_from_prequantized` + 2 次 `SingleTokenRefinerBlock` 张量调用 `ck.int8_attention`）。
- **跨卡加速比与 CU 基准说明**：单视频 CU 按 G4 实测等效费率 **`7.2046 CU / hr`**（`0.0020012778 CU / s`）折算；跨卡加速比按实验 02 A100 基线表列均值常量 `A100_BASELINE_E2E_SEC = 149.22s` 折算，跨卡省 CU 比例按实验 02 A100 基线精确成本 `A100_BASELINE_CU = 0.280617 CU`（`149.2165s × 6.77 / 3600`）折算。

| Run | 简码 | 方案族与核心配置 | 去噪耗时 $T_{\text{den}}$ (vs SDPA / Sage) | 端到端 $T_{\text{e2e}}$ (vs SDPA / A100) | 单视频 CU (`条/100 CU`) | 相对 G4 SDPA / A100 基线省 CU | 注意力入口与融合 Block 计数 (`Dense + Sparse` / `FusedMod`) | 峰值显存 | $S_{\text{ref}}$ (35%) | $S_{\text{vbench}}$ (25%) | $S_{\text{fl2v}}$ (25%) | $S_{\text{audio}}$ (15%) | 综合质量 $Q_{\text{total}}$ ($Q_{\text{lin}}$) | 帕累托前沿 (`Q_total` / `Q_lin`) |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **01** | `[D0-SDPA]` | **Dense Baseline**: PyTorch cuDNN SDPA (BF16) | `78.06s` (`1.00x` / `0.88x`) | `82.42s` (`1.00x` / `1.81x`) | `0.1649 CU` (`606.25`) | `0.00%` / `-41.22%` | `402 SDPA` / `0 Mod` | `27.04 GiB` | `100.00` | `73.50` | `94.28` | `100.00` | **`91.95`** (`91.95`) | ⭐ **是 / 是** |
| **02** | `[D1-Sage]` | **ADR-02 Baseline**: `SageAttention 1.0.6` (INT8-QK) | `68.69s` (`1.14x` / `1.00x`) | `73.03s` (`1.13x` / `2.04x`) | `0.1461 CU` (`684.25`) | `-11.40%` / `-47.92%` | `402 Sage` / `0 Mod` | `27.18 GiB` | `63.90` | `73.30` | `94.25` | `91.64` | `78.00` (`75.98`) | 否 *(被 R03 支配)* |
| **03** | `[D2-CKInt8]` | **Native CK Pure-INT8 Dense** (`qk_int_sv_i8` QK+PV) | **`54.06s` (`1.44x` / `1.27x`)** | **`58.41s` (`1.41x` / `2.55x`)** | **`0.1169 CU` (`855.44`)** | **`-29.13%` / `-58.34%`** | `402 CK-Int8` / `0 Mod` | `27.04 GiB` | **`72.08`** | `72.72` | `94.11` | **`93.63`** | **`80.98`** (`79.15`) | ⭐ **是 / 是** |
| **04** | `[D3-Sage+Mod]` | `SageAttention 1.0.6` + Triton `H3FusedModulation` | `67.96s` (`1.15x` / `1.01x`) | `72.31s` (`1.14x` / `2.06x`) | `0.1447 CU` (`691.02`) | `-12.27%` / `-48.43%` | `402 Sage` / `400 Mod` | `27.46 GiB` | `63.90` | `73.30` | `94.25` | `91.64` | `78.00` (`75.98`) | 否 *(被 R03 支配)* |
| **05** | `[S1-CKChunk-Full]` | Stock 4K-Chunked `ck.sol_attn_chunked` (0–100%, $\tau=1.3$, `tok=0`) | `33.28s` (`2.35x` / `2.06x`) | `37.64s` (`2.19x` / `3.96x`) | `0.0753 CU` (`1327.62`) | `-54.34%` / `-73.16%` | `2 Sage + 400 Chunk` / `0 Mod` | `27.04 GiB` *(27.045)* | `38.56` | `74.24` | `94.66` | `70.20` | `66.25` (`65.63`) | 否 *(被 R07 支配)* |
| **06** | `[S2-CKChunk-W20-Tok256]` | Stock Schema Default `ck.sol_attn_chunked` (W20, $\tau=1.3$, `tok=256`) | `43.64s` (`1.79x` / `1.57x`) | `48.00s` (`1.72x` / `3.11x`) | `0.0961 CU` (`1040.96`) | `-41.76%` / `-65.77%` | `102 Sage + 300 Chunk` / `0 Mod` | `27.18 GiB` | `52.36` | `73.61` | `94.25` | `79.18` | `72.17` (`70.79`) | 否 *(被 R08/R12 支配)* |
| **07** | `[S3-CKDirect-Full]` | **Discovery 1 Direct Zero-Copy `ck.sol_attn`** (0–100%, $\tau=1.3$, `tok=0`) | **`32.76s` (`2.38x` / `2.10x`)** | **`37.11s` (`2.22x` / `4.02x`)** | **`0.0743 CU` (`1346.55`)** | **`-54.98%` / `-73.54%`** | `2 Sage + 400 Direct` / `0 Mod` | `27.04 GiB` | `39.08` | `74.03` | `94.57` | `70.78` | **`66.45`** (`65.91`) | ⭐ **是 / 是 (极速点)** |
| **08** | `[S4-CKDirect-W20-Tok256]` | **Direct Zero-Copy `ck.sol_attn`** (W20, $\tau=1.3$, `tok=256`) | **`43.01s` (`1.81x` / `1.60x`)** | **`47.36s` (`1.74x` / `3.15x`)** | **`0.0948 CU` (`1054.97`)** | **`-42.53%` / `-66.22%`** | `102 Sage + 300 Direct` / `0 Mod` | `27.39 GiB` | `52.31` | `73.83` | `94.30` | `79.55` | **`72.27`** (`70.84`) | ⭐ **是 / 否** *(Q_lin 被 R12 支配)* |
| **09** | `[S5-CKDirect-HiFi-t10]` | **HiFi Direct `ck.sol_attn`** (W20, $\tau=1.0$, `tok=256`, `dense="0,1,48,49"`) | **`46.36s` (`1.68x` / `1.48x`)** | **`50.72s` (`1.63x` / `2.94x`)** | **`0.1015 CU` (`985.22`)** | **`-38.47%` / `-63.83%`** | `126 Sage + 276 Direct` / `0 Mod` | `27.39 GiB` | **`54.38`** | **`74.44`** | `94.41` | **`80.37`** | **`73.30`** (`71.93`) | ⭐ **是 / 是 (高保真稀疏点)** |
| **10** | `[S6-Triton-Int8QK]` | Community `r-vage` Triton Sol (0–100%, $\tau=1.3$, `INT8-QK, BF16-PV`) | `49.82s` (`1.57x` / `1.38x`) | `54.17s` (`1.52x` / `2.75x`) | `0.1084 CU` (`922.44`) | `-34.28%` / `-61.37%` | `2 Sage + 400 Triton` / `0 Mod` | `27.04 GiB` | `37.33` | `73.26` | **`94.81`** | `79.65` | `67.03` (`65.77`) | 否 *(S_fl2v 前沿)* |
| **11** | `[S7-Triton-Sched]` | **Discovery 2 Fixed Scheduled Triton Sol** (W20, $\tau: 1.5\rightarrow 0.8$, `INT8-QK+PV`) | `55.04s` (`1.42x` / `1.25x`) | `59.39s` (`1.39x` / `2.51x`) | `0.1189 CU` (`841.39`) | `-27.95%` / `-57.65%` | `102 Sage + 300 Triton` / `0 Mod` | `27.18 GiB` | `52.50` | `73.86` | `94.74` | `83.90` | `73.11` (`72.02`) | 否 *(被 R03/R09 支配)* |
| **12** | `[S8-FullStack-Pareto]` | **Full-Stack Hybrid**: Direct `ck.sol_attn` (W20, $\tau=1.3$, `tok=256`) + `H3FusedModulation` | **`42.37s` (`1.84x` / `1.62x`)** | **`46.72s` (`1.76x` / `3.19x`)** | **`0.0935 CU` (`1069.50`)** | **`-43.31%` / `-66.68%`** | `102 Sage + 300 Direct` / `400 Mod` | `27.46 GiB` | `52.70` | `73.99` | `94.24` | `78.35` | **`72.25`** (`70.90`) | ⭐ **是 / 是 (推荐生产甜点)** |

> [!IMPORTANT]
> **不同质量目标下的帕累托前沿对比**：
> - **`E2E Latency vs. 主口径 Q_total`（6 个前沿点）**：**`[Run 07, Run 12, Run 08, Run 09, Run 03, Run 01]`**。
> - **`E2E Latency vs. 补充口径 Q_total_linear_supp`（5 个前沿点）**：**`[Run 07, Run 12, Run 09, Run 03, Run 01]`**——在此口径下，`Run 12`（`46.72s`, $Q_{\text{lin}} = 70.9032$）在速度与质量上同时严格支配 `Run 08`（`47.36s`, $Q_{\text{lin}} = 70.8356$），使前沿从 6 点收敛为 5 点。
> - **`E2E Latency vs. 首尾帧锚定 S_fl2v`（3 个前沿点）**：**`[Run 07 (94.57), Run 05 (94.66), Run 10 (94.81)]`**。
> - 若在 `(E2E, S_ref, S_vbench, S_fl2v, S_audio)` 五维向量空间中不做加权标量化，则除 `Run 02`（被 `Run 04` 在时间上严格支配且四维质量完全相等）外，其余 11 个点互不严格支配。本报告中各处「严格支配」均指 **`(端到端耗时/单视频 CU, 标量综合质量 Q_total 或 Q_lin)` 二维目标**下的帕累托支配关系。

### 4.2 十二组四维质量子项明细表（直接提取自 `g4_sol_attn_ablation_metrics.json`）

| Run | 配置简码 | PSNR 均值 / 最低 (dB) | SSIM 均值 / 最低 | LPIPS 均值 / 最高 | 首帧 SSIM / DINO / CLIP | 尾帧 SSIM / DINO / CLIP | 闪烁分 / 光流平滑 / 光流幅度(px) / MUSIQ | STFT-SNR (Log / Lin dB) / Log-Mel Cos / AV-Corr (`Retention`) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **01** | `[D0-SDPA]` | `100.00` / `100.00` | `1.0000` / `1.0000` | `0.0000` / `0.0000` | `0.9255` / `0.9891` / `0.9413` | `0.9031` / `0.9909` / `0.9672` | `0.9651` / `0.4405` / `2.322` / `0.6231` | `100.00` / `100.00` / `1.0000` / `0.2062` (`1.0000`) |
| **02** | `[D1-Sage]` | `20.19` / `13.06` | `0.7711` / `0.5292` | `0.1075` / `0.2889` | `0.9248` / `0.9886` / `0.9443` | `0.9031` / `0.9912` / `0.9642` | `0.9681` / `0.4234` / `2.441` / `0.6252` | `22.96` / `8.32` / `0.9990` / `0.2018` (`0.9956`) |
| **03** | `[D2-CKInt8]` | **`22.75` / `14.38`** | **`0.8257` / `0.6444`** | **`0.0672` / `0.2061`** | `0.9245` / `0.9877` / `0.9340` | `0.9023` / `0.9905` / `0.9651` | `0.9655` / `0.4068` / `2.914` / `0.6236` | **`24.76` / `11.66` / `0.9995`** / `0.1965` (`0.9903`) |
| **04** | `[D3-Sage+Mod]` | `20.19` / `13.06` | `0.7711` / `0.5292` | `0.1075` / `0.2889` | `0.9248` / `0.9886` / `0.9443` | `0.9031` / `0.9912` / `0.9642` | `0.9681` / `0.4234` / `2.441` / `0.6252` | `22.96` / `8.32` / `0.9990` / `0.2018` (`0.9956`) |
| **05** | `[S1-CKChunk-Full]` | `14.90` / `6.14` | `0.6234` / `0.1099` | `0.2630` / `0.8226` | `0.9266` / `0.9891` / `0.9665` | `0.9055` / `0.9937` / `0.9666` | `0.9666` / `0.4594` / `2.406` / `0.6179` | `9.59` / `1.63` / `0.9754` / `0.4063` (`0.7999`) |
| **06** | `[S2-CKChunk-W20-Tok256]` | `16.88` / `11.22` | `0.6881` / `0.4860` | `0.1649` / `0.4262` | `0.9236` / `0.9917` / `0.9491` | `0.9031` / `0.9909` / `0.9611` | `0.9685` / `0.4352` / `2.299` / `0.6268` | `15.32` / `3.15` / `0.9908` / `0.3393` (`0.8669`) |
| **07** | `[S3-CKDirect-Full]` | `14.94` / `6.37` | `0.6281` / `0.1186` | `0.2598` / `0.8145` | `0.9267` / `0.9906` / `0.9552` | `0.9048` / `0.9930` / `0.9696` | `0.9666` / `0.4444` / `2.707` / `0.6251` | `9.72` / `2.33` / `0.9782` / `0.3937` (`0.8125`) |
| **08** | `[S4-CKDirect-W20-Tok256]` | `16.85` / `11.39` | `0.6876` / `0.4751` | `0.1650` / `0.4157` | `0.9237` / `0.9927` / `0.9547` | `0.9029` / `0.9906` / `0.9606` | `0.9684` / `0.4478` / `2.098` / `0.6243` | `15.58` / `3.05` / `0.9908` / `0.3365` (`0.8697`) |
| **09** | `[S5-CKDirect-HiFi-t10]` | **`17.41` / `11.58`** | **`0.7019` / `0.4845`** | **`0.1543` / `0.4026`** | `0.9247` / `0.9916` / `0.9482` | `0.9049` / `0.9908` / `0.9674` | `0.9690` / **`0.4636`** / `1.834` / **`0.6268`** | **`16.70` / `4.69` / `0.9933`** / `0.3601` (`0.8461`) |
| **10** | `[S6-Triton-Int8QK]` | `14.78` / `7.27` | `0.6163` / `0.1367` | `0.2717` / `0.7893` | **`0.9272`** / `0.9891` / `0.9643` | `0.9089` / `0.9924` / `0.9717` | **`0.9715`** / `0.4176` / `2.750` / `0.6055` | `14.00` / `2.83` / `0.9883` / `0.1578` (`0.9516`) |
| **11** | `[S7-Triton-Sched]` | `16.86` / `11.70` | `0.6879` / `0.5056` | `0.1633` / `0.3561` | `0.9270` / `0.9912` / `0.9486` | **`0.9100`** / `0.9927` / `0.9739` | `0.9701` / `0.4466` / `2.078` / `0.6197` | `16.68` / **`7.22`** / `0.9928` / `0.1956` (**`0.9894`**) |
| **12** | `[S8-FullStack-Pareto]` | `16.98` / `11.51` | `0.6918` / `0.4839` | `0.1637` / `0.4187` | `0.9235` / `0.9917` / `0.9473` | `0.9035` / `0.9900` / `0.9615` | `0.9683` / `0.4515` / `2.046` / `0.6257` | `15.26` / `3.12` / `0.9909` / `0.3698` (`0.8364`) |

### 4.3 成本–速度–质量帕累托前沿与逐帧视觉对比

![Blackwell G4 成本–速度–质量帕累托前沿图](outputs/pareto_frontier_cost_speed_quality.png)

![12 组实验五关键帧横向对比长图](outputs/keyframe_strip_12runs.png)

![124 帧逐帧保真度轨迹曲线](outputs/frame_metrics_curves_12runs.png)

---

## 5. 五大核心科学发现深度剖析

### 5.1 为什么 `comfy-kitchen` 原生全 INT8 稠密注意力（`Run 03 [D2-CKInt8]`）在速度与参考保真度上双重超越 `SageAttention 1.0.6`（`Run 02`）？

在实验 02（ADR-02）中，`SageAttention 1.0.6` 是 Blackwell G4 上已测最快的全步稠密注意力后端（实验 02 原测为去噪 `68.01s` / 端到端 `72.34s` / $Q_{\text{lin}} = 75.98$；本次实验 03 同会话复测 `Run 02 [D1-Sage]` 为去噪 `68.69s` / 端到端 `73.03s` / $Q_{\text{total}} = 78.00$ / $Q_{\text{lin}} = 75.98$）。在本次实验中，启用 ComfyUI 内置的 `attention_comfy_kitchen_int8`（`Run 03 [D2-CKInt8]`）实现了显著的**速度与参考保真度双重提升**：
1. **去噪提速 `1.27x`（净省 `14.62s`）、端到端提速 `1.25x`（净省 `14.61s`）**：去噪耗时从 `68.69s` 降至 **`54.06s`**（端到端从 `73.03s` 降至 **`58.41s`**，相对 G4 SDPA 去噪加速 `1.44x` / E2E 加速 `1.41x`；相对实验 02 G4 `Run 2` 原测 E2E 净省 `13.93s`），单次算子微基准从 `118.353 ms`（`428.52 TFLOPS`）降至 **`81.966 ms`（`618.76 TFLOPS`）**。
   - **源码实现路径**：检查 `comfy-kitchen` v0.2.40 源码（`sage_attention.py:605`），当 `MiniMaxH3Model` 的 50 层 `DiTBlock` 传入 `AttentionTensorContainer` 时，`attention_comfy_kitchen_int8` 先调用 `prequantize_int8_attention → sage_sdpa_quantize`（对当前步 $Q/K$ 执行 Hadamard 正交旋转、按需减去代表性 key 向量并完成 INT8 量化，同时提前释放浮点容器内部张量引用以降低显存驻留），随后调用 C++/CUDA `ck.int8_attention_from_prequantized`（400 次 `DiTBlock` 调用）将 **$Q \cdot K^T$ 与 $P \cdot V$ 均在 SM 12.0 INT8 Tensor Core 上执行**；而 2 次 `SingleTokenRefinerBlock` 调用则直接走 `ck.int8_attention`（合计 `402` 次 `ck_int8` 入口命中）。相比之下，`sageattention 1.0.6` 在 SM 12.0 上由 Triton 执行 INT8 $Q \cdot K^T$ + BF16/FP16 $P \cdot V$，单次调用慢 `36.39 ms`。
2. **多步累积参考保真度反超 `+2.56 dB` PSNR（`22.75 dB` vs `20.19 dB`）**：
   - 在单层随机高斯张量微基准中，Sage（INT8-QK + BF16-PV）的单次余弦相似度（`0.999908`）略高于 `CK Pure-INT8`（`0.999855`）。但在 50 层 × 8 步的真实端到端生成中，`Run 03 [D2-CKInt8]` 的参考像素保真度（`PSNR = 22.75 dB` vs `20.19 dB`，`SSIM = 0.8257` vs `0.7711`，`LPIPS = 0.0672` vs `0.1075`，$S_{\text{ref}} = 72.08$ vs `63.90`）与音频保真度（`STFT-SNR = 24.76 dB` vs `22.96 dB`，$S_{\text{audio}} = 93.63$ vs `91.64`）显著高于 `Run 02 [D1-Sage]`（而 `Run 02` 的 $S_{\text{vbench}}$ `73.30 vs 72.72` 与 $S_{\text{fl2v}}$ `94.25 vs 94.11` 略高），使 `Run 03` 的综合得分 $Q_{\text{total}} = 80.98$（$Q_{\text{lin}} = 79.15$）比 `Run 02` 高出 **`+2.98` 分**。
   - **机制分析与证据边界**：`comfy-kitchen` v0.2.40 的 `sage_sdpa_quantize` 在量化前引入了 **$Q/K$ Hadamard 旋转与代表性 key 向量中心化**（而 `comfy/ldm/modules/attention.py` 中的 `attention_sage` 调用 `sageattn` 时未传 `smooth_k` 参数）。这为 `Run 03` 在多步生成中抑制离群通道量化误差提供了合理的源码级假说；不过本次实验未单独拆解 Hadamard 旋转、代表性 key 中心化与 INT8-PV 各自的子消融贡献，因此将「`Run 03` 在本基准下兼具更快速度与更高 $S_{\text{ref}} / Q_{\text{total}}$」作为严格实测结论呈现。

### 5.2 Sol-Attn 在多镜头切换（4-Shot Sequence）中的「0% 全步稀疏 vs. 20% 稠密预热 + 离群 Token 路由」对比

观察 [`outputs/frame_metrics_curves_12runs.png`](./outputs/frame_metrics_curves_12runs.png) 与 [`outputs/keyframe_strip_12runs.png`](./outputs/keyframe_strip_12runs.png)，可以清晰看到两组配置的鲜明差异：
1. **0% 预热全步稀疏组（`Run 05, 07, 10`，`start_percent = 0.0, extra_tokens = 0`）**：
   - 从 Step 0（$\sigma=1.0$）起全程启用 $\tau=1.3$ 的稀疏注意力，原生 `ck.sol_attn` 组的去噪耗时压至 **`32.76s ~ 33.28s`**（`Run 07, 05`；社区 Triton `Run 10` 去噪耗时为 `49.82s`），且由于 `sink_blocks=(0,75)` 保护了前 4,800 个条件与首尾参考帧潜空间 token，**首尾帧锚定依然保持 `S_fl2v = 94.57 ~ 94.81`**，整段视频内生画质同样稳定（`S_vbench = 73.26 ~ 74.24`）。
   - 但从逐帧曲线可见，在 **第 53 帧（镜头 1 $\rightarrow$ 镜头 2 切镜处）与第 76 帧（镜头 3 $\rightarrow$ 镜头 4 切镜处）**，0% 预热组的 PSNR 出现明显下探（最低帧 PSNR `6.14 ~ 7.27 dB`，均值 `14.78 ~ 14.94 dB`），且 [`outputs/keyframe_strip_12runs.png`](./outputs/keyframe_strip_12runs.png) 第 62 帧（`50%` 处微距滚轮特写）的内部电路板反射光偏冷蓝。
2. **20% 稠密预热 + 离群特征保护组（`Run 06, 08, 09, 11, 12`，`start_percent = 0.20`）**：
   - 在前 2 步（100 次 `DiTBlock` + 2 次精炼器 = 102 次 `SageAttention` 调用）执行全稠密注意力，后 6 步（300 次 `DiTBlock` 调用：`Run 06` 为 `300 Chunk`、`Run 08/12` 为 `300 Direct`、`Run 09` 为 `276 Direct + 24 Dense`、`Run 11` 为 `300 Triton`）切换至稀疏注意力并配合 `extra_tokens = 256`（`Run 06/08/09/12`）或动态 $\tau$ 退火（`Run 11`），**第 53 帧与第 76 帧的切镜偏差被显著抑制**：最低帧 PSNR 从 `6.14 ~ 7.27 dB` 跃升至 **`11.22 ~ 11.70 dB`**（其中 `ck.sol_attn` 直通组 `Run 08/09/12` 为 `11.39 ~ 11.58 dB`，分块组 `Run 06` 为 `11.22 dB`，Triton 退火组 `Run 11` 为 `11.70 dB`），平均 PSNR 提升至 **`16.85 ~ 17.41 dB`**，$S_{\text{ref}}$ 从 `37.33 ~ 39.08` 跃升至 **`52.31 ~ 54.38`**，$S_{\text{audio}}$ 提升至 **`78.35 ~ 83.90`**，去噪耗时为 **`42.37s ~ 55.04s`**（其中原生 `ck.sol_attn` 组仅 **`42.37s ~ 46.36s`**）。
   - *(方法学说明：`Run 05→06` 与 `Run 07→08` 同时改变了 `start_percent` (`0.0 → 0.20`) 与 `extra_tokens` (`0 → 256`)，`Run 11` 则结合了 `start_percent = 0.20` 与余弦 $\tau$ 退火（`extra_tokens = 0`）并同样达到 `16.86 dB` PSNR。因此，上述分镜对齐与色温保真度的提升反映了 **20% 稠密预热与离群 token 路由/退火策略的组合收益**。)*

### 5.3 原生 CUDA `comfy-kitchen` Sol-Attn vs. 社区 Triton `ComfyUI-sol-attn` 全面对比

| 对比维度 | 上游 `comfy-kitchen` v0.2.40 (`ck.sol_attn` 直通) | 社区 `r-vage/ComfyUI-sol-attn` v0.6.7 (Triton `sol_attn`) |
| :--- | :--- | :--- |
| **底层内核与量化精度** | 预编译 CUDA C++/PTX 二进制，块均值路由 + **INT8 $Q \cdot K^T$ + INT8 $P \cdot V$** (`qk_int_sv_i8`) | Python + Triton JIT 内核，对角线/块均值路由 + INT8 $Q \cdot K^T$ + 可选 BF16/INT8 $P \cdot V$ |
| **`sink_conditioning="exact_kv_and_rows"` 语义差异** | `sink_blocks=(0,75)` 保护全部前 75 块作为全局 $K/V$ sink；**`sink_q=(67,75)` 仅强制目标音频查询块 `[67,75)` 走精确稠密注意力**，前 67 块（含首尾帧参考潜变量）查询走快速稀疏路由。 | `sink_blocks=(0,75)`；但 **`sink_q=(0,75)` 将整个前 75 块（4,800 个前缀 token）查询全部强制跑稠密注意力**，额外增加了 `14.87 ms/call` 开销。 |
| **单算子微基准耗时 (`N=42,058, \tau=1.3`)** | **`23.95 ms`** (`tok=0`, `2117.8 TFLOPS`) / **`27.31 ms`** (`tok=256`, `1857.2 TFLOPS`) | `62.64 ms` (`sink_q=0..75`, `809.7 TFLOPS`) / `47.77 ms` (对齐 `sink_q=67..75`, `1061.7 TFLOPS`) |
| **0–100% 全步端到端表现 (`Run 07` vs `Run 10`)** | **`Run 07`：去噪 `32.76s` / E2E `37.11s` (`0.0743 CU`)**，$Q_{\text{total}} = 66.45$ | `Run 10`：去噪 `49.82s` / E2E `54.17s` (`0.1084 CU`)，$Q_{\text{total}} = 67.03$（由于 `sink_q=(0,75)` 全前缀稠密查询，`S_audio = 79.65`、`S_fl2v = 94.81` 略高，但慢 `17.06s`） |
| **20% 预热端到端表现 (`Run 09` vs `Run 11`)** | **`Run 09`：去噪 `46.36s` / E2E `50.72s` (`0.1015 CU`)**，$Q_{\text{total}} = \mathbf{73.30}$ (`PSNR = 17.41 dB`) | `Run 11`（修复 Discovery 2 后）：去噪 `55.04s` / E2E `59.39s` (`0.1189 CU`)，$Q_{\text{total}} = 73.11$ (`PSNR = 16.86 dB`, `S_audio = 83.90`) |
| **帕累托结论** | **全面占据 `(E2E, Q_total)` 稀疏段帕累托前沿 (`Run 07, 12, 08, 09`)** | 注意力内核在 SM 12.0 上比 `ck.sol_attn` 慢 `1.75x ~ 2.61x`；但其附带的 **`MiniMaxH3FusedModulation` 位级无损融合算子成功进入全栈帕累托最优解 (`Run 12`)**！ |

---

## 6. 场景化部署选型建议（Actionable Recommendations on Blackwell G4）

结合实验 01、02 与本次实验 03 的完整实证数据，在 **Google Colab Blackwell G4（`7.2046 CU/hr`）** 上部署 **MiniMax `H3-Base-FL2VA`（`1344×768 × 124 帧` 有声视频生成）** 时，推荐按以下五档帕累托最优配置选型（全面废弃实验 02 中破坏首尾帧锚定的 `EasyCache` 以及在 `(E2E, Q_total)` 上被 `CK Pure-INT8` 严格支配的 `SageAttention` 全稠密配置）：

| 推荐档位 | 对应实验组 | 核心技术栈组合 | 去噪 / 端到端耗时 (vs SDPA) | 单条视频 CU (`条 / 100 CU`) | 相对 A100 基线省 CU | 综合质量 $Q_{\text{total}}$ (`PSNR` / `S_fl2v`) | 适用业务场景 |
| :--- | :---: | :--- | :-: | :-: | :-: | :-: | :--- |
| **1. 极限极速/最低成本探索档** | **`Run 07`**<br>`[S3-CKDirect-Full]` | 零拷贝直通 `ck.sol_attn` (0–100% 步, $\tau=1.3$, `tok=0`) | **`32.76s` / `37.11s`**<br>(`2.38x` / `2.22x`) | **`0.0743 CU`**<br>(`1,346.55 条`) | **`-73.54%` 🚀** | `66.45`<br>(`14.94 dB` / **`94.57`**) | 大规模提示词/种子批量筛选与实时分镜预演；比实验 02 `Sage+EasyCache` (`55.36s`) 还快 **`33.0%`**，且完美保持首尾帧锚定 (`S_fl2v = 94.57` vs `EasyCache` `75.13`)。 |
| **2. 全栈均衡生产甜点档（主力推荐 ⭐）** | **`Run 12`**<br>`[S8-FullStack-Pareto]` | 零拷贝直通 `ck.sol_attn` (20% Sage 预热, $\tau=1.3$, `tok=256`) + Triton `MiniMaxH3FusedModulation` | **`42.37s` / `46.72s`**<br>(`1.84x` / `1.76x`) | **`0.0935 CU`**<br>(`1,069.50 条`) | **`-66.68%` 🚀** | **`72.25`**<br>(`16.98 dB` / `94.24`) | **日常主力生产档**：`46.72s` 生成一条 `1344×768`（768p 宽屏）5.17 秒有声视频，显著改善多镜头切换与色温一致性，每 `100 CU` 产出突破 **1,069 条**（是 A100 基线的 **`3.00 倍`**）。*(若追求无第三方节点依赖，可选 `Run 08`，`47.36s`，`0.0948 CU`，$Q_{\text{total}} = 72.27$。)* |
| **3. 高保真稀疏加速档** | **`Run 09`**<br>`[S5-CKDirect-HiFi-t10]` | 零拷贝直通 `ck.sol_attn` (20% Sage 预热, $\tau=1.0$, `tok=256`, `dense_blocks="0,1,48,49"`) | **`46.36s` / `50.72s`**<br>(`1.68x` / `1.63x`) | **`0.1015 CU`**<br>(`985.22 条`) | **`-63.83%` 🚀** | **`73.30`**<br>(`17.41 dB` / `94.41`) | 对动态平滑度（`S_vbench = 74.44` 全场最高）与音画细节要求更高的精细稀疏生成。 |
| **4. 准无损全稠密加速档（替代 ADR-02 Sage ⭐）** | **`Run 03`**<br>`[D2-CKInt8]` | `comfy-kitchen` 原生 CUDA 全 INT8 稠密注意力 (`attention_comfy_kitchen_int8` + 预量化容器) | **`54.06s` / `58.41s`**<br>(`1.44x` / `1.41x`) | **`0.1169 CU`**<br>(`855.44 条`) | **`-58.34%` 🚀** | **`80.98`**<br>(**`22.75 dB`** / `94.11`) | **高保真定稿首选**：100% 全稠密注意力（无块剪枝），比本实验 `Run 02 [D1-Sage]` 去噪快 **`14.62s`（E2E 快 `14.61s`）** 且 PSNR 高出 **`+2.56 dB`**，肉眼与原生 SDPA 几乎无异。 |
| **5. 绝对基准锚点档** | **`Run 01`**<br>`[D0-SDPA]` | 原生 PyTorch 2.11 cuDNN BF16 SDPA | `78.06s` / `82.42s`<br>(`1.00x` / `1.00x`) | `0.1649 CU`<br>(`606.25 条`) | `-41.22%` | **`91.95`**<br>(`100.0 dB` / `94.28`) | 用于评测对标与零近似误差要求的学术基准生成。 |

---

## 7. 实验方法学边界与审计披露（Methodological Caveats）

1. **单案例、单 Seed 与微小质量分差异的解释边界**：
   - 与实验 02 完全一致，本实验基于固定测试场景（透明电竞鼠标 4 镜头序列）、固定随机种子（`seed=42`）单次顺序执行，未包含跨多 Prompt / 多 Seed 或同配置多次重复运行的方差估计。
   - 在稠密组中，`Run 02 [D1-Sage]` 与叠加 `MiniMaxH3FusedModulation` 的 `Run 04 [D3-Sage+Mod]` 的所有质量子项与 124 帧逐帧数组完全一致，且导出的两个 MP4 文件 SHA-256 哈希值 100% 相同，结合单块微基准 `mod_max_abs_diff = 0.0`，严格证明了 `MiniMaxH3FusedModulation` 的位级等价性。
   - 在稀疏组中，`Run 08`（`47.36s`，$Q_{\text{total}} = 72.2748$，$Q_{\text{lin}} = 70.8356$）与叠加 `MiniMaxH3FusedModulation` 的 `Run 12`（`46.72s`，$Q_{\text{total}} = 72.2533$，$Q_{\text{lin}} = 70.9032$）存在微小数值差异（`PSNR 16.85 vs 16.98 dB`）。这可能源于 CUDA 稀疏内核在线路由或规约顺序的微小非确定性，但由于本次实验未对同一稀疏配置进行多次重复运行，我们不将该差值直接定性为固定的运行抖动界限。
2. **`Discovery 1` 在不同长宽比/帧数下的显存与分块权衡边界**：
   - 在本实验负载（$N = 42,058$ tokens，96 GB 显存的 Blackwell G4）下，移除 `h3_sparse_attention` 的 4K 分块生产者改走零拷贝直通 `ck.sol_attn` 仅使峰值分配显存从 `27.18 GiB`（`Run 06`）微增至 `27.39 GiB`（`Run 08`，增加 `0.21 GiB`），远低于 96 GB 显存上限。但若在 16 GB / 24 GB 消费级显卡上生成超长序列视频，4K 分块流式 QKV 投影仍有降低瞬时激活峰值的价值。
3. **`S_audio` 两种音频评分口径（主口径 vs. 补充口径）的完整公式差异对照**：
   - 为同时兼容实验 02 的 A100 主脚本（`run_ablation.py`）与 Blackwell 补充脚本（`run_h100_sage_ablation.py`），[`eval_metrics.py`](./code/eval_metrics.py) 与 [`outputs/g4_sol_attn_ablation_metrics.json`](./outputs/g4_sol_attn_ablation_metrics.json) 同时导出并计算了两种口径（所有子项含 `log_mel_cosine_sim_linear_supp` 均保存在 JSON 的 `audio_metrics` 中），二者的 7 项精确算法差异如下表所示：

| 音频评分维度 | 主口径 `S_audio` / `Q_total`（对齐 `run_ablation.py`） | 补充口径 `S_audio_linear_supp` / `Q_total_linear_supp`（对齐 `run_h100_sage_ablation.py`） |
| :--- | :--- | :--- |
| **1. 多分辨率 STFT 幅度定义** | 对数幅度谱 `log10(abs(Z) + 1e-7)`（`n_fft in [512, 1024, 2048]`） | 线性幅度谱 `abs(Z)`（`nperseg in [512, 1024, 2048]`） |
| **2. STFT 帧移（Hop Length）** | `hop = n_fft / 4`（`noverlap = n_fft - hop`） | SciPy `signal.stft` 默认值 `hop = n_fft / 2` |
| **3. STFT-SNR 多声道与多分辨率聚合顺序** | 对 `3` 个分辨率 × `2` 个立体声声道分别独立计算 `10 * log10(sig_pow / noise_pow)`，再对 `6` 个分贝值取算术平均 | 在每个分辨率下先跨双声道与全时频网格对信号功率和误差功率取均值 `np.mean(...)`，再取 `10 * log10(sig_power / err_power)` 并对 `3` 个分辨率取平均 |
| **4. STFT-SNR 归一化公式** | `snr_norm = clip(stft_snr_db / 30.0, 0, 1)` | `snr_lin_norm = clip((stft_snr_linear_db + 5.0) / 35.0, 0, 1)` |
| **5. Log-Mel 频谱图 FFT 窗口** | `n_fft = 1024, hop = 256, n_mels = 80`，加 `+1e-7` 取 `log10` | `n_fft = 2048, hop = 512, n_mels = 80`，加 `+1e-8` 取 `log10` |
| **6. 双声道 Mel 余弦聚合方式** | 左右声道分别计算 Log-Mel 余弦相似度后再取算术平均 | 先对左右声道功率谱 `abs(Z)**2` 求均值，再计算单次 Log-Mel 余弦相似度 |
| **7. 子项合成权重** | `100 * (0.40 * cos + 0.35 * snr_norm + 0.25 * av_sync_retention)` | `100 * (0.45 * cos_supp + 0.35 * snr_lin_norm + 0.20 * av_sync_retention)` |

4. **四维指标的代理属性与评测输入说明**：
   - 与实验 02 相同，`S_ref` 以同卡 `Run 01 [D0-SDPA]` 为参考锚点；`S_vbench` 为基于 DINOv2-small、CLIP ViT-B/32、RAFT-small 与 `pyiqa` MUSIQ 的代理指标；阶段 5 的四维指标均在 VAE 解码输出的**内存无损 `float32` 帧张量与原始浮点波形**上直接计算，随后交由后台线程编码为 H.264/AAC MP4 归档；[`outputs/comparison_grid_12runs.mp4`](./outputs/comparison_grid_12runs.mp4)（已通过 `ffprobe` 校验 `nb_read_frames = 124`，时长 `5.166667s`）复用 `Run 01` 音轨并带有明文配置标签，仅供直观定性检视。

---

## 8. 外部社区基准对比：与 `soren-labs/minimax-h3-colab` 同卡（Blackwell G4）方案的深度对标与根因拆解

在社区针对 Google Colab **Blackwell G4（`NVIDIA RTX PRO 6000 Blackwell Server Edition`）** 的公开基准仓库 [`soren-labs/minimax-h3-colab`](https://github.com/soren-labs/minimax-h3-colab) 中，作者同样在 G4 实例上测试了 `Comfy-Org/MiniMax-H3`（`minimax_h3_fl2va_pruned_int8_convrot.safetensors`，`res_multistep` 采样器，24 FPS），并评估了三项技术方案：
1. **源码编译 `SageAttention v2.2.0`**（Commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`，针对 `sm_120` 编译 `_qattn_sm80` / `_qattn_sm89`）；
2. **`Spectrum` 频谱特征外推跳步节点**（[`xmarre/ComfyUI-Spectrum-MiniMax-H3`](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) @ `b5fd9db3...`，基于论文 `arXiv:2603.01623`，在真实步捕获第 50 层 `DiTBlock` 输出的 `[target_audio | target_video]` 隐状态，使用切比雪夫多项式基底与在线岭回归预测后续步隐状态，从而跳过预测步的全部 50 层 DiT）；
3. **`TE-Speed-MiniMaxH3-OSS` 尾部层残差缓存节点**（[`HELPMEEADICE/TE-Speed-MiniMaxH3-OSS`](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS) @ `c1dacf47...`，向 `comfy/ldm/minimax/model.py` 注入 `("block_loop", 0)` 钩子，在满足 `0.1 <= pos <= 0.9` 且相邻步 $|\Delta \sigma| < 0.12$ 的缓存步仅重算前 `25%` 即前 12 层 `DiTBlock`，直接复用上一轮 Full 步的后 `75%` 即第 `12..49` 共 38 层残差 $h_{\text{full}} - h_{12}$）。

### 8.1 三项方案覆盖关系与方法学定位对比

| `soren-labs/minimax-h3-colab` 方案 | 加速维度 | 本项目（实验 02 / 实验 03）覆盖情况与技术结论 |
| :--- | :--- | :--- |
| **1. `SageAttention` (`sm_120`)** | 算子级：注意力 INT8/FP8 量化 | ✅ **实验 02 & 03 已完整评测（`Run 02 [D1-Sage]`）**，并在实验 03 中证明：上游 `comfy-kitchen 0.2.40` 原生 CUDA 全 INT8 注意力（`Run 03 [D2-CKInt8]`，`qk_int_sv_i8`）在 G4 上比 `SageAttention` **端到端快 `14.61s`（`58.41s` vs `73.03s`）且参考保真度高出 `+2.56 dB` PSNR（`22.75 dB` vs `20.19 dB`）**。 |
| **2. `Spectrum` (`ComfyUI-Spectrum-MiniMax-H3`)** | 步级（Step-Level）：切比雪夫多项式跨步特征预测（20 步中 `14 实 + 6 测`） | ⚠️ **实验 02 已评测同属「跨步特征复用/预测」维度的官方 `EasyCache`，实验 03 未单测 `Spectrum`**：<br>• `Spectrum` 面向 **20 步未蒸馏调度**（配置 `warmup_steps=5, tail_actual_steps=1`）。在我们的 **8 步 Turbo LoRA** 调度下，扣除前 5 步预热与尾部锚定步后最多仅能预测 **1 步**，却需额外承担隐状态历史归档开销（该仓库 `first-5s-comfyui.log` 实测 `history_mib = 3093.4 MiB`、`history_archive_s = 14.25s ~ 14.45s`）。 |
| **3. `TE-Speed-MiniMaxH3-OSS`** | 层级（Block-Loop）：跨步尾部 75% `DiTBlock` 残差缓存（20 步中 `8 实 + 12 缓`，跳过 `456/1000` 层） | ⚠️ **实验 02/03 未测该第三方补丁（与实验 02 `EasyCache` 同属跨步残差缓近似）**：<br>• `TE-Speed` 依赖 20 步密集采样下相邻步 $|\Delta \sigma| < 0.12$ 时尾部 38 层残差近似恒定的假设；且该仓库实证指出 **`TE-Speed` 与 `Spectrum` 互斥不可叠加**（叠加会抛出 `RuntimeError: native MiniMax H3 final transformer block was not executed`）。<br>• 在 **8 步 Turbo LoRA + $\alpha=12.0$ 流匹配偏移**下，中后段单步 $\Delta \sigma$ 达 `0.078 ~ 0.632`，且实验 02 已证实跨步残差复用在 8 步首尾帧受控任务（FL2VA）上会导致首尾帧像素锚定崩塌（SSIM 跌至 `0.557 / 0.590`）。 |

### 8.2 同卡（Colab Blackwell G4）实测速度与 CU 成本正面对标

将 `soren-labs/minimax-h3-colab` 仓库 `benchmarks/metadata/summary.json` 及 `benchmarks/logs/first-5s-comfyui.log`、`te-speed-native-ab-comfyui.log` 中的 G4 实测数据，与本报告实验 03 在同一型号 G4 显卡上的实测数据直接对比（统一按 G4 实测等效费率 `7.2046 CU/hr` 折算单视频成本）：

| 方案来源与实验组 | 分辨率 × 帧数 (时长) | 采样步数与任务类型 | 单步全量 DiT 耗时 | 端到端总耗时 $T_{\text{e2e}}$ | 单条视频 CU (`条 / 100 CU`) | 相对 `soren-labs` Spectrum I2V (`234.20s`) 加速比 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| `soren-labs`：**Native T2V Baseline** (`cu128`, SDPA) | `1344×768 × 124f` (`5.17s`) | 20 步 (无 LoRA, T2V) | `17.22s / step` | `355.83s` | `0.7121 CU` (`140.4 条`) | `0.66x` |
| `soren-labs`：**Spectrum + Sage v2.2.0 (T2V)** | `1344×768 × 124f` (`5.17s`) | 20 步 (`14 实 + 6 测`, T2V) | `14.66s / step` | `226.09s` | `0.4525 CU` (`221.0 条`) | `1.04x` |
| `soren-labs`：**Spectrum + Sage v2.2.0 (I2V)** | `1344×768 × 124f` (`5.17s`) | 20 步 (`14 实 + 6 测`, 首帧 I2V) | `15.56s / step` | `234.20s` | `0.4687 CU` (`213.4 条`) | **`1.00x` (外部基准锚点)** |
| `soren-labs`：**Native + Sage v2.2.0 (8s T2V)** | `1344×768 × 192f` (`8.00s`) | 20 步 (无 LoRA, T2V) | `24.82s / step` | `514.15s` | `1.0289 CU` (`97.2 条`) | — |
| `soren-labs`：**TE-Speed + Sage v2.2.0 (8s T2V)** | `1344×768 × 192f` (`8.00s`) | 20 步 (`8 实 + 12 缓`, T2V) | `24.84s` (实) / `6.05s` (缓) | `288.09s` | `0.5766 CU` (`173.4 条`) | — |
| **本报告 Exp 03：`Run 01 [D0-SDPA]`（无加速基线）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`9.76s / step`** | **`82.42s`** | **`0.1649 CU` (`606.2 条`)** | **快 `2.84x`（省 `64.8%` CU）** |
| **本报告 Exp 03：`Run 02 [D1-Sage]`（ADR-02 基线）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`8.59s / step`** | **`73.03s`** | **`0.1461 CU` (`684.2 条`)** | **快 `3.21x`（省 `68.8%` CU）** |
| **本报告 Exp 03：`Run 03 [D2-CKInt8]`（高保真全稠密 ⭐）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`6.76s / step`** | **`58.41s`** | **`0.1169 CU` (`855.4 条`)** | **快 `4.01x`（省 `75.1%` CU）🚀** |
| **本报告 Exp 03：`Run 09 [S5-CKDirect-HiFi-t10]`（高保真稀疏 ⭐）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`5.80s / step` (均值)** | **`50.72s`** | **`0.1015 CU` (`985.2 条`)** | **快 `4.62x`（省 `78.3%` CU）🚀** |
| **本报告 Exp 03：`Run 12 [S8-FullStack-Pareto]`（主力生产甜点 ⭐）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`5.30s / step` (均值)** | **`46.72s`** | **`0.0935 CU` (`1,069.5 条`)** | **快 `5.01x`（省 `80.1%` CU）🚀** |
| **本报告 Exp 03：`Run 07 [S3-CKDirect-Full]`（极速探索档 ⭐）** | **`1344×768 × 124f` (`5.17s`)** | **8 步 Turbo LoRA (FL2VA 首尾帧)** | **`4.10s / step`** | **`37.11s`** | **`0.0743 CU` (`1,346.6 条`)** | **快 `6.31x`（省 `84.2%` CU）🚀** |

### 8.3 为什么 `soren-labs/minimax-h3-colab` 在同款 G4 显卡上慢 4.0 ~ 6.3 倍？—— 两项底层根因审计

通过对 `soren-labs/minimax-h3-colab` 的源码（`h3_colab/remote_job.py`）与远端 ComfyUI 原始日志（`benchmarks/logs/first-5s-comfyui.log`、`te-speed-native-ab-comfyui.log`）进行逐行审计，我们查明了导致其性能落后的两项核心工程根因：

1. **根因一：`PyTorch 2.11.0+cu128` + 旧版 `comfy-kitchen 0.2.26` 导致 CUDA 与 Triton 量化算子后端全部被禁用（退化为 PyTorch `eager` 反量化）！**
   - 在 `soren-labs/minimax-h3-colab/benchmarks/logs/first-5s-comfyui.log` 第 7–11 行及 `te-speed-native-ab-comfyui.log` 第 7–11 行中，ComfyUI 启动时明确打印了如下告警：
     ```text
     [WARNING] WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.
     [INFO] Found comfy_kitchen backend eager: {'available': True, 'disabled': False, ...}
     [INFO] Found comfy_kitchen backend triton: {'available': True, 'disabled': True, ...}
     [INFO] Found comfy_kitchen backend cuda: {'available': True, 'disabled': True, ...}
     [INFO] pytorch version: 2.11.0+cu128 ... comfy-kitchen version: 0.2.26
     ```
   - 由于该仓库运行在 `cu128` 构建的 PyTorch 上，**`comfy-kitchen` 的 `cuda` 与 `triton` 后端双双处于 `disabled: True` 状态**！这意味着 H3 50 层 `DiTBlock` 中所有的 `int8_convrot` 线性投影（`qkv_proj`、`proj`、FFN）以及融合 `rms_rope_split_half_` 全部回退到了慢速的 `eager` 后端（每步在 PyTorch 中将 INT8 权重反量化回浮点再做矩阵乘），同时也导致无法调用 `comfy-kitchen` 原生 CUDA `ck.int8_attention` 与 `ck.sol_attn`。
   - **单步耗时铁证**：在 `1344×768 × 124 帧` 负载下，`soren-labs` 的单步真实 DiT 耗时高达 **`17.22s/step`（SDPA）** 与 **`14.66s ~ 15.56s/step`（Sage v2.2.0）**；而在本实验（`PyTorch 2.11.0+cu130` + `comfy-kitchen 0.2.40` 原生 SM 12.0 CUDA 后端全开）中，即便包含首尾双参考帧条件（$N=42,058$，比 `soren-labs` T2V 的 $N=37,849$ 和单首帧 I2V 的 $N=39,858$ 序列更长），我们的单步耗时也仅为 **`9.76s/step`（SDPA）**、**`8.59s/step`（Sage）**、**`6.76s/step`（CK Pure-INT8 稠密）** 与 **`4.10s/step`（CK Sol-Attn 稀疏）**——仅激活正确的 `cu130` + `comfy-kitchen` 原生 CUDA 后端与算子，单步速度即提升 **`1.8x ~ 3.8x`**！
2. **根因二：未加载 [ADR-01](../../docs/adr/0001-model-weights-loading-pipeline.md) 黄金五件套中的 8 步蒸馏 Turbo LoRA（`minimax_h3_fl2va_2505_ntk32_8steps_lora_fp16.safetensors`），被迫执行 20 步采样**：
   - 该仓库 `benchmarks/metadata/models.json` 与 `docs/deployment.md:39` 明确记载只下载了 4 个权重文件（刻意排除了官方 8 步 Turbo LoRA），因此所有实验均按 **20 步（`STEPS = 20`）** 执行。
   - 即便启用 `Spectrum`（在 20 步中预测 6 步、实际执行 **14 步**全量 DiT，外加 `14.25s ~ 14.45s` 的 `3,093.4 MiB` 隐状态历史归档开销）或 `TE-Speed`（在 20 步中执行 8 步全量 50 层 + 12 步前 12 层，跳过 `456 / 1000` 个 Block，折算执行 **`10.88` 步**全量 DiT = **`544 / 1000` 个 Block**），其实际执行的 Transformer 层数（`544 ~ 700` 个 `DiTBlock`）依然显著高于我们的 **8 步 Turbo LoRA 全步方案（`400` 个 `DiTBlock`）**。
   - 更关键的是，我们在实验 03 中采用的 **零拷贝直通 `ck.sol_attn`（`Run 12 / 09 / 07`）与全 INT8 稠密注意力（`Run 03`）不跳过任何一个去噪步、也不丢弃任何尾部 Transformer 层**，在将 `1344×768 × 124 帧` 端到端生成时间压缩至 **`37.11s ~ 58.41s`（比 `soren-labs` 的 `Spectrum` 快 `4.01x ~ 6.31x`）** 的同时，彻底规避了跳步/跳层缓存对首尾帧受控对齐（`S_fl2v > 94.1`）与多镜头切换的破坏。
