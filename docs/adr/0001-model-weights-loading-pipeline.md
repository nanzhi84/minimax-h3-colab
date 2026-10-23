<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# ADR-01：锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）为标准量化基座，并采用链路 A（HF 直连 `/content` $\rightarrow$ RAM $\rightarrow$ VRAM）作为标准加载方案

- **状态（Status）**：已采纳（Accepted）
- **日期（Date）**：2026-10-22
- **关联实验**：[`experiments/01-hf-vs-gdrive-speed`](../../experiments/01-hf-vs-gdrive-speed/README.md)（原始实测日志：[`a100_benchmark_raw.log`](../../experiments/01-hf-vs-gdrive-speed/outputs/a100_benchmark_raw.log)）

---

## 1. 背景与问题（Context）

在 Google Colab 无状态 GPU 虚拟机上运行 **MiniMax `H3-Base-FL2VA`** 视频生成模型时，首先要解决两个核心工程问题：

1. **权重组合选型问题**：
   - 官方源仓库 `MiniMaxAI/MiniMax-H3` 混装了多套模型及根目录转换副本（整仓约 `498 GB`），其中仅 `FL2VA/` 单目录就达约 `144 GB`，且包含约 `13B` 仅微调训练使用的 AdaLN 冗余参数，直接拉取极易撑爆 Colab 本地磁盘或导致冗长的 I/O 等待。
   - 社区维护的推理仓库 [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) 提供了剔除微调冗余分支并量化后的 **官方默认推理黄金组合（共 5 个 `.safetensors` 文件，合计 `42,030,035,159` 字节 = `42.03 GB` / `39.14 GiB`）**。
2. **云端权重加载链路选择问题**：
   针对这 `42.03 GB`（`39.14 GiB`）的 5 个核心权重文件，存在三条候选加载链路：
   - **链路 A（HF 直连本地盘流）**：每次启动 Colab 会话后，直接从 Hugging Face 高速下载至虚拟机本地磁盘 `/content` $\rightarrow$ `safetensors.torch.load_file(device="cpu")` $\rightarrow$ `.to("cuda")` 搬运至 GPU 显存。
   - **链路 B1（Google Drive 拷贝流）**：预先将权重持久化在 Google Drive，每次启动后从 `/content/drive` 拷贝至 `/content` 本地盘 $\rightarrow$ 加载至系统内存 RAM $\rightarrow$ 搬运至 GPU 显存。
   - **链路 B2（Google Drive 原地零拷贝流）**：预先将权重存入 Google Drive，不拷贝至 `/content`，直接对 `/content/drive` 挂载路径调用 `load_file(device="cpu")` $\rightarrow$ 搬运至 GPU 显存。

---

## 2. 实验证据（Evidence）

在 **Colab `A100-SXM4-80GB High-RAM`**（本地磁盘 `235.7 GiB`、系统内存 `167.1 GiB`、可见显存 `79.3 GiB`）上，对上述 `42.03 GB`（`39.14 GiB`）五件套进行了逐文件三阶段（① 落盘 $\rightarrow$ ② Disk 入 RAM $\rightarrow$ ③ RAM 入 VRAM 同步完成）实测对比：

| 链路方案（执行顺序：`A` $\rightarrow$ `B1` $\rightarrow$ `B2`） | ① 落盘耗时 (平均吞吐率) | ② Disk $\rightarrow$ RAM 耗时 | ③ RAM $\rightarrow$ VRAM 耗时 (等效吞吐率) | **三阶段累计耗时** |
| :--- | :---: | :---: | :---: | :---: |
| **链路 A：Hugging Face (Xet) $\rightarrow$ `/content` $\rightarrow$ RAM $\rightarrow$ VRAM 🏆** | **`127.8 s`** (`314 MiB/s` / `328.9 MB/s`) | **`0.2 s`** | **`11.7 s`** (`3.35 GiB/s` / `3.59 GB/s`) | **`139.7 s` (`2.33 min`)** |
| **链路 B2：Google Drive 原地零拷贝 $\rightarrow$ RAM $\rightarrow$ VRAM** | `0.0 s` (零拷贝) | `206.0 s`* | `109.1 s`* | **`315.0 s` (`5.25 min`)*** |
| **链路 B1：Google Drive $\rightarrow$ 拷贝至 `/content` $\rightarrow$ RAM $\rightarrow$ VRAM** | `481.3 s` (`83 MiB/s` / `87.3 MB/s`)* | `0.1 s` | `11.6 s` (`3.37 GiB/s` / `3.62 GB/s`) | **`493.0 s` (`8.22 min`)*** |

### 关键实测发现：
1. **Hugging Face 直连远快于 Google Drive FUSE 挂载读取**：
   - 在未在本会话预热的第 1 个大文件（`20.97 GB` / `19.53 GiB` 主干模型）上，**链路 A** 从 Hugging Face 下载仅耗时 **`52.8 s`（`379 MiB/s`）**；而 **链路 B1** 从 `/content/drive` 本会话首次拷贝耗时高达 **`432.2 s`（仅 `46 MiB/s`，慢 8.19 倍）**。
   - 即使链路 B1/B2 的第 2~5 个文件因同会话刚写入而受益于客户端 `DriveFS` 本地缓存，链路 A 全套累计耗时（`139.7 s`）依然比链路 B2（`315.0 s`）快 **2.25 倍**、比链路 B1（`493.0 s`）快 **3.53 倍**。
2. **Google Drive 原地零拷贝（链路 B2）在阶段 ③ 存在严重隐性阻塞**：
   - `safetensors.torch.load_file(..., device="cpu")` 采用 `mmap` 内存映射，在本地 `/content` 盘上阶段 ② 仅需 `0.1~0.2 s`，阶段 ③ `.to("cuda")` 搬运全套 `39.14 GiB` 仅需 **`11.7 s`**（等效 **`3.35 GiB/s` / `3.59 GB/s`**）。
   - 若直接从 `/content/drive` 零拷贝加载（链路 B2），第 1 个 `20.97 GB` 文件不仅阶段 ② 阻塞 `203.6 s`，**阶段 ③ `.to("cuda")` 触发映射页读取时在 FUSE 路径上又额外阻塞了 `95.8 s`（比本地盘的 `5.6 s` 慢 17.1 倍）**。

---

## 3. 决策内容（Decision）

后续所有实验与生产脚本统一锁定 **`Comfy-Org/MiniMax-H3` 黄金推理五件套（不再对模型量化位宽做重复消融）**，并采纳 **链路 A（Hugging Face 直连下载至 `/content` 本地 SSD $\rightarrow$ RAM $\rightarrow$ VRAM）** 作为标准加载方案：

1. **权重基座与量化方案锁定（黄金五件套，合计 `42.03 GB` / `39.14 GiB`）**：
   默认唯一基座拉取 `Comfy-Org/MiniMax-H3` 的 5 个官方打包权重文件，后续加速消融实验均基于此五件套在 **ComfyUI 原生推理管线（`comfy-kitchen`）** 上展开：
   - **FL2VA 生成主干（剪枝 + ConvRot INT8）**：`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（`20.97 GB` / `19.53 GiB`，剔除 13B 微调冗余 AdaLN 分支，采用 Group-256 正交旋转 INT8 量化，兼顾全系 GPU Tensor Core 硬件加速与近 BF16 画质，无需再测低效的 `fp8_scaled` 或臃肿的未剪枝 `bf16`）。
   - **多模态文本/图像编码器（NVFP4-AWQ）**：`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（`15.69 GB` / `14.61 GiB`，基于 AWQ 4-bit 压缩，不绑定 Blackwell 架构，在 A100/H100/L4/G4 上通用）。
   - **视频解码器（ConvRot INT8）**：`vae/minimax_h3_video_vae_int8_convrot.safetensors`（`2.81 GB` / `2.62 GiB`，比 FP16 原版省一半显存且在 ComfyUI 中解码提速约 2 倍）。
   - **8步蒸馏加速 LoRA（BF16）**：`loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`（`1.96 GB` / `1.82 GiB`）。
   - **音频解码器（FP32）**：`vae/minimax_h3_audio_vae_fp32.safetensors`（`0.61 GB` / `0.56 GiB`）。
2. **标准加载路径**：
   - 每次 Colab 会话启动后，直接将权重下载至虚拟机本地磁盘 `/content/models/...`（或 ComfyUI 对应的本地 `/content/ComfyUI/models/...` 目录）。
   - 在导入 `huggingface_hub` **之前**，显式配置环境变量开启 Xet 高性能模式并禁用隐式 Token 轮询：
     ```python
     import os
     os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
     os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
     os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
     ```
3. **Google Drive 的职责收敛**：
   - **不再使用 Google Drive 传输或原地加载几十 GB 的基础公开模型权重**（既避免挂载 FUSE 慢速 I/O 浪费 GPU 计费时长，也避免在无人值守 CLI 脚本中强依赖交互式 `drivemount` OAuth 授权）。
   - Google Drive（或 `colab push/pull`）仅用于持久化**体积小的私有输入素材**（如首尾帧测试图片、Prompt 配置）、**实验输出的 MP4 视频与评测日志**，以及在必要时作为离线灾备。

---

## 4. 影响与后果（Consequences）

### 正向收益（Pros）
- **大幅缩短 GPU 计费等待时间**：全套 `42.03 GB` 模型从拉取到可进入显存仅需约 **2.3 分钟**，相比从 Google Drive 冷读拷贝节省 **6~12 分钟/次** 的 GPU 算力单元（CU）消耗。
- **本地 SSD 满速入显存**：权重落盘 `/content` 后，`mmap` + `.to("cuda")` 阶段等效吞吐率达 **`3.35 GiB/s`（`3.59 GB/s`）**，即使在 24GB/40GB 显存机型上启用多阶段 CPU Offload 也能保证极速切换。
- **全自动化无阻塞**：通过 `colab-cli` 执行纯 Python 脚本时无需每次等待人工点击 Google Drive 挂载授权链接，真正实现一键拉起、跑完自动 `colab stop` 释放算力。

### 约束与注意事项（Cons & Mitigations）
- **每次新建 VM 需重新拉取 42.03 GB**：由于 Colab 本地盘 `/content` 随会话销毁而清空，每次新起 VM 需约 2 分钟下载。因此**在同一会话内应批量跑完多组测试样例后再执行 `colab stop`**。
- **对 Hugging Face 可用性的依赖**：若极端情况下 Hugging Face 仓库发生限流或故障，可传入 `HF_TOKEN` 提升配额，或退回已保存在 `/content/drive/MyDrive/models/Comfy-MiniMax-H3-FL2VA` 的冷备副本。
