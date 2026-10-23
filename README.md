<p align="center">
  <!-- <img src="docs/assets/logo.svg" alt="minimax-h3-colab Logo" width="96"> -->
</p>

<h1 align="center">minimax-h3-colab</h1>

<p align="center"><strong>面向 Google Colab Blackwell G4 的 MiniMax <code>H3-Base-FL2VA</code> 极速推理与零浪费异构批处理生产工作流</strong></p>

<p align="center">
  <code>minimax-h3-colab</code> 基于实测算子消融与会话级算力单元（CU）账本审计，将 <code>Comfy-Org/MiniMax-H3</code> 声画同源视频生成模型在 Google Colab Blackwell 96GB（<code>--gpu G4</code>）上的单条 <code>1344×768 × 124 帧</code> 推理耗时压降至 <strong>45.9s ~ 46.7s（边际成本 0.092 CU）</strong>，并通过“本地 0-CU 策划预投料 + 云端阶段式条件去重预编码与常驻 <code>Run 12</code> 渲染 + 2 秒机房内网 Google Drive REST 卸载停机”三阶段异构架构，消除冷启动与跨境慢网对 GPU 算力的双向阻塞。
</p>

## Core ideas

- **黄金五件套直连加载（[ADR-0001](docs/adr/0001-model-weights-loading-pipeline.md)）**：锁定 `Comfy-Org/MiniMax-H3` 量化推理五件套（`int8_convrot` + `nvfp4_awq`，合计 `42.03 GB`），采用 Hugging Face Xet 8 线程直连 `/content` 本地盘加载（`328.9 MB/s`），规避 Google Drive FUSE 挂载冷读与 `mmap` 缺页阻塞。
- **Blackwell G4 全栈混合稀疏加速（[ADR-0002](docs/adr/0002-blackwell-gpu-and-sageattention-acceleration.md) · [ADR-0003](docs/adr/0003-sol-attn-and-g4-pareto-frontier.md)）**：锁定 `Run 12 [S8-FullStack-Pareto]`（前 2/8 步 `SageAttention` 稠密预热 + 后 6/8 步零拷贝直通 `ck.sol_attn` + 位级无损 `MiniMaxH3FusedModulation`），在不跳步、保持首尾帧高像素锚定精度（`S_fl2v > 94`）的前提下实现相对 A100 基线 `3.19x` 端到端加速。
- **两段式零交互 Google Drive REST 内网中转（[ADR-0004](docs/adr/0004-two-stage-gdrive-rest-egress-relay.md)）**：远端通过机房骨干网（`54 ~ 80 MB/s`）在 `1.5s ~ 2.3s` 内将未压缩 `.tar` 归档包直传 Google Drive v3 API 并立即触发 `colab stop` 停机，随后在本地以 `0.00 CU` 执行 8 线程 `HTTP Range` 并发拉取，彻底消除 Jupyter `/api/contents` 的 `+36.84%` Base64 膨胀税与弱网 GPU 陪跑。
- **三阶段异构批处理生产管线（[ADR-0005](docs/adr/0005-staged-batch-production-workflow.md)）**：在 [`studio/api/`](studio/api/) 中实现本地 0-CU 静态校验与预投料、云端 `Qwen3-VL-32B` SHA-256 条件去重预编码后立即卸载（释放 `17.21 GiB` 显存）、常驻 `DiT` 单流连续渲染（模式 P1）与后台 CPU 异步 `ffmpeg` 封装，将多任务批次的单视频综合算力成本降低 `62% ~ 75%`。

## Get started

```bash
# 1. 运行仓库结构、架构图约定与文档死链巡检
yoryon-harness doctor

# 2. 本地 0-CU 干跑校验分镜清单、查看 SHA-256 条件去重率与 CU 成本预估
python3 studio/api/colab_batch_runner.py \
  --manifest studio/api/sample_single_manifest.json \
  --dry-run

# 3. 全自动执行 Colab Blackwell G4 批处理渲染与 0-CU 归档拉取
python3 studio/api/colab_batch_runner.py \
  --manifest studio/api/sample_30s_shortfilm_manifest.json \
  --session h3-studio-g4
```

## Architecture Decision Records (ADR)

| 编号 | 决策标题 | 状态 | 日期 | 关联实验 |
| :---: | :--- | :---: | :---: | :--- |
| [ADR-0001](docs/adr/0001-model-weights-loading-pipeline.md) | 锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`int8_convrot` + `nvfp4_awq`，`42.03 GB`）为标准量化基座，并采用链路 A（HF 直连 `/content` $\rightarrow$ RAM $\rightarrow$ VRAM）加载 | ✅ 已采纳 | 2026-10-22 | [`01-hf-vs-gdrive-speed`](experiments/01-hf-vs-gdrive-speed/README.md) |
| [ADR-0002](docs/adr/0002-blackwell-gpu-and-sageattention-acceleration.md) | 锁定 Google Colab Blackwell 96GB（`--gpu G4`）为标准推理机型，加速策略锁定为单开 `SageAttention`（`[S=1, C=0, T=0]`），禁用 `EasyCache` 与 `torch.compile` | 🔄 加速策略由 ADR-0003 升级 | 2026-10-23 | [`02-a100-acceleration-ablation`](experiments/02-a100-acceleration-ablation/report.md) |
| [ADR-0003](docs/adr/0003-sol-attn-and-g4-pareto-frontier.md) | 在 Blackwell G4 上正式锁定 `Run 12 [S8-FullStack-Pareto]`（零拷贝直通 `ck.sol_attn` + 前 2/8 步 `SageAttention` 预热 + `MiniMaxH3FusedModulation`，`46.72s` / `0.0935 CU`）为默认标准推理配置，并确立零浪费 GPU 生命周期工程规范 | ✅ 已采纳 | 2026-10-23 | [`03-g4-sol-attn-pareto-ablation`](experiments/03-g4-sol-attn-pareto-ablation/report.md) |
| [ADR-0004](docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) | 锁定两段式零交互 Google Drive v3 REST API 云端内网中转卸载（`Colab G4 -> Google Drive REST API` $\rightarrow$ 立即 `colab stop` $\rightarrow$ 停机后 `0 CU` 8 线程 `HTTP Range` 本地拉取）为唯一默认标准尾端回传方案 | ✅ 已采纳 | 2026-10-23 | [`04-gdrive-vs-http-egress-transfer`](experiments/04-gdrive-vs-http-egress-transfer/report.md) |
| [ADR-0005](docs/adr/0005-staged-batch-production-workflow.md) | 锁定“本地 0-CU 策划攒批预投料 + 云端阶段式 `Qwen3-VL` SHA-256 去重预编码并立即卸载 + 常驻 `Run 12` 模式 P1 单流渲染与后台 CPU 异步 `ffmpeg` + 2 秒内网卸载停机”为标准批处理视频生产工作流 | ✅ 已采纳 | 2026-10-23 | [`05-production-batch-workflow`](experiments/05-production-batch-workflow/report.md) |

## Learn more

- **Agent 与项目特异规则**：[AGENTS.md](AGENTS.md) · [PROJECTS.md](PROJECTS.md)
- **文档导航与全局架构**：[文档维护规范](docs/README.md) · [全局架构现状](docs/architecture/overview.md)
- **参考资料与版本记录**：[参考资料索引](docs/reference/INDEX.md) · [CHANGELOG.md](CHANGELOG.md)
