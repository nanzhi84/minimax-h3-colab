<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# Experiment 04: Colab-to-Local HTTP Egress (`colab download`) vs. Colab-to-Google-Drive Egress Relay (`GDrive REST Relay`)

本目录包含 **实验 04（尾端产物回传链路消融实验：HTTP 流式直拉 vs. Google Drive 云端内网中转卸载）** 的完整基准测试脚本、可复用两段式回传中转工具、原始指标 JSON、账本审计摘录及可视化分析图表。

## 1. 核心结论

| 归档包档位 | 原始体积 (`raw_bytes`) | `/api/contents` 膨胀后流量 (`wire_bytes`) | 方案 A: `colab download` HTTP 直拉 (GPU 计费时长 / CU) | **方案 B2 (推荐): GDrive REST 中转阶段 1 (GPU 唯一计费时长 / CU)** | **方案 B2 阶段 2: 停机后本地 8 线程拉取 (0 CU)** | **方案 B2 端到端总耗时** | **GPU 计费时长加速比 & CU 节省率** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1_Single_Run`** *(单视频 `Run 12`)* | `10.63 MB` | `14.55 MB` (`+36.85%`) | `3.76s` (`0.00752 CU`) | **`1.04s`** (**`0.00209 CU`**) | `2.84s` (`0 CU`) | **`3.88s`** | **`3.61x` / 省 `72.3%` CU** |
| **`T2_4Run_Batch`** *(4 视频小批量)* | `41.09 MB` | `56.23 MB` (`+36.84%`) | `7.88s` (`0.01577 CU`) | **`1.29s`** (**`0.00259 CU`**) | `5.02s` (`0 CU`) | **`6.32s`** | **`6.09x` / 省 `83.6%` CU** |
| **`T3_12Run_Exp03`** *(实验 03 完整 12 视频包)* | `122.42 MB` | `167.52 MB` (`+36.84%`) | 实测 `17.69s` (`0.03540 CU`)<br>实验03 `26.60s` (`0.05323 CU`) | **`2.27s`** *(Media)* / **`2.16s`** *(Resumable)*<br>(**`0.00453 CU`**) | `10.94s` (`0 CU`) | **`13.21s`** | **`7.81x~11.74x` / 省 `87.2%~91.5%` CU** |
| **`T4_36Run_Sweep`** *(36 视频大批量扫描)* | `367.26 MB` | `502.56 MB` (`+36.84%`) | `48.05s`* (`0.09615 CU`) | **`4.60s`** (**`0.00920 CU`**) | `44.52s` (`0 CU`) | **`49.12s`** | **`10.45x` / 省 `90.4%` CU** |

## 2. 目录结构

- [`report.md`](report.md)：完整中文技术报告（协议 `+36.84%` Base64 膨胀解剖、四档位实测数据、本地网速敏感度解析模型、盈亏平衡点定理及 CU 账本审计）。
- [`gdrive_egress_relay.py`](code/gdrive_egress_relay.py)：**生产级可复用两段式回传中转工具**（阶段 1 云端内网直传 Google Drive v3 API $\rightarrow$ 立即 `colab stop` 停机 $\rightarrow$ 阶段 2 本地 8 线程 HTTP `Range` 零 CU 并发拉取 + `SHA-256` 校验 + 自动清理云端临时文件）。
- [`run_egress_benchmark.py`](code/run_egress_benchmark.py)：端到端自动化实验编排与遥测采集脚本。
- [`render_visuals_local.py`](code/render_visuals_local.py)：本地零 CU 可视化图表渲染脚本。
- [`outputs/`](outputs/)：
  - [`exp04_egress_transfer_metrics.json`](outputs/exp04_egress_transfer_metrics.json)：完整实测指标与本地网速敏感度矩阵 JSON。
  - [`colab_session_ledger_excerpt.json`](outputs/colab_session_ledger_excerpt.json)：`h3-exp04-g4` 会话 `/tun/m/ccu-info` 与 `assign`/`unassign` 原始 HTTP 时间戳审计摘录。
  - [`remote_probe_results.json`](outputs/remote_probe_results.json)：远端 G4 节点硬件信息与 Google Drive 骨干网上传原始采样记录。
  - [`exp04_benchmark_raw.log`](outputs/exp04_benchmark_raw.log)：完整终端执行日志。
  - [`egress_gpu_time_and_cu_comparison.png`](outputs/egress_gpu_time_and_cu_comparison.png)：四档位 GPU 计费耗时、CU 消耗与端到端总耗时拆解对比图。
  - [`bandwidth_sensitivity_and_protocol_bloat.png`](outputs/bandwidth_sensitivity_and_protocol_bloat.png)：本地网速敏感度曲线与机房内网吞吐率扩展曲线图。
