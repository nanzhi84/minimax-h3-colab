<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# ADR-04：锁定两段式零交互 Google Drive v3 REST API 云端内网中转卸载（`Colab G4 -> Google Drive REST API` $\rightarrow$ 立即 `colab stop` $\rightarrow$ 停机后 `0 CU` 8 线程 `HTTP Range` 本地拉取）为唯一默认标准尾端回传方案

- **状态（Status）**：已采纳（Accepted，正式锁定为后续所有实验与生产任务的**唯一默认标准尾端回传形式**，全面替代 `colab download` 的 Jupyter `/api/contents` HTTP 直拉回传方式及交互式 `colab drivemount` FUSE 挂载回传方式）
- **日期（Date）**：2026-10-23
- **关联实验**：[`experiments/04-gdrive-vs-http-egress-transfer`](../../experiments/04-gdrive-vs-http-egress-transfer/README.md)（完整实验报告：[`report.md`](../../experiments/04-gdrive-vs-http-egress-transfer/report.md)；原始实测指标：[`exp04_egress_transfer_metrics.json`](../../experiments/04-gdrive-vs-http-egress-transfer/outputs/exp04_egress_transfer_metrics.json)；会话账本摘录：[`colab_session_ledger_excerpt.json`](../../experiments/04-gdrive-vs-http-egress-transfer/outputs/colab_session_ledger_excerpt.json)；生产级可复用工具：[`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py)）
- **前置依赖**：
  - [ADR-01：锁定 `Comfy-Org/MiniMax-H3` 黄金五件套（`42.03 GB`）与 HF 直连加载链路](./0001-model-weights-loading-pipeline.md)
  - [ADR-02：锁定 Google Colab Blackwell 96GB（`--gpu G4`）为标准推理机型](./0002-blackwell-gpu-and-sageattention-acceleration.md)
  - [ADR-03：在 Blackwell G4 上正式锁定 `Run 12 [S8-FullStack-Pareto]` 为默认标准推理配置，并确立零浪费 GPU 生命周期工程规范](./0003-sol-attn-and-g4-pareto-frontier.md)

---

## 1. 背景与问题（Context）

在 [ADR-03](./0003-sol-attn-and-g4-pareto-frontier.md) 确立“尾端单包 `.tar` 归档 + 本地渲染卸载”后，实验 03 将 12 条生成的 MP4 视频、首尾参考帧与 JSON 评测日志打包为单一未压缩归档包（`/content/exp03_g4_outputs_bundle.tar`，远端磁盘原始二进制体积 `122,419,200 B` = `122.42 MB`），通过 `colab download` 拉回本地耗时 **`26.598s`**（消耗 **`0.0532 CU`**），随后触发 `colab stop` 销毁实例。

然而，在日常生产（单视频 `~10.6 MB` 或小批量 `~41 MB`）与大规模消融实验（12 视频 `122.4 MB` 乃至 36 视频 `367.3 MB`）的尾端产物回传阶段，存在三个核心工程问题：

1. **`colab download`（方案 A：Jupyter `/api/contents?content=1` 直拉）存在 `+36.8421%` 协议膨胀税，且受本地下载网速单流阻塞 GPU**：
   - `colab download` 底层调用 Jupyter Contents API（`GET /api/contents/{path}?content=1`）。远端单核读取二进制文件后，调用 Python `base64.encodebytes()`（每 3 字节变为 4 个 ASCII 字符，且每 76 字符插入一个换行符 `\n`），再经 `json.dumps()` 将每个 `\n` 转义为 2 字节的 `\\n`，导致**网络层实际传输字节数精确膨胀了 $\gamma = \frac{4}{3} \times \frac{78}{76} = \frac{52}{38} \approx \mathbf{1.368421\times}$（`+36.8421%`）**（例如实验 03 的 `122,419,200 B` 归档包在 HTTP 响应中膨胀为 `167,521,347 B` = `167.52 MB`）。
   - 更严重的是，`/api/contents` 返回单一巨型 JSON 响应体，**不支持 `HTTP Range` 多线程分块下载或断点续传**。在整个跨境单流下载期间（当本地网络波动至 `0.5 ~ 2.0 MB/s` 时，拉取 `122.42 MB` 包需 `84.5s ~ 335.8s`），费率高达 **`7.2046 CU/hr`** 的 Blackwell G4 GPU 全程处于 100% 空转陪跑状态。
2. **传统 `colab drivemount`（方案 B1：`/content/drive` FUSE 挂载拷贝）存在交互式 OAuth 阻塞与异步写缓存丢数据风险**：
   - `google.colab.drive.mount('/content/drive')` 使用的 `dfs_ephemeral` 凭据绑定在单台临时虚拟机上，每次新建 Colab 实例均需人工在浏览器点击 OAuth 授权；实验 01 账本记录显示，每次挂载总耗时 `91.23s ~ 131.11s`，即便扣除人工点击时间，纯 `drive-fs` FUSE 守护进程启动仍需 **`11.60s ~ 15.22s`（均值 `13.41s`）**。
   - 此外，向 `/content/drive/MyDrive/` 执行 `shutil.copy2` 仅写入虚拟机本地磁盘缓存（`/root/.config/Google/DriveFS`）便立即返回；若此时直接执行 `colab stop` 销毁实例，后台尚未刷入云端的字节会直接丢失，必须额外阻塞调用 `drive.flush_and_unmount()`。
3. **能否通过零交互的 Google Drive v3 REST API 实现“机房骨干网极速中转 $\rightarrow$ 秒级关停 GPU $\rightarrow$ 停机后 0 CU 本地并发拉取”？**
   - `google-colab-cli` 的本地授权文件 `~/.config/colab-cli/token.json` 已内置 `https://www.googleapis.com/auth/drive.file` 与 `cloud-platform` 权限范围。若在 GCP 配额项目（`<your-gcp-quota-project>`）启用免费的 `drive.googleapis.com` 服务，即可在不挂载 FUSE、无需任何人工点击的情况下，直接从 Colab G4 实例通过 Google 机房内部骨干网向 Google Drive v3 API 直传原始二进制流，并在 `200 OK` 返回瞬间关停 GPU。

---

## 2. 实验证据（Evidence）

在实验 04（[`experiments/04-gdrive-vs-http-egress-transfer`](../../experiments/04-gdrive-vs-http-egress-transfer/report.md)，G4 节点 `gpu-g4-s-kkb-use5c2-3vzs5htjt6j8q`）中，我们使用实验 03 产出的真实 H.264 MP4 视频与 JSON 指标（`zlib` 压缩比 `1.0001`，完全不可二次压缩）构建了四个代表性负载档位（`T1` ~ `T4`），对三种尾端回传架构进行了严格对账实测：

### 2.1 四档位尾端回传实测对比表

| 归档包档位 (代表场景) | 原始二进制体积 (`raw_bytes`) | `/api/contents` 膨胀后流量 (`wire_bytes`) | 方案 A: `colab download` HTTP 直拉 (占用 GPU 时长 / CU) | 方案 B1: 传统 FUSE 挂载含守护进程初始化 (占用 GPU 时长 / CU) | **方案 B2 (本决策): GDrive REST 阶段 1 机房内网直传 (唯一占用 GPU 时长 / CU)** | **方案 B2 阶段 1 机房内网吞吐率** | **方案 B2 阶段 2: 停机后本地 8 线程并发拉取 (0 CU)** | **方案 B2 端到端总墙钟耗时 (阶段1+阶段2)** | **方案 B2 相对方案 A 的 GPU 释放加速比 & 尾端 CU 节省率** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1_Single_Run`**<br>*(日常单视频: 1x `Run 12` MP4 + JSON)* | `10.63 MB`<br>(`10,629,120 B`) | `14.55 MB`<br>(`+36.85%`) | `3.76s`<br>(`0.00752 CU`) | `17.20s`<br>(`0.03442 CU`) | **`1.04s`** *(Media)*<br>`1.29s` *(Resumable)*<br>(**`0.00209 CU`**) | `10.20 MB/s` | `2.84s`<br>(`0.0000 CU`) | **`3.88s`** | **`3.61x` 释放 GPU**<br>**省 `72.28%` CU** |
| **`T2_4Run_Batch`**<br>*(4 视频小批量消融 / 抽卡)* | `41.09 MB`<br>(`41,093,120 B`) | `56.23 MB`<br>(`+36.84%`) | `7.88s`<br>(`0.01577 CU`) | `17.59s`<br>(`0.03521 CU`) | **`1.29s`** *(Media)*<br>`1.68s` *(Resumable)*<br>(**`0.00259 CU`**) | `31.77 MB/s` | `5.02s`<br>(`0.0000 CU`) | **`6.32s`**<br>*(总耗时快 `1.25x`)* | **`6.09x` 释放 GPU**<br>**省 `83.58%` CU** |
| **`T3_12Run_Exp03`**<br>*(实验 03 完整 12 视频归档包)* | `122.42 MB`<br>(`122,419,200 B`) | `167.52 MB`<br>(`+36.84%`) | 实验04: `17.69s` (`0.03540 CU`)<br>实验03: `26.60s` (`0.05323 CU`) | `18.07s`<br>(`0.03616 CU`) | **`2.27s`** *(Media)*<br>**`2.16s`** *(Resumable)*<br>(**`0.00453 CU`**) | **`54.03 MB/s`**<br>*(Resumable `56.70 MB/s`)* | `10.94s`<br>(`0.0000 CU`) | **`13.21s`**<br>*(总耗时快 `1.34x~2.01x`!)* | **`7.81x ~ 11.74x`**<br>**省 `87.19% ~ 91.48%` CU** |
| **`T4_36Run_Sweep`**<br>*(36 视频大规模网格搜索)* | `367.26 MB`<br>(`367,257,600 B`) | `502.56 MB`<br>(`+36.84%`) | `48.05s`*<br>(`0.09615 CU`) | `20.78s`<br>(`0.04159 CU`) | **`4.60s`** *(Media)*<br>`4.87s` *(Resumable)*<br>(**`0.00920 CU`**) | **`79.89 MB/s`**<br>*(单次峰值 `91.23 MB/s`)* | `44.52s`<br>(`0.0000 CU`) | **`49.12s`** | **`10.45x` 释放 GPU**<br>**省 `90.43%` CU** |

### 2.2 本地弱网敏感度与数学盈亏平衡点

针对实验 03 完整 12 视频包（`T3`，原始体积 $S = 122.42\text{ MB}$），建立本地下载带宽 $B_{\text{local}}$ 的解析模型：

- **GPU 计费时长与 CU 维度（无条件胜出）**：
  - 方案 A 的 GPU 占用时长为 $T_{\text{GPU}}^{(A)} = 2.14\text{ s} + \frac{167.52\text{ MB}}{B_{\text{local}}}$，随本地网络变慢而反比激增；当本地网速处于 `0.5 ~ 2.0 MB/s` 的跨境弱网区间时，仅下载阶段就阻塞 GPU 长达 **`84.5s ~ 335.8s`（白白消耗 `0.169 ~ 0.672 CU`，相当于浪费 2 到 7 条完整 `Run 12` 视频的生成算力）**。
  - 方案 B2 的 GPU 占用时长恒定为 Google 机房内网传输时间 **`2.27s`（`0.00453 CU`）**，**与本地网速 $B_{\text{local}}$ 100% 解耦**，在弱网下节省高达 **`97.32% ~ 99.33%`** 的尾端 GPU 算力。
- **端到端总墙钟耗时维度（$B_{\text{local}} < 30.2\text{ MB/s}$ 时单流亦胜出，8 线程并发全面胜出）**：
  - 即便阶段 2 仅使用单线程下载，由于方案 B2 传输的是 `122.42 MB` 原始二进制流（比方案 A 少传 `45.10 MB` 的 Base64 膨胀字节），只要本地网速 $B_{\text{local}} < \mathbf{30.2\text{ MB/s}}$（`242 Mbps`），节省的冗余下载时间即超过阶段 1 的 `2.27s` 机房上传耗时，使得**方案 B2 两段接力的总墙钟耗时比方案 A 直接下载还要快**。
  - 叠加阶段 2 的 **8 线程 `HTTP Range` 分块并发拉取**（有效吞吐从单线程 `5.87 MB/s` 提升至 `11.19 MB/s`），`T3` 端到端总墙钟耗时压缩至 **`13.21s`**，比方案 A（`17.69s ~ 26.60s`）快 **`4.48s ~ 13.39s`**。

---

## 3. 决策内容（Decision）

自本 ADR 起，所有 Colab GPU 实验与日常视频生成任务的**尾端产物回传（Egress）**统一锁定采用 **两段式零交互 Google Drive v3 REST API 中转卸载方案（方案 B2，由 [`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py) 标准工具执行）**：

### 3.1 标准尾端回传执行协议（Two-Stage Zero-Click Egress Protocol）

1. **远端无压缩单包归档（In-VM Tar Bundling）**：
   - 由于 H.264 MP4 视频本身已是高熵压缩流（实测 `zlib` 二次压缩比为 `1.0001`），远端一律使用未压缩的 `tar -cf /content/outputs_bundle.tar -C /content/outputs .` 将所有 MP4 视频与 JSON 评测指标合并为单一归档包，严禁在 GPU 实例上开启 `gzip`/`xz` 浪费 CPU 与 GPU 挂钟时长。
2. **阶段 1：Colab GPU $\rightarrow$ Google Drive v3 REST API 机房内网极速直传 + 立即 `colab stop`（唯一 GPU 计费阶段）**：
   - 严禁调用需要交互式浏览器点击且启动缓慢的 `colab drivemount`（`/content/drive` FUSE 挂载）。
   - 直接读取本地 `~/.config/colab-cli/token.json` 的 OAuth2 刷新令牌（携带 `https://www.googleapis.com/auth/drive.file` 权限与配额项目头 `X-Goog-User-Project: <your-gcp-quota-project>`），在远端通过 `POST https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum`（或大于 `200 MB` 时使用 `uploadType=resumable`）将原始二进制 `.tar` 包直传至 Google Drive。
   - 远端断言校验返回的 `int(info["size"]) == raw_bytes` 且 `info["md5Checksum"] == local_md5`（确保 100% 落盘 Google 云端且无缓存未刷盘风险）后，向本地编排器输出 `file_id`，并在**毫秒级内立即执行 `colab stop -s <session>`（`POST /tun/m/unassign`）终止 GPU 计费**。
3. **阶段 2：停机后本地 8 线程 `HTTP Range` 零 CU 并发拉取与自动清理（`0.00 CU/hr` 阶段）**：
   - 在确认 GPU 实例已销毁（`assignmentsCount == 0`，消耗 `0.00 CU/hr`）后，本地 Mac 调用 `GET https://www.googleapis.com/drive/v3/files/{fileId}?alt=media` 并附带 `Range: bytes=start-end` 请求头，使用 `ThreadPoolExecutor(max_workers=8)`（默认分块 `8 MiB`，单分块含 4 次指数退避自动重试，彻底规避跨境代理单长连接 `SSLEOFError` 断流问题）并发拉回原始二进制归档包。
   - 本地完成完整字节流 `SHA-256` 校验并解包后，默认自动调用 `DELETE https://www.googleapis.com/drive/v3/files/{fileId}` 清理 Google Drive 上的临时中转包（保持云盘零垃圾残留；若需云盘持久化备份则显式传入 `--keep-gdrive`）。

### 3.2 标准调用命令与代码接口

- **命令行一键调用**：
  ```bash
  ~/.local/share/uv/tools/google-colab-cli/bin/python \
    experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py \
    -s <session_name> \
    --remote-path /content/outputs_bundle.tar \
    --local-path ./outputs/outputs_bundle.tar \
    --workers 8
  ```
- **Python 编排脚本直接集成**：
  ```python
  from pathlib import Path
  from gdrive_egress_relay import stage1_remote_push_and_stop, stage2_local_parallel_pull

  # 阶段 1: 远端直传 Google Drive (1~2.3s) 并在 finally 块中立即 colab stop 销毁 GPU
  s1 = stage1_remote_push_and_stop(
      session_name="h3-prod-g4",
      remote_file_path="/content/outputs_bundle.tar",
      stop_immediately=True,
  )

  # 阶段 2: GPU 已停机 (0.00 CU)，本地 8 线程 HTTP Range 并发拉取 + SHA-256 校验 + 自动清理云端临时包
  s2 = stage2_local_parallel_pull(
      file_id=s1["file_id"],
      local_dest_path=Path("./outputs/outputs_bundle.tar"),
      expected_raw_bytes=s1["raw_bytes"],
      expected_sha256=s1["sha256"],
      workers=8,
      delete_from_gdrive_after_pull=True,
  )
  ```

### 3.3 与 [ADR-01](./0001-model-weights-loading-pipeline.md) 的职责边界协同（云端 I/O 完整闭环）

| I/O 方向与阶段 | 数据特征与体量 | 锁定标准方案 | 禁用方案及原因 |
| :--- | :--- | :--- | :--- |
| **入向（Ingress）：模型与评测权重加载** ([ADR-01](./0001-model-weights-loading-pipeline.md)) | 公开只读大模型权重（`42.03 GB` 黄金五件套） | **Hugging Face (`hf_xet` 8 线程并发) $\rightarrow$ `/content` 本地盘** (`328.9 MB/s`) | 禁用 `/content/drive` FUSE 读取（冷读仅 `46 MB/s` 且触发 `mmap` 缺页阻塞与交互式授权） |
| **出向（Egress）：生成视频与指标回传** (**本 ADR-04**) | 会话内新生成的视频与评测日志归档包（`.tar`） | **强制统一采用两段式 Google Drive REST 中转（阶段 1：`Colab -> Google Drive v3 REST API` (`54~80 MB/s`) $\rightarrow$ 立即 `colab stop`；阶段 2：本地 8 线程 `HTTP Range` 零 CU 拉取）** | 全面弃用 `colab download` 直拉回传（`+36.84%` Base64 膨胀 + 单流慢网阻塞 GPU）及交互式 `colab drivemount` FUSE 挂载 |

---

## 4. 影响与后果（Consequences）

### 正向收益（Pros）
- **尾端回传 GPU 算力消耗锐减 `72.3% ~ 91.5%`（弱网下锐减 `97.3% ~ 99.3%`）**：将单次实验的尾端 GPU 占用时长从 `17.69s ~ 26.60s`（弱网下数分钟）刚性锁定在 **`1.04s ~ 2.27s`（仅 `0.0021 ~ 0.0045 CU`）**，彻底消除本地网络波动对昂贵 GPU 算力的拖累。
- **端到端总等待时间同步缩短且具备抗断流能力**：消除 `+36.84%` Base64 膨胀并在阶段 2 引入 8 线程 `HTTP Range` 分块重试拉取后，不仅总墙钟耗时缩短至 **`13.21s`**（比 `colab download` 快 `1.34x ~ 2.01x`），而且解决了跨境单长连接下载大文件时易发生的 `SSLEOFError` 连接中断重传问题。
- **100% 无人值守与零云盘残留**：无需任何浏览器点击授权，且阶段 2 拉取校验完成后自动删除云端临时文件，不侵占用户的 Google Drive 存储配额。

### 约束与注意事项（Cons & Mitigations）
- **依赖 GCP 配额项目已启用 `drive.googleapis.com`**：使用 `~/.config/colab-cli/token.json` 的 ADC 类凭据调用 Google Drive v3 API 时，请求头必须携带已启用 Google Drive API 的 `X-Goog-User-Project`（已在项目 `<your-gcp-quota-project>` 中完成永久启用并封装为 [`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py) 的默认常量）。
- **OAuth 访问令牌有效期刷新**：单次 `access_token` 有效期为 1 小时；[`gdrive_egress_relay.py`](../../experiments/04-gdrive-vs-http-egress-transfer/code/gdrive_egress_relay.py) 在发起阶段 1 远端调用前会自动通过 `refresh_token` 刷新令牌，确保长时实验收尾阶段凭证始终有效。
