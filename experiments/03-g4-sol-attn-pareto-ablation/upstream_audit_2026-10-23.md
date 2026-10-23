# H3 稀疏注意力与融合调制源码审计

审计日期：2026-10-23，Asia/Singapore。范围：上游固定版本源码、当前主分支对照、GitHub issue/PR、既有实验工件和 CPU 调度重放。本次没有重新执行 GPU 推理，没有修改实验实现，也没有向上游发帖。

## 结论

1. Discovery 1 的首个稀疏步双遍 QKV 投影确实存在，当前上游也保留它。但这是源码明确说明的流式显存优化取舍，不是意外重复消费生成器。实验验证了在本次 96GB / 8 步配置下绕过该路径的收益；称为普适的“关键架构缺陷”过强。
2. Discovery 2 的进度映射确实没有撤销 H3 flow shift，当前社区版本尚未修正。按节点“前一定比例采样保持稠密”的语义以及 ComfyUI 的 percent-to-sigma 约定，这是可以单独上报的调度语义问题。公开检索未发现完全对应的 issue/PR。
3. 融合调制是社区原有实现，v0.6.0 已引入；本实验的贡献是集成与在该工作负载上的验证。上游另有调制融合 PR，但不是同一实现，也不能混同精度与性能承诺。

## 仓库和版本

| 角色 | 仓库 | 本次核验版本 |
|---|---|---|
| H3 模型、采样设置、稀疏节点与 4K QKV 生产者 | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 实验 `f5ea88af1c46c50a39792d00af80e5af941f99f6`；当前 `d07dff4565ea9804277fff8cdfa0c817e08c3d44` |
| 原生 CUDA Sol-Attn、首步标定及分块接口 | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | `v0.2.40` = `f8b7e97a1418d2e529bd5405fbf86312be198ae6`，也是本次读取的 main |
| 实验实际使用的 Triton 节点、调度器、调制融合 | [r-vage/ComfyUI-sol-attn](https://github.com/r-vage/ComfyUI-sol-attn) | `v0.6.7`，`69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3`，实验与当前 HEAD 相同 |
| 社区实现的原始来源 | [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | 当前 `930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`，v0.6.2 |
| Sol-Attn 算法/参考实现来源 | [NVlabs/Sana，sol-engine 分支](https://github.com/NVlabs/Sana/tree/sol-engine) | 本次分支头 `9f0af774e7be7a82cb429de7428e678e8157c750`；仅作算法来源与相关问题检索，不归责于此 |

ComfyUI 的 `nodes_sparse_attention.py` 和 `comfy/ldm/minimax/model.py` 在实验 commit 与本次主分支头逐字相同。Kitchen CUDA Python 包装文件在 v0.2.40 与本次 main 逐字相同。

原始 Saganaki22 仓库本次 API 读取成功，其历史 issue 也可读。r-vage README 关于原仓库不可用的描述不宜当作当前事实。

## Discovery 1：源码支持双遍投影，但需要正确解释代价

调用链：

```text
ComfyUI BlockSparseAttention
  → 给每个 H3 double_block 注册 make_h3_block_patch
  → h3_sparse_attention
  → chunks() 每 4096 token 调用一次 attn.qkv_proj
  → comfy-kitchen sol_attn_chunked
  → 无统计时 produce(dummy) 收集 K 均值与 V 幅度
  → produce(kmean, vscale) 重新生成并量化
  → sol_attn_core
```

源码位置：

- [ComfyUI 分块生产与统计缓存，248–301 行](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy_extras/nodes_sparse_attention.py#L248-L301)。缓存按 block、token 数和 conditioning UUID 区分；首次分支同时缺少 `kmean` 与 `vscale`。
- [Kitchen 接口说明，3119–3126 行](https://github.com/Comfy-Org/comfy-kitchen/blob/f8b7e97a1418d2e529bd5405fbf86312be198ae6/comfy_kitchen/backends/cuda/__init__.py#L3119-L3126)明确写了无统计时先测量再量化，执行两遍。
- [Kitchen 迭代和 bootstrap，3143–3187 行](https://github.com/Comfy-Org/comfy-kitchen/blob/f8b7e97a1418d2e529bd5405fbf86312be198ae6/comfy_kitchen/backends/cuda/__init__.py#L3143-L3187)。传入 callable 时每遍重新执行投影；传入 iterable 且无统计时则先物化列表，避免重新投影但保留全部分片。
- [H3 原生全序列投影，174–201 行](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy/ldm/minimax/model.py#L174-L201)和[普通注意力 override，171–212 行](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy_extras/nodes_sparse_attention.py#L171-L212)构成实验绕过 block patch 后的 direct 路径。

对于本次 42,058 token，`ceil(42058 / 4096) = 11`。首次 22 次、后续 11 次 `qkv_proj`，50 层对应 1,100/550 次模块调用；direct 为 50 次。**22 次分片投影总计覆盖两遍序列，不等于 22 次全序列矩阵乘。** 本地仪表实际计数的是 Python `qkv_proj.forward`，不是 CUDA profiler 的 GEMM kernel launch 计数。

“零拷贝”是指 split/view/transpose 沿用 QKV 缓冲区，避免单独复制 q/k/v；完整 BF16 QKV 仍然会被分配。按本次维度估算，它约为 1.6846 GiB，而单个 4K 分片约 0.1641 GiB；这不是总显存峰值，尚未计入输出、工作区和模型其他驻留。

既有实验数据支持：

| 对照 | E2E 净省 | Q_total 变化 |
|---|---:|---:|
| Run 05 → 07，全稀疏 | 0.5289 s | +0.1966 |
| Run 06 → 08，20% 预热 | 0.6373 s | +0.1072 |

单块微基准为 cold chunked 67.633 ms、warm chunked 52.365 ms、direct 53.303 ms。warm 分块在该样本中反而略快；收益主要来自首次标定成本的摊销。cold 只计时一次，warm/direct 各循环五次；端到端每配置是一次请求。这支持“两个已测配置观察到收益”，不能证明跨 seed、长度和重复运行的稳定收益或稳定画质提升。

分块路径采用上一步统计，而 direct 重新计算当前输入统计；二者并非纯粹更改投影粒度的位级等价路径。原有微基准也记录了非零相对 L1 差异。

已有讨论：

| 项目 | 当前状态 | 与本问题的关系 |
|---|---|---|
| [Kitchen #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117) | 已合并 | 原始实现已说明分块减少完整 QKV 显存、统计跨步复用、首次自测。该取舍有明确设计背景。 |
| [ComfyUI #16378](https://github.com/Comfy-Org/ComfyUI/pull/16378) | 已关闭、未合并 | 提议 4K → 24K；[维护者 CUDA 实测](https://github.com/Comfy-Org/ComfyUI/pull/16378#issuecomment-5712767180)显示 84K token 时约慢 16%，临时显存更多。后续定位为另一瓶颈并撤回。 |
| [ComfyUI #16404](https://github.com/Comfy-Org/ComfyUI/pull/16404) | Open、未合并 | 在 VBAR fallback 下持有 INT8 权重 cast，避免每片重新物化权重；保持 4K，并明确排除独立的 bootstrap QKV 缓存实验。有关联但不是消除本次首步双遍投影。 |
| [Spectrum #106](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/106) | 已合并 | 已审计 cold → primed 的 `kmean/vscale` 生命周期，用于预测历史衔接；没有移除双遍 QKV。 |

可上游化的方向是显式的显存/速度策略或首步投影复用，而不是无条件删除低显存路径。实验中的 `pop("dit", None)` 会清除整个 dit replacement 字典，在可组合节点环境中应改为精确管理自己的 block hook。

## Discovery 2：sigma 比例不能直接代表未偏移进度

[社区 `_TauSchedule.progress()`，464–498 行](https://github.com/r-vage/ComfyUI-sol-attn/blob/69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3/minimax.py#L464-L498)只用两个 sigma 端点做线性归一化；[398 行](https://github.com/r-vage/ComfyUI-sol-attn/blob/69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3/minimax.py#L396-L399)据此判断 dense gate。在端点 1 和 0 时就是 `1 - sigma`。同一个 progress 也驱动 tau，所以问题同时影响预热和退火速度。

H3 的[采样设置](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy/supported_models.py#L986-L994)为 video shift 12、audio shift 3；注意力节点读取的是 sampler 的 video sigma。ComfyUI 的[正向映射](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy/model_sampling.py#L289-L336)为：

```text
sigma = alpha * t / (1 + (alpha - 1) * t)
t = sigma / (alpha - (alpha - 1) * sigma)
progress = 1 - t
```

本次用真实社区类的源码片段进行 CPU 重放，结果：

| 步（从 1 开始） | sigma，显示到六位 | 原 progress | 逆映射 progress | 原路由 → 修正路由 |
|---|---:|---:|---:|---|
| 1 | 1.000000 | 0.0000 | 0.000 | 稠密 → 稠密 |
| 2 | 0.988235 | 0.0118 | 0.125 | 稠密 → 稠密 |
| 3 | 0.972973 | 0.0270 | 0.250 | 稠密 → 稀疏 |
| 4 | 0.952381 | 0.0476 | 0.375 | 稠密 → 稀疏 |
| 5 | 0.923077 | 0.0769 | 0.500 | 稠密 → 稀疏 |
| 6 | 0.878049 | 0.1220 | 0.625 | 稠密 → 稀疏 |
| 7 | 0.800000 | 0.2000 | 0.750 | 稠密 → 稀疏 |
| 8 | 0.631579 | 0.3684 | 0.875 | 稀疏 → 稀疏 |

第 7 步存在容易漏掉的浮点边界：FP32 sigma 实际是 `0.800000011920929`，所以原 progress 是 `0.19999998807907104 < 0.20`。用四舍五入后的表格直接判断严格小于会造成歧义。

本地 Run 11 记录 300 次 Triton Sol、102 次 Sage；其中 100 次 Sage 对应 2 × 50 层预热，另外 2 次是 token refiner。CPU 重放、原始调度表和该运行计数一致。社区 `dense_percent` 默认 0.0，不能描述为默认配置自动触发 20% 预热缺陷；配置为 0.2 才出现所述退化，tau 进度偏斜则独立存在。

检索结果：r-vage 全部公开 issues 和 pulls 均为空；原始 Saganaki22 的 10 条 issue/PR 条目中未发现对应根因；全局 `_TauSchedule`、`H3 dense_percent` 的 issue/PR 搜索也未命中。[原仓库 #9](https://github.com/Saganaki22/ComfyUI-sol-attn/issues/9)报告“不提速”，回复涉及 PyTorch 版本，不能认定为这个 bug。[NVlabs/Sana #517](https://github.com/NVlabs/Sana/pull/517)也涉及 dense gate 错位，但根因是 FirstBlockCache 破坏调用计数，不是 flow shift。检索未命中不等于证明从未有人报告。

正式修复应读取有效 model sampling shift，而不是硬编码 12；还应约定“比例”是未偏移时间还是实际采样步序。非均匀、自定义或截断 sigma 序列下，这两者不相同。本实验的全局 `_TauSchedule.progress = ...` monkeypatch 只适合串行受控消融，不应直接作为支持多模型并发的节点补丁。

## 融合调制：已存在的优化，精度边界要明确

原始实现由 [v0.6.0 commit `bd28909`](https://github.com/r-vage/ComfyUI-sol-attn/commit/bd28909e6a152ba5db4e3590d2d9df1c249943f2)于 2026-08-09 引入，并由 r-vage 延续。[changelog](https://github.com/r-vage/ComfyUI-sol-attn/blob/69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3/changelog.md#L101-L122)已经明确记录 exact H3 fusion。

- [原生 DiTBlock](https://github.com/Comfy-Org/ComfyUI/blob/f5ea88af1c46c50a39792d00af80e5af941f99f6/comfy/ldm/minimax/model.py#L231-L298)每层执行两次 scale/shift、两次 gated residual，各自遍历 modality segments。
- [社区包装](https://github.com/r-vage/ComfyUI-sol-attn/blob/69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3/minimax.py#L182-L250)替换为两种 Triton 内核各执行两次。是每层四次融合调用，不是将四处工作合为一个 kernel，也没有把 RMSNorm、attention 和 MLP 全部融合。
- [内核](https://github.com/r-vage/ComfyUI-sol-attn/blob/69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3/sol_kernel/h3_fusion.py#L13-L108)显式保留 eager BF16 的中间舍入边界；gate 对齐 addcmul 的 FP32 opmath。CUDA BF16、无梯度、标量 segment row 是加速路径前提；per-token modulation row 会回退 eager。

本次重新计算已有 MP4 的 SHA-256：Run 02 和 Run 04 都是 `1acf4d6abf726e5b0db65e03fd37c1dc44c40cbddffb052a0d7d7a3d04d37b02`。这证实该稠密端到端对照的文件逐字节一致。既有单块测试的 modulation/gate max absolute diff 均为 0。Run 08 与 Run 12 的 MP4 哈希不同、Q_total 相差 -0.0215，不能把所有组合概括为“端到端 100% 位级无损”。有限输入的 max-diff 检查本身也不是全域位模式证明。

已有相似提案：

| 提案 | 当前状态 | 区别 |
|---|---|---|
| [ComfyUI #15621](https://github.com/Comfy-Org/ComfyUI/pull/15621) | Open、未合并 | RMSNorm+AdaLN 融合及 segment 向量化；作者报告 FP16 最大误差 `3.05e-05`，不是社区 BF16 位级等价契约。 |
| [ComfyUI #15619](https://github.com/Comfy-Org/ComfyUI/pull/15619) | 已关闭、未合并 | 更广的融合与输入缓存提案，不能当作已上线功能。 |
| [ComfyUI #16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) | Open、未合并 | 将 H3 MLP FC2 INT8 输出与 indexed gate 融合，范围更靠近 GEMM epilogue；依赖 Kitchen #219/#223。 |
| [Kitchen #219](https://github.com/Comfy-Org/comfy-kitchen/pull/219)、[#223](https://github.com/Comfy-Org/comfy-kitchen/pull/223) | 均 Open、未合并 | 新增 indexed gate/residual 与输入量化组合接口；并非本实验使用的两个 Triton elementwise kernel。 |

另一个直接相关的数值线索是 [Kitchen issue #241](https://github.com/Comfy-Org/comfy-kitchen/issues/241)及[修复 PR #242](https://github.com/Comfy-Org/comfy-kitchen/pull/242)，二者仍 Open。它们记录 `token_aug=256` 时跨 CTA 的原子预留在溢出下导致保留 token 集合非确定，覆盖 direct/chunked 路径，并有 RTX PRO 6000 Blackwell 证据。这与本实验稀疏组差异值得联合排查；没有本实验同配置 A/A 重复和 workspace 证据，不能断定就是该根因。

## 本次工件与复现

- [固定源码片段、blob 标识、PR 状态与检索记录](audit/2026-10-23/source_evidence.json)
- [CPU 审计重放脚本](audit/2026-10-23/replay_audit.py)
- [重放结果、七项断言、视频哈希及历史配对差值](audit/2026-10-23/replay_result.json)

运行：

```bash
python3 experiments/03-g4-sol-attn-pareto-ablation/audit/2026-10-23/replay_audit.py
```

环境：本次 macOS / Python 3.14.6；仅用标准库，无 CUDA、PyTorch、网络和账号依赖。前提：保留本目录源码证据 JSON、原有 metrics JSON 和 Run 02/04/08/12 MP4。输出为独立审计目录下的 `replay_result.json`，只读取既有实验数据，不启动 GPU，不读取账单或凭据。

结果：**7/7 断言通过**。它们核验原调度 7/8、逆映射 2/8、与原记录路由一致、分块调用次数算术、Run 11 调用计数、稠密组哈希相同和稀疏组哈希不同。它们不是新一轮 GPU/E2E 验收；GPU 耗时与算子误差仍来自此前实验工件。

## 对原总结的建议表述

> 实验确认并量化了 H3 默认 4K 流式 QKV 路径在首次稀疏调用时的双遍标定开销，并在 96GB、8 步工作负载下通过直通路径观察到约 0.53–0.64 秒端到端收益；修正了社区调度器将 shifted sigma 线性归一化为进度的语义偏差，恢复预期的 2 步稠密、6 步稀疏调度；集成社区既有 BF16 舍入保持型调制融合，在已测稠密对照中验证输出完全一致并获得约 0.72 秒收益，在稀疏组合中观察到约 0.64 秒收益及微小输出差异。
