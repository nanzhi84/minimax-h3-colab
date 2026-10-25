<!-- harness-ignore LNK001: outputs/ 中的上游工作流快照为忽略入库的本地调研工件 -->
# MiniMax H3 开源工作流与 Colab 后端调研

调研日期：2026-10-24。部署目标：本地浏览器或策划程序负责操作，推理后端继续使用Colab G4，自行运行开源H3权重。本轮没有安装节点、启动Colab或消耗GPU CU；已核对README、部分实现及实际工作流JSON，并保存7份原始工作流快照。

建议首先保留官方原生模板作为基线，再选择AIMixer导演台作为操作层；参考片复现优先验证Ref2VA／多关键帧／Fun ControlNet，续镜优先验证Motion Context。导演台改善操作方式，运动条件和采样配置才直接影响生成过程；这些方案尚未在本项目的Colab环境实跑，不能据调研宣布已经解决丝滑度。

## 候选项目

| 项目与来源 | 已核实的能力 | 对当前目标的价值 | Colab接入方式与范围 |
| --- | --- | --- | --- |
| [Comfy-Org 原生模板](https://github.com/Comfy-Org/workflow_templates) | 实际JSON包含本地UNET、CLIP、视频/音频VAE和采样节点；T2V、I2V/FL2VA、R2V、Multiframe、Fun ControlNet | 作为少依赖的推理基线；R2V能输入视频参考，多关键帧在单次生成中设置内部锚点 | Colab运行匹配版本ComfyUI；本地浏览器编辑或提交API Prompt。不是云API模板 |
| [AIMixer 导演台](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) | 中文时间轴、多组首尾帧、参考素材、源视频分段、原生音频、段间引导与可选二采；仓库含可导入JSON | 最接近可交互的导演工作流；能分别管理月夜、红叶和各动作段 | 节点Python和前端扩展安装在Colab ComfyUI，本地浏览器操作。FL2V用FL2VA；V2V/RV2V用Ref2VA |
| [NikoDemon80 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | 源码处理上一段视频/音频latent尾窗、上下文锚定、前缀裁剪、latent保存/加载和接缝诊断 | 针对只传末帧导致动作重新起步的问题；多帧上下文提供运动历史 | 作为Colab节点接入生成链。需保存采样器AV latent；不能只保留最终MP4。连续性是待验目标，不是物理保证 |
| [H3 Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot) | 多段脚本、连续采样、GGUF/safetensors加载、CORE及完整工作流；完整版本可接Motion Context和本地LLM | 更适合未来较长片段自动接力与脚本化生产 | 可自部署到Colab；先评估CORE，再决定是否加入LLM、缓存和加速依赖。本轮未核验其无缝效果 |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/MiniMax-H3.md) | Python MiniMaxH3Pipeline，FL2VA、Ref2VA、ControlNet和低显存示例；NF4文档给出7GB起步的特定路径 | 适合保留当前本地清单→Colab批处理的使用方式，提供另一套推理基线 | 在Colab运行Python，不提供同等导演时间轴。低显存数字依赖卸载与示例规格，不代表G4应选该配置 |

[Aeverlumi Segment Cache](https://github.com/Aeverlumi/ComfyUI-H3-Segment-Cache)提供落盘分段缓存、预览和合并模板，可作为生产管理补充；它本身不解决动作生成。8GB整合包如[YixuAn-13](https://github.com/YixuAn-13/minimax-h3-comfyui-all-in-one)适合研究功能组合，但含更多第三方节点、社区量化模型及平台依赖，暂不作为当前G4质量基线。

## 与现有工作流的关键区别

现有[生产引擎](../../studio/api/remote_batch_engine.py)直接调用官方H3节点，但固定加载8-step Turbo LoRA和Run12的Sage／Sol稀疏注意力组合；它没有运行完整ComfyUI计算图，也没有把上一段的AV latent传入下一段。[清单格式](../../studio/api/manifest_schema.py)与ComfyUI工作流JSON不同，因此导入新图需要适配，不能直接覆盖现有BatchManifest。

### 先建立质量基线

[Comfy官方指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)区分不启用Turbo的20步基线和8步加速，参考匹配建议20–25步。AIMixer的两份已读取示例实际设置为25步、`res_multistep/simple`、video/audio shift 12/3。建议先用无Turbo、默认稠密注意力做短动作对照，再逐项恢复加速；这是工作流比较，不是已证明当前某一个算子导致失真。取消加速后需要重新计时和核算预算，不能沿用8步耗时。

### 用视频或结构参考约束动作

- [R2V](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)使用Ref2VA权重，将参考视频明确指定为动作／镜头参考。我们此前只提供首尾图片，模型并未看到整段动作。原片应按场景分开取短参考；官方输入限制为每个视频2–15秒、视频总时长不超过15秒。
- [Multiframe Reference](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe)通过Add Guide把中间姿势放进同一次采样，区别于分别生成多条短片后拼接。它约束构图节点，不保证节点之间自动物理正确。
- [Fun ControlNet Union](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet)可用Pose、Depth、Canny等控制视频约束动作和结构。当前官方ComfyUI已有原生模板；[社区节点](https://github.com/wyzborrero/ComfyUI-H3-FunControl)也是一条实现路径。官方模板需要0.35.0+，多关键帧需要0.34.0+，部署时需固定兼容版本。

参考视频必须按时间重采样到24fps，不能只改帧率标签。官方节点把输入帧当24fps处理；直接把30fps帧序列作为24fps输入会改变时长。画面参考和参考音轨也要分别接入，载入视频不等于音轨自动参与条件。

### 续镜携带运动历史

Motion Context保留的是一小段尾部视频和音频上下文；优先传递上一段采样器的latent，可以减少像素解码再编码的往返。下一段生成后裁掉用于锚定的重叠前缀。AIMixer段间引导默认关闭，作者推荐22帧上下文；需要显式启用并核验。

同场景连续动作适合测试这种接力；月夜切换红叶属于明确换景，应单独规划新场景，不强行延续上一场的空间和声音。最终仍要正常速度与半速检查身体、剑路、运动方向及接点，不能把接缝指标当作丝滑度验收。

## 后端继续使用 Colab

```mermaid
flowchart LR
    A["本地浏览器 / 策划程序\n导演时间轴、素材与任务清单"] --> B["ComfyUI API Prompt\n或现有清单的适配层，待实现"]
    B --> C["Colab G4\nComfyUI、节点、权重与实际采样"]
    C --> D["Drive机房卸载\n立即停止GPU"]
    D --> E["本地回收\n审片、选片、精剪与报告"]
```

[Comfy官方API示例](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example.py)提供`/prompt`、WebSocket进度和`/history`结果查询。可沿用当前Colab生命周期与Drive两阶段传输，让新工作流成为渲染任务；完整图和自定义节点需在Colab执行，本地不承担CUDA推理。工作流画布JSON需要通过ComfyUI导出API格式或编写适配，不能把画布JSON原样当API Prompt提交。

开发时先在隔离环境验证节点、版本和短片，不修改现有锁定Run12生产路径。素材、提示词、图编译和校验在分配GPU前完成；批次内部完成必要的连续段计算和缓存，卸载后立即停机，再进行本地审片。仅调研不能视为这些集成已经完成。

官方[MiniMax H3仓库](https://github.com/MiniMax-AI/MiniMax-H3)把本地H3-Base和完整2K系统分开：Context-IR和Regenerate-2K仍有未开源的托管环节。选择自部署768p基础链路时，应使用`video_minimax_h3_*.json`原生模板；`api_minimax_h3_max_*`等云API模板不是在Colab加载权重推理。

## 已保存的工作流与核验范围

7份原始JSON保存在[本地工作流快照](../../experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/outputs/research/github-workflows-20261024/workflows/)：

- 官方：T2V、I2V/首尾帧、R2V、Multiframe Reference、Fun ControlNet Union。
- AIMixer：`minimax_h3_director_fl2v.json`、`minimax_h3_director_v2v.json`。

[来源与校验记录](../../experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/outputs/research/github-workflows-20261024/sources.json)记录下载地址、字节数、Git blob SHA及SHA-256。7份均与GitHub返回的blob哈希一致，JSON可解析、顶层节点ID无重复。blob SHA是文件标识，不是仓库commit；部署时仍需固定依赖版本。未下载模型权重，未安装环境，未排队生成，未将静态核验写成推理验收。

在仓库根目录可重复校验下载字节：

```bash
cd experiments/06-sakuga-fl2va-director-workflow/sub03-reference-motion-continuity/outputs/research/github-workflows-20261024
shasum -a 256 -c SHA256SUMS
```

JSON不包含模型、节点环境或模板引用的示例媒体；实际导入后需选择自己的图片／视频。`harness doctor`核对本调研的结构与链接，不能代替Colab推理验收。

适合下一步的小规模验证顺序：官方无Turbo短动作基线 → Ref2VA视频参考／多关键帧 → 同场景Motion Context接力 → 导演台操作与批处理适配。是否进入实际部署、生成及相应CU预算，属于后续执行范围。
