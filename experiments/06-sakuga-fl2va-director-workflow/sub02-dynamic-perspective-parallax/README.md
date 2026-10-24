<!-- harness-ignore LNK001: outputs/ directory is ignored in git -->
# Experiment 06 — 子实验 2 (`sub02-dynamic-perspective-parallax`)：大跨度三维透视重构、景别跳跃与真实刚体受击反馈工作流

> **状态**：🔄 第一阶段（大跨度三维透视重构关键帧 `Frame A ~ Frame F` 生成与校验）进行中  
> **演进基线**：针对 [子实验 1 (`sub01-static-canvas-lineage`)](../sub01-static-canvas-lineage/report.md) 暴露的“背景锁死导致零光流视差”与“图层叠加式假碰撞”两大问题进行彻底重构  
> **前置架构基石**：[ADR-03](../../../docs/adr/0003-sol-attn-and-g4-pareto-frontier.md) · [ADR-04](../../../docs/adr/0004-two-stage-gdrive-rest-egress-relay.md) · [ADR-05](../../../docs/adr/0005-staged-batch-production-workflow.md)

---

## 1. 子实验 2 的两大核心革新（Core Innovations vs. Sub-Exp 01）

### 1.1 空间调度革新：放弃原图局部微调，强制首尾帧“跨景别 + 跨俯仰机位 + 三维透视重构”
在子实验 2 中，前序帧仅作为**角色与场景的视觉身份锚点（Character & Scene Identity Bible）**传入生图模型，严禁沿用同一张静态背景画布。每一组首尾帧（`A -> B`、`B -> C`、`D -> E`、`E -> F`）均强制执行**剧烈的摄像机空间位移与焦段/透视重构**，使首尾帧之间的背景像素差分（`Background Parallax MAE`）从子实验 1 的 `< 15` 提升至 **`> 55`**，从数学边界条件上逼迫 `MiniMax-H3` 的 3D 时空注意力计算高速相机推进、俯仰翻转与环绕视差：

| 分镜序号与 `shot_id` | 首帧机位与透视 (`first_frame`) | 尾帧机位与透视重构 (`last_frame`) | 逼迫模型合成的三维空间运动与视差 |
| :--- | :--- | :--- | :--- |
| **`shot_01_thunder_dash`**<br>(`Frame A -> B`, `4.0s / 90f`) | **`Frame A`**：`24mm` **高角度越肩俯拍大远景（High-Angle OTS Wide）**<br>左前景剑士半蹲蓄力（占画面 35%），右远景台阶上方武士持炎刀戒备，纵深庭院一览无余 | **`Frame B`**：`14mm` **贴地虫视超广角倾斜仰拍近景（Worm's-Eye Dutch Low-Angle）**<br>摄像机贴近碎裂石阶以 `45°` 仰看两人上半身与交击刀锋，神庙巨柱与屋檐向天空陡峭汇聚 | 摄像机从左后方高处贴着地面极速俯冲推近，在冲至武士身前的瞬间下潜并剧烈上仰，产生强烈的地面掠过与立柱仰视视差 |
| **`shot_02_blade_clash_shockwave`**<br>(`Frame B -> C`, `5.0s / 124f`) | **`Frame B`**：`14mm` **贴地虫视超广角仰拍近景**<br>双刃在镜头正上方刚体咬合，上挑冲击力顶起武士前脚 | **`Frame C`**：`35mm` **120° 大角度环绕侧跟拍中近景（Lateral Orbital Tracking）**<br>摄像机绕至武士右侧后方，背景廊柱呈水平高速拉丝视差，剑士突入左中景完成贯穿横斩 | 摄像机围绕交锋中心执行大跨度 3D 环绕旋转并侧向跟拍，背景神庙廊柱快速横向掠过 |
| **`shot_03_eye_awakening_slash`**<br>(`Frame D -> E`, `3.0s / 73f`) | **`Frame D`**：`85mm` **长焦微距面部与刀锋极近景（Macro Telephoto Close-Up）**<br>剑士双瞳雷光与贴近镜头的横刃占满 85% 画幅 | **`Frame E`**：`12mm` **高空三点透视夸张近大远小跃斩（Ultra-Wide Foreshortened Leap）**<br>镜头极速拉远升空，剑士腾空跃向画面右前景，挥出的刀锋与咆哮雷龙首级呈金田式夸张透视扑向镜头，神庙地面在脚下急剧下陷 | 从面部微距瞬间拉出至超广角空中俯冲透视，产生极强的纵深拉伸感与破空跃动感 |
| **`shot_04_dragon_finale_blast`**<br>(`Frame E -> F`, `5.0s / 124f`) | **`Frame E`**：`12mm` **空中超广角近大远小雷龙扑镜**<br>雷龙巨口与剑锋紧贴右前景 | **`Frame F`**：`16mm` **30 米高空鸟瞰全景急拉（High-Altitude Bird's-Eye Pull-Out）**<br>镜头升至神庙屋顶正上方俯瞰整座建筑群，雷龙贯穿大殿将武士轰飞撞穿石墙，半座屋顶瓦片呈环形反重力掀飞 | 摄像机冲破神庙屋顶直拉高空鸟瞰，完整展现雷龙贯穿整座神庙建筑的毁天灭地空间尺度 |

---

### 1.2 动作交互革新：强制刚体碰撞体积、动量传递与受击躯干扭曲（Visceral Hit Deformation）
为彻底消除子实验 1 中“刀剑浮空贴纸叠加、防守方原地不动”的问题，子实验 2 在 `Frame B`、`Frame C`、`Frame F` 中注入了严格的刚体力学受击反馈：
1. **刚体咬合与防线崩解（`Frame B`）**：双刀在三维透视焦点处正面死锁咬合，剑士由下至上的暴烈挑斩直接顶歪武士的双手护手，武士前脚被迫离地、肩甲向后拧转、手腕肌肉紧绷变形。
2. **破防重创与躯干折弯（`Frame C`）**：剑士二段回旋斩破开防线，刀锋重重劈入武士黑色胸甲——**胸甲钢板明显凹陷崩裂并炸开蓝赤电火，武士整个躯干因剧烈受击向后折成痛苦的 C 字形反弓，兜鍪（头盔）歪斜，重心彻底崩溃，双膝弯曲，铁靴在神庙石板地面上向后硬生生犁出两条冒烟的深沟与飞溅碎石**。
3. **终章贯穿与击飞破坏（`Frame F`）**：苍蓝雷龙正面吞没武士，将其整个人向后轰飞撞穿神庙后殿石墙，建筑梁柱与屋瓦顺着冲击波矢量方向粉碎掀飞。

---

## 2. 目录结构与执行命令

```text
experiments/06-sakuga-fl2va-director-workflow/sub02-dynamic-perspective-parallax/
├── README.md                              # 本子实验设计方案
├── code/
│   ├── assets/                            # 6 张 1344×768 大跨度透视重构关键帧 (Frame A ~ Frame F)
│   ├── exp06_sub02_manifest.json          # 子实验 2 的 4 分镜 × 5 Seeds = 20 任务批处理清单
│   └── director_cut.py                    # 子实验 2 关键帧视差检验、自动选卡与 15.0s 导演精剪脚本
└── outputs/                               # 故事板、Sub01 vs Sub02 视差对比图、20 条候选视频与 15.0s 成片
```

```bash
# Step 1 (0.00 CU): 对齐子实验 2 的 Frame A..F 为 1344x768，计算首尾帧视差重构幅度并生成故事板
python3 experiments/06-sakuga-fl2va-director-workflow/sub02-dynamic-perspective-parallax/code/director_cut.py --align-keyframes

# Step 2 (0.00 CU): 干跑校验子实验 2 清单
python3 studio/api/colab_batch_runner.py \
  --manifest experiments/06-sakuga-fl2va-director-workflow/sub02-dynamic-perspective-parallax/code/exp06_sub02_manifest.json \
  --dry-run
```
