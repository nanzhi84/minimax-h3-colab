# PROJECTS.md — minimax-h3-colab 项目特异规范与领域红线

## 1. 项目定位与命名约定
- **项目名称**：`minimax-h3-colab`（面向 Google Colab Blackwell G4 的 MiniMax `H3-Base-FL2VA` 声画同源视频生成推理基准与零浪费异构批处理生产系统）。
- **核心模块与目录约定**：
  - `studio/api/`：生产级三阶段异构批处理调度器与云端常驻推理引擎（含 [`manifest_schema.py`](studio/api/manifest_schema.py)、[`local_planner.py`](studio/api/local_planner.py)、[`colab_batch_runner.py`](studio/api/colab_batch_runner.py)、[`remote_batch_engine.py`](studio/api/remote_batch_engine.py)）。
  - `experiments/01..05-*/`：支撑架构决策的五个自包含实测基准沙盒（每个实验目录保持 `README.md`、`report.md`、`code/` 与忽略入库的 `outputs/` 结构）。
  - `docs/adr/`：四位数编号的架构决策记录（`0001` 至 `0005`）。

## 2. 领域红线与不变量
- **零个人密钥与隐私入库红线**：严禁在任何受控代码、配置清单、Notebook 或文档中硬编码 OAuth `access_token`、`refresh_token`、`client_secret`、Hugging Face Token、个人 GCP 配额项目号或本地绝对用户路径；运行时凭据统一从本地 `~/.config/colab-cli/` 或环境变量读取，并在注入远端临时脚本时设置 `0600` 权限且在 `finally` 块中立即销毁。
- **产出物与大体积归档隔离红线**：所有实验与生产生成的 `.mp4` 视频、`.tar` 归档包、原始运行日志及衍生评测产物必须统一输出至 `output/` 或 `outputs/` 目录（已由 [`.gitignore`](.gitignore) 全局忽略），严禁将生成产物或临时归档包提交至 Git 历史。
- **零浪费 GPU 生命周期红线**：严禁在挂载计费 GPU 的状态下执行本地慢网直传/直拉（`colab download`）或交互式代码调试；所有云端会话必须在 `try...finally` 保护下于 Google Drive REST 机房内网卸载完成瞬间立即执行 `colab stop` 释放实例。
- **架构与决策入口**：全局架构现状详见 [docs/architecture/overview.md](docs/architecture/overview.md)，核心架构权衡与算子选型详见 [ADR-0001](docs/adr/0001-model-weights-loading-pipeline.md) 至 [ADR-0005](docs/adr/0005-staged-batch-production-workflow.md)。
