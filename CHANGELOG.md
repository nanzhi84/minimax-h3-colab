# Changelog

本项目的所有重要版本变更、新特性汇总与修复记录均归档于此。

## [Unreleased]

### Added
- 初始化 `minimax-h3-colab` Universal Harness 目录结构、架构全景图（[`docs/architecture/overview.md`](docs/architecture/overview.md)）与项目领域红线（[`PROJECTS.md`](PROJECTS.md)）。
- 归档五项核心架构决策记录（[ADR-0001](docs/adr/0001-model-weights-loading-pipeline.md) 至 [ADR-0005](docs/adr/0005-staged-batch-production-workflow.md)）及配套五组实测基准实验（`experiments/01..05`）。
- 交付生产级三阶段异构批处理引擎与 CLI 调度器（[`studio/api/`](studio/api/)），支持本地 0-CU 策划预投料、云端 `Qwen3-VL` SHA-256 条件去重预编码、Blackwell G4 常驻 `Run 12 [S8-FullStack-Pareto]` 连续渲染以及两段式 Google Drive REST 极速卸载停机。
