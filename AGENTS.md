# Agent 协作与仓库规范 (Universal Harness)

未经授权，严禁修改本文件及各目录下的规范说明。

## 1. 核心工作流与验收纪律

- **静态与契约检查**：任何代码或文档变更、移动后，必须优先通过类型检查、Linter 以及结构、文本与死链巡检（运行 `harness doctor`；可通过 `harness rule --all` 查阅规则细则，确需破例时在文档中使用 `<!-- harness-ignore CODE: 原因 -->` 标注豁免）。
- **测试驱动开发（TDD）与真实执行验收**：任何实质业务改动必须严格遵循测试驱动开发（先写测试 → 确认失败 → 编写实现 → 验证通过），严禁编写依赖 Mock 或拼凑覆盖率的无效单元测试。测试代码必须由全新上下文的独立子代理（Fresh Subagent）严格依据业务需求独立编写，严禁主 Agent 对着代码实现补写测试；验收时必须利用 `work/` 目录构造真实或最小可复现输入（可通过 `harness run <task> -- <cmd>` 自动隔离运行），直接执行测试、CLI、接口或端到端流程，通过核对 `artifacts/` 中的实际输出产物、日志或状态回执来完成自验。
- **高价值黑盒测试与命名规范**：测试必须聚焦于真实输入与输出的黑盒契约，禁止针对内部实现细节编写脆弱断言。测试文件必须按领域能力命名（`tests/test_<domain>.py`，严禁使用 `test_phase1`、`test1` 等排期或序号命名）；测试用例统一采用 `class Test<被测对象>:` + `def test_<场景或输入条件>_<预期行为结果>()` 层级结构，涵盖“被测对象、输入条件、预期外在行为”三要素，且严禁将底层实现库名写入测试名称。
- **Issue Checklist 诚实打勾与评论留痕**：所有 Issue（含本地 `docs/spec/issue#<N>/spec.md` 与 GitHub 线上 Issue）中的实施与验收 Checklist 必须随任务推进逐一核对并如实打勾：已通过真实运行验收的项勾选 `[x]`，遇到阻塞或卡点未跑通的项诚实保持 `[ ]` 未勾选；原 Issue 正文仅允许勾选状态框（严禁擅自篡改原 Issue 正文表述），具体测试结论与受阻原因统一在 **Issue Comment** 中补充说明。

## 2. 根目录工作区约定

- `README.md`：项目门面介绍（居中 Logo 与简介、`Core ideas`、`Get started`、`Learn more`）。
- `PROJECTS.md`：仅存放项目定位、统一命名约定与项目特异身份红线（如 `dd-xxx` 来源编号规则）；严禁混入架构设计权衡、模块选型或技术栈细节（此类内容必须统一归入 `docs/adr/`）。
- `tests/`：仅存放核心链路的高价值黑盒与集成测试。
- `work/` & `artifacts/`：日常开发与端到端执行验收产生的临时运行数据和构建产物（已忽略不入库，随时可清理）。
- `experiments/<name>/`：独立的技术选型与原型探索沙盒，内部环境与配置自包含，严禁为日常功能改动新建实验。
- `CHANGELOG.md`：集中记录版本发布、能力演进与破坏性变更。

## 3. 渐进式上下文路由（按需读取）

请根据当前任务性质，主动读取以下规范，未涉及的领域无需加载：

- **开始任何业务代码编写、修改数据流或外部集成前**：必须先读取根目录的 [PROJECTS.md](PROJECTS.md) 与 [docs/architecture/](docs/architecture)，严格遵守当前项目的领域红线与架构现状。
- **读写、移动或归档任何文档前**：必须先读取 [docs/README.md](docs/README.md)，严格遵守 `architecture`、`spec`（`issue#<N>/`）、`adr`、`runbooks`、`reference` 的生命周期与归档纪律（可使用 `harness spec new`、`harness adr new`、`harness adr supersede` 辅助流转）。
