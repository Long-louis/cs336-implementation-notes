# CS336 Assignment 1 代理工作指南（精简版）

## 1) 项目概览（WHY）
- 本仓库核心目标是完成 **CS336 Assignment 1: Building a Transformer LM**，不是单一 BPE 项目（`MinerU_markdown_cs336_spring2025_assignment1_basics_2022353073144655872.md:1`, `...:9`）。
- 讨论实现细节时，默认先对齐作业 handout 的 Problem 要求、分值与实验约束，再落到具体代码（`...:259`, `...:337`, `...:878`, `...:1204`, `...:1514`）。

## 2) 技术栈（WHAT）
- Python >= 3.11（`pyproject.toml:6`）。
- 依赖与执行：`uv`（`README.md:9-27`, `pyproject.toml:24-35`）。
- 核心库：`torch`, `numpy`, `regex`, `tiktoken`, `pytest` 等（`pyproject.toml:7-22`）。

## 3) 关键目录与用途（WHAT）
- `cs336_basics/`：你的实现与训练脚本。
  - `cs336_basics/nn/`：模型基础组件（当前已有 `Linear`）。
  - `cs336_basics/tokenizer/`：Tokenizer 子包入口（`BpeTokenizerTrainer` / `BpeTokenizer`）。
  - `cs336_basics/bpe_tokenizer.py`：当前作为兼容层与历史实现承载文件。
- `tests/`：评分导向的验证入口；`tests/adapters.py` 是实现对接边界（`tests/adapters.py:13-566`）。
- `data/`：TinyStories/OWT 数据与导出产物。
- `MinerU_markdown_cs336_spring2025_assignment1_basics_2022353073144655872.md`：**作业主说明文档**（长文，按章节定位阅读）。
- `docs/architectural_patterns.md`：跨文件设计模式与约定索引。

## 4) 必备构建/测试命令（HOW）
- 全量测试：`uv run pytest`（`README.md:26`）。
- 详细测试 + junit：`uv run pytest -v ./tests --junitxml=test_results.xml`（`make_submission.sh:4`）。
- 提交打包：`bash make_submission.sh`（`make_submission.sh:2-36`）。

## 5) 作业主说明文档导航（优先阅读）
- 总览与执行：Overview / What you will implement / What you will run（`...:9`, `...:13`, `...:23`）。
- Part 2 Tokenizer：Unicode、BPE 训练、编码解码、实验（`...:87-376`）。
  - `train_bpe`（`...:259`）、`tokenizer`（`...:337`）。
- Part 3 Model：Linear/Embedding、RMSNorm、Attention、Transformer LM（`...:541-906`）。
  - `linear`（`...:565`）、`embedding`（`...:599`）、`rmsnorm`（`...:659`）、`scaled_dot_product_attention`（`...:808`）、`transformer_lm`（`...:878`）。
- Part 4 Optimization：cross-entropy、AdamW、LR schedule、gradient clipping（`...:951-1132`）。
- Part 5 Training loop：data loading、checkpoint、整体训练（`...:1136-1204`）。
- Part 6 Decoding（`...:1216-1250`）。
- Part 7 Experiments：日志、TinyStories、ablation、OWT、leaderboard（`...:1262-1514`）。

## 6) 测试到任务的快速映射
- 模型组件：`tests/test_model.py:21-193` ↔ adapters 中 `run_*` 模型函数（`tests/adapters.py:13-384`）。
- 数值与训练工具：`tests/test_nn_utils.py:9-62`, `tests/test_optimizer.py:29-52`, `tests/test_data.py:10`, `tests/test_serialization.py:57`。
- tokenizer/BPE：`tests/test_tokenizer.py:77-436`, `tests/test_train_bpe.py:8-65`。

## 7) Additional Documentation（按需渐进阅读）
- 架构模式与跨文件约定：`docs/architectural_patterns.md`
- 环境与基础运行：`README.md:9-37`
- 变更记录：`CHANGELOG.md`
- 课程作业提交文档：`cs336_basics/writeup.md`

> 工作顺序建议：先定位 handout 章节与对应 Problem，再看 `tests/adapters.py` 的目标接口，最后实现 `cs336_basics/` 并用 `pytest` 回归。
