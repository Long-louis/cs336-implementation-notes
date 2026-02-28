# 架构模式与跨文件约定

## 0) 目录组织模式（按领域分层）
- `cs336_basics/nn/`：承载模型基础组件（如 `Linear`），避免与 tokenizer、训练脚本混放。
- `cs336_basics/tokenizer/`：承载 tokenizer 对外入口（`BpeTokenizerTrainer` / `BpeTokenizer`）。
- `cs336_basics/bpe_tokenizer.py`：当前作为兼容层/历史实现承载文件，子包通过 re-export 对外提供稳定入口。
- 设计目标：**按问题域拆分模块**，保证“可读、可测、可迁移”。

## 1) 适配器驱动的实现边界（Adapter Boundary）
- `tests/adapters.py` 定义了作业要求实现的统一函数接口，测试只通过这些入口调用学生实现（`tests/adapters.py:1-595`）。
- 多个测试文件都依赖该边界，而不是直接依赖具体实现模块（`tests/test_train_bpe.py:4-35`, `tests/test_tokenizer.py:11-75`）。
- 设计含义：实现可以替换，但对外函数签名和语义必须稳定。

## 2) “核心算法 + 任务脚本”分层
- 核心逻辑由 `BpeTokenizerTrainer` / `BpeTokenizer` 承载；当前通过 `cs336_basics/tokenizer/` 暴露统一入口。
- TinyStories 与 OWT 使用独立脚本承载参数、I/O 与实验流程（`cs336_basics/train_bpe_tinystories.py`, `cs336_basics/train_bpe_expts_owt.py`）。
- 两个脚本共享同一训练器 API（`BpeTokenizerTrainer.train`），体现“算法复用、任务配置解耦”。

## 3) 字节级词表表示与 latin-1 持久化约定
- 训练与 tokenizer 内部统一以 `bytes` 表示 token 与 merge 元素。
- 导出 JSON / 文本时统一采用 `latin-1` 双向映射保留 0-255 字节信息（见 `train_bpe_*` 脚本中的序列化函数）。
- 测试侧同样做 GPT-2 字节映射还原，保证比较时语义一致（`tests/common.py`, `tests/test_train_bpe.py`, `tests/test_tokenizer.py`）。

## 4) 两阶段 BPE 管线：并行预分词 + 增量合并
- 第一阶段：文件切块与并行预分词计数（`_find_chunk_boundaries`, `_load_and_split`, `ProcessPoolExecutor`）。
- 第二阶段：基于 `pair_counts` 与 `pair_to_words` 的增量 merge，避免全量重算。
- 对应测试存在性能上限约束，驱动实现采用增量更新而非朴素重扫（`tests/test_train_bpe.py`）。

## 5) 数据边界与前置校验（Fail Fast）
- OWT 训练脚本在正式训练前验证 boundary token 是否存在，缺失即抛错（`cs336_basics/train_bpe_expts_owt.py:27-43`）。
- 输入、输出、统计文件路径均在入口阶段确定，运行失败尽早暴露（`cs336_basics/train_bpe_expts_owt.py:91-124`, `cs336_basics/train_bpe_tinystories.py:68-90`）。

## 6) 可观测性模式：统计信息与资源指标
- 训练脚本统一收集耗时、峰值内存、词表规模与最长 token 等指标并输出 JSON。
- 测试中对速度和内存也有显式约束，形成“实现—评测”闭环（`tests/test_train_bpe.py`, `tests/test_tokenizer.py`）。

## 7) 命令行入口一致性（CLI Consistency）
- 两个训练脚本都采用 `argparse` + `main()` + `if __name__ == "__main__"` 入口模式。
- 参数命名保持一致（如 `--input`, `--vocab-size`, `--workers`, `--chunks`），便于实验切换。

## 8) 兼容迁移模式（Compatibility Layer）
- 新增子包时，优先让外部调用走 `cs336_basics.tokenizer` / `cs336_basics.nn` 的稳定导入路径。
- 历史路径（如 `cs336_basics/bpe_tokenizer.py`）可保留为兼容层，降低重构期间的破坏性。
- 测试适配层 `tests/adapters.py` 应尽量只依赖稳定入口，不绑定内部文件布局。

## 9) 提交流程自动化与结果产物约定
- `make_submission.sh` 将测试执行与打包串联，测试报告固定为 `test_results.xml`（`make_submission.sh:2-6`）。
- 打包时大量排除中间产物与大文件，确保提交内容可控（`make_submission.sh:12-35`）。

## 10) 包级元数据最小化
- 包版本通过 `importlib.metadata` 从安装元数据读取，而非手写常量（`cs336_basics/__init__.py:1-3`）。
- 该约定与 `pyproject.toml` 的包声明保持一致（`pyproject.toml:2-5`, `pyproject.toml:24-35`）。
