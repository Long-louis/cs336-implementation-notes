### Problem (unicode1)
(a) `chr(0)` 返回的是 U+0000 的 Unicode **空字符 (null character)**。

(b) 该字符的 `__repr__()` 表示为转义序列 `'\x00'`，而其直接打印表现通常是不可见的，不占用任何宽度。

(c) 当它出现在 Python 文本中时，它被视为一个普通的单一字符（不影响连接），但在某些系统显示或打印输出中可能会完全不可见、占据零宽度或引发系统特定的呈现问题（例如在 C 语言接口中被视为终止符）。

### Problem (unicode2)
(a) UTF-8 对 ASCII 和常见文本更具空间效率（通常为 1 个字节），而 UTF-16 或 UTF-32 会强制使用 2 或 4 个字节，且 UTF-8 对 ASCII 是后向兼容的。

(b) 示例：`b'\xe4\xb8\xad'`（汉字“中”的 UTF-8 编码）。原因是 UTF-8 是变长编码，单个多字节序列中的字节独立出来通常不是有效的 UTF-8 字符。

(c) 示例：`b'\xff\xff'`。解释：`0xFF` 在任何有效的 UTF-8 序列中都是不允许出现的非法起始字节。

### Problem (train_bpe)
see `bpe_tokenizer.py` for the implementation of `train_bpe`

### Problem (train_bpe_tinystories)
(a) 我在 `data/TinyStoriesV2-GPT4-train.txt` 上以 `vocab_size=10000`、特殊 token 为 `<|endoftext|>` 训练并序列化了词表与 merges；实测训练耗时约 151.37 秒（0.042 小时），峰值内存约 2.36 GB。词表中最长 token 是 `" accomplishment"`（15 bytes），这个结果是合理的，因为高频前导空格+词干在 BPE 中常被合并成单个 token。

(b) 我使用 `cProfile` + `pstats` 进行分析（可视化可用 `snakeviz`），热点主要在 BPE merge 阶段每轮寻找最佳 pair 的过程：`_merge` 内部反复调用 `max(pair_counts, key=lambda ...)`，其中 `max` 与其 key `lambda` 占据了最主要时间，因此当前瓶颈不是 I/O，而是迭代合并时的全表扫描选优。

### Problem (train_bpe_expts_owt)
(a) 我在 `data/owt_train.txt` 上以 `vocab_size=32000` 完成了 byte-level BPE 训练并序列化输出；实测训练耗时约 21926.16 秒（6.09 小时），峰值内存约 11.40 GB。词表中最长 token 长度为 64 bytes，对应内容表现为重复的乱码样式片段，这在开放网页语料中是合理的，通常来自编码污染或高频异常字节模式被持续 merge。

(b) 与 TinyStories 相比，OWT 训练出的 tokenizer 更“开放域”：合并次数更多（31743 vs 9743）、最长 token 更长（64 vs 15）、包含更多噪声/异常模式；而 TinyStories 语料更干净、叙事风格更集中，token 更偏向常见自然语言子词。总体上，OWT tokenizer 覆盖面更广但也更“杂”，TinyStories tokenizer 更规整、任务针对性更强。

### Problem (tokenizer)
我已在 `bpe_tokenizer.py` 中实现 `BpeTokenizer` 的核心接口（`__init__`、`from_files`、`encode`、`encode_iterable`、`decode`），并在 `tests/adapters.py` 中完成 `get_tokenizer` 适配。随后运行 `uv run pytest tests/test_tokenizer.py`，Tokenizer 相关测试已通过。