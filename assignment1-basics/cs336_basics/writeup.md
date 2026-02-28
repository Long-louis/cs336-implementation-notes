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

### Problem (learning_rate_tuning)

我按题目要求把 SGD toy 例子分别用学习率 $10^1,10^2,10^3$ 运行了 10 步（脚本：`cs336_basics/learning_rate_tuning.py`）。观察到：$10^1$ 时损失从 26.27 快速下降到 0.47（收敛更快）；$10^2$ 时损失基本不变（在本例中接近临界步长，几乎不下降）；$10^3$ 时损失迅速爆炸到 $10^{24}$ 量级（明显发散）。

### Problem (transformer_accounting)

以下记号用于统一表述：

- $V=\text{vocab\_size}$
- $n=\text{context\_length}$
- $L=\text{num\_layers}$
- $d=d_{\text{model}}$
- $d_{ff}$ 为 FFN 中间层维度

按本作业实现（无 bias、RoPE 无可训练参数、`Embedding` 与 `lm_head` 不共享权重、只统计矩阵乘 FLOPs），可写为：

- 参数量：

$$
P=2Vd+L(4d^2+3dd_{ff}+2d)+d
$$

- 前向矩阵乘 FLOPs：

$$
F=L(8nd^2+4n^2d+6ndd_{ff})+2ndV
$$

#### (a) GPT-2 XL 参数量与加载内存

给定 GPT-2 XL 配置：$V=50257, n=1024, L=48, d=1600, d_{ff}=6400$。

总参数量：

$$
P=2,127,057,600\ (\approx 2.127\text{B})
$$

若参数用 FP32（4 bytes）表示，仅加载模型参数所需内存：

$$
4P=8,508,230,400\ \text{bytes}\approx 8.51\ \text{GB}\approx 7.92\ \text{GiB}
$$

#### (b) GPT-2 XL 前向所需矩阵乘与 FLOPs

按“每层 + 最后输出层”分解：

1. 注意力线性投影（QKV+O）

   - $Q=xW_Q$: $2nd^2$
   - $K=xW_K$: $2nd^2$
   - $V=xW_V$: $2nd^2$
   - $\text{AttnOut}\cdot W_O$: $2nd^2$
   - 合计：$8nd^2$
2. 注意力核心矩阵乘

   - $QK^\top$: $2n^2d$
   - $AV$: $2n^2d$
   - 合计：$4n^2d$
3. FFN（SwiGLU 三个线性层）

   - $W_1$: $2ndd_{ff}$
   - $W_3$: $2ndd_{ff}$
   - $W_2$: $2ndd_{ff}$
   - 合计：$6ndd_{ff}$
4. 最终词表投影（lm\_head）

   - $2ndV$

总 FLOPs：

$$
F=L(8nd^2+4n^2d+6ndd_{ff})+2ndV
$$

代入 XL 数值：

$$
F=4,513,336,524,800\ \approx 4.51\times 10^{12}
$$

#### (c) 哪部分最耗 FLOPs（XL）

基于上式分项占比（$n=1024$）：

- FFN：约 $66.9\%$
- 注意力线性投影（QKV+O）：约 $22.3\%$
- 注意力矩阵乘（$QK^\top+AV$）：约 $7.1\%$
- 最终词表投影：约 $3.6\%$

因此 XL 在该上下文长度下主要 FLOPs 来自 FFN，其次是注意力投影。

#### (d) small / medium / large 的占比变化

统一取 $n=1024, V=50257, d_{ff}=4d$。

- GPT-2 small ($L=12,d=768$)

  - FFN: 49.75%
  - QKV+O: 16.58%
  - $QK^\top+AV$: 11.06%
  - lm\_head: 22.61%
- GPT-2 medium ($L=24,d=1024$)

  - FFN: 59.86%
  - QKV+O: 19.96%
  - $QK^\top+AV$: 9.98%
  - lm\_head: 10.20%
- GPT-2 large ($L=36,d=1280$)

  - FFN: 64.20%
  - QKV+O: 21.40%
  - $QK^\top+AV$: 8.56%
  - lm\_head: 5.84%

结论：模型变大时，$O(d^2)$ 相关部分（尤其 FFN 与投影）占比上升；$O(n^2d)$ 注意力核心与固定词表投影占比下降。

#### (e) XL 增大上下文到 16,384

当 XL 从 $n=1024$ 提升到 $n=16384$，总 FLOPs 变为：

$$
149,522,795,724,800\ \approx 1.495\times 10^{14}
$$

约为原来的 $33.1\times$（不是 $16\times$，因为注意力里有 $n^2$ 项）。

分项占比变为：

- 注意力矩阵乘（$QK^\top+AV$）：约 $55.1\%$
- FFN：约 $32.3\%$
- QKV+O：约 $10.8\%$
- lm\_head：约 $1.8\%$

可见长上下文下注意力核心矩阵乘成为主导计算瓶颈。

### 1. “注意力的线性投影” vs “注意力的矩阵乘”

虽然它们都涉及矩阵运算，但在 Transformer 内部的**物理意义**和**计算规模**完全不同：

#### **A. 注意力的线性投影 (Attention Projections，简称 QKV+O)**

这是指在注意力机制的前后，将“残差流”中的特征经过**可学习的参数矩阵**进行变换的过程。

* **物理含义**：这些是模型中真实的“层”，它们要把输入向量 $x$ 映射到不同的子空间（Query, Key, Value）。
* **涉及参数**：$W_Q, W_K, W_V$ 和 $W_O$。
* **计算内容 ($QKV+O$)**：
  1. $Q = x \cdot W_Q$：将输入 $x(n, d)$ 投影到查询空间。
  2. $K = x \cdot W_K$：投影到键空间。
  3. $V = x \cdot x \cdot W_V$：投影到值空间。
  4. $Output = Attn\_Result \cdot W_O$：将多头注意力的输出映射回 $d_{model}$ 维度。
* **FLOPs 计算**：每个投影都是 $(n, d) \times (d, d)$。根据 $2mnp$ 规则，每一项产生 $2nd^2$ FLOPs。
  * **公式汇总**：$4 \times 2nd^2 = \mathbf{8nd^2}$ (每层)。

#### **B. 注意力的矩阵乘 (Attention Math，简称 $QK^\top + AV$)**

这是注意力机制**内部**，“特征与特征”之间相互作用产生权重的过程，**没有可学习参数**参与这两个矩阵乘法。

* **物理含义**：

  * **$QK^\top$**（计算相关性）：Query 分别和每个 Key 算点积，决定每个词对其他词的关注程度，生成注意力得分矩阵（大小为 $n \times n$）。
  * **$AV$**（加权求和）：拿算出来的注意力权重矩阵（记为 $A$）去乘以 Value 矩阵。即：用生成的权重把所有的信息“捞”出来。
* **FLOPs 计算**：

  1. **$QK^\top$**：形状 $(n, d) \times (d, n) \rightarrow (n, n)$。FLOPs = $2n \cdot n \cdot d = 2n^2d$。
  2. **$AV$**：形状 $(n, n) \times (n, d) \rightarrow (n, d)$。FLOPs = $2n \cdot d \cdot n = 2n^2d$。

  * **公式汇总**：$2n^2d + 2n^2d = \mathbf{4n^2d}$ (每层)。

---

### 2. 为什么占比不一样？比例是怎么算出来的？

我们以 GPT-2 XL 为例：$d=1600, n=1024, d_{ff}=6400$。

#### **核心公式对比（每层）：**

1. **FFN (Feed-Forward)**：$3 \times (2nd \cdot d_{ff}) = 6ndd_{ff}$（因为有三个线性层 $W_1, W_2, W_3$）。
   * 代入：$6 \times 1024 \times 1600 \times 6400 \approx 62.9 \text{ GFLOPs}$
2. **投影 (QKV+O)**：$8nd^2$。
   * 代入：$8 \times 1024 \times 1600^2 \approx 20.9 \text{ GFLOPs}$
3. **注意力矩阵乘 ($QK^\top+AV$)**：$4n^2d$。
   * 代入：$4 \times 1024^2 \times 1600 \approx 6.7 \text{ GFLOPs}$

**为什么 FFN 最高？**
因为在大多数模型设计中，$d_{ff}$ 是 $d$ 的 $4$ 倍左右。

* 投影部分是 $8nd^2$。
* FFN 部分是 $6nd(4d) = 24nd^2$。
  **FFN 消耗的计算量通常是注意力投影部分的 3 倍左右**，这是因为 FFN 的中间层维度非常宽，要把信息进行高维映射和筛选。

**为什么投影比注意力矩阵乘（$QK^\top+AV$）大？**

* 比较 $8nd^2$ 和 $4n^2d$：其实就是在比较 $2d$ 和 $n$。
* 对于 GPT-2 XL，$2d = 3200$，而 $n = 1024$。因为 $d_{model}$ 明显大于序列长度 $n$，所以**参数变换（投影）**比**序列内部互动（注意力矩阵乘）**要耗费更多的算力。

---

### 3. $AV$ 具体是什么意思？

在注意力机制里，我们算出得分后会做 `softmax`：

$$
A = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})
$$

这里的 $A$ 是一个 $n \times n$ 的概率矩阵，表示每个词看其他词的权重。

**$AV$ 相乘就是：**

$$
Output = A \times V
$$

* $A$ 的形状是 $(n, n)$（序列长度 $\times$ 序列长度）。
* $V$ 的形状是 $(n, d)$（序列长度 $\times$ 向量维度）。
* **计算过程**：对于序列中的每一个位置 $i$，它都会查看 $A$ 矩阵的第 $i$ 行（即它对全序列所有词的控制权重），然后根据这些权重对 $V$ 矩阵（全序列的信息内容）进行加权平均。

### 总结（字母表示）：

| 模块                                      | FLOPs 公式 (单层) | 决定因素                   |
| :---------------------------------------- | :---------------- | :------------------------- |
| **FFN (SwiGLU)**                    | $6ndd_{ff}$     | 取决于模型深度和中间层宽度 |
| **Attention 投影 (QKV+O)**          | $8nd^2$         | 取决于$d_{model}$ 的平方 |
| **Attention 核心 ($QK^\top+AV$)** | $4n^2d$         | 取决于序列长度$n$ 的平方 |
| **LM Head (logits)**                | $2ndV$          | 取决于词表大小$V$        |

**这就是为什么在问题 (e) 中，当你把 $n$ 从 1024 增加到 16384（增加 16 倍）时，注意力核心部分的计算量会因为 $n^2$ 的存在暴增 256 倍，从而反超 FFN 成为最大的负担。**

### (a) 内存占用分析 (Algebraic Expression)

在大模型训练的资源结算中，显存主要由 **静态内存**（模型本身）和 **动态内存**（前向传播产生的激活值）组成。以下计算均基于 Float32 (4 bytes/float)。

#### 1. 静态内存 (Static Memory)

* **参数 (Parameters, $P$)**:
  * **Embedding**: $V \times d$ (词表) + $n \times d$ (位置，如果是绝对位置编码)。
  * **Transformer Blocks ($L$ 层)**:
    * QKV 投影 ($3d^2$) + 输出投影 $W_O$ ($d^2$) = $4d^2$。
    * SwiGLU FFN: $W_1, W_3$ (均为 $d \to 4d$) + $W_2$ ($4d \to d$) = $4d^2 + 4d^2 + 4d^2 = 12d^2$。
    * Norm 层及偏置: 约 $12d$。
  * **公式**: $P \approx 2Vd + L(16d^2)$。
* **梯度 (Gradients, $G$)**: 大小与参数 $P$ 相同。内存 = $4P$ 字节。
* **优化器状态 (Optimizer State, $O_{\text{opt}}$)**: AdamW 存储每个参数的一阶矩 $m$ 和二阶矩 $v$。内存 = $2 \times 4P = 8P$ 字节。
* **静态内存合计**: $M_{\text{static}} = 16P$ 字节。

#### 2. 激活值内存 (Activations, $A$) - 深度拆解

激活值是反向传播计算梯度所必需的中间张量。按照计算流逐项拆解每个 Block 在 **每个 Token** 位置需存储的量：

**A. Attention 机制与 Norm 1**

1. **RMSNorm 1 输入**: 用于计算 Norm 的梯度。 (**$1d$**)
2. **QKV 投影输入**: 即 Norm 后的结果，线性层反向传播需要输入张量。 (**$1d$**)
3. **Q, K 向量本身**: 用于计算 $QK^\top$ 的梯度。 (**$2d$**)
4. **V 向量本身**: 用于计算 $Attention \cdot V$ 的梯度。 (**$1d$**)
5. **注意力得分 (Pre-Softmax)**: 形状为 $(H, n)$，记录当前词对全序列的原始权重。 (**$1Hn$**)
6. **注意力概率 (Post-Softmax)**: 形状为 $(H, n)$，Softmax 梯度的计算必须依赖其输出。 (**$1Hn$**)
7. **Attention 输出**: 即 $W_O$ 投影层的输入。 (**$1d$**)

**B. SwiGLU FFN 与 Norm 2**
8.  **RMSNorm 2 输入**: 保留以计算梯度。 (**$1d$**)
9.  **FFN $W_1$ & $W_3$ 投影输入**: 即 Norm 2 后的结果。 (**$1d$**)
10. **$W_1$ 投影输出**: 即门控支路在进入 SiLU 激活前的值。 (**$4d$**)
11. **$W_3$ 投影输出**: 即值支路的值，待与门控支路点乘。 (**$4d$**)
12. **SiLU 激活输出**: 门控支路激活后的结果，用于计算点乘梯度。 (**$4d$**)
13. **$W_2$ 投影输入**: 两支路合并（逐元素点乘）后的结果，维度为 $4d$。 (**$4d$**)

**汇总 (Per Token):**

* **$d$ 项合计**: $1+1+2+1+1+1+1+4+4+4+4 = \mathbf{24d}$
* **$n$ 项合计**: $1Hn + 1Hn = \mathbf{2Hn}$
* **合计激活值公式**: **$24d + 2Hn$**

#### 3. 总体显存公式 (Bytes)

考虑到全模型 $L$ 层、Batch Size $B$、序列长度 $n$ 以及末端开销：

$$
M_{\text{total}} = 16P + 4 \times B \cdot n \cdot [L(24d + 2Hn) + 2d + V]
$$

---

### (b) GPT-2 XL 实例化与最大 batch_size

**参数**: $d=1600, L=48, n=1024, V=50257, H=25$。硬件：A100-80GB。

1. **静态内存**:
   * $P \approx 2.127 \times 10^9$。
   * $M_{\text{static}} = 16 \times 2.127 \text{ GB} \approx \mathbf{34.03 \text{ GB}}$。
2. **激活值内存 (随 B 变化)**:
   * 代入公式：每个 Batch 激活值 $\approx 4 \times 1024 \times [48(24 \times 1600 + 2 \times 25 \times 1024) + 50257] \approx \mathbf{18.4 \text{ GB}}$。
3. **最大 Batch Size 计算**:
   * 剩余可用显存 = $80 - 34.03 = 45.97 \text{ GB}$。
   * $18.4 \cdot B \le 45.97 \Rightarrow B \le 2.49$。
   * **结论**: 最大 Batch Size 为 **2**（若使用 SwiGLU）。

### (c) AdamW 步运行 FLOPs 解析

**结论**: 每次训练步所需的计算量约为 **$6PBn$**。

#### 为什么是 $6P$？ (详细推导)

在神经网络中，最核心的运算是矩阵乘法 $Y = XW$（其中 $X \in \mathbb{R}^{m \times k}, W \in \mathbb{R}^{k \times n}$）。其浮点运算次数（FLOPs）约为 $2mkn$。

- **前向传播 ($2P$ / token)**:
  对于每个 token ($m=1$)，计算线性层 $Y = XW$ 的运算量是 $2 \times 1 \times k \times n$。你会发现 $k \times n$ 正好是该层的参数量。因此，前向传播总 FLOPs 约等于 $2 \times \text{总参数量} = 2P$。
- **反向传播 ($4P$ / token)**:
  反向传播需要完成两个任务：

  1. **计算对输入的梯度 ($\frac{\partial \mathcal{L}}{\partial X}$)**: 用于向上一层传递。其计算复杂度与前向传播相同 ($2P$)。
  2. **计算对参数的梯度 ($\frac{\partial \mathcal{L}}{\partial W}$)**: 用于更新当前层权重。复杂度同样与前向传播相同 ($2P$)。
     根据题目提示（Backward 是 Forward 的两倍），反向传播总计 $4P$。
- **总计**: $2P (\text{fwd}) + 4P (\text{bwd}) = 6P$。

#### 为什么不用 Transformer 的精细公式？

你提到的“精细表达式”（如 $12Ld^2n + \dots$）主要用于描述注意力机制和投影层的具体实现。但在 **资源结算 (Accounting)** 中，$6P$ 是一个极佳的近似，原因有二：

1. **参数主导**: Transformer 中 99% 的参数都在这些线性层中，$6P$ 已经捕获了绝大多数计算。
2. **忽略项**: Softmax、LayerNorm、残差连接的 FLOPs 占比通常小于 1%，在宏观估算中可以忽略不计。

### (d) A100 训练时长预估

#### 1. 总 Token 数 ($T$) 是怎么来的？

题目给出：[400,000] steps，Batch Size [1024]，序列长度 $n=1024$。

$$
T = \text{Steps} \times B \times n = 400,000 \times 1024 \times 1024 \approx \mathbf{419.4 \times 10^9} \text{ tokens (约 419B)}
$$

#### 2. 总计算量 (Total FLOPs)

$$
F = 6 \cdot P \cdot T = 6 \times (2.127 \times 10^9) \times (419.4 \times 10^9) \approx \mathbf{5.35 \times 10^{21} \text{ FLOPs}}
$$

#### 3. 硬件实际产出 (Effective Throughput)

- **A100 FP32 峰值**: $19.5 \text{ TFLOPS}$。
- **50% MFU (Model FLOPs Utilization)**: 实际每秒运行 $19.5 \times 0.5 = \mathbf{9.75 \text{ TFLOPS}}$。

#### 4. 耗时计算

$$
\text{Time (days)} = \frac{5.35 \times 10^{21}}{9.75 \times 10^{12} \times 86400 \text{ (秒/天)}} \approx \mathbf{6352 \text{ 天}}
$$

*(注：由于题目要求使用 **Float32** 且峰值仅为 **19.5 TFLOPS**，这在单卡 A100 上是非常慢的。在实际生产中，我们会使用 FP16 Tensor Core，峰值可达 312 TFLOPS，时间会缩短 16 倍。但按此题给出的 FP32 条件，结果确实在 6000 天量级)*。

### 修正与改进列表

- **Token 数来源**: 修正了之前对话中模糊的“400B”，明确了由 $Steps \times B \times n$ 严格导出的 **419.4B**。
- **SwiGLU 的 $24d$ 详细推导**: 补充了对 $W_3$ 和 SiLU 激活后张量的解释，这是实现 FFN 部分 $18d$ 存储的关键。
- **FLOPs 简化逻辑**: 解释了为什么 $6P$ 是工业界标准近似，消除了你对“Transformer 精细公式”是否需要叠加的疑虑（即：$6P$ 已经包含了那些精细公式的大部分主要项）。
- **硬件参数对齐**: 严格使用题目给出的 **19.5 TFLOPS (FP32)**，而不是通用认知下的 312 TFLOPS，这解释了为什么最终训练天数如此巨大。

**下一步建议**：如果你需要将这个计算过程写进报告，我可以帮你把上述推导整理成标准的学术 LaTeX 论文格式。需要我这样做吗？
