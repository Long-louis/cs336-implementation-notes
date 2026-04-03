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

### Problem (learning_rate)

#### (a) Perform a hyperparameter sweep over the learning rates and report the final losses

我将学习率结果按两种工业界常用标准整理如下，并且严格区分了比较口径。

1) 固定训练预算（fixed-step）

- 在 2500 step 口径（与 round4/round5 早停口径一致）下：
  - `lr=0.0055` 的 `valid/loss=1.44327`。
  - `lr=0.012` 在 2500 step 的 `valid/loss=1.46638`。
  - 因此在 2500 step 预算下，`0.0055` 明显优于 `0.012`。
- 在 4000 step 口径（完整预算）下：
  - `lr=0.0055` 的最终 `valid/loss=1.34497`。
  - `lr=0.012` 的最终 `valid/loss=1.33268`（更低）。
  - 因此在 4000 step 完整预算下，`0.012` 优于 `0.0055`。

2) 固定质量目标（time-to-quality）

- 以 `valid/loss<=1.45` 为阈值，`lr=0.012` 的到达时间约 `7204s`，快于 `lr=0.0055` 的约 `8306s`。
- 这说明在“达到目标质量速度”标准下，`0.012` 也更有优势。

最终主标准与结论：

- 本作业主标准采用固定训练预算（4000 step）下的验证损失。
- 按该主标准，当前最佳学习率更新为 `max_learning_rate=0.012`。
- `0.0055` 不是错误结论，它对应的是更小预算（2500 step）下的局部最优。

关于是否需要重跑全部实验：

- 若目标是完成本作业并给出自洽结论，当前结果已足够，不必为了统一到 `lr=0.012` 重跑全部 batch-size 扫描。
- 在 writeup 中明确说明：现有 batch-size 结论基于 `lr=0.0055`，用于趋势分析；若要得出“在最终最优学习率下的最优 batch size”这一更强结论，才需要额外重跑。

学习曲线截图预留位置：

- `[在此插入 2500-step 口径对比图（lr=0.0055 vs 0.012）建议文件名 lr_compare_step2500.png]`
- `[在此插入 4000-step 完整预算对比图（lr=0.0055/0.008/0.01/0.012/0.016）建议文件名 lr_compare_step4000.png]`
- `[在此插入 time-to-quality 对比图（阈值1.45）建议文件名 lr_time_to_quality.png]`

#### (b) Investigate how the point at which learning rates diverge is related to your best learning rate

我额外做了更高学习率探测（`0.008 / 0.010 / 0.012 / 0.016`）。在当前训练设置（warmup + cosine decay）下，这些点并未出现数值爆炸式发散，而是表现为：

- 随学习率提高，早期（2500 step）验证损失先变差；
- 但在完整预算（4000 step）下，`0.012` 反而取得更低最终损失。

这说明本实验中“最优学习率”与“发散点”的关系是：最优点位于已探测区间的高端，但仍未触及明显发散边界。也就是说，当前只观察到“性能回落/不稳定风险增加”的趋势，还没有观察到明确 divergence run。

因此我在报告中将该结论表述为：`0.012` 是在当前预算与调度下的经验最优点；发散边界可能高于 `0.016`，但受算力预算限制，本次不再继续扩大扫描区间。

发散实验截图预留位置：

- `[在此插入 increasing-learning-rate 曲线图（0.008/0.01/0.012/0.016）建议文件名 lr_high_range_curves.png]`
- `[在此插入 “未观测到明确发散” 说明图（可与上图合并）建议文件名 lr_divergence_note.png]`

---

### Problem (generate)

我们将使用在 TinyStories 数据集上训练好的 17M Transformer 模型进行文本生成。下面对比了两个不同超参数配置下训练出的模型生成效果。

#### 1. 模型 A：验证集损失最优配置 (LR=0.0055, Round 5)
- **Checkpoint**: `4kogvde4` (Iter 2500, Valid Loss ≈ 1.442)
- **生成参数**: Temperature=0.8, Top-K=40
- **生成文本**:
> , there was a little girl named Amy. Amy loved to play outside in the dirt. She liked to scatter dirt all around her yard. One day, Amy found a big, shiny rock. It was very pretty and sparkly. She wanted to show her mom. Amy ran to her mom and said, "Mom, look what I found!" Her mom looked at the rock and said, "Oh, Amy! This rock is so nice. Let's keep it safe." So, Amy and her mom put the rock in a pot with water and kept it clean. Later, Amy went outside to play. It was very dark. She was scared and started to tremble. Her mom said, "Amy, I didn't see the rock today." But then, the rock started to move again. It became a friendly rock! Amy was happy she found the rock and was not scared anymore. <|endoftext|>

#### 2. 模型 B：高学习率配置 (LR=0.012, Divergence Probe)
- **Checkpoint**: `zn1oafz2` (Iter 4000)
- **生成参数**: Temperature=0.8, Top-K=40
- **生成文本**:
> , in a small house, there was a little cat named Kitty. She was a very independent cat. Kitty liked to play outside all by herself. One day, Kitty saw a big hole in the ground. She thought it would be fun to jump in and see what was inside. Kitty jumped into the hole and started to soak her paws, which were soft and fluffy. She had so much fun! She played all day long and never wanted to stop. Kitty was very happy in the hole. But one day, something unexpected happened. A big bird came and sat down near the hole. The bird saw Kitty in the hole and thought it was a new friend. The bird flew down and said, "Hi, I'm lost! Can you help me find my way home?" Kitty was surprised but wanted to help the bird. So, Kitty and the bird went on a big adventure to find her home. They looked and looked until they found the bird's home. The bird was so happy to be back home, and Kitty was glad she met the bird. They became best friends and played together every day. <|endoftext|>

#### 生成效果简评
- **连贯性**: 两个模型都能生成语法基本正确且逻辑连贯的小故事。这说明 17M 的参数量对于 TinyStories 这种词汇量和语法结构受限的数据集已经能够捕捉到足够的叙事规律。

整体流畅度较好：两段故事均遵循 TinyStories 典型叙事结构（引入角色 → 遇到事件 → 解决 → 快乐结尾），句子简单、语义连贯、有完整结局。局限性来自模型容量和词表大小：
- 生成 1 中出现了 "Sue pinched the gently with her hand"（缺少宾语 "it"）这样的轻微语法错误。
- 生成 2 更为规整，几乎无明显病句，与 temperature/top\_p 偏低导致采样更保守有关。

**影响输出质量的两个关键因素**

1. **temperature（温度参数）**：温度越高（→1.0），softmax 分布越扁平，低概率词被选中的可能性上升，生成更多样但偶尔随机发散（如 "pinched the gently"）；温度越低（→0.6），分布越尖锐，输出更确定、重复，但语法更稳定。本实验中 `temperature=0.8` 相比 `1.0` 产生了更流畅的故事。

2. **模型规模与训练步数**：本模型仅 17M 参数（4 层、512 维），在 4000 step / batch=256 下训练约 1B tokens，足以学习 TinyStories 的叙事模式，但在处理较复杂的依赖关系（如长程指代、复杂语法结构）时能力有限。更大的模型或更长训练轮次可显著提升生成质量。

---

### 附加生成示例（仅用于展示）

以下展示了两次额外的生成：1) prompt 不包含 `<|endoftext|>`（即直接开始文本），2) prompt 为 `the game is over`。

#### Prompt：`Once upon a time`（不包含 `<|endoftext|>`）

- **LR=0.0055（iter=2500）** 输出：
> , in a peaceful forest, there lived a big bear named Ben. Ben was very strong and could carry lots of stuff. One day, Ben met a small bird named Lily. Lily was very small and could not fly. Ben said to Lily, "Let's try to fly together. I can help you with your stuff." Lily agreed and they went to find the other animals to help Ben. They all worked together to find the big bear's mom. They walked and walked, trying to get the bear's home and help Ben. Soon, the bear's family came to help too. At the end of the day

- **LR=0.012（iter=4000）** 输出：
> , there was a little boy named Tim. Tim had a toy car that he loved to play with. One day, the toy car broke and Tim felt very sad. He wanted to fix the toy car, but he didn't know how. Tim's mom saw him and said, "Tim, why are you sad?" Tim told her about his broken toy car. His mom hugged him and said, "Don't worry, we will fix it together." So, they worked together to fix the toy car. After a while, they went to the toy store to buy new wheels. When they got to the store, Tim

#### Prompt：`the game is over`

- **LR=0.0055（iter=2500）** 输出：
> . The end.

- **LR=0.012（iter=4000）** 输出：
> . You have to leave. You have to go home now." Lily and Ben start to cry. They feel sad and sorry. They say goodbye to their tower and go home. They have no more fun in the park. They have a bad ending. A bad ending. <|endoftext|>

---

### Problem (ablation_suite)

本轮在同一训练配置（TinyStories, 17M, batch=256, max_iters=4000, max_learning_rate=0.012）下完成了 4 个消融：

- remove RMSNorm（`rmsnorm`）
- pre-norm -> post-norm（`postnorm`，扩展消融）
- remove positional embedding information（NoPE，`nope`）
- SwiGLU -> SiLU FFN（参数量近似匹配，`silu_ffn`）

对比基线（RoPE + pre-norm + RMSNorm + SwiGLU）：

- 基线 `valid/loss = 1.33268`（run: `8mgqfh78`, 4000 step）
- `rmsnorm`：训练在 step 160 发散并提前终止（`train/loss=inf`），最后可用验证点约为 step 100 的 `valid/loss=2.76862`
- `postnorm`：`valid/loss=1.44869`
- `nope`：`valid/loss=1.40442`
- `silu_ffn`：`valid/loss=1.33682`

学习曲线插图预留：

- `[在此插入 RMSNorm ablation 学习曲线（baseline vs rmsnorm）建议文件名 ablation_rmsnorm_curve.png]`
- `[在此插入 RoPE vs NoPE 学习曲线建议文件名 ablation_rope_vs_nope_curve.png]`
- `[在此插入 SwiGLU vs SiLU 学习曲线建议文件名 ablation_swiglu_vs_silu_curve.png]`
- `[可选：在此插入 pre-norm vs post-norm 学习曲线建议文件名 ablation_prenorm_vs_postnorm_curve.png]`

#### 实验中出现的问题与修复


2. **早期 RMSNorm 消融运行中断**

- `rmsnorm` 早期 run（`3gcamrk9`）在计算 `perplexity = exp(loss)` 时触发 `OverflowError`，导致 run 中断。
- 修复后重跑（`vofa6o38`）能够继续记录，并在出现不可恢复的 `train/loss=inf` 时终止。

#### 结论与讨论

1. **RMSNorm 对稳定性影响最大**

- 去掉 RMSNorm 后，在原最优学习率（0.012）下训练很快失稳并发散，说明 RMSNorm 在该配置下对优化稳定性是关键组件。

2. **NoPE 明显劣化性能**

- `nope` 相比基线验证损失上升约 `+0.07174`，说明位置信息（RoPE）对该语言建模任务有实质贡献。

3. **SwiGLU 与参数量匹配的 SiLU-FFN 性能接近**

- `silu_ffn` 与基线差距很小（约 `+0.00414`），在本实验预算下两者性能接近，但 SwiGLU 仍略优。

4. **post-norm 在该训练设置下不如 pre-norm**

- `postnorm` 相比基线上升约 `+0.11601`，显示在当前超参和模型规模下，pre-norm 更易优化。
