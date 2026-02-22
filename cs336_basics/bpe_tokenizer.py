import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import BinaryIO, Iterator, Optional

import regex as re
import tqdm  # 建议安装 tqdm 用于显示进度条: pip install tqdm


class BpeTokenizerTrainer:
    """
    BPE 分词器训练器，用于从语料库中提取字节对编码并训练词表。
    """
    def __init__(self, vocab_size: int, special_tokens: Optional[list[str]] = None, boundary_token: str = "<|endoftext|>", num_chunks: int = 4):
        """
        初始化 BPE 训练器。

        Args:
            vocab_size: 目标词表大小。
            special_tokens: 特殊 token 列表。
            boundary_token: 语料库中文档的边界标识，用于切分文件块进行并行处理。
            num_chunks: 切分文件的块数。
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.boundary_token = boundary_token
        self.num_chunks = num_chunks
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]

    def _initialize_training_state(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        初始化单次训练的运行时状态。
        """
        vocab: dict[int, bytes] = {}
        for i in range(0, 256):
            vocab[i] = bytes([i])
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in vocab.values():
                vocab[len(vocab)] = token_bytes
        merges: list[tuple[bytes, bytes]] = []
        return vocab, merges
        
    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        寻找文件中的块边界，以便独立统计每个块。
        如果边界恰好落在特殊 token 中间，会向后偏移以确保完整性。

        Args:
            file: 二进制文件句柄。
            desired_num_chunks: 希望切分的块数。
            split_special_token: 用于切分的特殊 token 字节。

        Returns:
            递增的字节偏移量列表（包含 0 和文件末尾）。
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # 获取文件总长度
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # 初始均匀分布的边界猜测点
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # 每次预读 4KB 查找特殊 token

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)

                # 到达文件末尾
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 在当前小块中查找特殊 token 以确定真实的块边界
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # 确保边界唯一且有序
        return sorted(set(chunk_boundaries))

    def _load_and_split(self, input_path: str | os.PathLike, num_chunks: int, boundary_token: str = "<|endoftext|>")-> Iterator[bytes]:
        """
        将输入文件切分为多个字节块进行迭代读取。

        Args:
            num_chunks: 切分块数。
            boundary_token: 边界 token。

        Yields:
            文件的字节块。
        """
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, num_chunks, boundary_token.encode('utf-8'))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start)
                yield chunk

    @staticmethod
    def _pretokenize_and_count(byte_sequence: bytes, special_tokens: Optional[list[str]] = None) -> dict[tuple[bytes, ...], int]:
        """
        对输入的字节序列进行预分词，并统计每个 token 序列（字符序列）的出现次数。

        Args:
            byte_sequence: 输入的字节序列。
            special_tokens: 特殊 token 列表。

        Returns:
            Counter 对象，键为由单字节组成的元组 (tuple[bytes, ...])，值为出现频率。
        """
        # 将字节序列解码为字符串（假设为 utf-8）
        text = byte_sequence.decode('utf-8')
        # 如果提供了特殊 token，则使用它们切分文本
        if special_tokens:
            escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
            special_token_pattern = "|".join(escaped)
            segments = re.split(special_token_pattern, text)
        else:
            segments = [text]

        # GPT-2 风格的预分词正则表达式
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 将每个单词片段进一步按单字节分解为元组序列
        token_generator = (
            tuple(bytes([b]) for b in token_str.encode('utf-8'))
            for segment in segments
            if segment
            for token_str in re.findall(pattern, segment)
        )

        return Counter(token_generator)

    def _get_initial_stats(self, frequency_table: dict[tuple[bytes, ...], int]) -> tuple[Counter, dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
        """
        初始化对频率统计和倒排索引。
        """
        pair_counts = Counter()
        pair_to_words = {}
        for token_seq, freq in frequency_table.items():
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] += freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(token_seq)
        return pair_counts, pair_to_words

    @staticmethod
    def _merge(
        frequency_table: dict[tuple[bytes, ...], int],
        pair_counts: Counter,
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
    ) -> bool:
        """
        执行一轮增量 BPE 合并操作。
        """
        if not pair_counts:
            return False

        # 1. 选取频率最高的对。如果频率相同，则按字节序降序排列（为了稳定性）
        best_pair: tuple[bytes, bytes] = max(pair_counts, key=lambda x: (pair_counts[x], x))
        
        # 如果最高频率已为 0，停止
        if pair_counts[best_pair] <= 0:
            return False

        new_token_id = len(vocab)
        new_token_bytes = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[new_token_id] = new_token_bytes

        # 2. 增量更新：只处理包含 best_pair 的单词序列
        # 注意：由于 frequency_table 的键是元组且不可变，我们要通过删除旧键增加新键的方式更新。
        affected_words = list(pair_to_words.get(best_pair, set()))
        
        for old_seq in affected_words:
            if old_seq not in frequency_table:
                continue
                
            freq = frequency_table[old_seq]
            
            # --- 这一步非常关键：在合并前，先移除该词对所有计数器的旧贡献 ---
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i + 1])
                pair_counts[p] -= freq
                if p in pair_to_words:
                    pair_to_words[p].discard(old_seq)
            
            # 执行合并逻辑
            new_seq_list = []
            i = 0
            while i < len(old_seq):
                if i < len(old_seq) - 1 and (old_seq[i], old_seq[i + 1]) == best_pair:
                    new_seq_list.append(new_token_bytes)
                    i += 2
                else:
                    new_seq_list.append(old_seq[i])
                    i += 1
            new_seq = tuple(new_seq_list)
            
            # 更新词频表
            del frequency_table[old_seq]
            frequency_table[new_seq] = freq
            
            # --- 合并后，重新加上该词新序列带来的新贡献 ---
            for i in range(len(new_seq) - 1):
                p = (new_seq[i], new_seq[i + 1])
                pair_counts[p] += freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(new_seq)

        # 清理计数为 0 的项，防止 max() 变慢 (每 50 轮清理一次以减少开销)
        if len(merges) % 50 == 0:
            for p in list(pair_counts.keys()):
                if pair_counts[p] <= 0:
                    del pair_counts[p]

        return True

    def train(self, input_path: str | os.PathLike, max_workers: int = 8)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        开始训练 BPE 词表。

        Args:
            input_path: 输入语料的文件路径。
            max_workers: 并行处理的最大进程数。

        Returns:
            tuple: 包含两个元素的元组:
                - **vocab** (dict[int, bytes]): 映射 token ID 到字节序列的字典。
                - **merges** (list[tuple[bytes, bytes]]): 合并记录列表。
        """
        vocab, merges = self._initialize_training_state()

        # 1. 并行预分词 (Parallel Pre-tokenization)
        print("Starting pre-tokenization...")
        
        # 加载语料并切分
        chunk_generator = self._load_and_split(input_path=input_path, num_chunks=self.num_chunks, boundary_token=self.boundary_token)
        
        # 初始化总频率计数表
        frequency_table = Counter()

        # 使用进程池并行执行预分词逻辑
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 使用 partial 固定特殊 token 参数
            worker_func = partial(self._pretokenize_and_count, special_tokens=self.special_tokens)
            
            # 分发任务并收集结果
            results = executor.map(worker_func, chunk_generator)
            
            for chunk_counts in tqdm.tqdm(results, total=self.num_chunks, desc="Counting tokens"):
                frequency_table += chunk_counts
                
        print(f"Pre-tokenization complete. Initial unique tokens: {len(frequency_table)}")

        # 2. 迭代合并 (Iterative Merging)
        print("Starting BPE merge...")
        pbar = tqdm.tqdm(total=self.vocab_size - len(vocab), desc="Merging")
        
        # 初始化持久化的状态：计数器和倒排索引
        pair_counts, pair_to_words = self._get_initial_stats(frequency_table)

        while len(vocab) < self.vocab_size:
            current_vocab_len = len(vocab)
            
            # 执行增量合并
            merged = self._merge(frequency_table, pair_counts, pair_to_words, vocab, merges)
            
            # 如果无法进一步合并（频率表已空或无重复对），则停止
            if (not merged) or len(vocab) == current_vocab_len:
                break
                
            pbar.update(1)
            
        pbar.close()
        
        return vocab, merges

class BpeTokenizer:
    """
    BPE 分词器，用于对文本进行编码和解码。
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        由给定词表、merge 列表和（可选）特殊 token 构造 tokenizer。

        Args:
            vocab: 词表映射，键为 token id，值为 token 对应的 bytes。
            merges: BPE 训练得到的 merge 序列，按创建顺序排列。
            special_tokens: 可选特殊 token 列表。
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BpeTokenizer":
        """
        从序列化的 vocab 与 merges 文件构造 tokenizer。

        Args:
            vocab_filepath: 词表文件路径。
            merges_filepath: merges 文件路径。
            special_tokens: 可选特殊 token 列表。

        Returns:
            BpeTokenizer: 构造得到的 tokenizer。
        """
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            for token_id_str, token_bytes_latin1 in vocab_json.items():
                token_id = int(token_id_str)
                token_bytes = token_bytes_latin1.encode("latin-1")
                vocab[token_id] = token_bytes

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                left_str, right_str = json.loads(line.strip())
                left_bytes = left_str.encode("latin-1")
                right_bytes = right_str.encode("latin-1")
                merges.append((left_bytes, right_bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token id 序列。

        Args:
            text: 输入字符串。

        Returns:
            list[int]: 编码后的 token id 序列。
        """
        if text == "":
            return []

        bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # 与训练阶段保持一致的 GPT-2 预分词正则
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 先按 special token 切分，special token 不参与普通 BPE merge
        if self.special_tokens:
            escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            # 注意：这里的切分模式会保留分隔符（特殊 token）作为独立的 segment，方便后续处理
            special_pattern = f"({'|'.join(escaped)})"
            segments = re.split(special_pattern, text)
        else:
            segments = [text]

        encoded_ids: list[int] = []

        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens:
                special_bytes = segment.encode("utf-8")
                encoded_ids.append(bytes_to_id[special_bytes])
                continue

            # 在每个 pre-token 内独立进行 merge，不跨边界
            for pre_token in re.findall(pattern, segment):
                # 为什么bytes([b])的b要用[]包裹？因为我们需要将每个单字节转换为一个长度为1的bytes对象，而bytes([b])正好可以实现这个功能。直接使用bytes(b)会将整数b解释为一个长度为b的全零字节序列，这不是我们想要的。
                token_sequence = [bytes([b]) for b in pre_token.encode("utf-8")]

                # 按训练时 merge 创建顺序依次应用
                for left, right in self.merges:
                    if len(token_sequence) < 2:
                        break

                    merged_sequence: list[bytes] = []
                    i = 0
                    while i < len(token_sequence):
                        if i < len(token_sequence) - 1 and token_sequence[i] == left and token_sequence[i + 1] == right:
                            merged_sequence.append(left + right)
                            i += 2
                        else:
                            merged_sequence.append(token_sequence[i])
                            i += 1
                    token_sequence = merged_sequence

                for token_bytes in token_sequence:
                    encoded_ids.append(bytes_to_id[token_bytes])

        return encoded_ids




    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        对字符串可迭代对象进行惰性编码，逐个产出 token id。

        Args:
            iterable: 字符串可迭代对象（如文件句柄逐行迭代）。

        Yields:
            int: 编码后的 token id。
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将 token id 序列解码为字符串。

        Args:
            ids: token id 序列。

        Returns:
            str: 解码后的文本。
        """
        bytes_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        return bytes_sequence.decode("utf-8", errors="replace")

