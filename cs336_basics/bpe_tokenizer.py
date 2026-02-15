import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from turtle import left
from typing import BinaryIO, Iterator, Optional

import regex as re
import tqdm  # 建议安装 tqdm 用于显示进度条: pip install tqdm


class BpeTokenizerTrainer:
    def __init__(self, input_path: str| os.PathLike, vocab_size: int, special_tokens: Optional[list[str]] = None, boundary_token: str = "<|endoftext|>", num_chunks: int = 4):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.vocab = {}
        for i in range(0, 256):
            self.vocab[i] = bytes([i])
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        self.merges:list[tuple[bytes, bytes]] = []
        self.boundary_token = boundary_token
        self.num_chunks = num_chunks
        
    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _load_and_split(self, num_chunks: int, boundary_token: str = "<|endoftext|>")-> Iterator[bytes]:
        with open(self.input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, num_chunks, boundary_token.encode('utf-8'))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start)
                yield chunk

    @staticmethod
    def _pretokenize_and_count(byte_sequence: bytes, special_tokens: Optional[list[str]] = None) -> dict[tuple[bytes, ...], int]:
        """
        对输入的字节序列进行预分词，并统计每个token序列的出现次数。
        Args:
            byte_sequence: 输入的字节序列。
            special_tokens: 需要特殊处理的token字符串列表。
        Returns:
            Counter对象，键为token字节元组，值为出现次数。
        """
        # 将字节序列解码为字符串
        text = byte_sequence.decode('utf-8')
        # 如果有特殊token，则先对其进行转义，构造正则表达式，按特殊token切分文本
        if special_tokens:
            escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
            special_token_pattern = "|".join(escaped)
            segments = re.split(special_token_pattern, text)
        else:
            segments = [text]

        # 定义分词正则表达式，兼容英文缩写、字母、数字、符号和空白
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 生成token序列，每个token用utf-8编码后再转为单字节元组
        token_generator = (
            tuple(bytes([b]) for b in token_str.encode('utf-8'))  # 单元素元组，替代冗余的 tuple([b])
            for segment in segments  # 遍历每个片段
            if segment  # 跳过空片段
            for token_str in re.findall(pattern, segment)  # 用正则表达式分词
        )

        # 统计每种token序列的出现次数
        return Counter(token_generator)

    def _merge(self, frequency_table: dict[tuple[bytes, ...], int]) -> tuple[dict[int, bytes], dict[tuple[bytes, ...], int]]:

        pair_counts = Counter()

        for token_seq, freq in frequency_table.items():
            # 跳过长度小于2的token序列
            if len(token_seq) < 2:
                continue
            for i in range(len(token_seq) - 1):
                left, right = token_seq[i], token_seq[i + 1]
                pair = (left, right)
                pair_counts[pair] += freq
            
        if not pair_counts:
            return self.vocab, frequency_table
        # 保证max的稳定性,主键使用freq，次键使用字典序
        best_pair = max(pair_counts, key=lambda x: (pair_counts[x],x))
        new_token_id = len(self.vocab)
        new_token_bytes = best_pair[0] + best_pair[1]
        self.merges.append(best_pair)
        self.vocab[new_token_id] = new_token_bytes
        # TODO 遍历算法有优化空间？
        new_frequency_table = {}
        for token_seq, freq in frequency_table.items():
            if len(token_seq) < 2:
                new_frequency_table[token_seq] = freq
                continue

            # 合并旧frequency table中的best_pair
            new_token_seq = []
            i = 0
            while i < len(token_seq):
                if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
                    new_token_seq.append(new_token_bytes)
                    i += 2
                else:
                    new_token_seq.append(token_seq[i])
                    i += 1
            new_frequency_table[tuple(new_token_seq)] = freq
        return self.vocab, new_frequency_table

    def train(self, max_workers: int = 8)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''
        Returns:
        - vocab: A dictionary mapping token IDs to byte sequences.
        - merges: A list of tuples representing the BPE merges.
        '''
        # 1. 并行 Pre-tokenization
        print("Starting pre-tokenization...")
        
        # 准备数据生成器
        chunk_generator = self._load_and_split(num_chunks=self.num_chunks, boundary_token=self.boundary_token)
        
        # 初始化总计数器
        frequency_table = Counter()

        # 使用 ProcessPoolExecutor 进行并行处理
        # max_workers 默认为 CPU 核心数
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 使用 partial 固定 special_tokens 参数，因为 map 只能传一个变动参数(chunk)
            worker_func = partial(self._pretokenize_and_count, special_tokens=self.special_tokens)
            
            # executor.map 会保持顺序返回结果，这对计数无所谓，但很方便
            # 此时 chunk_generator 里的数据会被分发给不同的进程
            results = executor.map(worker_func, chunk_generator)
            
            for chunk_counts in tqdm.tqdm(results, total=self.num_chunks, desc="Counting tokens"):
                frequency_table += chunk_counts
                
        print(f"Pre-tokenization complete. Initial vocab size: {len(frequency_table)}")

        # 2. 循环 Merge (这部分很难并行，因为每一步依赖上一步的结果)
        print("Starting BPE merge...")
        pbar = tqdm.tqdm(total=self.vocab_size - len(self.vocab), desc="Merging")
        
        while len(self.vocab) < self.vocab_size:
            # 记录当前词表大小用于判断是否还有 merge 发生
            current_vocab_len = len(self.vocab)
            
            self.vocab, frequency_table = self._merge(frequency_table)
            
            # 如果一轮 merge 下来词表没变大，说明没东西可合了，提前退出
            if len(self.vocab) == current_vocab_len:
                break
                
            pbar.update(1)
            
        pbar.close()
        
        return self.vocab, self.merges

class BpeTokenizer:
    ...