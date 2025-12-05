import os
from typing import BinaryIO, Iterator
import regex as re


class BpeTokenizerTrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str] = None, boundary_token: str = "<|endoftext|>", num_chunks: int = 4):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.vocab = {}
        for i in range(0, 256):
            self.vocab[i] = bytes([i])
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
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
            boundries = self._find_chunk_boundaries(f, num_chunks, boundary_token.encode('utf-8'))
            for start, end in zip(boundries[:-1], boundries[1:]):
                f.seek(start)
                chunk = f.read(end - start)
                yield chunk

    def _pretokenize_and_count(self, byte_sequence: bytes) -> dict[tuple[bytes, bytes], int]:
        text = byte_sequence.decode('utf-8')
        # escape special tokens
        if self.special_tokens:
            escaped = [re.escape(token) for token in self.special_tokens]
            split_pattern = "|".join(escaped)
            text = " ".join(re.split(split_pattern, text))
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = re.finditer(pattern, text)
        frequency_table = {}
        for token in tokens:
            token_bytes = tuple(token.group(0).encode('utf-8'))
            if token_bytes in frequency_table:
                frequency_table[token_bytes] += 1
            else:
                frequency_table[token_bytes] = 1
        return frequency_table

    def _merge(self, frequency_table: dict[tuple[bytes, bytes], int], merge_list:list[tuple[bytes, bytes]]) -> tuple[dict[int, bytes], dict[tuple[bytes, bytes], int]]:
        pair_counts = {}
        for token_seq, freq in frequency_table.items():
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += freq
                else:
                    pair_counts[pair] = freq
        if not pair_counts:
            return self.vocab, frequency_table
        # 更明确的写法
        # best_pair = max(pair_counts.keys(), key=lambda x:pair_counts[x])
        best_pair = max(pair_counts, key=pair_counts.get)
        new_token_id = len(self.vocab)
        new_token_bytes = best_pair[0] + best_pair[1]
        merge_list.append(best_pair)
        self.vocab[new_token_id] = new_token_bytes
        # TODO 遍历算法有优化空间？
        new_frequency_table = {}
        for token_seq, freq in frequency_table.items():
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

    def train(self)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''
        Returns:
        - vocab: A dictionary mapping token IDs to byte sequences.
        - merges: A list of tuples representing the BPE merges.
        '''
        chunk_generator = self._load_and_split(num_chunks=self.num_chunks, boundary_token=self.boundary_token)
        frequency_table = {}
        for chunk in chunk_generator:
            chunk_freq_table = self._pretokenize_and_count(chunk)
            for token_seq, freq in chunk_freq_table.items():
                if token_seq in frequency_table:
                    frequency_table[token_seq] += freq
                else:
                    frequency_table[token_seq] = freq
        while len(self.vocab) < self.vocab_size:
            self.vocab, frequency_table = self._merge(frequency_table, self.vocab, self.merges)
        return self.vocab, self.merges

class BpeTokenizer:
    ...