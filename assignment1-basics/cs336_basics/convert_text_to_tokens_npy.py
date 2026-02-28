import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy

from cs336_basics.tokenizer import BpeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将文本语料转换为 token-id 的 .npy 文件")
    parser.add_argument("--input", type=Path, required=True, help="输入文本文件路径（.txt）")
    parser.add_argument("--vocab", type=Path, required=True, help="词表 json 路径")
    parser.add_argument("--merges", type=Path, required=True, help="merges 文本路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 .npy 路径")
    parser.add_argument("--special-token", dest="special_tokens", action="append", default=[])
    parser.add_argument("--dtype", type=str, choices=["int32", "int64"], default="int32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=32)
    return parser.parse_args()


def build_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> BpeTokenizer:
    if not vocab_path.exists():
        raise ValueError(f"词表文件不存在: {vocab_path}")
    if not merges_path.exists():
        raise ValueError(f"merges 文件不存在: {merges_path}")
    return BpeTokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)


def resolve_dtype(dtype_name: str) -> numpy.dtype:
    if dtype_name == "int32":
        return numpy.int32
    if dtype_name == "int64":
        return numpy.int64
    raise ValueError(f"不支持的 dtype: {dtype_name}")


def count_tokens(input_path: Path, tokenizer: BpeTokenizer) -> int:
    if not input_path.exists():
        raise ValueError(f"输入文件不存在: {input_path}")

    token_count = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            token_count += len(tokenizer.encode(line))

    if token_count <= 0:
        raise ValueError(f"没有从输入文件中得到任何 token: {input_path}")
    return token_count


def write_tokens_to_npy(input_path: Path, tokenizer: BpeTokenizer, output_path: Path, token_count: int, dtype: numpy.dtype) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = numpy.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(token_count,))

    cursor = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            token_ids = tokenizer.encode(line)
            if not token_ids:
                continue
            next_cursor = cursor + len(token_ids)
            array[cursor:next_cursor] = numpy.asarray(token_ids, dtype=dtype)
            cursor = next_cursor

    if cursor != token_count:
        raise ValueError(f"写入 token 数与预估不一致: cursor={cursor}, token_count={token_count}")

    array.flush()


def write_tokens_to_npy_one_pass(input_path: Path, tokenizer: BpeTokenizer, output_path: Path, dtype: numpy.dtype) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_bin_path = output_path.with_suffix(output_path.suffix + ".tmp.bin")

    token_count = 0
    with input_path.open("r", encoding="utf-8") as in_f:
        with temp_bin_path.open("wb") as out_f:
            for line in in_f:
                token_ids = tokenizer.encode(line)
                if not token_ids:
                    continue
                token_array = numpy.asarray(token_ids, dtype=dtype)
                token_array.tofile(out_f)
                token_count += int(token_array.shape[0])

    if token_count <= 0:
        raise ValueError(f"没有从输入文件中得到任何 token: {input_path}")

    mmap_array = numpy.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(token_count,))
    bin_array = numpy.memmap(temp_bin_path, mode="r", dtype=dtype, shape=(token_count,))
    mmap_array[:] = bin_array[:]
    mmap_array.flush()

    del bin_array
    del mmap_array
    temp_bin_path.unlink()
    return token_count


def _find_chunk_boundaries(input_path: Path, num_chunks: int) -> list[int]:
    file_size = input_path.stat().st_size
    if num_chunks <= 1:
        return [0, file_size]

    boundaries = [0]
    with input_path.open("rb") as f:
        for i in range(1, num_chunks):
            target = (file_size * i) // num_chunks
            f.seek(target)
            f.readline()
            boundaries.append(f.tell())
    boundaries.append(file_size)
    return sorted(set(boundaries))


def _process_chunk(
    input_path: Path,
    start: int,
    end: int,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    part_index: int,
    temp_dir: Path,
    dtype: str,
) -> tuple[Path, int]:
    tokenizer = BpeTokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )
    numpy_dtype = resolve_dtype(dtype)
    part_path = temp_dir / f"part_{part_index:04d}.bin"

    token_count = 0
    with input_path.open("rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)

    with part_path.open("wb") as out_f:
        for line in lines:
            token_ids = tokenizer.encode(line)
            if not token_ids:
                continue
            token_array = numpy.asarray(token_ids, dtype=numpy_dtype)
            token_array.tofile(out_f)
            token_count += int(token_array.shape[0])

    return part_path, token_count


def _process_chunk_from_tuple(args: tuple) -> tuple[Path, int]:
    return _process_chunk(*args)


def convert_text_to_npy_parallel(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype: str,
    workers: int,
    chunks: int,
) -> int:
    if not input_path.exists():
        raise ValueError(f"输入文件不存在: {input_path}")

    boundaries = _find_chunk_boundaries(input_path=input_path, num_chunks=chunks)
    ranges: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end > start:
            ranges.append((start, end))

    temp_dir = output_path.parent / f".{output_path.stem}_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    futures_args = [
        (
            input_path,
            start,
            end,
            vocab_path,
            merges_path,
            special_tokens,
            idx,
            temp_dir,
            dtype,
        )
        for idx, (start, end) in enumerate(ranges)
    ]

    results: list[tuple[Path, int]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_process_chunk_from_tuple, futures_args):
            results.append(result)

    total_tokens = sum(count for _, count in results)
    if total_tokens <= 0:
        raise ValueError(f"没有从输入文件中得到任何 token: {input_path}")

    numpy_dtype = resolve_dtype(dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_array = numpy.lib.format.open_memmap(output_path, mode="w+", dtype=numpy_dtype, shape=(total_tokens,))

    cursor = 0
    for part_path, count in sorted(results, key=lambda item: item[0].name):
        if count == 0:
            part_path.unlink()
            continue
        part_array = numpy.memmap(part_path, mode="r", dtype=numpy_dtype, shape=(count,))
        next_cursor = cursor + count
        output_array[cursor:next_cursor] = part_array
        cursor = next_cursor
        del part_array
        part_path.unlink()

    output_array.flush()
    temp_dir.rmdir()
    return total_tokens


def main() -> None:
    args = parse_args()

    if not args.vocab.exists():
        raise ValueError(f"词表文件不存在: {args.vocab}")
    if not args.merges.exists():
        raise ValueError(f"merges 文件不存在: {args.merges}")

    print(f"[1/1] 并行编码并写入 npy: {args.input} -> {args.output}")
    token_count = convert_text_to_npy_parallel(
        input_path=args.input,
        output_path=args.output,
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=args.special_tokens,
        dtype=args.dtype,
        workers=args.workers,
        chunks=args.chunks,
    )
    print(f"token_count={token_count}")

    print("转换完成")


if __name__ == "__main__":
    main()
