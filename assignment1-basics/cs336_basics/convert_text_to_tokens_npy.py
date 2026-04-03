import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
from typing import Any

import numpy
from tqdm import tqdm

from cs336_basics.tokenizer import BpeTokenizer


_WORKER_TOKENIZER: BpeTokenizer | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将文本语料转换为 token-id 的 .npy 文件")
    parser.add_argument("--input", type=Path, required=True, help="输入文本文件路径（.txt）")
    parser.add_argument("--vocab", type=Path, required=True, help="词表 json 路径")
    parser.add_argument("--merges", type=Path, required=True, help="merges 文本路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 .npy 路径")
    parser.add_argument("--special-token", dest="special_tokens", action="append", default=[])
    parser.add_argument("--dtype", type=str, choices=["uint16", "int32", "int64"], default="uint16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--batch-lines", type=int, default=4096)
    return parser.parse_args()


def build_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> BpeTokenizer:
    if not vocab_path.exists():
        raise ValueError(f"词表文件不存在: {vocab_path}")
    if not merges_path.exists():
        raise ValueError(f"merges 文件不存在: {merges_path}")
    return BpeTokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "uint16":
        return numpy.uint16
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


def write_tokens_to_npy(input_path: Path, tokenizer: BpeTokenizer, output_path: Path, token_count: int, dtype: Any) -> None:
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


def _initialize_worker(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BpeTokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )


def _iter_line_batches(input_path: Path, batch_lines: int) -> tuple[list[bytes], int]:
    if batch_lines <= 0:
        raise ValueError(f"batch_lines 必须大于 0，当前为: {batch_lines}")

    current_batch: list[bytes] = []
    current_batch_bytes = 0

    with input_path.open("rb") as input_file:
        for raw_line in input_file:
            current_batch.append(raw_line)
            current_batch_bytes += len(raw_line)

            if len(current_batch) >= batch_lines:
                yield current_batch, current_batch_bytes
                current_batch = []
                current_batch_bytes = 0

    if current_batch:
        yield current_batch, current_batch_bytes


def _encode_batch(batch_lines: list[bytes], batch_bytes: int, dtype_name: str) -> tuple[int, int, bytes]:
    if _WORKER_TOKENIZER is None:
        raise ValueError("worker tokenizer 未初始化")

    numpy_dtype = resolve_dtype(dtype_name)
    token_ids: list[int] = []
    for raw_line in batch_lines:
        token_ids.extend(_WORKER_TOKENIZER.encode(raw_line.decode("utf-8")))

    if not token_ids:
        return batch_bytes, 0, b""

    token_array = numpy.asarray(token_ids, dtype=numpy_dtype)
    return batch_bytes, int(token_array.shape[0]), token_array.tobytes()


def _encode_batch_from_tuple(args: tuple[list[bytes], int, str]) -> tuple[int, int, bytes]:
    return _encode_batch(*args)


def cleanup_legacy_temp_artifacts(output_path: Path) -> None:
    legacy_parts_dir = output_path.parent / f".{output_path.stem}_parts"
    if legacy_parts_dir.exists():
        shutil.rmtree(legacy_parts_dir)

    temp_bin_path = output_path.with_suffix(output_path.suffix + ".tmp.bin")
    if temp_bin_path.exists():
        temp_bin_path.unlink()


def convert_text_to_npy_parallel(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype: str,
    workers: int,
    chunks: int,
    batch_lines: int,
) -> int:
    if not input_path.exists():
        raise ValueError(f"输入文件不存在: {input_path}")

    cleanup_legacy_temp_artifacts(output_path)

    file_size = input_path.stat().st_size
    numpy_dtype = resolve_dtype(dtype)
    temp_bin_path = output_path.with_suffix(output_path.suffix + ".tmp.bin")

    total_tokens = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encode_progress = tqdm(
        total=file_size,
        desc=f"编码 {input_path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )

    with temp_bin_path.open("wb") as temp_bin_file:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(vocab_path, merges_path, special_tokens),
        ) as executor:
            batch_iterator = (
                (batch_lines_bytes, batch_bytes, dtype)
                for batch_lines_bytes, batch_bytes in _iter_line_batches(input_path=input_path, batch_lines=batch_lines)
            )
            for processed_bytes, batch_token_count, batch_payload in executor.map(_encode_batch_from_tuple, batch_iterator, chunksize=1):
                if batch_payload:
                    temp_bin_file.write(batch_payload)
                total_tokens += batch_token_count
                encode_progress.update(processed_bytes)

    encode_progress.close()

    if total_tokens <= 0:
        raise ValueError(f"没有从输入文件中得到任何 token: {input_path}")

    output_array = numpy.lib.format.open_memmap(output_path, mode="w+", dtype=numpy_dtype, shape=(total_tokens,))
    temp_array = numpy.memmap(temp_bin_path, mode="r", dtype=numpy_dtype, shape=(total_tokens,))

    write_progress = tqdm(
        total=total_tokens,
        desc=f"写入 {output_path.name}",
        unit="tok",
        dynamic_ncols=True,
    )
    output_array[:] = temp_array[:]
    write_progress.update(total_tokens)
    output_array.flush()
    write_progress.close()

    del temp_array
    del output_array
    temp_bin_path.unlink()
    return total_tokens


def main() -> None:
    args = parse_args()

    if not args.vocab.exists():
        raise ValueError(f"词表文件不存在: {args.vocab}")
    if not args.merges.exists():
        raise ValueError(f"merges 文件不存在: {args.merges}")

    if args.workers <= 0:
        raise ValueError(f"workers 必须大于 0，当前为: {args.workers}")
    if args.chunks <= 0:
        raise ValueError(f"chunks 必须大于 0，当前为: {args.chunks}")
    if args.batch_lines <= 0:
        raise ValueError(f"batch_lines 必须大于 0，当前为: {args.batch_lines}")

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
        batch_lines=args.batch_lines,
    )
    print(f"token_count={token_count}")

    print("转换完成")


if __name__ == "__main__":
    main()
