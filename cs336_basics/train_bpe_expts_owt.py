import argparse
import json
import resource
import time
from pathlib import Path

from cs336_basics.tokenizer import BpeTokenizerTrainer


def _max_rss_gb() -> float:
    self_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_kb = max(self_rss_kb, children_rss_kb)
    return peak_kb / (1024 * 1024)


def _serialize_vocab(vocab: dict[int, bytes], output_path: Path) -> None:
    payload = {str(token_id): token_bytes.decode("latin-1") for token_id, token_bytes in vocab.items()}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_merges(merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    lines = [json.dumps([left.decode("latin-1"), right.decode("latin-1")], ensure_ascii=False) for left, right in merges]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_boundary_token_exists(input_path: Path, boundary_token: str) -> None:
    boundary = boundary_token.encode("utf-8")
    chunk_size = 4 * 1024 * 1024
    overlap = max(1, len(boundary) - 1)

    with input_path.open("rb") as f:
        tail = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            if boundary in data:
                return
            tail = data[-overlap:] if len(data) > overlap else data

    raise ValueError(f"boundary token `{boundary_token}` not found in {input_path}")


def train_bpe_expts_owt(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    vocab_out: Path,
    merges_out: Path,
    workers: int,
    chunks: int,
    boundary_token: str,
) -> dict[str, object]:
    _ensure_boundary_token_exists(input_path=input_path, boundary_token=boundary_token)

    trainer = BpeTokenizerTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_chunks=chunks,
        boundary_token=boundary_token,
    )

    start = time.perf_counter()
    vocab, merges = trainer.train(input_path=input_path, max_workers=workers)
    elapsed_seconds = time.perf_counter() - start
    peak_memory_gb = _max_rss_gb()

    _serialize_vocab(vocab=vocab, output_path=vocab_out)
    _serialize_merges(merges=merges, output_path=merges_out)

    longest_token = max(vocab.values(), key=len)
    stats = {
        "input_path": str(input_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_seconds / 3600,
        "peak_memory_gb": peak_memory_gb,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_utf8": longest_token.decode("utf-8", errors="replace"),
        "longest_token_bytes_latin1": longest_token.decode("latin-1"),
        "vocab_output": str(vocab_out),
        "merges_output": str(merges_out),
        "boundary_token": boundary_token,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--special-token", dest="special_tokens", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--boundary-token", type=str, default="<|endoftext|>")
    parser.add_argument("--output-dir", type=Path, default=Path("cs336_basics/owt"))
    parser.add_argument("--vocab-out", type=Path, default=Path("owt_vocab.json"))
    parser.add_argument("--merges-out", type=Path, default=Path("owt_merges.txt"))
    parser.add_argument("--stats-out", type=Path, default=Path("owt_train_bpe_stats.json"))
    args = parser.parse_args()

    special_tokens = args.special_tokens if args.special_tokens is not None else ["<|endoftext|>"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_out = args.vocab_out if args.vocab_out.is_absolute() else args.output_dir / args.vocab_out
    merges_out = args.merges_out if args.merges_out.is_absolute() else args.output_dir / args.merges_out
    stats_out = args.stats_out if args.stats_out.is_absolute() else args.output_dir / args.stats_out

    stats = train_bpe_expts_owt(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        vocab_out=vocab_out,
        merges_out=merges_out,
        workers=args.workers,
        chunks=args.chunks,
        boundary_token=args.boundary_token,
    )

    stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
