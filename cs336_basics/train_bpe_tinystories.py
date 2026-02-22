import argparse
import json
import resource
import time
from pathlib import Path

from cs336_basics.bpe_tokenizer import BpeTokenizerTrainer


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


def train_bpe_tinystories(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    vocab_out: Path,
    merges_out: Path,
    workers: int,
    chunks: int,
) -> dict[str, object]:
    trainer = BpeTokenizerTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_chunks=chunks,
        boundary_token="<|endoftext|>",
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
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", dest="special_tokens", action="append", default=["<|endoftext|>"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--vocab-out", type=Path, default=Path("tinystories_vocab.json"))
    parser.add_argument("--merges-out", type=Path, default=Path("tinystories_merges.txt"))
    parser.add_argument("--stats-out", type=Path, default=Path("tinystories_train_bpe_stats.json"))
    args = parser.parse_args()

    stats = train_bpe_tinystories(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        vocab_out=args.vocab_out,
        merges_out=args.merges_out,
        workers=args.workers,
        chunks=args.chunks,
    )

    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
