"""文本生成脚本：对比不同 checkpoint 的生成质量。"""
import argparse
from pathlib import Path

import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.nn.decoding import generate

VOCAB_PATH   = Path("data/tinystories/tinystories_vocab.json")
MERGES_PATH  = Path("data/tinystories/tinystories_merges.txt")

# 默认对比的两个 checkpoint：lr=0.0055 与 lr=0.012
DEFAULT_CKPTS = [
    Path("checkpoints/tinystories_lr_sweep_round5_refine_0050_0065/4kogvde4/final.pt"),
    Path("checkpoints/tinystories_lr_divergence_probe/8mgqfh78/final.pt"),
]

# 模型超参（与训练时一致）
MODEL_CFG = dict(
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000.0,
)

EOS_TOKEN_ID = 256  # <|endoftext|> 在 tinystories tokenizer 中的 ID
MAX_NEW_TOKENS = 400

DECODE_CONFIGS = [
    {"label": "温度=0.8, top_k=40", "temperature": 0.8, "top_k": 40, "top_p": 1.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同 checkpoint 的文本生成输出")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=DEFAULT_CKPTS,
        help="要对比的 checkpoint 路径列表",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="运行设备，auto 表示优先 cuda 其次 mps 否则 cpu",
    )
    parser.add_argument("--prompt", type=str, default="<|endoftext|>Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_tokenizer() -> BpeTokenizer:
    return BpeTokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )


def load_model(ckpt_path: Path, device: torch.device) -> TransformerLM:
    model = TransformerLM(**MODEL_CFG, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ 已加载 checkpoint（iter={ckpt['iteration']}）：{ckpt_path}")
    return model


def run_generation(
    model: TransformerLM,
    tokenizer: BpeTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long)

    output_tensor = generate(
        model=model,
        input_tokens=input_tensor,
        max_length=max_new_tokens,
        context_length=MODEL_CFG["context_length"],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=EOS_TOKEN_ID,
        device=device,
    )

    # 只解码 prompt 之后新生成的部分
    new_ids = output_tensor[len(prompt_ids):].tolist()
    return tokenizer.decode(new_ids)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"使用设备：{device}\n")

    tokenizer = load_tokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {repr(args.prompt)}  ({len(prompt_ids)} tokens)\n")
    print("=" * 80)

    for ckpt in args.checkpoints:
        model = load_model(ckpt, device)
        print(f"\n===== checkpoint: {ckpt} =====")
        for cfg in DECODE_CONFIGS:
            print(f"\n【{cfg['label']}】")
            print("-" * 80)
            text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=cfg["temperature"],
                top_k=cfg["top_k"],
                top_p=cfg["top_p"],
            )
            print(text)
            print("-" * 80)


if __name__ == "__main__":
    main()
