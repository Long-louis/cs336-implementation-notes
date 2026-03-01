import argparse
import math
import time
from pathlib import Path

import numpy
import torch
import wandb

from cs336_basics.nn.loss import cross_entropy_loss
from cs336_basics.nn.optimizer import AdamW
from cs336_basics.nn.training_utils import (
	get_batch,
	get_lr_cosine_schedule,
	gradient_clipping,
	load_checkpoint,
	save_checkpoint,
)
from cs336_basics.nn.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="使用自定义组件训练 Transformer LM")

	parser.add_argument("--train-tokens-path", type=Path, required=True)
	parser.add_argument("--valid-tokens-path", type=Path, required=True)

	parser.add_argument("--vocab-size", type=int, required=True)
	parser.add_argument("--context-length", type=int, default=256)
	parser.add_argument("--d-model", type=int, default=512)
	parser.add_argument("--num-layers", type=int, default=8)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--d-ff", type=int, default=1344)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")

	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--max-iters", type=int, default=2000)
	parser.add_argument("--max-learning-rate", type=float, default=3e-4)
	parser.add_argument("--min-learning-rate", type=float, default=3e-5)
	parser.add_argument("--warmup-iters", type=int, default=200)
	parser.add_argument("--cosine-cycle-iters", type=int, default=2000)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.95)
	parser.add_argument("--eps", type=float, default=1e-8)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--grad-clip-norm", type=float, default=1.0)

	parser.add_argument("--eval-interval", type=int, default=100)
	parser.add_argument("--eval-batches", type=int, default=20)
	parser.add_argument("--log-interval", type=int, default=10)

	parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
	parser.add_argument("--checkpoint-interval", type=int, default=200)
	parser.add_argument("--resume-from", type=Path, default=None)

	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--use-wandb", action="store_true")
	parser.add_argument("--wandb-project", type=str, default="cs336-assignment1")
	parser.add_argument("--wandb-entity", type=str, default=None)
	parser.add_argument("--wandb-run-name", type=str, default=None)
	parser.add_argument("--wandb-group", type=str, default=None)

	return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		if torch.backends.mps.is_available():
			return torch.device("mps")
		return torch.device("cpu")
	return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
	if dtype_arg == "float32":
		return torch.float32
	if dtype_arg == "float16":
		return torch.float16
	if dtype_arg == "bfloat16":
		return torch.bfloat16
	raise ValueError(f"不支持的 dtype: {dtype_arg}")


def load_tokens_memmap(path: Path) -> numpy.ndarray:
	if not path.exists():
		raise ValueError(f"数据文件不存在: {path}")
	if path.suffix != ".npy":
		raise ValueError(f"当前仅支持 .npy 文件进行 mmap 加载: {path}")

	data = numpy.load(path, mmap_mode="r")

	if data.ndim != 1:
		raise ValueError(f"token 数据必须为一维数组，当前 shape={data.shape}")
	if not numpy.issubdtype(data.dtype, numpy.integer):
		raise ValueError(f"token 数据 dtype 必须是整数类型，当前 dtype={data.dtype}")

	return data


def evaluate_loss(
	model: TransformerLM,
	valid_tokens: numpy.ndarray,
	batch_size: int,
	context_length: int,
	eval_batches: int,
	device: torch.device,
) -> float:
	model.eval()
	losses: list[float] = []

	with torch.no_grad():
		for _ in range(eval_batches):
			x, y = get_batch(valid_tokens, batch_size=batch_size, context_length=context_length, device=device)
			logits = model(x)
			loss = cross_entropy_loss(logits, y)
			losses.append(float(loss.detach().cpu()))

	model.train()
	return sum(losses) / len(losses)


def run_training(args: argparse.Namespace) -> None:
	device = resolve_device(args.device)
	dtype = resolve_dtype(args.dtype)

	numpy.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_tokens = load_tokens_memmap(args.train_tokens_path)
	valid_tokens = load_tokens_memmap(args.valid_tokens_path)

	min_required_tokens = args.context_length + 1
	if len(train_tokens) < min_required_tokens:
		raise ValueError("训练集 token 数过少，无法采样一个 batch")
	if len(valid_tokens) < min_required_tokens:
		raise ValueError("验证集 token 数过少，无法采样一个 batch")

	model = TransformerLM(
		num_layers=args.num_layers,
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		num_heads=args.num_heads,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		max_seq_len=args.context_length,
		device=device,
		dtype=dtype,
	)

	optimizer = AdamW(
		model.parameters(),
		lr=args.max_learning_rate,
		betas=(args.beta1, args.beta2),
		eps=args.eps,
		weight_decay=args.weight_decay,
	)

	if args.use_wandb:
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=args.wandb_run_name,
			group=args.wandb_group,
			config=vars(args),
		)
		wandb.define_metric("train/grad_step")
		wandb.define_metric("time/wallclock_seconds")
		wandb.define_metric("train/loss", step_metric="train/grad_step")
		wandb.define_metric("valid/loss", step_metric="train/grad_step")
		wandb.define_metric("train/perplexity", step_metric="train/grad_step")
		wandb.define_metric("valid/perplexity", step_metric="train/grad_step")

	args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	start_iter = 0
	if args.resume_from is not None:
		start_iter = load_checkpoint(args.resume_from, model=model, optimizer=optimizer)

	model.train()
	tokens_per_step = args.batch_size * args.context_length
	run_start_time = time.perf_counter()

	for it in range(start_iter, args.max_iters):
		step_start_time = time.perf_counter()

		lr = get_lr_cosine_schedule(
			it=it,
			max_learning_rate=args.max_learning_rate,
			min_learning_rate=args.min_learning_rate,
			warmup_iters=args.warmup_iters,
			cosine_cycle_iters=args.cosine_cycle_iters,
		)
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

		x, y = get_batch(
			data=train_tokens,
			batch_size=args.batch_size,
			context_length=args.context_length,
			device=device,
		)

		logits = model(x)
		loss = cross_entropy_loss(logits, y)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		gradient_clipping(model.parameters(), max_norm=args.grad_clip_norm)
		optimizer.step()

		step_time = time.perf_counter() - step_start_time
		elapsed_seconds = time.perf_counter() - run_start_time
		train_loss_value = float(loss.detach().cpu())
		train_perplexity = math.exp(train_loss_value)
		tokens_per_second = tokens_per_step / step_time
		current_iter = it + 1
		tokens_seen = current_iter * tokens_per_step

		should_eval = current_iter % args.eval_interval == 0
		should_log = current_iter % args.log_interval == 0 or should_eval
		should_ckpt = current_iter % args.checkpoint_interval == 0

		val_loss_value = None
		if should_eval:
			val_loss_value = evaluate_loss(
				model=model,
				valid_tokens=valid_tokens,
				batch_size=args.batch_size,
				context_length=args.context_length,
				eval_batches=args.eval_batches,
				device=device,
			)

		if should_log:
			metrics: dict[str, float | int] = {
				"train/grad_step": current_iter,
				"train/loss": train_loss_value,
				"train/perplexity": train_perplexity,
				"train/lr": lr,
				"train/step_time_seconds": step_time,
				"train/tokens_seen": tokens_seen,
				"train/tokens_per_second": tokens_per_second,
				"time/wallclock_seconds": elapsed_seconds,
			}
			if val_loss_value is not None:
				metrics["valid/loss"] = val_loss_value
				metrics["valid/perplexity"] = math.exp(val_loss_value)

			print(metrics)

			if args.use_wandb:
				wandb.log(metrics, step=current_iter)

		if should_ckpt:
			ckpt_path = args.checkpoint_dir / f"ckpt_iter_{current_iter}.pt"
			latest_path = args.checkpoint_dir / "latest.pt"
			save_checkpoint(model=model, optimizer=optimizer, iteration=current_iter, out=ckpt_path)
			save_checkpoint(model=model, optimizer=optimizer, iteration=current_iter, out=latest_path)

	final_path = args.checkpoint_dir / "final.pt"
	save_checkpoint(model=model, optimizer=optimizer, iteration=args.max_iters, out=final_path)

	if args.use_wandb:
		wandb.finish()


def main() -> None:
	args = parse_args()
	run_training(args)


if __name__ == "__main__":
	main()
