import math
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy
import os
import torch

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """计算给定迭代次数的学习率。

    Args:
        it: 当前迭代次数。
        max_learning_rate: 预热阶段结束时的最大学习率。
        min_learning_rate: 余弦周期结束时的最小学习率。
        warmup_iters: 预热阶段的迭代次数。
        cosine_cycle_iters: 余弦周期的迭代次数。
    """
    if it < warmup_iters:
        # 线性预热阶段：t < T_w
        return max_learning_rate * (it / warmup_iters)

    if it > cosine_cycle_iters:
        # 退火后阶段：t > T_c
        return min_learning_rate

    # 余弦退火阶段：T_w <= t <= T_c
    anneal_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1 + math.cos(anneal_progress * math.pi))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    parameters = list(params)
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)

    if float(clip_coef) < 1.0:
        for grad in grads:
            grad.detach().mul_(clip_coef)

def get_batch(data: numpy.ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] :
    """从数据中随机采样一个批次。

    Args:
        data: 包含训练数据的numpy数组。
        batch_size: 批次大小。
        context_length: 上下文长度，即输入序列的长度。
        device: 目标设备（CPU或GPU）。
    """
    max_start_index = len(data) - context_length
    start_indices = numpy.random.randint(0, max_start_index, size=batch_size)

    x_batch = [data[i : i + context_length] for i in start_indices]
    y_batch = [data[i + 1 : i + context_length + 1] for i in start_indices]
    
    # 避免“从 numpy 列表直接建 tensor 很慢”的 warning
    x_batch_array = numpy.asarray(x_batch)
    y_batch_array = numpy.asarray(y_batch)

    x_batch_tensor = torch.tensor(x_batch_array, dtype=torch.long, device=device)
    y_batch_tensor = torch.tensor(y_batch_array, dtype=torch.long, device=device)

    return x_batch_tensor, y_batch_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存训练检查点。

    Args:
        model: 需要保存状态的模型。
        optimizer: 需要保存状态的优化器。
        iteration: 当前训练迭代步数。
        out: 输出路径或二进制文件对象。
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载训练检查点。

    Args:
        src: 检查点路径或二进制文件对象。
        model: 需要恢复状态的模型。
        optimizer: 需要恢复状态的优化器。

    Returns:
        检查点中保存的训练迭代步数。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]



