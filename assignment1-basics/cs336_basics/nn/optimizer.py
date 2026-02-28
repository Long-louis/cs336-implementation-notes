import torch
from collections.abc import Iterable, Callable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    """AdamW 优化器的 PyTorch 实现。"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        """初始化 AdamW 优化器。

        Args:
            params: 可迭代的参数列表或参数组。
            lr: 学习率。
            betas: 用于计算梯度和梯度平方的移动平均的系数。
            eps: 为了数值稳定性而添加到分母中的项。
            weight_decay: 权重衰减（L2 正则化）系数。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable]=None):
        """执行一步优化。

        Args:
            closure: 一个可选的闭包函数，用于重新计算损失。
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # state 维护了每个参数的优化状态，包括动量和二阶矩估计 
                # 初始化
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # 更新步数
                state['step'] += 1
                # 更新动量和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # 计算更新后的lr
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                # 更新参数
                p.data -= step_size * exp_avg / (exp_avg_sq.sqrt() + eps) 
                # 应用权重衰减
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

        return loss

        