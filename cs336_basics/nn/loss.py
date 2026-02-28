import torch
from einops import einsum
from torch import Tensor

from jaxtyping import Float, Int

def cross_entropy_loss(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """交叉熵损失函数"""
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    # exp(logits - max_logit) 是为了数值稳定性，防止指数函数计算时出现溢出, 后面再加回来保持等式不变
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1)) + max_logits.squeeze(-1)
    # 选择正确类别的 logit
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()

