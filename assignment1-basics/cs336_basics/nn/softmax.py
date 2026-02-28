import torch
from jaxtyping import Float
from torch import Tensor, nn

def softmax(x: Float[Tensor, "... d_model"], dim: int) -> Float[Tensor, "... d_model"]:
    """对输入 x 的指定维度 dim 执行 softmax，输出形状与输入一致。"""
    return torch.exp(x - torch.max(x, dim=dim, keepdim=True).values) / torch.sum(torch.exp(x - torch.max(x, dim=dim, keepdim=True).values), dim=dim, keepdim=True)