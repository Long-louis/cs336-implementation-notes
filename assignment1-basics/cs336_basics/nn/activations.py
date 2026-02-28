import torch
from jaxtyping import Float
from torch import Tensor, nn

class SiLU(nn.Module):
    """SiLU 激活函数：SiLU(x) = x * sigmoid(x)。"""

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """对输入 x 应用 SiLU 激活。"""
        return x * torch.sigmoid(x)