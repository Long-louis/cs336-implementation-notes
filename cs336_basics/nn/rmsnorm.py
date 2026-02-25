import torch
from jaxtyping import Float
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMSNorm：对最后一维做均方根归一化并乘可学习增益。"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造 RMSNorm 模块。"""
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # TODO: 按作业要求初始化可学习参数（建议命名为 weight）
        # 期望形状: (d_model,)
        # 初始化: 全 1
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]) -> Float[Tensor, "batch sequence_length d_model"]:
        """对输入执行 RMSNorm，输出形状与输入一致。"""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化并乘 gain 参数
        result = x / rms * self.weight

        return result.to(in_dtype)
