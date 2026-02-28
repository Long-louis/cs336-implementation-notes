import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    """不带 bias 的线性层：y = Wx。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造线性变换模块。"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W: Float[Tensor, "out_features in_features"] = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        # TODO: 按作业要求初始化并注册参数 self.weight
        # 期望形状: (out_features, in_features)
        # 初始化: trunc_normal_(std=sqrt(2 / (in_features + out_features)), 截断到 ±3σ)
        self._init_weights()

    def _init_weights(self) -> None:
        # 初始化权重为截断正态分布
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        """对输入 x 应用线性变换。"""
        return einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
