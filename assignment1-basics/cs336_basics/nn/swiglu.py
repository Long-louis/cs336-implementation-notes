import torch
from jaxtyping import Float
from torch import Tensor, nn

from .linear import Linear
from .activations import SiLU

class SwiGLU(nn.Module):
    """位置前馈网络：SwiGLU(x) = W2(SiLU(W1x) ⊙ W3x)。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int|None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """构造 SwiGLU 前馈模块。"""
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff
        # TODO: 采用不带 bias 的线性层，构建 W1/W2/W3
        # W1: d_model -> d_ff
        # W2: d_ff -> d_model
        # W3: d_model -> d_ff
        # 提示：可复用你实现的 Linear 模块
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """对输入执行 SwiGLU 前馈变换。"""
        # TODO: 按公式实现
        # hidden = SiLU(W1(x))
        # gate = W3(x)
        # out = W2(hidden * gate)
        # 说明：本题允许直接使用 torch.sigmoid 实现 SiLU 的稳定版本
        hidden = self.silu(self.w1(x))
        gate = self.w3(x)
        out = self.w2(hidden * gate)
        return out