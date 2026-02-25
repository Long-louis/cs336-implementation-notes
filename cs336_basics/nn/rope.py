import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """RoPE：对最后一维按二维子空间做旋转位置编码。"""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """构造 RoPE 模块并按需预计算缓存。"""
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 要点：d_k 必须可按偶数维成对旋转
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        # 预计算 sin/cos 缓存，形状为 (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freq_base = theta ** (- torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = einsum(positions, freq_base, "sequence_length, half_d -> sequence_length half_d")

        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False
        )
        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False
        )
    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
        """将 RoPE 应用于输入张量。"""
        # x分为偶数维和奇数维两部分，分别乘以 cos 和 sin 后组合
        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        cos = self.cos_cache[token_positions]  # type: ignore # (sequence_length, d_k/2)
        sin = self.sin_cache[token_positions]  # type: ignore # (sequence_length, d_k/2)
        x_rotated_odd = x_odd * cos + x_even * sin
        x_rotated_even = x_even * cos - x_odd * sin
        # 交替组合偶数维和奇数维
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        return x_rotated