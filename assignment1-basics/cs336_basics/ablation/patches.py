from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

import cs336_basics.nn.attention as attention_mod
import cs336_basics.nn.transformer as transformer_mod
from cs336_basics.nn.activations import SiLU
from cs336_basics.nn.linear import Linear


class AblationIdentity(nn.Identity):
    """兼容 RMSNorm 构造签名的恒等映射模块。"""

    def __init__(self, *args, **kwargs):
        super().__init__()


class NoPositionalEmbedding(nn.Module):
    """兼容 RoPE 构造与调用签名的无位置编码模块。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, token_positions):
        return x


class SiLUFeedForward(nn.Module):
    """两层 MLP 版 SiLU FFN，参数量与 SwiGLU 近似匹配。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64

        # 匹配参数量：SwiGLU 约为 3*d_model*d_ff，SiLU-FFN 约为 2*d_model*h，因此取 h≈1.5*d_ff。
        hidden_dim = int((3 * d_ff) / 2)
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = Linear(d_model, hidden_dim, device=device, dtype=dtype)
        self.w2 = Linear(hidden_dim, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)))


PatchFn = Callable[[], str]


def patch_rmsnorm() -> str:
    """将 Transformer 中引用的 RMSNorm 替换为恒等映射。"""
    transformer_mod.RMSNorm = AblationIdentity
    return "rmsnorm"


def _postnorm_block_forward(self, x):
    """将 block 计算顺序改为 post-norm。"""
    x = self.norm1(x + self.self_attention(x))
    x = self.norm2(x + self.ffn(x))
    return x


def patch_postnorm() -> str:
    """将 TransformerBlock 从 pre-norm 改为 post-norm。"""
    transformer_mod.TransformerBlock.forward = _postnorm_block_forward
    return "postnorm"


def patch_nope() -> str:
    """移除 RoPE 的位置信息（NoPE）。"""
    attention_mod.RotaryPositionalEmbedding = NoPositionalEmbedding
    return "nope"


def patch_silu_ffn() -> str:
    """将 Transformer 前馈网络从 SwiGLU 替换为 SiLU-FFN。"""
    transformer_mod.SwiGLU = SiLUFeedForward
    return "silu_ffn"


PATCH_REGISTRY: dict[str, PatchFn] = {
    "rmsnorm": patch_rmsnorm,
    "postnorm": patch_postnorm,
    "nope": patch_nope,
    "silu_ffn": patch_silu_ffn,
}


def parse_ablation_list(raw: str | None) -> list[str]:
    if raw is None:
        return []

    parts = [item.strip().lower() for item in raw.split(",")]
    non_empty = [item for item in parts if item]

    deduped: list[str] = []
    seen: set[str] = set()
    for name in non_empty:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    return deduped


def apply_ablation_patches(ablation_names: list[str]) -> list[str]:
    applied: list[str] = []
    for name in ablation_names:
        if name not in PATCH_REGISTRY:
            supported = ", ".join(sorted(PATCH_REGISTRY.keys()))
            raise ValueError(f"未知 ablation: {name}。可选: {supported}")
        applied_name = PATCH_REGISTRY[name]()
        applied.append(applied_name)
    return applied
