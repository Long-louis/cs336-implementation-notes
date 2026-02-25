import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class Embedding(nn.Module):
    """Embedding 层：根据 token id 查表得到向量。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        初始化 Embedding 模块。

        Args:
            num_embeddings: 词表大小。
            embedding_dim: 每个嵌入向量的维度。
            device: 分配到的设备。
            dtype: 权重的数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight: Float[Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        """根据 token_ids 查找对应 embedding 向量。"""
        return self.weight[token_ids]
