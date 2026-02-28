import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .softmax import softmax


def scaled_dot_product_attention(Q: Float[Tensor, "batch_size ... seq_len d_k"], K: Float[Tensor, "batch_size ... seq_len d_k"], V: Float[Tensor, "batch_size ... seq_len d_v"], mask=None):
    """计算缩放点积注意力。
    
    参数:
    Q: 查询张量，形状为 (batch_size, ..., seq_len, d_k)
    K: 键张量，形状为 (batch_size, ..., seq_len, d_k)
    V: 值张量，形状为 (batch_size, ..., seq_len, d_v)
    mask: 可选的掩码张量，形状为 (batch_size, ..., seq_len, seq_len)，用于遮挡某些位置的注意力权重
    
    返回:
    输出张量，形状为 (batch_size, ..., seq_len, d_v)
    注意力权重张量，形状为 (batch_size, ..., seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        theta: float = 10000.0,  # 默认 10000
        max_seq_len: int = 512,  # 建议由外层传入，例如 Llama2 是 4096，作业 TinyStories 测试集常用 256/512
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=self.d_k, 
            max_seq_len=max_seq_len,
            device=device
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]

        # 一次性完成所有 head 的 Q/K/V 投影
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # [..., seq_len, d_model] -> [..., num_heads, seq_len, d_k]
        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)

        # 若未显式传入位置索引，则按 0..seq_len-1 自动构造并广播到 batch-like 维度
        if token_positions is None:
            # 目标：从 [seq_len] 生成与输入 x 对齐的位置索引 [..., seq_len]
            token_positions = torch.arange(seq_len, device=x.device)

            # x 的最后两维固定是 [seq_len, d_model]，前面都是 batch-like 维度
            # 例如 x.shape = [batch, heads, seq_len, d_model] 时，batch_shape = [batch, heads]
            batch_shape = x.shape[:-2]
            num_batch_dims = len(batch_shape)

            # 先把 [seq_len] 变成 [1, 1, ..., seq_len]（前面补 num_batch_dims 个 1）
            # 这里的 * 是“参数解包”：把列表/元组里的元素逐个作为位置参数传入。
            # 等价示例：view(1, 1, seq_len)
            token_positions = token_positions.view(*([1] * num_batch_dims), seq_len)

            # 再把前面的 1 广播成 batch_shape，得到 [..., seq_len]
            token_positions = token_positions.expand(*batch_shape, seq_len)
        token_positions = repeat(token_positions, "... seq_len -> ... num_heads seq_len", num_heads=self.num_heads)

        # RoPE 仅作用于 Q/K
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # 因果掩码：True 表示允许注意，False 表示屏蔽未来位置
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # [..., num_heads, seq_len, d_k] -> [..., seq_len, d_model]
        attn_out = rearrange(attn_out, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")

        return self.w_o(attn_out)
        