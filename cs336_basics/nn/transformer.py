import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float = 10000.0,  # 默认 10000
        max_seq_len: int = 512,  # 建议由外层传入，例如 Llama2 是 4096，作业 TinyStories 测试集常用 256/512
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            theta=rope_theta, 
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,  # 默认 10000
        max_seq_len: int = 512,  
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                rope_theta=rope_theta, 
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits