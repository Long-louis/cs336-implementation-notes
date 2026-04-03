import torch
import torch.nn as nn


def sample_next_token(
    model: nn.Module,
    input_tokens: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """给定输入 token 序列，使用模型预测下一个 token 的 ID。

    Args:
        model: 语言模型。
        input_tokens: 输入 token 序列，形状为 (context_length,)，数据类型为 torch.long。
        temperature: softmax 温度参数，越低越确定，越高越随机。
        top_k: 若 > 0，只保留概率最高的 top_k 个 token。
        top_p: nucleus sampling 阈值，只保留累积概率达到 p 的最小集合（0 < p <= 1）。
    Returns:
        预测的下一个 token 的 ID，形状为 (1,)，数据类型为 torch.long。
    """
    model.eval()
    with torch.no_grad():
        input_tokens = input_tokens.unsqueeze(0)  # 添加 batch 维度，变为 (1, context_length)
        logits = model(input_tokens)  # 模型输出形状为 (1, context_length, vocab_size)
        next_token_logits = logits[:, -1, :]  # 获取最后一个位置的 logits，形状为 (1, vocab_size)
        next_token_logits = next_token_logits / temperature  # 应用温度参数
        if top_k > 0:
            # 将 logits 中小于 top_k 最大值的元素设为负无穷
            values, _ = torch.topk(next_token_logits, top_k)
            next_token_logits = next_token_logits.masked_fill(next_token_logits < values[:, -1:], float('-inf'))
        # 使用 softmax 生成概率分布
        probs = torch.softmax(next_token_logits, dim=-1)
        if top_p < 1.0:
            # nucleus (top-p) sampling：按概率降序排列，找出累积概率达到 p 的最小集合
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 移除累积概率超过 p 之后的 token（保留使累积值刚好 >= p 的最后一个）
            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0
            # 将过滤后的概率散回原始顺序
            probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id


def generate(
    model: nn.Module,
    input_tokens: torch.Tensor,
    max_length: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """从输入 token 序列开始，生成新的 token 序列。

    Args:
        model: 语言模型。
        input_tokens: 输入 token 序列，形状为 (prompt_length,)，数据类型为 torch.long。
        max_length: 最多生成的新 token 数量（不含 prompt）。
        context_length: 模型支持的最大上下文长度。
        temperature: softmax 温度参数。
        top_k: 若 > 0，启用 top-k 过滤。
        top_p: nucleus sampling 阈值，默认 1.0 表示不过滤。
        eos_token_id: 若指定，生成到该 token 时立即停止。
        device: 设备，None 表示沿用 input_tokens 所在设备。
    Returns:
        生成的完整 token 序列（含 prompt），形状为 (prompt_length + generated,)。
    """
    if device is not None:
        input_tokens = input_tokens.to(device)
    generated_tokens = input_tokens.tolist()  # 将输入 token 转换为列表，方便追加生成的 token

    for _ in range(max_length):
        current_input = torch.tensor(generated_tokens[-context_length:], dtype=torch.long, device=device)
        next_token_id = sample_next_token(
            model, current_input, temperature=temperature, top_k=top_k, top_p=top_p
        )
        token = next_token_id.item()
        generated_tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break

    return torch.tensor(generated_tokens, dtype=torch.long)
