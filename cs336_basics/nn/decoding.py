import torch
import torch.nn as nn

def sample_next_token(model: nn.Module, input_tokens: torch.LongTensor, temperature: float = 1.0, top_k: int = 0) -> torch.LongTensor:
    """给定输入 token 序列，使用模型预测下一个 token 的 ID。

    Args:
        model: 语言模型。
        input_tokens: 输入 token 序列，形状为 (context_length,)，数据类型为 torch.long。
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
            values, indices = torch.topk(next_token_logits, top_k)
            next_token_logits = next_token_logits.masked_fill(next_token_logits < values[:, -1:], float('-inf'))
        # 使用 softmax 生成概率分布并采样下一个 token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id


def generate(model: nn.Module, input_tokens: torch.LongTensor, max_length: int, context_length: int, temperature: float = 1.0, top_k: int = 0, device=None) -> torch.LongTensor:
    """从输入 token 序列开始，生成新的 token 序列。

    Args:
        model: 语言模型。
        input_tokens: 输入 token 序列，形状为 (context_length,)，数据类型为 torch.long。
        max_length: 生成的最大 token 数量。
    Returns:生成的 token 序列
    """
    if device is not None:
        input_tokens = input_tokens.to(device)
    generated_tokens = input_tokens.tolist()  # 将输入 token 转换为列表，方便追加生成的 token

    for _ in range(max_length):
        current_input = torch.tensor(generated_tokens[-context_length:], dtype=torch.long, device=device)  # 获取当前上下文
        next_token_id = sample_next_token(model, current_input, temperature=temperature, top_k=top_k)
        generated_tokens.append(next_token_id.item())  # 将生成的 token ID 添加到列表中

    return torch.tensor(generated_tokens, dtype=torch.long)
