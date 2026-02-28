import torch


def run_sgd_toy_experiment(lr: float, steps: int = 10, seed: int = 0) -> list[float]:
    """运行作业中的 SGD toy 例子并返回每步 loss。"""
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), dtype=torch.float32))
    optimizer = torch.optim.SGD([weights], lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        optimizer.step()
    return losses


def main() -> None:
    for lr in [1e1, 1e2, 1e3]:
        losses = run_sgd_toy_experiment(lr=lr, steps=10, seed=0)
        print(f"lr={lr:.0e}")
        print(", ".join(f"{x:.6g}" for x in losses))


if __name__ == "__main__":
    main()
