from __future__ import annotations

import sys

import cs336_basics.train as base_train
from cs336_basics.ablation.patches import apply_ablation_patches, parse_ablation_list


def _strip_ablation_cli_args(argv: list[str]) -> tuple[list[str], str | None]:
    cleaned: list[str] = []
    ablate_value: str | None = None

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "--ablate":
            if i + 1 >= len(argv):
                raise ValueError("参数 --ablate 缺少值，例如 --ablate rmsnorm")
            ablate_value = argv[i + 1]
            i += 2
            continue

        if arg.startswith("--ablate="):
            ablate_value = arg.split("=", maxsplit=1)[1]
            i += 1
            continue

        cleaned.append(arg)
        i += 1

    return cleaned, ablate_value


def main() -> None:
    cleaned_argv, ablate_raw = _strip_ablation_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *cleaned_argv]

    args = base_train.parse_args()

    requested_ablations = parse_ablation_list(ablate_raw)
    applied_ablations = apply_ablation_patches(requested_ablations)

    if args.use_wandb and len(applied_ablations) > 0:
        suffix = "-".join(applied_ablations)
        if args.wandb_group is None:
            args.wandb_group = f"ablation-{suffix}"
        else:
            args.wandb_group = f"{args.wandb_group}-ablation-{suffix}"

    print({"ablation/requested": requested_ablations, "ablation/applied": applied_ablations})

    base_train.run_training(args)


if __name__ == "__main__":
    main()
