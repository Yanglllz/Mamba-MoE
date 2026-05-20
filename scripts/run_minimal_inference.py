import argparse

import torch

from mamba_moe import DeterministicHMoE, build_mamba_moe_sharedhead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", choices=["MRI", "CT", "PET"], default="MRI")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mamba_moe_sharedhead(device=device, base_ckpt=args.checkpoint).eval()
    DeterministicHMoE.CURRENT_TASK = args.task

    x = torch.randn(1, 1, args.height, args.width, device=device)
    with torch.no_grad():
        y, _ = model(x)
    print({"task": args.task, "input_shape": tuple(x.shape), "output_shape": tuple(y.shape)})


if __name__ == "__main__":
    main()

