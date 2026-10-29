"""
Train eps_theta on a 2-D dataset (Algorithm 1).

    python train.py                          # letters "DDPM"
    python train.py --dataset swissroll
    python train.py --dataset text --text HI
"""

import argparse
import os

import torch

from data import get_dataset
from ddpm import DDPM
from model import NoisePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="text", choices=["text", "swissroll", "moons"])
    parser.add_argument("--text", default="DDPM")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():      # Apple-silicon GPU
        device = "mps"
    else:
        device = "cpu"
    print(f"training on {device}")

    data = get_dataset(args.dataset, text=args.text).to(device)
    ddpm = DDPM(T=args.T, device=device)
    model = NoisePredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    losses = []
    for step in range(1, args.steps + 1):
        x0 = data[torch.randint(len(data), (args.batch,), device=device)]   # x_0 ~ q_data
        loss = ddpm.loss(model, x0)                                          # Algorithm 1

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(loss.detach())                 # no .item() here: it would force a GPU sync every step
        if step % 1000 == 0:
            print(f"step {step:6d} | loss {torch.stack(losses[-1000:]).mean().item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{args.dataset}.pt"
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(args), "losses": torch.stack(losses).tolist()}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
