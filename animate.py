"""
Make the figures and animations for the README from a trained checkpoint.

    python animate.py                 # uses checkpoints/text.pt
    python animate.py --dataset swissroll

Outputs (in assets/):
    diffusion_<dataset>.gif     forward process (data -> noise) followed by DDPM sampling (noise -> data)
    strip_<dataset>.png         snapshots of the reverse chain at a few timesteps
    field_<dataset>.png         the learned denoising direction -eps_theta(x, t)
    schedule.png                sqrt(abar_t), sqrt(1 - abar_t) and their squares
    loss_<dataset>.png          training curve
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data import get_dataset
from ddpm import DDPM
from model import NoisePredictor

BG = "#0d1117"        # GitHub dark background
FG = "#e6edf3"
MUTED = "#8b949e"
CMAP = plt.get_cmap("plasma")

FORWARD_EQ = r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$"
REVERSE_EQ = (r"$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}"
              r"\,\epsilon_\theta(x_t,t)\right)+\sigma_t z$")


def frame_times(T, n_frames):
    """Timesteps to draw, denser near t = 0 where the shape actually forms."""
    u = np.linspace(0, 1, n_frames)
    return np.unique(np.round(T * u ** 2).astype(int))


def setup_axes(lims, size=(8, 5)):
    fig, ax = plt.subplots(figsize=size, dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-lims[0], lims[0])
    ax.set_ylim(-lims[1], lims[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08)
    return fig, ax


def render(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def make_gif(ddpm, model, x0, colors_fwd, lims, path):
    T = ddpm.T
    times = frame_times(T, 70)
    frames, durations = [], []

    fig, ax = setup_axes(lims)
    scat = ax.scatter([], [], s=3, linewidths=0)
    title = fig.text(0.5, 0.94, "", ha="center", va="center", color=FG, fontsize=15, weight="bold")
    eq = fig.text(0.5, 0.875, "", ha="center", va="center", color=MUTED, fontsize=12)
    info = fig.text(0.5, 0.035, "", ha="center", va="center", color=MUTED, fontsize=11, family="monospace")

    # Arrow field of the learned denoising direction -eps_theta(x, t), drawn during sampling.
    gx, gy = np.meshgrid(np.linspace(-lims[0], lims[0], 32), np.linspace(-lims[1], lims[1], 20))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    quiver = ax.quiver(gx, gy, np.zeros_like(gx), np.zeros_like(gy), color=FG, alpha=0.18,
                       scale=40, width=0.0018, headwidth=4)

    def draw(points, colors, t, phase):
        scat.set_offsets(points)
        scat.set_color(colors)
        ab = ddpm.alpha_bars[t].item()
        info.set_text(f"t = {t:4d}    signal sqrt(abar_t) = {ab ** 0.5:.2f}    noise sqrt(1-abar_t) = {(1 - ab) ** 0.5:.2f}")
        if phase == "forward":
            title.set_text("Forward process: adding noise")
            eq.set_text(FORWARD_EQ)
            quiver.set_visible(False)
        else:
            title.set_text("Reverse process: DDPM sampling")
            eq.set_text(REVERSE_EQ)
            quiver.set_visible(t > 0)
            if t > 0:
                with torch.no_grad():
                    d = -model(grid, torch.full((len(grid),), t, dtype=torch.long)).numpy()
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
                quiver.set_UVC(d[:, 0], d[:, 1])
        return render(fig)

    # ---- Forward: run the actual chain x_{t-1} -> x_t (Eq. 4) ----
    x = x0.clone()
    for t in range(0, T + 1):
        if t > 0:
            x = ddpm.q_step(x, t)
        if t in times:
            frames.append(draw(x.numpy(), colors_fwd, t, "forward"))
            durations.append(1200 if t == 0 else 40)
    durations[-1] = 700

    # ---- Reverse: Algorithm 2, colour each point by where it ends up ----
    torch.manual_seed(1)
    _, traj = ddpm.sample(model, x0.shape, return_trajectory=True)   # traj[k] = x_{T-k}
    final = traj[-1].numpy()
    colors_rev = CMAP(normalize(final[:, 0]))
    for t in sorted(times, reverse=True):
        frames.append(draw(traj[T - t].numpy(), colors_rev, t, "reverse"))
        durations.append(40)
    durations[-1] = 2500

    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"saved {path}")
    return traj


def normalize(v):
    lo, hi = np.percentile(v, [1, 99])     # robust to a few stray samples
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)


def make_strip(ddpm, traj, colors, lims, path):
    T = ddpm.T
    show = [T, 500, 200, 100, 30, 0]
    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.4 * lims[1] / lims[0] + 0.6), dpi=110)
    fig.patch.set_facecolor(BG)
    for ax, t in zip(axes, show):
        pts = traj[T - t].numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c=colors, linewidths=0)
        ax.set_xlim(-lims[0], lims[0])
        ax.set_ylim(-lims[1], lims[1])
        ax.set_aspect("equal")
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.set_title(f"$x_{{{t}}}$", color=FG, fontsize=14)
    fig.suptitle(f"Reverse chain:  pure noise  $x_{{{T}}}$  →  generated data  $x_0$", color=FG, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"saved {path}")


def make_field(ddpm, model, data, lims, path):
    """The learned denoising direction -eps_theta(x, t) at a few timesteps: early on it points to the
    centre of the data, late it points to the nearest letter stroke."""
    show = [800, 300, 100, 20]
    gx, gy = np.meshgrid(np.linspace(-lims[0], lims[0], 22), np.linspace(-lims[1], lims[1], 15))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.2 * lims[1] / lims[0] + 0.7), dpi=110)
    fig.patch.set_facecolor(BG)
    for ax, t in zip(axes, show):
        with torch.no_grad():
            d = -model(grid, torch.full((len(grid),), t, dtype=torch.long)).numpy()
        mag = np.linalg.norm(d, axis=1)
        d /= mag[:, None] + 1e-8
        ax.quiver(gx, gy, d[:, 0], d[:, 1], mag, cmap="viridis", scale=28, width=0.005, alpha=0.9)
        ax.scatter(data[:, 0], data[:, 1], s=0.5, color="#8b949e", alpha=0.5, linewidths=0)
        ax.set_xlim(-lims[0], lims[0])
        ax.set_ylim(-lims[1], lims[1])
        ax.set_aspect("equal")
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.set_title(f"$t={t}$", color=FG, fontsize=14)
    fig.suptitle(r"Learned denoising direction  $-\epsilon_\theta(x,t)$  (colour = magnitude)", color=FG, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"saved {path}")


def make_schedule(ddpm, path):
    t = np.arange(ddpm.T + 1)
    ab = ddpm.alpha_bars.cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xlabel("timestep $t$")
        ax.set_xlim(0, ddpm.T)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.plot(t, np.sqrt(ab), color="#0f766e", lw=2.2, label=r"signal  $\sqrt{\bar\alpha_t}$")
    ax1.plot(t, np.sqrt(1 - ab), color="#b45309", lw=2.2, label=r"noise  $\sqrt{1-\bar\alpha_t}$")
    ax1.set_title("Amplitudes (what multiplies $x_0$ and $\\epsilon$)")
    ax1.legend()
    ax2.stackplot(t, ab, 1 - ab, colors=["#0f766e", "#b45309"], alpha=0.8,
                  labels=[r"$\bar\alpha_t$", r"$1-\bar\alpha_t$"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Variances: the squares always add to 1")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def make_loss(losses, path):
    losses = np.array(losses)
    k = 200
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=110)
    ax.plot(losses, color="#93c5fd", lw=0.5, alpha=0.6, label="per step")
    ax.plot(np.arange(k - 1, len(losses)), smooth, color="#2563eb", lw=2, label=f"moving avg ({k})")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\| \epsilon - \epsilon_\theta \|^2$")
    ax.set_title(r"Training loss $L_{\mathrm{simple}}$")
    ax.set_ylim(0, np.percentile(losses, 99.5))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="text")
    parser.add_argument("--n", type=int, default=2000, help="number of points to animate")
    args = parser.parse_args()

    ckpt = torch.load(f"checkpoints/{args.dataset}.pt", map_location="cpu")
    cfg = ckpt["args"]
    model = NoisePredictor()
    model.load_state_dict(ckpt["model"])
    model.eval()
    ddpm = DDPM(T=cfg["T"])

    torch.manual_seed(0)
    data = get_dataset(cfg["dataset"], text=cfg["text"])
    x0 = data[torch.randperm(len(data))[: args.n]]
    d = data.numpy()
    lims = (3.6, 2.4)          # data lives in [-2.5, 2.5]; x_T ~ N(0, I) mostly inside +-2.5 too
    colors_fwd = CMAP(normalize(x0[:, 0].numpy()))

    os.makedirs("assets", exist_ok=True)
    name = args.dataset
    traj = make_gif(ddpm, model, x0, colors_fwd, lims, f"assets/diffusion_{name}.gif")
    colors_rev = CMAP(normalize(traj[-1][:, 0].numpy()))
    make_strip(ddpm, traj, colors_rev, lims, f"assets/strip_{name}.png")
    make_field(ddpm, model, d, lims, f"assets/field_{name}.png")
    make_schedule(ddpm, "assets/schedule.png")
    make_loss(ckpt["losses"], f"assets/loss_{name}.png")


if __name__ == "__main__":
    main()
