"""
Numerically check the derivation before trusting the code. No training needed.

    python sanity_check.py
"""

import torch

from ddpm import DDPM

torch.manual_seed(0)
ddpm = DDPM(T=1000)
T = ddpm.T
ab = ddpm.alpha_bars

print("1) Variance budget (Section 2): unit variance stays unit variance, Var(x_t) = (1 - beta_t) v + beta_t")
x = torch.randn(200_000, 1)                # Var(x_0) = 1
for s in range(1, T + 1):
    x = ddpm.q_step(x, s)
    if s in (1, 100, 1000):
        print(f"   t={s:4d}  Var(x_t) = {x.var():.3f}")

print("\n2) Forward shortcut (Eq. 11) matches running the chain step by step (Eq. 4)")
x0 = torch.full((200_000, 1), 2.0)         # every sample starts at x_0 = 2
for t_check in [10, 100, 500]:
    x = x0.clone()
    for s in range(1, t_check + 1):
        x = ddpm.q_step(x, s)
    shortcut = ddpm.q_sample(x0, torch.full((len(x0),), t_check), torch.randn_like(x0))
    print(f"   t={t_check:4d}  chain: mean {x.mean():+.3f} var {x.var():.3f}"
          f"  | shortcut: mean {shortcut.mean():+.3f} var {shortcut.var():.3f}"
          f"  | theory: mean {2 * ab[t_check].sqrt():+.3f} var {1 - ab[t_check]:.3f}")

print("\n3) Posterior mean: x_0 form (Eq. 22) == noise form (Eq. 26)")
x0 = torch.randn(5, 2, dtype=torch.float64)
eps = torch.randn(5, 2, dtype=torch.float64)
betas = ddpm.betas.double()                # float64 so rounding error does not hide the identity
alphas = 1 - betas
ab = torch.cumprod(alphas, dim=0)
for t_check in [2, 50, 999]:
    x_t = ab[t_check].sqrt() * x0 + (1 - ab[t_check]).sqrt() * eps
    a, b = alphas[t_check], betas[t_check]
    mu_x0 = (a.sqrt() * (1 - ab[t_check - 1]) / (1 - ab[t_check])) * x_t \
          + (ab[t_check - 1].sqrt() * b / (1 - ab[t_check])) * x0
    mu_eps = (x_t - b / (1 - ab[t_check]).sqrt() * eps) / a.sqrt()
    print(f"   t={t_check:4d}  max difference = {(mu_x0 - mu_eps).abs().max():.2e}")

print(f"\n4) End of the chain: abar_T = {ab[T]:.2e}, so x_T is almost pure N(0, I) noise")
