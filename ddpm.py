"""
DDPM core: the math from "Wait... DDPM Finally Makes Sense!" turned into code.

Every method below points to the equation it implements (Eq. numbers refer to
the PDF write-up). Notation follows the paper exactly:

    beta_t       noise added at step t
    alpha_t      = 1 - beta_t
    alpha_bar_t  = alpha_1 * alpha_2 * ... * alpha_t      (alpha_bar_0 = 1)

To keep the code identical to the math, all schedule tensors are indexed by
t = 0, 1, ..., T. Index 0 is a dummy (beta_0 = 0, alpha_bar_0 = 1), so
`self.betas[t]` really means beta_t.
"""

import torch


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, sigma="beta", device="cpu"):
        self.T = T
        self.device = device

        # Linear schedule from Ho et al. (2020): beta_1 = 1e-4, ..., beta_T = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        self.betas = torch.cat([torch.zeros(1), betas]).to(device)   # beta_0 = 0 (dummy)
        self.alphas = 1.0 - self.betas                                # alpha_t = 1 - beta_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)           # alpha_bar_t, alpha_bar_0 = 1

        # Posterior variance, Eq. (21): tilde_beta_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t
        self.tilde_betas = torch.zeros_like(self.betas)
        self.tilde_betas[1:] = (1 - self.alpha_bars[:-1]) / (1 - self.alpha_bars[1:]) * self.betas[1:]

        # Reverse-process variance sigma_t^2 (Section 6.1): either beta_t or tilde_beta_t
        self.sigmas = (self.betas if sigma == "beta" else self.tilde_betas).sqrt()

    @staticmethod
    def _bcast(coef, x):
        """Reshape a per-sample coefficient of shape (B,) so it broadcasts against x of shape (B, ...)."""
        return coef.view(-1, *([1] * (x.dim() - 1)))

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------
    def q_step(self, x_prev, t):
        """One forward step, Eq. (4):
            x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * eps_t
        Only used for visualisation / sanity checks; training uses the shortcut below.
        """
        eps = torch.randn_like(x_prev)
        return self.alphas[t].sqrt() * x_prev + (1 - self.alphas[t]).sqrt() * eps

    def q_sample(self, x0, t, eps):
        """Forward shortcut, Eq. (11): jump from x_0 straight to any x_t.
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        t is a LongTensor of shape (B,) with values in {1, ..., T}.
        """
        ab = self._bcast(self.alpha_bars[t], x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    # ------------------------------------------------------------------
    # Training (Algorithm 1)
    # ------------------------------------------------------------------
    def loss(self, model, x0):
        """Simplified loss, Eq. (30):
            L_simple = E || eps - eps_theta( sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, t ) ||^2
        """
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)   # t ~ Uniform{1, ..., T}
        eps = torch.randn_like(x0)                                   # eps ~ N(0, I)
        x_t = self.q_sample(x0, t, eps)                              # build x_t in one jump
        eps_pred = model(x_t, t)                                     # network guesses the noise
        return ((eps - eps_pred) ** 2).mean()

    # ------------------------------------------------------------------
    # Sampling (Algorithm 2)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_mean(self, model, x_t, t):
        """Model mean, Eq. (27):
            mu_theta = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - abar_t) * eps_theta(x_t, t) )
        """
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)
        coef = self.betas[t] / (1 - self.alpha_bars[t]).sqrt()
        return (x_t - coef * eps_pred) / self.alphas[t].sqrt()

    @torch.no_grad()
    def p_step(self, model, x_t, t):
        """One reverse step, Eq. (29):
            x_{t-1} = mu_theta(x_t, t) + sigma_t * z,   z ~ N(0, I),   and z = 0 when t = 1
        """
        mean = self.p_mean(model, x_t, t)
        if t == 1:
            return mean
        z = torch.randn_like(x_t)
        return mean + self.sigmas[t] * z

    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False):
        """Run the full reverse chain x_T -> x_{T-1} -> ... -> x_0.

        If return_trajectory is True, also return [x_T, x_{T-1}, ..., x_0]
        (so trajectory[k] is x_{T-k}).
        """
        x = torch.randn(shape, device=self.device)   # x_T ~ N(0, I)
        trajectory = [x.cpu()]
        for t in range(self.T, 0, -1):               # t = T, ..., 1
            x = self.p_step(model, x, t)
            if return_trajectory:
                trajectory.append(x.cpu())
        return (x, trajectory) if return_trajectory else x
