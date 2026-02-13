import math
from typing import Dict, Tuple
import numpy as np
import torch

# r' = a*r−λ(r−μ)^2

class Shaping: 
    def __init__(
        self,
        *,
        alpha_fidelity: float,
        alpha_alignment: float,
        fidelity_reward: Dict[int, np.ndarray],
        alignment_reward: Dict[int, np.ndarray],
        a: float = 1.0,
        frac: float = 0.3,
        q_lo: float = 0.05,
        q_hi: float = 0.95,
        use_median: bool = True,
        clip_k: float = 1.0, # clip 할 때, std 앞에 곱하는 계수 
        eps: float = 1e-8,
    ):
        self.alpha_fidelity = float(alpha_fidelity)
        self.alpha_alignment = float(alpha_alignment)
        self.fidelity_reward = fidelity_reward
        self.alignment_reward = alignment_reward

        self.a = float(a)
        self.frac = float(frac)
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self.use_median = bool(use_median)
        self.clip_k = float(clip_k)
        self.eps = float(eps)

        # fold별 파라미터/통계 캐시
        self._cache: Dict[int, Dict[str, float]] = {}

    def _pool_for_fold(self, fold: int) -> torch.Tensor:
        pool = self.alpha_fidelity * self.fidelity_reward[fold] + self.alpha_alignment * self.alignment_reward[fold]
        return torch.as_tensor(pool, dtype=torch.float32)

    @torch.no_grad()
    def _params_for_fold(self, fold: int) -> Dict[str, float]:
        if fold in self._cache:
            return self._cache[fold]

        t = self._pool_for_fold(fold)
        q_lo = torch.quantile(t, self.q_lo)
        q_hi = torch.quantile(t, self.q_hi)
        mu = t.median() if self.use_median else t.mean()

        radius = torch.maximum(q_hi - mu, mu - q_lo).clamp_min(1e-8)
        lam_max = self.a / (2.0 * float(radius))
        lam = float(self.frac * lam_max)

        R_pool = self.a * t - lam * (t - mu) ** 2
        mu_R = float(R_pool.mean().item())
        sd_R = float(R_pool.std(unbiased=False).item())
        if (not math.isfinite(sd_R)) or (sd_R < self.eps):
            sd_R = self.eps

        params = {
            "a": float(self.a),
            "mu": float(mu.item()),
            "lam": float(lam),
            "r_lo": float(q_lo.item()),
            "r_hi": float(q_hi.item()),
            "mu_R": mu_R,
            "sd_R": sd_R,
        }
        self._cache[fold] = params
        return params

    @torch.no_grad()
    def shape(self, fold: int, imm: float) -> Tuple[float, Dict[str, float]]:
        """ clip & z-score norm """
        if not math.isfinite(imm):
            imm = 0.0

        p = self._params_for_fold(fold)
        a, mu, lam = p["a"], p["mu"], p["lam"]
        mu_R, sd_R = p["mu_R"], p["sd_R"]

        R_before_clip = a * imm - lam * (imm - mu) ** 2
        lo, hi = (mu_R - self.clip_k * sd_R, mu_R + self.clip_k * sd_R)
        R_after_clip = float(min(max(R_before_clip, lo), hi))
        R_znorm = (R_after_clip - mu_R) / sd_R

        logs = {
            "R_before_clip": R_before_clip,
            "R_after_clip": R_after_clip,
            "R_znorm": R_znorm,
            "mu": mu,
            "lam": lam,
            "mu_R": mu_R,
            "sd_R": sd_R,
            "lo_R": lo,
            "hi_R": hi,
        }
        return float(R_znorm), logs
