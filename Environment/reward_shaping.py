import math
from typing import Dict, Tuple
import numpy as np
import torch

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
        clip_k: float = 1.0,
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
        
        self._cache: Dict[int, Dict[str, float]] = {}
    
    def _pool_for_fold(self, fold: int) -> torch.Tensor:
        pool = (
            self.alpha_fidelity * self.fidelity_reward[fold] + 
            self.alpha_alignment * self.alignment_reward[fold]
        )
        return torch.as_tensor(pool, dtype=torch.float32)
    
    @torch.no_grad()
    def _params_for_fold(self, fold: int) -> Dict[str, float]:
        if fold in self._cache:
            return self._cache[fold]
        
        t = self._pool_for_fold(fold)
        
        # Quantiles
        q_lo = torch.quantile(t, self.q_lo)
        q_hi = torch.quantile(t, self.q_hi)
        
        # Center (median or mean)
        mu = t.median() if self.use_median else t.mean()
        
        # Radius
        radius = torch.maximum(q_hi - mu, mu - q_lo).clamp_min(1e-8)
        
        # Lambda
        lam_max = self.a / (2.0 * float(radius))
        lam = float(self.frac * lam_max)
        
        # Shaped pool statistics
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
        """
        단일 scalar 보상 shaping (전역 보상용)
        imm: immediate reward (scalar)
        returns: (shaped_reward, logs)
        """
        if not math.isfinite(imm):
            imm = 0.0
        
        p = self._params_for_fold(fold)
        a, mu, lam = p["a"], p["mu"], p["lam"]
        mu_R, sd_R = p["mu_R"], p["sd_R"]
        
        # Quadratic shaping
        R_before_clip = a * imm - lam * (imm - mu) ** 2
        
        # Clipping
        lo = mu_R - self.clip_k * sd_R
        hi = mu_R + self.clip_k * sd_R
        R_after_clip = float(min(max(R_before_clip, lo), hi))
        
        # Z-score normalization
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
    
    @torch.no_grad()
    def shape_batch(self, fold: int, imm_batch: torch.Tensor) -> torch.Tensor:
        if imm_batch.numel() == 0:
            return imm_batch
        
        p = self._params_for_fold(fold)
        a, mu, lam = p["a"], p["mu"], p["lam"]
        mu_R, sd_R = p["mu_R"], p["sd_R"]
        
        # Vectorized quadratic shaping
        R_before_clip = a * imm_batch - lam * (imm_batch - mu) ** 2
        
        # Clipping
        lo = mu_R - self.clip_k * sd_R
        hi = mu_R + self.clip_k * sd_R
        R_after_clip = torch.clamp(R_before_clip, lo, hi)
        
        # Z-score normalization
        R_znorm = (R_after_clip - mu_R) / sd_R
        
        return R_znorm