import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import wandb


class PPO:
    def __init__(self, actor, critic, config, device=None, logger=None):
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor   = actor.to(self.device)
        self.critic  = critic.to(self.device)
        self.config  = config
        self.logger  = logger

        # === hyperparameters ===
        self.clip_ratio   = config.clip_ratio
        self.n_epochs     = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.adv_scale    = config.adv_scale
        self.temperature  = config.temperature

        # === (옵션) 전역 EMA baseline (로그용/백업용) ===
        self.use_ema_baseline = getattr(config, "use_ema_baseline", True)
        self.baseline_beta    = getattr(config, "baseline_beta", 0.9)
        self.baseline_ema     = getattr(config, "baseline_init", 0.0)

        # === 토큰별 크레딧: 선택/미선택 이중 EMA ===
        self.use_token_credit = getattr(config, "use_token_credit", True)
        self.b_sel      = float(getattr(config, "baseline_sel_init", 0.0))
        self.b_nosel    = float(getattr(config, "baseline_nosel_init", 0.0))
        self.beta_sel   = float(getattr(config, "baseline_sel_beta", 0.9))     # 0.9~0.99
        self.beta_nosel = float(getattr(config, "baseline_nosel_beta", 0.9))

        # critic 학습은 선택(원스텝 + 이중 EMA면 False 권장)
        self.train_critic = bool(getattr(config, "train_critic", False))

        # === Entropy targeting ===
        self.target_entropy = float(getattr(config, "target_entropy", 0.30))   # Bernoulli/2-class: 0.2~0.5
        self.entropy_beta   = float(getattr(config, "entropy_beta", 0.02))
        self.entropy_min    = float(getattr(config, "entropy_min", 1e-4))
        self.entropy_max    = float(getattr(config, "entropy_max", 5e-2))

        # === KL targeting / adaptive LR ===
        self.target_kl    = float(getattr(config, "target_kl", 0.01))
        self.kl_stop_mult = float(getattr(config, "kl_stop_mult", 2.0))
        self.lr_actor_now = float(config.lr_actor)
        self.lr_min       = float(getattr(config, "lr_min", 1e-5))
        self.lr_max       = float(getattr(config, "lr_max", 1e-3))

        # === Logit regularization & clamp ===
        self.logit_l2    = float(getattr(config, "logit_l2", 1e-4))
        self.logit_clamp = float(getattr(config, "logit_clamp", 6.0))          # prevent saturation

        # === (Optional) temperature adaptation ===
        self.temperature_min = float(getattr(config, "temperature_min", 1.0))
        self.temperature_max = float(getattr(config, "temperature_max", 2.0))
        self.temp_up         = float(getattr(config, "temp_up", 1.02))
        self.temp_down       = float(getattr(config, "temp_down", 0.98))

        # === optimizers ===
        def split_param_groups(module, head_name="head"):
            decay, no_decay = [], []
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if ("bias" in n) or ("norm" in n.lower()) or ("layernorm" in n.lower()) or (head_name in n):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        lr_actor  = config.lr_actor
        wd_actor  = config.wd_actor
        act_decay, act_no_decay = split_param_groups(self.actor, head_name="head")
        self.optimizer_actor = torch.optim.AdamW(
            [{"params": act_decay, "weight_decay": wd_actor},
             {"params": act_no_decay, "weight_decay": 0.0}],
            lr=lr_actor,
        )

        lr_critic = config.lr_critic
        wd_critic = config.wd_critic
        cri_decay, cri_no_decay = split_param_groups(self.critic, head_name="head")
        self.optimizer_critic = torch.optim.AdamW(
            [{"params": cri_decay, "weight_decay": wd_critic},
             {"params": cri_no_decay, "weight_decay": 0.0}],
            lr=lr_critic,
        )

        self.buffer = []
        self.actor_losses, self.critic_losses = [], []

    @staticmethod
    @torch.no_grad()
    def kl_divergence(old_logits: torch.Tensor, new_logits: torch.Tensor, reduce: str = 'mean') -> torch.Tensor:
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        old_probs     = old_log_probs.exp()
        kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)
        if reduce == 'mean':
            return kl.mean()
        elif reduce == 'sum':
            return kl.sum()
        else:
            return kl.mean(dim=-1)

    @torch.no_grad()
    def act(self, state):
        """
        state: [1, N, D]
        returns: action[N], logp_joint, old_logits_scaled[N,2], h_last[1,H]
        """
        s = state.to(self.device)
        logits, h_last = self.actor.forward_with_hidden(s)   # [1,N,2], [1,H]
        scaled_logits = logits[0] / self.temperature
        dist = Categorical(logits=scaled_logits)
        a    = dist.sample()                 # [N] in {0,1}
        logp_tok = dist.log_prob(a)          # [N]
        logp_joint = logp_tok.sum()
        return a.cpu(), logp_joint.cpu(), scaled_logits.detach().cpu(), h_last.detach().cpu()

    def store_transition(self, state, action, reward, done, logp, old_logits, h_last):
        self.buffer.append({
            "state":      state.detach().cpu(),      # [1,N,D]
            "action":     action.detach().cpu(),     # [N]
            "reward":     float(reward),
            "done":       bool(done),
            "log_prob":   logp.detach().cpu(),
            "old_logits": old_logits.detach().cpu(), # [N,2] (scaled)
            "h_last":     h_last.detach().cpu(),
        })

    def learn(self, fold=None, step_idx=None, grad_clip=None, eps=1e-8):
        assert len(self.buffer) > 0, "Buffer is empty."

        self.actor.train()
        self.critic.train()

        # === stack buffer ===
        rewards = torch.tensor([b['reward'] for b in self.buffer], dtype=torch.float32, device=self.device)   # [T]
        actions = torch.stack([b['action'].to(self.device) for b in self.buffer], dim=0)                      # [T,N]
        old_logits_stack = torch.stack([b['old_logits'].to(self.device) for b in self.buffer])                # [T,N,2]
        states_tensor = torch.cat([b['state'].to(self.device) for b in self.buffer], dim=0)                   # [T,N,D]
        h_lasts = torch.cat([b['h_last'].to(self.device) for b in self.buffer], dim=0)                        # [T,H]

        T = rewards.shape[0]
        mb_size = min(getattr(self, "mb_episodes", getattr(self, "mb_size", getattr(getattr(self, "config", object()), "mb_episodes", T))), T)

        ent_accum, clip_accum, kl_accum = 0.0, 0.0, 0.0
        advstd_accum = 0.0
        selratio_accum, pold_accum, pnew_accum = 0.0, 0.0, 0.0
        batches_seen = 0
        epochs_done  = 0

        for epoch in range(self.n_epochs):
            perm = torch.randperm(T, device=self.device)
            early_stop = False

            for start in range(0, T, mb_size):
                idx = perm[start:start + mb_size]

                s_mb   = states_tensor[idx]           # [M,N,D]
                a_mb   = actions[idx]                 # [M,N]
                old_mb = old_logits_stack[idx]        # [M,N,2]
                r_mb   = rewards[idx]                 # [M]
                h_mb   = h_lasts[idx]                 # [M,H]

                # === per-token 확률 (old policy 권장) ===
                with torch.no_grad():
                    p_old = F.softmax(old_mb, dim=-1)[..., 1]    # [M,N], P(a=1)

                # === 혼합 베이스라인: b_bar = p_old*b_sel + (1-p_old)*b_nosel ===
                b_sel   = torch.as_tensor(self.b_sel,   device=self.device, dtype=torch.float32)
                b_nosel = torch.as_tensor(self.b_nosel, device=self.device, dtype=torch.float32)
                b_bar   = p_old * b_sel + (1.0 - p_old) * b_nosel                                   # [M,N]

                # === per-token advantage ===
                adv_tok_raw = r_mb.unsqueeze(1) - b_bar                                             # [M,N]
                adv_tok = (adv_tok_raw - adv_tok_raw.mean()) / (adv_tok_raw.std(unbiased=False) + 1e-8)
                adv_tok = adv_tok * self.adv_scale

                # === actor ===
                new_logits_mb, _ = self.actor.forward_with_hidden(s_mb)     # [M,N,2]
                if self.logit_clamp is not None and self.logit_clamp > 0:
                    new_logits_mb = new_logits_mb.clamp(-self.logit_clamp, self.logit_clamp)

                new_dist = Categorical(logits=new_logits_mb / self.temperature)
                old_dist = Categorical(logits=old_mb)                        # old는 이미 scaled

                new_lp_tok = new_dist.log_prob(a_mb)                         # [M,N]
                old_lp_tok = old_dist.log_prob(a_mb)                         # [M,N]
                ratio_tok  = (new_lp_tok - old_lp_tok).exp()                 # [M,N]

                surr1 = ratio_tok * adv_tok
                surr2 = torch.clamp(ratio_tok, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_tok

                entropy_term     = new_dist.entropy().mean()
                logit_l2_penalty = (new_logits_mb ** 2).mean()

                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy_term \
                             + self.logit_l2 * logit_l2_penalty

                # === critic (옵션) ===
                if self.train_critic:
                    V_new = self.critic(h_mb).squeeze(-1)
                    critic_loss = F.mse_loss(V_new, r_mb)
                else:
                    critic_loss = torch.tensor(0.0, device=self.device)

                # === step ===
                self.optimizer_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
                self.optimizer_actor.step()

                if self.train_critic:
                    self.optimizer_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
                    self.optimizer_critic.step()

                # === EMA 업데이트(선택/미선택 별) & 진단 ===
                with torch.no_grad():
                    # 선택/미선택 수
                    sel_mask   = a_mb.float()
                    cnt_sel    = sel_mask.sum()
                    cnt_nosel  = sel_mask.numel() - cnt_sel
                    # b_nosel이 업데이트되지 않을 때 대안
                    if cnt_nosel == 0:
                        # 전역 평균 사용 또는 b_sel의 일정 비율
                        effective_b_nosel = self.baseline_ema * 0.5  # 이미 float
                        self.b_nosel = float(self.beta_nosel * self.b_nosel + (1.0 - self.beta_nosel) * effective_b_nosel)
                    if cnt_sel > 0:
                        m_sel = (r_mb.unsqueeze(1) * sel_mask).sum() / cnt_sel
                        self.b_sel = float(self.beta_sel * self.b_sel + (1.0 - self.beta_sel) * m_sel.item())
                    if cnt_nosel > 0:
                        m_nosel = (r_mb.unsqueeze(1) * (1.0 - sel_mask)).sum() / cnt_nosel
                        self.b_nosel = float(self.beta_nosel * self.b_nosel + (1.0 - self.beta_nosel) * m_nosel.item())

                    # KL/clip/entropy
                    kl_now = self.kl_divergence(old_mb, new_logits_mb / self.temperature, reduce=None).mean().item()
                    kl_accum   += kl_now
                    ent_accum  += float(entropy_term.item())
                    clip_accum += float(((ratio_tok - 1.0).abs() > self.clip_ratio).float().mean().item())
                    advstd_accum += float(adv_tok_raw.std(unbiased=False).item())

                    # 선택비율 / 확률 로깅용
                    p_new = F.softmax(new_logits_mb / self.temperature, dim=-1)[..., 1]
                    selratio_accum += float(sel_mask.mean().item())
                    pold_accum     += float(p_old.mean().item())
                    pnew_accum     += float(p_new.mean().item())
                    batches_seen   += 1

                    # ① Entropy targeting
                    H = float(entropy_term.item())
                    self.entropy_coef *= math.exp(self.entropy_beta * (self.target_entropy - H))
                    self.entropy_coef = float(min(max(self.entropy_coef, self.entropy_min), self.entropy_max))

                    # ② KL targeting: LR 적응
                    if kl_now < 0.5 * self.target_kl:
                        self.lr_actor_now = min(self.lr_actor_now * 1.25, self.lr_max)
                    elif kl_now > 2.0 * self.target_kl:
                        self.lr_actor_now = max(self.lr_actor_now / 1.25, self.lr_min)
                    for g in self.optimizer_actor.param_groups:
                        g['lr'] = self.lr_actor_now

                    # ③ (선택) Temperature 적응
                    if (kl_now < 0.5 * self.target_kl) and (H < self.target_entropy):
                        self.temperature = min(self.temperature * self.temp_up, self.temperature_max)
                    elif kl_now > 2.0 * self.target_kl:
                        self.temperature = max(self.temperature * self.temp_down, self.temperature_min)

                    # 조기 종료(과대 KL)
                    if kl_now > self.kl_stop_mult * self.target_kl:
                        early_stop = True

                if early_stop:
                    break

            epochs_done += 1
            if early_stop:
                break

        # === (옵션) 전역 EMA baseline 업데이트 ===
        with torch.no_grad():
            batch_mean_r = rewards.mean().item()
            self.baseline_ema = self.baseline_beta * self.baseline_ema + (1.0 - self.baseline_beta) * batch_mean_r

        # === final stats ===
        with torch.no_grad():
            kl_mean      = kl_accum / max(1, batches_seen)
            entropy_val  = ent_accum / max(1, batches_seen)
            clip_frac    = clip_accum / max(1, batches_seen)
            mean_reward  = float(rewards.mean().item())
            adv_std_raw  = advstd_accum / max(1, batches_seen)
            sel_ratio    = selratio_accum / max(1, batches_seen)
            p_old_mean   = pold_accum / max(1, batches_seen)
            p_new_mean   = pnew_accum / max(1, batches_seen)

        if wandb.run is not None:
            wandb.log({
                f"Fold{fold}/PPO/mean_reward": mean_reward,
                f"Fold{fold}/Loss/actor": float(actor_loss.item()),
                f"Fold{fold}/Loss/critic": float(critic_loss.item()),
                f"Fold{fold}/PPO/entropy": float(entropy_val),
                f"Fold{fold}/PPO/kl_mean": float(kl_mean),
                f"Fold{fold}/PPO/clip_frac": float(clip_frac),
                f"Fold{fold}/PPO/adv_std_raw": float(adv_std_raw),
                f"Fold{fold}/PPO/baseline_ema": float(self.baseline_ema),
                f"Fold{fold}/PPO/entropy_coef": float(self.entropy_coef),
                f"Fold{fold}/PPO/lr_actor_now": float(self.lr_actor_now),
                f"Fold{fold}/PPO/temperature": float(self.temperature),
                f"Fold{fold}/PPO/logit_l2": float(self.logit_l2),
                f"Fold{fold}/PPO/sel_ratio": float(sel_ratio),
                f"Fold{fold}/PPO/p_old_mean": float(p_old_mean),
                f"Fold{fold}/PPO/p_new_mean": float(p_new_mean),
                f"Fold{fold}/PPO/b_sel": float(self.b_sel),
                f"Fold{fold}/PPO/b_nosel": float(self.b_nosel),
                f"Fold{fold}/PPO/b_gap": float(self.b_sel - self.b_nosel),
            })

        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(float(critic_loss.item()))
        self.buffer.clear()

        plt.figure(); plt.plot(self.actor_losses, label='Actor Loss'); plt.legend()
        plt.savefig(f"ppo_actor_loss_fold{fold}.png"); plt.close()
        plt.figure(); plt.plot(self.critic_losses, label='Critic Loss'); plt.legend()
        plt.savefig(f"ppo_critic_loss_fold{fold}.png"); plt.close()

        self.last_stats = {
            "kl_mean": float(kl_mean),
            "entropy": float(entropy_val),
            "clip_frac": float(clip_frac),
            "adv_std_raw": float(adv_std_raw),
            "lr_actor_now": float(self.lr_actor_now),
            "entropy_coef": float(self.entropy_coef),
            "temperature": float(self.temperature),
            "sel_ratio": float(sel_ratio),
            "p_old_mean": float(p_old_mean),
            "p_new_mean": float(p_new_mean),
            "b_sel": float(self.b_sel),
            "b_nosel": float(self.b_nosel),
        }