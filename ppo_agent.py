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

        # === Critic baseline ===
        self.train_critic = True  
        self.critic_coef = float(getattr(config, "critic_coef", 0.5))  

        # === Entropy targeting ===
        self.target_entropy = float(getattr(config, "target_entropy", 0.30))
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
        self.logit_clamp = float(getattr(config, "logit_clamp", 6.0))

        # === Temperature adaptation ===
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

    def store_transition(self, state, action, reward, done, logp, old_logits, h_last, info=None):
        """
        info: dict with optional 'per_sample_rewards' key
        """
        per_sample_rewards = None
        if info is not None and 'per_sample_rewards' in info:
            per_sample_rewards = info['per_sample_rewards'].detach().cpu()
        
        self.buffer.append({
            "state":      state.detach().cpu(),      # [1,N,D]
            "action":     action.detach().cpu(),     # [N]
            "reward":     float(reward),             # scalar 
            "per_sample_rewards": per_sample_rewards,  # [N] or None
            "done":       bool(done),
            "log_prob":   logp.detach().cpu(),
            "old_logits": old_logits.detach().cpu(), # [N,2] (scaled)
            "h_last":     h_last.detach().cpu(),     # [1,H]
        })

    def learn(self, fold=None, step_idx=None, grad_clip=None, eps=1e-8):
        assert len(self.buffer) > 0, "Buffer is empty."

        self.actor.train()
        self.critic.train()

        # === Stack buffer ===
        rewards = torch.tensor([b['reward'] for b in self.buffer], dtype=torch.float32, device=self.device)
        actions = torch.stack([b['action'].to(self.device) for b in self.buffer], dim=0)
        old_logits_stack = torch.stack([b['old_logits'].to(self.device) for b in self.buffer])
        states_tensor = torch.cat([b['state'].to(self.device) for b in self.buffer], dim=0)
        h_lasts = torch.cat([b['h_last'].to(self.device) for b in self.buffer], dim=0)

        # === Per-sample rewards ===
        per_sample_rewards_list = []
        for b in self.buffer:
            psr = b.get('per_sample_rewards')
            if psr is not None:
                per_sample_rewards_list.append(psr.to(self.device))
            else:
                N = b['action'].shape[0]
                per_sample_rewards_list.append(
                    torch.full((N,), b['reward'], dtype=torch.float32, device=self.device)
                )
        
        per_sample_rewards = torch.stack(per_sample_rewards_list, dim=0)  # [T, N]

        T = rewards.shape[0]
        N = actions.shape[1]  
        mb_size = min(getattr(self, "mb_episodes", getattr(self, "mb_size", getattr(getattr(self, "config", object()), "mb_episodes", T))), T)

        ent_accum, clip_accum, kl_accum = 0.0, 0.0, 0.0
        advstd_accum = 0.0
        selratio_accum, pold_accum, pnew_accum = 0.0, 0.0, 0.0
        critic_loss_accum = 0.0
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
                r_sample_mb = per_sample_rewards[idx]  # [M,N] 

                # === Critic forward  ===
                with torch.no_grad():
                    V_old = self.critic(h_mb)  # [M]
              
                V_new = self.critic(h_mb)  # [M]
                
                # === Per-token advantage with Critic baseline ===
                with torch.no_grad():
                    adv_tok_raw = r_sample_mb - V_old.unsqueeze(1)  # [M,N] - [M,1] → [M,N]
                
                adv_tok = (adv_tok_raw - adv_tok_raw.mean()) / (adv_tok_raw.std(unbiased=False) + 1e-8)
                adv_tok = adv_tok * self.adv_scale

                # === Actor ===
                new_logits_mb, _ = self.actor.forward_with_hidden(s_mb)
                if self.logit_clamp is not None and self.logit_clamp > 0:
                    new_logits_mb = new_logits_mb.clamp(-self.logit_clamp, self.logit_clamp)

                new_dist = Categorical(logits=new_logits_mb / self.temperature)
                old_dist = Categorical(logits=old_mb)

                new_lp_tok = new_dist.log_prob(a_mb)  # [M,N]
                old_lp_tok = old_dist.log_prob(a_mb)  # [M,N]
                ratio_tok  = (new_lp_tok - old_lp_tok).exp()  # [M,N]

                surr1 = ratio_tok * adv_tok
                surr2 = torch.clamp(ratio_tok, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_tok

                entropy_term     = new_dist.entropy().mean()
                logit_l2_penalty = (new_logits_mb ** 2).mean()

                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy_term \
                             + self.logit_l2 * logit_l2_penalty

                critic_loss = F.mse_loss(V_new, r_mb)

                # Actor update
                self.optimizer_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
                self.optimizer_actor.step()

                # Critic update
                self.optimizer_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
                self.optimizer_critic.step()

                with torch.no_grad():
                    sel_mask = a_mb.float()  # [M,N]
                    
                    # KL/clip/entropy
                    p_old = F.softmax(old_mb, dim=-1)[..., 1]  # [M,N]
                    kl_now = self.kl_divergence(old_mb, new_logits_mb / self.temperature, reduce=None).mean().item()
                    kl_accum   += kl_now
                    ent_accum  += float(entropy_term.item())
                    clip_accum += float(((ratio_tok - 1.0).abs() > self.clip_ratio).float().mean().item())
                    advstd_accum += float(adv_tok_raw.std(unbiased=False).item())
                    critic_loss_accum += float(critic_loss.item())

                    p_new = F.softmax(new_logits_mb / self.temperature, dim=-1)[..., 1]
                    selratio_accum += float(sel_mask.mean().item())
                    pold_accum     += float(p_old.mean().item())
                    pnew_accum     += float(p_new.mean().item())
                    batches_seen   += 1

                    # Entropy targeting
                    H = float(entropy_term.item())
                    self.entropy_coef *= math.exp(self.entropy_beta * (self.target_entropy - H))
                    self.entropy_coef = float(min(max(self.entropy_coef, self.entropy_min), self.entropy_max))

                    # KL targeting
                    if kl_now < 0.5 * self.target_kl:
                        self.lr_actor_now = min(self.lr_actor_now * 1.25, self.lr_max)
                    elif kl_now > 2.0 * self.target_kl:
                        self.lr_actor_now = max(self.lr_actor_now / 1.25, self.lr_min)
                    for g in self.optimizer_actor.param_groups:
                        g['lr'] = self.lr_actor_now

                    # Temperature adaptation
                    if (kl_now < 0.5 * self.target_kl) and (H < self.target_entropy):
                        self.temperature = min(self.temperature * self.temp_up, self.temperature_max)
                    elif kl_now > 2.0 * self.target_kl:
                        self.temperature = max(self.temperature * self.temp_down, self.temperature_min)

                    # Early stop
                    if kl_now > self.kl_stop_mult * self.target_kl:
                        early_stop = True

                if early_stop:
                    break

            epochs_done += 1
            if early_stop:
                break

        # === Final stats ===
        with torch.no_grad():
            kl_mean      = kl_accum / max(1, batches_seen)
            entropy_val  = ent_accum / max(1, batches_seen)
            clip_frac    = clip_accum / max(1, batches_seen)
            mean_reward  = float(rewards.mean().item())
            adv_std_raw  = advstd_accum / max(1, batches_seen)
            sel_ratio    = selratio_accum / max(1, batches_seen)
            p_old_mean   = pold_accum / max(1, batches_seen)
            p_new_mean   = pnew_accum / max(1, batches_seen)
            critic_loss_mean = critic_loss_accum / max(1, batches_seen)
            
            with torch.no_grad():
                V_mean = self.critic(h_lasts).mean().item()

        if wandb.run is not None:
            wandb.log({
                f"Fold{fold}/PPO/mean_reward": mean_reward,
                f"Fold{fold}/Loss/actor": float(actor_loss.item()),
                f"Fold{fold}/Loss/critic": critic_loss_mean,
                f"Fold{fold}/PPO/entropy": float(entropy_val),
                f"Fold{fold}/PPO/kl_mean": float(kl_mean),
                f"Fold{fold}/PPO/clip_frac": float(clip_frac),
                f"Fold{fold}/PPO/adv_std_raw": float(adv_std_raw),
                f"Fold{fold}/PPO/V_mean": float(V_mean),
                f"Fold{fold}/PPO/entropy_coef": float(self.entropy_coef),
                f"Fold{fold}/PPO/lr_actor_now": float(self.lr_actor_now),
                f"Fold{fold}/PPO/temperature": float(self.temperature),
                f"Fold{fold}/PPO/logit_l2": float(self.logit_l2),
                f"Fold{fold}/PPO/sel_ratio": float(sel_ratio),
                f"Fold{fold}/PPO/p_old_mean": float(p_old_mean),
                f"Fold{fold}/PPO/p_new_mean": float(p_new_mean),
            })

        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(critic_loss_mean)
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
            "V_mean": float(V_mean),
            "critic_loss": critic_loss_mean,
        }