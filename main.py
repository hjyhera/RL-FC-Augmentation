import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import logging
import math
import random
import sys
import time
import traceback
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
GCGAN_DIR = PROJECT_ROOT / "GC-GAN-main" / "GC-GAN-main"
FOLDS = range(1, 6)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self):
        # ── Data paths ────────────────────────────────────────────────────────
        split_root = PROJECT_ROOT / "MDD_Harvard_site20" / "20210331151631_MDD_site20_group"
        val_test_root = PROJECT_ROOT / "MDD_val_test"
        self.train_folds = {f: str(split_root / f"MDD_trainvalid_data_list_fold_{f}.txt") for f in FOLDS}
        self.valid_folds = {f: str(val_test_root / f"fold_{f}_val.txt") for f in FOLDS}
        self.test_folds  = {f: str(val_test_root / f"fold_{f}_test.txt") for f in FOLDS}

        self.data_dir             = str(PROJECT_ROOT / "MS_MDD_Harvard_FC_fisher-20250107T100946Z-001" / "MS_MDD_Harvard_FC_fisher")
        self.topology_dir         = str(PROJECT_ROOT / "GAN" / "Edge_topology")
        self.model_dir            = str(PROJECT_ROOT / "GAN" / "Model")
        self.data_dir_gan         = str(PROJECT_ROOT / "GAN" / "Pseudo_data" / "model_20250830_103826")
        self.model_alignment_root = str(PROJECT_ROOT / "GraphAE" / "Model" / "model_20250828_165235")
        self.info_dir             = str(PROJECT_ROOT / "logs" / "synthetic_info")
        self.precomputed_path     = str(PROJECT_ROOT / "precomputed_rewards_main.npz")

        # ── Reward ───────────────────────────────────────────────
        self.reward_weights = {"F": 1 / 6, "A": 1 / 6, "D": 1 / 3, "U": 1 / 3} 
        self.gpm_p = 1.0                  
        self.alignment_noise_std = 0.1      
        self.clf_every = 100                

        # ── PPO ────────────────────────────────────────────────────────
        self.clip_eps         = 0.5
        self.lr               = 3e-4        
        self.entropy_coef     = 0.5         
        self.n_epochs         = 2
        self.rollout_episodes = 256         
        self.mb_episodes      = 64

        # ── Shared transformer encoder ────────────────────────────────────────
        self.d_model         = 64
        self.nhead           = 2
        self.num_layers      = 2
        self.dim_feedforward = 256
        self.dropout         = 0.25

        # ── Training loop ─────────────────────────────────────────────────────
        self.seed         = 100
        self.max_episodes = 20000
        self.ma_window    = 200             # moving average of r^GPM (logging only)

        # ── Logging ───────────────────────────────────────────────────────────
        self.log_to_wandb   = True
        self.wandb_project  = "RL-FC-Augmentation"
        self.wandb_run_name = "ppo_" + time.strftime("%Y%m%d_%H%M%S")


# ══════════════════════════════════════════════════════════════════════════════
# Seeds
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


# ══════════════════════════════════════════════════════════════════════════════
# FC utilities & datasets
# ══════════════════════════════════════════════════════════════════════════════

def flatten_fc(matrix: np.ndarray) -> np.ndarray:
    """Return the upper triangle of a square FC matrix, excluding the diagonal."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def unflatten_fc(flat_vector: torch.Tensor, n: int = 112) -> torch.Tensor:
    """Restore a symmetric FC matrix (zero diagonal) from an upper-triangular vector."""
    flat_vector = flat_vector.reshape(-1)
    indices = torch.triu_indices(n, n, offset=1, device=flat_vector.device)
    matrix = torch.zeros(n, n, dtype=flat_vector.dtype, device=flat_vector.device)
    matrix[indices[0], indices[1]] = flat_vector
    return matrix + matrix.T


def minmax_to_unit(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale values to [0, 1], returning neutral scores for a constant array."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < eps:
        return np.full_like(values, 0.5)
    return ((values - lo) / (hi - lo)).astype(np.float32)


class BrainDataset(torch.utils.data.Dataset):
    """Real subject FC matrices listed in a split file (label 1 = MDD, 0 = NC)."""

    def __init__(self, txt_file, data_dir):
        with open(txt_file, "r") as f:
            self.file_list = f.read().splitlines()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        fc_matrix = flatten_fc(np.load(os.path.join(self.data_dir, file_name)))
        label = 1 if file_name.split("-")[1] == "1" else 0
        return torch.tensor(fc_matrix, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class GANSyntheticDataset(torch.utils.data.Dataset):
    """Fixed candidate pool drawn from the fold's pretrained GAN generator (§2.1)."""

    def __init__(self, data_dir_gan, fold):
        files = os.listdir(data_dir_gan)
        data_files  = sorted(f for f in files if f.startswith(f"pseudo_data_{fold}_") and f.endswith(".npy"))
        label_files = sorted(f for f in files if f.startswith(f"pseudo_label_{fold}_") and f.endswith(".npy"))

        if not data_files:
            raise FileNotFoundError(f"No GAN samples found for fold {fold} in {data_dir_gan}")
        if len(data_files) != len(label_files):
            raise ValueError(f"Fold {fold} has {len(data_files)} data files and {len(label_files)} label files")

        all_fc, all_label = [], []
        for data_file, label_file in zip(data_files, label_files):
            data = np.load(os.path.join(data_dir_gan, data_file))     # (num_samples, n, n)
            labels = np.load(os.path.join(data_dir_gan, label_file))  # (num_samples,)
            if len(data) != len(labels):
                raise ValueError(f"{data_file} and {label_file} contain different sample counts")
            all_fc.append(np.asarray([flatten_fc(sample) for sample in data]))
            all_label.append(labels)
        self.data = torch.tensor(np.concatenate(all_fc, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(all_label, axis=0), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dataset_to_tensors(dataset):
    if len(dataset) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(xs), torch.stack(ys)


def load_fold_data(cfg, fold):
    return {
        "train": dataset_to_tensors(BrainDataset(cfg.train_folds[fold], cfg.data_dir)),
        "val":   dataset_to_tensors(BrainDataset(cfg.valid_folds[fold], cfg.data_dir)),
        "test":  dataset_to_tensors(BrainDataset(cfg.test_folds[fold], cfg.data_dir)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Classifier 
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "acc":  float(accuracy_score(y_true, y_pred)),
        "sens": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1":   float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def classifier(train, evals, seed):
    """
    train : (X, y) numpy arrays used for fitting
    evals : {split_name: (X, y)} evaluated after fitting
    Returns {split_name: metrics}
    """
    X_train, y_train = train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 64),
        activation="tanh",
        solver="sgd",
        alpha=0.001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.02,
        max_iter=200,
        shuffle=True,
        random_state=seed,
        early_stopping=False,
        tol=1e-4,
    )
    model.fit(X_train, y_train)

    results = {}
    for name, (X, y) in evals.items():
        if len(X) == 0:
            continue
        X = scaler.transform(X)
        results[name] = calculate_metrics(y, model.predict(X), model.predict_proba(X))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Reward components
# ══════════════════════════════════════════════════════════════════════════════

def load_gcgan_modules():
    """Import the GC-GAN generator/discriminator/denoising-AE definitions.

    GC-GAN's config parses sys.argv at import time, so it must not see this script's CLI.
    """
    if str(GCGAN_DIR) not in sys.path:
        sys.path.insert(0, str(GCGAN_DIR))
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from model import ACDiscriminator, myGraphAE
    finally:
        sys.argv = argv
    return ACDiscriminator, myGraphAE


@torch.no_grad()
def fidelity_score(discriminator, mat, topo):
    logits, _ = discriminator(mat, topo)                 # [1, 4]: NC_real, MDD_real, NC_fake, MDD_fake
    return float((logits[0, 0] + logits[0, 1] - logits[0, 2] - logits[0, 3]).item())


@torch.no_grad()
def alignment_score(autoencoder, flat_vec, label, topo, noise_std):
    n = 112
    mat = unflatten_fc(flat_vec, n)
    noisy = mat + noise_std * torch.randn_like(mat)
    label_emb = autoencoder.embedding(label.view(1)).view(1, n, 1)

    # Enc
    h = torch.cat([noisy.unsqueeze(0), label_emb], dim=2)
    h = autoencoder.act(autoencoder.econv1(h, topo))
    z = autoencoder.act(autoencoder.econv2(h, topo))
    # Dec
    h = torch.cat([z, label_emb], dim=2)
    h = autoencoder.act(autoencoder.gconv1(h, topo))
    h = autoencoder.act(autoencoder.gconv2(h, topo))
    recon = autoencoder.tanh(autoencoder.linear(h.reshape(1, -1))).reshape(n, n)
    recon = torch.arctanh(0.999 * 0.5 * (recon + recon.T))

    iu = torch.triu_indices(n, n, offset=1, device=recon.device)
    return float(-((flat_vec - recon[iu[0], iu[1]]) ** 2).sum().item())


def precompute_rewards(cfg, device):
    ACDiscriminator, myGraphAE = load_gcgan_modules()
    seed_everything(cfg.seed, deterministic=False)

    out = {}
    for fold in FOLDS:
        topo_path = os.path.join(cfg.topology_dir, f"mrmr_MDD_Harvard_FC_map_fold_{fold}.npy")
        topo = torch.tensor(np.load(topo_path), dtype=torch.float32, device=device)

        discriminator = ACDiscriminator().to(device).eval()
        discriminator.load_state_dict(torch.load(
            os.path.join(cfg.model_dir, f"D_model_{fold}_final.pth"), map_location=device, weights_only=True))
        autoencoder = myGraphAE().to(device).eval()
        autoencoder.load_state_dict(torch.load(
            os.path.join(cfg.model_alignment_root, f"GAE_model_{fold}_final.pth"), map_location=device, weights_only=True))

        r_f, r_a = [], []
        for flat_vec, label in tqdm(GANSyntheticDataset(cfg.data_dir_gan, fold), desc=f"Fold {fold}"):
            flat_vec, label = flat_vec.to(device), label.to(device)
            r_f.append(fidelity_score(discriminator, unflatten_fc(flat_vec), topo))
            r_a.append(alignment_score(autoencoder, flat_vec, label, topo, cfg.alignment_noise_std))

        out[f"fidelity_raw_{fold}"]  = np.asarray(r_f, dtype=np.float32)
        out[f"alignment_raw_{fold}"] = np.asarray(r_a, dtype=np.float32)
        out[f"fidelity_{fold}"]      = minmax_to_unit(out[f"fidelity_raw_{fold}"])
        out[f"alignment_{fold}"]     = minmax_to_unit(out[f"alignment_raw_{fold}"])
        print(f"[Fold {fold}] {len(r_f)} candidates | "
              f"r_F raw mean={np.mean(r_f):.4f} | r_A raw mean={np.mean(r_a):.4f}")

    np.savez_compressed(cfg.precomputed_path, **out)
    print(f"Saved rewards → {cfg.precomputed_path}")


def gpu_LS(real: torch.Tensor, gen: torch.Tensor) -> float:
    if len(real) < 2 or len(gen) < 2:
        return 0.0

    dist_real = torch.cdist(real, real)                               # ICD 1
    dist_real = torch.flatten(torch.tril(dist_real, diagonal=-1))     # remove repeats
    dist_real = dist_real[dist_real.nonzero()].flatten()              # remove distance=0 for distances btw same data points
    dist_real = dist_real.cpu().numpy()

    dist_gen = torch.cdist(gen, gen)                                  # ICD 2
    dist_gen = torch.flatten(torch.tril(dist_gen, diagonal=-1))
    dist_gen = dist_gen[dist_gen.nonzero()].flatten()
    dist_gen = dist_gen.cpu().numpy()

    distbtw = torch.flatten(torch.cdist(gen, real)).cpu().numpy()     # BCD

    D_Sep_1, _ = ks_2samp(dist_real, distbtw)
    D_Sep_2, _ = ks_2samp(dist_gen, distbtw)

    return float(1 - np.max([D_Sep_1, D_Sep_2]))  # LS = 1 − DSI


def utility_score(val_loss: float, ref_loss: float) -> float:
    """
    r_U relative to the real-only classifier's validation loss L₀:
        clip(0.5 + (L₀ − L) / (2·L₀), 0, 1)   → 0.5 at L = L₀, 1 at L = 0, 0 at L = 2·L₀
    """
    return float(np.clip(0.5 + (ref_loss - val_loss) / (2 * ref_loss), 0.0, 1.0))


def generalized_power_mean(components: dict, weights: dict, p: float):
    return sum(weights[k] * components[k] ** p for k in weights) ** (1.0 / p)


# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

class Environment:
    """
    State  : candidate FC matrices s_t (the fold's fixed GAN pool)
    Action : a_t ∈ {0, 1} per candidate (1 = select)
    Reward : R(S) = GPM of [r_F, r_D, r_A, r_U] computed on the selected set S, shared by every candidate
    """

    def __init__(self, cfg, fold, real, synthetic, val, test, device):
        self.cfg = cfg
        self.fold = fold
        self.device = device

        self.real_x, self.real_y = (t.to(device) for t in real)
        self.val, self.test = val, test
        self.cand_x = synthetic.data.to(device)          # [N, D]
        self.cand_y = synthetic.labels.to(device)        # [N]

        rewards = np.load(cfg.precomputed_path)
        self.r_fidelity  = torch.tensor(rewards[f"fidelity_{fold}"],  device=device)
        self.r_alignment = torch.tensor(rewards[f"alignment_{fold}"], device=device)
        if len(self.r_fidelity) != len(self.cand_x):
            raise ValueError(f"Precomputed rewards cover {len(self.r_fidelity)} candidates, pool has {len(self.cand_x)}")

        # Reference for r_U: validation loss of the classifier trained on real data only
        no_selection = torch.empty(0, dtype=torch.long, device=device)
        self.ref_val_loss = self.evaluate_classifier(no_selection, include_test=False)["val"]["loss"]
        self.r_utility = None

        self.episode = 0
        self.last_reward_components: dict = {}

    def reset(self) -> torch.Tensor:
        """Returns state [1, N, D] — all candidate FC vectors of the fold."""
        return self.cand_x.unsqueeze(0)

    def step(self, action: torch.Tensor):
        """
        action : [N] tensor, 1 = select
        Returns (episode reward R(S), per-candidate rewards [N] all equal to R(S))
        """
        self.episode += 1
        sel = action.to(self.device).nonzero(as_tuple=False).squeeze(1)

        # r_U: negative validation loss of the classifier trained on real + selected
        if (self.episode - 1) % self.cfg.clf_every == 0:
            metrics = self.evaluate_classifier(sel, include_test=False)
            self.r_utility = utility_score(metrics["val"]["loss"], self.ref_val_loss)

        # r_D: likeness score between the selected set and the real training set 
        r_diversity = gpu_LS(self.real_x, self.cand_x[sel])

        # r_F, r_A: mean over the selected set
        r_fidelity  = float(self.r_fidelity[sel].mean()) if len(sel) else 0.0
        r_alignment = float(self.r_alignment[sel].mean()) if len(sel) else 0.0

        # Set-level reward of the selected combination, received by every candidate
        components = {"F": r_fidelity, "D": r_diversity, "A": r_alignment, "U": self.r_utility}
        episode_reward = float(generalized_power_mean(components, self.cfg.reward_weights, self.cfg.gpm_p))
        rewards = torch.full((len(action),), episode_reward, dtype=torch.float32, device=self.device)

        self.last_reward_components = {
            "rF":         r_fidelity,
            "rD":         r_diversity,
            "rA":         r_alignment,
            "rU":         self.r_utility,
            "n_selected": len(sel),
            "sel_ratio":  len(sel) / len(action),
            "reward":     episode_reward,
        }
        return episode_reward, rewards

    def evaluate_classifier(self, sel: torch.Tensor, include_test: bool) -> dict:
        """Train the classifier on real + selected candidates (Fig. 1C) and evaluate it."""
        train_x = torch.cat([self.real_x, self.cand_x[sel]]).cpu().numpy()
        train_y = torch.cat([self.real_y, self.cand_y[sel]]).cpu().numpy()
        evals = {"val": self.val}
        if include_test:
            evals["test"] = self.test
        evals = {k: (x.numpy(), y.numpy()) for k, (x, y) in evals.items()}
        return classifier((train_x, train_y), evals, seed=self.cfg.seed)


# ══════════════════════════════════════════════════════════════════════════════
# Actor–critic
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.25):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Actor head: π_θ(a_t | s_t) logits per candidate
        self.actor_head = nn.Linear(d_model, 2)
        # Critic head: V_φ(s_t) per candidate
        self.critic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero logits at init → every candidate starts with π(a_t = 1 | s_t) = 0.5
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic_head[-1].weight)
        nn.init.zeros_(self.critic_head[-1].bias)

    def forward(self, x: torch.Tensor):
        """x [B, N, D] → (logits [B, N, 2], value [B, N])"""
        h = self.encoder(self.input_proj(x) * math.sqrt(self.d_model))
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# PPO
# ══════════════════════════════════════════════════════════════════════════════

class PPO:
    def __init__(self, model: ActorCritic, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.buffer: list = []
        self.last_stats: dict = {}
        self.last_p_select = 0.5

    @torch.no_grad()
    def act(self, state: torch.Tensor, greedy: bool = False):
        """state [1, N, D] → (action [N], log_prob [N], value [N])"""
        self.model.eval()
        logits, value = self.model(state.to(self.device))
        dist = Categorical(logits=logits[0])
        action = logits[0].argmax(dim=-1) if greedy else dist.sample()
        self.last_p_select = float(dist.probs[:, 1].mean())
        return action, dist.log_prob(action), value[0]

    def store_transition(self, state, action, log_prob, value, rewards):
        self.buffer.append({
            "state": state, "action": action, "log_prob": log_prob, "value": value, "reward": rewards,
        })

    def learn(self):
        self.model.train()
        n_transitions = len(self.buffer)
        mb_size = min(self.cfg.mb_episodes, n_transitions)
        stats = {}

        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(n_transitions).tolist()
            for start in range(0, n_transitions, mb_size):
                mb = [self.buffer[i] for i in perm[start:start + mb_size]]
                states   = torch.cat([b["state"] for b in mb])          # [M, N, D]
                actions  = torch.stack([b["action"] for b in mb])       # [M, N]
                old_logp = torch.stack([b["log_prob"] for b in mb])     # [M, N]
                old_v    = torch.stack([b["value"] for b in mb])        # [M, N]
                rewards  = torch.stack([b["reward"] for b in mb])       # [M, N]

                logits, value = self.model(states)
                dist = Categorical(logits=logits)
                # Per-candidate ratio, advantage and clipping: ρ_t, A_t, clip(ρ_t) are all [M, N]
                ratio = (dist.log_prob(actions) - old_logp).exp()               # [M, N]
                adv = rewards - old_v                                           # [M, N]
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)     # normalised over the mini-batch

                surrogate = torch.min(ratio * adv,
                                      ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv)  # [M, N]
                entropy = dist.entropy().mean()
                actor_loss = -surrogate.mean() - self.cfg.entropy_coef * entropy
                critic_loss = ((rewards - value) ** 2).mean()

                loss = actor_loss + critic_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norms = {
                    "grad_norm/total":       grad_norm(self.model.parameters()),
                    "grad_norm/encoder":     grad_norm(list(self.model.input_proj.parameters())
                                                       + list(self.model.encoder.parameters())),
                    "grad_norm/actor_head":  grad_norm(self.model.actor_head.parameters()),
                    "grad_norm/critic_head": grad_norm(self.model.critic_head.parameters()),
                }
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = ratio.log()
                    mb_stats = {
                        "loss/actor":     float(actor_loss),
                        "loss/critic":    float(critic_loss),
                        "loss/total":     float(loss),
                        "ppo/entropy":    float(entropy),
                        "ppo/approx_kl":  float(((ratio - 1) - log_ratio).mean()),
                        "ppo/clip_frac":  float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean()),
                        "ppo/ratio_mean": float(ratio.mean()),
                        "value/mean":     float(value.mean()),
                        "adv/mean":       float(adv.mean()),
                        "adv/std":        float(adv.std()),
                        **grad_norms,
                    }
                for k, v in mb_stats.items():
                    stats.setdefault(k, []).append(v)

        # Averages over all mini-batches of this update
        self.last_stats = {k: float(np.mean(v)) for k, v in stats.items()}
        self.last_stats["ppo/sel_ratio"] = float(torch.stack([b["action"] for b in self.buffer]).float().mean())
        self.buffer.clear()


def grad_norm(params) -> float:
    """L2 norm over the gradients of the given parameters (no clipping applied)."""
    norms = [p.grad.detach().norm() for p in params if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation per fold
# ══════════════════════════════════════════════════════════════════════════════

def get_device(fold, force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{(fold - 1) % torch.cuda.device_count()}")


def build_actor_critic(cfg, input_dim):
    return ActorCritic(
        input_dim=input_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )


def save_selection(env, sel, out_dir):
    """Selected indices and class-wise mean FC of the final selection (for FC-motif analysis)."""
    fold = env.fold
    sel_idx = sel.tolist()
    labels = env.cand_y[sel].tolist()
    with open(os.path.join(out_dir, f"selected_idx_fold{fold}.txt"), "w") as f:
        f.write(f"Total selected: {len(sel_idx)}\nIndex\tLabel\n")
        for idx, lbl in zip(sel_idx, labels):
            f.write(f"{idx}\t{lbl}\n")

    x, y = env.cand_x[sel].cpu(), env.cand_y[sel].cpu()
    means = {}
    for name, c in (("nc_mean", 0), ("mdd_mean", 1)):
        if (y == c).any():
            means[name] = unflatten_fc(x[y == c].mean(0)).numpy()
    if len(means) == 2:
        means["mdd_minus_nc"] = means["mdd_mean"] - means["nc_mean"]
    np.savez(os.path.join(out_dir, f"selection_stats_fold{fold}.npz"),
             selected_idx=np.asarray(sel_idx, dtype=np.int64), **means)


def train_single_fold(fold, cfg_dict, result_queue, force_cpu=False):
    try:
        seed_everything(cfg_dict["seed"])
        cfg = Config()
        cfg.__dict__.update(cfg_dict)

        device = get_device(fold, force_cpu)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        print(f"[Fold {fold}] Training on {device} (PID {os.getpid()})")

        logger = logging.getLogger(f"Fold{fold}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(cfg.info_dir, f"fold{fold}_run.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers = [fh]

        run = None
        if cfg.log_to_wandb:
            run = wandb.init(project=cfg.wandb_project, group=cfg.wandb_run_name,
                             name=f"{cfg.wandb_run_name}_fold{fold}", job_type="train", config=cfg_dict)

        data = load_fold_data(cfg, fold)
        synthetic = GANSyntheticDataset(cfg.data_dir_gan, fold)
        env = Environment(cfg, fold, data["train"], synthetic, data["val"], data["test"], device)

        model = build_actor_critic(cfg, input_dim=synthetic.data.shape[1])
        ppo = PPO(model, cfg, device)

        last_path = os.path.join(cfg.info_dir, f"last_actor_fold{fold}.pt")
        episode_rewards = deque(maxlen=cfg.ma_window)
        selection_history = []

        pbar = tqdm(range(1, cfg.max_episodes + 1), desc=f"Fold {fold}", position=fold - 1, leave=True, ncols=120)
        for episode in pbar:
            state = env.reset()
            action, log_prob, value = ppo.act(state)
            reward, rewards = env.step(action)
            ppo.store_transition(state, action, log_prob, value, rewards)

            episode_rewards.append(reward)
            selection_history.append({
                "episode": episode,
                "sel_ratio": float(action.float().mean()),
                "reward": reward,
                "reward_components": dict(env.last_reward_components),
            })

            comps = env.last_reward_components
            log = {
                "reward/episode":       reward,
                "reward/rF":            comps["rF"],
                "reward/rD":            comps["rD"],
                "reward/rA":            comps["rA"],
                "reward/rU":            comps["rU"],
                "selection/ratio":      comps["sel_ratio"],
                "selection/n_selected": comps["n_selected"],
                "policy/p_select_mean": ppo.last_p_select,
            }

            if len(ppo.buffer) >= cfg.rollout_episodes:
                ppo.learn()
                log.update(ppo.last_stats)
                logger.info(f"episode={episode} " + " ".join(f"{k}={v:.4f}" for k, v in ppo.last_stats.items()))

            score = float(np.mean(episode_rewards))
            log["reward/moving_avg"] = score
            if run is not None:
                run.log(log, step=episode)

            pbar.set_postfix({
                "reward": f"{score:.3f}",
                "sel": f"{selection_history[-1]['sel_ratio']:.2f}",
            })
        pbar.close()
        torch.save(ppo.model.state_dict(), last_path)

        with open(os.path.join(cfg.info_dir, f"selection_history_fold{fold}.json"), "w") as f:
            json.dump(selection_history, f, indent=2)

        # Final selection with the last policy (greedy), then classifier training + inference (Fig. 1C, 1D)
        action, _, _ = ppo.act(env.reset(), greedy=True)
        sel = action.nonzero(as_tuple=False).squeeze(1)
        metrics = env.evaluate_classifier(sel, include_test=True)
        save_selection(env, sel, cfg.info_dir)

        logger.info(f"final selection: k={len(sel)}, ratio={len(sel) / len(action):.4f}")
        for split in ("val", "test"):
            logger.info(f"{split}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics[split].items()))

        if run is not None:
            final = {f"final/{split}_{k}": v for split in ("val", "test") for k, v in metrics[split].items()}
            final.update({"final/n_selected": len(sel), "final/sel_ratio": len(sel) / len(action),
                          "final/episodes_trained": episode})
            run.summary.update(final)
            run.finish()

        result_queue.put({
            "fold": fold,
            "test_metrics": metrics["test"],
            "n_selected": len(sel),
            "episodes_trained": episode,
            "success": True,
        })
        print(f"[Fold {fold}] Done ({episode} episodes, {len(sel)} candidates selected)")

    except Exception as e:
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        result_queue.put({"fold": fold, "error": str(e), "success": False})


def summarize_cv(all_metrics, title, log_to_wandb=False):
    keys = all_metrics[0].keys()
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in keys}
    print(f"\n=== {title} ({len(all_metrics)}-fold CV, test) ===")
    for k in keys:
        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
    if log_to_wandb and wandb.run is not None:
        wandb.run.summary.update({**{f"cv/{k}_mean": avg[k] for k in keys}, **{f"cv/{k}_std": std[k] for k in keys}})
    return avg, std


def train(force_cpu=False):
    cfg = Config()
    os.makedirs(cfg.info_dir, exist_ok=True)
    if not os.path.exists(cfg.precomputed_path):
        raise FileNotFoundError(f"{cfg.precomputed_path} not found — run `python main.py precompute` first")

    result_queue = mp.Queue()
    start = time.time()
    processes = [mp.Process(target=train_single_fold, args=(fold, vars(cfg), result_queue, force_cpu))
                 for fold in FOLDS]
    for p in processes:
        p.start()
    results = [result_queue.get() for _ in processes]
    for p in processes:
        p.join()

    results.sort(key=lambda r: r["fold"])
    for r in results:
        if not r["success"]:
            print(f"Fold {r['fold']} failed: {r['error']}")
    test_metrics = [r["test_metrics"] for r in results if r["success"]]
    if test_metrics:
        if cfg.log_to_wandb:
            wandb.init(project=cfg.wandb_project, group=cfg.wandb_run_name,
                       name=f"{cfg.wandb_run_name}_cv", job_type="cv", config=vars(cfg))
        summarize_cv(test_metrics, "PPO selection", log_to_wandb=cfg.log_to_wandb)
        if cfg.log_to_wandb:
            wandb.finish()
    print(f"\nTotal time: {(time.time() - start) / 60:.1f} minutes")
    return results


def baseline():
    """No-augmentation control: classifier trained on real training subjects only."""
    cfg = Config()
    seed_everything(cfg.seed)
    all_metrics = []
    for fold in FOLDS:
        data = load_fold_data(cfg, fold)
        to_np = lambda split: tuple(t.numpy() for t in data[split])
        metrics = classifier(to_np("train"), {"test": to_np("test")}, seed=cfg.seed)
        print(f"Fold {fold}: {metrics['test']}")
        all_metrics.append(metrics["test"])
    summarize_cv(all_metrics, "No augmentation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["precompute", "baseline", "train"])
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()

    if args.command == "precompute":
        use_cuda = torch.cuda.is_available() and not args.cpu
        precompute_rewards(Config(), torch.device("cuda" if use_cuda else "cpu"))
    elif args.command == "baseline":
        baseline()
    else:
        mp.set_start_method("spawn", force=True)
        train(force_cpu=args.cpu)
