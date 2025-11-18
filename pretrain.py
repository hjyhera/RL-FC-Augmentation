import os
import math
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_gan_data import GANSyntheticDataset
from dual_network import Actor, Critic  

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_state(X: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    onehot = F.one_hot(y.to(torch.long), num_classes=num_classes).float()
    S = torch.cat([X.float(), onehot], dim=1)
    return S

def build_p_ini(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    N = y.numel()
    counts = torch.bincount(y.to(torch.long), minlength=num_classes).float()  # [C]
    ratios = counts / counts.sum().clamp_min(1.0)                             # [C]
    sel_prob = ratios[y.to(torch.long)]                                       # [N]
    p_ini = torch.stack([1.0 - sel_prob, sel_prob], dim=1)                    # [N, 2]
    return p_ini

def entropy_floor_from_labels(y: torch.Tensor, num_classes: int) -> float:
    counts = torch.bincount(y.to(torch.long), minlength=num_classes).float()  # [C]
    r = counts / counts.sum().clamp_min(1.0)                                  # [C]
    eps = 1e-12
    # H(Bernoulli(r_c)) = - r_c ln r_c - (1-r_c) ln(1-r_c)
    Hc = -(r.clamp_min(eps).log() * r + (1 - r).clamp_min(eps).log() * (1 - r))  # [C]
    # 평균: E_c[H] = sum_c p(c)*H(r_c) = sum_c r_c * H(r_c)
    floor = (r * Hc).sum().item()
    return floor

@torch.no_grad()
def _epoch_eval(agent, X, p_ini, batch_tokens: int, device):
    agent.eval()
    N = X.size(0)
    losses = []
    for start in range(0, N, batch_tokens):
        end = min(start + batch_tokens, N)
        seq = X[start:end].unsqueeze(0).to(device)     # [1, T, F] 
        tgt = p_ini[start:end].unsqueeze(0).to(device) # [1, T, 2]
        logits = agent(seq)                             # [1, T, 2]
        logp = F.log_softmax(logits, dim=-1)
        loss = -(tgt * logp).sum(dim=-1).mean()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def pretrain_agent(
    X_trn: np.ndarray,
    y_trn: np.ndarray,
    *,
    ## actor hyperparameter ##
    d_model: int = 256,
    nhead: int = 8,
    tf_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    lr: float = 1e-3,
    #################
    epochs: int = 30,
    batch_tokens: int = 512,
    grad_clip: float = 1.0,
    seed: int = 42,
    save_path: str = "actor_pretrain.pt",
) -> Tuple[Actor, dict]:
    seed_everything(seed)
    device = get_device()

    X = torch.as_tensor(X_trn, dtype=torch.float32)
    y = torch.as_tensor(y_trn, dtype=torch.long)
    N, feat_dim = X.shape
    C = int(y.max().item()) + 1

    p_ini = build_p_ini(y, C)    # [N, 2]

    agent = Actor(
        input_dim=feat_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=tf_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.Adam(agent.parameters(), lr=lr, weight_decay=0.0)

    stats = {"epoch": [], "loss": [], "val_loss": []}
    idx = torch.arange(N)

    for ep in range(1, epochs + 1):
        agent.train()
        perm = idx[torch.randperm(N)] 

        epoch_losses = []
        for start in range(0, N, batch_tokens):
            end = min(start + batch_tokens, N)
            sel = perm[start:end]

            seq = X[sel].unsqueeze(0).to(device)       # [1, T, F]
            tgt = p_ini[sel].unsqueeze(0).to(device)   # [1, T, 2]

            logits = agent(seq)                        # [1, T, 2]
            logp = F.log_softmax(logits, dim=-1)
            loss = -(tgt * logp).sum(dim=-1).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            optim.step()

            epoch_losses.append(loss.item())

        val_loss = _epoch_eval(agent, X, p_ini, batch_tokens, device)

        stats["epoch"].append(ep)
        stats["loss"].append(float(np.mean(epoch_losses)))
        stats["val_loss"].append(val_loss)

        print(f"[Pretrain] Epoch {ep:03d} | loss={np.mean(epoch_losses):.4f} | val_loss={val_loss:.4f}")

    torch.save({
        "model_state": agent.state_dict(),
        "input_dim": feat_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": tf_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    }, save_path)
    print(f"Saved pretrain agent → {save_path}")

    return agent, stats


if __name__ == "__main__":
    data_dir_gan = r'/home/user/Desktop/intern_project_final/ppo_final/GAN/Pseudo_data/model_20250830_103826'
    
    for fold in range(1, 6):
        print(f"[Fold {fold}] 데이터 로드 및 사전학습 시작")

        gan_ds = GANSyntheticDataset(data_dir_gan, fold)
        X = gan_ds.data.numpy()    # [N, F] 
        y = gan_ds.labels.numpy()  # [N]

        print(f"[Fold {fold}] X shape: {X.shape}, y shape: {y.shape}")

        agent, stats = pretrain_agent(
            X_trn=X,
            y_trn=y,
            d_model=256,
            nhead=8,
            tf_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            lr=1e-3,             
            epochs=50,
            batch_tokens=256,       
            grad_clip=1.0,
            save_path=f"actor_pretrain_fold{fold}.pt"
        )

        print(f"[Fold {fold}] 학습 완료. 모델 저장 → actor_pretrain_fold{fold}.pt\n")
