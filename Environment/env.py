# import os
# import random
# import numpy as np
# import math

# SEED = 100

# os.environ["PYTHONHASHSEED"] = str(SEED)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# import torch  
# import wandb
# from collections import deque

# def seed_rng(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     g = torch.Generator()   
#     g.manual_seed(seed)
#     return g

# def make_dl_generator(seed: int):
#     g = torch.Generator()
#     g.manual_seed(seed)
#     return g

# def seed_worker(worker_id):
#     worker_seed = (torch.initial_seed() + worker_id) % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# import sys
# import time
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
# from load_brain_data import BrainDataset
# from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
# import wandb
# from compute_LS import gpu_LS
# import logging

# from mlp_classifier import classifier
# from load_gan_data import GANSyntheticDataset
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from .reward_shaping import Shaping

# class Environment:
#     def __init__(self, config, real_data, synthetic_data, val_data, test_data):
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # dataset
#         self.real_data = real_data
#         self.synthetic_data = synthetic_data
#         self.val_data = val_data
#         self.test_data = test_data

#         # imm reward 가중치
#         self.alpha_fidelity = config.alpha_fidelity
#         self.alpha_alignment = config.alpha_alignment
#         self.alpha_diversity = config.alpha_diversity

#         self.fidelity_reward, self.alignment_reward = self._load_precomputed(config.precomputed_path)

#         # 선택 기록
#         self.selected_data    = {f: [] for f in synthetic_data.keys()}
#         self.selected_indices = {f: [] for f in synthetic_data.keys()}

#         # shaping
#         self.shaping = Shaping(
#             alpha_fidelity=self.alpha_fidelity,
#             alpha_alignment=self.alpha_alignment,
#             fidelity_reward=self.fidelity_reward,
#             alignment_reward=self.alignment_reward,
#             a=config.a,
#             frac=config.frac,
#             q_lo=config.q_lo,
#             q_hi=config.q_hi,
#             use_median=config.use_median,
#             clip_k=config.clip_k,
#             eps=config.eps,
#         )


#         # validation reward
#         self.clf_every       = config.clf_every
#         self.val_metric      = config.val_metric
#         self.alpha_utility      = config.alpha_utility
#         self.fold_steps      = {f: 0 for f in synthetic_data.keys()}
#         self.last_val_reward = {f: 0.0 for f in synthetic_data.keys()}

#     def _load_precomputed(self, path):
#         npz = np.load(path, allow_pickle=True)
#         if "fidelity_good" in npz and "alignment_good" in npz:
#             fidelity = npz["fidelity_good"].item()
#             alignment = npz["alignment_good"].item()
#         else:
#             raise KeyError(f"Precomputed file missing expected keys. Found: {list(npz.keys())}.")
#         fidelity = {int(f): np.asarray(v, dtype=np.float32) for f, v in fidelity.items()}
#         alignment = {int(f): np.asarray(v, dtype=np.float32) for f, v in alignment.items()}
#         return fidelity, alignment

#     def _flat_to_mat(self, flat_vec: torch.Tensor) -> torch.Tensor:
#         """
#         flat_vec: [F] (F = n*(n-1)/2). 반환: [n,n] 대칭행렬
#         """
#         F = int(flat_vec.numel())
#         n = int((1 + math.sqrt(1 + 8*F)) // 2)
#         device = flat_vec.device
#         idx = torch.triu_indices(n, n, offset=1, device=device)
#         mat = torch.zeros(n, n, device=device, dtype=flat_vec.dtype)
#         mat[idx[0], idx[1]] = flat_vec
#         return mat + mat.T

#     @torch.no_grad()
#     def save_selection_artifacts(self, fold: int, out_dir: str):
#         import seaborn as sns

#         os.makedirs(out_dir, exist_ok=True)
#         sel_idx = self.selected_indices.get(fold, [])
#         if not sel_idx:
#             print(f"[Fold {fold}] No selected samples. Skipping save.")
#             return

#         # 선택 벡터/라벨 수집
#         vecs, labels = [], []
#         for i in sel_idx:
#             v, lbl = self.synthetic_data[fold][i]
#             vecs.append(v.detach().cpu().float())
#             labels.append(int(lbl))
#         X = torch.stack(vecs, dim=0)           # [k, F]
#         y = torch.tensor(labels, dtype=torch.long)

#         # 그룹 분리
#         is_mdd = (y == 1)
#         is_nc  = (y == 0)

#         def vecs_to_mean_std_mats(Xg: torch.Tensor):
#             if Xg.numel() == 0:
#                 return None, None
#             mean_vec = Xg.mean(dim=0)
#             std_vec  = Xg.std(dim=0, unbiased=False)
#             mean_mat = self._flat_to_mat(mean_vec).cpu().numpy()
#             std_mat  = self._flat_to_mat(std_vec).cpu().numpy()
#             return mean_mat, std_mat

#         mdd_mean, mdd_std = vecs_to_mean_std_mats(X[is_mdd]) if is_mdd.any() else (None, None)
#         nc_mean,  nc_std  = vecs_to_mean_std_mats(X[is_nc])  if is_nc.any()  else (None, None)

#         # 1) 인덱스+라벨 저장(.txt)
#         txt_path = os.path.join(out_dir, f"selected_idx_fold{fold}.txt")
#         with open(txt_path, "w") as f:
#             f.write(f"Total selected: {len(sel_idx)}\nIndex\tLabel\n")
#             for idx, lbl in zip(sel_idx, labels):
#                 f.write(f"{idx}\t{lbl}\n")
#         print(f"[Fold {fold}] Saved indices -> {txt_path}")

#         # 2) 통계 행렬 저장(.npz)
#         npz_path = os.path.join(out_dir, f"selection_stats_fold{fold}.npz")
#         np.savez(
#             npz_path,
#             selected_idx=np.asarray(sel_idx, dtype=np.int64),
#             mdd_mean=mdd_mean, mdd_std=mdd_std,
#             nc_mean=nc_mean,   nc_std=nc_std,
#         )
#         print(f"[Fold {fold}] Saved stats -> {npz_path}")

#         # ---- 공통 저장 유틸 ----
#         def _save_mat_csv_npy(name: str, mat: np.ndarray):
#             if mat is None:
#                 return
#             np.save(os.path.join(out_dir, f"{name}.npy"), mat)
#             np.savetxt(os.path.join(out_dir, f"{name}.csv"), mat, delimiter=",")

#         def _save_heatmap_jet(name: str, mat: np.ndarray, vmin=None, vmax=None):
#             if mat is None:
#                 return
#             fig, ax = plt.subplots(figsize=(6, 5))
#             sns.heatmap(
#                 mat, ax=ax, annot=False, fmt=".2f",
#                 linewidths=0, cmap="jet", cbar=True,
#                 xticklabels=False, yticklabels=False,
#                 vmin=vmin, vmax=vmax
#             )
#             ax.set_title(name)
#             fig.tight_layout()
#             png_path = os.path.join(out_dir, f"{name}.png")
#             fig.savefig(png_path, dpi=200)
#             plt.close(fig)
#             print(f"[Fold {fold}] Saved heatmap -> {png_path}")

#         # 3) 선택 전체 평균 FC (스케일 기준)
#         mats = [self._flat_to_mat(x).cpu().numpy() for x in X]  # list of [n,n]
#         mean_selected = np.mean(np.stack(mats, axis=0), axis=0)

#         _save_mat_csv_npy(f"mean_selected_fold{fold}", mean_selected)
#         # 전체 평균 스케일(예시와 동일): vmin=-0.1, vmax=1.5
#         vmin_mean, vmax_mean = -0.1, 1.5
#         _save_heatmap_jet(f"mean_generated_fold{fold}", mean_selected, vmin=vmin_mean, vmax=vmax_mean)

#         # 4) 그룹 평균: 히트맵은 '전체 평균'과 같은 vmin/vmax를 강제
#         _save_mat_csv_npy(f"fold{fold}_MDD_mean", mdd_mean)
#         _save_heatmap_jet(f"fold{fold}_MDD_mean", mdd_mean, vmin=vmin_mean, vmax=vmax_mean)

#         _save_mat_csv_npy(f"fold{fold}_NC_mean",  nc_mean)
#         _save_heatmap_jet(f"fold{fold}_NC_mean",  nc_mean,  vmin=vmin_mean, vmax=vmax_mean)

#         # 5) 표준편차: 히트맵 없음, 수치만 저장
#         _save_mat_csv_npy(f"fold{fold}_MDD_std",  mdd_std)
#         _save_mat_csv_npy(f"fold{fold}_NC_std",   nc_std)

#         # 6) 그룹 평균 차이: delta = MDD_mean - NC_mean
#         if (mdd_mean is not None) and (nc_mean is not None):
#             delta = mdd_mean - nc_mean
#             _save_mat_csv_npy(f"fold{fold}_MDD_minus_NC", delta)
#             vmax_abs = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).all() else 0.0
#             if vmax_abs <= 0.0:
#                 vmax_abs = 1e-6
#             _save_heatmap_jet(f"fold{fold}_MDD_minus_NC", delta, vmin=-vmax_abs, vmax=+vmax_abs)

#         print(f"[Fold {fold}] Saved: overall mean (scale ref), group means (matched scale), and mean-diff heatmap (jet) into {out_dir}")


#     def reset(self, fold):
#         syn_dataset = self.synthetic_data[fold]
#         features = torch.stack([vec for vec, _ in syn_dataset]).to(self.device)  # [N, F]
#         return features.unsqueeze(0)  # [1, N, F]

#     def step(self, action_mask, fold):
#         sel_idx = action_mask.nonzero(as_tuple=False).squeeze(1).tolist()
#         total_samples = len(action_mask)
#         selection_ratio = len(sel_idx) / total_samples if total_samples > 0 else 0.0

#         self.selected_indices[fold] = sel_idx
#         if sel_idx:
#             self.selected_data[fold] = [
#                 (self.synthetic_data[fold][i][0].detach().cpu(), int(self.synthetic_data[fold][i][1].item()))
#                 for i in sel_idx
#             ]
#         else:
#             self.selected_data[fold] = []

#         # ====== Quality-weighted Immediate reward ======
#         imm = 0.0
#         fidelity_mean = 0.0
#         alignment_mean = 0.0
        
#         if sel_idx and len(sel_idx) > 0:  
#             try:
#                 fidelity_vals = self.fidelity_reward[fold][sel_idx] 
#                 alignment_vals = self.alignment_reward[fold][sel_idx]
                
#                 # NaN 안전 평균 계산
#                 if len(fidelity_vals) > 0:
#                     fidelity_mean = float(fidelity_vals.mean())
#                     if not math.isfinite(fidelity_mean):
#                         fidelity_mean = 0.0
                        
#                 if len(alignment_vals) > 0:
#                     alignment_mean = float(alignment_vals.mean())
#                     if not math.isfinite(alignment_mean):
#                         alignment_mean = 0.0
                
#                 # Quality-weighted reward with selection ratio penalty
#                 base_quality = self.alpha_fidelity * fidelity_mean + self.alpha_alignment * alignment_mean
                
#                 # Calculate selection ratio first
#                 N_total = len(self.synthetic_data[fold])
#                 k = len(sel_idx)
#                 sel_ratio = k / max(1, N_total)
                
#                 # Selection ratio penalty/bonus for balanced selection
#                 optimal_ratio = 0.5  # 50% 선택이 이상적
#                 ratio_penalty = abs(sel_ratio - optimal_ratio) * 50.0  # 5.0 → 15.0 (극강화)
                
#                 # Diversity bonus (선택된 샘플들의 다양성)
#                 diversity_bonus = 0.0
#                 if len(sel_idx) > 1:
#                     # 선택된 샘플들 간의 표준편차로 다양성 측정
#                     combined_scores = fidelity_vals + alignment_vals
#                     diversity_bonus = float(combined_scores.std()) * 0.5
                
#                 imm = base_quality - ratio_penalty + diversity_bonus
#                 print(f"[IMM] base={base_quality:.4f}, ratio_penalty={ratio_penalty:.4f}, diversity={diversity_bonus:.4f}, final={imm:.4f}")
#             except (IndexError, RuntimeError) as e:
#                 print(f"[WARNING] Error in immediate reward calculation: {e}")
#                 imm = 0.0
#                 fidelity_mean = 0.0
#                 alignment_mean = 0.0
#         else:
#             # 아무것도 선택하지 않을 때 큰 페널티
#             imm = -5.0  # -2.0 → -5.0 (선택 안함에 대한 극강 페널티)
#             print(f"[IMM] No selection penalty, imm={imm:.4f}")

#         # reward shaping
#         shaped, c_logs = self.shaping.shape(fold, imm)
        
#         # Store reward components for visualization
#         self.last_reward_components = {
#             'immediate': imm,
#             'shaped': shaped,
#             'base_quality': base_quality if 'base_quality' in locals() else 0.0,
#             'ratio_penalty': ratio_penalty if 'ratio_penalty' in locals() else 0.0,
#             'diversity_bonus': diversity_bonus if 'diversity_bonus' in locals() else 0.0,
#             'fidelity_mean': fidelity_mean if 'fidelity_mean' in locals() else 0.0,
#             'alignment_mean': alignment_mean if 'alignment_mean' in locals() else 0.0
#         }

#         # selected synthetic set 구성 
#         X_sel = None
#         if len(sel_idx) > 1:
#             X_sel = torch.stack([self.synthetic_data[fold][i][0] for i in sel_idx], 0).to(self.device)

#         N_total = len(self.synthetic_data[fold])
#         k = len(sel_idx)
#         sel_ratio = k / max(1, N_total)

#         # Balanced Likeness score
#         ls = 0.0
#         if sel_idx:
#             real_feats = torch.stack([v for v, _ in self.real_data[fold]], dim=0).to(self.device)
#             syn_feats  = torch.stack([self.synthetic_data[fold][i][0] for i in sel_idx], dim=0).to(self.device)
#             if real_feats.numel() > 0 and syn_feats.numel() > 0:
#                 raw_ls = float(gpu_LS(real_feats, syn_feats))
                
#                 # Selection size penalty - 너무 적거나 많이 선택하면 페널티
#                 size_penalty = 0.0
#                 if len(sel_idx) < 0.2 * N_total:  # 20% 미만 선택
#                     size_penalty = (0.2 - sel_ratio) * 5.0  # 1.5 → 5.0 (대폭 강화)
#                 elif len(sel_idx) > 0.8 * N_total:  # 80% 초과 선택
#                     size_penalty = (sel_ratio - 0.8) * 25.0  # 10.0 → 25.0 (극강화)
                
#                 ls = max(0.0, raw_ls - size_penalty)
#         else:
#             ls = -3.0  # -1.0 → -3.0 (선택 안함에 대한 극강 페널티)


#         # Validation reward
#         self.fold_steps[fold] += 1
#         step_i = self.fold_steps[fold]

#         val_reward = self.last_val_reward[fold]
#         if step_i % self.clf_every == 0:
#             metrics = self.calculate_validation_and_test(fold)
#             if self.val_metric == "val_loss":
#                 val_loss = float(metrics["val"]["loss"])
#                 val_reward = -val_loss
#             elif self.val_metric == "val_acc":
#                 val_acc = float(metrics["val"]["accuracy"])
#                 val_reward = val_acc
#             else:
#                 raise ValueError(f"Unknown val_metric: {self.val_metric}")

#             self.last_val_reward[fold] = val_reward

#             if wandb.run is not None:
#                 wandb.log({
#                     f"Fold{fold}/clf/step": step_i,
#                     f"Fold{fold}/clf/clf_reward": val_reward,
#                     f"Fold{fold}/LS/likeness_score": ls,
#                     f"Fold{fold}/clf/train_loss": metrics["train"].get("loss", float('nan')),
#                     f"Fold{fold}/clf/train_acc": metrics["train"].get("acc", float('nan')),
#                     f"Fold{fold}/clf/train_sen": metrics["train"].get("sens", float('nan')),
#                     f"Fold{fold}/clf/train_spec": metrics["train"].get("spec", float('nan')),
#                     f"Fold{fold}/clf/train_f1": metrics["train"].get("f1", float('nan')),
#                     f"Fold{fold}/clf/val_loss": metrics["val"].get("loss", float('nan')),
#                     f"Fold{fold}/clf/val_acc": metrics["val"].get("acc", float('nan')),
#                     f"Fold{fold}/clf/val_sen": metrics["val"].get("sens", float('nan')),
#                     f"Fold{fold}/clf/val_spec": metrics["val"].get("spec", float('nan')),
#                     f"Fold{fold}/clf/val_f1": metrics["val"].get("f1", float('nan')),
#                     f"Fold{fold}/clf/test_loss": metrics["test"].get("loss", float('nan')),
#                     f"Fold{fold}/clf/test_acc": metrics["test"].get("acc", float('nan')),
#                     f"Fold{fold}/clf/test_sen": metrics["test"].get("sens", float('nan')),
#                     f"Fold{fold}/clf/test_spec": metrics["test"].get("spec", float('nan')),
#                     f"Fold{fold}/clf/test_f1": metrics["test"].get("f1", float('nan')),
#                 })

#         # 최종 보상 (NaN 안전 처리)
#         def safe_float(value, default=0.0):
#             """NaN/Inf 안전 변환"""
#             if value is None:
#                 return default
#             try:
#                 val = float(value)
#                 return default if (math.isnan(val) or math.isinf(val)) else val
#             except (ValueError, TypeError):
#                 return default
        
#         # 각 구성요소 NaN 체크 (selection_penalty 제거)
#         shaped_safe = safe_float(shaped, 0.0)
#         ls_safe = safe_float(ls, 0.0)
#         val_reward_safe = safe_float(val_reward, 0.0)
        
#         # Reward 정규화 (스케일 문제 해결)
#         # shaped: -10~+10 -> -1~+1
#         shaped_normalized = shaped_safe / 10.0
#         # ls: 0~1 -> 0~1 (이미 정규화됨)
#         ls_normalized = ls_safe
#         # val_reward: -5~+1 -> -1~+0.2
#         val_normalized = val_reward_safe / 5.0
        
#         reward = shaped_normalized + self.alpha_diversity * ls_normalized + self.alpha_utility * val_normalized
        
#         # Update reward components with final values
#         self.last_reward_components.update({
#             'likeness_score': ls_safe,
#             'validation_reward': val_reward_safe,
#             'shaped_normalized': shaped_normalized,
#             'ls_normalized': ls_normalized,
#             'val_normalized': val_normalized,
#             'final_reward': reward,
#             'alpha_diversity': self.alpha_diversity,
#             'alpha_utility': self.alpha_utility
#         })
        
#         # 최종 reward NaN 체크
#         reward = safe_float(reward, 0.0)
        
#         # 정규화된 reward 범위 제한 (-3 ~ +3)
#         reward = max(-3.0, min(3.0, reward))
        
#         # NaN 발생 시 디버깅 로그
#         if shaped != shaped_safe or ls != ls_safe or val_reward != val_reward_safe:
#             print(f"[WARNING] NaN detected in reward components for Fold {fold}:")
#             print(f"  shaped: {shaped} -> {shaped_safe}")
#             print(f"  ls: {ls} -> {ls_safe}")
#             print(f"  val_reward: {val_reward} -> {val_reward_safe}")
#             print(f"  final_reward: {reward}")

#         log = {
#             f"Fold{fold}/selection/k":          int(k),
#             f"Fold{fold}/selection/ratio":      float(sel_ratio),
#             f"Fold{fold}/reward/imm_raw":       float(imm),
#             f"Fold{fold}/reward/fidelity_mean":     float(fidelity_mean),
#             f"Fold{fold}/reward/alignment_mean":     float(alignment_mean),
#             f"Fold{fold}/reward/total":         float(reward),
#         }
        
#         if wandb.run is not None:
#             wandb.log(log)

#         return None, reward, True, {}


#     def calculate_validation_and_test(self, fold: int):
#         real_ds = self.real_data[fold]
#         if self.selected_data.get(fold):
#             feats, lbls = zip(*self.selected_data[fold])
#             syn_ds = TensorDataset(torch.stack(feats), torch.tensor(lbls))
#             train_ds = ConcatDataset([real_ds, syn_ds])
#         else:
#             train_ds = real_ds

#         g = make_dl_generator(getattr(self.config, "seed", 100) + fold)

#         train_loader = DataLoader(train_ds, batch_size=len(train_ds),
#                                   shuffle=True, num_workers=0, pin_memory=True,
#                                   worker_init_fn=seed_worker, generator=g)
#         val_loader = DataLoader(self.val_data[fold], batch_size=len(self.val_data[fold]),
#                                 shuffle=False, num_workers=0, pin_memory=True,
#                                 worker_init_fn=seed_worker, generator=g)
#         test_loader = DataLoader(self.test_data[fold], batch_size=len(self.test_data[fold]),
#                                  shuffle=False, num_workers=0, pin_memory=True,
#                                  worker_init_fn=seed_worker, generator=g)

#         clf_results = classifier(train_loader, val_loader, test_loader, config=self.config)
#         nested_metrics = {'train': {}, 'val': {}, 'test': {}}
#         for key, value in clf_results.items():
#             try:
#                 dataset_name, metric_name = key.split('_', 1)
#                 if dataset_name in nested_metrics:
#                     nested_metrics[dataset_name][metric_name] = value
#             except ValueError:
#                 pass
#         return nested_metrics


import os
import random
import numpy as np
import math

SEED = 100

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  
import wandb
from collections import deque

def seed_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()   
    g.manual_seed(seed)
    return g

def make_dl_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
from load_brain_data import BrainDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import wandb
from compute_LS import gpu_LS
import logging

from mlp_classifier import classifier
from load_gan_data import GANSyntheticDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .reward_shaping import Shaping

class Environment:
    def __init__(self, config, real_data, synthetic_data, val_data, test_data):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.val_data = val_data
        self.test_data = test_data

        # imm reward 가중치
        self.alpha_fidelity = config.alpha_fidelity
        self.alpha_alignment = config.alpha_alignment
        self.alpha_diversity = config.alpha_diversity

        self.fidelity_reward, self.alignment_reward = self._load_precomputed(config.precomputed_path)

        # 선택 기록
        self.selected_data    = {f: [] for f in synthetic_data.keys()}
        self.selected_indices = {f: [] for f in synthetic_data.keys()}

        # shaping
        self.shaping = Shaping(
            alpha_fidelity=self.alpha_fidelity,
            alpha_alignment=self.alpha_alignment,
            fidelity_reward=self.fidelity_reward,
            alignment_reward=self.alignment_reward,
            a=config.a,
            frac=config.frac,
            q_lo=config.q_lo,
            q_hi=config.q_hi,
            use_median=config.use_median,
            clip_k=config.clip_k,
            eps=config.eps,
        )

        # validation reward
        self.clf_every       = config.clf_every
        self.val_metric      = config.val_metric
        self.alpha_utility      = config.alpha_utility
        self.fold_steps      = {f: 0 for f in synthetic_data.keys()}
        self.last_val_reward = {f: 0.0 for f in synthetic_data.keys()}

    def _load_precomputed(self, path):
        npz = np.load(path, allow_pickle=True)
        if "fidelity_good" in npz and "alignment_good" in npz:
            fidelity = npz["fidelity_good"].item()
            alignment = npz["alignment_good"].item()
        else:
            raise KeyError(f"Precomputed file missing expected keys. Found: {list(npz.keys())}.")
        fidelity = {int(f): np.asarray(v, dtype=np.float32) for f, v in fidelity.items()}
        alignment = {int(f): np.asarray(v, dtype=np.float32) for f, v in alignment.items()}
        return fidelity, alignment

    def _flat_to_mat(self, flat_vec: torch.Tensor) -> torch.Tensor:
        """
        flat_vec: [F] (F = n*(n-1)/2). 반환: [n,n] 대칭행렬
        """
        F = int(flat_vec.numel())
        n = int((1 + math.sqrt(1 + 8*F)) // 2)
        device = flat_vec.device
        idx = torch.triu_indices(n, n, offset=1, device=device)
        mat = torch.zeros(n, n, device=device, dtype=flat_vec.dtype)
        mat[idx[0], idx[1]] = flat_vec
        return mat + mat.T

    @torch.no_grad()
    def save_selection_artifacts(self, fold: int, out_dir: str):
        import seaborn as sns

        os.makedirs(out_dir, exist_ok=True)
        sel_idx = self.selected_indices.get(fold, [])
        if not sel_idx:
            print(f"[Fold {fold}] No selected samples. Skipping save.")
            return

        # 선택 벡터/라벨 수집
        vecs, labels = [], []
        for i in sel_idx:
            v, lbl = self.synthetic_data[fold][i]
            vecs.append(v.detach().cpu().float())
            labels.append(int(lbl))
        X = torch.stack(vecs, dim=0)           # [k, F]
        y = torch.tensor(labels, dtype=torch.long)

        # 그룹 분리
        is_mdd = (y == 1)
        is_nc  = (y == 0)

        def vecs_to_mean_std_mats(Xg: torch.Tensor):
            if Xg.numel() == 0:
                return None, None
            mean_vec = Xg.mean(dim=0)
            std_vec  = Xg.std(dim=0, unbiased=False)
            mean_mat = self._flat_to_mat(mean_vec).cpu().numpy()
            std_mat  = self._flat_to_mat(std_vec).cpu().numpy()
            return mean_mat, std_mat

        mdd_mean, mdd_std = vecs_to_mean_std_mats(X[is_mdd]) if is_mdd.any() else (None, None)
        nc_mean,  nc_std  = vecs_to_mean_std_mats(X[is_nc])  if is_nc.any()  else (None, None)

        # 1) 인덱스+라벨 저장(.txt)
        txt_path = os.path.join(out_dir, f"selected_idx_fold{fold}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Total selected: {len(sel_idx)}\nIndex\tLabel\n")
            for idx, lbl in zip(sel_idx, labels):
                f.write(f"{idx}\t{lbl}\n")
        print(f"[Fold {fold}] Saved indices -> {txt_path}")

        # 2) 통계 행렬 저장(.npz)
        npz_path = os.path.join(out_dir, f"selection_stats_fold{fold}.npz")
        np.savez(
            npz_path,
            selected_idx=np.asarray(sel_idx, dtype=np.int64),
            mdd_mean=mdd_mean, mdd_std=mdd_std,
            nc_mean=nc_mean,   nc_std=nc_std,
        )
        print(f"[Fold {fold}] Saved stats -> {npz_path}")

        # ---- 공통 저장 유틸 ----
        def _save_mat_csv_npy(name: str, mat: np.ndarray):
            if mat is None:
                return
            np.save(os.path.join(out_dir, f"{name}.npy"), mat)
            np.savetxt(os.path.join(out_dir, f"{name}.csv"), mat, delimiter=",")

        def _save_heatmap_jet(name: str, mat: np.ndarray, vmin=None, vmax=None):
            if mat is None:
                return
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                mat, ax=ax, annot=False, fmt=".2f",
                linewidths=0, cmap="jet", cbar=True,
                xticklabels=False, yticklabels=False,
                vmin=vmin, vmax=vmax
            )
            ax.set_title(name)
            fig.tight_layout()
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(png_path, dpi=200)
            plt.close(fig)
            print(f"[Fold {fold}] Saved heatmap -> {png_path}")

        # 3) 선택 전체 평균 FC (스케일 기준)
        mats = [self._flat_to_mat(x).cpu().numpy() for x in X]
        mean_selected = np.mean(np.stack(mats, axis=0), axis=0)

        _save_mat_csv_npy(f"mean_selected_fold{fold}", mean_selected)
        vmin_mean, vmax_mean = -0.1, 1.5
        _save_heatmap_jet(f"mean_generated_fold{fold}", mean_selected, vmin=vmin_mean, vmax=vmax_mean)

        # 4) 그룹 평균
        _save_mat_csv_npy(f"fold{fold}_MDD_mean", mdd_mean)
        _save_heatmap_jet(f"fold{fold}_MDD_mean", mdd_mean, vmin=vmin_mean, vmax=vmax_mean)

        _save_mat_csv_npy(f"fold{fold}_NC_mean",  nc_mean)
        _save_heatmap_jet(f"fold{fold}_NC_mean",  nc_mean,  vmin=vmin_mean, vmax=vmax_mean)

        # 5) 표준편차
        _save_mat_csv_npy(f"fold{fold}_MDD_std",  mdd_std)
        _save_mat_csv_npy(f"fold{fold}_NC_std",   nc_std)

        # 6) 그룹 평균 차이
        if (mdd_mean is not None) and (nc_mean is not None):
            delta = mdd_mean - nc_mean
            _save_mat_csv_npy(f"fold{fold}_MDD_minus_NC", delta)
            vmax_abs = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).all() else 0.0
            if vmax_abs <= 0.0:
                vmax_abs = 1e-6
            _save_heatmap_jet(f"fold{fold}_MDD_minus_NC", delta, vmin=-vmax_abs, vmax=+vmax_abs)

        print(f"[Fold {fold}] Saved all artifacts into {out_dir}")

    def reset(self, fold):
        syn_dataset = self.synthetic_data[fold]
        features = torch.stack([vec for vec, _ in syn_dataset]).to(self.device)  # [N, F]
        return features.unsqueeze(0)  # [1, N, F]

    def step(self, action_mask, fold):
        sel_idx = action_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        N_total = len(action_mask)
        selection_ratio = len(sel_idx) / N_total if N_total > 0 else 0.0

        self.selected_indices[fold] = sel_idx
        if sel_idx:
            self.selected_data[fold] = [
                (self.synthetic_data[fold][i][0].detach().cpu(), 
                 int(self.synthetic_data[fold][i][1].item()))
                for i in sel_idx
            ]
        else:
            self.selected_data[fold] = []

        # ============= Per-sample rewards 초기화 =============
        per_sample_rewards = torch.zeros(N_total, dtype=torch.float32, device=self.device)
        
        # ===== 1) Per-sample quality scores (개별 보상) =====
        imm = 0.0
        fidelity_mean = 0.0
        alignment_mean = 0.0
        base_quality = 0.0
        ratio_penalty = 0.0
        
        if sel_idx and len(sel_idx) > 0:  
            try:
                # NumPy → Torch tensor 변환
                fidelity_vals = torch.tensor(
                    self.fidelity_reward[fold][sel_idx], 
                    dtype=torch.float32,
                    device=self.device
                )
                alignment_vals = torch.tensor(
                    self.alignment_reward[fold][sel_idx],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # 개별 샘플 quality scores
                quality_scores = (
                    self.alpha_fidelity * fidelity_vals + 
                    self.alpha_alignment * alignment_vals
                )  # [k]
                
                # 통계 (로깅용)
                fidelity_mean = float(fidelity_vals.mean().item())
                if not math.isfinite(fidelity_mean):
                    fidelity_mean = 0.0
                    
                alignment_mean = float(alignment_vals.mean().item())
                if not math.isfinite(alignment_mean):
                    alignment_mean = 0.0
                
                base_quality = float(quality_scores.mean().item())
                
                # Selection ratio penalty
                k = len(sel_idx)
                sel_ratio = k / max(1, N_total)
                optimal_ratio = 0.5
                ratio_penalty = abs(sel_ratio - optimal_ratio) * 50.0
                
                # Immediate reward (전역 평균)
                imm = base_quality - ratio_penalty
                
                # ===== Per-sample quality 정규화 및 할당 =====
                if quality_scores.numel() > 1:
                    q_mean = quality_scores.mean()
                    q_std = quality_scores.std(unbiased=False) + 1e-8
                    quality_normalized = (quality_scores - q_mean) / q_std
                else:
                    quality_normalized = quality_scores
                
                # 선택된 샘플에 개별 quality 할당
                per_sample_rewards[sel_idx] = quality_normalized
                
                print(f"[PER-SAMPLE] k={k}, quality: [{quality_normalized.min():.3f}, {quality_normalized.max():.3f}], mean={quality_normalized.mean():.3f}")
                
            except (IndexError, RuntimeError) as e:
                print(f"[WARNING] Error in quality calculation: {e}")
                imm = 0.0
        else:
            imm = -5.0
            print(f"[IMM] No selection penalty, imm={imm:.4f}")

        # ===== 2) Reward shaping (전역) =====
        shaped, c_logs = self.shaping.shape(fold, imm)
        shaped_safe = shaped if math.isfinite(shaped) else 0.0
        shaped_normalized = shaped_safe / 10.0
        
        # Store reward components
        self.last_reward_components = {
            'immediate': imm,
            'shaped': shaped,
            'base_quality': base_quality,
            'ratio_penalty': ratio_penalty,
            'fidelity_mean': fidelity_mean,
            'alignment_mean': alignment_mean,
        }

        # ===== 3) Likeness Score (집합 보상) =====
        ls = 0.0
        if sel_idx:
            real_feats = torch.stack([v for v, _ in self.real_data[fold]], dim=0).to(self.device)
            syn_feats = torch.stack([self.synthetic_data[fold][i][0] for i in sel_idx], dim=0).to(self.device)
            if real_feats.numel() > 0 and syn_feats.numel() > 0:
                raw_ls = float(gpu_LS(real_feats, syn_feats))
                
                # Selection size penalty
                size_penalty = 0.0
                if len(sel_idx) < 0.2 * N_total:
                    size_penalty = (0.2 - selection_ratio) * 5.0
                elif len(sel_idx) > 0.8 * N_total:
                    size_penalty = (selection_ratio - 0.8) * 25.0
                
                ls = max(0.0, raw_ls - size_penalty)
        else:
            ls = -3.0
        
        ls_safe = ls if math.isfinite(ls) else 0.0
        ls_normalized = ls_safe

        # ===== 4) Validation reward (집합 보상) =====
        self.fold_steps[fold] += 1
        step_i = self.fold_steps[fold]
        val_reward = self.last_val_reward[fold]
        
        if step_i % self.clf_every == 0:
            metrics = self.calculate_validation_and_test(fold)
            if self.val_metric == "val_loss":
                val_reward = -float(metrics["val"]["loss"])
            elif self.val_metric == "val_acc":
                val_reward = float(metrics["val"]["accuracy"])
            else:
                raise ValueError(f"Unknown val_metric: {self.val_metric}")
            
            self.last_val_reward[fold] = val_reward
            
            if wandb.run is not None:
                wandb.log({
                    f"Fold{fold}/clf/step": step_i,
                    f"Fold{fold}/clf/clf_reward": val_reward,
                    f"Fold{fold}/LS/likeness_score": ls,
                    f"Fold{fold}/clf/train_loss": metrics["train"].get("loss", float('nan')),
                    f"Fold{fold}/clf/train_acc": metrics["train"].get("acc", float('nan')),
                    f"Fold{fold}/clf/train_sen": metrics["train"].get("sens", float('nan')),
                    f"Fold{fold}/clf/train_spec": metrics["train"].get("spec", float('nan')),
                    f"Fold{fold}/clf/train_f1": metrics["train"].get("f1", float('nan')),
                    f"Fold{fold}/clf/val_loss": metrics["val"].get("loss", float('nan')),
                    f"Fold{fold}/clf/val_acc": metrics["val"].get("acc", float('nan')),
                    f"Fold{fold}/clf/val_sen": metrics["val"].get("sens", float('nan')),
                    f"Fold{fold}/clf/val_spec": metrics["val"].get("spec", float('nan')),
                    f"Fold{fold}/clf/val_f1": metrics["val"].get("f1", float('nan')),
                    f"Fold{fold}/clf/test_loss": metrics["test"].get("loss", float('nan')),
                    f"Fold{fold}/clf/test_acc": metrics["test"].get("acc", float('nan')),
                    f"Fold{fold}/clf/test_sen": metrics["test"].get("sens", float('nan')),
                    f"Fold{fold}/clf/test_spec": metrics["test"].get("spec", float('nan')),
                    f"Fold{fold}/clf/test_f1": metrics["test"].get("f1", float('nan')),
                })
        
        val_reward_safe = val_reward if math.isfinite(val_reward) else 0.0
        val_normalized = val_reward_safe / 5.0

        # ===== 5) 집합 보상 (set-level bonus) =====
        # 선택된 모든 샘플이 협력적으로 받는 보상
        set_level_bonus = (
            shaped_normalized + 
            self.alpha_diversity * ls_normalized + 
            self.alpha_utility * val_normalized
        )
        
        # ===== 6) 최종 per-sample rewards =====
        # 개별 quality + 집합 보너스
        if sel_idx:
            per_sample_rewards[sel_idx] += set_level_bonus
        
        # 미선택 샘플: 기회비용 패널티
        unselected_mask = torch.ones(N_total, dtype=torch.bool, device=self.device)
        if sel_idx:
            unselected_mask[sel_idx] = False
        per_sample_rewards[unselected_mask] = -0.1
        
        # ===== 7) 전역 평균 (backward compatibility) =====
        global_reward = float(per_sample_rewards.mean().item())
        global_reward = max(-3.0, min(3.0, global_reward))
        
        # ===== 로깅 =====
        self.last_reward_components.update({
            'likeness_score': ls_safe,
            'validation_reward': val_reward_safe,
            'shaped_normalized': shaped_normalized,
            'ls_normalized': ls_normalized,
            'val_normalized': val_normalized,
            'set_level_bonus': set_level_bonus,
            'final_reward': global_reward,
            'alpha_diversity': self.alpha_diversity,
            'alpha_utility': self.alpha_utility,
            'per_sample_mean': float(per_sample_rewards[sel_idx].mean().item()) if sel_idx else 0.0,
            'per_sample_std': float(per_sample_rewards[sel_idx].std(unbiased=False).item()) if sel_idx else 0.0,
        })
        
        log = {
            f"Fold{fold}/selection/k": int(len(sel_idx)),
            f"Fold{fold}/selection/ratio": float(selection_ratio),
            f"Fold{fold}/reward/imm_raw": float(imm),
            f"Fold{fold}/reward/fidelity_mean": float(fidelity_mean),
            f"Fold{fold}/reward/alignment_mean": float(alignment_mean),
            f"Fold{fold}/reward/set_level_bonus": float(set_level_bonus),
            f"Fold{fold}/reward/total": float(global_reward),
            f"Fold{fold}/reward/per_sample_mean": self.last_reward_components['per_sample_mean'],
            f"Fold{fold}/reward/per_sample_std": self.last_reward_components['per_sample_std'],
        }
        
        if wandb.run is not None:
            wandb.log(log)

        return None, global_reward, True, {
            'per_sample_rewards': per_sample_rewards.detach().cpu()  # [N_total]
        }

    def calculate_validation_and_test(self, fold: int):
        real_ds = self.real_data[fold]
        if self.selected_data.get(fold):
            feats, lbls = zip(*self.selected_data[fold])
            syn_ds = TensorDataset(torch.stack(feats), torch.tensor(lbls))
            train_ds = ConcatDataset([real_ds, syn_ds])
        else:
            train_ds = real_ds

        g = make_dl_generator(getattr(self.config, "seed", 100) + fold)

        train_loader = DataLoader(train_ds, batch_size=len(train_ds),
                                  shuffle=True, num_workers=0, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(self.val_data[fold], batch_size=len(self.val_data[fold]),
                                shuffle=False, num_workers=0, pin_memory=True,
                                worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(self.test_data[fold], batch_size=len(self.test_data[fold]),
                                 shuffle=False, num_workers=0, pin_memory=True,
                                 worker_init_fn=seed_worker, generator=g)

        clf_results = classifier(train_loader, val_loader, test_loader, config=self.config)
        nested_metrics = {'train': {}, 'val': {}, 'test': {}}
        for key, value in clf_results.items():
            try:
                dataset_name, metric_name = key.split('_', 1)
                if dataset_name in nested_metrics:
                    nested_metrics[dataset_name][metric_name] = value
            except ValueError:
                pass
        return nested_metrics