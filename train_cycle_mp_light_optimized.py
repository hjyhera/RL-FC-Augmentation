import os
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Value
import pickle
import traceback
import gc

from seeds import setup_determinism

from copy import deepcopy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
import wandb
from datetime import datetime

from load_brain_data import BrainDataset
from compute_LS import gpu_LS

import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent / "GC-GAN-main" / "GC-GAN-main"
assert repo_root.is_dir(), f"경로 없음: {repo_root}"

sys.path.insert(0, str(repo_root))

from model import ACDiscriminator

from mlp_classifier import classifier
from Environment.env import Environment
from ppo_agent import PPO
from ppo_config import PPOConfig, get_ppo_config
from load_gan_data import GANSyntheticDataset
import logging
import math
from dual_network_light import Actor, Critic  # 경량화 모델 사용
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def round_metrics(m, ndigits=4):
    out = {}
    for k, v in m.items():
        try:
            out[k] = round(float(v), ndigits)
        except Exception:
            out[k] = v
    return out

def save_best_actor(actor, meta, path, selection_count: int):
    payload = {
        **meta,
        "model_state": actor.state_dict(),
        "selection_count": selection_count,
    }
    torch.save(payload, path)

def get_device_for_process(process_id, total_processes, force_cpu=False):
    """
    메모리 최적화된 디바이스 할당
    """
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # 멀티 GPU: 각 프로세스를 다른 GPU에 할당
        gpu_id = process_id % num_gpus
        return torch.device(f"cuda:{gpu_id}")
    else:
        # 단일 GPU: 메모리 부족 시 CPU로 fallback
        try:
            device = torch.device("cuda:0")
            # 메모리 테스트
            test_tensor = torch.randn(100, 100, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Fold {process_id}] GPU memory insufficient, falling back to CPU")
                return torch.device("cpu")
            else:
                raise e

def build_light_actor_and_critic(device, input_dim=6216):
    """
    경량화된 모델 생성
    """
    # 경량화된 하이퍼파라미터
    meta = {
        "input_dim": input_dim,
        "d_model": 128,      # 256 → 128
        "nhead": 4,          # 8 → 4  
        "num_layers": 2,     # 4 → 2
        "dim_feedforward": 256,  # 512 → 256
    }
    
    actor = Actor(
        input_dim=meta["input_dim"],
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dim_feedforward=meta["dim_feedforward"],
    ).to(device)
    
    critic = Critic(h_dim=meta["d_model"], hidden=64).to(device)  # 128 → 64
    
    return actor, critic, meta

def build_actor_and_critic_from_ckpt(ckpt_path, device, critic_hidden=64, default_input_dim=6216):
    """
    체크포인트에서 모델 로드 (경량화 버전으로 변환)
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state", ckpt)

        if "d_model" in ckpt:
            # 기존 모델을 경량화 모델로 변환
            print(f"Converting pretrained model to lightweight version...")
            
            # 경량화된 모델 생성
            actor, critic, meta = build_light_actor_and_critic(device, default_input_dim)
            
            # 가능한 가중치만 로드 (크기가 맞는 것만)
            actor_state = actor.state_dict()
            pretrained_state = state
            
            loaded_keys = []
            for key in actor_state.keys():
                if key in pretrained_state:
                    if actor_state[key].shape == pretrained_state[key].shape:
                        actor_state[key] = pretrained_state[key]
                        loaded_keys.append(key)
            
            actor.load_state_dict(actor_state)
            print(f"Loaded {len(loaded_keys)}/{len(actor_state)} layers from pretrained model")
            
            return actor, critic, meta
        else:
            # 체크포인트가 없으면 새로 생성
            return build_light_actor_and_critic(device, default_input_dim)
            
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        print("Creating new lightweight model...")
        return build_light_actor_and_critic(device, default_input_dim)


@torch.no_grad()
def evaluate_with_best(cfg, fold, device, env, ckpt_path):
    def _is_num(x):
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    def _fmt(x, nd=4):
        return f"{float(x):.{nd}f}" if _is_num(x) else "NA"

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
        # 경량화 모델로 로드
        actor, _, meta = build_actor_and_critic_from_ckpt(ckpt_path, device)
        actor.eval()

        print(f"[Fold {fold}] [Evaluation] Using lightweight model on {device}")

        # ===== 상태/로짓/확률 =====
        state = env.reset(fold)
        s = state.to(device)
        logits, _ = actor.forward_with_hidden(s)    # [1, N, 2]
        probs = torch.softmax(logits[0], dim=-1)    # [N, 2]
        probs1 = probs[:, 1].detach().cpu().float() # [N]

        # argmax 선택 
        a_star = probs.argmax(dim=-1)               # [N] {0,1}
        action = a_star.to(torch.long).cpu()
        
        # NaN 안전 selection ratio 계산
        if len(action) > 0:
            sel_ratio = float(action.float().mean().item())
            if not math.isfinite(sel_ratio):
                sel_ratio = 0.0
        else:
            sel_ratio = 0.0

        # NaN 안전 통계 계산
        if len(probs1) > 0:
            p_mean = float(probs1.mean().item())
            p_std  = float(probs1.std(unbiased=False).item())
            p_min  = float(probs1.min().item())
            p_max  = float(probs1.max().item())
            
            # NaN 체크 및 기본값 설정
            if not math.isfinite(p_mean): p_mean = 0.0
            if not math.isfinite(p_std): p_std = 0.0
            if not math.isfinite(p_min): p_min = 0.0
            if not math.isfinite(p_max): p_max = 1.0
        else:
            p_mean, p_std, p_min, p_max = 0.0, 0.0, 0.0, 1.0

        # 통계 
        p_mean = float(probs1.mean().item())
        p_std  = float(probs1.std(unbiased=False).item())
        p_min  = float(probs1.min().item())
        p_max  = float(probs1.max().item())

        # 히스토그램 저장
        log_dir  = Path(getattr(cfg, "eval_log_dir", "eval_logs")); log_dir.mkdir(parents=True, exist_ok=True)
        hist_png = log_dir / f"fold{fold}_eval_prob_hist.png"
        plt.figure(figsize=(6,4))
        plt.hist(probs1.numpy(), bins=20, range=(0,1), alpha=0.85)
        plt.title(f"Fold {fold} eval probs[:,1]\nmean={p_mean:.4f}, std={p_std:.4f}")
        plt.xlabel("p(select)"); plt.ylabel("count"); plt.tight_layout()
        plt.savefig(hist_png, dpi=160); plt.close()

        # validation & test 
        env.step(action, fold)
        sel_idx = env.selected_indices[fold]
        sel_labels = [int(env.synthetic_data[fold][i][1].item()) for i in sel_idx]

        nested = env.calculate_validation_and_test(fold)
        val_metrics  = nested.get("val", {})
        test_metrics = nested.get("test", {})

        # .log 파일 저장
        log_path = log_dir / f"eval_fold{fold}.log"
        lines = []
        lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] fold={fold}")
        lines.append(f"  ckpt: {ckpt_path}")
        lines.append(f"  model: lightweight (d_model={meta.get('d_model', 128)})")
        lines.append(f"  selection: k={len(sel_idx)}, ratio={_fmt(sel_ratio)}")
        lines.append(f"  probs1: mean={_fmt(p_mean)}, std={_fmt(p_std)}, min={_fmt(p_min)}, max={_fmt(p_max)}")
        if val_metrics:
            lines.append("  val:")
            for kname in sorted(val_metrics.keys()):
                lines.append(f"    {kname}: {_fmt(val_metrics[kname])}")
        if test_metrics:
            lines.append("  test:")
            for kname in sorted(test_metrics.keys()):
                lines.append(f"    {kname}: {_fmt(test_metrics[kname])}")
        lines.append("")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Fold {fold}] [Eval] 로그 저장 완료 → {log_path}")

        return test_metrics, sel_idx, sel_labels
        
    except Exception as e:
        print(f"[Fold {fold}] Evaluation failed: {e}")
        return {}, [], []

def compute_policy_score(mean_reward, stats, cfg, episode=0):
    """
    개선된 policy score 계산 (NaN 안전 처리 포함)
    - Warmup 기간 동안 완화된 평가
    - 음수 보상에 대한 완화된 처리
    - 적응적 가중치 적용
    - NaN/Inf 값에 대한 안전 처리
    - Selection ratio 관련 보상 제거
    """
    import math
    
    # NaN 안전 변환 함수
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            val = float(value)
            return default if (math.isnan(val) or math.isinf(val)) else val
        except (ValueError, TypeError):
            return default
    
    # 입력값 NaN 안전 처리 (selection ratio 제거)
    mean_reward = safe_float(mean_reward, 0.0)
    kl = safe_float(stats.get("kl_mean", None), None)
    entropy = safe_float(stats.get("entropy", None), None)
    clip_frac = safe_float(stats.get("clip_frac", None), None)
    adv_std = safe_float(stats.get("adv_std_raw", None), None)
    
    # Warmup 기간 확인
    warmup_episodes = getattr(cfg, "warmup_episodes", 50)
    is_warmup = episode < warmup_episodes

    # ---- 가중치 (warmup 기간 동안 완화) ----
    w_r   = getattr(cfg, "score_w_reward",   1.0)
    w_kl  = getattr(cfg, "score_w_kl",       0.2 if is_warmup else 0.3)
    w_clip= getattr(cfg, "score_w_clip",     0.2 if is_warmup else 0.3)
    w_ent = getattr(cfg, "score_w_entropy",  0.1 if is_warmup else 0.15)
    w_adv = getattr(cfg, "score_w_advstd",   0.05 if is_warmup else 0.1)

    # ---- 임계값 (warmup 기간 동안 완화) ----
    kl_hi        = getattr(cfg, "es_kl_hi",          0.3 if is_warmup else 0.2)
    clip_frac_hi = getattr(cfg, "es_clip_frac_hi",   0.95 if is_warmup else 0.9)
    entropy_lo   = getattr(cfg, "es_entropy_lo",     0.02 if is_warmup else 0.04)
    adv_std_lo   = getattr(cfg, "es_adv_std_lo",     0.001 if is_warmup else 0.005)
    
    # ---- 기본 점수 (NaN 안전 처리) ----
    if mean_reward >= 0:
        reward_score = w_r * mean_reward
    else:
        # 음수 보상에 대한 완화 처리 (warmup 시 더 관대)
        penalty_factor = 0.3 if is_warmup else 0.5
        reward_score = w_r * mean_reward * penalty_factor
    
    score = reward_score

    # ---- Selection ratio 관련 보상 제거됨 ----

    # ---- 기존 규제 패널티들 (NaN 안전 처리) ----
    if kl is not None and kl > kl_hi:
        penalty = w_kl * (kl - kl_hi)
        score -= penalty
        
    if clip_frac is not None and clip_frac > clip_frac_hi:
        penalty = w_clip * (clip_frac - clip_frac_hi)
        score -= penalty
        
    if entropy is not None and entropy < entropy_lo:
        penalty = w_ent * (entropy_lo - entropy)
        score -= penalty
        
    if adv_std is not None and adv_std < adv_std_lo:
        penalty = w_adv * (adv_std_lo - adv_std)
        score -= penalty

    # 최종 NaN 체크 및 안전 반환
    final_score = safe_float(score, 0.0)
    
    # 디버깅을 위한 로그 (극단적인 경우만)
    if abs(final_score) > 100 or final_score != score:
        print(f"[WARNING] Policy score clamped: {score:.6f} -> {final_score:.6f}")
        print(f"  mean_reward={mean_reward:.6f}")
    
    return final_score

def train_single_fold(fold, cfg_dict, result_queue, progress_dict, force_cpu=False):
    """
    단일 fold를 훈련하는 함수 (성능 최적화 버전)
    """
    try:
        # 프로세스별 시드 설정
        setup_determinism(100 + fold)
        seed_everything(cfg_dict['seed'] + fold)
        
        # cfg 객체 재구성
        cfg = PPOConfig()
        for key, value in cfg_dict.items():
            setattr(cfg, key, value)
        

        # 메모리 최적화된 디바이스 설정
        device = get_device_for_process(fold, 5, force_cpu=force_cpu)
        print(f"[Fold {fold}] Starting optimized lightweight training on {device} (Process ID: {os.getpid()})")
        
        # GPU 메모리 최적화
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            # 메모리 사용량 모니터링
            allocated = torch.cuda.memory_allocated(device)/1024**3
            print(f"[Fold {fold}] GPU Memory: {torch.cuda.get_device_name(device)} - Allocated: {allocated:.1f}GB")
        
        # 로거 설정
        logger = logging.getLogger(f"Fold{fold}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(cfg.info_dir, f"fold{fold}_optimized_run.log"), mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.handlers = [fh]

        # 데이터 로드
        real_data = BrainDataset(cfg.train_folds[fold], cfg.data_dir)
        val_data = BrainDataset(cfg.valid_folds[fold], cfg.data_dir)  
        test_data = BrainDataset(cfg.test_folds[fold], cfg.data_dir)
        synthetic_data = GANSyntheticDataset(cfg.data_dir_gan, fold)

        # 환경 설정
        env = Environment(cfg, {fold: real_data}, {fold: synthetic_data}, {fold: val_data}, {fold: test_data})

        # 경량화 모델 로드
        ckpt_path = f"actor_pretrain_fold{fold}.pt"
        actor, critic, meta = build_actor_and_critic_from_ckpt(ckpt_path, device, critic_hidden=64)
        print(f"[Fold {fold}] Optimized lightweight model loaded: {meta}")

        ppo = PPO(actor=actor, critic=critic, config=cfg, device=device, logger=logger)  

        # 메모리 효율적인 보상 추적 (고정 크기 deque 사용)
        from collections import deque
        total_rewards = deque(maxlen=1000)  # 최대 1000개만 유지
        best_score = -np.inf
        patience_counter = 0

        # 베스트 액터 저장 경로 (새로운 파일명으로 기존 모델과 구분)
        best_actor_path = os.path.join(cfg.info_dir, f"best_fixed_actor_fold{fold}.pt")

        # Selection ratio tracking for visualization
        selection_history = []
        
        # 진행률 초기화
        progress_dict[f'fold_{fold}'] = {
            'current_episode': 0,
            'total_episodes': cfg.max_episodes,
            'best_score': -np.inf,
            'status': 'training',
            'reward_mean': 0.0,
            'patience': 0
        }

        # 훈련 루프 (tqdm 진행률 표시)
        pbar = tqdm(range(1, cfg.max_episodes + 1), 
                   desc=f"Fold {fold}", 
                   position=fold-1,
                   leave=True,
                   ncols=120)
        
        for episode in pbar:
            state = env.reset(fold)
            action, logp, old_logits, h_last = ppo.act(state)  

            with torch.no_grad():
                probs = torch.softmax(old_logits, dim=-1)      # [N,2]
                probs1 = probs[:,1].detach().cpu()
                p_mean = float(probs1.mean().item())
                sel_ratio = float(action.float().mean().item())
                
                # Save selection ratio for visualization
                selection_history.append({
                    'episode': episode,
                    'sel_ratio': sel_ratio,
                    'reward': 0.0,  # Will be updated after env.step
                    'reward_components': {}  # Will store detailed reward breakdown
                })

            _, reward, done, _ = env.step(action, fold)
            
            # Update reward and components in selection history
            if selection_history:
                selection_history[-1]['reward'] = reward
                # Get detailed reward components from environment
                if hasattr(env, 'last_reward_components'):
                    selection_history[-1]['reward_components'] = env.last_reward_components.copy()
            ppo.store_transition(state, action, reward, done, logp, old_logits, h_last)
            total_rewards.append(reward)

            if len(ppo.buffer) >= cfg.rollout_episodes:
                ppo.learn(fold=fold, step_idx=episode) 

            last_stats = {}
            if hasattr(ppo, "last_stats") and isinstance(ppo.last_stats, dict):
                last_stats = ppo.last_stats
            
            # Selection ratio 정보 추가
            if len(action) > 0:
                current_sel_ratio = float(action.sum().item()) / len(action)
                # NaN 체크 및 안전한 변환
                if torch.isnan(torch.tensor(current_sel_ratio)) or not torch.isfinite(torch.tensor(current_sel_ratio)):
                    current_sel_ratio = 0.5  # 기본값
                last_stats["selection_ratio"] = current_sel_ratio
            else:
                last_stats["selection_ratio"] = 0.5  # 빈 action일 때 기본값

            # 정책 점수 계산 (효율적인 이동평균)
            ma_window = min(200, len(total_rewards))
            if len(total_rewards) == 0:
                mean_reward_ma = 0.0
            else:
                mean_reward_ma = float(np.mean(list(total_rewards)[-ma_window:]))

            policy_score = compute_policy_score(mean_reward_ma, last_stats, cfg, episode)

            # 베스트 액터 저장 및 early stopping 체크
            if policy_score > best_score + cfg.min_delta:
                best_score = policy_score
                patience_counter = 0
                current_selection_count = int(action.sum().item())

                save_best_actor(
                    ppo.actor,
                    meta,
                    best_actor_path,
                    selection_count=current_selection_count
                )
            else:
                patience_counter += 1

            # Early stopping 체크
            if patience_counter >= cfg.patience:
                print(f"[Fold {fold}] Early stopping at episode {episode} (patience={cfg.patience})")
                pbar.set_description(f"Fold {fold} [EARLY STOP]")
                break

            # 진행률 업데이트
            progress_dict[f'fold_{fold}'] = {
                'current_episode': episode,
                'total_episodes': cfg.max_episodes,
                'best_score': best_score,
                'status': 'training',
                'reward_mean': mean_reward_ma,
                'patience': patience_counter
            }

            # tqdm 설명 업데이트 (더 많은 정보 표시)
            pbar.set_postfix({
                'reward': f'{mean_reward_ma:.3f}',
                'best': f'{best_score:.3f}',
                'sel': f'{sel_ratio:.2f}',
                'pat': f'{patience_counter}/{cfg.patience}'
            })

            # 주기적 메모리 정리 (매 100 에피소드마다)
            if episode % 100 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        pbar.close()

        # GPU 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 진행률 상태 업데이트
        progress_dict[f'fold_{fold}']['status'] = 'evaluating'

        # Save selection ratio history for visualization
        selection_log_path = os.path.join(cfg.info_dir, f"selection_history_fold{fold}.json")
        try:
            import json
            with open(selection_log_path, 'w') as f:
                json.dump(selection_history, f, indent=2)
            print(f"[Fold {fold}] Selection history saved to {selection_log_path}")
        except Exception as e:
            print(f"[Fold {fold}] Warning: Could not save selection history: {e}")

        # fold 종료: best actor 로드해서 최종 평가
        final_test_metrics_for_fold, sel_idx, sel_labels = evaluate_with_best(cfg, fold, device, env, best_actor_path)

        try:
            env.save_selection_artifacts(fold, out_dir=cfg.info_dir)
        except Exception as e:
            print(f"[Fold {fold}] Warning: Could not save selection artifacts: {e}")

        save_path = os.path.join(cfg.info_dir, f"selected_idx_fold{fold}.txt")
        with open(save_path, "w") as f:
            f.write(f"Total selected: {len(sel_idx)}\n")
            f.write("Index\tLabel\n")
            for idx, label in zip(sel_idx, sel_labels):
                f.write(f"{idx}\t{label}\n")

        plt.figure(figsize=(8,4))
        plt.plot(range(1, len(total_rewards) + 1), list(total_rewards), marker='o', linestyle='-')
        plt.title(f"Fold {fold} Reward Trajectory (Optimized)")
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)
        plt.savefig(os.path.join(cfg.info_dir, f"fold{fold}_optimized_reward_trajectory.png"))
        plt.close()

        # 진행률 상태 업데이트
        progress_dict[f'fold_{fold}']['status'] = 'completed'

        # 결과를 큐에 저장
        result_queue.put({
            'fold': fold,
            'test_metrics': final_test_metrics_for_fold,
            'sel_idx': sel_idx,
            'sel_labels': sel_labels,
            'device': str(device),
            'model_type': 'optimized_lightweight',
            'episodes_trained': episode,
            'early_stopped': patience_counter >= cfg.patience,
            'success': True
        })
        
        print(f"[Fold {fold}] Optimized training completed successfully on {device}! (Episodes: {episode})")
        
    except Exception as e:
        print(f"[Fold {fold}] Error occurred: {str(e)}")
        traceback.print_exc()
        progress_dict[f'fold_{fold}']['status'] = 'failed'
        result_queue.put({
            'fold': fold,
            'error': str(e),
            'success': False
        })


def summarize_cv(all_test_metrics, log_to_wandb=False):
    print("\nOptimized lightweight training complete. Summarizing 5-Fold CV results.")
    if not all_test_metrics or all_test_metrics[0] is None:
        print("No test metrics were recorded."); return

    keys = all_test_metrics[0].keys()
    avg = {k: np.mean([m[k] for m in all_test_metrics if m]) for k in keys}
    std = {k: np.std([m[k] for m in all_test_metrics if m]) for k in keys}

    print("\n=== PPO 5-Fold CV Test Metrics (optimized lightweight model) ===")
    for k in keys:
        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

    if log_to_wandb:
        wandb.log({**{f"cv/{k}_mean": avg[k] for k in keys},
                   **{f"cv/{k}_std":  std[k] for k in keys}})


def train(force_cpu=False):
    cfg = get_ppo_config()
    os.makedirs(cfg.info_dir, exist_ok=True)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name + "_optimized", config=vars(cfg))

    # GPU 정보 출력
    if torch.cuda.is_available() and not force_cpu:
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s) detected")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        print("Using optimized lightweight models for maximum performance")
    else:
        print("Using CPU for optimized lightweight training")

    # cfg를 딕셔너리로 변환 (멀티프로세싱 전달용)
    cfg_dict = vars(cfg)
    
    # 멀티프로세싱 설정
    num_processes = min(5, mp.cpu_count())  # 최대 5개 프로세스 (fold 수)
    device_mode = "CPU" if force_cpu or not torch.cuda.is_available() else "GPU"
    print(f"Starting optimized multiprocessing training with {num_processes} processes on {device_mode}")
    print(f"Optimizations: validation 5x less frequent, early stopping, memory management")
    print("d_model: 256→128, layers: 4→2, heads: 8→4, ff: 512→256")
    print("clf_every: 10→50, rollout: 256→128, patience: 500")
    print("=" * 80)
    
    # 결과 수집용 큐 및 진행률 공유 딕셔너리
    result_queue = Queue()
    manager = Manager()
    progress_dict = manager.dict()
    
    # 프로세스 생성 및 시작
    processes = []
    start_time = time.time()
    
    print("\n" + "="*90)
    print("Optimized Training Progress Monitor (tqdm)")
    print("="*90)
    
    for fold in range(1, 6):
        p = Process(target=train_single_fold, args=(fold, cfg_dict, result_queue, progress_dict, force_cpu))
        p.start()
        processes.append(p)
        print(f"Started optimized process for fold {fold} (PID: {p.pid})")
    
    # 모든 프로세스 완료 대기
    print("Waiting for all optimized processes to complete...")
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 결과 수집
    all_test_metrics = []
    results = {}
    early_stopped_count = 0
    
    while not result_queue.empty():
        result = result_queue.get()
        if result['success']:
            fold = result['fold']
            device_used = result.get('device', 'unknown')
            model_type = result.get('model_type', 'optimized_lightweight')
            episodes_trained = result.get('episodes_trained', 'unknown')
            early_stopped = result.get('early_stopped', False)
            
            results[fold] = result
            all_test_metrics.append(result['test_metrics'])
            
            status = "EARLY STOPPED" if early_stopped else "COMPLETED"
            if early_stopped:
                early_stopped_count += 1
                
            print(f"Fold {fold} {status} on {device_used} ({model_type}) - Episodes: {episodes_trained}")
        else:
            print(f"Fold {result['fold']} failed: {result['error']}")
    
    # 결과 요약
    if all_test_metrics:
        summarize_cv(all_test_metrics, log_to_wandb=cfg.log_to_wandb)
    
    print(f"\nOptimized training summary:")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average time per fold: {total_time/5/60:.1f} minutes")
    print(f"Early stopped folds: {early_stopped_count}/5")
    
    if cfg.log_to_wandb:
        wandb.finish()
    
    print("All optimized folds completed!")
    return results


if __name__ == "__main__":
    # 멀티프로세싱을 위한 설정
    mp.set_start_method('spawn', force=True)
    
    print("=" * 80)
    print("Optimized Lightweight Multiprocessing PPO Training")
    print("Memory optimized: ~75% reduction in model size")
    print("Performance optimized: 5x less validation, early stopping, memory management")
    print("Model architecture changes:")
    print("  - d_model: 256→128")
    print("  - layers: 4→2")
    print("  - heads: 8→4")
    print("  - ff: 512→256")
    print("Training changes:")
    print("  - clf_every: 10→50")
    print("  - rollout: 256→128")
    print("  - patience: 500")
    print("=" * 80)
    
    # 최적화된 경량화 모델 사용
    train(force_cpu=False)
