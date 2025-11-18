import os
from datetime import datetime

class PPOConfig:
    def __init__(self):
        self.train_folds = {
            1: r'/home/user/Desktop/intern_project_final/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_1.txt',
            2: r'/home/user/Desktop/intern_project_final/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_2.txt',
            3: r'/home/user/Desktop/intern_project_final/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_3.txt',
            4: r'/home/user/Desktop/intern_project_final/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_4.txt',
            5: r'/home/user/Desktop/intern_project_final/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_5.txt',
        }
        self.valid_folds = {
            1: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_1_val.txt',
            2: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_2_val.txt',
            3: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_3_val.txt',
            4: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_4_val.txt',
            5: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_5_val.txt',
        }
        self.test_folds = {
            1: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_1_test.txt',
            2: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_2_test.txt',
            3: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_3_test.txt',
            4: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_4_test.txt',
            5: r'/home/user/Desktop/intern_project_final/ppo_final/MDD_val_test/fold_5_test.txt',
        }
        
        # Data directories
        self.data_dir = r'/home/user/Desktop/intern_project_final/MS_MDD_Harvard_FC_fisher-20250107T100946Z-001/MS_MDD_Harvard_FC_fisher'
        self.topology_dir = r'/home/user/Desktop/intern_project_final/ppo_final/GAN/Edge_topology'
        self.model_dir = r'/home/user/Desktop/intern_project_final/ppo_final/GAN/Model'
        self.data_dir_gan = r'/home/user/Desktop/intern_project_final/ppo_final/GAN/Pseudo_data/model_20250830_103826'
        self.model_dcae_root = r'/home/user/Desktop/intern_project_final/ppo_final/GraphAE/Model/model_20250828_165235'
        self.info_dir = 'logs/synthetic_info'
        self.precomputed_path = 'precomputed_rewards.npz'
        self.state_dim = 6216

        # Training parameters
        self.max_episodes = 20000
        self.seed_base = 56
        self.seed = 56 #56
        self.rollout_episodes = 256
        self.min_delta = 0.000
        # self.grad_clip = None     # UNUSED: replaced by grad_clip_actor/critic
        self.temperature = 1.5        # 1.5 

        # Reward weights (rebalanced for better selection)
        self.alpha_disc = 1.0      # 1.0 → 0.5 (base quality 감소)
        self.alpha_dcae = 1.0     # 1.0 → 0.5 (base quality 감소)
        self.alpha_ls = 0.0        # 1.0 → 1.5 (likeness 중요도 증가)
        self.alpha_val = 0.5       # 0.1 → 0.2 (validation 중요도 증가)
        # self.alpha_div = 0.5      # UNUSED: diversity reward weight

        # Classifier parameters (overfitting prevention)
        self.C = 1.0           # 0.1 → 1.0 (정규화 완화)
        self.l1_ratio = 0.7    # 0.5 → 0.7 (L1 정규화 강화)
        self.max_iter = 5000    # 500 → 300 (조기 종료)

        # PPO parameters (optimized values)
        self.clip_ratio = 0.5
        self.n_epochs = 2
        self.entropy_coef = 0.5        # 0.1 → 0.2 (극단적 선택 방지)
        self.adv_scale = 0.9           # 5.0 → 1.0 (스케일 문제 해결)
        self.lr_actor = 3e-4
        self.lr_critic = 3e-6          # 1e-4 → 3e-4 (3배 증가로 빠른 수렴)
        self.wd_actor = 1e-5
        self.wd_critic = 5e-5          # 1e-3 → 5e-4 (critic 정규화 완화)
        self.mb_episodes = 64
        
        # Gradient clipping for stability
        self.grad_clip_actor = 1.4
        self.grad_clip_critic = 1.4
        
        # Learning rate scheduling (UNUSED)
        # self.lr_decay_factor = 0.99
        # self.lr_decay_interval = 100
        # self.min_lr_factor = 0.1
        
        # Selection ratio constraints (정책 붕괴 방지)
        self.min_selection_ratio = 0.15
        self.max_selection_ratio = 0.85
        
        # Policy score parameters (균형잡힌 정책 평가) - UNUSED
        # self.warmup_episodes = 50
        # self.score_w_reward = 1.0
        # self.score_w_kl = 0.3
        # self.score_w_clip = 0.4
        # self.score_w_entropy = 0.15
        # self.score_w_advstd = 0.1
        # self.score_w_selection = 0.75
        
        # Early stopping thresholds (UNUSED)
        # self.es_kl_hi = 0.2
        # self.es_clip_frac_hi = 0.9
        # self.es_entropy_lo = 0.04
        # self.es_adv_std_lo = 0.005
        self.patience = 1000
        
        # Reward shaping
        self.a = 1.0
        self.frac = 0.3
        self.q_lo = 0.05
        self.q_hi = 0.95
        self.use_median = True
        self.clip_k = 1.0
        self.eps = 1e-8

        # Diversity reward
        self.m = 10
        self.softness = 0.05
        self.margin = 0.05
        self.ema_beta = 0.9
        self.soft = True

        # Validation reward
        self.clf_every = 100            # 10 → 50 (5배 감소)
        self.val_metric = "val_loss"

        # === NEW PPO AGENT CONFIG PARAMETERS ===
        
        # EMA baseline parameters
        self.use_ema_baseline = True
        self.baseline_beta = 0.9
        self.baseline_init = 0.0
        
        # Token-level credit assignment (dual EMA)
        self.use_token_credit = True
        self.baseline_sel_init = 0.0
        self.baseline_nosel_init = 0.0
        self.baseline_sel_beta = 0.9
        self.baseline_nosel_beta = 0.9
        
        # Critic training (recommended False for dual EMA)
        self.train_critic = False
        
        # Entropy targeting
        self.target_entropy = 0.30    # Bernoulli/2-class: 0.2~0.5
        self.entropy_beta = 0.02
        self.entropy_min = 1e-4
        self.entropy_max = 5e-2
        
        # KL targeting / adaptive LR
        self.target_kl = 0.01
        self.kl_stop_mult = 2.0
        self.lr_min = 1e-5
        self.lr_max = 1e-3
        
        # Logit regularization & clamp
        self.logit_l2 = 1e-4
        self.logit_clamp = 6.0        # prevent saturation
        
        # Temperature adaptation
        self.temperature_min = 1.0
        self.temperature_max = 2.0
        self.temp_up = 1.1
        self.temp_down = 0.9

        # Wandb
        self.wandb_project = 'intern_project_apollox'  # 최신 프로젝트명 사용
        self.wandb_run_name = 'ppo_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_to_wandb = True

def get_ppo_config():
    return PPOConfig()
