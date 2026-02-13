import os
from datetime import datetime

class PPOConfig:
    def __init__(self):
        self.train_folds = {
            1: r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_1.txt',
            2: r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_2.txt',
            3: r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_3.txt',
            4: r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_4.txt',
            5: r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_5.txt',
        } 
        self.valid_folds = {
            1: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_1_val.txt',
            2: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_2_val.txt',
            3: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_3_val.txt',
            4: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_4_val.txt',
            5: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_5_val.txt',
        } 
        self.test_folds = {
            1: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_1_test.txt',
            2: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_2_test.txt',
            3: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_3_test.txt',
            4: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_4_test.txt',
            5: r'/home/user/RL-FC-Augmentation/MDD_val_test/fold_5_test.txt',
        }

        # Data directories
        self.data_dir = r'/home/user/RL-FC-Augmentation/MS_MDD_Harvard_FC_fisher-20250107T100946Z-001/MS_MDD_Harvard_FC_fisher'
        self.topology_dir = r'/home/user/RL-FC-Augmentation/GAN/Edge_topology'
        self.model_dir = r'/home/user/RL-FC-Augmentation/GAN/Model'
        self.data_dir_gan = r'/home/user/RL-FC-Augmentation/GAN/Pseudo_data/model_20250830_103826'
        self.model_alignment_root = r'/home/user/RL-FC-Augmentation/GraphAE/Model/model_20250828_165235'
        self.info_dir = 'logs/synthetic_info'
        self.precomputed_path = 'precomputed_rewards.npz'

        # Training parameters
        self.state_dim = 6216
        self.max_episodes = 20000
        self.seed_base = 56
        self.seed = 56 
        self.rollout_episodes = 256
        self.min_delta = 0.000
        self.temperature = 1.5     

        # Reward weights 
        self.alpha_fidelity = 0.5      
        self.alpha_alignment = 0.5     
        self.alpha_diversity = 1.0      
        self.alpha_utility = 1.0      

        # Classifier parameters 
        self.C = 1.0          
        self.l1_ratio = 0.7    
        self.max_iter = 5000   

        # PPO parameters 
        self.clip_ratio = 0.5
        self.n_epochs = 2
        self.entropy_coef = 0.5       
        self.adv_scale = 0.9          
        self.lr_actor = 3e-4
        self.lr_critic = 3e-6       
        self.wd_actor = 1e-5
        self.wd_critic = 5e-5       
        self.mb_episodes = 64
        
        # Gradient clipping for stability
        self.grad_clip_actor = 1.4
        self.grad_clip_critic = 1.4
        
        # Selection constraints 
        self.min_selection_ratio = 0.15
        self.max_selection_ratio = 0.85
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

        # Utility reward
        self.clf_every = 100           
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
        self.wandb_project = 'RL-FC-Augmentation'
        self.wandb_run_name = 'ppo_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_to_wandb = True

def get_ppo_config():
    return PPOConfig()
