# Reinforcement Learning with Multi Objective Rewards for Functional Connectivity Augmentation

## 🔥 Motivation

Functional connectivity (FC) derived from resting-state fMRI provides promising biomarkers for Major Depressive Disorder (MDD).  
However, limited and noisy annotations make deep diagnostic models highly brittle.    
While generative augmentation can alleviate data scarcity, uncontrolled synthetic samples often introduce biologically implausible artifacts.  

To address this, we propose a reinforcement learning–based synthetic data filtering framework that curates synthetic FC samples using **multi-objective rewards**, balancing fidelity, diversity, alignment, and diagnostic utility.

## 📋 Overview
<p align="center">
  <img src="./images/overview.jpg" width="90%" alt="Overview">
</p>
    
> **Fig 1.**
> Our framework operates in four stages:

- **Candidate Generation:** Pretrain a GAN to synthesize FC candidates  
- **RL-based Selection:** A PPO agent filters candidates under multi-objective rewards  
- **Classifier Training:** Train the diagnosis model using curated augmentations  
- **Inference:** Apply the trained classifier to unseen subjects

## 🎯 Multi-Objective Reward Formulation

We define a vector-valued reward:


r($s_t$, $a_t$) = ( $r_F$, $r_D$, $r_A$, $r_U$)


- **Fidelity ($r_F$):** discriminator realism score  
- **Diversity ($r_D$):** encourages coverage and prevents mode collapse  
- **Alignment ($r_A$):** preserves manifold structure via autoencoder reconstruction  
- **Utility ($r_U$):** improves downstream diagnostic performance  

Rewards are aggregated using **Generalized Power Mean scalarization** for stable training.

## ⭐ Contributions

(i) **RL-based Synthetic Data Filtering Framework.**  
We propose a reinforcement learning–based synthetic data filtering framework that formulates sample selection as a **vector-valued multi-objective reward modeling** problem, and optimizes a scalarized objective to enable **quality-controlled functional connectivity augmentation** for MDD diagnosis.

(ii) **Interpretable Selection Dynamics Analysis.**  
We introduce an analysis framework that quantifies **selection dynamics** throughout training and connects high-reward synthetic samples to **interpretable functional connectivity motifs**, ensuring transparent and reproducible curation.

## 📂 Dataset & Experimental Setup

- Dataset: REST-meta-MDD
```bash
Yan, C. G., Wang, X., Zuo, X. N., & Zang, Y. F. (2016). DPABI: Data Processing & Analysis for (Resting-State) Brain Imaging. 
Neuroinformatics, 14(3), 339-351. https://doi.org/10.1007/s12021-016-9312-x

REST-meta-MDD Project: https://rfmri.org/REST-meta-MDD
```
- Subjects: 249 MDD / 228 Normal Controls  
- Parcellation: Harvard-Oxford Atlas (112 ROIs)  
- FC Construction: Pearson correlation + Fisher z-transform  
- Evaluation: Subject-disjoint 5-fold cross-validation

## 📊 Results

We evaluated our framework on the **REST-meta-MDD** dataset using a 5-fold cross-validation scheme.   
The results demonstrate that our multi-objective RL agent consistently selects high-quality synthetic samples that improve downstream diagnostic performance.

### 1. Comparison with GAN-based Augmentation and RL-based Selection Approaches
Our method outperforms representative GAN-based augmentation and RL-based selection baselines across all metrics (Accuracy, Sensitivity, Specificity, and F1-score).

<p align="center">
  <img src="./images/Table1.png" width="60%" alt="Comparison with baselines">
</p>

> **Table 1.** Classification performance comparison. Our method achieves the highest accuracy (**67.71%**), sensitivity (**70.00%**), specificity (**65.22%**) and F1-score (**69.18%**), significantly surpassing the no-augmentation baseline and other competitive methods.

### 2. Dynamics of Reward-Selection Interactions
The following graph illustrates the evolution of reward signals and the selection ratio during training.

<p align="center">
  <img src="./images/average_across_folds_only.png" width="60%" alt="Training Dynamics">
</p>

> **Fig 2.** Temporal dynamics of multi-objective rewards and selection ratio. The learned policy consistently accepts approximately 55% of synthetic candidates on average. A three-phase pattern emerges: (1) **Utility ($r_U$)-driven expansion**, (2) **Diversity ($r_D$)-driven coverage**, and (3) **Fidelity/Alignment ($r_F, r_A$)-constrained stabilization**, resulting in a robust selection policy. 

### 3. Comparison of Selection Strategies
<p align="center">
  <img src="./images/Table2.png" width="60%" alt="Performance across selection methods.">
</p>

> **Table 2.** We investigated downstream effects of candidate selection under a fixed acceptance budget of approximately 55%.
- Random: uniform, structure-agnostic baseline.
- Coverage: farthest-first dispersion in the embedding space, prone to outliers.
- Centroid: cluster then keep medoids, sensitive to k and cluster stability.
- Realism: rank by discriminator-based $r_F$, high fidelity at potential cost of diversity.
- Ours: PPO-based multi-objective policy optimizing a balanced trade-off among fidelity, diversity, alignment, and downstream utility.


## 🧪 Ablation Studies & Analysis
To validate the effectiveness of our proposed components, we conducted extensive ablation studies.

### 1. Impact of Reward Components
We analyzed the contribution of each reward objective ($r_F, r_D, r_A, r_U$).

<p align="center">
  <img src="./images/Table3.png" width="60%" alt="Ablation Study on Rewards">
</p>

> **Table 3:** Ablation study on the contribution of each reward component. Removing any single objective leads to performance degradation, confirming the necessity of the multi-objective framework.

### 2. Impact of Scalarization Strategy
We investigated the effect of reward scalarization strategies: uniform vs. tuned weights.

<p align="center">
  <img src="./images/Table4.png" width="60%" alt="Scalarization Strategy">
</p>

> **Table 4.** Uniform linear scalarization(lambda_{all}=1.0) reduces F1 by 1.8% and doubles variance(std 2.45 to 4.85), as dense structural rewards(r_F,r_A) overshadow sparse utility reward(r_U), biasing optimization toward easier objectives. Prioritizing primary targets(lambda=1.0) over regularizers(lambda=0.5) balances learning, achieving the highest Accuracy(67.71%) and Stability(std: 1.04).

### 3. Impact of PPO
We further evaluate the impact of using a learned PPO-based policy for candidate selection, compared to a context-unaware static scoring baseline. 

<p align="center">
  <img src="./images/Table5.png" width="60%" alt="Static Scoring">
</p>

> **Table 5.** Compared to a context-unaware "Static Scoring" baseline(selecting features independently), our RL framework improved F1 by 3.1%. The agent analyzes functional connectivity(s_t) to identify redundancy, mitigating mode collapse and ensuring diversity crucial for robust diagnostic generalization. 

## ⚙️ Implementation Details

### 1. Computational Resources
All experiments were conducted on a workstation equipped with:

- GPU: NVIDIA RTX 3090 (24GB)
- Framework: Python 3.9, PyTorch 2.0 
- Training time: approx. 12 hours on a single RTX 3090

### 2. PPO Training Hyperparameters

The PPO-based selection agent was trained with the following configuration:

- **Policy network:** 2-layer MLP (hidden dim = 256)
- **AdamW Learning rate:** 3e-4  
- **Discount factor ($\gamma$):** 0.99   
- **Clip range ($\epsilon$):** 0.5  
- **Entropy coefficient:** 0.5  
- **Batch size:** 256  
- **Training steps:** 1M timesteps  

### 3. Classifier Hyperparameters

The classifier was trained with the following configuration:

- **Classifier network:** 3-layer MLP (hidden dim = 256)
- **Learning rate:** 1e-4
- **Batch size:** 256
- **Training steps:** 200 epochs
  
## 🔧 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/hjyhera/Reinforcement-Learning-with-Multi-Objective-Rewards-for-Functional-Connectivity-Augmentation.git
```

### 2. Create Environment
We recommend using a conda environment:

```bash
conda create -n py39 python=3.9
conda activate py39
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Generator (GAN)
First, train the FC generator to synthesize candidate samples:

```bash

```

### 5. Train the PPO Selection Agent
Train the PPO agent with multi-objective rewards:

```bash
python train_cycle_mp_light_optimized.py
```

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License  
(**CC BY-NC-SA 4.0**).

- You are free to share and adapt the material for **non-commercial** purposes.
- Proper attribution must be given.
- Any derivative work must be distributed under the same license.

For details, see: https://creativecommons.org/licenses/by-nc-sa/4.0/

## 🙏 Acknowledgements

This work was supported by the Ministry of Science and ICT (MSIT), Korea,  
through the Institute of Information & Communications Technology Planning & Evaluation (IITP)  
[Artificial Intelligence Graduate School Program at Korea University, No. RS-2019-II190079;  
Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration, No. 2022-0-00871],  
the National Research Foundation of Korea (NRF) [No. RS-2023-00212498],  
and the Korea Health Industry Development Institute (KHIDI)  
under the Federated Learning-based Drug Discovery Acceleration Project (K-MELLODDY)  
[No. RS-2025-16066488].

## 📬 Contact

For questions, discussions, or collaborations, feel free to contact:

**Jiyoung Hwang**  
Korea University  
Email: hjyhera@korea.ac.kr  
GitHub: https://github.com/hjyhera

## 📌 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{hwang2026rlfc,
  title     = {Reinforcement Learning with Multi-Objective Rewards for Functional Connectivity Augmentation},
  author    = {Hwang, Jiyoung and others},
  booktitle = {Proceedings of ICASSP 2026},
  year      = {2026}
}


