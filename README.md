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

\[
r(s_t,a_t) = [r_F, r_D, r_A, r_U]^T
\]

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

- Dataset: **REST-meta-MDD**  
- Subjects: 249 MDD / 228 Normal Controls  
- Parcellation: Harvard-Oxford Atlas (112 ROIs)  
- FC Construction: Pearson correlation + Fisher z-transform  
- Evaluation: Subject-disjoint 5-fold cross-validation  


## 📊 Results

We evaluated our framework on the **REST-meta-MDD** dataset using a 5-fold cross-validation scheme. The results demonstrate that our multi-objective RL agent consistently selects high-quality synthetic samples that improve downstream diagnostic performance.

### 1. Comparison with State-of-the-Art Methods
Our method outperforms representative GAN-based augmentation and RL-based selection baselines across all metrics (Accuracy, Sensitivity, Specificity, and F1-score).

<p align="center">
  <img src="./images/Table1.png" width="90%" alt="Comparison with SOTA">
</p>

> **Table 1.** Classification performance comparison. Our method achieves the highest accuracy (**67.71%**) and F1-score (**69.18%**), significantly surpassing the no-augmentation baseline and other competitive methods.

<p align="center">
  <img src="./images/Table2.png" width="90%" alt="Performance across selection methods.">
</p>

> **Table 1.** Performance across selection methods.

---

### 2. Ablation Studies & Analysis
To validate the effectiveness of our proposed components, we conducted extensive ablation studies.

#### Impact of Reward Components & Selection Strategy
We analyzed the contribution of each reward objective ($r_F, r_D, r_A, r_U$) and compared our policy against heuristic selection strategies (Random, Coverage, Centroid, Fidelity).

<p align="center">
  <img src="./images/Table3.png" width="48%" alt="Ablation Study on Rewards">
</p>

> **Table 3:** Ablation study on the contribution of each reward component. Removing any single objective leads to performance degradation, confirming the necessity of the multi-objective framework[cite: 156, 171].

#### Impact of Scalarization Strategy
We further investigated the effect of reward scalarization weights on the agent's learning.

<p align="center">
  <img src="./images/Table4.png" width="60%" alt="Scalarization Strategy">
</p>

> **Table 4.** Uniform linear scalarization(lambda_{all}=1.0) reduces F1 by 1.8% and doubles variance(std 2.45 to 4.85), as dense structural rewards(r_F,r_A) overshadow sparse utility reward(r_U), biasing optimization toward easier objectives. Prioritizing primary targets(lambda=1.0) over regularizers(lambda=0.5) balances learning, achieving the highest Accuracy(67.71%) and Stability(std: 1.04).

<p align="center">
  <img src="./images/Table5.png" width="60%" alt="Static Scoring">
</p>

> **Table 5.** Compared to a context-unaware "Static Scoring" baseline(selecting features independently), our RL framework improved F1 by 3.1%. The agent analyzes functional connectivity(s_t) to identify redundancy, mitigating mode collapse and ensuring diversity crucial for robust diagnostic generalization. 

---

### 3. Training Dynamics
The following graph illustrates the evolution of reward signals and the selection ratio during training.

<p align="center">
  <img src="./images/average_across_folds_only.png" width="80%" alt="Training Dynamics">
</p>

> **Fig 2.** Temporal dynamics of multi-objective rewards and selection ratio. [cite_start]A three-phase pattern emerges: (1) **Utility ($r_U$)-driven expansion**, (2) **Diversity ($r_D$)-driven coverage**, and (3) **Fidelity/Alignment ($r_F, r_A$)-constrained stabilization**, resulting in a robust selection policy.
