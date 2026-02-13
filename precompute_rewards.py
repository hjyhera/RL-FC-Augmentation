import os
import sys
import torch
import numpy as np
from load_gan_data import GANSyntheticDataset
from load_brain_data import BrainDataset
from ppo_config import get_ppo_config

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from math import erf, sqrt

REPO_DIR = Path(__file__).resolve().parent / "GC-GAN-main" / "GC-GAN-main"
sys.path.insert(0, str(REPO_DIR))

from model import myGraphAE, ACDiscriminator
from config import config 


# Load topology
def load_topology(topology_dir: str):
    topology_data = {}
    for fold in range(1, 6):
        file_path = os.path.join(topology_dir, f"mrmr_MDD_Harvard_FC_map_fold_{fold}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Topology 파일을 찾을 수 없습니다: {file_path}")
        topology_data[fold] = np.load(file_path)
    return topology_data


# ====== 유틸: 로버스트 정규화 → [0,1] ======
def robust_zscore_to_unit(x, use_median: bool = True, k: float = 3.0, eps: float = 1e-8):
    """
    1) 로버스트 표준화(median/MAD 또는 mean/std)
    2) z를 [-k, k]로 클리핑
    3) 정규 CDF: 0.5*(1 + erf(z/sqrt(2))) → [0,1]  (torch.erf로 벡터화)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x

    if use_median:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4826 * mad  # robust std
        if scale < eps:
            return np.full_like(x, 0.5)
        z = (x - med) / (scale + eps)
    else:
        mu, sd = float(x.mean()), float(x.std())
        if sd < eps:
            return np.full_like(x, 0.5)
        z = (x - mu) / (sd + eps)

    z = np.clip(z, -k, k).astype(np.float32)

    z_t = torch.from_numpy(z)
    u_t = 0.5 * (1.0 + torch.erf(z_t / np.sqrt(2.0)))   # [N]
    return u_t.numpy().astype(np.float32)


# ====== 유틸: 개별/폴드 분포 저장(따로) ======
def save_distribution_plots_separate(d_vals, m_vals, fold, outdir="figs", bins: int = 30, share_bins: bool = True):
    """
    d_vals: ALIGNMENT 'good' (예: -MSE 정규화 후 [0,1])
    m_vals: FIDELITY 'good' (예: log-odds 정규화 후 [0,1])
    """
    os.makedirs(outdir, exist_ok=True)
    d_vals = np.asarray(d_vals, dtype=float)
    m_vals = np.asarray(m_vals, dtype=float)

    # 통계 출력
    def _desc(name, x):
        print(f"[Fold {fold}] {name} | n={x.size}, mean={x.mean():.4f}, std={x.std():.4f}, "
              f"min={x.min():.4f}, p50={np.median(x):.4f}, max={x.max():.4f}")
    _desc("ALIGNMENT(good)", d_vals)
    _desc("FIDELITY(good)", m_vals)

    # 공통 bin 경계
    if share_bins:
        lo = float(min(d_vals.min(), m_vals.min()))
        hi = float(max(d_vals.max(), m_vals.max()))
        bin_edges = np.linspace(lo, hi, bins + 1)
    else:
        bin_edges = bins

    # ALIGNMENT 히스토그램 (단독)
    plt.figure(figsize=(6, 4))
    plt.hist(d_vals, bins=bin_edges)
    plt.title(f"ALIGNMENT (good) Histogram - Fold {fold}")
    plt.xlabel("Value"); plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"reward_hist_alignment_fold{fold}.png"))
    plt.close()

    # FIDELITY 히스토그램 (단독)
    plt.figure(figsize=(6, 4))
    plt.hist(m_vals, bins=bin_edges if share_bins else bins)
    plt.title(f"FIDELITY (good) Histogram - Fold {fold}")
    plt.xlabel("Value"); plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"reward_hist_fidelity_fold{fold}.png"))
    plt.close()


# ====== 모델별 스코어 계산 ======
def precompute_rewards(topologies, synthetic_data, model_alignment_root, discriminators, device, save_path=None,
                       figs_outdir="figs", use_median=True, clip_k=3.0):
    """
    - ALIGNMENT: 재구성 MSE (작을수록 좋음) → good = robust_zscore_to_unit(-MSE)
    - FIDELITY: log-odds(real vs fake) (클수록 좋음) → good = robust_zscore_to_unit(logodds)
    저장:
      - d_raw_mse, m_raw_logodds (원시)
      - d_good, m_good ([0,1] 정규화)
    """
    mse_loss = torch.nn.MSELoss()

    def comp_alignment(alignment, mat, lbl, topo):
        # 단일 샘플 처리 (입력: mat [n,n])
        args = config()
        lbl_emb = alignment.embedding(lbl).view(1, 112, 1)
        with torch.no_grad():
            h = torch.cat([mat.unsqueeze(0), lbl_emb], dim=2)  # [1,112,2]
            h = alignment.act(alignment.econv1(h, topo))
            z = alignment.act(alignment.econv2(h, topo))
            h = torch.cat([z, lbl_emb], dim=2)
            h = alignment.act(alignment.gconv1(h, topo))
            recon = alignment.act(alignment.gconv2(h, topo))
            recon = recon.reshape((-1, 112 * args.g_channels[2]))
            recon = alignment.tanh(alignment.linear(recon))
            recon = recon.reshape((1, 112, 112))
            recon = 0.5 * (recon + recon.transpose(1, 2))
            recon = torch.arctanh(0.999 * recon)
            return mse_loss(mat.unsqueeze(0), recon).item()  # 스칼라 MSE

    def comp_fidelity_logodds(fidelity, mat, topo):
        # 단일 샘플 처리: 4-클래스 로짓에서 log-odds(real vs fake)
        with torch.no_grad():
            logits, _ = fidelity(mat, topo)           # [1,4] 가정: [NC_r, MDD_r, NC_f, MDD_f]
            l_real = torch.logsumexp(logits[:, :2], dim=1)   # [1]
            l_fake = torch.logsumexp(logits[:, 2:], dim=1)   # [1]
            logodds = l_real - l_fake                        # [1]
            return float(logodds.item())

    # 결과 저장 구조
    alignment_good = {f: [] for f in range(1, 6)}       # [0,1] good
    fidelity_good = {f: [] for f in range(1, 6)}       # [0,1] good
    alignment_raw_mse = {f: [] for f in range(1, 6)}    # 원시 MSE
    fidelity_raw_logodds = {f: [] for f in range(1, 6)}# 원시 log-odds

    for fold in range(1, 6):
        alignment = myGraphAE().to(device).eval()
        fn = f"GAE_model_{fold}_final.pth"
        path = os.path.join(model_alignment_root, fn)
        alignment.load_state_dict(torch.load(path, map_location=device, weights_only=True))

        topo = torch.tensor(topologies[fold], dtype=torch.float32, device=device)

        # 원시값 수집
        for flat_vec, lbl in synthetic_data[fold]:
            mat = flat_to_mat(flat_vec, device)
            lbl = lbl.to(device)

            d_mse = comp_alignment(alignment, mat, lbl, topo)                 # 작을수록 좋음
            m_lo  = comp_fidelity_logodds(discriminators[fold], mat, topo)  # 클수록 좋음

            alignment_raw_mse[fold].append(d_mse)
            fidelity_raw_logodds[fold].append(m_lo)

        # numpy 변환
        d_raw = np.asarray(alignment_raw_mse[fold], dtype=np.float32)         # MSE
        m_raw = np.asarray(fidelity_raw_logodds[fold], dtype=np.float32)     # log-odds

        # "좋을수록 큰 값"으로 변환 후 [0,1] 정규화 
        d_good = robust_zscore_to_unit(-d_raw, use_median=use_median, k=clip_k)  # -MSE → good
        m_good = robust_zscore_to_unit( m_raw, use_median=use_median, k=clip_k)  # log-odds → good

        alignment_good[fold] = d_good.tolist()
        fidelity_good[fold] = m_good.tolist()

        # 분포 히스토그램(각각)
        save_distribution_plots_separate(
            d_vals=alignment_good[fold],
            m_vals=fidelity_good[fold],
            fold=fold,
            outdir=figs_outdir,
            bins=30,
            share_bins=True
        )

    if save_path:
        # 원시/정규화 값 모두 저장
        np.savez_compressed(
            save_path,
            alignment_good=alignment_good,
            fidelity_good=fidelity_good,
            alignment_raw_mse=alignment_raw_mse,
            fidelity_raw_logodds=fidelity_raw_logodds,
        )
        print(f"Saved rewards → {save_path}")

    return alignment_good, fidelity_good


# ====== 유틸: 벡터→대칭행렬 ======
def flat_to_mat(flat_vec, device, n: int = 112):
    idx = torch.triu_indices(n, n, offset=1)
    mat = torch.zeros(n, n, device=device)
    mat[idx[0], idx[1]] = flat_vec.to(device)
    return mat + mat.T


# ====== 유틸: 판별기 로드 ======
def load_discriminator(fold, model_dir, device):
    path = os.path.join(model_dir, f"D_model_{fold}_final.pth")
    fidelity = ACDiscriminator().to(device)
    fidelity.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    fidelity.eval()
    return fidelity


# ====== 메인 ======
if __name__ == "__main__":
    args = get_ppo_config()

    topology_dir     = args.topology_dir
    data_dir_gan     = args.data_dir_gan
    model_alignment_root  = args.model_alignment_root
    model_dir        = args.model_dir
    save_path        = "precomputed_rewards.npz"
    figs_outdir      = "figs"  # 필요 시 변경

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Precompute] Using device: {device}")

    # 토폴로지 로드
    topologies = load_topology(topology_dir)

    # 판별기 로드
    discriminators = {f: load_discriminator(f, model_dir, device) for f in range(1, 6)}

    # 합성데이터 로드 (fold 1..5)
    synthetic_data = {f: GANSyntheticDataset(data_dir_gan, f) for f in range(1, 6)}

     # 실행
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    d_good, m_good = precompute_rewards(
        topologies=topologies,
        synthetic_data=synthetic_data,
        model_alignment_root=model_alignment_root,
        discriminators=discriminators,
        device=device,
        save_path=save_path,
        figs_outdir=figs_outdir,
        use_median=True,   # median/MAD 기반 로버스트 표준화
        clip_k=3.0         # z 클리핑 강도
    )
    print(f"[Precompute] Done. Saved to: {save_path}")