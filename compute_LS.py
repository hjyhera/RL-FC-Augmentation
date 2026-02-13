import torch
import numpy as np
from scipy.stats import ks_2samp
from load_brain_data import BrainDataset
from load_gan_data import GANSyntheticDataset
from ppo_config import get_ppo_config

def gpu_LS(real,gen):
    t_gen = gen 
    t_real = real  

    dist_real = torch.cdist(t_real, t_real)  # ICD 1
    dist_real = torch.flatten(torch.tril(dist_real, diagonal=-1))  # remove repeats
    dist_real = dist_real[dist_real.nonzero()].flatten()  # remove distance=0 for distances btw same data points
    dist_real = dist_real.cpu().numpy()

    dist_gen = torch.cdist(t_gen, t_gen)  # ICD 2
    dist_gen = torch.flatten(torch.tril(dist_gen, diagonal=-1))  # remove repeats
    dist_gen = dist_gen[dist_gen.nonzero()].flatten()  # remove distance=0 for distances btw same data points
    dist_gen = dist_gen.cpu().numpy()

    distbtw = torch.cdist(t_gen, t_real)  # BCD
    distbtw = torch.flatten(distbtw)
    distbtw = distbtw.cpu().numpy()

    D_Sep_1, _ = ks_2samp(dist_real, distbtw)
    D_Sep_2, _ = ks_2samp(dist_gen, distbtw)

    return 1- np.max([D_Sep_1, D_Sep_2])  # LS=1-DSI



if __name__ == "__main__":
    args = get_ppo_config()
    for fold in range(1, 6):
        txt_file = args.train_folds[fold]
        brain_dataset = BrainDataset(txt_file, args.data_dir)
        gan_dataset = GANSyntheticDataset(args.data_dir_gan, fold=fold)

        real_data = torch.stack([brain_dataset[i][0] for i in range(len(brain_dataset))])
        gen_data = torch.stack([gan_dataset[i][0] for i in range(len(gan_dataset))])

        print(f"Fold {fold} | real_data shape: {real_data.shape}, gen_data shape: {gen_data.shape}")
        ls_score = gpu_LS(real_data, gen_data)
        print(f"Fold {fold} LS score: {ls_score}")



