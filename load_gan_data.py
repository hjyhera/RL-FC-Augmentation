import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

data_dir_gan = r'D:\intern_project_final\ppo_final\GAN\Pseudo_data\model_20250830_103826'


class GANSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_gan, fold):
        self.data_dir_gan = data_dir_gan
        self.data_files = sorted([f for f in os.listdir(data_dir_gan)
                                  if f.startswith(f'pseudo_data_{fold}_') and f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(data_dir_gan)
                                   if f.startswith(f'pseudo_label_{fold}_') and f.endswith('.npy')])

        all_fc = []
        all_label = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            data = np.load(os.path.join(data_dir_gan, data_file))  # (num_samples, n, n)
            labels = np.load(os.path.join(data_dir_gan, label_file))  # (num_samples,)
            n = data.shape[1]
            triu_idx = np.triu_indices(n, k=1)
            all_fc.append(np.array([sample[triu_idx] for sample in data]))
            all_label.append(labels)
        self.data = torch.tensor(np.concatenate(all_fc, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(all_label, axis=0), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
if __name__ == "__main__":
    dataset = GANSyntheticDataset(data_dir_gan, fold=5)  
    print(f"총 샘플 수: {len(dataset)}")
    fc_data, label = dataset[4]
    print("\n=== GANSyntheticDataset 첫 샘플 예시 ===")
    print(f"fc_data shape: {fc_data.shape}, dtype: {fc_data.dtype}, sample values: {fc_data[:10]}")
    print(f"label shape: {label.shape}, dtype: {label.dtype}, value(s): {label.flatten()[:10]}")

    

