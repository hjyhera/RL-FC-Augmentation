import numpy as np
import os
import torch


class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, data_dir):
        with open(txt_file, 'r') as f:
            self.file_list = f.read().splitlines()
        self.data_dir = data_dir
        self.mdd_count = sum(1 for file_name in self.file_list if file_name.split('-')[1] == "1")
        self.nc_count = len(self.file_list) - self.mdd_count

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        n = data.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        fc_matrix = data[triu_idx] 
        label = 1 if self.file_list[idx].split('-')[1] == "1" else 0

        fc_data = torch.tensor(fc_matrix, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return fc_data, label


if __name__ == "__main__":
    data_dir= r'/home/user/RL-FC-Augmentation/MS_MDD_Harvard_FC_fisher-20250107T100946Z-001/MS_MDD_Harvard_FC_fisher'
    txt_file = r'/home/user/RL-FC-Augmentation/MDD_Harvard_site20-20250107T050634Z-001/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_1.txt'
    dataset = BrainDataset(txt_file, data_dir)
    print(f"Total number of samples: {len(dataset)}")
    fc_data, label = dataset[0]
    print("\n=== First sample from BrainDataset ===")
    print(f"fc_data shape: {fc_data.shape}, dtype: {fc_data.dtype}, sample values: {fc_data[:10]}")
    print(f"label shape: {label.shape}, dtype: {label.dtype}, value(s): {label.flatten()[:10]}")

    