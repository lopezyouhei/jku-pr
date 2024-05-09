import torch
from torch.utils.data import Dataset

class MAEDataset(Dataset):
    def __init__(self, view1_path, view2_path):

        self.data1 = torch.load(view1_path)
        self.data2 = torch.load(view2_path)
        assert self.data1.shape == self.data2.shape, "view1 and view2 must have the same shape"
        self.length = self.data1.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]