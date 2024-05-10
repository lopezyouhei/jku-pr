import torch
from torch.utils.data import Dataset

def remove_classes_randomly(view1, view2, labels, reduce_factor=1.0):
    unique_labels = labels.unique() # 1000 for imagenet-1k
    num_classes_to_remove = int(len(unique_labels) * (1 - reduce_factor))
    classes_to_remove = torch.randperm(len(unique_labels))[:num_classes_to_remove]
    mask = torch.ones(len(labels), dtype=torch.bool)
    for c in classes_to_remove:
        # if labels == c, mask = False and if labels != c, mask = True
        mask = mask & (labels != c)
    return view1[mask], view2[mask], labels[mask]

class MAEDataset(Dataset):
    def __init__(self, view1_path, view2_path, labels_path, reduce_factor=1.0):

        self.view1 = torch.load(view1_path)
        self.view2 = torch.load(view2_path)
        self.labels = torch.load(labels_path)

        assert self.view1.shape == self.view2.shape, "view1 and view2 must have the same shape"
        assert self.view1.shape[0] == self.labels.shape[0], "data and labels must have the same length"
        assert 0 < reduce_factor <= 1, "reduce factor must be greater than 0 and less than or equal to 1"

        if reduce_factor < 1:
            self.view1, self.view2, self.labels = remove_classes_randomly(self.view1, self.view2, self.labels, reduce_factor)

        self.length = self.view1.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.view1[idx], self.view2[idx], self.labels[idx]