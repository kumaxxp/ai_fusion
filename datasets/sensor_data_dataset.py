import torch
from torch.utils.data import Dataset

class SensorDataDataset(Dataset):
    def __init__(self, size=100, seq_length=32, feature_size=512, output_size=2):
        self.data = torch.randn(size, seq_length, feature_size)
        self.labels = torch.rand(size, output_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
