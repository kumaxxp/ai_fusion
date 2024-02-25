import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, size=1000, seq_length=32, feature_size=512):
        # データセットのサイズ、シーケンスの長さ、特徴量のサイズを指定
        self.size = size
        self.seq_length = seq_length
        self.feature_size = feature_size
        
        # ダミーデータの生成
        self.data = torch.randn(size, seq_length, feature_size)
        # ダミーのターゲット（ステアリングとスロットルの値）を生成
        self.targets = torch.rand(size, 2)  # 2はステアリングとスロットルの値を示す

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# データセットのインスタンス化
dataset = CustomDataset()

# データローダーの設定
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# データローダーの使用例
for data, targets in data_loader:
    print(data.shape, targets.shape)
    break  # デモのため、最初のバッチのみ表示
