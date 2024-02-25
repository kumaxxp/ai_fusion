import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor

import sys
import os
sys.path.append(os.path.abspath('/mnt/c/work/ai_fusion'))

# feature_extractor.py
from utils.data_generation import generate_dummy_monocular_image


# ダミーデータの前処理（正規化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 単眼カメラデータをTensorに変換し、正規化
dummy_monocular_image = generate_dummy_monocular_image()  # 関数を呼び出し
dummy_monocular_tensor = to_tensor(dummy_monocular_image)  # NumPy配列をTensorに変換
dummy_monocular_tensor = dummy_monocular_tensor.unsqueeze(0)  # ダミーデータに適用


# CNNモデルの定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 120 * 160, 128),  # 画像サイズに注意
            nn.ReLU(),
            nn.Linear(128, 10)  # 仮の出力層サイズ
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# モデルのインスタンス化
model = SimpleCNN()

# ダミーデータをモデルに入力し、特徴を抽出
features = model(dummy_monocular_tensor)
print("Extracted features shape:", features.shape)
