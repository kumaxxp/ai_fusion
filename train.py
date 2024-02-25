import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets.custom_dataset import CustomDataset  # カスタムデータセットのインポート
from models.sensor_fusion_transformer import SensorFusionTransformerWithDecisionLayer  # 仮定のモデル

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # モデルを訓練モードに設定
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 勾配を0で初期化
            outputs = model(inputs)  # モデルによる予測
            loss = criterion(outputs, targets)  # 損失の計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータの更新

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, test_loader, criterion):
    model.eval()  # モデルを評価モードに設定
    running_loss = 0.0
    with torch.no_grad():  # 勾配計算を無効化
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    final_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {final_loss:.4f}')


# ダミーデータでの動作確認
train_dataset = CustomDataset(size=1000, seq_length=32, feature_size=512)
test_dataset = CustomDataset(size=200, seq_length=32, feature_size=512)

# DataLoaderのインスタンス化
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# モデル、損失関数、最適化アルゴリズムの設定
model = SensorFusionTransformerWithDecisionLayer(num_layers=2, nhead=4, feature_size=512, output_size=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoaderの使用例
for data, targets in train_loader:
    print("Train batch - data shape:", data.shape, "targets shape:", targets.shape)
    break  # デモのため、最初のバッチのみ表示

for data, targets in test_loader:
    print("Test batch - data shape:", data.shape, "targets shape:", targets.shape)
    break  # デモのため、最初のバッチのみ表示

# 訓練と評価の実行
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader, criterion)

