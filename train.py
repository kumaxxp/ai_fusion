import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets.custom_dataset import CustomDataset  # カスタムデータセットのインポート
from models.sensor_fusion_transformer import SensorFusionTransformerWithDecisionLayer  # 仮定のモデル

import matplotlib.pyplot as plt

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()  # モデルを訓練モードに設定
        running_loss = 0.0
        running_acc = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 勾配を0で初期化
            outputs = model(inputs)  # モデルによる予測
            loss = criterion(outputs, targets)  # 損失の計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータの更新

            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, targets)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 検証データセットでの評価
        model.eval()  # モデルを評価モードに設定
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():  # 勾配計算を無効化
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item() * inputs.size(0)
                running_val_acc += accuracy(outputs, targets)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = running_val_acc / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

    # 学習曲線のプロット
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
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
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader, criterion)

