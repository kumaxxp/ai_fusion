import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.sensor_fusion_transformer import SensorFusionTransformerWithDecisionLayer


# モデルの初期化
feature_size = 512  # 特徴量のサイズ
num_layers = 6  # Transformerのレイヤー数
nhead = 8  # マルチヘッドアテンションのヘッド数
output_size = 2  # ステアリングとスロットルのための出力サイズ
model = SensorFusionTransformerWithDecisionLayer(feature_size, num_layers, nhead, output_size)

# ダミーデータの生成
batch_size = 10
seq_length = 32
dummy_input = torch.rand(batch_size, seq_length, feature_size)  # (バッチサイズ, シーケンス長, 特徴量のサイズ)

# モデルを通してダミーデータを実行
output = model(dummy_input)
print("Output shape:", output.shape)