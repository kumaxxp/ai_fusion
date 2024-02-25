import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SensorFusionTransformerWithDecisionLayer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, output_size):
        super(SensorFusionTransformerWithDecisionLayer, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)  # Transformer層の出力を変換するための層
        
        # 意思決定層：ステアリングとスロットルの出力を生成する
        self.decision_layer = nn.Linear(feature_size, output_size)  # 出力サイズはステアリングとスロットルの2つ

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        # シーケンスの次元に沿って平均を計算し、バッチサイズ x 出力サイズのテンソルを得る
        output = torch.mean(output, dim=1)
        decision_output = self.decision_layer(output)  # 意思決定層を通して最終的な出力を得る
        return decision_output

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
