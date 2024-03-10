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

