import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SensorFusionTransformerWithDecisionLayer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, output_size):
        super(SensorFusionTransformerWithDecisionLayer, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)
        
        # 各センサーデータを処理するためのダミーモデル
        self.image_model = nn.Linear(feature_size, feature_size)
        self.depth_model = nn.Linear(feature_size, feature_size)
        self.imu_model = nn.Linear(feature_size, feature_size)
        
        self.decision_layer = nn.Linear(feature_size, output_size)

    def forward(self, image_data, depth_data, imu_data):
        # 各センサーデータを個別に処理
        image_features = self.image_model(image_data)
        depth_features = self.depth_model(depth_data)
        imu_features = self.imu_model(imu_data)
        
        # 特徴を結合
        combined_features = torch.cat((image_features, depth_features, imu_features), dim=1)
        
        output = self.transformer_encoder(combined_features)
        output = self.decoder(output)
        output = torch.mean(output, dim=1)
        decision_output = self.decision_layer(output)
        return decision_output