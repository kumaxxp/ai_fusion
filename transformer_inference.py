import torch
import numpy as np
from models.sensor_fusion_transformer import SensorFusionTransformerWithDecisionLayer

def load_model(model_path):
    # モデルのインスタンス化（適切な引数を設定してください）
    model = SensorFusionTransformerWithDecisionLayer(num_layers=2, nhead=4, feature_size=512, output_size=2)
    # モデルの状態辞書をロード
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 推論モードに設定
    return model

def generate_dummy_data():
    # ダミーデータの生成（ここではランダムデータを生成しています）
    dummy_image = torch.rand((1, 3, 224, 224))  # 例: 画像データ
    dummy_depth = torch.rand((1, 1, 224, 224))  # 例: 深度データ
    dummy_imu = torch.rand((1, 6))  # 例: IMUデータ（加速度 + ジャイロスコープ）
    return dummy_image, dummy_depth, dummy_imu

def inference(model, image, depth, imu):
    # データに対する推論を行う
    with torch.no_grad():  # 勾配計算を無効化
        inputs = (image, depth, imu)  # 入力をタプルにパック
        outputs = model(*inputs)  # アンパックしてモデルに渡す
    return outputs

if __name__ == "__main__":
    # モデルの読み込み
    model_path = './model.pth'
    model = load_model(model_path)
    
    # ダミーデータの生成と推論の実行
    dummy_image, dummy_depth, dummy_imu = generate_dummy_data()
    
    # 推論の実行（すべてのダミーデータを使用）
    outputs = inference(model, dummy_image, dummy_depth, dummy_imu)
    
    # 推論結果の表示
    print(outputs)
