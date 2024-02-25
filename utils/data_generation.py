import torch
import numpy as np
import random

def set_seed(seed_value=42):
    """全ての乱数生成器のシードを設定する"""
    torch.manual_seed(seed_value) # PyTorch用
    torch.cuda.manual_seed_all(seed_value) # CUDA環境でのPyTorch用
    np.random.seed(seed_value) # NumPy用
    random.seed(seed_value) # Python標準ライブラリのrandom用
    torch.backends.cudnn.deterministic = True # CuDNNの動作を決定論的に
    torch.backends.cudnn.benchmark = False # ネットワークが固定の場合のパフォーマンス向上をFalseに

def generate_dummy_data(size=(1, 3, 224, 224)):
    """ダミーデータを生成する簡単な例"""
    return torch.rand(size)


def generate_dummy_monocular_image(width=640, height=480, channels=3):
    """単眼カメラデータを生成する"""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

def generate_dummy_depth_data(width=640, height=480):
    """深度センサーデータを生成する"""
    # 0.5mから4mまでのランダムな深度値
    return np.random.uniform(0.5, 4.0, (height, width))

def generate_dummy_imu_data():
    """IMUデータを生成する"""
    # 加速度: -10から10 m/s^2
    accel = np.random.uniform(-10, 10, 3)
    # 角速度: -500から500 °/s
    gyro = np.random.uniform(-500, 500, 3)
    return accel, gyro

# ダミーデータを生成
dummy_monocular_image = generate_dummy_monocular_image()
dummy_depth_data = generate_dummy_depth_data()
dummy_accel, dummy_gyro = generate_dummy_imu_data()

print(dummy_monocular_image.shape, dummy_depth_data.shape, dummy_accel, dummy_gyro)


