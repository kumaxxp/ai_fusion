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
