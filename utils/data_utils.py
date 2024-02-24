import torch

def generate_dummy_data(size=(1, 3, 224, 224)):
    """ダミーデータを生成する簡単な例"""
    return torch.rand(size)
