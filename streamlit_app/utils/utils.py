import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.model import Net


def load_checkpoint(path, model, optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def get_model(model_path=r'src\checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth'):
    model = Net()
    model, _, _ = load_checkpoint(model_path, model, None)
    model.eval()
    return model
