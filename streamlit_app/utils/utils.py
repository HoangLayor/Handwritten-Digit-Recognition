import torch
from models.model import Net


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def get_model():
    model_path = r'src\checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth'

    model = Net()
    model, _, _ = load_checkpoint(model_path, model, None)
    model.eval()
    return model