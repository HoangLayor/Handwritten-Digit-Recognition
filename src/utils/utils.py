import torch
import os

def create_checkpoint_dir(model_path):
    # Create the save_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    checkpoint = len(os.listdir(model_path)) + 1
    if not os.path.exists(f'{model_path}/checkpoint_{checkpoint}'):
        os.makedirs(f'{model_path}/checkpoint_{checkpoint}')
    return checkpoint

def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']