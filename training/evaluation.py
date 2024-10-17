import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model import Net
from data.dataset import get_data_loader
from utils.utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    return correct / total * 100

def eval(model_path, batch_size=64):
    train_loader, valid_loader = get_data_loader(batch_size)
    
    model = Net()
    model, _, epoch = load_checkpoint(model_path, model, None)
    print(f'Model loaded from epoch {epoch+1}')
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in valid_loader:
            image, label = image.to(device), label.to(device)
                                                      
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
    print(f'Accuracy: {100 * correct / total:.2f}%')