import torch
from models.model import Net
from data.dataset import get_data_loader

def evaluate_model(model_path, batch_size=64):
    train_loader, valid_loader = get_data_loader(batch_size)
    
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print(f'Accuracy: {100 * correct / total:.2f}%')