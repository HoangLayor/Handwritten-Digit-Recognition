import torch
import torch.optim as optim
import torch.nn as nn
from models.model import Net
from data.dataset import get_data_loader
from training.evaluation import evaluate_model
import os

def train_model(epochs=5, batch_size=64, learning_rate=0.001, model_path='./checkpoints', eval=False):
    train_loader, valid_loader = get_data_loader(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, and optimizer
    model = Net()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        model.train()
        loss = 0.0
        accuracy = 0.0

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            batch_loss = loss_function(output, label)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss / len(train_loader):.4f},", end = " ")

        if eval:
            correct = 0
            total = 0

            with torch.no_grad():
                for image, label in valid_loader:
                    image, label = image.to(device), label.to(device)

                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                
            print(f'Accuracy: {100 * correct / total:.2f}%')
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    torch.save(model.state_dict(), f'{model_path}/mnist_model_epoch_{epochs}.pth')