import torch
import torch.optim as optim
import torch.nn as nn
from models.model import Net
from data.dataset import get_data_loader
from training.evaluation import evaluate_model
import os

def train_model(epochs=5, batch_size=64, learning_rate=0.001, model_path='./checkpoints', eval=False):
    train_loader, valid_loader = get_data_loader(batch_size)
    train_N, valid_N = len(train_loader), len(valid_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, and optimizer
    model = Net()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss, valid_loss = 0.0, 0.0
        train_acc, valid_acc = 0.0, 0.0

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            batch_loss = loss_function(output, label)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
        train_acc = evaluate_model(model, train_loader)
        print(f"Epoch {epoch+1}/{epochs}:\nTrain - Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

        if eval:
            with torch.no_grad():
                for image, label in valid_loader:
                    image, label = image.to(device), label.to(device)
                    output = model(image)

                    valid_loss += loss_function(output, label).item()
            valid_acc = evaluate_model(model, valid_loader)
            print(f"Valid - Loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {valid_acc:.4f}")
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    torch.save(model.state_dict(), f'{model_path}/mnist_model_epoch_{epochs}.pth')