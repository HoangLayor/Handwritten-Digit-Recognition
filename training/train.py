import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
from models.model import Net
from data.dataset import get_data_loader
from training.evaluation import evaluate_model
from utils.utils import save_checkpoint, create_checkpoint_dir

def train_model(epochs=5, batch_size=64, learning_rate=0.001, save_dir='./checkpoints', eval=True):
    # Create the save_dir
    checkpoint = create_checkpoint_dir(save_dir)

    train_loader, valid_loader = get_data_loader(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, and optimizer
    model = Net()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            batch_loss = loss_function(output, label)
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()

        # Evaluate the model in training set
        train_acc = evaluate_model(model, train_loader)
        print(f"-------------")
        print(f"Epoch {epoch+1}/{epochs}:\nTrain - Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

        # Evaluate the model in validation set
        if eval:
            best_valid_acc = 0.0
            with torch.no_grad():
                for image, label in valid_loader:
                    image, label = image.to(device), label.to(device)
                    output = model(image)

                    valid_loss += loss_function(output, label).item()
            valid_acc = evaluate_model(model, valid_loader)
            print(f"Valid - Loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {valid_acc:.4f}")
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if not os.path.exists(f'{save_dir}/checkpoint_{checkpoint}/best accuracy'):
                    os.makedirs(f'{save_dir}/checkpoint_{checkpoint}/best accuracy')
                save_checkpoint(model, optimizer, epoch+1, f'{save_dir}/checkpoint_{checkpoint}/best accuracy/mnist_model_best.pth')
        
    save_checkpoint(model, optimizer, epochs, f'{save_dir}/checkpoint_{checkpoint}/mnist_model_epoch_{epochs}.pth')