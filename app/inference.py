import torch
from models.model import Net
from utils.preprocess import preprocess_image
from utils.utils import load_checkpoint

model = Net()
model_path = r"checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth"
model, _, _ = load_checkpoint(model_path, model, None)
model.eval()

def predict(image_path):
    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)

    return predicted.item()