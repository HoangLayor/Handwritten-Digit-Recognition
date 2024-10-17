import torch
from models.model import Net
from utils.preprocess import preprocess_image

model = Net()
model_path = r"checkpoints\mnist_model_epoch_10.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(image_path):
    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)

    return predicted.item()