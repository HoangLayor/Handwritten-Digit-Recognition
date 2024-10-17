import torch
from models.model import Net
from utils.preprocess import preprocess_image
from utils.utils import load_checkpoint

model = Net()
model = Net()
model, _, _ = load_checkpoint(r"checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth", model, None)
# model.eval()

def predict(image):
    if type(image) == str:
        image = preprocess_image(image)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted = torch.argmax(output, dim=1)

    return predicted.item()

# # Test
# image_path = r"test_images\01.png"
# print(predict(image_path))