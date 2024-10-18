import torch
from utils.preprocess import transform_image

def predict(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted = torch.argmax(output, dim=1)

    return predicted.item()
