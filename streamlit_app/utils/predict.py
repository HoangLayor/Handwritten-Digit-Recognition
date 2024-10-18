import torch

def predict(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted = torch.argmax(output, dim=1)

    return predicted.item()
