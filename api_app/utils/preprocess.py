from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Invert the image to match the training data
        transforms.Lambda(lambda x: 1 - x),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    if type(image) == str:
        image = Image.open(image)

    image = transform(image)

    return image