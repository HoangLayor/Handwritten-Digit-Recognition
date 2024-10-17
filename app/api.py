from fastapi import FastAPI, UploadFile, File
from inference import predict
from PIL import Image
from torchvision import transforms
from utils.utils import load_checkpoint
from utils.preprocess import preprocess_image
from models.model import Net
import torch
import io

app = FastAPI()

# API để upload file ảnh
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Đọc nội dung file
    image_data = await file.read()

    # Mở ảnh bằng Pillow
    image = Image.open(io.BytesIO(image_data))

    # Bạn có thể xử lý ảnh ở đây, ví dụ lấy kích thước ảnh
    width, height = image.size

    return {"image": image}

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        prediction = predict(image)
        return {'prediction': prediction}
    except Exception as e:
        return {'error': str(e)}
    
    # Giả sử bạn đã load model MNIST đã train xong
# Ví dụ, bạn lưu model vào file model_mnist.pth sau khi train
model = Net()
model, _, _ = load_checkpoint(r"checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth", model, None)
# model.eval()

# Chuyển đổi ảnh về dạng mà model yêu cầu
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh sang grayscale
    transforms.Resize((28, 28)),  # Resize về 28x28 giống MNIST
    transforms.ToTensor(),  # Chuyển ảnh sang Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize giống như cách bạn đã train
])

# API để upload file ảnh và nhận diện
@app.post("/predict_123/")
async def predict(file: UploadFile = File(...)):
    # Đọc file ảnh
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    img_tensor = preprocess_image(image)
    pred = predict(img_tensor)
    return {pred}
    