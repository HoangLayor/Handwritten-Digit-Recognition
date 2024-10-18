from fastapi import FastAPI, UploadFile, File
# from inference import predict
from PIL import Image
from torchvision import transforms
from utils.utils import load_checkpoint
from utils.preprocess import preprocess_image
from models.model import Net
import torch
import io
import asyncio

app = FastAPI()

model = Net()
model, _, _ = load_checkpoint(r'checkpoints\checkpoint_1\best accuracy\mnist_model_best.pth', model, None)

# API để upload file ảnh
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Đọc nội dung file
    image_data = await file.read()

    # Mở ảnh bằng Pillow
    image = Image.open(io.BytesIO(image_data))

    return {"image": image}
    
# API để upload file ảnh và nhận diện
@app.post("/predict_123/")
async def predict(file: UploadFile = File(...)):
    # Đọc file ảnh
    image = Image.open(io.BytesIO(await file.read()))

    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        predicted = torch.argmax(output, dim=1)

    return {"prediction": predicted.item()}
    