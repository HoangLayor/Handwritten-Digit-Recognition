from fastapi import FastAPI, UploadFile, File
from inference import predict
from PIL import Image
import io

app = FastAPI()

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        prediction = predict(image)
        return {'prediction': prediction}
    except Exception as e:
        return {'error': str(e)}