import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
from torchvision import transforms
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.dataloader import transform
from models.model import model

app = FastAPI()

model.load_state_dict(torch.load("outputs/model_weights.pth"))
model.eval()

class ImageInput(BaseModel):
    image_path: str

def preprocess_image(image_path):
    image = Image.open(image_path)
    # Preprocess image
    image = transform(image).unsqueeze(0)
    return image

@app.get("/")
async def read_root():
    return {"message": "Watermark classifier"}

@app.post("/predict")
async def predict(image_data: ImageInput):
    try:
        image = preprocess_image(image_data.image_path)
        
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
