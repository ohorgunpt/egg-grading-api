from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# PENTING: Agar website di InfinityFree bisa memanggil Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "API Deteksi Telur Aktif"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    results = model.predict(source=img, conf=0.5)
    
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": model.names[class_id],
                "confidence": float(box.conf[0])
            })
            
    return {"predictions": detections}
