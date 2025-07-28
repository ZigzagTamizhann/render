from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")  # Place this model file in same folder
person_class_id = 0

@app.post("/detect/")
async def detect_person(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if class_id == person_class_id and confidence >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": "person",
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })

    return JSONResponse(content={"detections": detections})
