from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import uuid

app = FastAPI()
model = YOLO("yolov8n.pt")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Unique ID for saving files
    uid = str(uuid.uuid4())[:8]

    # Run prediction
    results = model.predict(img, conf=0.25, save=False, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for bbox, conf, cls in zip(xyxy, confs, classes):
                detections.append({
                    "bbox": [float(x) for x in bbox],
                    "confidence": float(conf),
                    "class": int(cls)
                })

        # Save image with boxes
        save_path = os.path.join(OUTPUT_DIR, f"{uid}_pred.jpg")
        result.save(filename=save_path)

    # Save detections JSON
    json_path = os.path.join(OUTPUT_DIR, f"{uid}_pred.json")
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=4)

    return {
        "detections": detections,
        "image_path": save_path,
        "json_path": json_path
    }
