# Object Detection Microservice 

A lightweight object detection microservice built with **FastAPI** and **YOLOv8** (Ultralytics), designed as two modular services:

- **UI Backend** – Handles image uploads from users.
- **AI Backend** – Performs object detection and returns JSON results + output images.

Both services are **containerized using Docker** and orchestrated with Docker Compose for easy deployment.

---

## Architecture

User → UI Backend (/upload) → AI Backend (/detect) → YOLOv8 Model → JSON + Output Images

yaml


- UI Backend forwards requests and responses.
- AI Backend runs the detection model.
- Separation allows modularity and easier scaling.

---

## Features

- Run YOLOv8 inference on uploaded images.
- Returns bounding boxes, confidence scores, and class IDs.
- Saves prediction images and JSON outputs.
- Lightweight and CPU-friendly model.
- Fully containerized with Docker.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Shethakshi/object_detection_microservice.git
cd object_detection_microservice

### 2. Build and run with Docker Compose
```
docker-compose up --build

### 3. Test with cURL
```
curl -X POST "http://localhost:5000/upload/" -F "file=@path_to_your_image/image.jp
