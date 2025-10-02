FROM python:3.9-slim

# Install system dependencies (including libgl1 for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Add /app and yolov3 to Python path so "from utils" works
ENV PYTHONPATH=/app:/app/yolov3

# Copy requirements first (better cache usage)
COPY req.txt .
COPY yolov3/requirements.txt ./yolov3_requirements.txt

# Install PyTorch (CPU version only, you can switch to CUDA if needed)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install yolov3 + project dependencies
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r req.txt \
    && pip install --no-cache-dir --timeout=120 --retries=5 -r yolov3_requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 5000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
