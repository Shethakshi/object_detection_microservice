FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (better cache usage)
COPY req.txt .

# Install PyTorch separately from official wheels (CPU only)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r req.txt

# Copy project files
COPY . .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
