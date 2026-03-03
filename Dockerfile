FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer — uses headless OpenCV for smaller image)
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt gunicorn==21.2.0

# Copy app code
COPY attendance_system/ .

# Download ONNX models (not stored in git — too large)
RUN mkdir -p models && \
    wget -q -O models/face_detection_yunet_2023mar.onnx \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" && \
    wget -q -O models/face_recognition_sface_2021dec.onnx \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" && \
    echo "Models downloaded:" && ls -lh models/*.onnx

# Create upload directories
RUN mkdir -p static/uploads/dataset logs

# Expose port
EXPOSE 10000

# Run with gunicorn
CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120", "--worker-class", "gthread", "--threads", "4"]
