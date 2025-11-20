FROM python:3.11-slim

# Install system dependencies
# poppler-utils: for pdf2image (if needed)
# tesseract-ocr: for OCR
# libgl1: for opencv/rapidocr
# git: for installing git dependencies
# chromium: for Selenium (download.py)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    git \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome/Chromium binary location for Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set python path to include current directory
ENV PYTHONPATH=/app

CMD ["python", "pipeline/download.py"]
