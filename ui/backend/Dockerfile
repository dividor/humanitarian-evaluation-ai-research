FROM python:3.11-slim

WORKDIR /app

# Install pipeline dependencies first
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install UI backend dependencies
COPY ui/backend/requirements.txt /app/ui/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/ui/backend/requirements.txt

# Copy project files
COPY . /app/

# Set python path
ENV PYTHONPATH=/app

WORKDIR /app/ui/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
