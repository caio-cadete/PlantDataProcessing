# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for better memory management
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code first
COPY . .

# Set Python path to include the current directory
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
# Memory optimization for Python
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

# Train the models if they haven't been trained yet
RUN if [ ! -d "/app/trained-models" ] || [ -z "$(ls -A /app/trained-models)" ]; then \
    python -m models.train_model; \
fi

# Expose port 8080
EXPOSE 8080

# Run the API using uvicorn with memory optimization
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--access-log"]