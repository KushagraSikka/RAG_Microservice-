# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_microservice/ /app/
COPY data/ /app/data/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CORPUS_DOCUMENTS_PATH=/app/data/professional_info
ENV CORPUS_DOCUMENTS_FILE_EXT=txt
ENV TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV GENERATOR_MODEL=google/flan-t5-large

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]