#!/bin/bash
docker build . -t kushagrasikka/rag-microservice:latest
docker run -d --rm \
    -p 8000:8000 \
    --env-file .env \
    -v ./data:/data \
    -v ./rag_microservice:/app \
    kushagrasikka/rag-microservice:latest
