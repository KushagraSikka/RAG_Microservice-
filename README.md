# RAG Microservice with Docker and Python

A Retrieval Augmented Generation (RAG) microservice that demonstrates the integration of document retrieval with OpenAI's language models. Created by Kushagra Sikka, this example uses Rick and Morty episode descriptions as the knowledge base.

## What is RAG?

RAG (Retrieval Augmented Generation) is a hybrid modeling technique that combines:

- Retrieval-based systems
- Generative models (Large Language Models)

Key advantages:

- Combines accuracy of retrieval systems with the flexibility of generative models
- Uses dense embeddings for document retrieval
- Allows for knowledge-based generation using specific document corpus

## Prerequisites

- Docker
- Python 3.12
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kushagrasikka/rag-microservice-python.git
cd rag-microservice-python
```

2. Create and configure your environment file:

```bash
# Create .env file
touch .env

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

3. Install dependencies (if running locally):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   └── rick_and_morty_episodes/
│       ├── season1.txt
│       ├── season2.txt
│       └── ...
├── rag_microservice/
│   └── app.py
├── Dockerfile
├── requirements.txt
└── run_app.sh
```

## Dependencies

```
fastapi
haystack
python-dotenv
sentence-transformers
scikit-learn
numpy
pydantic
uvicorn
```

## Running the Service

1. Using the shell script:

```bash
./run_app.sh
```

This script will:

- Build the Docker image
- Run the container
- Mount necessary volumes
- Set up port forwarding
- Clean up the container when stopped

2. Manual Docker commands:

```bash
# Build the image
docker build -t kushagra-rag-microservice:latest .

# Run the container
docker run -d --rm \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/rag_microservice:/root/app \
    --env-file .env \
    kushagra-rag-microservice:latest
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Ask Questions

```bash
# Example by Kushagra Sikka
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a Meeseeks box?"}'
```

## Docker Management

View running containers:

```bash
docker ps
```

View logs:

```bash
docker logs <container-id>
```

Stop the service:

```bash
docker stop <container-id>
```

Clean up resources:

```bash
# Remove stopped containers
docker container prune

# Remove images
docker image rm kushagra-rag-microservice:latest
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
- `GENERATOR_MODEL`: OpenAI model for text generation (default: "gpt-4-mini")
- `DATA_PATH`: Path to document corpus

## Architecture

The microservice uses:

- FastAPI for the web server
- Haystack for the RAG pipeline
- SentenceTransformers for document embeddings
- OpenAI's API for text generation
- Docker for containerization

The RAG pipeline consists of:

1. Document store creation
2. Document embedding
3. Retriever setup
4. Prompt template creation
5. Generator configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Kushagra Sikka**  
Master's in Computer Science  
University of Florida  
Class of 2024

For questions or collaborations, please reach out to me on [GitHub](https://github.com/kushagrasikka).
