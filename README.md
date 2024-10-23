# RAG Microservice for Rick and Morty Episode Data

This project demonstrates a Retrieval Augmented Generation (RAG) microservice built using Haystack, FastAPI, Sentence Transformers, and OpenAI's API.  The microservice answers questions about Rick and Morty episodes based on a corpus of episode descriptions.  This project is part of my coursework as a Masters student at the University of Florida, focusing on data engineering and DevOps.

**Author:** Kushagra Sikka (kushgrasikka@gmail.com) - Masters Student, University of Florida


## Functionality

The microservice provides two main endpoints:

- **`/health` (GET):**  A simple health check endpoint to verify the service is running.
- **`/ask` (POST):**  Accepts a JSON payload containing a question (`{"question": "Your question here"}`) and returns a JSON response with the generated answer.


## Setup

1. **Prerequisites:**

   - Docker
   - Python 3.12 (or compatible version specified in `requirements.txt`)
   - An OpenAI API key.

2. **Environment Variables:**

   Create a `.env` file in the project's root directory with the following environment variable:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Data:**

   The project includes a `data/rick_and_morty_episodes` directory containing text files with episode descriptions.  You should not need to modify these files.

4. **Installation:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application (using Docker):**

   The easiest way to run the application is using Docker.  Navigate to the project's root directory and run:

   ```bash
   ./run_app.sh
   ```

   This script builds the Docker image and starts a container, mapping port 8000 to the host machine.  The data and application code are mounted as volumes, allowing changes to the code or data to be reflected in the running container without rebuilding.


## Usage

After running the application, you can test the endpoints using `curl`:

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Ask a Question:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What happened in Rick Potion #9?"}' http://localhost:8000/ask
```


## Technology Stack

- **FastAPI:** A modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **Haystack:** An open-source framework for building NLP applications.  Used here for embedding generation, retrieval, and prompt building.
- **Sentence Transformers:** Used for generating embeddings of both the documents and user queries.
- **OpenAI API:** The large language model used for question answering.
- **Docker:** Used for containerization and deployment.


## Future Improvements

- **More Robust Error Handling:** Implement more specific error handling and logging to improve the application's resilience.
- **Input Validation and Sanitization:** Add more robust input validation to prevent injection attacks and handle unexpected input.
- **Advanced Retrieval Techniques:** Explore more sophisticated retrieval methods beyond simple in-memory embedding retrieval.
- **Scalability:** Implement strategies for scaling the application to handle a higher volume of requests.
- **Unit and Integration Tests:** Add comprehensive testing to improve the code quality and maintainability.


## Contact

For any questions or inquiries, feel free to contact me at kushagrasikka@gmail.com.
