
# RAG Microservice - Professional Profile Assistant

The **RAG Microservice** is an intelligent assistant designed to provide detailed, structured, and professional responses about **Kushagra Sikka's professional profile**. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate well-formatted, concise answers tailored for recruiters and collaborators.

## **Use Case**

This system is tailored to **Kushagra's recruiters** and professional network, allowing them to:
- Fetch **structured professional summaries**.
- Retrieve details about **technical skills** with usage examples.
- Explore **academic achievements**, **work experience**, and **projects**.
- Gain insights into **recent contributions** and **research focus**.
- Access contact information and links to **professional profiles**.

The assistant ensures:
- **Accurate and verified information** retrieval from curated documents.
- **Well-formatted responses** using bullet points for readability.
- A **professional tone** to enhance user experience and utility.

### Example Questions
- **"Who is Kushagra Sikka?"**  
  Provides a professional overview, recent impact, and technical expertise.
  
- **"What are Kushagra's technical skills?"**  
  Details programming languages, cloud platforms, and tools with usage examples.
  
- **"Tell me about Kushagra's achievements."**  
  Highlights teaching impact, technical accomplishments, and academic recognition.

---

## **Architecture**

The system integrates:
- **Backend**: FastAPI + Haystack for RAG implementation.
- **Frontend**: React + TailwindCSS for a user-friendly interface.
- **Deployment**: Docker, Jenkins, and AWS EC2 for production readiness.

### **Backend Workflow**
1. **Document Store**: Stores structured professional data (e.g., skills, projects, achievements).
2. **Retriever**: Fetches relevant documents based on the query.
3. **Prompt Builder**: Constructs dynamic prompts tailored to the query type.
4. **Generator**: Uses a text generation model to produce well-formatted, bullet-pointed responses.

### **Frontend Features**
- Interactive UI for querying Kushagra's profile.
- Real-time response rendering with React.
- Mobile-responsive design using TailwindCSS.

---

## **Local Development**

Follow these steps to set up the project locally:

### **1. Clone the Repository**
```bash
git clone https://github.com/YourUsername/RAG_Microservice.git
cd RAG_Microservice
```

### **2. Set Up the Backend**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Set Up the Frontend**
```bash
cd rag-frontend
npm install
```

### **4. Create Environment Variables**
Copy the example `.env` file and update it with your configuration:
```bash
cp .env.example .env
```

### **5. Run the Application Locally**

#### **Backend**
```bash
uvicorn rag_microservice.app:app --reload
```

#### **Frontend**
In a new terminal:
```bash
cd rag-frontend
npm start
```

---

## **Docker Deployment**

To deploy the application using Docker:
```bash
docker-compose up -d
```

---

## **Production Deployment**

1. Configure an AWS EC2 instance with the necessary environment.
2. Set up Docker and Jenkins for continuous integration and deployment.
3. Refer to the `deployment/README.md` file for step-by-step instructions.

---

## **Project Structure**

```
RAG_Microservice/
├── rag_microservice/         # Backend code
│   └── app.py                # Main FastAPI application
├── rag-frontend/             # React frontend
├── data/                     # Data directory
├── docker-compose.yml        # Docker compose configuration
├── Jenkinsfile               # CI/CD pipeline
└── deployment/               # Deployment documentation and scripts
```

---

## **Environment Variables**

### **Backend**
| Variable               | Description                                       |
|------------------------|---------------------------------------------------|
| `CORPUS_DOCUMENTS_PATH` | Path to the document corpus (e.g., professional details). |
| `TEXT_EMBEDDING_MODEL`  | Model for text embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`). |
| `GENERATOR_MODEL`       | Model for text generation (e.g., `google/flan-t5-large`). |

### **Frontend**
| Variable                | Description                                   |
|-------------------------|-----------------------------------------------|
| `REACT_APP_API_URL`     | Backend API URL (e.g., `http://localhost:8000`). |

---

## **Features**

### **Backend**
- Implements RAG using Haystack components:
  - **Document Splitter**: Chunks documents for better retrieval.
  - **Retriever**: Retrieves relevant documents based on queries.
  - **Prompt Builder**: Constructs dynamic prompts for generation.
  - **Generator**: Produces well-structured responses.
- Pre-processed corpus includes:
  - Professional profile, achievements, and technical skills.
  - Work experience and key projects.
  - Contact information.

### **Frontend**
- Query interface with a clean, professional design.
- Supports real-time query results.
- Mobile-friendly and intuitive layout.

---

## **How It Works**

1. **Ask a Question**: The user asks a question, such as "What are Kushagra's skills?".
2. **Document Retrieval**: The system retrieves relevant sections from the document store.
3. **Dynamic Prompt Creation**: A prompt is generated based on the question type.
4. **Answer Generation**: The model generates a structured, bullet-pointed response.
5. **Response Display**: The frontend displays the response in a user-friendly format.

---

## **Use Case for Recruiters**

### **Objective**
To provide a **quick, reliable, and structured way** for recruiters to:
- Understand Kushagra's professional profile.
- Explore technical expertise and projects.
- Gain insights into recent achievements and research focus.

### **Advantages**
- **Time-Saving**: Direct answers without the need to parse lengthy resumes.
- **Structured Format**: Responses are concise and categorized for better readability.
- **Accurate Data**: Fetches verified information only.

### **Example Queries**
- **"Who is Kushagra Sikka?"**
  - Learn about Kushagra's current roles, educational background, and recent impact.
- **"What are Kushagra's achievements?"**
  - Understand his teaching impact, technical accomplishments, and academic recognition.
- **"Tell me about Kushagra's skills."**
  - Explore his technical expertise with usage examples.

---

## **Contributing**

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request.

---

## **Contact**

For further queries or contributions, reach out to **Kushagra Sikka**:
- **Email**: kushagrasikka@gmail.com
- **Portfolio**: [kushagrasikka.com](https://www.kushagrasikka.com)
- **GitHub**: [github.com/KushagraSikka](https://github.com/KushagraSikka)
- **LinkedIn**: [linkedin.com/in/kushagrasikka](https://linkedin.com/in/kushagrasikka)
