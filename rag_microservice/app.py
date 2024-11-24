from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack import Pipeline
from pathlib import Path
import os
import time
import logging
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
CORPUS_DOCUMENTS_PATH = os.getenv(
    "CORPUS_DOCUMENTS_PATH", "./data/rick_and_morty_episodes")
CORPUS_DOCUMENTS_FILE_EXT = os.getenv("CORPUS_DOCUMENTS_FILE_EXT", "txt")
TEXT_EMBEDDING_MODEL = os.getenv(
    "TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google/flan-t5-small")

# Constants
MAX_CHUNK_SIZE = 200  # Maximum number of words per chunk
MAX_CHUNKS_TO_INCLUDE = 3  # Maximum number of chunks to include in context
OVERLAP_SIZE = 50  # Number of overlapping words between chunks

# FastAPI app initialization
app = FastAPI(title="RAG Microservice",
              description="Retrieval-Augmented Generation system for Kushagra's personal Assitant",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models


class QuestionRequest(BaseModel):
    question: str


class ProcessingMetrics(BaseModel):
    timestamp: float
    tokens_processed: int
    processing_time: float
    step: str


class Answer(BaseModel):
    answer: str
    metrics: List[ProcessingMetrics]


def create_document_store() -> InMemoryDocumentStore:
    """Initialize the document store"""
    logger.info("Creating document store")
    return InMemoryDocumentStore()


def create_document_splitter() -> DocumentSplitter:
    """Create a document splitter with specified configuration"""
    logger.info(
        f"Creating document splitter (chunk_size={MAX_CHUNK_SIZE}, overlap={OVERLAP_SIZE})")
    return DocumentSplitter(
        split_by="word",
        split_length=MAX_CHUNK_SIZE,
        split_overlap=OVERLAP_SIZE
    )


def create_document_embedder() -> SentenceTransformersDocumentEmbedder:
    """Create document embedder with specified model"""
    logger.info(
        f"Creating document embedder with model {TEXT_EMBEDDING_MODEL}")
    return SentenceTransformersDocumentEmbedder(model=TEXT_EMBEDDING_MODEL)


def create_text_embedder() -> SentenceTransformersTextEmbedder:
    """Create text embedder with specified model"""
    logger.info(f"Creating text embedder with model {TEXT_EMBEDDING_MODEL}")
    return SentenceTransformersTextEmbedder(model=TEXT_EMBEDDING_MODEL)


def create_retriever(document_store: InMemoryDocumentStore) -> InMemoryEmbeddingRetriever:
    """Create retriever with specified document store"""
    logger.info(f"Creating retriever (max_chunks={MAX_CHUNKS_TO_INCLUDE})")
    return InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=MAX_CHUNKS_TO_INCLUDE
    )


def create_prompt_builder() -> PromptBuilder:
    """Create dynamic prompt builder with question-specific formatting"""
    template = """
Using the information provided below, respond to the question about Kushagra Sikka.

Verified Information:
{%- for document in documents %}
{{ document.content }}
{%- endfor %}

Question: {{ question }}

Instructions:
1. Only include information from the provided context.
2. Provide a concise answer using bullet points under relevant subheadings.
3. Do not write long sentences; keep each bullet point brief.
4. Use clear and direct language.
5. Maintain a professional tone.

Answer:
"""
    return PromptBuilder(template=template)


def get_question_specific_prompt(question: str) -> str:
    """Return specific prompt template based on question type"""
    question_lower = question.lower()

    if "who" in question_lower or "about" in question_lower:
        return """
Please provide a comprehensive introduction of Kushagra Sikka using the following format:

**Professional Overview**
- Current role and location
- Primary responsibilities
- Educational background

**Recent Impact**
- Key achievements with metrics
- Notable contributions
- Research focus

**Technical Expertise**
- Main technical skills
- Cloud and infrastructure experience
- Development tools and where they've been applied

**Contact Information**
- Professional email
- Location
- Professional profiles
"""

    elif "skill" in question_lower or "expertise" in question_lower:
        return """
Please detail Kushagra Sikka's technical skills using the following format, and include where he has used each technology:

**Programming & Development**
- Programming languages (e.g., Python: used in data analysis projects)
- Development tools (e.g., Git: used for version control in all projects)
- Frameworks (e.g., TensorFlow: used in machine learning models)

**Cloud & Infrastructure**
- Cloud platforms (e.g., AWS EC2: deployed web applications)
- DevOps tools (e.g., Jenkins: automated CI/CD pipelines)
- Infrastructure management (e.g., Terraform: managed cloud resources)

**Data & AI**
- ML/AI technologies (e.g., PyTorch: developed deep learning models)
- Database systems (e.g., PostgreSQL: managed relational databases)
- Data engineering tools (e.g., Apache Spark: processed large datasets)
"""

    # Default template for other questions
    return """
Please provide relevant information about Kushagra Sikka using this structure:

**Main Points**
- Current role and relevance
- Key metrics and achievements
- Specific examples

**Additional Details**
- Supporting information
- Relevant experience
- Technical context

**Contact**
- Professional email
- Location
"""


def format_achievements_response(answer: str) -> str:
    """Specifically format achievement-related responses"""
    achievements = {
        "teaching": [
            "Instructing 60+ students in assembly mechanics",
            "Reducing administrative time by 50% through automation",
            "Enhancing student engagement by 25%"
        ],
        "professional": [
            "Increasing sales by 7.3% through time series analysis",
            "Reducing inventory costs by 15%",
            "Improving database performance by 20%",
            "Reducing deployment times by 40%"
        ],
        "academic": [
            "$4,500 Academic Scholarship at UF",
            "GPA Achievement Award (4.0 in final year)",
            "Three published research papers in 2024"
        ]
    }

    formatted_answer = "Key Achievements:\n\n"

    # Teaching Achievements
    formatted_answer += "Teaching Impact:\n"
    for achievement in achievements["teaching"]:
        formatted_answer += f"* {achievement}\n"

    # Professional Achievements
    formatted_answer += "\nProfessional Impact:\n"
    for achievement in achievements["professional"]:
        formatted_answer += f"* {achievement}\n"

    # Academic Achievements
    formatted_answer += "\nAcademic Recognition:\n"
    for achievement in achievements["academic"]:
        formatted_answer += f"* {achievement}\n"

    return formatted_answer


def chunk_content(content: str, file_path: Path) -> List[Document]:
    """Chunk content with professional context preservation"""
    # Split by sections
    sections = [s.strip() for s in content.split('\n\n') if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for section in sections:
        section_words = section.split()
        section_word_count = len(section_words)

        if section_word_count > MAX_CHUNK_SIZE:
            # Handle large sections (like project descriptions)
            words = section.split()
            for i in range(0, len(words), MAX_CHUNK_SIZE - OVERLAP_SIZE):
                chunk_words = words[i:i + MAX_CHUNK_SIZE]
                chunk_text = ' '.join(chunk_words)

                # Add metadata for context
                meta = {
                    'file_name': file_path.name,
                    'category': file_path.stem,
                    'section_type': 'partial_section',
                    'section_position': f"part_{i // (MAX_CHUNK_SIZE - OVERLAP_SIZE)}"
                }

                # Add formatting hints for common sections
                if 'experience' in file_path.stem.lower():
                    meta['context_type'] = 'professional_experience'
                elif 'education' in file_path.stem.lower():
                    meta['context_type'] = 'education'
                elif 'skills' in file_path.stem.lower():
                    meta['context_type'] = 'technical_skills'
                elif 'projects' in file_path.stem.lower():
                    meta['context_type'] = 'projects'

                chunks.append(Document(content=chunk_text, meta=meta))

        elif current_word_count + section_word_count > MAX_CHUNK_SIZE:
            # Create new chunk when current one is full
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Document(
                    content=chunk_text,
                    meta={
                        'file_name': file_path.name,
                        'category': file_path.stem,
                        'section_type': 'complete_sections'
                    }
                ))
            current_chunk = [section]
            current_word_count = section_word_count
        else:
            current_chunk.append(section)
            current_word_count += section_word_count

    # Add remaining content
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append(Document(
            content=chunk_text,
            meta={
                'file_name': file_path.name,
                'category': file_path.stem,
                'section_type': 'complete_sections'
            }
        ))

    return chunks


def create_generator() -> HuggingFaceLocalGenerator:
    """Create enhanced language model generator"""
    generator = HuggingFaceLocalGenerator(model=GENERATOR_MODEL)

    # Configure generation parameters after creation
    generation_kwargs = {
        'max_new_tokens': 512,
        'min_length': 50,
        'do_sample': False,
        'num_beams': 5,
        'early_stopping': True,
        'repetition_penalty': 1.1,
        'no_repeat_ngram_size': 2,
        'length_penalty': 1.0
    }

    # If the model has an EOS token, add it to the kwargs
    if hasattr(generator, 'tokenizer') and hasattr(generator.tokenizer, 'eos_token_id'):
        generation_kwargs['eos_token_id'] = generator.tokenizer.eos_token_id

    # Create new instance with all parameters
    return HuggingFaceLocalGenerator(
        model=GENERATOR_MODEL,
        generation_kwargs=generation_kwargs
    )


# Update these lines near the top of app.py
CORPUS_DOCUMENTS_PATH = os.getenv(
    "CORPUS_DOCUMENTS_PATH", "./data/professional_info")  # Changed default path
CORPUS_DOCUMENTS_FILE_EXT = os.getenv("CORPUS_DOCUMENTS_FILE_EXT", "txt")

# Update embed_documents function with better error handling


def create_initial_files(directory: str) -> None:
    """Create well-structured professional profile data"""
    files = {
        'profile.txt': """
PROFESSIONAL PROFILE
Name: Kushagra Sikka
Current Role: Graduate Teaching Assistant & Research Assistant
Institution: University of Florida
Location: Gainesville, FL
Contact: 
- Email: kushagrasikka@gmail.com, kushagrasikka@ufl.edu
- Phone: +1 (352) 740-6029
Links:
- GitHub: https://github.com/KushagraSikka
- LinkedIn: in/kushagrasikka
- Portfolio: https://www.kushagrasikka.com""",

        'current_role.txt': """
CURRENT POSITIONS (2024-Present)

Graduate Teaching Assistant - University of Florida
- Instructing 60+ students in assembly mechanics and memory integrity
- Improved student grades by 20% across the cohort
- Automated coursework management reducing administrative time by 50%
- Enhanced student engagement by 25%

Research Assistant - Trustworthy-Engineered-Autonomy-Lab
- Focus: Safe autonomous systems and robotics
- Research Areas: DevOps integration with AI/ML workflows
- Published researcher in Computer Science & Digital Technologies""",

        'work_experience.txt': """
PROFESSIONAL EXPERIENCE

Data Engineer Intern - Salescode AI (May 2022 - July 2022)
- Implemented time series analysis increasing sales by 7.3%
- Engineered predictive models reducing inventory costs by 15%
- Published technical whitepaper leading to Pre-Placement Offer
- Integrated client data with custom recommendation engine

Software Engineer Intern - VKS ValveCraft Solutions (April 2021 - April 2022)
- Optimized SQL database performance by 20%
- Reduced deployment times by 40% through Jenkins pipeline improvements
- Developed resume-ranking algorithm achieving 89% accuracy
- Automated test workflows improving efficiency by 32%""",

        'education.txt': """
EDUCATION

Master's in Computer Science - University of Florida (2023-2025)
- GPA: 3.6
- Focus Areas: Advanced Data Structures, Distributed Systems, AI Ethics
- Research: Trustworthy Autonomous Systems
- Academic Scholarship: $4,500

Bachelor's in Computer Science - Manipal University (2019-2023)
- GPA: 3.7
- Key Areas: Data Structures, Deep Learning, Computer Vision
- Achievement: GPA Achievement Award (4.0 in final year)""",

        'skills.txt': """
TECHNICAL EXPERTISE

Programming Languages:
- Primary: Python, Java
- Secondary: Ruby, Go

Cloud & Infrastructure:
- AWS: EC2, S3, RDS, SageMaker, Lambda
- Other: Microsoft Azure, GCP
- Infrastructure as Code: Terraform

DevOps & Tools:
- CI/CD: Jenkins, GitHub Actions
- Containerization: Docker, Kubernetes
- Version Control: Git
- API Testing: Postman
- Scripting: Bash

Databases:
- SQL: PostgreSQL, MySQL
- NoSQL: MongoDB
- Cloud: Amazon RDS

AI/ML Technologies:
- Frameworks: TensorFlow, PyTorch
- Libraries: Scikit-learn
- Cloud ML: AWS SageMaker
- NLP: Hugging Face

Professional Skills:
- Technical Leadership
- Project Management
- Research & Publication
- Technical Writing""",

        'projects.txt': """
KEY PROJECTS

RAG Microservice System:
- Enhanced AI response relevance by 25%
- Implemented Docker containerization
- Improved deployment efficiency by 35%
- Tech Stack: FastAPI, React, Docker

Incident Data Enrichment Project:
- Processed 10,000+ police reports monthly
- Reduced deployment time by 50% using Terraform
- Improved data accuracy by 35%
- Implemented ELK Stack monitoring

Research Publications:
1. Web Service Classification Analysis (IC2SDT 2024)
   DOI: 10.1109/IC2SDT62152.2024.1069639
2. Deep Learning for Plant Disease Classification (IC2SDT 2024)
   DOI: 10.1109/IC2SDT62152.2024.10696395
3. Twitter Sentiment Analysis for Elections (IC2SDT 2024)
   DOI: 10.1109/IC2SDT62152.2024.10696204"""
    }

    for filename, content in files.items():
        file_path = Path(directory) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        logger.info(f"Created file: {filename}")


def process_response(answer: str, question: str) -> str:
    """Process and enhance response based on question type"""
    question_lower = question.lower()

    if "achievement" in question_lower or "accomplishment" in question_lower:
        return format_achievements_response(answer)

    if "who" in question_lower or "about" in question_lower:
        return format_profile_response(answer)

    if "skill" in question_lower or "expertise" in question_lower:
        return format_skills_response(answer)

    return format_general_response(answer)


def format_profile_response(answer: str) -> str:
    """Format profile-related responses with bullet points and subheadings."""
    # Remove any unwanted characters
    answer = answer.replace('\r', '').strip()

    # Split the answer into lines
    lines = answer.split('\n')

    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            # Subheading detected
            formatted_lines.append(f"\n{line}\n")
        elif line.startswith('- '):
            # Bullet point detected
            formatted_lines.append(line)
        elif line:
            # Any other non-empty line
            formatted_lines.append(f"- {line}")

    formatted_answer = '\n'.join(formatted_lines).strip()
    return formatted_answer


def format_skills_response(answer: str) -> str:
    """Format skills-related responses"""
    skills = {
        "Programming": [
            "Python, Java (Primary)",
            "Ruby, Go (Secondary)",
            "Full-stack development capabilities"
        ],
        "Cloud & DevOps": [
            "AWS (EC2, S3, RDS, SageMaker, Lambda)",
            "Docker, Kubernetes, Jenkins",
            "Terraform for infrastructure"
        ],
        "Data & AI": [
            "TensorFlow, PyTorch, Scikit-learn",
            "SQL and NoSQL databases",
            "Machine Learning deployment"
        ]
    }

    formatted = "Technical Expertise:\n"
    for category, items in skills.items():
        formatted += f"\n{category}:\n"
        for item in items:
            formatted += f"* {item}\n"

    return formatted.strip()


def format_general_response(answer: str) -> str:
    """Format general responses with better structure"""
    # Clean up formatting
    answer = answer.replace("[", "")
    answer = answer.replace("]", "")
    answer = answer.replace(" - ", "\n* ")

    # Ensure proper sections
    if not any(section in answer for section in ["Overview:", "Summary:", "Details:"]):
        parts = answer.split("\n\n")
        formatted = "Overview:\n"
        formatted += "* " + parts[0].replace("*", "").strip() + "\n\n"
        if len(parts) > 1:
            formatted += "Details:\n"
            for part in parts[1:]:
                points = part.split("\n")
                for point in points:
                    if point.strip():
                        formatted += "* " + \
                            point.replace("*", "").strip() + "\n"
        answer = formatted

    return answer.strip()


@app.post("/ask", response_model=Answer)
async def ask_question(request: QuestionRequest):
    """Process a question with improved response handling"""
    try:
        start_time = time.time()
        metrics = []

        def log_metrics(step: str, tokens: int) -> None:
            current_time = time.time()
            metrics.append(ProcessingMetrics(
                timestamp=current_time * 1000,
                tokens_processed=tokens,
                processing_time=current_time - start_time,
                step=step
            ))

        if not hasattr(app.state, "rag_pipeline"):
            raise HTTPException(
                status_code=500, detail="RAG pipeline not initialized")

        question = request.question
        question_tokens = len(question.split())
        log_metrics("input", question_tokens)

        try:
            response = app.state.rag_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                }
            )

            retrieved_docs = response.get("retriever", {}).get("documents", [])

            # Log retrieved context for debugging
            for doc in retrieved_docs:
                logger.info(
                    f"Retrieved context from {doc.meta.get('file_name', 'unknown')}: {doc.content[:100]}..."
                )

            retrieved_tokens = sum(len(doc.content.split())
                                   for doc in retrieved_docs)
            log_metrics("retrieval", retrieved_tokens)

            raw_answer = response["llm"]["replies"][0]
            processed_answer = process_response(raw_answer, question)
            answer_tokens = len(processed_answer.split())
            log_metrics("generation", answer_tokens)

            # Validate answer quality
            if len(processed_answer.split()) < 10 or "I don't have" in processed_answer:
                logger.warning("Generated potentially low-quality answer")

            return Answer(answer=processed_answer, metrics=[m.dict() for m in metrics])

        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Pipeline error: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def embed_documents(document_store: InMemoryDocumentStore) -> None:
    """Process and embed documents with improved chunking"""
    abs_path = os.path.abspath(CORPUS_DOCUMENTS_PATH)
    logger.info(f"Processing documents from: {abs_path}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(abs_path, exist_ok=True)

        all_files = list(Path(abs_path).glob(
            f"**/*.{CORPUS_DOCUMENTS_FILE_EXT}"))
        if not all_files:
            # Create example files if none exist
            create_initial_files(abs_path)
            all_files = list(Path(abs_path).glob(
                f"**/*.{CORPUS_DOCUMENTS_FILE_EXT}"))

        logger.info(f"Found {len(all_files)} documents")

        # Process each file with improved chunking
        all_chunks = []
        for file_path in all_files:
            try:
                content = file_path.read_text()
                logger.info(
                    f"Processing {file_path.name}: {len(content)} characters")

                # Use improved chunking strategy
                file_chunks = chunk_content(content, file_path)
                logger.info(
                    f"Split {file_path.name} into {len(file_chunks)} chunks")

                all_chunks.extend(file_chunks)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        if not all_chunks:
            raise ValueError("No document chunks could be created")

        # Embed chunks
        logger.info(f"Embedding {len(all_chunks)} chunks")
        doc_embedder = create_document_embedder()
        doc_embedder.warm_up()
        docs_with_embeddings = doc_embedder.run(all_chunks)
        document_store.write_documents(docs_with_embeddings["documents"])
        logger.info(f"Successfully embedded {len(all_chunks)} document chunks")

    except Exception as e:
        logger.error(f"Error in embed_documents: {str(e)}")
        raise


def create_rag_pipeline(
    text_embedder: SentenceTransformersTextEmbedder,
    retriever: InMemoryEmbeddingRetriever,
    prompt_builder: PromptBuilder,
    generator: HuggingFaceLocalGenerator,
) -> Pipeline:
    """Create the RAG pipeline with all components"""
    logger.info("Creating RAG pipeline")
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)

    # Connect components
    rag_pipeline.connect("text_embedder.embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    return rag_pipeline


@app.on_event("startup")
async def load_pipeline():
    """Initialize the RAG pipeline on startup"""
    try:
        document_store = create_document_store()
        embed_documents(document_store)

        text_embedder = create_text_embedder()
        retriever = create_retriever(document_store)
        prompt_builder = create_prompt_builder()
        generator = create_generator()

        rag_pipeline = create_rag_pipeline(
            text_embedder=text_embedder,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generator=generator,
        )

        app.state.rag_pipeline = rag_pipeline
        logger.info("Successfully initialized RAG pipeline")

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        raise


@app.post("/ask", response_model=Answer)
async def ask_question(request: QuestionRequest):
    """Process a question with improved response handling"""
    try:
        start_time = time.time()
        metrics = []

        def log_metrics(step: str, tokens: int) -> None:
            current_time = time.time()
            metrics.append(ProcessingMetrics(
                timestamp=current_time * 1000,
                tokens_processed=tokens,
                processing_time=current_time - start_time,
                step=step
            ))

        if not hasattr(app.state, "rag_pipeline"):
            raise HTTPException(
                status_code=500, detail="RAG pipeline not initialized")

        question = request.question
        question_tokens = len(question.split())
        log_metrics("input", question_tokens)

        try:
            response = app.state.rag_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                }
            )

            retrieved_docs = response.get("retriever", {}).get("documents", [])

            # Log retrieved context for debugging
            for doc in retrieved_docs:
                logger.info(
                    f"Retrieved context from {doc.meta.get('file_name', 'unknown')}: {doc.content[:100]}..."
                )

            retrieved_tokens = sum(len(doc.content.split())
                                   for doc in retrieved_docs)
            log_metrics("retrieval", retrieved_tokens)

            raw_answer = response["llm"]["replies"][0]
            processed_answer = process_response(raw_answer, question)
            answer_tokens = len(processed_answer.split())
            log_metrics("generation", answer_tokens)

            # Validate answer quality
            if answer_tokens < 10 or "I don't have" in processed_answer:
                logger.warning("Generated potentially low-quality answer")

            return Answer(answer=processed_answer, metrics=[m.dict() for m in metrics])

        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Pipeline error: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the service"""
    try:
        if not hasattr(app.state, "rag_pipeline"):
            return {"status": "unhealthy", "error": "RAG pipeline not initialized"}
        return {
            "status": "healthy",
            "model_info": {
                "embedder": TEXT_EMBEDDING_MODEL,
                "generator": GENERATOR_MODEL
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

CORPUS_DOCUMENTS_PATH = os.getenv(
    "CORPUS_DOCUMENTS_PATH", "./data/professional_info")
