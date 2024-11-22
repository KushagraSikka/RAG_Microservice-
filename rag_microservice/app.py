from fastapi import FastAPI, HTTPException
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
CORPUS_DOCUMENTS_PATH = os.getenv(
    "CORPUS_DOCUMENTS_PATH", "./data/rick_and_morty_episodes")
CORPUS_DOCUMENTS_FILE_EXT = os.getenv("CORPUS_DOCUMENTS_FILE_EXT", "txt")
TEXT_EMBEDDING_MODEL = os.getenv(
    "TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATOR_MODEL = "google/flan-t5-small"

# Constants
MAX_CHUNK_SIZE = 200  # Maximum number of words per chunk
MAX_CHUNKS = 3  # Maximum number of chunks to include in context

# FastAPI app initialization
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


def create_document_store() -> InMemoryDocumentStore:
    logger.info("Instantiating RAG document store")
    return InMemoryDocumentStore()


def create_document_splitter() -> DocumentSplitter:
    logger.info("Creating document splitter")
    return DocumentSplitter(
        split_by="word",
        split_length=MAX_CHUNK_SIZE,
        split_overlap=50
    )


def create_document_embedder() -> SentenceTransformersDocumentEmbedder:
    logger.info(
        f"Creating document embedder with model {TEXT_EMBEDDING_MODEL}")
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=TEXT_EMBEDDING_MODEL
    )
    return document_embedder


def create_text_embedder() -> SentenceTransformersTextEmbedder:
    logger.info(f"Creating text embedder with model {TEXT_EMBEDDING_MODEL}")
    text_embedder = SentenceTransformersTextEmbedder(
        model=TEXT_EMBEDDING_MODEL
    )
    return text_embedder


def create_retriever(document_store: InMemoryDocumentStore) -> InMemoryEmbeddingRetriever:
    logger.info("Creating retriever")
    return InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=MAX_CHUNKS
    )


def create_prompt_builder() -> PromptBuilder:
    template = """
    Answer the question about Rick and Morty based on the following context. 
    Keep the answer concise and relevant to the question.
    
    Context:
    {% for document in documents[:3] %}
    {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer: """

    logger.info("Creating prompt builder")
    return PromptBuilder(template=template)


def create_generator() -> HuggingFaceLocalGenerator:
    """
    Creates a HuggingFace generator with fallback options if the primary model fails.
    """
    logger.info(
        f"Creating Hugging Face generator with model {GENERATOR_MODEL}")
    try:
        return HuggingFaceLocalGenerator(
            model=GENERATOR_MODEL,
            generation_kwargs={
                'max_length': 150,
                'do_sample': True,
                'temperature': 0.7,
                'num_return_sequences': 1,
                'min_length': 30,  # Ensure we get meaningful responses
                'no_repeat_ngram_size': 3,  # Avoid repetition
                'early_stopping': True
            }
        )
    except Exception as e:
        logger.error(
            f"Failed to load primary model {GENERATOR_MODEL}: {str(e)}")
        # Fallback to a simpler model
        logger.info("Attempting to load fallback model t5-small")
        return HuggingFaceLocalGenerator(
            model="t5-small",
            generation_kwargs={
                'max_length': 100,
                'do_sample': False,  # More deterministic for fallback
                'num_return_sequences': 1
            }
        )


def embed_documents(document_store: InMemoryDocumentStore) -> None:
    abs_path = os.path.abspath(CORPUS_DOCUMENTS_PATH)
    logger.info(f"Looking for documents in: {abs_path}")

    try:
        all_files = list(Path(abs_path).glob(
            f"**/*.{CORPUS_DOCUMENTS_FILE_EXT}"))
        logger.info(
            f"Found {len(all_files)} files: {[f.name for f in all_files]}")

        if not all_files:
            raise ValueError(
                f"No .{CORPUS_DOCUMENTS_FILE_EXT} files found in {abs_path}")

        # Create document splitter
        splitter = create_document_splitter()

        # Process each file
        all_chunks = []
        for file_path in all_files:
            try:
                content = file_path.read_text()
                logger.info(
                    f"Processing {file_path.name}: {len(content)} characters")

                # Create initial document
                doc = Document(content=content)

                # Split into chunks
                chunks = splitter.run([doc])
                logger.info(
                    f"Split {file_path.name} into {len(chunks['documents'])} chunks")

                all_chunks.extend(chunks['documents'])

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        if not all_chunks:
            raise ValueError("No document chunks could be created")

        # Embed chunks
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


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not hasattr(app.state, "rag_pipeline"):
            raise HTTPException(
                status_code=500, detail="RAG pipeline not initialized")

        question = request.question
        logger.info(f"Received question: {question}")

        response = app.state.rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )

        answer = response["llm"]["replies"][0]
        logger.info(f"Generated answer: {answer}")

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        if not hasattr(app.state, "rag_pipeline"):
            return {"status": "unhealthy", "error": "RAG pipeline not initialized"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
