import os
import sys
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core_rag.document_loader import load_and_split_docs_dir
from core_rag.embedding import create_embeddings, create_vector_store

load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__file__)

if __name__ == "__main__":
    logger.info("Loading and splitting documents from directory: data")
    docs = load_and_split_docs_dir("data")
    logger.info(f"Loaded and split {len(docs)} documents.")

    embedding_model = os.getenv("EMBEDDING_MODEL")
    embeddings = create_embeddings(str(embedding_model))
    logger.info("Creating vector store...")
    create_vector_store(docs, embeddings)
    logger.info("Vector store creation complete.")