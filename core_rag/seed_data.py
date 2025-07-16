import os
import sys
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core_rag.document_loader import load_documents, split_documents
from core_rag.embedding import create_embeddings, create_vector_store

load_dotenv()
if __name__ == "__main__":
   # Load documents
   documents = load_documents("data")
   chunks = split_documents(documents)
   
   # Create embeddings
   embedding_model = os.getenv("EMBEDDING_MODEL")
   embeddings = create_embeddings(embedding_model)
   create_vector_store(chunks, embeddings)