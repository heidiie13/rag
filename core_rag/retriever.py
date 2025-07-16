import os
from langchain_qdrant import QdrantVectorStore
from core_rag.embedding import create_embeddings
from dotenv import load_dotenv
load_dotenv()

def load_retriever(collection_name: str = "rag", **kwargs):
    embeddings = create_embeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        prefer_grpc=True,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    return vector_store.as_retriever(search_kwargs={**kwargs})