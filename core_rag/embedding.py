import os

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

def create_embeddings(model_name: str = "intfloat/multilingual-e5-small") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def create_vector_store(documents: list[Document], 
                       embeddings: HuggingFaceEmbeddings,
                       collection_name: str = "rag") -> QdrantVectorStore | None:

    if not documents or not embeddings:
        return None
    
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        prefer_grpc=True,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    return vector_store