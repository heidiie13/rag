from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from core.embedding import create_embeddings

def load_retriever(collection_name: str = "rag_collection", persist_directory="vectorstore/", **kwargs):
    embeddings = create_embeddings()
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": f"{persist_directory}/milvus.db"},
        collection_name=collection_name
    )
    return vectorstore.as_retriever(search_kwargs={**kwargs})