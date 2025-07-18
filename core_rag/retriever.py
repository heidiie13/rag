import os
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from core_rag.embedding import create_embeddings
from dotenv import load_dotenv

load_dotenv()

def load_vector_store(collection_name="rag"):
    embeddings = create_embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        prefer_grpc=True,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
def load_retriever(**kwargs):
    return load_vector_store().as_retriever(search_kwargs={**kwargs})


def load_ensemble_retriever(collection_name="rag", k=5, n=20, m=10):
    """
    Creates a hybrid retriever combining vector and BM25 methods to fetch documents from a collection.
    
    Args:
        collection_name: Name of the collection (default: "rag").
        k: Number of documents to return (default: 5).
        n: Number of documents to fetch using vector retriever (default: 20).
        m: Number of documents to fetch using BM25 (default: 10).
    
    Returns:
        A hybrid_retriever(query) function that returns a list of relevant documents.
    """
    
    vector_store = load_vector_store(collection_name)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": n})

    def hybrid_retriever(query: str):
        docs = vector_retriever.invoke(query)

        bm25_retriever = BM25Retriever.from_documents(docs, k=m)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        combined = ensemble.invoke(query)

        seen, final = set(), []
        for doc in combined:
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if key not in seen:
                seen.add(key)
                final.append(doc)
                if len(final) == k:
                    break
        return final

    return hybrid_retriever