import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

def create_embeddings(model_name: str = "intfloat/multilingual-e5-small") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def create_vector_store(documents: list[Document], 
                       embeddings: HuggingFaceEmbeddings,
                       collection_name: str = "rag_collection",
                       persist_directory: str = "vectorstore") -> Milvus | None:

    if not documents or not embeddings:
        return None
    
    os.makedirs(persist_directory, exist_ok=True)
    
    URI = os.path.join(persist_directory, "milvus.db")
    
    try:
        vector_store = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={"uri": URI},
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            drop_old=True
        )
        
        return vector_store
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None