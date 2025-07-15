from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader

def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def load_documents(data_dir: str = "data") -> list[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    return documents

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    if not documents:
        return []
    
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    return text_splitter.split_documents(documents)