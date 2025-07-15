from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

def create_llm():
    pass

def create_prompt_template() -> PromptTemplate:
    """
    Tạo prompt template cho RAG
    """
    prompt_template =\
    """Bạn là một trợ lý AI thông minh và hữu ích. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp trong ngữ cảnh.
    Ngữ cảnh:
    {context}
    
    Hướng dẫn:
    1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp để trả lời câu hỏi
    2. Nếu ngữ cảnh không chứa đủ thông tin để trả lời, hãy nói "Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời câu hỏi này"
    3. Trả lời một cách chính xác, rõ ràng và có cấu trúc
    4. Nếu có thể, hãy đề cập đến nguồn hoặc tài liệu liên quan
    5. Trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu
    
    Câu hỏi: {question}
    
    Trả lời:"""
    
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

def create_qa_chain(llm: ChatOpenAI, retriever) -> RetrievalQA:
    if not llm or not retriever:
        return None

    prompt = create_prompt_template()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain
