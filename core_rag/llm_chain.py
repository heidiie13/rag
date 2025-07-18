import os
from dotenv import load_dotenv

from langchain.chains import (
      create_history_aware_retriever,
      create_retrieval_chain,
)
from langchain_core.runnables import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

def create_llm():
   if not os.getenv("OPENAI_BASE_URL") or not os.getenv("OPENAI_API_KEY"):
      raise ValueError("OPENAI_BASE_URL, OPENAI_API_KEY environment variables must be set")

   return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.7,
        default_headers={"App-Code": "fresher"},
        extra_body= {
            "service": "test_rag_for_langchain_app",
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            },
        )

def create_qa_system_prompt() -> ChatPromptTemplate:
   qa_system_prompt = """Bạn là một trợ lý AI pháp lý đáng tin cậy và am hiểu, chuyên về các vấn đề liên quan đến Thuế - Phí - Lệ phí. Nhiệm vụ của bạn là hỗ trợ người dùng bằng cách trả lời các câu hỏi dựa trên thông tin pháp lý được cung cấp dưới đây.

<context>:
{context}
</context>

HƯỚNG DẪN TRẢ LỜI:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp để trả lời câu hỏi.
2. Cung cấp câu trả lời chính xác, rõ ràng và có cấu trúc tốt.
3. LUÔN trả lời bằng cùng ngôn ngữ với câu hỏi.
4. Luôn trích dẫn các văn bản pháp lý liên quan hỗ trợ câu trả lời, nếu có trong ngữ cảnh (Không tự ý tạo ra hoặc giả định các tham chiếu pháp lý).
5. Nếu ngữ cảnh không chứa đủ thông tin để trả lời đầy đủ câu hỏi, hãy tóm tắt các thông tin liên quan một phần và nêu rõ thông tin cụ thể nào còn thiếu. Không từ chối câu hỏi; thay vào đó, hãy giải thích những chi tiết bổ sung cần thiết.
6. Tổ chức câu trả lời với thông tin quan trọng nhất trước, bao gồm ngày hiệu lực hoặc ngày ban hành của văn bản pháp lý (nếu có).
7. Sử dụng giọng văn và mức độ trang trọng phù hợp với bối cảnh pháp lý.
8. Tránh đưa ra lời khuyên pháp lý cá nhân hoặc suy đoán vượt quá ngữ cảnh được cung cấp.

"""
   return ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def create_qa_chain(llm: ChatOpenAI, retriever) -> Runnable:
   contextualize_q_system_prompt = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, just "
      "reformulate it if needed and otherwise return it as is."
   )
   contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
         ("system", contextualize_q_system_prompt),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}"),
      ]
   )
   history_aware_retriever = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
   )
   
   qa_prompt = create_qa_system_prompt()
   
   question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
   
   rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
      )
   
   return rag_chain