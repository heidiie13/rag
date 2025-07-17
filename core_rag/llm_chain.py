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
        model=os.getenv("MODEL_NAME"),
        )
   
   # return ChatOpenAI(
   #      base_url=os.getenv("OPENAI_BASE_URL"),
   #      api_key=os.getenv("OPENAI_API_KEY"),
   #      model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
   #      temperature=0.7,
   #      default_headers={"App-Code": "fresher"},
   #      extra_body= {
   #          "service": "test_rag_for_langchain_app",
   #          "chat_template_kwargs": {
   #              "enable_thinking": False
   #              }
   #          },
   #      )

def create_qa_system_prompt() -> ChatPromptTemplate:
   qa_system_prompt = """You are a knowledgeable and reliable AI legal assistant specializing in matters of taxes, fees, and charges (Thuế - Phí - Lệ phí). Your task is to assist users by answering questions based on the legal information provided in the context below.

<context>:
{context}
</context>

RESPONSE GUIDELINES:
1. Use only the information from the provided context to answer the question.
2. Provide accurate, clear, and well-structured answers.
3. ALWAYS respond in the same language as the question.
4. Always cite relevant legal documents that support your answer, if such references appear in the context.
5. If the context does not contain enough information to fully answer the question, summarize any partially relevant information, and clearly state what specific information is missing. Do not reject the question outright; instead, explain what additional details would be needed.
6. Organize your response with the most important information first, including the effective date or issuance date of the legal document when available.
7. Use an appropriate tone and formality based on the legal context.
8. Avoid providing personal legal advice or speculating beyond the provided context.

IMPORTANT:
- Your answers must be **based strictly on the provided context**.
- Do not invent or assume legal references; **only mention laws or resolutions if they appear in the context.**
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