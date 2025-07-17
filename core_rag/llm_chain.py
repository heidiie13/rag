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
   #  return ChatOpenAI(
   #      base_url=os.getenv("OPENAI_BASE_URL"),
   #      api_key=os.getenv("OPENAI_API_KEY"),
   #      model=os.getenv("MODEL_NAME"),
   #      )
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
   qa_system_prompt = """You are an intelligent and helpful AI assistant specializing in legal matters, specifically acting as a chatbot for taxes, fees, and charges (Thuế - Phí - Lệ phí). Answer questions using only the information provided in the context below. 
   <context>:
   {context}
   </context>
   
   RESPONSE GUIDELINES:
   1. Use only the information from the provided context to answer the question.
   2. Provide accurate, clear, and well-structured answers.
   3. ALWAYS respond in the same language as the question (Vietnamese → Vietnamese, English → English).
   4. If the question requires information not available in the context, clearly indicate this limitation.
   5. Organize your response with the most important information first.
   6. Use an appropriate tone and formality based on the legal context.
   7. Avoid providing personal legal advice or speculating beyond the provided context.

   IMPORTANT: Ensure accuracy and never speculate beyond the provided context.
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