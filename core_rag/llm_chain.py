import os
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

def create_llm():
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME"),
        )
    # return ChatOpenAI(
    #     base_url=os.getenv("OPENAI_BASE_URL"),
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model=os.getenv("MODEL_NAME"),
    #     temperature=0.7,
    #     default_headers={"App-Code": "fresher"},
    #     extra_body= {
    #         "service": "test_rag_for_langchain_app",
    #         "chat_template_kwargs": {
    #             "enable_thinking": False
    #             }
    #         },
    #     )


def create_prompt_template() -> PromptTemplate:
    prompt_template =\
    """You are an intelligent and helpful AI assistant. Your task is to answer questions based on the information provided in the context.
    Context:
    {context}
    
    Instructions:
    1. Only use information from the provided context to answer the question.
    2. If the context does not contain enough information to answer, state "I do not have enough information in the provided context to answer this question."
    3. Answer accurately, clearly, and structurally.
    4. Answer in natural and easy-to-understand Vietnamese.
    
    Question: 
    {question}
    
    Answer:"""
    
    return PromptTemplate.from_template(template=prompt_template)

def create_qa_chain(llm: ChatOpenAI, retriever):
    if not llm or not retriever:
        return None

    prompt = create_prompt_template()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain
