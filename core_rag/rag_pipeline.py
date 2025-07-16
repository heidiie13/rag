from core_rag.llm_chain import create_llm, create_qa_chain
from core_rag.retriever import load_retriever

class RAGPipeline:
   def __init__(self):
      self.retriever = load_retriever()
      self.llm = create_llm()
      self.qa_chain = create_qa_chain(self.llm, self.retriever)
      
   def get_answer(self, query):
      answer = self.qa_chain.invoke({"input": query})
      return answer.get("answer","")
   
   def get_answer_stream(self, query):
      for chunk in self.qa_chain.stream({"input": query}):
         yield chunk.get("answer","")