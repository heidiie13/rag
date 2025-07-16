from core_rag.llm_chain import create_llm, create_qa_chain
from core_rag.retriever import load_retriever

class RAGPipeline:
   def __init__(self):
      self.retriever = load_retriever()
      self.llm = create_llm()
      self.qa_chain = create_qa_chain(self.llm, self.retriever)
      
   def get_answer(self, query, context=None):
      payload = {"question": query}
      if context:
         payload["chat_history"] = context
      answer = self.qa_chain.invoke(payload)

      if isinstance(answer, dict):
         return answer.get("answer","")
      return answer

   def get_answer_stream(self, query, context=None):
      payload = {"question": query}
      if context:
         payload["chat_history"] = context
      for chunk in self.qa_chain.stream(payload):
         if isinstance(chunk, dict):
            yield chunk.get("answer","")
         else:
            yield chunk