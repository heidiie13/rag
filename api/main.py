from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core_rag.rag_pipeline import RAGPipeline


app = FastAPI()
rag_pipeline = RAGPipeline()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*']
)

class QueryRequest(BaseModel):
   query: str
   
@app.post("/chat/stream")
async def get_answer_stream(question: QueryRequest):
   return StreamingResponse(rag_pipeline.get_answer_stream(question.query), media_type="text/event-stream")

@app.post("/chat")
async def get_answer(question: QueryRequest):
   return rag_pipeline.get_answer(question.query)