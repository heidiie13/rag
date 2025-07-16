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
    question: str
    chat_history: list = []
   
@app.post("/chat/stream")
def get_answer_stream(request: QueryRequest):
    return StreamingResponse(
        rag_pipeline.get_answer_stream(request.question, request.chat_history),
        media_type="text/event-stream"
    )

@app.post("/chat")
def get_answer(request: QueryRequest):
    return rag_pipeline.get_answer(request.question, request.chat_history)