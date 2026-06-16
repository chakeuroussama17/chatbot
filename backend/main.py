from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os

from rag_engine import RAGEngine

app = FastAPI(title="FAQ Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()


class ChatRequest(BaseModel):
    question: str
    chat_history: List[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    chunks_used: int


@app.get("/")
def health():
    return {"status": "ok", "docs_loaded": rag.get_doc_count()}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Ingest one or more documents into the vector store."""
    results = []
    for file in files:
        if not file.filename:
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".txt", ".docx"]:
            raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF, TXT, or DOCX.")

        contents = await file.read()
        result = rag.ingest_document(contents, file.filename, ext)
        results.append(result)

    return {"message": f"Ingested {len(results)} file(s)", "details": results}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer a question using RAG over ingested documents."""
    if rag.get_doc_count() == 0:
        raise HTTPException(400, "No documents loaded. Please upload documents first.")

    answer, sources, chunks_used = rag.answer(req.question, req.chat_history)
    return ChatResponse(answer=answer, sources=sources, chunks_used=chunks_used)


@app.delete("/documents")
def clear_documents():
    """Clear all documents from the vector store."""
    rag.clear()
    return {"message": "All documents cleared."}


@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    return {"documents": rag.list_documents()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
