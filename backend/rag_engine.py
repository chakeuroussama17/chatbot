"""
RAG Engine — the heart of the chatbot.
Handles: document parsing → chunking → embedding → FAISS indexing → retrieval → LLM answer
"""

import os
import io
import json
import faiss
import numpy as np
import tiktoken
from typing import List, Tuple, Optional
from dotenv import load_dotenv

import fitz  # PyMuPDF
import docx
from openai import OpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # tokens per chunk
CHUNK_OVERLAP = 60        # token overlap between chunks
TOP_K = 5                 # how many chunks to retrieve
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def split_into_chunks(text: str, source: str) -> List[dict]:
    """Split text into overlapping token-based chunks."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
        })

        chunk_index += 1
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP  # slide window with overlap

    return chunks


def extract_text(contents: bytes, filename: str, ext: str) -> str:
    """Parse PDF, DOCX, or TXT → raw text string."""
    if ext == ".pdf":
        doc = fitz.open(stream=contents, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(io.BytesIO(contents))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        return contents.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch embed a list of strings via OpenAI. Returns (N, EMBED_DIM) float32 array."""
    # OpenAI allows up to 2048 inputs per call; batch in groups of 100 to be safe
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype="float32")


# ── RAG Engine ───────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks: List[dict] = []       # parallel to FAISS index rows
        self.doc_names: List[str] = []     # track ingested filenames

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_document(self, contents: bytes, filename: str, ext: str) -> dict:
        """Parse → chunk → embed → add to FAISS index."""
        # 1. Extract text
        raw_text = extract_text(contents, filename, ext)
        if not raw_text.strip():
            return {"file": filename, "status": "skipped", "reason": "empty content"}

        # 2. Chunk
        chunks = split_into_chunks(raw_text, source=filename)

        # 3. Embed
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # 4. Normalise (for cosine similarity via inner product)
        faiss.normalize_L2(embeddings)

        # 5. Build or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product = cosine after normalisation

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        if filename not in self.doc_names:
            self.doc_names.append(filename)

        return {
            "file": filename,
            "status": "ok",
            "chunks": len(chunks),
            "tokens": sum(c["token_count"] for c in chunks),
        }

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, question: str, k: int = TOP_K) -> List[dict]:
        """Embed the question and return the top-K most relevant chunks."""
        q_embedding = embed_texts([question])
        faiss.normalize_L2(q_embedding)

        scores, indices = self.index.search(q_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Answer Generation ────────────────────────────────────────────────────

    def answer(self, question: str, chat_history: list) -> Tuple[str, List[dict], int]:
        """Retrieve relevant chunks, build prompt, call Claude, return answer + sources."""
        retrieved = self.retrieve(question)

        if not retrieved:
            return "I couldn't find any relevant information in the documents.", [], 0

        # Build context block
        context_parts = []
        for i, chunk in enumerate(retrieved):
            context_parts.append(
                f"[Source {i+1}: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build system prompt
        system_prompt = f"""You are a precise FAQ assistant. Answer the user's question based ONLY on the context below.

Rules:
- If the context contains the answer, give a clear, helpful response.
- If the context does NOT contain the answer, say: "I don't have information about that in the provided documents."
- Never make up information not in the context.
- When relevant, reference which source the information came from.
- Be concise and direct.

CONTEXT:
{context}"""

        # Build messages (include chat history for multi-turn)
        messages = []
        for turn in chat_history[-6:]:  # last 3 turns = 6 messages
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        # Prepend system prompt as the first message (OpenAI style)
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            max_tokens=1024,
            messages=full_messages,
        )

        answer_text = response.choices[0].message.content

        # De-duplicate sources for the response
        seen = set()
        sources = []
        for chunk in retrieved:
            key = (chunk["source"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": chunk["source"],
                    "chunk": chunk["chunk_index"],
                    "score": round(chunk["score"], 3),
                    "preview": chunk["text"][:150] + "...",
                })

        return answer_text, sources, len(retrieved)

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        return len(self.chunks)

    def list_documents(self) -> List[str]:
        return self.doc_names

    def clear(self):
        self.index = None
        self.chunks = []
        self.doc_names = []
