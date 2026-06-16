# FAQ Chatbot with RAG

A retrieval-augmented FAQ chatbot. Upload company documents (PDF/DOCX/TXT), ask questions in natural language, get grounded answers with source citations.

## Architecture

```
Documents → Chunker (400 tokens, 60 overlap)
                ↓
          OpenAI Embeddings (text-embedding-3-small, 1536-dim)
                ↓
          FAISS Vector Store (cosine similarity)
                ↓
User Query → Embed → Top-5 Chunks → Claude claude-sonnet-4-6 → Answer + Sources
```

## Project Structure

```
faq-chatbot/
├── backend/
│   ├── main.py          # FastAPI routes
│   ├── rag_engine.py    # Core RAG logic
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── railway.toml     # Railway deploy config
│   └── .env.example
└── frontend/
    ├── index.html       # React SPA (no build step)
    └── vercel.json      # Vercel deploy config
```

## Local Setup

### 1. Clone and set up backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create your .env file

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Open the frontend

Just open `frontend/index.html` in your browser. It auto-detects localhost.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + doc count |
| POST | `/upload` | Upload and ingest documents |
| POST | `/chat` | Ask a question |
| GET | `/documents` | List ingested documents |
| DELETE | `/documents` | Clear all documents |

### Chat request body
```json
{
  "question": "What is the refund policy?",
  "chat_history": []
}
```

### Chat response
```json
{
  "answer": "According to the policy document...",
  "sources": [
    {
      "file": "policy.pdf",
      "chunk": 3,
      "score": 0.891,
      "preview": "Refunds are processed within 5-7 business days..."
    }
  ],
  "chunks_used": 5
}
```

---

## Deploy to Production

### Backend → Railway (free tier)

1. Push `backend/` to a GitHub repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables in Railway dashboard:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
5. Railway auto-deploys on push. Copy your `.railway.app` URL.

### Frontend → Vercel (free tier)

1. Push `frontend/` to GitHub (can be same repo)
2. Go to [vercel.com](https://vercel.com) → New Project → Import repo
3. Set root directory to `frontend/`
4. **Edit `frontend/index.html`** — replace the `API_BASE` URL:
   ```js
   const API_BASE = window.location.hostname === "localhost"
     ? "http://localhost:8000"
     : "https://YOUR-APP.railway.app";  // ← your Railway URL
   ```
5. Deploy. Done.

---

## How It Works (Key Concepts)

### Chunking
Text is split into 400-token chunks with 60-token overlap. Overlap ensures sentences that span chunk boundaries are still findable.

### Embeddings
Each chunk is converted to a 1536-dimensional vector using OpenAI's `text-embedding-3-small`. Semantically similar text → similar vectors.

### FAISS (IndexFlatIP)
Stores all vectors. On each query, does exact inner-product (= cosine similarity after L2 normalisation) search in O(n) time. Fast enough for thousands of chunks without approximate methods.

### Retrieval
Top-5 most similar chunks are retrieved. Score threshold isn't applied — the LLM decides relevance from context.

### Generation
Claude receives a system prompt containing the retrieved chunks and is instructed to answer ONLY from that context. If the answer isn't there, it says so — no hallucination.

---

## Tuning Parameters (in rag_engine.py)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CHUNK_SIZE` | 400 tokens | Larger = more context per chunk, but less precise retrieval |
| `CHUNK_OVERLAP` | 60 tokens | Larger = fewer missed boundaries, more redundancy |
| `TOP_K` | 5 | More chunks = more context, higher cost |
| `EMBED_MODEL` | text-embedding-3-small | Switch to `text-embedding-3-large` for better quality |

## Scaling Up

- **Persistent storage**: Save FAISS index to disk (`faiss.write_index`) so documents survive restarts
- **Auth**: Add API key middleware to FastAPI
- **Multi-tenant**: Namespace the FAISS index per user/organisation
- **Better chunking**: Use semantic chunking (split on sentence boundaries) instead of token count
- **Reranking**: Add a cross-encoder reranker after FAISS retrieval for better precision
