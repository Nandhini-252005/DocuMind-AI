# DocuMind AI

DocuMind AI is a lightweight, self-hosted document question-answering assistant. Upload PDFs, it indexes them with semantic embeddings, and you can ask natural-language questions about the content. The project also auto-generates summaries and quizzes from uploaded documents.

## Repository layout

- backend/ — FastAPI server and processing logic
  - [backend/main.py](backend/main.py) — FastAPI routes and orchestration
  - [backend/pdf_processor.py](backend/pdf_processor.py) — PDF extraction & chunking
  - [backend/embedder.py](backend/embedder.py) — embeddings + FAISS index management
  - [backend/chat.py](backend/chat.py) — LLM prompts and response generation (Groq client)
  - [backend/auth.py](backend/auth.py) — user creation, login, JWT verification
  - [backend/database.py](backend/database.py) — SQLite persistence for users, docs, history
  - [backend/requirements.txt](backend/requirements.txt)
- frontend/ — single-file client app (HTML/CSS/JS)
  - [frontend/index.html](frontend/index.html)
- uploads/ — uploaded PDF files (created at runtime)
- documind.db — SQLite database (created at runtime in project root)

## Features

- PDF text extraction using `pdfplumber` with PyMuPDF fallback
- Chunking of document text into overlapping segments
- SentenceTransformer embeddings (model `all-MiniLM-L6-v2`) and FAISS vector search
- LLM-powered conversational answers, summaries and quiz generation (Groq API)
- Simple user auth (bcrypt + JWT) and chat/document persistence in SQLite
- Frontend with drag/drop upload, chat UI, voice input, and export to PDF

## Quick start (development)

1. Clone repo and open terminal in project root.

2. Backend: create a virtual environment and install requirements

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
python -m pip install -r requirements.txt
```

3. Provide environment variables (create `.env` or set in your shell):

- `GROQ_API_KEY` — API key for the Groq LLM client used in `backend/chat.py`.
- (Optional) `SECRET` — a replacement JWT secret; by default `auth.py` has a hard-coded secret which is unsafe for production.

4. Run the API server (from `backend`):

```powershell
uvicorn main:app --reload
```

The server defaults to `http://127.0.0.1:8000`.

5. Frontend: serve the `frontend/` folder (simple static server) and open in a browser:

```powershell
cd frontend
python -m http.server 3000
# open http://localhost:3000
```

6. Use the UI: upload a PDF, then ask questions in the chat box. If you sign up/log in, documents and chat history are stored in the SQLite DB.

## Important configuration & runtime notes

- Database path: `documind.db` created at project root by `backend/database.py` (DB_PATH defaults to `../documind.db` from backend folder).
- Uploads folder: `uploads/` is created automatically by `backend/main.py` (UPLOAD_DIR = `../uploads`).
- Vector store is in-memory only (FAISS indices stored in a Python dict). Restarting the backend will clear loaded documents. Consider adding persistence or a vector DB for production.
- The SentenceTransformer model downloads to your machine on first run; embeddings are created in-process and may take time for large PDFs.

## Environment & dependencies

See [backend/requirements.txt](backend/requirements.txt) for the Python dependency list. Notable packages:

- `fastapi`, `uvicorn` — API server
- `pymupdf`, `pdfplumber` — PDF parsing
- `sentence-transformers`, `faiss-cpu` — embeddings + vector search
- `groq` and `python-dotenv` — LLM integration
- `PyJWT`, `bcrypt` — auth

## Security & production recommendations

- Replace the hard-coded JWT secret in `backend/auth.py` with a secure secret loaded from environment.
- Restrict CORS (currently `allow_origins=["*"]`) and add rate limiting.
- Persist FAISS indices to disk or use a managed vector DB (Qdrant, Milvus, Weaviate) so indexes survive restarts.
- Move long-running tasks (embedding/indexing) to a background worker or queue (Celery/RQ) and return job status to the client.
- Add optional OCR (Tesseract) for scanned PDFs.
- Add proper logging, monitoring, and structured error handling before exposing the app publicly.

## Troubleshooting

- If embeddings or FAISS fail, ensure `faiss-cpu` is compatible with your platform and Python version.
- If Groq calls fail, confirm `GROQ_API_KEY` is set and network access is available.
- For empty extraction results, the PDF may be image-only — add OCR preprocessing.

## Suggested next steps

- Persist and reload FAISS indexes on startup/shutdown.
- Add OCR for scanned PDFs and progress reporting for large uploads.
- Add tests for PDF parsing and prompt-to-JSON parsing for quiz generation.

---

If you'd like, I can now:
- Add index persistence and a save/load utility for FAISS, or
- Create a minimal `.env.example` and update `auth.py` to read the JWT secret from env, or
- Generate a short CONTRIBUTORS / developer guide with run/debug tips.

Tell me which you'd like next.