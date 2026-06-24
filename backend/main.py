import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

from pdf_processor import process_pdf
# Make sure load_persisted_indices is imported cleanly
from embedder import build_index, search, get_doc_count, remove_doc, load_persisted_indices
from chat import ask_groq, generate_summary, generate_quiz
from auth import create_user, login_user, verify_token
from database import (
    save_chat, get_chat_history, save_document,
    get_user_documents, init_db, delete_document_db
)

app = FastAPI(title="DocuMind AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
security = HTTPBearer(auto_error=False)


@app.on_event("startup")
async def startup():
    # 1. Initialize SQLite Database Tables layout structure
    init_db()
    # 2. RESTORE STEP: Pull indexes straight from local binary disk blocks back into RAM
    load_persisted_indices()
    print("[startup] Vector store synchronized and aligned successfully.")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    return verify_token(credentials.credentials)

class SignupRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

class QuizRequest(BaseModel):
    num_questions: int = 5

@app.post("/auth/signup")
async def signup(req: SignupRequest):
    if not create_user(req.username, req.password):
        raise HTTPException(status_code=400, detail="Username already exists.")
    return {"message": "Account created successfully!"}

@app.post("/auth/login")
async def login(req: SignupRequest):
    token = login_user(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"token": token, "username": req.username}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks = process_pdf(save_path)
        doc_id = build_index(chunks, file.filename)
        summary = generate_summary(chunks[:10]) if chunks else "No content available."

        if user and "user_id" in user:
            save_document(int(user["user_id"]), doc_id, file.filename, summary)

        return {
            "filename": file.filename,
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(user=Depends(get_current_user)):
    if user and "user_id" in user:
        return {"documents": get_user_documents(int(user["user_id"]))}
    return {"documents": []}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user=Depends(get_current_user)):
    if user and "user_id" in user:
        delete_document_db(int(user["user_id"]), doc_id)
    remove_doc(doc_id)
    return {"message": "Document deleted"}

@app.post("/chat")
async def chat(request: ChatRequest, user=Depends(get_current_user)):
    try:
        relevant_chunks = search(request.question, doc_id=request.doc_id, top_k=3)
        answer = ask_groq(request.question, relevant_chunks)

        if user and "user_id" in user:
            save_chat(int(user["user_id"]), request.doc_id or "unknown", request.question, answer)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(user=Depends(get_current_user)):
    if not user or "user_id" not in user:
        return {"history": []}
    return {"history": get_chat_history(int(user["user_id"]))}

@app.post("/quiz")
async def generate_quiz_route(request: QuizRequest, user=Depends(get_current_user)):
    try:
        from embedder import get_all_chunks
        chunks = get_all_chunks()
        if not chunks:
            raise HTTPException(status_code=400, detail="No document loaded.")
        return {"quiz": generate_quiz(chunks[:15], request.num_questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"doc_count": get_doc_count()}