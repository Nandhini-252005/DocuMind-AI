import sqlite3
import json
from datetime import datetime

DB_PATH = "../documind.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            doc_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            summary TEXT,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            doc_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    db.commit()
    db.close()
    print("[database] Tables initialized")

def save_document(user_id: int, doc_id: str, filename: str, summary: str):
    db = get_db()
    db.execute(
        "INSERT INTO documents (user_id, doc_id, filename, summary) VALUES (?,?,?,?)",
        (user_id, doc_id, filename, summary)
    )
    db.commit()
    db.close()

def get_user_documents(user_id: int) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT doc_id, filename, summary, uploaded_at FROM documents WHERE user_id=? ORDER BY uploaded_at DESC",
        (user_id,)
    ).fetchall()
    result = [dict(r) for r in rows]
    db.close()
    return result

def delete_document_db(user_id: int, doc_id: str):
    db = get_db()
    db.execute("DELETE FROM documents WHERE user_id=? AND doc_id=?", (user_id, doc_id))
    db.commit()
    db.close()

def save_chat(user_id: int, doc_id: str, question: str, answer: str):
    db = get_db()
    db.execute(
        "INSERT INTO chat_history (user_id, doc_id, question, answer) VALUES (?,?,?,?)",
        (user_id, doc_id, question, answer)
    )
    db.commit()
    db.close()

def get_chat_history(user_id: int, limit: int = 50) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT doc_id, question, answer, created_at FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    result = [dict(r) for r in rows]
    db.close()
    return result

def clear_chat_history(user_id: int):
    db = get_db()
    db.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
    db.commit()
    db.close()