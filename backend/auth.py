import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from database import get_db

SECRET = "documind-secret-key-change-in-production"
ALGORITHM = "HS256"

def create_user(username: str, password: str) -> bool:
    db = get_db()
    try:
        existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            return False
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (username, hashed))
        db.commit()
        return True
    finally:
        db.close()

def login_user(username: str, password: str) -> str | None:
    db = get_db()
    try:
        row = db.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            return None
        db_hash = row["password_hash"]
        if isinstance(db_hash, str):
            db_hash = db_hash.encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), db_hash):
            return None
        exp_time = datetime.now(timezone.utc) + timedelta(days=7)
        token = jwt.encode(
            {"user_id": int(row["id"]), "username": str(row["username"]), "exp": exp_time},
            SECRET, 
            algorithm=ALGORITHM
        )
        return token
    finally:
        db.close()

def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None