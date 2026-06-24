import faiss
import numpy as np
import uuid
import os
import json
import requests

store = {}
STORAGE_DIR = "../uploads/vector_store"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Free, serverless embedding via Hugging Face Inference API
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
# No token needed for low-frequency hobby usage, or use your HF token if you have one
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"} if os.getenv('HF_TOKEN') else {}

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Fetch embeddings from cloud API to save server RAM."""
    try:
        response = requests.post(HF_API_URL, json={"inputs": texts, "options": {"wait_for_model": True}}, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Hugging Face API Error: {response.text}")
        embeddings = np.array(response.json(), dtype=np.float32)
        return embeddings
    except Exception as e:
        print(f"[embedder] API Fallback error: {e}")
        # Tiny zero-vector fallback so it doesn't hard-crash if API spikes
        return np.zeros((len(texts), 384), dtype=np.float32)

def build_index(chunks: list[dict], doc_name: str, doc_id: str = None) -> str:
    if not doc_id:
        doc_id = str(uuid.uuid4())[:8]

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    store[doc_id] = {
        "chunks": chunks,
        "index": index,
        "doc_name": doc_name
    }

    index_file_path = os.path.join(STORAGE_DIR, f"{doc_id}.faiss")
    faiss.write_index(index, index_file_path)

    meta_file_path = os.path.join(STORAGE_DIR, f"{doc_id}.json")
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump({"doc_name": doc_name, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    return doc_id

def load_persisted_indices():
    if not os.path.exists(STORAGE_DIR):
        return
    files = [f for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]
    for file in files:
        doc_id = file.replace(".json", "")
        meta_path = os.path.join(STORAGE_DIR, file)
        index_path = os.path.join(STORAGE_DIR, f"{doc_id}.faiss")
        if os.path.exists(index_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                loaded_index = faiss.read_index(index_path)
                store[doc_id] = {
                    "chunks": meta_data["chunks"],
                    "index": loaded_index,
                    "doc_name": meta_data["doc_name"]
                }
            except Exception as e:
                print(f"[embedder] Restoring broken index failed: {e}")

def search(query: str, doc_id: str = None, top_k: int = 3) -> list[dict]:
    if not store:
        load_persisted_indices()
        if not store:
            return []

    query_vec = get_embeddings([query])
    faiss.normalize_L2(query_vec)
    results = []

    targets = {doc_id: store[doc_id]} if (doc_id and doc_id in store) else store

    for did, data in targets.items():
        actual_k = min(top_k, len(data["chunks"]))
        if actual_k == 0: continue

        scores, indices = data["index"].search(query_vec, actual_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            chunk = data["chunks"][idx]
            results.append({
                "index": chunk["index"],
                "text": chunk["text"],
                "score": float(score),
                "doc_name": data["doc_name"],
                "doc_id": did
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

def remove_doc(doc_id: str):
    if doc_id in store: del store[doc_id]
    for ext in [".faiss", ".json"]:
        p = os.path.join(STORAGE_DIR, f"{doc_id}{ext}")
        if os.path.exists(p): os.remove(p)

def get_all_chunks() -> list[dict]:
    all_chunks = []
    for data in store.values(): all_chunks.extend(data["chunks"])
    return all_chunks

def get_doc_count() -> int:
    return len(store)