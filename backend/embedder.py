import faiss
import numpy as np
import uuid
import os
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Multi-document in-memory layout store mapping
store = {}
STORAGE_DIR = "../uploads/vector_store"
os.makedirs(STORAGE_DIR, exist_ok=True)


def build_index(chunks: list[dict], doc_name: str, doc_id: str = None) -> str:
    """Build a flat IP FAISS index and commit both chunks and vectors to local disk."""
    if not doc_id:
        doc_id = str(uuid.uuid4())[:8]

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save to local active runtime state dictionary
    store[doc_id] = {
        "chunks": chunks,
        "index": index,
        "doc_name": doc_name
    }

    # PERSIST TO DISK: Save the raw mathematical index file binary
    index_file_path = os.path.join(STORAGE_DIR, f"{doc_id}.faiss")
    faiss.write_index(index, index_file_path)

    # PERSIST TO DISK: Save the structural raw text chunks array mapping references
    meta_file_path = os.path.join(STORAGE_DIR, f"{doc_id}.json")
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump({"doc_name": doc_name, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    print(f"[embedder] Serialized and locked '{doc_name}' to disk as doc_id={doc_id}")
    return doc_id


def load_persisted_indices():
    """Scan local workspace disk registers and load indexes back into RAM cache on reboot."""
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
                print(f"[embedder] Restored tracking link state for doc_id={doc_id} ({meta_data['doc_name']})")
            except Exception as e:
                print(f"[embedder] Refusing broken index parsing sequence for {doc_id}: {e}")


def search(query: str, doc_id: str = None, top_k: int = 3) -> list[dict]:
    """Search across a specific document or pool balanced chunks from all active documents."""
    if not store:
        # Gracefully handle cold queries by attempting an inline reload scan check first
        load_persisted_indices()
        if not store:
            raise ValueError("No active indices are currently present in your workspace storage mapping pipelines.")

    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    results = []

    if doc_id and doc_id in store:
        targets = {doc_id: store[doc_id]}
        k_per_doc = top_k
    else:
        targets = store
        k_per_doc = max(2, top_k)

    for did, data in targets.items():
        actual_k = min(k_per_doc, len(data["chunks"]))
        if actual_k == 0:
            continue

        scores, indices = data["index"].search(query_vec, actual_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
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
    """Wipe tracing cache records from both RAM registry layouts and matching disk elements."""
    if doc_id in store:
        del store[doc_id]
    
    # Remove file vectors from local system disk safely
    for ext in [".faiss", ".json"]:
        p = os.path.join(STORAGE_DIR, f"{doc_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
    print(f"[embedder] Erased disk file vectors matching signature tracking target: {doc_id}")


def get_all_chunks() -> list[dict]:
    all_chunks = []
    for data in store.values():
        all_chunks.extend(data["chunks"])
    return all_chunks

def get_doc_count() -> int:
    return len(store)