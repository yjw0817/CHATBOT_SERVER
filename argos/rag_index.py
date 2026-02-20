"""Embedding-based RAG index using FAISS + bge-m3 (Ollama)."""
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import requests

from database import get_connection
from llm_client import get_llm_mode

# --- Config ---
FAISS_DIR = Path(__file__).parent / "faiss_data"
# Auto-detected dimension (set on first embedding call)
_embed_dim: int | None = None

# --- In-memory cache ---
# {apt_id: {"index": faiss.Index, "chunk_ids": [...], "mtime": float}}
_index_cache: dict = {}
# {apt_id: {chunk_id: {chunk detail dict}}}
_chunk_cache: dict = {}


def _get_embed_config():
    """Return (base_url, api_key, model) for embedding API."""
    mode = get_llm_mode()
    if mode == "local":
        base_url = os.getenv("LLM_LOCAL_URL", "http://127.0.0.1:11434")
        api_key = ""
    else:
        base_url = os.getenv("LLM_REMOTE_URL", os.getenv("LLM_BASE_URL", ""))
        api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("EMBED_MODEL", "bge-m3")
    return base_url.rstrip("/"), api_key, model


# ── Embedding helpers ──────────────────────────────────────────────


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a single text via Ollama /api/embeddings."""
    base_url, api_key, model = _get_embed_config()
    url = f"{base_url}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        url,
        json={"model": model, "prompt": text},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama /api/embeddings returns {"embedding": [float, ...]}
    embedding = data["embedding"]
    # Auto-detect dimension on first call
    global _embed_dim
    if _embed_dim is None:
        _embed_dim = len(embedding)
        print(f"[RAG_INDEX] Auto-detected embed dim={_embed_dim} for model={model}")
    return embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts (sequential, Ollama has no batch API)."""
    results = []
    for i, text in enumerate(texts):
        results.append(get_embedding(text))
        if (i + 1) % 50 == 0:
            print(f"[RAG_INDEX] Embedded {i + 1}/{len(texts)} chunks")
    return results


# ── FAISS index management ─────────────────────────────────────────


def _index_path(apt_id: str) -> Path:
    return FAISS_DIR / f"{apt_id}.index"


def _meta_path(apt_id: str) -> Path:
    return FAISS_DIR / f"{apt_id}.meta.json"


def _load_index(apt_id: str):
    """Load FAISS index + meta from cache or disk. Returns (index, chunk_ids) or (None, None)."""
    idx_path = _index_path(apt_id)
    meta_path = _meta_path(apt_id)

    if not idx_path.exists() or not meta_path.exists():
        return None, None

    disk_mtime = idx_path.stat().st_mtime

    # Check cache
    cached = _index_cache.get(apt_id)
    if cached and cached["mtime"] >= disk_mtime:
        return cached["index"], cached["chunk_ids"]

    # Load from disk
    t0 = time.time()
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunk_ids = meta["chunk_ids"]

    # Cache it
    _index_cache[apt_id] = {"index": index, "chunk_ids": chunk_ids, "mtime": disk_mtime}
    print(f"[RAG_INDEX] Loaded index for {apt_id} from disk ({index.ntotal} vectors, {round(time.time()-t0, 3)}s)")

    return index, chunk_ids


def _load_chunk_details(apt_id: str, chunk_ids: list[str]) -> dict:
    """Load chunk details from cache or DB. Returns {chunk_id: dict}."""
    if apt_id in _chunk_cache:
        cached = _chunk_cache[apt_id]
        # Check if all requested chunk_ids are in cache
        if all(cid in cached for cid in chunk_ids):
            return cached

    # Load all chunks for this apt_id at once
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.chunk_id, c.doc_id, c.section_name, c.chunk_text, d.title as doc_title
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE d.apt_id = %s AND d.status = 'APPROVED'
    """, (apt_id,))
    rows = {r["chunk_id"]: dict(r) for r in cursor.fetchall()}
    conn.close()

    _chunk_cache[apt_id] = rows
    return rows


def build_index(apt_id: str) -> dict:
    """Build FAISS index from APPROVED chunks for an apt_id.

    - Generates embeddings for chunks missing them, saves to DB.
    - Writes .index + .meta.json files.
    Returns stats dict.
    """
    t0 = time.time()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.chunk_id, c.chunk_text, c.embedding
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE d.apt_id = %s AND d.status = 'APPROVED'
    """, (apt_id,))
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        invalidate_index(apt_id)
        return {"chunk_count": 0, "embedded": 0, "time_s": 0}

    chunk_ids = []
    vectors = []
    newly_embedded = 0

    for i, row in enumerate(rows):
        chunk_id = row["chunk_id"]
        chunk_text = row["chunk_text"]
        emb_blob = row["embedding"]

        if emb_blob:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
        else:
            emb = get_embedding(chunk_text)
            vec = np.array(emb, dtype=np.float32)
            # Save embedding to DB
            cursor.execute(
                "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
                (vec.tobytes(), chunk_id),
            )
            newly_embedded += 1
            if newly_embedded % 50 == 0:
                print(f"[RAG_INDEX] Embedded {newly_embedded} new chunks...")

        # L2 normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        chunk_ids.append(chunk_id)
        vectors.append(vec)

    conn.commit()
    conn.close()

    # Build FAISS index (Inner Product = cosine similarity after normalization)
    matrix = np.stack(vectors).astype(np.float32)
    dim = _embed_dim or matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save to disk
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_index_path(apt_id)))
    with open(_meta_path(apt_id), "w", encoding="utf-8") as f:
        json.dump({"chunk_ids": chunk_ids}, f)

    # Update in-memory cache
    _index_cache[apt_id] = {
        "index": index,
        "chunk_ids": chunk_ids,
        "mtime": _index_path(apt_id).stat().st_mtime,
    }
    _chunk_cache.pop(apt_id, None)  # Invalidate chunk cache

    elapsed = round(time.time() - t0, 2)
    print(f"[RAG_INDEX] Built index for apt_id={apt_id}: {len(chunk_ids)} chunks, {newly_embedded} newly embedded, {elapsed}s")
    return {"chunk_count": len(chunk_ids), "embedded": newly_embedded, "time_s": elapsed}


def search(apt_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search FAISS index for top_k chunks matching query.

    Returns list of dicts with chunk_id, doc_id, section_name, chunk_text, doc_title, score.
    """
    t_total = time.time()

    # Load index (from cache or disk)
    index, chunk_ids = _load_index(apt_id)

    # Build index if not loaded
    if index is None:
        print(f"[RAG_INDEX] No index for {apt_id}, building...")
        build_index(apt_id)
        index, chunk_ids = _load_index(apt_id)
    if index is None or index.ntotal == 0:
        return []

    # Query embedding
    t_emb = time.time()
    q_emb = np.array(get_embedding(query), dtype=np.float32)
    norm = np.linalg.norm(q_emb)
    if norm > 0:
        q_emb = q_emb / norm
    q_emb = q_emb.reshape(1, -1)
    t_emb = round(time.time() - t_emb, 3)

    # FAISS search
    t_search = time.time()
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_emb, k)
    t_search = round(time.time() - t_search, 3)

    # Get chunk details (from cache)
    hit_ids = [chunk_ids[idx] for idx in indices[0] if idx >= 0]
    if not hit_ids:
        return []

    all_details = _load_chunk_details(apt_id, hit_ids)

    # Return in score order
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        cid = chunk_ids[idx]
        if cid in all_details:
            row = dict(all_details[cid])  # copy to avoid mutating cache
            row["score"] = float(scores[0][i])
            results.append(row)

    t_total = round(time.time() - t_total, 3)
    print(f"[RAG_INDEX] search({apt_id}): embed={t_emb}s faiss={t_search}s total={t_total}s hits={len(results)}")

    return results


def invalidate_index(apt_id: str):
    """Delete index files + clear cache to trigger rebuild on next search."""
    for p in (_index_path(apt_id), _meta_path(apt_id)):
        if p.exists():
            p.unlink()
    _index_cache.pop(apt_id, None)
    _chunk_cache.pop(apt_id, None)
    print(f"[RAG_INDEX] Invalidated index for apt_id={apt_id}")
