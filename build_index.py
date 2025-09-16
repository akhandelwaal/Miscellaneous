#!/usr/bin/env python3
"""
Build a local FAISS index for RAG:
- Loads PDF/DOCX/TXT/MD from DOCS_DIR
- Chunks text
- Embeds with sentence-transformers/all-MiniLM-L6-v2
- Saves FAISS index + metadata to INDEX_DIR

Configure paths in the CONFIG section below.
"""

import os
import re
import glob
import pickle
from typing import List, Tuple

# ========== CONFIG ==========
DOCS_DIR = "./docs"               # folder with your documents
INDEX_DIR = "./index_store"       # where FAISS + metadata will be stored
CHUNK_TOKENS = 256                # target chunk "tokens" (heuristic)
CHUNK_OVERLAP_TOKENS = 40         # overlap "tokens" between chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
# ============================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _walk_documents(doc_dir: str) -> List[str]:
    exts = ["**/*.pdf", "**/*.docx", "**/*.txt", "**/*.md"]
    paths = []
    for pattern in exts:
        paths.extend(glob.glob(os.path.join(doc_dir, pattern), recursive=True))
    # de-dup + sort for stable order
    return sorted(list(set(paths)))

def _load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader = PdfReader(path)
            parts = []
            for p in reader.pages:
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts)
        elif ext == ".docx":
            try:
                import docx
            except ImportError as e:
                raise RuntimeError("Please `pip install python-docx` to read .docx") from e
            d = docx.Document(path)
            return "\n".join(p.text for p in d.paragraphs)
        elif ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # simple sentence boundary on .?! followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _chunk_text(text: str, target_chunk_tokens: int, overlap_tokens: int) -> List[str]:
    def est_tokens(s: str) -> int:
        # crude estimate: ~0.75 * words
        return max(1, int(0.75 * max(1, len(s.split()))))

    sents = _split_sentences(text)
    if not sents:
        return []

    chunks, curr, curr_toks = [], [], 0
    for s in sents:
        t = est_tokens(s)
        if curr and curr_toks + t > target_chunk_tokens:
            chunks.append(" ".join(curr).strip())
            if overlap_tokens > 0:
                back, acc = [], 0
                for prev in reversed(curr):
                    tt = est_tokens(prev)
                    back.append(prev)
                    acc += tt
                    if acc >= overlap_tokens:
                        break
                curr = list(reversed(back))
                curr_toks = sum(est_tokens(x) for x in curr)
            else:
                curr, curr_toks = [], 0
        curr.append(s)
        curr_toks += t

    if curr:
        chunks.append(" ".join(curr).strip())

    # filter tiny fragments
    return [c for c in chunks if len(c.split()) >= 5]

class _Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True):
        import numpy as np
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=normalize
        )
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs.astype("float32")

class _FaissStore:
    def __init__(self, dim: int, index_dir: str):
        import faiss
        self.dim = dim
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "vectors.faiss")
        self.meta_path = os.path.join(index_dir, "meta.pkl")
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.id_map: List[Tuple[str, int]] = []
        self.index = faiss.IndexFlatIP(dim)  # cosine via inner product (with normalized vectors)

    def add(self, vectors, texts: List[str], sources: List[str], id_pairs: List[Tuple[str, int]]):
        self.index.add(vectors)
        self.texts.extend(texts)
        self.sources.extend(sources)
        self.id_map.extend(id_pairs)

    def save(self):
        import faiss
        _ensure_dir(self.index_dir)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "texts": self.texts,
                    "sources": self.sources,
                    "id_map": self.id_map,
                },
                f,
            )

def build_index(
    docs_dir: str = DOCS_DIR,
    index_dir: str = INDEX_DIR,
    chunk_tokens: int = CHUNK_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    embed_model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> int:
    paths = _walk_documents(docs_dir)
    if not paths:
        print(f"[WARN] No documents found in {docs_dir}")
        return 0

    all_chunks, all_sources, id_pairs = [], [], []
    print(f"[INFO] Found {len(paths)} files. Chunking...")
    doc_id = 0
    for p in paths:
        txt = _load_text_from_file(p)
        if not txt.strip():
            continue
        chunks = _chunk_text(txt, chunk_tokens, chunk_overlap_tokens)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_sources.append(p)
            id_pairs.append((f"doc{doc_id}", i))
        doc_id += 1

    print(f"[INFO] Total chunks: {len(all_chunks)}. Embedding with {embed_model} ...")
    emb = _Embedder(embed_model)
    vecs = emb.encode(all_chunks, batch_size=batch_size, normalize=True)

    print(f"[INFO] Building FAISS index @ {index_dir}")
    store = _FaissStore(emb.dim, index_dir=index_dir)
    store.add(vecs, all_chunks, all_sources, id_pairs)
    store.save()
    print(f"[OK] Saved index: {index_dir}")
    return len(all_chunks)

# When run directly (you can also import build_index from your UI code)
if __name__ == "__main__":
    build_index()
