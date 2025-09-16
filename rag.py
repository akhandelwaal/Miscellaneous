#!/usr/bin/env python3
# pip install sentence-transformers faiss-cpu llama-cpp-python pypdf python-docx
# (or PyPDF2 if pypdf fails)
"""
RAG pipeline: parse -> chunk -> embed (all-MiniLM-L6-v2) -> store (FAISS) -> retrieve -> generate (llama.cpp)

Usage examples:

# 1) Build/update the vector store from a folder of docs:
python rag_local_llama.py build --docs ./docs --index_dir ./index_store

# 2) Ask a question (retrieval + local LLM answer):
python rag_local_llama.py ask \
  --index_dir ./index_store \
  --model_path /path/to/your/model.gguf \
  --question "What are the key points about X?"

# 3) Interactive chat over your corpus:
python rag_local_llama.py chat \
  --index_dir ./index_store \
  --model_path /path/to/your/model.gguf
"""

import os
import re
import sys
import json
import math
import faiss
import time
import glob
import pickle
import argparse
from typing import List, Dict, Tuple

# -----------------------------
# Document loading (PDF/DOCX/TXT/MD)
# -----------------------------
def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader  # fallback
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
        elif ext in (".docx",):
            try:
                import docx  # python-docx
            except ImportError as e:
                raise RuntimeError("Please `pip install python-docx` to read .docx files") from e
            d = docx.Document(path)
            return "\n".join([p.text for p in d.paragraphs])
        elif ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
        return ""

def walk_documents(doc_dir: str) -> List[str]:
    exts = ["**/*.pdf", "**/*.docx", "**/*.txt", "**/*.md"]
    paths = []
    for pattern in exts:
        paths.extend(glob.glob(os.path.join(doc_dir, pattern), recursive=True))
    return sorted(list(set(paths)))

# -----------------------------
# Chunking
# -----------------------------
def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter (avoid heavy deps); good enough for most prose.
    # Splits on ., ?, ! followed by space/newline. Keeps delimiter.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(
    text: str,
    target_chunk_tokens: int = 256,
    overlap_tokens: int = 40,
) -> List[str]:
    """
    Token-agnostic heuristic chunker (works well with MiniLM).
    We guesstimate tokens as ~0.75 * words (roughly correct for English).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    curr = []
    curr_tokens = 0

    def est_tokens(s: str) -> int:
        # crude estimate: tokens ~ 0.75 * words
        return max(1, int(0.75 * max(1, len(s.split()))))

    for sent in sentences:
        s_tok = est_tokens(sent)
        if curr_tokens + s_tok > target_chunk_tokens and curr:
            chunks.append(" ".join(curr).strip())
            # build overlap by rolling window on already added sentences
            if overlap_tokens > 0:
                # slide from the end to keep approx overlap
                back_tokens = 0
                overlap = []
                for s in reversed(curr):
                    t = est_tokens(s)
                    overlap.append(s)
                    back_tokens += t
                    if back_tokens >= overlap_tokens:
                        break
                curr = list(reversed(overlap))
                curr_tokens = sum(est_tokens(s) for s in curr)
            else:
                curr = []
                curr_tokens = 0

        curr.append(sent)
        curr_tokens += s_tok

    if curr:
        chunks.append(" ".join(curr).strip())

    # Filter very short chunks
    chunks = [c for c in chunks if len(c.split()) >= 5]
    return chunks

# -----------------------------
# Embeddings (SentenceTransformers)
# -----------------------------
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True):
        import numpy as np
        vecs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
        if normalize:
            # Ensure unit norm (cosine sim via inner product)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs

# -----------------------------
# FAISS Store (cosine via inner product)
# -----------------------------
class FaissStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "vectors.faiss")
        self.meta_path = os.path.join(index_dir, "meta.pkl")
        self.id_map: List[Tuple[str, int]] = []  # (doc_id, chunk_idx)
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.index = faiss.IndexFlatIP(dim)  # use inner product (expects normalized vectors)

    def _ensure_dir(self):
        os.makedirs(self.index_dir, exist_ok=True)

    def add(self, vectors, texts: List[str], sources: List[str], id_pairs: List[Tuple[str, int]]):
        import numpy as np
        assert vectors.shape[0] == len(texts) == len(sources) == len(id_pairs)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype="float32")
        if vectors.dtype != "float32":
            vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)
        self.sources.extend(sources)
        self.id_map.extend(id_pairs)

    def save(self):
        self._ensure_dir()
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "id_map": self.id_map,
                    "texts": self.texts,
                    "sources": self.sources,
                    "dim": self.dim,
                },
                f,
            )

    def load(self):
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            raise FileNotFoundError("No existing index found. Build it first.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
        self.id_map = data["id_map"]
        self.texts = data["texts"]
        self.sources = data["sources"]
        self.dim = data["dim"]

    def search(self, query_vec, k: int = 5):
        import numpy as np
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != "float32":
            query_vec = query_vec.astype("float32")
        scores, idxs = self.index.search(query_vec, k)
        return scores[0].tolist(), idxs[0].tolist()

# -----------------------------
# Prompting & Generation (llama-cpp-python)
# -----------------------------
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using the provided context. "
    "If the answer is not in the context, say you don't know. Be concise and cite source filenames when relevant."
)

def format_prompt(system_msg: str, question: str, contexts: List[Tuple[str, str]], max_ctx_chars: int = 8000) -> str:
    """
    contexts: list of (source_path, text)
    """
    ctx_blocks = []
    total = 0
    for src, txt in contexts:
        block = f"[Source: {os.path.basename(src)}]\n{txt}\n"
        if total + len(block) > max_ctx_chars:
            break
        ctx_blocks.append(block)
        total += len(block)

    context_str = "\n---\n".join(ctx_blocks) if ctx_blocks else "No retrieved context."
    user_block = f"Question: {question}\n\nUse the context above to answer. If uncertain, say so."
    return f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n[CONTEXT]\n{context_str}\n\n[USER]\n{user_block}"

class LocalLLM:
    """
    Wrapper around llama-cpp-python. Install:
      pip install llama-cpp-python
    Provide a quantized GGUF model path, e.g.:
      - Meta Llama 3 8B Instruct Q4_K_M GGUF
      - Mistral 7B Instruct Q4_K_M GGUF
      - Qwen 2 7B Instruct Q4_K_M GGUF

    Notes on quantization:
      Q2_K (smallest, fastest, least accurate)
      Q4_K_M (great speed/quality tradeoff)
      Q5_K_M (better quality, needs more RAM)
      Q8_0 (near-FP16 quality, heavy)
    """
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = max(2, os.cpu_count() or 4),
        n_batch: int = 512,
        n_gpu_layers: int = 0,  # keep 0 for pure CPU
        verbose: bool = False,
    ):
        from llama_cpp import Llama
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def generate(self, prompt: str, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 512):
        out = self.llm(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</s>", "###", "[/USER]"],
        )
        return out["choices"][0]["text"].strip()

# -----------------------------
# RAG Orchestrator
# -----------------------------
class RAGPipeline:
    def __init__(self, index_dir: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.embedder = Embedder(embed_model, device=device)
        self.store = FaissStore(dim=self.embedder.dim, index_dir=index_dir)

    def build_from_dir(
        self,
        doc_dir: str,
        chunk_tokens: int = 256,
        chunk_overlap: int = 40,
        batch_size: int = 32,
    ):
        print(f"[INFO] Scanning documents in: {doc_dir}")
        paths = walk_documents(doc_dir)
        if not paths:
            print("[WARN] No documents found.")
            return

        all_chunks = []
        all_sources = []
        all_idpairs = []
        doc_id = 0
        for p in paths:
            text = load_text_from_file(p)
            if not text.strip():
                continue
            chunks = chunk_text(text, target_chunk_tokens=chunk_tokens, overlap_tokens=chunk_overlap)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                all_sources.append(p)
                all_idpairs.append((f"doc{doc_id}", i))
            doc_id += 1

        print(f"[INFO] Total chunks: {len(all_chunks)} (embedding ...)")
        # Embed in batches
        vectors = self.embedder.encode(all_chunks, batch_size=batch_size, normalize=True)
        self.store.add(vectors, texts=all_chunks, sources=all_sources, id_pairs=all_idpairs)
        self.store.save()
        print(f"[OK] Index saved to {self.store.index_dir}. Chunks indexed: {len(all_chunks)}.")

    def _ensure_loaded(self):
        if not self.store.texts:
            self.store.load()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str, str]]:
        """
        Returns list of (score, source, text) sorted by score desc
        """
        import numpy as np
        self._ensure_loaded()
        qvec = self.embedder.encode([query], batch_size=1, normalize=True)[0]
        scores, idxs = self.store.search(qvec, k=top_k)
        results = []
        for s, i in zip(scores, idxs):
            if i == -1:
                continue
            results.append((float(s), self.store.sources[i], self.store.texts[i]))
        return results

    def answer(
        self,
        question: str,
        model_path: str,
        system_prompt: str = DEFAULT_SYSTEM,
        top_k: int = 5,
        max_ctx_chars: int = 8000,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> Dict:
        self._ensure_loaded()
        hits = self.retrieve(question, top_k=top_k)
        contexts = [(src, txt) for (_s, src, txt) in hits]
        prompt = format_prompt(system_prompt, question, contexts, max_ctx_chars=max_ctx_chars)

        llm = LocalLLM(model_path=model_path)
        answer = llm.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        return {
            "question": question,
            "answer": answer,
            "contexts": [{"score": round(hits[i][0], 4), "source": contexts[i][0], "chunk_preview": contexts[i][1][:400]} for i in range(len(contexts))],
        }

# -----------------------------
# CLI
# -----------------------------
def build_cmd(args):
    rag = RAGPipeline(index_dir=args.index_dir, embed_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    rag.build_from_dir(
        doc_dir=args.docs,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )

def ask_cmd(args):
    rag = RAGPipeline(index_dir=args.index_dir, embed_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out = rag.answer(
        question=args.question,
        model_path=args.model_path,
        top_k=args.top_k,
        max_ctx_chars=args.max_ctx_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== TOP CONTEXTS ===\n")
    for c in out["contexts"]:
        print(f"[{c['score']}] {c['source']}\n{c['chunk_preview']}\n---")

def chat_cmd(args):
    rag = RAGPipeline(index_dir=args.index_dir, embed_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("Entering chat mode. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if q.lower() in ("exit", "quit"):
            break
        out = rag.answer(
            question=q,
            model_path=args.model_path,
            top_k=args.top_k,
            max_ctx_chars=args.max_ctx_chars,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        print("\nAssistant:\n" + out["answer"] + "\n")

def main():
    parser = argparse.ArgumentParser(description="Simple local-CPU RAG with MiniLM embeddings + llama.cpp generation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Parse, chunk, embed, and index documents")
    p_build.add_argument("--docs", required=True, help="Folder containing documents (pdf/docx/txt/md)")
    p_build.add_argument("--index_dir", required=True, help="Where to save the FAISS index & metadata")
    p_build.add_argument("--chunk_tokens", type=int, default=256)
    p_build.add_argument("--chunk_overlap", type=int, default=40)
    p_build.add_argument("--batch_size", type=int, default=32)
    p_build.set_defaults(func=build_cmd)

    p_ask = sub.add_parser("ask", help="Ask a single question (retrieval + local LLM answer)")
    p_ask.add_argument("--index_dir", required=True)
    p_ask.add_argument("--model_path", required=True, help="Path to quantized GGUF model (e.g., llama-3-8b-instruct.Q4_K_M.gguf)")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top_k", type=int, default=5)
    p_ask.add_argument("--max_ctx_chars", type=int, default=8000)
    p_ask.add_argument("--temperature", type=float, default=0.2)
    p_ask.add_argument("--top_p", type=float, default=0.9)
    p_ask.add_argument("--max_tokens", type=int, default=512)
    p_ask.set_defaults(func=ask_cmd)

    p_chat = sub.add_parser("chat", help="Interactive chat over your corpus")
    p_chat.add_argument("--index_dir", required=True)
    p_chat.add_argument("--model_path", required=True)
    p_chat.add_argument("--top_k", type=int, default=5)
    p_chat.add_argument("--max_ctx_chars", type=int, default=8000)
    p_chat.add_argument("--temperature", type=float, default=0.2)
    p_chat.add_argument("--top_p", type=float, default=0.9)
    p_chat.add_argument("--max_tokens", type=int, default=512)
    p_chat.set_defaults(func=chat_cmd)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
