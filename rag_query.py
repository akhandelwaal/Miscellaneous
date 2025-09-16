#!/usr/bin/env python3
"""
RAG retrieval + local generation:
- Loads FAISS index + metadata from INDEX_DIR
- Retrieves top-K chunks for a question
- Generates with a local quantized GGUF model via llama-cpp-python (CPU)

Configure paths in the CONFIG section below.
Expose `rag_answer(question)` for your UI code.
"""

import os
import pickle
from typing import List, Tuple, Dict

# ========== CONFIG ==========
INDEX_DIR = "./index_store"   # must match build_index.py
MODEL_PATH = "/path/to/your/quantized-model.gguf"  # e.g., llama-3-8b-instruct.Q4_K_M.gguf
TOP_K = 5
MAX_CTX_CHARS = 8000
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 512
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the user. "
    "If the answer is not in the context, say you don't know. Be concise and cite source filenames."
)
# ============================

class _Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True):
        import numpy as np
        vecs = self.model.encode(
            texts, batch_size=1, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=normalize
        )
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs.astype("float32")

class _FaissReader:
    def __init__(self, index_dir: str):
        import faiss
        meta_path = os.path.join(index_dir, "meta.pkl")
        index_path = os.path.join(index_dir, "vectors.faiss")
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(f"Missing index files in {index_dir}. Build them first.")
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.texts: List[str] = data["texts"]
        self.sources: List[str] = data["sources"]
        self.dim: int = data["dim"]
        self._faiss = faiss.read_index(index_path)

    def search(self, qvec, k: int) -> List[Tuple[float, int]]:
        # returns list of (score, idx)
        scores, idxs = self._faiss.search(qvec.reshape(1, -1), k)
        return list(zip(scores[0].tolist(), idxs[0].tolist()))

def _format_prompt(system_msg: str, question: str, contexts: List[Tuple[str, str]], max_ctx_chars: int) -> str:
    blocks, total = [], 0
    for src, txt in contexts:
        b = f"[Source: {os.path.basename(src)}]\n{txt}\n"
        if total + len(b) > max_ctx_chars:
            break
        blocks.append(b)
        total += len(b)
    context_str = "\n---\n".join(blocks) if blocks else "No retrieved context."
    user_block = f"Question: {question}\n\nUse the context above to answer. If uncertain, say so."
    return f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n[CONTEXT]\n{context_str}\n\n[USER]\n{user_block}"

class _LocalLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = max(2, os.cpu_count() or 4),
                 n_batch: int = 512, n_gpu_layers: int = 0, verbose: bool = False):
        from llama_cpp import Llama
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model GGUF not found: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        out = self.llm(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</s>", "###", "[/USER]"],
        )
        return out["choices"][0]["text"].strip()

# ---------- Public API you can call from your UI ----------
_embedder = _Embedder()          # lazy global (CPU)
_store = _FaissReader(INDEX_DIR) # loads FAISS + metadata
_llm = _LocalLLM(MODEL_PATH)     # loads your quantized model

def rag_answer(
    question: str,
    top_k: int = TOP_K,
    max_ctx_chars: int = MAX_CTX_CHARS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict:
    # 1) encode query
    qvec = _embedder.encode([question])[0]
    # 2) retrieve
    hits = _store.search(qvec, k=top_k)
    results: List[Tuple[float, str, str]] = []
    for score, idx in hits:
        if idx == -1:
            continue
        results.append((float(score), _store.sources[idx], _store.texts[idx]))
    # 3) format prompt with contexts
    contexts = [(src, txt) for (_s, src, txt) in results]
    prompt = _format_prompt(system_prompt, question, contexts, max_ctx_chars=max_ctx_chars)
    # 4) generate
    answer = _llm.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    # 5) return structured result
    return {
        "question": question,
        "answer": answer,
        "contexts": [
            {"score": round(results[i][0], 4), "source": contexts[i][0], "chunk_preview": contexts[i][1][:400]}
            for i in range(len(contexts))
        ],
    }

# Optional: quick demo if running this file directly
if __name__ == "__main__":
    demo_q = "Summarize the main ideas from the corpus."
    out = rag_answer(demo_q)
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== TOP CONTEXTS ===\n")
    for c in out["contexts"]:
        print(f"[{c['score']}] {c['source']}\n{c['chunk_preview']}\n---")
