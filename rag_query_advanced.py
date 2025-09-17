#!/usr/bin/env python3
"""
RAG retrieval + local generation for table-heavy factsheets.

Works with the index produced by the updated build_index.py:
- meta.pkl contains: {"dim": int, "payloads": List[Dict]]}
- vectors.faiss contains the corresponding vectors

Retrieval strategy:
- Query -> embedding -> FAISS top-N
- Prefer table facts (type="table_fact") while still allowing some text chunks
- Build prompt with a dedicated [FACTS] section and a [CONTEXT] section
- Instruct the LLM to cite (file/page/table) and to say "don't know" if not found

Exported entry point:
- rag_answer(question: str) -> Dict with "answer" and "contexts"
"""

import os
import pickle
from typing import List, Dict, Tuple, Any

# ========== CONFIG ==========
INDEX_DIR = "./index_store"   # must match build_index.py output folder
MODEL_PATH = "/path/to/your/quantized-model.gguf"  # e.g., llama-3-8b-instruct.Q4_K_M.gguf

# Retrieval mix
TOP_K_TOTAL = 10          # how many items to pull from FAISS initially
MAX_FACTS = 6             # prefer up to this many table facts in the final prompt
MAX_TEXTS = 4             # and up to this many text chunks

# Prompting / generation
MAX_CTX_CHARS = 9000
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 512
SYSTEM_PROMPT = (
    "You are a careful financial assistant. Use the provided data to answer. "
    "Prioritize numeric facts from tables. If the answer is not present, say you don't know. "
    "Always include brief citations like [file:page(:table)] after key numbers."
)
# ============================


# ---------- Embeddings ----------
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


# ---------- FAISS loader ----------
class _FaissReader:
    def __init__(self, index_dir: str):
        import faiss
        meta_path = os.path.join(index_dir, "meta.pkl")
        index_path = os.path.join(index_dir, "vectors.faiss")
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(f"Missing index files in {index_dir}. Build them first.")
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.payloads: List[Dict[str, Any]] = data["payloads"]
        self.dim: int = data["dim"]
        self.faiss = faiss.read_index(index_path)

    def search(self, qvec, k: int) -> List[Tuple[float, int]]:
        scores, idxs = self.faiss.search(qvec.reshape(1, -1), k)
        return list(zip(scores[0].tolist(), idxs[0].tolist()))


# ---------- LLM (llama-cpp) ----------
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


# ---------- Utilities ----------
def _short_cite(p: Dict[str, Any]) -> str:
    """
    Make a compact citation tag like: [S&P500.pdf:3:1] or [S&P500.pdf:3]
    """
    base = os.path.basename(p.get("source", "?"))
    page = p.get("page")
    tbl = p.get("table_index", None)
    if page and tbl is not None:
        return f"[{base}:{page}:{tbl}]"
    elif page:
        return f"[{base}:{page}]"
    return f"[{base}]"


def _format_facts_block(facts: List[Dict[str, Any]]) -> str:
    """
    Render table facts as numbered bullet lines with citations.
    Each fact payload has fields like: row_label, col_header, value, etc.
    """
    lines = []
    for i, p in enumerate(facts, 1):
        # 'text' already looks like: "<file> page N | 2018 | Total Return = -4.38%"
        # But we also surface structured elements for clarity.
        row = p.get("row_label") or ""
        col = p.get("col_header") or ""
        val = p.get("value") or p.get("value_raw") or ""
        cite = _short_cite(p)
        if row or col or val:
            core = " | ".join([s for s in [row, col] if s]) or p.get("text", "")
            if val and core:
                lines.append(f"{i}. {core} = {val} {cite}")
            else:
                lines.append(f"{i}. {p.get('text', '')} {cite}")
        else:
            lines.append(f"{i}. {p.get('text', '')} {cite}")
    return "\n".join(lines)


def _format_context_block(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    out, used = [], 0
    for p in chunks:
        t = p.get("text", "")
        if not t:
            continue
        block = f"[{_short_cite(p)}]\n{t}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


def _build_prompt(system_msg: str,
                  question: str,
                  facts: List[Dict[str, Any]],
                  texts: List[Dict[str, Any]],
                  max_ctx_chars: int) -> str:
    facts_block = _format_facts_block(facts) if facts else "No table facts retrieved."
    context_block = _format_context_block(texts, max_ctx_chars=max(0, max_ctx_chars - len(facts_block)))
    user = (
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1) Prefer numeric facts in [FACTS] when answering.\n"
        "2) If the exact value/year/metric is not in the facts, say you don't know.\n"
        "3) Include short citations like [file:page(:table)] after key numbers.\n"
        "4) Keep the answer concise."
    )
    return (
        f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n"
        f"[FACTS]\n{facts_block}\n\n"
        f"[CONTEXT]\n{context_block}\n\n"
        f"[USER]\n{user}"
    )


# ---------- Public API ----------
_embedder = _Embedder()
_store = _FaissReader(INDEX_DIR)
_llm = _LocalLLM(MODEL_PATH)


def _retrieve(question: str,
              top_k_total: int = TOP_K_TOTAL,
              max_facts: int = MAX_FACTS,
              max_texts: int = MAX_TEXTS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[float, Dict[str, Any]]]]:
    """
    Retrieve top candidates and split into (facts, texts).
    Returns (facts_selected, texts_selected, scored_all) where scored_all = [(score, payload), ...]
    """
    # 1) Encode query
    qvec = _embedder.encode([question])[0]

    # 2) FAISS search
    hits = _store.search(qvec, k=top_k_total)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for score, idx in hits:
        if idx == -1:
            continue
        payload = _store.payloads[idx]
        scored.append((float(score), payload))

    # 3) Split by type and select
    facts = [p for s, p in scored if p.get("type") == "table_fact"]
    texts = [p for s, p in scored if p.get("type") == "text_chunk"]

    # Simple heuristic re-order within each group by their original FAISS score (already sorted)
    # (If needed, you can add extra boosting for exact years/% matches here.)

    facts_selected = facts[:max_facts]
    texts_selected = texts[:max_texts]
    return facts_selected, texts_selected, scored


def rag_answer(
    question: str,
    top_k_total: int = TOP_K_TOTAL,
    max_facts: int = MAX_FACTS,
    max_texts: int = MAX_TEXTS,
    max_ctx_chars: int = MAX_CTX_CHARS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    Main entry point for your UI.
    """
    # Retrieve
    facts, texts, scored = _retrieve(
        question,
        top_k_total=top_k_total,
        max_facts=max_facts,
        max_texts=max_texts,
    )

    # Build prompt
    prompt = _build_prompt(system_prompt, question, facts, texts, max_ctx_chars)

    # Generate
    answer = _llm.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    # Package contexts (with compact previews)
    def preview(p: Dict[str, Any], n: int = 300) -> str:
        t = p.get("text", "")
        return t[:n] + ("…" if len(t) > n else "")

    contexts = []
    for p in facts:
        contexts.append({
            "type": "table_fact",
            "source": p.get("source"),
            "page": p.get("page"),
            "table_index": p.get("table_index"),
            "row_label": p.get("row_label"),
            "col_header": p.get("col_header"),
            "value": p.get("value") or p.get("value_raw"),
            "text": preview(p),
            "citation": _short_cite(p),
        })
    for p in texts:
        contexts.append({
            "type": "text_chunk",
            "source": p.get("source"),
            "page": p.get("page"),
            "text": preview(p),
            "citation": _short_cite(p),
        })

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "debug": {
            "retrieved_count": len(scored),
            "facts_selected": len(facts),
            "texts_selected": len(texts),
        },
    }


# Optional: quick demo if running directly
if __name__ == "__main__":
    demo_q = "How did the S&P 500 perform in total % return in calendar year 2018?"
    out = rag_answer(demo_q)
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== CONTEXTS ===\n")
    for c in out["contexts"]:
        print(c["type"], c["citation"], c.get("row_label"), c.get("col_header"), c.get("value", ""), "\n", c["text"], "\n---")
