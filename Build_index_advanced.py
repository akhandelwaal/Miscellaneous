#!/usr/bin/env python3
"""
Build a FAISS index from fund factsheets that contain complex tables.

Key features:
- PDF table extraction via pdfplumber (no JVM requirement).
- Each numeric table cell becomes an atomic "fact sentence" for RAG retrieval.
- Running text is also chunked and indexed.
- Sentence-Transformers embeddings (all-MiniLM-L6-v2) on CPU.
- FAISS (Inner Product) with normalized vectors for cosine similarity.
- Rich metadata saved alongside vectors (source file, page, table idx, row/col, etc.).

You can import build_index() from your UI bootstrap, or run this file directly.
"""

import os
import re
import glob
import pickle
from typing import List, Tuple, Dict, Any, Optional

# ========= CONFIG (edit to your environment) =========
DOCS_DIR = "./docs"                 # folder with your factsheets
INDEX_DIR = "./index_store"         # where FAISS + metadata will be stored
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

# Running-text chunking (heuristic token proxy)
CHUNK_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 40

# pdfplumber table extraction settings (tweak if needed)
# "lattice" style tables (with ruling lines) work well with the 'lines' strategies below.
# If your PDFs are more "stream" style (no clear lines), switch both strategies to "text".
PDFPLUMBER_TABLE_SETTINGS = {
    "vertical_strategy": "lines",     # or "text"
    "horizontal_strategy": "lines",   # or "text"
    # "explicit_vertical_lines": [],  # leave unset unless you need to force positions
    # "explicit_horizontal_lines": [],
}
# =====================================================


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _walk_pdfs(doc_dir: str) -> List[str]:
    """Only PDFs for factsheets."""
    return sorted(list(set(glob.glob(os.path.join(doc_dir, "**/*.pdf"), recursive=True))))


# ---------- Text helpers ----------

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # simple sentence boundary on .?! followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, target_chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """Token-agnostic chunker tuned for MiniLM."""
    def est_tokens(s: str) -> int:
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

    return [c for c in chunks if len(c.split()) >= 5]


# ---------- PDF extraction (tables + running text) ----------

def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # collapse inner whitespace/newlines
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\u00A0", " ", s)  # non-breaking space
    return s


def _looks_numeric(s: str) -> bool:
    """Detect if a cell likely contains a numeric datum (%, numbers, signed, decimals)."""
    if not s:
        return False
    return bool(re.search(r"^[\+\-]?\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?\s*%?$", s))


def _normalize_value(s: str) -> str:
    """
    Keep original human-friendly form but also create a normalized variant if needed.
    Examples:
      "1.38%" -> "1.38%" (retain %)
      "  -4.38 % " -> "-4.38%"
      "14,304.68" -> "14304.68"
    """
    s = s.strip()
    s = s.replace(" %", "%")
    s = re.sub(r"\s+", " ", s)
    if s.endswith("%"):
        # normalize number before %
        num = s[:-1].replace(",", "")
        try:
            float(num)
            return f"{num}%"
        except Exception:
            return s
    # plain number normalization
    s2 = s.replace(",", "")
    try:
        float(s2)
        return s2
    except Exception:
        return s


def _page_text_blocks(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1_based, text) using pdfplumber.
    """
    import pdfplumber
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = txt.strip()
            if txt:
                out.append((i + 1, txt))
    return out


def _page_tables(pdf_path: str) -> List[Tuple[int, int, List[List[str]]]]:
    """
    Returns list of (page_number_1_based, table_index_on_page, table_rows_as_cells),
    where table_rows_as_cells is a 2D list of strings.
    """
    import pdfplumber
    results: List[Tuple[int, int, List[List[str]]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables(PDFPLUMBER_TABLE_SETTINGS) or []
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables):
                # Clean all cells to strings
                cleaned = [[_clean_cell(c) for c in row] for row in (table or [])]
                # drop empty rows
                cleaned = [row for row in cleaned if any(cell for cell in row)]
                if cleaned:
                    results.append((i + 1, t_idx, cleaned))
    return results


def _headers_from_table(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Very simple header detection:
    - If first row has at least one non-empty and seems like headers, use it.
    - Otherwise make generic headers col_0..col_{n-1}.
    Returns (headers, body_rows).
    """
    if not rows:
        return [], []
    n_cols = max(len(r) for r in rows)
    first = rows[0] if rows else []
    header_confidence = sum(1 for c in first if c) >= max(1, n_cols // 2)
    if header_confidence:
        headers = [first[j] if j < len(first) and first[j] else f"col_{j}" for j in range(n_cols)]
        body = rows[1:]
    else:
        headers = [f"col_{j}" for j in range(n_cols)]
        body = rows
    # pad/truncate body rows to n_cols
    body2 = []
    for r in body:
        r = (r + [""] * n_cols)[:n_cols]
        body2.append(r)
    return headers, body2


def _row_label(row: List[str], headers: List[str]) -> str:
    """
    Use first non-empty cell in the row as a 'row label'. This often captures 'Year' or category.
    If none, synthesize from any textual cells.
    """
    for c in row:
        if c and not _looks_numeric(c):
            return c
    # fallback: join first 2 text-like cells
    texts = [c for c in row if c]
    return texts[0] if texts else ""


def _facts_from_table(
    source_path: str,
    page_num: int,
    table_idx: int,
    rows: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Convert a table into atomic fact records (later turned into 'fact sentences').
    Only keeps cells that look numeric, so we don't bloat the index with labels.
    """
    facts: List[Dict[str, Any]] = []
    headers, body = _headers_from_table(rows)
    n_cols = len(headers)

    for r_i, row in enumerate(body):
        row_lbl = _row_label(row, headers).strip()
        for c_j in range(n_cols):
            col_hdr = headers[c_j].strip()
            cell = row[c_j].strip() if c_j < len(row) else ""
            if not cell:
                continue
            if _looks_numeric(cell):
                val_norm = _normalize_value(cell)
                f = {
                    "source": source_path,
                    "page": page_num,
                    "table_index": table_idx,
                    "row_index": r_i,
                    "col_index": c_j,
                    "row_label": row_lbl,
                    "col_header": col_hdr,
                    "value_raw": cell,
                    "value": val_norm,
                }
                facts.append(f)
    return facts


def _fact_to_sentence(f: Dict[str, Any]) -> str:
    """
    Turn a numeric cell into a compact, embedding-friendly sentence with metadata baked in.
    Example:
      "<S&P500_factsheet.pdf> page 3 | Calendar Year Performance | 2018 | Total Return = 1.38%"
    This helps retrieval for queries like: "total % return in 2018".
    """
    base = os.path.basename(f["source"])
    parts = [
        f"<{base}>",
        f"page {f['page']}",
    ]
    # Try to build a hierarchical label: Row (e.g., Year) + Column header (e.g., Total Return)
    row_lbl = f.get("row_label") or ""
    col_hdr = f.get("col_header") or ""
    if row_lbl:
        parts.append(row_lbl)
    if col_hdr:
        parts.append(col_hdr)
    # final value
    val = f.get("value") or f.get("value_raw") or ""
    sentence = " | ".join(parts) + f" = {val}"
    return sentence.strip()


# ---------- Embedding + FAISS ----------

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
        self.index = faiss.IndexFlatIP(dim)  # cosine via inner product (with normalized vectors)
        # Payload metadata
        self.payloads: List[Dict[str, Any]] = []  # one entry per vector

    def add(self, vectors, payloads: List[Dict[str, Any]]):
        """
        payloads[i] should contain at least:
            - "type": "table_fact" | "text_chunk"
            - "text": the actual string used for embedding
            - "source": file path
            - plus any other fields (page/table indices, headers, etc.)
        """
        self.index.add(vectors)
        self.payloads.extend(payloads)

    def save(self):
        import faiss
        _ensure_dir(self.index_dir)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "payloads": self.payloads,
                },
                f,
            )


# ---------- Build pipeline ----------

def _index_pdf(pdf_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (texts_to_embed, payloads).
    - Table cells -> fact sentences (type="table_fact")
    - Running text -> chunked (type="text_chunk")
    """
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    # 1) Tables -> facts
    table_data = _page_tables(pdf_path)
    all_facts: List[Dict[str, Any]] = []
    for pg, t_idx, rows in table_data:
        facts = _facts_from_table(source_path=pdf_path, page_num=pg, table_idx=t_idx, rows=rows)
        all_facts.extend(facts)

    for f in all_facts:
        s = _fact_to_sentence(f)
        if not s:
            continue
        texts.append(s)
        payloads.append({
            "type": "table_fact",
            "text": s,
            **f,  # include metadata (source, page, row/col/header/value)
        })

    # 2) Running text -> chunked
    page_texts = _page_text_blocks(pdf_path)
    for pg, txt in page_texts:
        chunks = _chunk_text(txt, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        for ch in chunks:
            texts.append(ch)
            payloads.append({
                "type": "text_chunk",
                "text": ch,
                "source": pdf_path,
                "page": pg,
            })

    return texts, payloads


def build_index(
    docs_dir: str = DOCS_DIR,
    index_dir: str = INDEX_DIR,
    embed_model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> int:
    pdfs = _walk_pdfs(docs_dir)
    if not pdfs:
        print(f"[WARN] No PDFs found under {docs_dir}")
        return 0

    # Aggregate all texts & payloads from all PDFs
    all_texts: List[str] = []
    all_payloads: List[Dict[str, Any]] = []

    print(f"[INFO] Found {len(pdfs)} PDF(s). Extracting tables and text...")
    for path in pdfs:
        t, p = _index_pdf(path)
        if t:
            all_texts.extend(t)
            all_payloads.extend(p)
        else:
            print(f"[WARN] No extractable content in {path}")

    if not all_texts:
        print("[WARN] Nothing to embed.")
        return 0

    # Embeddings
    print(f"[INFO] Total items to embed: {len(all_texts)}. Model: {embed_model}")
    emb = _Embedder(embed_model)
    vectors = emb.encode(all_texts, batch_size=batch_size, normalize=True)

    # FAISS store
    print(f"[INFO] Saving FAISS index to {index_dir}")
    store = _FaissStore(dim=emb.dim, index_dir=index_dir)
    store.add(vectors, all_payloads)
    store.save()
    print(f"[OK] Saved {len(all_texts)} vectors with metadata.")
    return len(all_texts)


# Optional: run directly
if __name__ == "__main__":
    build_index()
