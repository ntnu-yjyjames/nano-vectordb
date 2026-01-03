#!/usr/bin/env python3
"""
build_vecbin_chunked.py (standalone, repo-local)

CSV -> (section split) -> (sentence-aware chunking) -> embed -> .vecbin + rowmeta.jsonl

Input default:
  ./arxiv_data/arxiv_cornell_title_abstract.csv

Expected columns:
  - id (or article_id/paper_id)  [optional]
  - title
  - abstract
Optional full-text columns (first found is used):
  - full_text, body_text, paper_text, text

Chunking (matches your spec):
  1) If full-text exists: split into sections by regex headings.
     If not: single ("abstract", abstract_text).
  2) For each section: split into <= max_chars_per_chunk, prefer last sentence-ending punctuation near end.
  3) Embed text formatted as:
     {title}\n[SECTION: {section_name}]\n{chunk_text}

Output vecbin formats (C++ compatible):
  - vecbin64 (default): 64B header with dtype=f32 + float32 payload
  - raw12: u32 count, u32 reserved=0, u32 dim + float32 payload

Also outputs:
  - <out>.rowmeta.jsonl (if --export-metadata)
"""

import argparse
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- vecbin64 header constants (must match C++ vecbin_format.h) ----------
K_MAGIC   = 0x4E56444256454331  # uint64 "NVDBVEC1"
K_VERSION = 1
DTYPE_F32 = 1  # Float32


FULLTEXT_CANDIDATES = ["full_text", "body_text", "paper_text", "text"]

# Common academic headings (extend as needed)
HEADING_KEYWORDS = [
    "abstract", "introduction", "related work", "background", "method", "methods",
    "materials", "experiments", "results", "discussion", "conclusion", "conclusions",
    "acknowledgements", "acknowledgments", "references",
]

# Sentence end punctuation set
SENT_END = set([".", "!", "?", "。", "！", "？"])


@dataclass
class Config:
    csv_path: str
    out_vecbin: str
    out_format: str            # vecbin64 or raw12
    model_name: str
    batch_size: int
    csv_chunksize: int
    max_docs: Optional[int]
    max_chars_per_chunk: int
    prefer_fulltext: bool
    export_metadata: bool

    id_col: str
    title_col: str
    abstract_col: str


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Standalone chunking + embedding + vecbin export for Nano-VectorDB.")
    ap.add_argument("--csv-path", default="./arxiv_data/arxiv_cornell_title_abstract.csv")
    ap.add_argument("--out", required=True, help="Output vecbin path (e.g., ./vecbin_full/embeddings_500k.vecbin)")
    ap.add_argument("--format", default="vecbin64", choices=["vecbin64", "raw12"])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--csv-chunksize", type=int, default=2000, help="Rows per CSV chunk read (streaming).")
    ap.add_argument("--max-docs", type=int, default=None, help="Limit number of documents (rows) for quick tests.")
    ap.add_argument("--max-chars-per-chunk", type=int, default=2000)
    ap.add_argument("--prefer-fulltext", action="store_true",
                    help="If set, use full-text column when available; otherwise fall back to abstract.")
    ap.add_argument("--export-metadata", action="store_true", help="Write <out>.rowmeta.jsonl")

    ap.add_argument("--id-col", default="id")
    ap.add_argument("--title-col", default="title")
    ap.add_argument("--abstract-col", default="abstract")

    a = ap.parse_args()
    return Config(
        csv_path=a.csv_path,
        out_vecbin=a.out,
        out_format=a.format,
        model_name=a.model,
        batch_size=a.batch_size,
        csv_chunksize=a.csv_chunksize,
        max_docs=a.max_docs,
        max_chars_per_chunk=a.max_chars_per_chunk,
        prefer_fulltext=a.prefer_fulltext,
        export_metadata=a.export_metadata,
        id_col=a.id_col,
        title_col=a.title_col,
        abstract_col=a.abstract_col,
    )


# ---------- helpers: header writing ----------
def write_raw12_header(f, count_u32: int, dim_u32: int) -> None:
    f.write(struct.pack("<III", count_u32, 0, dim_u32))

def write_vecbin64_header(f, count_u64: int, dim_u32: int, dtype_u32: int = DTYPE_F32) -> None:
    base = struct.pack("<QIIIIQ", K_MAGIC, K_VERSION, dtype_u32, dim_u32, 0, count_u64)
    pad = 64 - len(base)
    f.write(base + b"\x00" * pad)

def patch_raw12_header(f, count_u32: int, dim_u32: int) -> None:
    f.seek(0)
    write_raw12_header(f, count_u32, dim_u32)
    f.seek(0, os.SEEK_END)

def patch_vecbin64_header(f, count_u64: int, dim_u32: int, dtype_u32: int = DTYPE_F32) -> None:
    f.seek(0)
    write_vecbin64_header(f, count_u64, dim_u32, dtype_u32)
    f.seek(0, os.SEEK_END)


# ---------- chunking: section splitting ----------
def normalize_heading(h: str) -> str:
    h = h.strip().lower()
    h = re.sub(r"[\s\-_:]+", " ", h)
    return h

def build_heading_regex() -> re.Pattern:
    # matches lines like:
    #   "1 Introduction", "I. RELATED WORK", "2.1 Methods", "Conclusion", etc.
    # Uses multiline mode.
    kw = "|".join(re.escape(k) for k in sorted(HEADING_KEYWORDS, key=len, reverse=True))
    # optional numeric/roman prefix, optional punctuation, heading keywords
    pat = rf"(?im)^\s*(?:\(?\s*(?:\d+(\.\d+)*|[ivx]+)\s*\)?[.\-:]?\s+)?({kw})\s*$"
    return re.compile(pat)

HEADING_RE = build_heading_regex()

def split_into_sections(full_text: str) -> List[Tuple[str, str]]:
    """
    Split full text into (section_name, section_text) using heading regex.
    If no headings are found, return one section ('body', full_text).
    """
    if not full_text:
        return []

    matches = list(HEADING_RE.finditer(full_text))
    if not matches:
        return [("body", full_text.strip())]

    sections: List[Tuple[str, str]] = []
    # text before first heading is ignored unless non-trivial; you can keep it as 'preamble'
    for idx, m in enumerate(matches):
        sec_name = normalize_heading(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        sec_text = full_text[start:end].strip()
        if sec_text:
            sections.append((sec_name, sec_text))
    if not sections:
        sections = [("body", full_text.strip())]
    return sections


# ---------- chunking: length-based sentence-aware split ----------
def find_last_sentence_end(window: str) -> int:
    # return index (exclusive) of last sentence end char, or -1 if none
    for i in range(len(window) - 1, -1, -1):
        if window[i] in SENT_END:
            return i + 1
    return -1

def chunk_by_max_chars(text: str, max_chars: int) -> List[str]:
    """
    Split text into chunks <= max_chars, prefer cutting at last sentence-ending punctuation
    near the end of the window. If none found, hard cut.
    """
    t = (text or "").strip()
    if not t:
        return []

    chunks: List[str] = []
    pos = 0
    n = len(t)

    while pos < n:
        remaining = t[pos:]
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break

        window = remaining[:max_chars]
        # search for sentence end in the last ~25% region first, else whole window
        tail_region = window[int(max_chars * 0.6):]
        cut = find_last_sentence_end(tail_region)
        if cut != -1:
            cut = int(max_chars * 0.6) + cut
        else:
            cut = find_last_sentence_end(window)

        if cut == -1:
            cut = max_chars  # hard cut

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        pos += cut

    return chunks


def choose_text_source(row: pd.Series, cfg: Config) -> Tuple[str, Optional[str]]:
    """
    Decide which text to chunk: full-text if prefer_fulltext and available; else abstract.
    Returns (source_text, source_label).
    """
    # find a full-text column if exists
    full_col = None
    for c in FULLTEXT_CANDIDATES:
        if c in row and isinstance(row[c], str) and row[c].strip():
            full_col = c
            break

    abstract = row.get(cfg.abstract_col, "") if isinstance(row.get(cfg.abstract_col, ""), str) else ""
    full_text = row.get(full_col, "") if full_col else ""

    if cfg.prefer_fulltext and full_col and full_text.strip():
        return full_text, full_col
    # fallback
    return abstract, None


def iter_csv(cfg: Config) -> Iterator[pd.DataFrame]:
    # keep_default_na=False prevents NaN
    for chunk in pd.read_csv(cfg.csv_path, chunksize=cfg.csv_chunksize, dtype=str, keep_default_na=False):
        yield chunk


def main():
    cfg = parse_args()
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    out_path = os.path.abspath(cfg.out_vecbin)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rowmeta_path = os.path.splitext(out_path)[0] + ".rowmeta.jsonl"

    print(f"[INFO] Loading model: {cfg.model_name}")
    model = SentenceTransformer(cfg.model_name)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"[INFO] dim={dim} max_chars_per_chunk={cfg.max_chars_per_chunk}")

    # Open outputs
    f = open(out_path, "wb")

    # Write placeholder header (count=0), patch later
    if cfg.out_format == "raw12":
        write_raw12_header(f, 0, dim)
    else:
        write_vecbin64_header(f, 0, dim, DTYPE_F32)

    meta_fp = None
    if cfg.export_metadata:
        meta_fp = open(rowmeta_path, "w", encoding="utf-8")
        print(f"[INFO] Writing row metadata: {rowmeta_path}")

    total_docs = 0
    total_chunks = 0

    # For embedding batching at chunk-level
    batch_texts: List[str] = []
    batch_meta: List[Dict[str, object]] = []

    def flush_batch():
        nonlocal total_chunks
        if not batch_texts:
            return
        emb = model.encode(
            batch_texts,
            batch_size=cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        f.write(emb.tobytes(order="C"))

        if meta_fp is not None:
            for rec in batch_meta:
                meta_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_chunks += emb.shape[0]
        batch_texts.clear()
        batch_meta.clear()

    for df in iter_csv(cfg):
        for _, row in df.iterrows():
            if cfg.max_docs is not None and total_docs >= cfg.max_docs:
                break

            doc_idx = total_docs  # stable internal doc index
            paper_id = row.get(cfg.id_col, "") if cfg.id_col in row else ""
            title = row.get(cfg.title_col, "") if cfg.title_col in row else ""
            title = (title or "").strip()

            source_text, source_label = choose_text_source(row, cfg)

            # stage 1: section split (if full-text exists and prefer_fulltext)
            sections: List[Tuple[str, str]]
            if cfg.prefer_fulltext and source_label is not None:
                sections = split_into_sections(source_text)
                # If abstract column exists and no "abstract" section was found, prepend abstract section
                abstract = row.get(cfg.abstract_col, "") if cfg.abstract_col in row else ""
                abstract = (abstract or "").strip()
                if abstract and all(sec != "abstract" for sec, _ in sections):
                    sections = [("abstract", abstract)] + sections
            else:
                abstract = (source_text or "").strip()
                sections = [("abstract", abstract)] if abstract else []

            # stage 2: chunk each section by max chars, sentence-aware
            chunk_index = 0
            for sec_name, sec_text in sections:
                chunks = chunk_by_max_chars(sec_text, cfg.max_chars_per_chunk)
                for chunk_text in chunks:
                    embed_text = f"{title}\n[SECTION: {sec_name}]\n{chunk_text}".strip()

                    batch_texts.append(embed_text)
                    batch_meta.append({
                        "row": total_chunks + len(batch_texts) - 1,  # row index in embeddings matrix
                        "doc_idx": doc_idx,
                        "id": paper_id,
                        "title": title,
                        "section": sec_name,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                    # flush at doc-level or when batch is large (to control memory)
                    if len(batch_texts) >= 4096:
                        flush_batch()

            total_docs += 1

        if cfg.max_docs is not None and total_docs >= cfg.max_docs:
            break

        # flush per CSV chunk
        flush_batch()

        if total_docs % 50000 == 0 and total_docs > 0:
            print(f"[INFO] docs={total_docs} chunks={total_chunks}")

    flush_batch()

    # Patch header with final count (chunks)
    if cfg.out_format == "raw12":
        if total_chunks > 0xFFFFFFFF:
            raise ValueError(f"raw12 header uses uint32 count; got count={total_chunks}")
        patch_raw12_header(f, int(total_chunks), dim)
    else:
        patch_vecbin64_header(f, int(total_chunks), dim, DTYPE_F32)

    f.close()
    if meta_fp is not None:
        meta_fp.close()

    print(f"[INFO] Export done: {out_path}")
    print(f"[INFO] docs={total_docs} total_chunks={total_chunks} dim={dim}")
    if cfg.export_metadata:
        print(f"[INFO] rowmeta={rowmeta_path}")


if __name__ == "__main__":
    main()
