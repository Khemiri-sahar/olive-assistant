"""
rag/indexer.py — PDF corpus → text chunks → FAISS vector index.

Run this ONCE before starting the server:
    python -m rag.indexer --pdf_dir ./corpus/pdfs

Produces:
    corpus/index.faiss
    corpus/metadata.json
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using pdfminer."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(pdf_path))
        return text or ""
    except Exception as e:
        logger.warning(f"pdfminer failed for {pdf_path.name}: {e}")
        try:
            import PyPDF2
            text_parts = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e2:
            logger.error(f"All PDF extractors failed for {pdf_path.name}: {e2}")
            return ""


def chunk_text(
    text: str,
    source_name: str,
    source_tag: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks of ~chunk_size characters.
    Returns list of dicts with 'text', 'source', 'tag', 'chunk_id'.
    """
    # Clean up whitespace
    text = " ".join(text.split())
    if not text.strip():
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at a sentence boundary
        if end < len(text):
            last_period = max(
                chunk.rfind(". "),
                chunk.rfind(".\n"),
                chunk.rfind("! "),
                chunk.rfind("? "),
            )
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]

        if len(chunk.strip()) > 50:   # skip tiny fragments
            chunks.append({
                "text":     chunk.strip(),
                "source":   source_name,
                "tag":      source_tag,
                "chunk_id": f"{source_tag}_{chunk_id:04d}",
            })
            chunk_id += 1

        start += step

    return chunks


# ── Index builder ─────────────────────────────────────────────────────────────

class CorpusIndexer:
    def __init__(self, embedding_model: str, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")

    def build_from_pdfs(self, pdf_dir: Path) -> int:
        """
        Process all PDFs in pdf_dir, build FAISS index, save to disk.
        Returns number of chunks indexed.
        """
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

        logger.info(f"Found {len(pdf_files)} PDFs to index")

        all_chunks: List[Dict] = []

        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
            # Derive tag from filename (e.g. "FAO_olive_manual.pdf" → "FAO")
            tag = pdf_path.stem.split("_")[0].upper()
            name = pdf_path.stem.replace("_", " ").title()

            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"Empty text from {pdf_path.name}")
                continue

            chunks = chunk_text(text, source_name=name, source_tag=tag)
            all_chunks.extend(chunks)
            logger.info(f"  {pdf_path.name}: {len(chunks)} chunks")

        if not all_chunks:
            raise ValueError("No chunks produced — check PDF content")

        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info("Generating embeddings...")

        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity via inner product
        )

        # Build FAISS index (inner product = cosine since normalized)
        index = faiss.IndexFlatIP(self.dim)
        index.add(np.array(embeddings, dtype=np.float32))

        # Save index and metadata
        faiss_path = self.index_dir / "index.faiss"
        meta_path  = self.index_dir / "metadata.json"

        faiss.write_index(index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Index saved: {faiss_path}")
        logger.info(f"✅ Metadata saved: {meta_path}")
        return len(all_chunks)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build FAISS index from PDF corpus")
    parser.add_argument("--pdf_dir",  type=Path, default=Path("corpus/pdfs"))
    parser.add_argument("--out_dir",  type=Path, default=Path("corpus"))
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    args = parser.parse_args()

    indexer = CorpusIndexer(
        embedding_model=args.model,
        index_dir=args.out_dir,
    )
    n = indexer.build_from_pdfs(args.pdf_dir)
    print(f"\n✅ Done! {n} chunks indexed.")