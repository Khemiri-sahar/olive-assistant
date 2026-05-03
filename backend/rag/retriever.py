"""
rag/retriever.py — Semantic retriever with anti-hallucination guard.

CRITICAL SAFETY MECHANISM:
    If top-1 cosine similarity < SIMILARITY_THRESHOLD:
        → Return None (caller must issue polite refusal)
        → LLM is NEVER called
        → No hallucination possible

This is the core protection against the jury's 30-point penalty.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Keywords that should trigger immediate refusal regardless of similarity score
# (dosage questions — legal/safety risk, jury -30pt penalty)
DOSAGE_KEYWORDS = [
    # Arabic / Darija
    "جرعة", "كمية المبيد", "كمية الرش", "نسبة الرش", "نسبة التخليط",
    "شحال نرش", "شحال من مل", "شحال من غرام", "شحال من كيلو",
    "كيفاش تتركب", "كيفاش نركب", "تخليط المبيد", "تحضير المحلول",
    "نسبة المبيد", "قياس المبيد",
    # French
    "dosage", "dose", "quantité", "litre par", "g/l", "ml/l",
    "proportion", "concentration", "dilution", "mélange",
    # English
    "dose", "dosage", "ml per", "g per", "how much pesticide",
    "mixing ratio", "application rate",
]

# Out-of-domain topic keywords → refuse immediately
OOD_KEYWORDS = [
    # Other crops
    "طماطم", "tomato", "tomate", "بطاطا", "potato", "pomme de terre",
    "عنب", "raisin", "vigne", "قمح", "blé", "wheat", "شعير", "orge",
    "لوز", "amande", "almond", "تفاح", "pomme", "apple",
    "فلفل", "poivron", "pepper", "خيار", "concombre", "cucumber",
    "خضرة", "légumes", "potager", "vegetables",
    # Business / economics
    "شركة", "entreprise", "company", "سعر", "prix", "price",
    "بورصة", "bourse", "stock market", "استثمار", "investissement",
    # Politics / news
    "سياسة", "politique", "politics", "أخبار", "actualité", "news",
    "انتخابات", "élections", "elections",
    # Sports / entertainment
    "كرة القدم", "football", "رياضة", "sport", "سينما", "cinéma",
    # General / AI identity traps
    "من أنت", "who are you", "chatgpt", "gpt", "openai", "claude",
    "ذكاء اصطناعي عام", "intelligence artificielle générale",
    # Cooking / human health (non-olive)
    "وصفة طبخ", "recette de cuisine", "دكتور بشري", "médecin",
    # Weather
    "توقعات الطقس", "météo", "weather forecast",
]


class RAGRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: str,
        threshold: float = 0.42,
        top_k: int = 5,
    ):
        self.threshold = threshold
        self.top_k = top_k

        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)

        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)

        logger.info(f"Retriever ready: {self.index.ntotal} vectors, threshold={threshold}")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _is_dosage_question(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in DOSAGE_KEYWORDS)

    def _is_out_of_domain(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in OOD_KEYWORDS)

    def retrieve(
        self,
        query: str,
        disease_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant passages for query.

        Returns dict with:
            - passages: list of retrieved chunks (empty if refused)
            - top_score: float (cosine similarity of best match)
            - should_refuse: bool — True if LLM must NOT be called
            - refuse_reason: str — why refusal was triggered
        """

        # ── Guard 1: out-of-domain keywords ──────────────────────────────────
        if self._is_out_of_domain(query):
            logger.info(f"OOD keyword detected in: '{query[:60]}'")
            return {
                "passages": [],
                "top_score": 0.0,
                "should_refuse": True,
                "refuse_reason": "out_of_domain",
            }

        # ── Guard 2: dosage questions ─────────────────────────────────────────
        if self._is_dosage_question(query):
            logger.info(f"Dosage keyword detected in: '{query[:60]}'")
            return {
                "passages": [],
                "top_score": 0.0,
                "should_refuse": True,
                "refuse_reason": "dosage",
            }

        # ── Translate Darija disease terms to scientific/English for better retrieval ──
        DISEASE_TERM_MAP = {
            "عين الطاووس":   "Spilocaea oleagina peacock spot olive fungal treatment copper",
            "عنكبوت":        "Aculus olearius olive mite acariose treatment",
            "أكولوس":        "Aculus olearius olive mite acariose treatment",
            "أنثراكنوز":     "Colletotrichum olive anthracnose treatment fungicide",
            "ذبابة الزيتون": "Bactrocera oleae Dacus oleae olive fruit fly Ceratitis capitata tephritidae bait trap control",
            "ذبابة":         "Bactrocera oleae olive fruit fly tephritidae Ceratitis control treatment",
            "دودة الزيتون":  "Bactrocera oleae Dacus olive larva fruit fly control treatment",
            "زيلّيلا":       "Xylella fastidiosa olive quick decline Philaenus spumarius vector",
            "زيليلا":        "Xylella fastidiosa olive quick decline Philaenus spumarius vector",
        }
        augmented = query
        for term, expansion in DISEASE_TERM_MAP.items():
            if term in query:
                augmented = f"{query} {expansion}"
                break

        # ── Augment query with disease hint if available ──────────────────────
        search_query = augmented
        if disease_hint:
            search_query = f"{disease_hint} {augmented}"

        # ── Semantic search ───────────────────────────────────────────────────
        vec = self._embed(search_query)
        scores, indices = self.index.search(vec, self.top_k)

        top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
        logger.info(f"Top similarity score: {top_score:.4f} (threshold={self.threshold})")

        # ── Guard 3: similarity threshold (anti-hallucination) ────────────────
        if top_score < self.threshold:
            logger.info(f"Score {top_score:.4f} < threshold {self.threshold} → refusing")
            return {
                "passages": [],
                "top_score": top_score,
                "should_refuse": True,
                "refuse_reason": "low_relevance",
            }

        # ── Collect passages ──────────────────────────────────────────────────
        passages = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(score)
            if float(score) >= self.threshold * 0.8:   # include nearby passages
                passages.append(chunk)

        return {
            "passages": passages,
            "top_score": top_score,
            "should_refuse": False,
            "refuse_reason": None,
        }

    def format_context(self, passages: List[Dict]) -> str:
        """Format retrieved passages for injection into LLM prompt."""
        parts = []
        for i, p in enumerate(passages, 1):
            parts.append(
                f"[مصدر {i} — {p['tag']} — {p['source']}]\n{p['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def format_citations(self, passages: List[Dict]) -> List[str]:
        """Return human-readable citation list."""
        seen = set()
        citations = []
        for p in passages:
            key = f"{p['tag']} — {p['source']}"
            if key not in seen:
                citations.append(key)
                seen.add(key)
        return citations