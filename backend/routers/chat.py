"""
routers/chat.py — Main /api/ask endpoint.

Pipeline:
    1. Receive question text + optional CNN result
    2. Run RAG retrieval (similarity check)
    3. If score < threshold → immediate refusal (no LLM call)
    4. Otherwise → strict LLM prompt → response in Darija
    5. Return text + citations + TTS audio (base64)
"""

import base64
import logging
from typing import Optional

import anthropic
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:    str          = Field(..., description="User question in Darija")
    disease_id:  Optional[int] = Field(None, description="CNN class ID (0-4)")
    tts_enabled: bool          = Field(True,  description="Include audio in response")


class AskResponse(BaseModel):
    answer:    str
    citations: list[str]
    refused:   bool
    refuse_reason: Optional[str]
    top_score: float
    audio_b64: Optional[str]   = None   # MP3 base64 if tts_enabled


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_disease_info(disease_id: Optional[int]) -> Optional[dict]:
    if disease_id is None:
        return None
    from config import DISEASE_CLASSES
    return DISEASE_CLASSES.get(disease_id)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic Claude with strict RAG prompt."""
    from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS

    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "LLM not configured — set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    """
    Main conversational endpoint.
    Receives Darija question + optional disease_id from CNN.
    Returns Darija answer with citation + optional TTS audio.
    """
    state = request.state.app_state

    rag: object = state.get("rag")
    if rag is None:
        raise HTTPException(503, "RAG index not loaded — run indexer first")

    # ── 1. Get disease context from CNN result ────────────────────────────────
    disease_info = _get_disease_info(body.disease_id)
    disease_hint = None
    if disease_info:
        disease_hint = f"{disease_info['en']} {disease_info.get('eppo', '')}"

    # ── 2. RAG retrieval + anti-hallucination guard ────────────────────────────
    retrieval = rag.retrieve(
        query=body.question,
        disease_hint=disease_hint,
    )

    # ── 3. Refusal path (no LLM involved) ─────────────────────────────────────
    if retrieval["should_refuse"]:
        from rag.prompts import get_refusal_message
        refusal_text = get_refusal_message(retrieval["refuse_reason"])

        audio_b64 = None
        if body.tts_enabled:
            audio_b64 = await _tts_b64(state, refusal_text)

        logger.info(
            f"REFUSED [{retrieval['refuse_reason']}] "
            f"score={retrieval['top_score']:.3f} q='{body.question[:50]}'"
        )

        return AskResponse(
            answer=refusal_text,
            citations=[],
            refused=True,
            refuse_reason=retrieval["refuse_reason"],
            top_score=retrieval["top_score"],
            audio_b64=audio_b64,
        )

    # ── 4. Build LLM prompt with retrieved context ────────────────────────────
    from rag.prompts import SYSTEM_PROMPT, build_user_prompt

    context    = rag.format_context(retrieval["passages"])
    user_prompt = build_user_prompt(
        question=body.question,
        context=context,
        disease_info=disease_info,
    )

    logger.info(
        f"ANSWERING score={retrieval['top_score']:.3f} "
        f"passages={len(retrieval['passages'])} q='{body.question[:50]}'"
    )

    # ── 5. LLM call ───────────────────────────────────────────────────────────
    try:
        answer = _call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(500, f"LLM error: {str(e)}")

    citations = rag.format_citations(retrieval["passages"])

    # ── 6. TTS ────────────────────────────────────────────────────────────────
    audio_b64 = None
    if body.tts_enabled:
        audio_b64 = await _tts_b64(state, answer)

    return AskResponse(
        answer=answer,
        citations=citations,
        refused=False,
        refuse_reason=None,
        top_score=retrieval["top_score"],
        audio_b64=audio_b64,
    )


async def _tts_b64(state: dict, text: str) -> Optional[str]:
    """Generate TTS audio and return as base64 MP3."""
    try:
        from audio.tts import EdgeTTS
        from config import TTS_VOICE, TTS_RATE, TTS_VOLUME

        tts = EdgeTTS(voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME)
        audio_bytes = await tts.synthesize_async(text)
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
        return None