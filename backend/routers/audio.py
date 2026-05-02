"""
routers/audio.py — ASR and TTS endpoints.

POST /api/transcribe — audio file → Darija text (Whisper)
POST /api/tts        — Arabic text → MP3 audio (edge-tts)
"""

import base64
import logging

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Transcription ─────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    text:     str
    language: str

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file (webm/mp4/wav/ogg)")
):
    """
    Transcribe Darija voice to Arabic text using Whisper.
    Accepts audio blobs from the browser MediaRecorder API (typically webm/opus).
    """
    state = request.state.app_state
    asr   = state.get("asr")

    if asr is None:
        raise HTTPException(503, "Whisper ASR not loaded")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
    if len(audio_bytes) > 25 * 1024 * 1024:   # 25MB
        raise HTTPException(413, "Audio too large (max 25MB)")

    # Detect format from content-type or filename
    ext = "webm"
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
    elif file.content_type:
        ct_map = {
            "audio/webm": "webm",
            "audio/ogg":  "ogg",
            "audio/wav":  "wav",
            "audio/mp4":  "mp4",
            "audio/mpeg": "mp3",
        }
        ext = ct_map.get(file.content_type, "webm")

    logger.info(f"Transcribing {len(audio_bytes)} bytes ({ext})")

    result = asr.transcribe_bytes(audio_bytes, file_ext=ext)

    if not result["text"]:
        raise HTTPException(422, "No speech detected in audio")

    logger.info(f"Transcript: '{result['text'][:80]}'")
    return TranscribeResponse(text=result["text"], language=result["language"])


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=2000, description="Arabic text to synthesize")

@router.post("/tts")
async def text_to_speech(body: TTSRequest):
    """
    Synthesize Arabic text to MP3 audio using edge-tts.
    Returns audio/mpeg binary directly.
    """
    from audio.tts import EdgeTTS
    from config import TTS_VOICE, TTS_RATE, TTS_VOLUME

    if not body.text.strip():
        raise HTTPException(400, "Empty text")

    try:
        tts = EdgeTTS(voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME)
        audio_bytes = await tts.synthesize_async(body.text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(500, f"TTS error: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )