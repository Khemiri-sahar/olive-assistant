"""
audio/asr.py — Whisper ASR wrapper for Darija (Tunisian Arabic) transcription.

Whisper-large-v3 handles Tunisian Arabic well without fine-tuning.
For production on CPU use 'base' or 'small' for faster inference.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperASR:
    def __init__(self, model_name: str = "base", language: str = "ar"):
        """
        model_name: 'tiny' | 'base' | 'small' | 'medium' | 'large-v3'
        language:   'ar' forces Arabic — better than auto-detect for Darija
        """
        logger.info(f"Loading Whisper model: {model_name}")
        import whisper
        self.model    = whisper.load_model(model_name)
        self.language = language
        logger.info("✅ Whisper ready")

    def transcribe_bytes(self, audio_bytes: bytes, file_ext: str = "webm") -> dict:
        """
        Transcribe audio from raw bytes.
        audio_bytes: raw audio data (webm/mp4/wav/ogg)
        file_ext:    extension hint for ffmpeg

        Returns dict:
            text:       str — transcribed text
            language:   str — detected language code
            segments:   list — time-stamped segments
        """
        import whisper

        # Write to temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            result = self.model.transcribe(
                tmp_path,
                language=self.language,
                task="transcribe",
                # These options improve Darija recognition:
                condition_on_previous_text=False,
                temperature=0.0,          # greedy decoding — more deterministic
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
            )
            return {
                "text":     result["text"].strip(),
                "language": result.get("language", self.language),
                "segments": result.get("segments", []),
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_file(self, audio_path: Path) -> dict:
        """Transcribe from a file on disk."""
        import whisper
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return {
            "text":     result["text"].strip(),
            "language": result.get("language", self.language),
            "segments": result.get("segments", []),
        }