"""
audio/tts.py — edge-tts wrapper for Tunisian Arabic text-to-speech.

Uses Microsoft's edge-tts (free, no API key, no rate limit).
ar-TN-ReemNeural is the Tunisian Arabic voice — closest to Darija.
Fallback: ar-SA-HamedNeural (MSA) if TN voice unavailable.
"""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Preferred Tunisian Arabic voice
TN_VOICE    = "ar-TN-ReemNeural"
# Fallback MSA voice if TN unavailable
MSA_VOICE   = "ar-SA-HamedNeural"


class EdgeTTS:
    def __init__(
        self,
        voice: str = TN_VOICE,
        rate:  str = "+0%",
        volume: str = "+0%",
    ):
        self.voice  = voice
        self.rate   = rate
        self.volume = volume
        logger.info(f"TTS initialized: voice={voice}")

    async def _synthesize_async(self, text: str) -> bytes:
        """Async core of synthesis — returns MP3 bytes."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            raise RuntimeError("edge-tts returned no audio data")

        return b"".join(audio_chunks)

    def synthesize(self, text: str) -> bytes:
        """
        Synchronous wrapper — returns MP3 bytes.
        Can be called from sync FastAPI routes via run_in_executor.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — schedule as a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._synthesize_async(text))
                    return future.result()
            else:
                return loop.run_until_complete(self._synthesize_async(text))
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Try fallback voice
            if self.voice != MSA_VOICE:
                logger.warning(f"Retrying with fallback voice {MSA_VOICE}")
                original_voice = self.voice
                self.voice = MSA_VOICE
                try:
                    result = self.synthesize(text)
                    self.voice = original_voice
                    return result
                except Exception as e2:
                    self.voice = original_voice
                    raise RuntimeError(f"TTS failed with both voices: {e}, {e2}")
            raise

    async def synthesize_async(self, text: str) -> bytes:
        """Async version — preferred in async FastAPI routes."""
        return await self._synthesize_async(text)

    @staticmethod
    async def list_voices() -> list:
        """List all available Arabic voices."""
        import edge_tts
        voices = await edge_tts.list_voices()
        return [v for v in voices if v["Locale"].startswith("ar-")]