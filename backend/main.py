"""
main.py — FastAPI application entry point for the Olive Assistant.

Starts all services (Whisper, CNN, FAISS) at startup and registers routers.
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    WHISPER_MODEL, CNN_MODEL_PATH, FAISS_INDEX,
    METADATA_FILE, EMBEDDING_MODEL
)
from routers import chat, vision, audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shared state (loaded once at startup) ─────────────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy models at startup — not per-request."""
    logger.info("🫒  Starting Olive Assistant — loading models...")

    # 1. Load Whisper ASR
    logger.info(f"Loading Whisper ({WHISPER_MODEL})...")
    try:
        from audio.asr import WhisperASR
        app_state["asr"] = WhisperASR(model_name=WHISPER_MODEL)
        logger.info("✅ Whisper loaded")
    except Exception as e:
        logger.warning(f"⚠️  Whisper failed to load: {e} — ASR will be unavailable")
        app_state["asr"] = None

    # 2. Load CNN classifier
    logger.info("Loading CNN classifier...")
    try:
        from vision.model import OliveCNN
        app_state["cnn"] = OliveCNN(model_path=CNN_MODEL_PATH)
        logger.info("✅ CNN loaded")
    except Exception as e:
        logger.warning(f"⚠️  CNN failed to load: {e} — Vision will be unavailable")
        app_state["cnn"] = None

    # 3. Load FAISS retriever
    logger.info("Loading FAISS index + embeddings...")
    try:
        from rag.retriever import RAGRetriever
        app_state["rag"] = RAGRetriever(
            index_path=FAISS_INDEX,
            metadata_path=METADATA_FILE,
            embedding_model=EMBEDDING_MODEL
        )
        logger.info("✅ RAG retriever loaded")
    except Exception as e:
        logger.warning(f"⚠️  RAG failed to load: {e} — will need to build index first")
        app_state["rag"] = None

    logger.info("🚀  Olive Assistant ready!")
    yield

    # Shutdown cleanup
    app_state.clear()
    logger.info("Olive Assistant shut down.")


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="مساعد الزيتون — Olive Farmer Assistant",
    description="Multimodal AI assistant for Tunisian olive farmers (Darija + Vision + RAG)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Inject shared state into routers ─────────────────────────────────────────
@app.middleware("http")
async def inject_state(request, call_next):
    request.state.app_state = app_state
    return await call_next(request)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat.router,   prefix="/api", tags=["Chat"])
app.include_router(vision.router, prefix="/api", tags=["Vision"])
app.include_router(audio.router,  prefix="/api", tags=["Audio"])

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr":    app_state.get("asr") is not None,
        "cnn":    app_state.get("cnn") is not None,
        "rag":    app_state.get("rag") is not None,
    }

# ── Serve frontend static files ───────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")