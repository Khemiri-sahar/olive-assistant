"""
config.py — Central configuration for the Olive Assistant.
All tuneable parameters live here. Adjust before jury demo.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
CORPUS_DIR      = BASE_DIR / "corpus"
MODELS_DIR      = BASE_DIR / "models"
FAISS_INDEX     = Path(os.getenv("FAISS_INDEX_PATH", CORPUS_DIR / "index.faiss"))
METADATA_FILE   = CORPUS_DIR / "metadata.json"
CNN_MODEL_PATH  = Path(os.getenv("CNN_MODEL_PATH", MODELS_DIR / "olive_cnn.pth"))

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
LLM_MODEL           = "llama-3.3-70b-versatile"           # most capable Groq model for Darija
LLM_MAX_TOKENS      = 400                          # keep responses concise

# ── ASR (Whisper) ─────────────────────────────────────────────────────────────
WHISPER_MODEL       = os.getenv("WHISPER_MODEL", "base")  # use large-v3 on GPU
WHISPER_LANGUAGE    = "ar"

# ── TTS (edge-tts) ───────────────────────────────────────────────────────────
# Tunisian Arabic voice — best match for Darija
TTS_VOICE           = "ar-TN-ReemNeural"
TTS_RATE            = "+0%"        # speech speed (+10% faster, -10% slower)
TTS_VOLUME          = "+0%"

# ── RAG / FAISS ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE          = 500          # tokens per chunk
CHUNK_OVERLAP       = 80           # token overlap between chunks
TOP_K               = 5            # number of passages to retrieve
# *** CRITICAL: hallucination guard threshold ***
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.42))

# ── CNN ───────────────────────────────────────────────────────────────────────
CNN_INPUT_SIZE      = 224
CNN_CONFIDENCE_MIN  = 0.50         # below this = "uncertain" → don't assert disease

DISEASE_CLASSES = {
    0: {
        "ar": "سليم",
        "fr": "Sain",
        "en": "Healthy",
        "eppo": None,
        "advice_ar": "الوڤة سليمة، واصل الرعاية الاعتيادية."
    },
    1: {
        "ar": "عنكبوت الزيتون (أكولوس)",
        "fr": "Acariose de l'olivier",
        "en": "Aculus Olearius",
        "eppo": "ACULOS",
        "advice_ar": "حلم الزيتون — يسبب تشوه الأوراق. رشّ بمبيدات الأكاروسات (ديكوفول أو أبامكتين) في الربيع."
    },
    2: {
        "ar": "عين الطاووس",
        "fr": "Œil de paon",
        "en": "Olive Peacock Spot",
        "eppo": "SPIOLEA",
        "advice_ar": "مرض فطري (Spilocaea oleagina) — رشّ بمبيدات النحاس قبل الأمطار الخريفية."
    },
}

# ── Corpus source URLs ────────────────────────────────────────────────────────
CORPUS_SOURCES = [
    {
        "name": "FAO Olive Production Manual",
        "url": "https://www.fao.org/3/y4252e/y4252e.pdf",
        "lang": "en",
        "tag": "FAO"
    },
    {
        "name": "EPPO Spilocaea oleagina",
        "url": "https://gd.eppo.int/taxon/SPIOLEA/documents",
        "lang": "en",
        "tag": "EPPO"
    },
    {
        "name": "CIHEAM Olive Production Med",
        "url": "https://om.ciheam.org/om/pdf/a56/00800108.pdf",
        "lang": "fr",
        "tag": "CIHEAM"
    },
]

# ── Refusal messages (in Darija) ──────────────────────────────────────────────
REFUSAL_MESSAGE = (
    "آسف، ما عنديش المعلومة هاذي في قاعدة البيانات متاعي. "
    "يُنصح تتصل بمرشد زراعي مختص باش يعاونك."
)
REFUSAL_MESSAGE_DOSAGE = (
    "ما نقدرش نعطيك الجرعة الدقيقة — هاذا خطر. "
    "رجع للفيشة التقنية للمبيد أو اتصل بمرشد زراعي."
)