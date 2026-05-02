# 🫒 مساعد الزيتون — Olive Farmer AI Assistant

A multimodal AI assistant for Tunisian olive farmers speaking Darija.  
Voice → Whisper ASR → RAG (anti-hallucination) → Groq LLM → edge-tts output.

## Architecture

```
Camera Photo ──► CNN Classifier ──────────────────────────┐
                                                           ▼
Darija Voice ──► Whisper ASR ──► Text ──► FAISS RAG ──► LLM (strict) ──► edge-tts ──► Voice
                                              │
                                    score < threshold
                                              │
                                              ▼
                                    Polite Refusal (no hallucination)
```

## Project Structure

```
olive-assistant/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # All configuration / thresholds
│   ├── routers/
│   │   ├── chat.py              # POST /api/ask  — main chat endpoint
│   │   ├── vision.py            # POST /api/classify — CNN leaf classifier
│   │   └── audio.py             # POST /api/transcribe, POST /api/tts
│   ├── rag/
│   │   ├── indexer.py           # PDF → chunks → FAISS index builder
│   │   ├── retriever.py         # Semantic search + hallucination guard
│   │   └── prompts.py           # Strict Arabic prompts
│   ├── vision/
│   │   ├── model.py             # CNN model definition (MobileNetV3)
│   │   └── train.py             # Training script (olive disease CNN)
│   └── audio/
│       ├── asr.py               # Whisper wrapper
│       └── tts.py               # edge-tts wrapper
├── corpus/                      # Built at setup time (gitignored)
│   ├── pdfs/                    # Downloaded source PDFs
│   ├── index.faiss              # FAISS vector index
│   └── metadata.json            # Chunk metadata
├── models/                      # Place trained CNN weights here
│   └── olive_cnn.pth            # (not included — train or request)
├── frontend/
│   ├── index.html               # PWA shell
│   ├── manifest.json            # PWA manifest
│   ├── sw.js                    # Service Worker (offline cache)
│   └── src/
│       ├── app.js               # Main app logic
│       ├── camera.js            # Camera capture
│       ├── recorder.js          # Audio recording
│       └── style.css            # RTL Arabic UI styles
├── scripts/
│   ├── download_corpus.py       # Download FAO/CIHEAM PDFs
│   └── evaluate.py              # Jury test protocol runner
├── .env.example                 # Environment variable template
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | Tested on 3.9 |
| ffmpeg | any | Required by Whisper for audio decoding |
| Git | any | |
| Groq API key | — | Free at [console.groq.com](https://console.groq.com) |

**Install ffmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd olive-assistant

python3.9 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
GROQ_API_KEY=gsk_...            # required — get free key at console.groq.com
WHISPER_MODEL=base              # base (CPU-friendly) or large-v3 (GPU, 3GB)
```

All other variables have sensible defaults and can be left as-is.

### 4. Build the knowledge corpus (FAISS index)

This downloads the olive farming PDFs (FAO, CIHEAM) and builds the vector index.  
**Run once — takes 5–10 minutes on first run.**

```bash
# Step 4a: download PDFs
python scripts/download_corpus.py

# Step 4b: build the FAISS index
python -m backend.rag.indexer --pdf_dir corpus/pdfs --out_dir corpus
```

This produces `corpus/index.faiss` and `corpus/metadata.json`.

### 5. (Optional) Place CNN model weights

The vision endpoint (`/api/classify`) needs a trained `olive_cnn.pth` checkpoint.  
Without it the server still starts — the endpoint returns HTTP 503.

```bash
mkdir -p models
# Place your olive_cnn.pth here, or train from scratch:
python backend/vision/train.py --data_dir ./data/olive_disease --epochs 30
```

### 6. Start the backend

```bash
source venv/bin/activate        # if not already active

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 7. Start the frontend

Open a second terminal and serve the frontend on its own port:

```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

The frontend talks to the backend at `http://localhost:8000` (hardcoded in `src/app.js`).  
CORS is open on the backend so cross-origin requests work out of the box.

> **Alternative:** FastAPI also mounts `frontend/` at `/` as a fallback, so  
> `http://localhost:8000` works too if you prefer a single port.

---

## Docker (recommended)

```bash
# Copy and fill in your .env first (step 3 above)
docker-compose up --build

# App (frontend + backend): http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ask` | Main chat — Darija question → Darija answer + TTS |
| `POST` | `/api/classify` | Upload olive leaf image → disease class + advice |
| `POST` | `/api/transcribe` | Upload audio → Darija text (Whisper) |
| `POST` | `/api/tts` | Arabic text → MP3 audio |
| `GET` | `/health` | Service status (asr / cnn / rag loaded?) |
| `GET` | `/docs` | Swagger UI |

---

## Configuration Reference

All settings live in `backend/config.py` and can be overridden via `.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `CNN_MODEL_PATH` | `models/olive_cnn.pth` | Path to trained CNN weights |
| `FAISS_INDEX_PATH` | `corpus/index.faiss` | Path to built FAISS index |
| `SIMILARITY_THRESHOLD` | `0.42` | Anti-hallucination guard — lower = more permissive |
| `TTS_VOICE` | `ar-TN-ReemNeural` | edge-tts voice (Tunisian Arabic) |

### Whisper model guide

| Model | Size | Speed (CPU) | Recommended for |
|---|---|---|---|
| `tiny` | 75 MB | ~5s | Testing only |
| `base` | 145 MB | ~10s | Development / low-resource machines |
| `small` | 465 MB | ~30s | Good accuracy on CPU |
| `large-v3` | 3 GB | ~3 min | Best accuracy — use with GPU |

---

## Anti-Hallucination Protocol

- Cosine similarity threshold: **0.42** (configurable)
- If top-1 FAISS score < threshold → immediate polite refusal, LLM is **never called**
- Strict system prompt: LLM instructed to use **only** retrieved passages
- Every response includes source citation (document + tag)
- Pesticide dosages always redirected to an agronomist

---

## Disease Classes (CNN)

| ID | English | Arabic |
|---|---|---|
| 0 | Healthy | سليم |
| 1 | Peacock Eye | عين الطاووس |
| 2 | Anthracnose | أنثراكنوز |
| 3 | Verticillium Wilt | ذبول الفرتيسيليوم |
| 4 | Cercospora Leaf Spot | تبقع السيركوسبورا |

---

## Jury Test Scenarios

```bash
python scripts/evaluate.py --scenario in_corpus    # 5 questions answered from corpus
python scripts/evaluate.py --scenario out_corpus   # 3 trap questions (should refuse)
python scripts/evaluate.py --scenario photos       # 3 disease photos (CNN)
```
