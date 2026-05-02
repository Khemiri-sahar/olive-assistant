# 🫒 مساعد الزيتون — Olive Farmer AI Assistant

A multimodal AI assistant for Tunisian olive farmers speaking Darija.  
Voice → Whisper ASR → RAG (anti-hallucination) → LLM → edge-tts output.

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
│   │   ├── chat.py              # POST /ask  — main chat endpoint
│   │   ├── vision.py            # POST /classify — CNN leaf classifier
│   │   └── audio.py             # POST /transcribe, GET /tts
│   ├── rag/
│   │   ├── indexer.py           # PDF → chunks → FAISS index builder
│   │   ├── retriever.py         # Semantic search + hallucination guard
│   │   └── prompts.py           # Strict Arabic prompts
│   ├── vision/
│   │   ├── model.py             # CNN model definition (MobileNetV3)
│   │   └── train.py             # Training script (PlantVillage)
│   └── audio/
│       ├── asr.py               # Whisper wrapper
│       └── tts.py               # edge-tts wrapper
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
│   ├── download_corpus.py       # Download & index FAO/EPPO PDFs
│   └── evaluate.py              # Jury test protocol runner
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and index the corpus
```bash
python scripts/download_corpus.py
# This downloads PDFs from FAO, EPPO, CIHEAM and builds the FAISS index
# Produces: corpus/index.faiss + corpus/metadata.json
```

### 3. Train the CNN (or use pretrained weights)
```bash
# Download PlantVillage dataset first, then:
python backend/vision/train.py --data_dir ./data/plantvillage --epochs 30
# OR skip training and use the provided pretrained checkpoint
```

### 4. Run the backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Serve the frontend
```bash
# Any static server works — the PWA is pure HTML/CSS/JS
cd frontend
python -m http.server 3000
# Visit http://localhost:3000
```

### 6. Docker (recommended for jury demo)
```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## Anti-Hallucination Protocol

- Cosine similarity threshold: **0.42** (configurable in config.py)
- If top-1 FAISS score < threshold → immediate polite refusal, LLM never called
- Strict system prompt: LLM instructed to use ONLY retrieved passages
- Every response includes source citation (document + page)
- Pesticide dosages always redirected to agronomist

## Jury Test Scenarios

```bash
python scripts/evaluate.py --scenario in_corpus    # 5 questions in corpus
python scripts/evaluate.py --scenario out_corpus   # 3 trap questions
python scripts/evaluate.py --scenario photos       # 3 disease photos
```

## Disease Classes (CNN)

| Class | Arabic | Code |
|-------|--------|------|
| Healthy | سليم | 0 |
| Peacock Eye | عين الطاووس | 1 |
| Anthracnose | أنثراكنوز | 2 |
| Verticillium | ذبول الفرتيسيليوم | 3 |
| Cercospora | تبقع الزيتون | 4 |

## Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...   # Or any OpenAI-compatible LLM
WHISPER_MODEL=large-v3          # tiny/base/small/medium/large-v3
CNN_MODEL_PATH=models/olive_cnn.pth
FAISS_INDEX_PATH=corpus/index.faiss
SIMILARITY_THRESHOLD=0.42
TTS_VOICE=ar-TN-ReemNeural      # Tunisian Arabic voice
```