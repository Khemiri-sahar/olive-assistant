/**
 * app.js — Olive Assistant PWA main logic
 *
 * Pipeline:
 *   Camera capture → POST /api/classify (CNN)
 *   Mic record → MediaRecorder → POST /api/transcribe (Whisper)
 *   Text/transcript + disease_id → POST /api/ask (RAG + LLM)
 *   Response audio (base64 MP3) → Audio playback
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
// Use dynamic host for mobile testing. If served via 3000, point to 8000.
const API_BASE = window.location.port === '3000' || window.location.port === '5173'
  ? window.location.protocol + '//' + window.location.hostname + ':8000'
  : '';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  currentDisease: null,   // {class_id, class_ar, class_name, confidence, ...}
  mediaRecorder:  null,
  audioChunks:    [],
  cameraStream:   null,
  isRecording:    false,
  lastAudioB64:   null,   // for playback
};

// ── Tab navigation ─────────────────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const content = document.getElementById(`tab-${tabId}`);
  const btn = document.querySelector(`[data-tab="${tabId}"]`);
  if (content) content.classList.add('active');
  if (btn)     btn.classList.add('active');
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    const video = document.getElementById('camera-video');
    video.srcObject = state.cameraStream;
    video.style.display = 'block';
    document.getElementById('camera-placeholder').style.display = 'none';
    document.getElementById('preview-img').style.display = 'none';
    document.getElementById('live-controls').style.display = 'flex';
    document.querySelector('.camera-controls').style.display = 'none';
    document.getElementById('cnn-result').style.display = 'none';
    document.getElementById('low-conf-warning').style.display = 'none';
  } catch (err) {
    alert('تعذّر فتح الكاميرا: ' + err.message);
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  document.getElementById('camera-video').style.display = 'none';
  document.getElementById('camera-placeholder').style.display = 'block';
  document.getElementById('live-controls').style.display = 'none';
  document.querySelector('.camera-controls').style.display = 'flex';
}

function capturePhoto() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('capture-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  stopCamera();
  canvas.toBlob(blob => classifyLeaf(blob), 'image/jpeg', 0.9);

  // Show preview
  const img = document.getElementById('preview-img');
  img.src = canvas.toDataURL('image/jpeg');
  img.style.display = 'block';
  document.getElementById('btn-clear-photo').style.display = 'block';
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('camera-placeholder').style.display = 'none';
    document.getElementById('btn-clear-photo').style.display = 'block';
  };
  reader.readAsDataURL(file);
  classifyLeaf(file);
}

// ── CNN Classification ─────────────────────────────────────────────────────────
async function classifyLeaf(imageBlob) {
  document.getElementById('cnn-result').style.display = 'none';
  document.getElementById('low-conf-warning').style.display = 'none';

  const formData = new FormData();
  formData.append('file', imageBlob, 'leaf.jpg');

  try {
    const res  = await fetch(`${API_BASE}/api/classify`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Classification failed');

    state.currentDisease = data;
    displayCNNResult(data);
  } catch (err) {
    console.error('CNN error:', err);
    showAlert('خطأ في التصنيف: ' + err.message);
  }
}

function displayCNNResult(data) {
  const badge   = document.getElementById('disease-badge');
  const conf    = document.getElementById('conf-score');
  const resAr   = document.getElementById('result-ar');
  const resEppo = document.getElementById('result-eppo');
  const advice  = document.getElementById('result-advice');

  badge.textContent = data.class_ar;
  badge.className = 'result-badge' + (data.class_id === 0 ? ' healthy' : '');
  conf.textContent = Math.round(data.confidence * 100) + '%';
  resAr.textContent = `${data.class_ar} (${data.class_fr})`;
  resEppo.textContent = data.eppo_code ? `EPPO: ${data.eppo_code}` : '';
  advice.textContent = data.advice_ar;

  document.getElementById('cnn-result').style.display = 'block';

  if (data.low_conf) {
    document.getElementById('low-conf-warning').style.display = 'block';
  }
}

function proceedToAsk() {
  const disease = state.currentDisease;
  if (!disease) return;
  const question = `شنوة هاذا المرض اللي في زيتوني ؟ شحال هو خطير وكيفاش نعالجه ؟`;
  submitQuestion(question);
}

// ── Audio Recording ────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (state.isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioChunks = [];
    state.mediaRecorder = new MediaRecorder(stream, {
      mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/mp4',
    });

    state.mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) state.audioChunks.push(e.data);
    };

    state.mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
      await transcribeAudio(audioBlob);
    };

    state.mediaRecorder.start(250);   // collect chunks every 250ms
    state.isRecording = true;

    document.getElementById('mic-btn').classList.add('recording');
    document.getElementById('mic-label').textContent = 'جاري التسجيل... اضغط للإيقاف';
    document.getElementById('recording-wave').classList.add('active');
    document.getElementById('transcript-box').style.display = 'none';
  } catch (err) {
    alert('تعذّر الوصول للميكروفون: ' + err.message);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.isRecording) {
    state.mediaRecorder.stop();
    state.isRecording = false;
    document.getElementById('mic-btn').classList.remove('recording');
    document.getElementById('mic-label').textContent = 'جاري التحليل...';
    document.getElementById('recording-wave').classList.remove('active');
  }
}

// ── Whisper Transcription ──────────────────────────────────────────────────────
async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');

  try {
    const res  = await fetch(`${API_BASE}/api/transcribe`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Transcription failed');

    document.getElementById('transcript-text').textContent = data.text;
    document.getElementById('transcript-box').style.display = 'block';
    document.getElementById('mic-label').textContent = 'اضغط وتكلّم بالدارجة';
  } catch (err) {
    console.error('Transcription error:', err);
    document.getElementById('mic-label').textContent = 'اضغط وتكلّم بالدارجة';
    showAlert('خطأ في التعرف على الصوت: ' + err.message);
  }
}

function submitTranscript() {
  const text = document.getElementById('transcript-text').textContent.trim();
  if (text) submitQuestion(text);
}

function submitText() {
  const text = document.getElementById('text-input').value.trim();
  if (text) {
    submitQuestion(text);
    document.getElementById('text-input').value = '';
  }
}

function clearPhoto() {
  const img = document.getElementById('preview-img');
  img.src = '';
  img.style.display = 'none';
  document.getElementById('camera-placeholder').style.display = 'block';
  document.getElementById('cnn-result').style.display = 'none';
  document.getElementById('low-conf-warning').style.display = 'none';
  document.querySelector('.camera-controls').style.display = 'flex';
  document.getElementById('file-input').value = '';
  state.currentDisease = null;
}

const NEEDS_IMAGE_KEYWORDS = ['هاذي المرض', 'هاذا المرض', 'هاذي الوڤة', 'هاذي الوقة', 'هاذا اللي في زيتوني'];

function askText(question) {
  const needsImage = NEEDS_IMAGE_KEYWORDS.some(kw => question.includes(kw));
  if (needsImage && !state.currentDisease) {
    switchTab('scan');
    showAlert('صوّر الوڤة أولاً باش نقدر نشخص المرض 📷');
    return;
  }
  submitQuestion(question);
}

// ── Main ask pipeline ──────────────────────────────────────────────────────────
async function submitQuestion(question) {
  switchTab('result');
  showLoading(true);

  const payload = {
    question:    question,
    disease_id:  state.currentDisease ? state.currentDisease.class_id : null,
    tts_enabled: true,
  };

  setLoadingStep('جاري البحث في قاعدة البيانات...', 30);

  try {
    const res  = await fetch(`${API_BASE}/api/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Request failed');

    setLoadingStep('جاري توليد الجواب...', 80);

    await new Promise(r => setTimeout(r, 300));   // brief UX pause

    setLoadingStep('اكتمل!', 100);
    await new Promise(r => setTimeout(r, 200));

    showLoading(false);
    displayAnswer(data);
  } catch (err) {
    console.error('Ask error:', err);
    showLoading(false);
    showFallbackError(err.message);
  }
}

function displayAnswer(data) {
  const loading  = document.getElementById('result-loading');
  const answer   = document.getElementById('result-answer');
  const refused  = document.getElementById('result-refused');
  const empty    = document.getElementById('result-empty');

  loading.style.display = 'none';
  answer.style.display  = 'none';
  refused.style.display = 'none';
  empty.style.display   = 'none';

  if (data.refused) {
    // ── Refusal path ────────────────────────────────────────────
    document.getElementById('refusal-text').textContent = data.answer;
    refused.style.display = 'block';

    if (data.audio_b64) {
      state.lastAudioB64 = data.audio_b64;
      const audioEl = document.getElementById('audio-refused');
      audioEl.src = `data:audio/mpeg;base64,${data.audio_b64}`;
      document.getElementById('audio-player-refused').style.display = 'block';
      audioEl.play().catch(() => {});
    }
  } else {
    // ── Answer path ─────────────────────────────────────────────
    const scorePct = Math.round(data.top_score * 100);
    document.getElementById('score-fill').style.width = scorePct + '%';
    document.getElementById('score-pct').textContent  = scorePct + '%';
    document.getElementById('answer-text').textContent = data.answer;

    // Citations
    const citList = document.getElementById('citations-list');
    citList.innerHTML = '';
    (data.citations || []).forEach(cit => {
      const li = document.createElement('li');
      li.textContent = cit;
      citList.appendChild(li);
    });
    document.getElementById('citations-box').style.display =
      data.citations?.length ? 'block' : 'none';

    // Audio
    if (data.audio_b64) {
      state.lastAudioB64 = data.audio_b64;
      const audioEl = document.getElementById('audio-el');
      const btn = document.getElementById('btn-play');
      audioEl.src = `data:audio/mpeg;base64,${data.audio_b64}`;
      audioEl.onended = () => { btn.textContent = '🔊 استمع للجواب'; };
      audioEl.onpause = () => { btn.textContent = '🔊 استمع للجواب'; };
      document.getElementById('audio-player').style.display = 'block';
      audioEl.play().then(() => {
        btn.textContent = '⏸ إيقاف';
      }).catch(() => {});
    } else {
      document.getElementById('audio-player').style.display = 'none';
    }

    answer.style.display = 'block';
  }
}

function playAudio() {
  const audio = document.getElementById('audio-el');
  const btn = document.getElementById('btn-play');
  if (!state.lastAudioB64) return;
  if (!audio.paused) {
    audio.pause();
    return;
  }
  audio.play().then(() => {
    btn.textContent = '⏸ إيقاف';
  }).catch(() => {});
}

function playRefusedAudio() {
  document.getElementById('audio-refused').play();
}

// ── Loading helpers ───────────────────────────────────────────────────────────
function showLoading(show) {
  const loading = document.getElementById('result-loading');
  const answer  = document.getElementById('result-answer');
  const refused = document.getElementById('result-refused');
  const empty   = document.getElementById('result-empty');

  loading.style.display = show ? 'flex' : 'none';
  if (show) {
    answer.style.display  = 'none';
    refused.style.display = 'none';
    empty.style.display   = 'none';
    document.getElementById('progress-fill').style.width = '5%';
  }
}

function setLoadingStep(text, pct) {
  document.getElementById('loading-step').textContent = text;
  document.getElementById('progress-fill').style.width = pct + '%';
}

function showFallbackError(msg) {
  document.getElementById('refusal-text').textContent =
    `حدث خطأ تقني: ${msg}\n\nتحقق من الاتصال بالإنترنت وحاول مجدداً.`;
  document.getElementById('result-refused').style.display = 'block';
}

// ── Utility ────────────────────────────────────────────────────────────────────
function showAlert(msg) {
  // Simple in-app alert (avoid browser alert() for mobile UX)
  const div = document.createElement('div');
  div.style.cssText = `
    position: fixed; bottom: 1rem; left: 1rem; right: 1rem;
    background: #1a1a1a; color: white; padding: 0.85rem 1rem;
    border-radius: 10px; font-size: 0.85rem; z-index: 1000;
    direction: rtl; text-align: right; font-family: Tajawal, sans-serif;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  `;
  div.textContent = msg;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 4000);
}

// ── Health check ────────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    const dot  = document.getElementById('status-dot');
    const allOk = data.asr && data.cnn && data.rag;
    dot.style.background   = allOk ? '#4ade80' : '#f59e0b';
    dot.style.boxShadow    = allOk ? '0 0 6px #4ade80' : '0 0 6px #f59e0b';
    dot.title = `ASR: ${data.asr ? '✅' : '❌'} CNN: ${data.cnn ? '✅' : '❌'} RAG: ${data.rag ? '✅' : '❌'}`;
  } catch {
    const dot = document.getElementById('status-dot');
    dot.style.background = '#ef4444';
    dot.style.boxShadow  = '0 0 6px #ef4444';
    dot.title = 'غير متصل بالخادم';
  }
}

// Run health check on load and every 30 seconds
checkHealth();
setInterval(checkHealth, 30000);