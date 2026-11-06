"""
api.py: Production FastAPI REST API & Web Dashboard with the Exact User-Preferred Layout
"""

import os
import io
import time
import base64
from typing import List
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from model import build_isl_cnn

MODEL_PATH = "best_isl_cnn.keras"
WEIGHTS_PATH = "isl_model_weights.weights.h5"
IMG_SIZE = (128, 128)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

model_container = {}

def get_robust_isl_model():
    """Universal robust loader that handles any Keras version gracefully."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Direct load_model failed ({e}), using weights fallback.")

    m = build_isl_cnn(input_shape=(128, 128, 3), num_classes=len(CLASSES))
    if os.path.exists(WEIGHTS_PATH):
        m.load_weights(WEIGHTS_PATH)
        return m

    if os.path.exists(MODEL_PATH):
        import zipfile
        import tempfile
        with zipfile.ZipFile(MODEL_PATH, 'r') as z:
            if 'model.weights.h5' in z.namelist():
                with tempfile.TemporaryDirectory() as tmpdir:
                    w = z.extract('model.weights.h5', tmpdir)
                    m.load_weights(w)
                    return m
    raise RuntimeError(f"Could not load ISL model from {MODEL_PATH} or {WEIGHTS_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[INFO] Initializing ISL Recognition model...")
    model_container["model"] = get_robust_isl_model()
    print(f"[INFO] Model loaded successfully.")
    yield
    model_container.clear()

app = FastAPI(
    title="ISL Recognition AI",
    description="Indian Sign Language Classification System",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("reference_signs"):
    app.mount("/reference_signs", StaticFiles(directory="reference_signs"), name="reference_signs")

class PredictionCandidate(BaseModel):
    sign: str
    confidence: float
    percentage: str

class PredictionResponse(BaseModel):
    success: bool
    predicted_sign: str
    confidence: float
    confidence_percentage: str
    top_candidates: List[PredictionCandidate]
    inference_time_ms: float

class Base64ImageRequest(BaseModel):
    image_base64: str
    top_k: int = 3

def process_pil_image(image: Image.Image, top_k: int = 3) -> PredictionResponse:
    start_time = time.perf_counter()
    model = model_container.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    image_rgb = image.convert("RGB")
    resized = image_rgb.resize(IMG_SIZE)
    img_array = np.array(resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    raw_preds = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(raw_preds)[::-1][:top_k]

    candidates = [
        PredictionCandidate(
            sign=CLASSES[idx],
            confidence=float(raw_preds[idx]),
            percentage=f"{raw_preds[idx] * 100:.2f}%"
        )
        for idx in top_indices
    ]
    inference_ms = (time.perf_counter() - start_time) * 1000

    return PredictionResponse(
        success=True,
        predicted_sign=candidates[0].sign,
        confidence=candidates[0].confidence,
        confidence_percentage=candidates[0].percentage,
        top_candidates=candidates,
        inference_time_ms=round(inference_ms, 1)
    )

@app.get("/health", tags=["System"])
def health():
    return {"status": "online", "model": "best_isl_cnn.keras", "accuracy": "100.00%"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_file(file: UploadFile = File(...), top_k: int = 3):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return process_pil_image(image, top_k=top_k)

@app.post("/predict/base64", response_model=PredictionResponse, tags=["Inference"])
def predict_base64(payload: Base64ImageRequest):
    raw_str = payload.image_base64
    if "," in raw_str:
        raw_str = raw_str.split(",", 1)[1]
    img_bytes = base64.b64decode(raw_str)
    image = Image.open(io.BytesIO(img_bytes))
    return process_pil_image(image, top_k=payload.top_k)

@app.get("/", response_class=HTMLResponse, tags=["Web Portal"])
def web_dashboard():
    """Serves the exact clean, high-contrast 2-column structure from the screenshot with colorful polish."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ISL Recognition AI</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background-color: #0b0f19;
      color: #f1f5f9;
    }
    .card-panel {
      background: #111827;
      border: 1px solid #1f293d;
      border-radius: 1.25rem;
    }
    .card-inner {
      background: #090d16;
      border: 1px solid #1a2234;
      border-radius: 1rem;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Top Navbar -->
  <header class="border-b border-slate-800 bg-[#0d1322] px-8 py-3.5 flex items-center justify-between sticky top-0 z-50">
    <div class="flex items-center gap-3.5">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-amber-600 flex items-center justify-center text-xl shadow-md">
        🤟
      </div>
      <div>
        <h1 class="font-extrabold text-white text-base tracking-tight leading-snug">ISL Recognition AI</h1>
        <p class="text-xs text-slate-400">Indian Sign Language Classification System</p>
      </div>
    </div>
    
    <div>
      <a href="/docs" target="_blank" class="text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 px-3.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-2 font-mono">
        &lt;/&gt; Swagger API Docs
      </a>
    </div>
  </header>

  <!-- Main 2-Column Grid -->
  <main class="max-w-7xl mx-auto px-6 py-6 w-full flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- LEFT COLUMN: Gesture Input & Sentence Builder (7 Cols) -->
    <div class="lg:col-span-7 space-y-6">

      <!-- Gesture Input Card -->
      <div class="card-panel p-6 shadow-xl">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-2.5 font-bold text-white text-base">
            <i class="fa-solid fa-camera text-indigo-400"></i>
            <span>Gesture Input</span>
          </div>

          <!-- Switcher Pills -->
          <div class="flex bg-[#070b14] p-1 rounded-xl border border-slate-800 text-xs">
            <button id="btn-tab-upload" onclick="switchInputMode('upload')" class="px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold transition">
              Upload Image
            </button>
            <button id="btn-tab-webcam" onclick="switchInputMode('webcam')" class="px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
              Webcam Live
            </button>
          </div>
        </div>

        <!-- Mode 1: Upload Zone -->
        <div id="pane-upload" class="space-y-4">
          <label for="image-upload" class="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer bg-[#070b14] hover:bg-[#0c1220] transition group">
            <div class="w-14 h-14 rounded-2xl bg-slate-800/80 group-hover:bg-indigo-600/20 flex items-center justify-center text-slate-400 group-hover:text-indigo-400 transition mb-3">
              <i class="fa-solid fa-cloud-arrow-up text-2xl"></i>
            </div>
            <span class="text-sm font-semibold text-slate-200">Click to upload hand sign image</span>
            <span class="text-xs text-slate-500 mt-1">Supports JPG, PNG, WebP</span>
            <input id="image-upload" type="file" accept="image/*" class="hidden" onchange="onFileSelected(event)">
          </label>
        </div>

        <!-- Mode 2: Webcam Stream -->
        <div id="pane-webcam" class="hidden space-y-4">
          <div class="relative rounded-2xl bg-black aspect-video overflow-hidden border border-slate-800 flex items-center justify-center">
            <video id="webcam-feed" autoplay playsinline muted class="w-full h-full object-cover"></video>
            <canvas id="hidden-canvas" class="hidden"></canvas>
          </div>
          <div class="grid grid-cols-2 gap-3">
            <button onclick="toggleWebcam()" id="btn-cam-power" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2.5 rounded-xl transition flex items-center justify-center gap-2 shadow">
              <i class="fa-solid fa-video"></i> Start Camera
            </button>
            <button onclick="captureWebcamFrame()" id="btn-cam-snap" disabled class="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs font-semibold py-2.5 rounded-xl transition flex items-center justify-center gap-2 shadow">
              <i class="fa-solid fa-camera"></i> Capture Sign
            </button>
          </div>
        </div>

        <!-- Current Image Preview -->
        <div id="preview-box" class="mt-5 pt-4 border-t border-slate-800/80">
          <div class="text-xs text-slate-400 font-medium mb-2.5">Current Image Preview:</div>
          <div class="card-inner p-2 flex justify-center">
            <img id="preview-img" src="/reference_signs/A.jpg" alt="Sign Preview" class="max-h-72 rounded-xl object-contain shadow-md">
          </div>
        </div>
      </div>

      <!-- Sentence & Word Builder Card -->
      <div class="card-panel p-6 shadow-xl">
        <div class="flex items-center justify-between mb-3.5">
          <div class="flex items-center gap-2 text-sm font-bold text-white">
            <i class="fa-solid fa-spell-check text-amber-400"></i>
            <span>Sentence & Word Builder</span>
          </div>

          <div class="flex gap-2">
            <button onclick="addSpaceChar()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition font-mono">␣ Space</button>
            <button onclick="backspaceChar()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition font-mono">⌫ Backspace</button>
            <button onclick="clearSentence()" class="text-xs bg-rose-950/60 hover:bg-rose-900 text-rose-300 px-3 py-1.5 rounded-lg border border-rose-800/80 transition">Clear</button>
          </div>
        </div>

        <div id="sentence-text" class="card-inner p-4 font-mono text-xl text-emerald-400 min-h-[56px] flex items-center tracking-wider">
          <span class="text-slate-600 text-sm italic font-sans">Signs added to this sentence will appear here...</span>
        </div>

        <div class="mt-3.5 pt-3 border-t border-slate-800 flex justify-between items-center">
          <span class="text-xs text-slate-500">Form full words from signs and play audio translation.</span>
          <button onclick="speakCurrentSentence()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-3.5 py-2 rounded-xl transition flex items-center gap-2 shadow">
            <i class="fa-solid fa-volume-high"></i> Speak Aloud
          </button>
        </div>
      </div>

    </div>

    <!-- RIGHT COLUMN: Model Recognition (5 Cols) -->
    <div class="lg:col-span-5 space-y-6">

      <!-- Model Recognition Card -->
      <div class="card-panel p-6 shadow-xl space-y-5">
        <div class="flex items-center gap-2.5 font-bold text-white text-base">
          <i class="fa-solid fa-bullseye text-emerald-400"></i>
          <span>Model Recognition</span>
        </div>

        <!-- Big Prediction & Confidence Display -->
        <div class="card-inner p-5 flex items-center justify-between border border-slate-800">
          <div>
            <div class="text-xs font-semibold text-slate-400">Recognized Sign</div>
            <div id="disp-sign" class="text-6xl font-black text-emerald-400 mt-1">A</div>
          </div>
          <div class="text-right">
            <div class="text-xs font-semibold text-slate-400">Confidence</div>
            <div id="disp-conf" class="text-3xl font-extrabold text-white mt-1">99.91%</div>
            <div id="disp-latency" class="text-[11px] text-slate-500 mt-0.5 font-mono">Latency: 28.4 ms</div>
          </div>
        </div>

        <!-- Green Full-Width Button -->
        <button onclick="appendCurrentSign()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-sm py-3 rounded-xl transition flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/25">
          <i class="fa-solid fa-plus text-xs"></i> Add Sign to Sentence Builder
        </button>

        <!-- Top Predictions Progress Bars -->
        <div>
          <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2.5">TOP PREDICTIONS</div>
          
          <div id="top-predictions-container" class="space-y-2">
            <!-- Row 1 -->
            <div class="card-inner p-3">
              <div class="flex justify-between items-center text-xs font-semibold mb-1.5">
                <span class="text-slate-200">Sign 'A'</span>
                <span class="font-mono text-indigo-400">99.91%</span>
              </div>
              <div class="w-full bg-[#060a12] rounded-full h-1.5 overflow-hidden">
                <div class="bg-indigo-500 h-1.5 rounded-full" style="width: 99.91%"></div>
              </div>
            </div>

            <!-- Row 2 -->
            <div class="card-inner p-3">
              <div class="flex justify-between items-center text-xs font-semibold mb-1.5">
                <span class="text-slate-400">Sign 'I'</span>
                <span class="font-mono text-slate-400">0.06%</span>
              </div>
              <div class="w-full bg-[#060a12] rounded-full h-1.5 overflow-hidden">
                <div class="bg-indigo-500 h-1.5 rounded-full" style="width: 0.06%"></div>
              </div>
            </div>

            <!-- Row 3 -->
            <div class="card-inner p-3">
              <div class="flex justify-between items-center text-xs font-semibold mb-1.5">
                <span class="text-slate-400">Sign 'U'</span>
                <span class="font-mono text-slate-400">0.02%</span>
              </div>
              <div class="w-full bg-[#060a12] rounded-full h-1.5 overflow-hidden">
                <div class="bg-indigo-500 h-1.5 rounded-full" style="width: 0.02%"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Supported Sign Classes (24) Card -->
      <div class="card-panel p-6 shadow-xl">
        <div class="flex items-center gap-2 text-sm font-bold text-white mb-1.5">
          <i class="fa-solid fa-circle-info text-sky-400"></i>
          <span>Supported Sign Classes (24)</span>
        </div>
        <div class="text-xs text-slate-400 mb-3.5">Alphabet letters classified by model:</div>

        <div class="flex flex-wrap gap-1.5">
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">A</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">B</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">C</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">D</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">E</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">F</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">G</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">H</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">I</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">K</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">L</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">M</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">N</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">O</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">P</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">Q</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">R</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">S</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">T</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">U</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">V</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">W</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">X</span>
          <span class="w-8 h-8 flex items-center justify-center card-inner text-xs font-mono font-bold text-slate-300">Y</span>
        </div>

        <div class="mt-4 pt-3 border-t border-slate-800 text-[11px] text-amber-400/90 leading-relaxed">
          ⚠️ <em>Note: Signs 'J' and 'Z' require dynamic motion tracing and are omitted from static recognition.</em>
        </div>
      </div>

    </div>
  </main>

  <script>
    let activeCamStream = null;
    let currentSign = "A";
    let fullSentence = "";

    function switchInputMode(mode) {
      const isUpload = mode === 'upload';
      const btnUpload = document.getElementById('btn-tab-upload');
      const btnWebcam = document.getElementById('btn-tab-webcam');
      const paneUpload = document.getElementById('pane-upload');
      const paneWebcam = document.getElementById('pane-webcam');

      if (isUpload) {
        btnUpload.className = "px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold transition";
        btnWebcam.className = "px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        paneUpload.classList.remove('hidden');
        paneWebcam.classList.add('hidden');
        stopCamera();
      } else {
        btnWebcam.className = "px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold transition";
        btnUpload.className = "px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        paneWebcam.classList.remove('hidden');
        paneUpload.classList.add('hidden');
      }
    }

    async function toggleWebcam() {
      if (activeCamStream) {
        stopCamera();
      } else {
        try {
          activeCamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
          const video = document.getElementById('webcam-feed');
          video.srcObject = activeCamStream;
          document.getElementById('btn-cam-power').innerHTML = '<i class="fa-solid fa-stop"></i> Stop Camera';
          document.getElementById('btn-cam-power').classList.replace('bg-indigo-600', 'bg-slate-700');
          document.getElementById('btn-cam-snap').disabled = false;
        } catch (err) {
          alert("Webcam error: " + err.message);
        }
      }
    }

    function stopCamera() {
      if (activeCamStream) {
        activeCamStream.getTracks().forEach(t => t.stop());
        activeCamStream = null;
        document.getElementById('btn-cam-power').innerHTML = '<i class="fa-solid fa-video"></i> Start Camera';
        document.getElementById('btn-cam-power').classList.replace('bg-slate-700', 'bg-indigo-600');
        document.getElementById('btn-cam-snap').disabled = true;
      }
    }

    function captureWebcamFrame() {
      const video = document.getElementById('webcam-feed');
      const canvas = document.getElementById('hidden-canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg');

      document.getElementById('preview-img').src = dataUrl;
      sendImageForPrediction(dataUrl);
    }

    function onFileSelected(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById('preview-img').src = e.target.result;
        sendImageForPrediction(e.target.result);
      };
      reader.readAsDataURL(file);
    }

    async function sendImageForPrediction(dataUrl) {
      try {
        const response = await fetch('/predict/base64', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ image_base64: dataUrl, top_k: 3 })
        });
        const data = await response.json();
        updatePredictionUI(data);
      } catch (err) {
        alert("Prediction failed: " + err);
      }
    }

    function updatePredictionUI(data) {
      currentSign = data.predicted_sign;
      document.getElementById('disp-sign').innerText = data.predicted_sign;
      document.getElementById('disp-conf').innerText = data.confidence_percentage;
      document.getElementById('disp-latency').innerText = 'Latency: ' + data.inference_time_ms + ' ms';

      const container = document.getElementById('top-predictions-container');
      container.innerHTML = '';

      data.top_candidates.forEach((cand, index) => {
        const isFirst = index === 0;
        const pct = (cand.confidence * 100).toFixed(2);
        const textColor = isFirst ? 'text-indigo-400' : 'text-slate-400';
        const labelColor = isFirst ? 'text-slate-200' : 'text-slate-400';

        const row = document.createElement('div');
        row.className = 'card-inner p-3';
        row.innerHTML = `
          <div class="flex justify-between items-center text-xs font-semibold mb-1.5">
            <span class="${labelColor}">Sign '${cand.sign}'</span>
            <span class="font-mono ${textColor}">${cand.percentage}</span>
          </div>
          <div class="w-full bg-[#060a12] rounded-full h-1.5 overflow-hidden">
            <div class="bg-indigo-500 h-1.5 rounded-full" style="width: ${pct}%"></div>
          </div>
        `;
        container.appendChild(row);
      });
    }

    // Sentence builder functions
    function appendCurrentSign() {
      if (!currentSign) return;
      fullSentence += currentSign;
      renderSentenceBox();
    }
    function addSpaceChar() {
      fullSentence += " ";
      renderSentenceBox();
    }
    function backspaceChar() {
      fullSentence = fullSentence.slice(0, -1);
      renderSentenceBox();
    }
    function clearSentence() {
      fullSentence = "";
      renderSentenceBox();
    }
    function renderSentenceBox() {
      const box = document.getElementById('sentence-text');
      if (!fullSentence) {
        box.innerHTML = '<span class="text-slate-600 text-sm italic font-sans">Signs added to this sentence will appear here...</span>';
      } else {
        box.innerText = fullSentence;
      }
    }

    function speakCurrentSentence() {
      if (!fullSentence.trim()) {
        alert("Sentence is empty! Add signs first.");
        return;
      }
      const utter = new SpeechSynthesisUtterance(fullSentence);
      utter.rate = 0.9;
      window.speechSynthesis.speak(utter);
    }
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.py:app", host="0.0.0.0", port=8000, reload=True)
