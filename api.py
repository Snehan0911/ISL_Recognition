"""
api.py: Production FastAPI REST API Backend for Indian Sign Language (ISL) Recognition
Includes Swagger Docs, CORS Middleware, Base64 & Multipart File Endpoints, and Static Web UI Serving.
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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Configuration ---
MODEL_PATH = "best_isl_cnn.keras"
IMG_SIZE = (128, 128)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

model_container = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on server startup."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please run train.py first.")
    print(f"[INFO] Loading trained model from {MODEL_PATH}...")
    model_container["model"] = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded successfully with {len(CLASSES)} classes.")
    yield
    model_container.clear()

app = FastAPI(
    title="Indian Sign Language (ISL) Recognition API",
    description="High-performance REST API for real-time ISL alphabet recognition from images.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for browser access from any domain/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Request/Response Models ---
class PredictionCandidate(BaseModel):
    sign: str = Field(..., description="Recognized ISL alphabet letter")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    percentage: str = Field(..., description="Confidence expressed as formatted percentage")

class PredictionResponse(BaseModel):
    success: bool
    predicted_sign: str
    confidence: float
    confidence_percentage: str
    top_candidates: List[PredictionCandidate]
    inference_time_ms: float

class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string (with or without data URI prefix)")
    top_k: int = Field(3, ge=1, le=24, description="Number of top candidate classes to return")

# --- Helper Utilities ---
def process_pil_image(image: Image.Image, top_k: int = 3) -> PredictionResponse:
    start_time = time.perf_counter()
    model = model_container.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Resize to model input shape
    image_rgb = image.convert("RGB")
    resized = image_rgb.resize(IMG_SIZE)
    img_array = np.array(resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Perform inference
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
        inference_time_ms=round(inference_ms, 2)
    )

# --- REST Endpoints ---
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for cloud monitoring and load balancers."""
    return {
        "status": "healthy",
        "model_loaded": "model" in model_container,
        "classes_count": len(CLASSES),
        "supported_classes": CLASSES
    }

@app.get("/classes", tags=["Metadata"])
def get_supported_classes():
    """Returns the list of supported ISL signs and omitted dynamic letters."""
    return {
        "total_classes": len(CLASSES),
        "classes": CLASSES,
        "omitted_dynamic_signs": ["J", "Z"],
        "reason_for_omission": "J and Z are dynamic motion gestures that require spatio-temporal video tracking."
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_file(file: UploadFile = File(...), top_k: int = 3):
    """
    Accepts an uploaded image file (JPEG, PNG, WebP) and predicts the ISL hand sign.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image."
        )
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )
    
    return process_pil_image(image, top_k=top_k)

@app.post("/predict/base64", response_model=PredictionResponse, tags=["Inference"])
def predict_base64(payload: Base64ImageRequest):
    """
    Accepts a base64 encoded image string (ideal for webcams and mobile frontends).
    """
    raw_str = payload.image_base64
    if "," in raw_str:
        raw_str = raw_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_str)
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode base64 image: {str(e)}"
        )

    return process_pil_image(image, top_k=payload.top_k)

# --- Beautiful Built-in Web Portal ---
@app.get("/", response_class=HTMLResponse, tags=["Web Portal"])
def web_dashboard():
    """Serves a modern standalone responsive web interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indian Sign Language (ISL) Recognition</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  <!-- Navbar -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-indigo-600 flex items-center justify-center text-xl shadow-lg shadow-indigo-500/20">
          🤟
        </div>
        <div>
          <h1 class="font-bold text-lg text-white leading-tight">ISL Recognition AI</h1>
          <p class="text-xs text-slate-400">Indian Sign Language Classification System</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <a href="/docs" target="_blank" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-2">
          <i class="fa-solid fa-code"></i> Swagger API Docs
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto px-4 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
    
    <!-- Left Column: Input Methods -->
    <div class="lg:col-span-7 space-y-6">
      
      <!-- Input Mode Switcher -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div class="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
          <h2 class="font-semibold text-slate-200 flex items-center gap-2 text-base">
            <i class="fa-solid fa-camera text-indigo-400"></i> Gesture Input
          </h2>
          <div class="flex bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
            <button id="tab-upload" onclick="switchTab('upload')" class="px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-medium transition">Upload Image</button>
            <button id="tab-camera" onclick="switchTab('camera')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">Webcam Live</button>
          </div>
        </div>

        <!-- Upload Pane -->
        <div id="pane-upload" class="space-y-4">
          <label for="image-input" class="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer bg-slate-950/50 hover:bg-slate-950 transition group">
            <i class="fa-solid fa-cloud-arrow-up text-4xl text-slate-500 group-hover:text-indigo-400 mb-3 transition"></i>
            <span class="text-sm font-medium text-slate-300">Click to upload hand sign image</span>
            <span class="text-xs text-slate-500 mt-1">Supports JPG, PNG, WebP</span>
            <input id="image-input" type="file" accept="image/*" class="hidden" onchange="handleFileSelect(event)">
          </label>
        </div>

        <!-- Camera Pane -->
        <div id="pane-camera" class="hidden space-y-4">
          <div class="relative rounded-2xl overflow-hidden bg-black aspect-video flex items-center justify-center border border-slate-800">
            <video id="webcam-video" autoplay playsinline class="w-full h-full object-cover"></video>
            <canvas id="snapshot-canvas" class="hidden"></canvas>
          </div>
          <div class="flex gap-3">
            <button onclick="startWebcam()" id="btn-start-cam" class="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold py-2.5 rounded-xl transition">
              <i class="fa-solid fa-video mr-1"></i> Start Camera
            </button>
            <button onclick="captureSnapshot()" id="btn-capture-cam" disabled class="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold py-2.5 rounded-xl transition">
              <i class="fa-solid fa-camera mr-1"></i> Capture Sign
            </button>
          </div>
        </div>

        <!-- Preview Image Box -->
        <div id="preview-container" class="hidden mt-4 pt-4 border-t border-slate-800">
          <p class="text-xs text-slate-400 mb-2">Current Image Preview:</p>
          <img id="preview-image" src="" class="max-h-64 rounded-xl mx-auto border border-slate-700 shadow-md object-contain">
        </div>
      </div>

      <!-- Sentence Builder -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold text-slate-200 text-sm flex items-center gap-2">
            <i class="fa-solid fa-spell-check text-amber-400"></i> Sentence & Word Builder
          </h3>
          <div class="flex gap-2">
            <button onclick="addSpace()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">Space</button>
            <button onclick="backspaceWord()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">Backspace</button>
            <button onclick="clearWord()" class="text-xs bg-red-950/60 hover:bg-red-900 text-red-300 px-2.5 py-1 rounded-lg border border-red-800 transition">Clear</button>
          </div>
        </div>
        <div id="sentence-display" class="bg-slate-950 border border-slate-800 p-4 rounded-xl font-mono text-lg text-emerald-400 min-h-[56px] flex items-center tracking-wider">
          <span class="text-slate-600 text-sm italic">Signs added to this sentence will appear here...</span>
        </div>
      </div>

    </div>

    <!-- Right Column: Predictions & Details -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- Prediction Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <h2 class="font-semibold text-slate-200 text-base mb-4 flex items-center gap-2">
          <i class="fa-solid fa-bullseye text-emerald-400"></i> Model Recognition
        </h2>

        <div id="idle-state" class="py-12 text-center text-slate-500">
          <i class="fa-solid fa-hand text-5xl mb-3 opacity-30"></i>
          <p class="text-sm">Upload or capture a sign gesture to begin</p>
        </div>

        <div id="loading-state" class="hidden py-12 text-center text-indigo-400">
          <i class="fa-solid fa-circle-notch fa-spin text-4xl mb-3"></i>
          <p class="text-sm font-medium">Running Deep CNN Inference...</p>
        </div>

        <div id="result-state" class="hidden space-y-5">
          <div class="flex items-center justify-between bg-slate-950 p-4 rounded-2xl border border-slate-800">
            <div>
              <p class="text-xs text-slate-400 font-medium">Recognized Sign</p>
              <h3 id="result-sign" class="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-300 tracking-wide">-</h3>
            </div>
            <div class="text-right">
              <p class="text-xs text-slate-400 font-medium">Confidence</p>
              <h4 id="result-confidence" class="text-2xl font-bold text-white mt-1">-</h4>
              <p id="result-time" class="text-[10px] text-slate-500 mt-0.5">Latency: -</p>
            </div>
          </div>

          <button onclick="appendCurrentSign()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs py-2.5 rounded-xl shadow-lg shadow-emerald-600/20 transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-plus"></i> Add Sign to Sentence Builder
          </button>

          <!-- Top 3 Probability Breakdown -->
          <div>
            <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Top Predictions</p>
            <div id="top-candidates-container" class="space-y-2">
              <!-- Dynamically populated -->
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Reference Bar -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 class="font-semibold text-slate-200 text-sm mb-2 flex items-center gap-2">
          <i class="fa-solid fa-circle-info text-indigo-400"></i> Supported Sign Classes (24)
        </h3>
        <p class="text-xs text-slate-400 mb-3">Alphabet letters classified by model:</p>
        <div class="flex flex-wrap gap-1.5">
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">A</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">B</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">C</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">D</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">E</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">F</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">G</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">H</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">I</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">K</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">L</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">M</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">N</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">O</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">P</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">Q</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">R</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">S</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">T</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">U</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">V</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">W</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">X</span>
          <span class="px-2 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300">Y</span>
        </div>
        <p class="text-[11px] text-amber-400/80 mt-3">⚠️ <em>Note: Signs 'J' and 'Z' require dynamic motion tracing and are omitted from static image recognition.</em></p>
      </div>

    </div>
  </main>

  <script>
    let activeStream = null;
    let latestRecognizedSign = "";
    let accumulatedText = "";

    function switchTab(mode) {
      const uploadTab = document.getElementById("tab-upload");
      const cameraTab = document.getElementById("tab-camera");
      const paneUpload = document.getElementById("pane-upload");
      const paneCamera = document.getElementById("pane-camera");

      if (mode === "upload") {
        uploadTab.className = "px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-medium transition";
        cameraTab.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        paneUpload.classList.remove("hidden");
        paneCamera.classList.add("hidden");
        stopWebcam();
      } else {
        cameraTab.className = "px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-medium transition";
        uploadTab.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        paneCamera.classList.remove("hidden");
        paneUpload.classList.add("hidden");
      }
    }

    async function startWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        activeStream = stream;
        const video = document.getElementById("webcam-video");
        video.srcObject = stream;
        document.getElementById("btn-capture-cam").disabled = false;
        document.getElementById("btn-start-cam").innerText = "Camera Active";
        document.getElementById("btn-start-cam").classList.replace("bg-indigo-600", "bg-slate-700");
      } catch (err) {
        alert("Webcam access denied or unavailable: " + err.message);
      }
    }

    function stopWebcam() {
      if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
        document.getElementById("btn-capture-cam").disabled = true;
        document.getElementById("btn-start-cam").innerText = "Start Camera";
        document.getElementById("btn-start-cam").classList.replace("bg-slate-700", "bg-indigo-600");
      }
    }

    function captureSnapshot() {
      const video = document.getElementById("webcam-video");
      const canvas = document.getElementById("snapshot-canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg");
      
      showPreview(dataUrl);
      predictBase64(dataUrl);
    }

    function handleFileSelect(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        showPreview(e.target.result);
      };
      reader.readAsDataURL(file);

      // Perform upload
      predictFile(file);
    }

    function showPreview(src) {
      const container = document.getElementById("preview-container");
      const img = document.getElementById("preview-image");
      img.src = src;
      container.classList.remove("hidden");
    }

    function setLoading(isLoading) {
      document.getElementById("idle-state").classList.add("hidden");
      if (isLoading) {
        document.getElementById("loading-state").classList.remove("hidden");
        document.getElementById("result-state").classList.add("hidden");
      } else {
        document.getElementById("loading-state").classList.add("hidden");
        document.getElementById("result-state").classList.remove("hidden");
      }
    }

    async function predictFile(file) {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/predict?top_k=3", { method: "POST", body: formData });
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert("Prediction failed: " + err);
        setLoading(false);
      }
    }

    async function predictBase64(dataUrl) {
      setLoading(true);
      try {
        const res = await fetch("/predict/base64", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: dataUrl, top_k: 3 })
        });
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert("Prediction failed: " + err);
        setLoading(false);
      }
    }

    function renderResult(data) {
      setLoading(false);
      latestRecognizedSign = data.predicted_sign;
      document.getElementById("result-sign").innerText = data.predicted_sign;
      document.getElementById("result-confidence").innerText = data.confidence_percentage;
      document.getElementById("result-time").innerText = "Latency: " + data.inference_time_ms + " ms";

      const topContainer = document.getElementById("top-candidates-container");
      topContainer.innerHTML = "";

      data.top_candidates.forEach((cand, idx) => {
        const pct = (cand.confidence * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs";
        row.innerHTML = `
          <div class="flex justify-between items-center mb-1 font-medium">
            <span class="text-slate-300">Sign '${cand.sign}'</span>
            <span class="text-emerald-400">${cand.percentage}</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
            <div class="bg-indigo-500 h-1.5 rounded-full" style="width: ${pct}%"></div>
          </div>
        `;
        topContainer.appendChild(row);
      });
    }

    // Sentence builder actions
    function appendCurrentSign() {
      if (!latestRecognizedSign) return;
      accumulatedText += latestRecognizedSign;
      updateSentence();
    }
    function addSpace() {
      accumulatedText += " ";
      updateSentence();
    }
    function backspaceWord() {
      accumulatedText = accumulatedText.slice(0, -1);
      updateSentence();
    }
    function clearWord() {
      accumulatedText = "";
      updateSentence();
    }
    function updateSentence() {
      const display = document.getElementById("sentence-display");
      if (accumulatedText.length === 0) {
        display.innerHTML = '<span class="text-slate-600 text-sm italic">Signs added to this sentence will appear here...</span>';
      } else {
        display.innerText = accumulatedText;
      }
    }
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.py:app", host="0.0.0.0", port=8000, reload=True)

