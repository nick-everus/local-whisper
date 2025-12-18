import os
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

# If you serve the HTML from the same server, CORS isn't needed.
# But leaving it here makes it easy to serve the page separately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pick a small model first; upgrade later.
# Options: "tiny", "base", "small", "medium", "large-v3"
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

def webm_to_wav(webm_path: str, wav_path: str) -> None:
    # Convert WebM/Opus -> WAV 16kHz mono for Whisper
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Save upload to temp file
    with tempfile.TemporaryDirectory() as td:
        webm_path = os.path.join(td, "audio.webm")
        wav_path = os.path.join(td, "audio.wav")

        data = await audio.read()
        with open(webm_path, "wb") as f:
            f.write(data)

        webm_to_wav(webm_path, wav_path)

        segments, info = model.transcribe(wav_path, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return {"language": info.language, "text": text}

