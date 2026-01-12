import os
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

OUTPUT_MP3 = "source.mp3"  # saved only if checkbox is true

def run_ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-y", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    save_mp3: str = Form("false"),  # "true" / "false"
):
    want_mp3 = str(save_mp3).lower() in ("1", "true", "yes", "on")

    with tempfile.TemporaryDirectory() as td:
        raw_path = os.path.join(td, "input_audio")
        wav_path = os.path.join(td, "audio.wav")

        data = await audio.read()
        with open(raw_path, "wb") as f:
            f.write(data)

        # ✅ Optional: save MP3 if requested
        saved_audio = None
        if want_mp3:
            # Convert uploaded audio to an MP3 file (overwrites source.mp3)
            run_ffmpeg(["-i", raw_path, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", OUTPUT_MP3])
            saved_audio = OUTPUT_MP3

        # Convert to WAV 16kHz mono for Whisper
        run_ffmpeg(["-i", raw_path, "-ac", "1", "-ar", "16000", wav_path])

        segments, info = model.transcribe(wav_path, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return {"language": info.language, "text": text, "saved_audio": saved_audio}