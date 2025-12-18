# 🎙️ Local Whisper STT (Mic → Text)

A **minimal local Speech-to-Text web app** using **OpenAI Whisper (via faster-whisper)**.  
Runs entirely on your machine — no cloud APIs, no UI frameworks.

- 🎧 Browser microphone → audio upload
- 🧠 Local Whisper model inference
- ⚡ FastAPI backend
- 🌐 Single-page HTML frontend
- 🖥️ macOS / Linux friendly

---

## ✨ Features

- One-click microphone permission
- Start / stop recording from the browser
- Local transcription using Whisper
- Supports multiple Whisper model sizes
- No React, no build step, no external services

---

## 🧰 Tech Stack

- **Python 3.10+**
- **FastAPI** – backend API
- **faster-whisper** – optimized Whisper inference
- **FFmpeg** – audio conversion
- **Vanilla HTML + JS** – frontend