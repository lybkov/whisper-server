 🎙️ Whisper API Server

A lightweight REST API server built with **Flask** and **Gunicorn** that leverages **OpenAI Whisper** for high-accuracy speech-to-text transcription.

## 💡 Implementation Decisions

During development, I compared **OpenAI Whisper** and **Faster Whisper**. Although Faster Whisper demonstrated superior processing speeds, it presented significant deployment complexities in the server environment. For the sake of stability and easier maintenance, I decided to proceed with the official **OpenAI Whisper** implementation.

## 🛠 Tech Stack

* **Python 3.12**
* **OpenAI Whisper** (Speech recognition)
* **Gunicorn** (Production WSGI server)
* **FFmpeg** (Audio processing)

## 🚀 Installation & Setup

### 1. System Dependencies

Install **FFmpeg** (required for audio decoding):

```bash
apt update && apt install ffmpeg -y
```

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/lybkov/whisper-server.git
cd whisper-server
python -m venv .venv && source .venv/bin/activate && pip install openai-whisper flask gunicorn
```

### 3. Running the Server

Start the server using Gunicorn. The high timeout is essential for processing long audio files:

```bash
gunicorn --workers 1 --timeout 600 --bind 0.0.0.0:5000 main:app
```

## 📡 API Reference

### Transcribe Audio

**Endpoint:** `POST /transcription`

**Request:**
Send an audio file using `multipart/form-data`.

**cURL Example:**

```bash
curl -X POST -F "file=@/path/to/music.mp3" http://127.0.0.1:5000/transcription
```

**Success Response:**

```json
{
  "text": "Hello, this is a transcribed text from the audio file."
}
```

## ⚙️ Deployment (Systemd)

To keep the server running in the background, create a service file: `/etc/systemd/system/whisper.service`

```ini
[Unit]
Description=Gunicorn instance to serve Whisper Server
After=network.target

[Service]
User=root
WorkingDirectory=/root/whisper-server
Environment="PATH=/root/whisper-server/.venv/bin"
ExecStart=/root/whisper-server/.venv/bin/gunicorn --workers 1 --timeout 600 --bind 0.0.0.0:5000 main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

**Manage the service:**

```bash
systemctl daemon-reload
systemctl start whisper
systemctl enable whisper
```
