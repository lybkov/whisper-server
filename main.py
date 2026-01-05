from pathlib import Path

import torch
from flask import Flask, request, jsonify
import logging
import whisper
import uuid

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = whisper.load_model("small", device=device)
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__)
app.config['static'] = STATIC

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

app.logger.info(f"Whisper loaded on device: {device}")

@app.route("/transcription", methods=['POST'])
def transcription():
    if 'upload-file' not in request.files:
        app.logger.warning('No file part')
        return jsonify({'message': 'No file part'}, 400)

    audio = request.files['upload-file']

    if not audio.filename.endswith('.mp3'):
        app.logger.warning('File is not audio')
        return jsonify({'message': 'File is not audio'}, 400)

    filename = f'{str(uuid.uuid4())}.mp3'

    file_path = STATIC / filename

    audio.save(str(file_path))

    try:
        result = MODEL.transcribe(str(file_path))

        return jsonify(result)
    except Exception:
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'message': 'Transcription error'}, 400)
    finally:
        if file_path.exists():
            file_path.unlink()
