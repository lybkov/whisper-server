from pathlib import Path

from flask import Flask, request, jsonify
from logger import logger
import whisper
import uuid

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3'}
MODEL = whisper.load_model("base")

app = Flask(__name__)
app.config['static'] = STATIC

@app.route("/transcription", methods=['POST'])
def transcription():
    if 'upload-file' not in request.files:
        logger.warning('No file part')
        return jsonify({'message': 'No file part'}, 400)

    audio = request.files['upload-file']

    if not audio.filename.endswith('.mp3'):
        logger.warning('File is not audio')
        return jsonify({'message': 'File is not audio'}, 400)

    filename = f'{str(uuid.uuid4())}.mp3'

    file_path = STATIC / filename

    audio.save(str(file_path))

    try:
        result = MODEL.transcribe(str(file_path))

        return jsonify(result)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'message': 'Transcription error'}, 400)
    finally:
        if file_path.exists():
            file_path.unlink()
