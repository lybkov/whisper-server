from pathlib import Path
from flask import Flask, request, jsonify, Response
import logging
from faster_whisper import WhisperModel
import uuid

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

try:
    MODEL = WhisperModel("turbo", device="cuda", compute_type="float16")
    device_name = "cuda"
except Exception:
    MODEL = WhisperModel("turbo", device="cpu", compute_type="int8")
    device_name = "cpu"

app = Flask(__name__)
app.config['static'] = STATIC

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

app.logger.info(f"Whisper loaded on device: {device_name}")


@app.route("/transcription", methods=['POST'])
def transcription() -> tuple[Response, int] | Response:
    if 'upload-file' not in request.files:
        app.logger.warning('No file part')
        return jsonify({'message': 'No file part'}), 400

    audio = request.files['upload-file']
    if not audio.filename.endswith('.mp3'):
        app.logger.warning('File is not audio')
        return jsonify({'message': 'File is not audio'}), 400

    filename = f'{uuid.uuid4()}.mp3'
    file_path = STATIC / filename

    try:
        audio.save(str(file_path))

        segments, info = MODEL.transcribe(str(file_path), beam_size=5)

        full_text = "".join([segment.text for segment in segments]).strip()

        return jsonify({
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        })
    except Exception:
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'message': 'Transcription error'}), 400
    finally:
        if file_path.exists():
            file_path.unlink()
