import whisper
import torch
import logging
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, Response
import threading
from transcription import transcription as transcription_worker

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['static'] = STATIC

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

try:
    if torch.cuda.is_available():
        model_name = "base"
        device_name = "cuda"
        MODEL = whisper.load_model(model_name, device=device_name)
    else:
        raise Exception("CUDA device not found")

except Exception as e:
    app.logger.error('!!! GPU Error, falling back to CPU: %s', e)
    MODEL = whisper.load_model("base", device="cpu")
    device_name = "cpu"

app.logger.info('Whisper loaded on device: %s', device_name)


@app.route("/transcription", methods=['POST'])
def transcription() -> tuple[Response, int]:
    if 'upload-file' not in request.files:
        app.logger.warning('No file part')
        return jsonify({'message': 'No file part'}), 400

    audio = request.files['upload-file']
    if audio.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = f'{uuid.uuid4()}.mp3'
    file_path = STATIC / filename

    threading.Thread(
        target=transcription_worker,
        args=(file_path,MODEL, device_name, app),
        daemon=True).start()

    return jsonify({'message': 'File received successfully'}), 202


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
