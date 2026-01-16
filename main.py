import logging
import queue
import threading
import uuid
from pathlib import Path

import torch
import whisper
from flask import Flask, Response, jsonify, request

from transcription import transcription as transcription_worker

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

task_queue = queue.Queue()

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


def worker():
    while True:
        file_path, model, device, flask_app = task_queue.get()

        try:
            app.logger.info('Start transcription work')

            transcription_worker(file_path, model, device, flask_app)

        except Exception as e:
            app.logger.error('Worker error: %s', e)
        finally:
            task_queue.task_done()

threading.Thread(target=worker, daemon=True).start()


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

    audio.save(file_path)

    task_queue.put((file_path, MODEL, device_name, app))

    return jsonify({'message': 'File received successfully'}), 202


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
