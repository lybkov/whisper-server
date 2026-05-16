import logging
import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

import torch
from faster_whisper import WhisperModel
from flask import Flask, Response, jsonify, request

from transcription import transcription as transcription_worker

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

task_queue = queue.Queue()

app = Flask(__name__)
app.config['static'] = STATIC

# Настройка системного логгера
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)


def worker():
    try:
        if torch.cuda.is_available():
            model_name = "base"
            device_name = "cuda"
            model = WhisperModel(model_name, device=device_name, compute_type="float16")
        else:
            raise Exception("CUDA device not found")

    except Exception as e:
        gunicorn_logger.error('!!! GPU Error, falling back to CPU: %s', e)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        device_name = "cpu"

    gunicorn_logger.info('Whisper loaded on device: %s', device_name)

    while True:
        # Убрали flask_app из очереди, передаем только путь и id
        file_path, transcription_id = task_queue.get()

        try:
            gunicorn_logger.info('Start transcription work')
            # Вызываем воркер без передачи app
            transcription_worker(file_path, model, transcription_id)

        except Exception as e:
            gunicorn_logger.error('Worker error: %s', e)
        finally:
            task_queue.task_done()


# Запуск фонового потока
threading.Thread(target=worker, daemon=True).start()


@app.route("/transcription/", defaults={'transcription_id': None}, methods=['POST'])
@app.route("/transcription/<transcription_id>", methods=['POST'])
def transcription(transcription_id: Optional[str]) -> tuple[Response, int]:
    if 'upload-file' not in request.files:
        app.logger.warning('No file part')
        return jsonify({'message': 'No file part'}), 400

    audio = request.files['upload-file']
    if audio.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = f'{uuid.uuid4()}.mp3'
    file_path = STATIC / filename

    audio.save(file_path)

    # Передаем в очередь только необходимые данные (без объекта app)
    task_queue.put((file_path, transcription_id))

    return jsonify({'message': 'File received successfully'}), 202


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)