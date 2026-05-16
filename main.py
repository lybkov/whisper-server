import logging
import multiprocessing
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request
from transcription import transcription as transcription_worker

STATIC = Path(__file__).resolve().parent / 'static'
STATIC.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['static'] = STATIC

# Настройка системного логгера Gunicorn
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)


def run_in_separate_process(file_path: Path, transcription_id: Optional[str]):
    """
    Эта функция запускается в отдельном процессе.
    Модель инициализируется внутри процесса, выполняет работу и умирает,
    гарантированно очищая 100% RAM и VRAM.
    """
    import torch
    from faster_whisper import WhisperModel

    try:
        if torch.cuda.is_available():
            model_name = "base"
            device_name = "cuda"
            # Передаем напрямую параметры, чтобы не держать лишний кэш
            model = WhisperModel(model_name, device=device_name, compute_type="float16")
        else:
            raise Exception("CUDA device not found")
    except Exception as e:
        gunicorn_logger.error('!!! GPU Error, falling back to CPU: %s', e)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        device_name = "cpu"

    gunicorn_logger.info('Whisper loaded inside child process on device: %s', device_name)

    try:
        gunicorn_logger.info('Start transcription work in isolated process')
        transcription_worker(file_path, model, transcription_id)
    except Exception as e:
        gunicorn_logger.error('Worker process error: %s', e)


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

    # ВМЕСТО очереди запускаем изолированный процесс
    p = multiprocessing.Process(
        target=run_in_separate_process,
        args=(file_path, transcription_id)
    )
    p.start()

    return jsonify({'message': 'File received successfully, processing started'}), 202


if __name__ == '__main__':
    # Это важно для стабильности работы multiprocessing в Flask
    multiprocessing.set_start_method('spawn', force=True)
    app.run(host='127.0.0.1', port=5000)