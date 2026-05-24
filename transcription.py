import hashlib
import hmac
import json
import logging
import subprocess
from pathlib import Path

import torch
from dotenv import dotenv_values
from faster_whisper import WhisperModel
import httpx

logger = logging.getLogger('gunicorn.error')

env = dotenv_values('.env')
key = env.get('TOKEN')


def transcription(file_path: Path, model: WhisperModel, transcription_id: str, reverse_url: str) -> None:
    logger.info(f'[ШАГ 1] Начало обработки файла: {file_path}')

    chunks_dir = file_path.parent / f"chunks_{transcription_id}"
    chunks_dir.mkdir(exist_ok=True)

    try:
        logger.info('[ШАГ 2] Нарезка файла на чанки по 20 минут через FFmpeg...')
        cmd = [
            'ffmpeg', '-i', str(file_path),
            '-f', 'segment', '-segment_time', '1200',
            '-c', 'copy', str(chunks_dir / 'chunk_%03d.mp3')
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        chunk_files = sorted(list(chunks_dir.glob('chunk_*.mp3')))
        logger.info(f'Файл успешно набит на куски. Всего чанков: {len(chunk_files)}')

        full_text = []
        result_segments = []
        time_shift = 0.0

        for idx, chunk_path in enumerate(chunk_files):
            logger.info(f' Обработка чанка {idx + 1}/{len(chunk_files)}: {chunk_path.name}')

            segments_generator, info = model.transcribe(str(chunk_path), beam_size=5)

            for segment in segments_generator:
                full_text.append(segment.text)
                result_segments.append({
                    'start': segment.start + time_shift,
                    'end': segment.end + time_shift,
                    'text': segment.text.strip(),
                })

            time_shift += info.duration

            chunk_path.unlink()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f'[ШАГ 4] Все чанки обработаны. Всего сегментов: {len(result_segments)}')

        payload = {
            'text': ''.join(full_text).strip(),
            'segments': result_segments,
        }
        segments_json = json.dumps(payload)

    except Exception as e:
        logger.error('[ОШИБКА] Сбой во время транскрибации: %s', e, exc_info=True)
        return
    finally:
        if chunks_dir.exists():
            for f in chunks_dir.glob('*'): f.unlink()
            chunks_dir.rmdir()
        if file_path.exists():
            file_path.unlink()

    signature = hmac.new(key.encode(), segments_json.encode(), hashlib.sha256).hexdigest()

    headers = {
        'x-signature': signature,
        'Content-Type': 'application/json',
    }

    try:
        logger.info(f'[ШАГ 6] Отправка webhook на URL: {reverse_url}')
        with httpx.Client() as client:
            response = client.post(
                headers=headers,
                content=segments_json,
                url=f'{reverse_url}/{transcription_id}',
                timeout=15.0,
            )
        logger.info(f'[ШАГ 7] Webhook успешно отправлен! Статус: {response.status_code}')
    except Exception as e:
        logger.error('[ОШИБКА] Сбой при отправке webhook: %s', e, exc_info=True)

