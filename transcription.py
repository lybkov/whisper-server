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
webhook_url = env.get('WEBHOOK_URL')


def transcription(file_path: Path, model: WhisperModel, transcription_id: str) -> None:
    logger.info(f'[ШАГ 1] Начало обработки файла: {file_path}')

    # Создаем временную папку для кусочков аудио рядом с файлом
    chunks_dir = file_path.parent / f"chunks_{transcription_id}"
    chunks_dir.mkdir(exist_ok=True)

    try:
        logger.info('[ШАГ 2] Нарезка файла на чанки по 20 минут через FFmpeg...')
        # Нарезаем файл средствами FFmpeg без потери качества (-c copy) прямо на HDD
        # Куски будут называться chunk_000.mp3, chunk_001.mp3 и т.д.
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
        time_shift = 0.0  # Сдвиг по времени для склейки таймкодов

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

            # Увеличиваем временной сдвиг на длительность текущего обработанного чанка
            time_shift += info.duration

            # Сразу удаляем отработанный чанк с HDD, чтобы освободить место
            chunk_path.unlink()

            # Чистим кэш после каждого куска
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
        # Очистка временных файлов при любом исходе
        if chunks_dir.exists():
            for f in chunks_dir.glob('*'): f.unlink()
            chunks_dir.rmdir()
        if file_path.exists():
            file_path.unlink()

    # --- Код отправки Webhook (Остается прежним) ---
    signature = hmac.new(key.encode(), segments_json.encode(), hashlib.sha256).hexdigest()
    base_url = webhook_url.rstrip('/')
    url = f'{base_url}/{transcription_id!s}' if transcription_id else base_url

    headers = {
        'x-signature': signature,
        'Content-Type': 'application/json',
    }

    try:
        logger.info(f'[ШАГ 6] Отправка webhook на URL: {url}')
        with httpx.Client() as client:
            response = client.post(headers=headers, content=segments_json, url=url, timeout=15.0)
        logger.info(f'[ШАГ 7] Webhook успешно отправлен! Статус: {response.status_code}')
    except Exception as e:
        logger.error('[ОШИБКА] Сбой при отправке webhook: %s', e, exc_info=True)