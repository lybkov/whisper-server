import hashlib
import hmac
import json
import logging
from pathlib import Path

import httpx
import torch  # Добавили для очистки кэша GPU
from dotenv import dotenv_values
from faster_whisper import WhisperModel

# Используем системный логгер Gunicorn, чтобы не тащить Flask-контекст
logger = logging.getLogger('gunicorn.error')

env = dotenv_values('.env')
key = env.get('TOKEN')
webhook_url = env.get('WEBHOOK_URL')


def transcription(file_path: Path, model: WhisperModel, transcription_id: str) -> None:
    logger.info(f'[ШАГ 1] Processing file: {file_path}, size: {file_path.stat().st_size} bytes')

    try:
        logger.info('[ШАГ 2] Вызов model.transcribe...')
        segments_generator, _info = model.transcribe(
            str(file_path),
            beam_size=5,
        )
        logger.info('[ШАГ 3] Генератор успешно создан. Начало итерации...')

        full_text = []
        result_segments = []
        segment_count = 0

        for segment in segments_generator:
            segment_count += 1
            full_text.append(segment.text)
            result_segments.append(
                {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                },
            )
            if segment_count % 50 == 0:
                logger.info(f'  ... обработано {segment_count} сегментов ...')

        logger.info(f'[ШАГ 4] Итерация завершена. Всего сегментов: {segment_count}')

        payload = {
            'text': ''.join(full_text).strip(),
            'segments': result_segments,
        }
        segments_json = json.dumps(payload)
        logger.info(f'[ШАГ 5] JSON сформирован. Размер: {len(segments_json)} символов.')

        # Очищаем тяжелые переменные сразу, как только сформировали JSON string
        del segments_generator
        del result_segments
        del full_text

    except Exception as e:
        logger.error('[ОШИБКА] Сбой во время транскрибации: %s', e, exc_info=True)
        if file_path.exists():
            file_path.unlink()
            logger.info('[ОЧИСТКА] Файл удален после ошибки транскрибации.')
        return
    finally:
        # Важнейший шаг: принудительно очищаем кэш аллокатора CUDA памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            response = client.post(
                headers=headers,
                content=segments_json,
                url=url,
                timeout=15.0,
            )
        logger.info(f'[ШАГ 7] Webhook успешно отправлен! Статус ответа сервера: {response.status_code}')
    except Exception as e:
        logger.error('[ОШИБКА] Сбой при отправке webhook: %s', e, exc_info=True)
        logger.error('Response url: %s', url)
    finally:
        if file_path.exists():
            file_path.unlink()
            logger.info(f'[ШАГ 8] Файл {file_path} успешно удален (финал).')