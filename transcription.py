import hashlib
import hmac
import json
from pathlib import Path

import httpx
from dotenv import dotenv_values
from faster_whisper import WhisperModel
from flask import Flask

env = dotenv_values('.env')
key = env.get('TOKEN')
webhook_url = env.get('WEBHOOK_URL')


def transcription(file_path: Path, model: WhisperModel, transcription_id: str, app: Flask) -> None:
    app.logger.info(f'[ШАГ 1] Processing file: {file_path}, size: {file_path.stat().st_size} bytes')

    try:
        app.logger.info('[ШАГ 2] Вызов model.transcribe...')
        segments_generator, _info = model.transcribe(
            str(file_path),
            beam_size=5,
        )
        app.logger.info('[ШАГ 3] Генератор успешно создан. Начало итерации...')

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
            # Логируем каждые 50 сегментов, чтобы видеть прогресс и не засорять логи
            if segment_count % 50 == 0:
                app.logger.info(f'  ... обработано {segment_count} сегментов ...')

        app.logger.info(f'[ШАГ 4] Итерация завершена. Всего сегментов: {segment_count}')

        payload = {
            'text': ''.join(full_text).strip(),
            'segments': result_segments,
        }
        segments_json = json.dumps(payload)
        app.logger.info(f'[ШАГ 5] JSON сформирован. Размер: {len(segments_json)} символов.')

    except Exception as e:
        app.logger.error('[ОШИБКА] Сбой во время транскрибации: %s', e, exc_info=True)
        if file_path.exists():
            file_path.unlink()
            app.logger.info('[ОЧИСТКА] Файл удален после ошибки транскрибации.')
        return

    signature = hmac.new(key.encode(), segments_json.encode(), hashlib.sha256).hexdigest()
    base_url = webhook_url.rstrip('/')
    url = f'{base_url}/{transcription_id!s}' if transcription_id else base_url

    headers = {
        'x-signature': signature,
        'Content-Type': 'application/json',
    }

    try:
        app.logger.info(f'[ШАГ 6] Отправка webhook на URL: {url}')
        with httpx.Client() as client:
            response = client.post(
                headers=headers,
                content=segments_json,
                url=url,
                timeout=15.0,
            )
        app.logger.info(f'[ШАГ 7] Webhook успешно отправлен! Статус ответа сервера: {response.status_code}')
    except Exception as e:
        app.logger.error('[ОШИБКА] Сбой при отправке webhook: %s', e, exc_info=True)
        app.logger.error('Response url: %s', url)
    finally:
        if file_path.exists():
            file_path.unlink()
            app.logger.info(f'[ШАГ 8] Файл {file_path} успешно удален (финал).')