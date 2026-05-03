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
    app.logger.info(f'Processing file: {file_path}, size: {file_path.stat().st_size} bytes')

    try:
        segments_generator, _info = model.transcribe(
            str(file_path),
            beam_size=5,
        )

        full_text = []
        result_segments = []

        for segment in segments_generator:
            full_text.append(segment.text)
            result_segments.append(
                {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                },
            )

        payload = {
            'text': ''.join(full_text).strip(),
            'segments': result_segments,
        }
        segments_json = json.dumps(payload)

    except Exception as e:
        app.logger.error('Error transcription: %s', e)
        if file_path.exists():
            file_path.unlink()
        return

    signature = hmac.new(key.encode(), segments_json.encode(), hashlib.sha256).hexdigest()
    base_url = webhook_url.rstrip('/')
    url = f'{base_url}/{transcription_id!s}' if transcription_id else base_url

    headers = {
        'x-signature': signature,
        'Content-Type': 'application/json',
    }

    try:
        with httpx.Client() as client:
            client.post(
                headers=headers,
                content=segments_json,
                url=url,
                timeout=15.0,
            )
    except Exception as e:
        app.logger.error('Error to send response: %s', e)
        app.logger.error('Response url: %s', url)
    finally:
        if file_path.exists():
            file_path.unlink()
