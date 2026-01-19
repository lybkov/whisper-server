import hashlib
import hmac
import json
from pathlib import Path

import httpx
import whisper
from dotenv import dotenv_values
from flask import Flask

env = dotenv_values('.env')
key = env.get('TOKEN')
webhook_url = env.get('WEBHOOK_URL')

def transcription(
        file_path: Path,
        model: whisper.Whisper,
        device_name: str,
        transcription_id: str,
        app: Flask) -> None:
    try:
        result = model.transcribe(
            str(file_path),
            fp16=(device_name == "cuda"),
            verbose=False,
        )

    except Exception as e:
        app.logger.error('Error transcription: %s', e)
        return

    segments = json.dumps({
        "text": result["text"].strip(),
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
            for segment in result["segments"]
        ],
    })

    signature = hmac.new(key.encode(), segments.encode(), hashlib.sha256).hexdigest()

    base_url = webhook_url.rstrip('/')

    url = f'{base_url}/{transcription_id!r}' if transcription_id else base_url

    headers = {
        'x-signature': signature,
        'Content-Type': 'application/json',
    }
    try:
        with httpx.Client() as client:
            client.post(
                headers=headers,
                content=segments,
                url=url,
            )
    except Exception as e:
        app.logger.error('Error to send response: %s', e)
        app.logger.error('Response url: %s', url)
    finally:
        file_path.unlink()

