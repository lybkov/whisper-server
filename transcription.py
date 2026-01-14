import hashlib
import json
from pathlib import Path
import hmac
import httpx

import whisper
from flask import Flask


def transcription(
        file_path: Path,
        model: whisper.Whisper,
        device_name: str,
        app: Flask) -> None:
    try:
        result = model.transcribe(
            str(file_path),
            fp16=(device_name == "cuda"),
            verbose=False,
        )

    except Exception as e:
        app.logger.error('Error transition: %s', e)
        return

    file_path.unlink()

    segments = json.dumps({
        "text": result["text"].strip(),
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]
    })

    signature = hmac.new('key'.encode(), segments.encode(), hashlib.sha256).hexdigest()

    headers = {
        'x-signature': signature
    }
    try:
        with httpx.Client() as client:
            client.post(headers=headers, content=segments, url='http://192.168.1.44/api/v1/webhook/transcription')
    except Exception as e:
        app.logger.error('Error to send response: %s', e)
