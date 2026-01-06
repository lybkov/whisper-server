Python 3.12

gunicorn --workers 1 --timeout 600 --bind 0.0.0.0:5000 main:app
apt update && apt install ffmpeg

curl -X POST -F "file=@/music.mp3" http://127.0.0.1:5000/transcription