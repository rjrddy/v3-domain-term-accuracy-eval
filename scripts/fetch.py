"""Fetch public Whisper test audio and two open ASR models."""
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1];cache=root/'.cache';cache.mkdir(exist_ok=True)
if not (cache/'jfk.flac').exists():urllib.request.urlretrieve('https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac',cache/'jfk.flac')
for name in ['tiny','base']:snapshot_download('Systran/faster-whisper-'+name,local_dir=cache/('whisper-'+name))
