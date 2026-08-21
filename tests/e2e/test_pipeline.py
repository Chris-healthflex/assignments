import asyncio
import sys
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.parse_service import process_audio_file

WAV_URL = "https://jmdkwtfuagwjloemfthj.supabase.co/storage/v1/object/public/assignment-assets/clinical_assessment.wav"
DATA_DIR = Path(__file__).parent.parent.parent / "assets"
DATA_DIR.mkdir(exist_ok=True)
WAV_PATH = DATA_DIR / "clinical_assessment.wav"

def ensure_audio():
    if not WAV_PATH.exists():
        print(f"Downloading {WAV_URL}")
        urllib.request.urlretrieve(WAV_URL, WAV_PATH)
    return WAV_PATH

async def main():
    wav_path = ensure_audio()
    with open(wav_path, "rb") as f:
        data = f.read()
    file = UploadFile(filename=wav_path.name, file=BytesIO(data))
    result = await process_audio_file(file)
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())