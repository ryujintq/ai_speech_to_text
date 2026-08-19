import os

import nvidia.cublas

# nvidia-cublas-cu12 ships cublas64_12.dll inside the package instead of on
# the system PATH. ctranslate2 loads it with a plain LoadLibrary call, which
# only consults PATH, so os.add_dll_directory (LoadLibraryEx-only) doesn't
# help here.
os.environ["PATH"] = os.path.join(nvidia.cublas.__path__[0], "bin") + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel
from config import model

stt = WhisperModel(model, device="cuda", compute_type="float16")

def convert_audio_to_text(recording):
    print('converting audio to text')

    segments, _ = stt.transcribe(recording, language="en")
    text = " ".join(segment.text for segment in segments).strip()

    print('Conversion done.')

    return text
