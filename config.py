from pathlib import Path

import espeakng_loader

BASE = Path(__file__).resolve().parent

STT_MODEL = "medium"
STT_LANG = "es"
STT_PROMPT = ""

TTS_ENGINE = "piper"

TTS_MODEL = BASE / "models" / "kokoro-multilang.onnx"
TTS_VOICES = BASE / "models" / "kokoro-voices.bin"
TTS_TOKENS = BASE / "models" / "kokoro-tokens.txt"
TTS_DATA_DIR = Path(espeakng_loader.get_data_path())
TTS_LANG = "es"
TTS_SID = 1
TTS_SPEED = 1.0
TTS_THREADS = 2
TTS_MAX_SENTENCES = 2

PIPER_MODEL = BASE / "models" / "piper" / "es_ES-sharvard-medium.onnx"
PIPER_CONFIG = BASE / "models" / "piper" / "es_ES-sharvard-medium.onnx.json"
PIPER_SPEAKER_ID = 1

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "edith"
SYSTEM_PROMPT = ""

SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 1
MAX_RECORD_SECONDS = 20
