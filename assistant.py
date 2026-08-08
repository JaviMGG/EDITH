import collections
import threading
import time

import numpy as np

import espeakng_loader
espeakng_loader.load_library()

import sherpa_onnx
import sounddevice as sd
import webrtcvad
import httpx
from faster_whisper import WhisperModel

import config
import tts as tts_mod

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)


def capturar_frase(vad, max_seg=config.MAX_RECORD_SECONDS):
    frames = collections.deque()
    pre = collections.deque(maxlen=10)
    hablando = False
    inicio = None
    fin = threading.Event()

    def esperar_tecla():
        linea = input()
        fin.set()
        if linea.strip().lower() in ("salir", "exit", "quit"):
            fin.parar = True

    hilo = threading.Thread(target=esperar_tecla, daemon=True)
    hilo.start()

    with sd.RawInputStream(samplerate=config.SAMPLE_RATE, blocksize=FRAME_SAMPLES, channels=1, dtype="int16") as stream:
        while not fin.is_set():
            chunk, _ = stream.read(FRAME_SAMPLES)
            voz = vad.is_speech(chunk, config.SAMPLE_RATE)
            pre.append(chunk)
            if not hablando:
                if voz:
                    hablando = True
                    inicio = time.time()
                    frames.extend(pre)
            else:
                frames.append(chunk)
                if time.time() - inicio >= max_seg:
                    break
    return (np.concatenate(frames) if len(frames) > 1 else frames[0]) if frames else None, getattr(fin, "parar", False)


def transcribir_audio(model, audio):
    muestras = np.frombuffer(audio.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        muestras,
        language=config.STT_LANG,
        initial_prompt=config.STT_PROMPT,
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def preguntar(historial):
    r = httpx.post(
        config.OLLAMA_URL,
        json={"model": config.OLLAMA_MODEL, "messages": historial, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    print(f"EDITH lista. Modelo Ollama: {config.OLLAMA_MODEL}. Escribe 'salir' + Enter o Ctrl+C para salir.")
    model = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8")
    tts = tts_mod.crear_tts()
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    historial = []
    if config.SYSTEM_PROMPT:
        historial.append({"role": "system", "content": config.SYSTEM_PROMPT})

    while True:
        print("  Habla; pulsa Enter cuando termines (o escribe 'salir' y Enter).")
        audio, salir = capturar_frase(vad)
        if salir:
            break
        if audio is None:
            continue
        texto = transcribir_audio(model, audio)
        if not texto:
            print("  (no te he entendido)")
            continue
        print(f"  Tu: {texto}")
        historial.append({"role": "user", "content": texto})
        try:
            respuesta = preguntar(historial)
        except httpx.HTTPError as e:
            print(f"  Error con Ollama: {e}")
            continue
        historial.append({"role": "assistant", "content": respuesta})
        del historial[1:-12]
        print(f"  EDITH: {respuesta}")
        tts_mod.hablar(tts, respuesta)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAdiós.")
