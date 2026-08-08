import argparse

from faster_whisper import WhisperModel

import config


def transcribir(ruta, model_size=config.STT_MODEL, lang=config.STT_LANG, prompt=config.STT_PROMPT):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(ruta, language=lang, initial_prompt=prompt, vad_filter=True)
    return [seg.text.strip() for seg in segments], info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transcribe un archivo de audio a texto.")
    ap.add_argument("ruta", help="Ruta al archivo de audio (wav, mp3, ...)")
    ap.add_argument("-m", "--modelo", default=config.STT_MODEL, help="Tamaño del modelo (tiny/base/small/medium)")
    args = ap.parse_args()

    segmentos, info = transcribir(args.ruta, args.modelo)
    print(f"Idioma detectado: {info.language} (p={info.language_probability:.2f})")
    for s in segmentos:
        print(s)
