import argparse

import numpy as np

import espeakng_loader
espeakng_loader.load_library()

import soundfile as sf
import sounddevice as sd
import sherpa_onnx

import config


class MotorPiper:
    def __init__(self):
        from piper import PiperVoice, SynthesisConfig
        self.voice = PiperVoice.load(config.PIPER_MODEL, config.PIPER_CONFIG)
        self.synth = SynthesisConfig(speaker_id=config.PIPER_SPEAKER_ID)

    def generar(self, texto):
        chunks = list(self.voice.synthesize(texto, self.synth))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        return audio, chunks[0].sample_rate


class MotorKokoro:
    def __init__(self):
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(config.TTS_MODEL),
                    voices=str(config.TTS_VOICES),
                    tokens=str(config.TTS_TOKENS),
                    data_dir=str(config.TTS_DATA_DIR),
                    lang=config.TTS_LANG,
                ),
                num_threads=config.TTS_THREADS,
            ),
            max_num_sentences=config.TTS_MAX_SENTENCES,
        )
        self.tts = sherpa_onnx.OfflineTts(cfg)

    def generar(self, texto):
        audio = self.tts.generate(texto, sid=config.TTS_SID, speed=config.TTS_SPEED)
        return audio.samples, audio.sample_rate


def crear_tts():
    if config.TTS_ENGINE == "piper":
        return MotorPiper()
    return MotorKokoro()


def hablar(tts, texto, out=None, play=True):
    audio, sr = tts.generar(texto)
    if out:
        sf.write(out, audio, sr)
    if play:
        sd.play(audio, sr)
        sd.wait()
    return audio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convierte texto a voz y lo reproduce.")
    ap.add_argument("texto", help="Texto a decir")
    ap.add_argument("-o", "--out", default=None, help="Guardar el audio en un archivo wav")
    ap.add_argument("-s", "--sid", type=int, default=config.TTS_SID, help="Voz de Kokoro (0=af_maple, 1=af_sol, 2=bf_vale)")
    ap.add_argument("--engine", choices=["piper", "kokoro"], default=config.TTS_ENGINE, help="Motor TTS")
    ap.add_argument("--no-play", action="store_true", help="No reproducir, solo generar")
    args = ap.parse_args()

    config.TTS_ENGINE = args.engine
    config.TTS_SID = args.sid
    tts = crear_tts()
    hablar(tts, args.texto, args.out, play=not args.no_play)
    print("OK")
