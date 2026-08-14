"""EDITH — Interfaz de usuario (TUI) del asistente de voz local.

Flujo completo: micrófono → STT (faster-whisper) → Ollama → TTS → altavoz.

Ejecutar con:
    ./.venv/bin/python main.py

La interfaz muestra un "cerebro" animado con partículas azul claro en el
centro que reacciona al estado: escuchando, pensando, hablando...
"""

import asyncio
import collections
import math
import threading
import time

import numpy as np
import httpx
import sounddevice as sd
import webrtcvad

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, RichLog

import config
import tts as tts_mod

from tqdm.std import TqdmDefaultWriteLock

TqdmDefaultWriteLock()

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)

AZUL_FONDO = "#04060e"
AZUL_BORDE = "#2a4a75"
AZUL_CAMINO = "#103055"
AZUL_PARTICULA = "#dff5ff"
AZUL_TITULO = "#bfe8ff"
AZUL_TEXTO = "#9fd4ff"


def _mezclar_hex(a: str, b: str, t: float) -> str:
    def _canal(h: str):
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)

    ra, ga, ba = _canal(a)
    rb, gb, bb = _canal(b)
    return "#{:02x}{:02x}{:02x}".format(
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )


class Cerebro(Widget):
    """Marca central "EDITH": texto centrado con brillo pulsante que
    reacciona al estado (escuchando, pensando, hablando…)."""

    BORDER_TITLE = "EDITH"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t = 0.0
        self._intensidad = 0.4

    def on_mount(self) -> None:
        self.set_interval(0.07, self._tick)

    def set_estado(self, intensidad: float) -> None:
        self._intensidad = intensidad

    def _tick(self) -> None:
        self._t += 0.07
        self.refresh()

    def render(self) -> Text:
        cw, ch = self.content_size.width, self.content_size.height
        palabra = "EDITH"
        if cw < len(palabra) or ch < 3:
            return Text(palabra, style=f"bold {AZUL_TITULO}")

        pulso = 0.5 + 0.5 * math.sin(self._t * math.tau / 4.0)
        k = max(0.0, min(1.0, self._intensidad))
        t = max(0.0, min(1.0, 0.3 + 0.7 * k * pulso))
        color = _mezclar_hex(AZUL_TITULO, AZUL_PARTICULA, t)
        halo = _mezclar_hex(AZUL_CAMINO, color, 0.55)

        separado = " ".join(palabra)
        if cw < len(separado):
            separado = palabra
        x = max(0, (cw - len(separado)) // 2)
        y = ch // 2

        letras = set()
        for dx, car in enumerate(separado):
            if car != " ":
                letras.add((x + dx, y))
        halo_celdas = set()
        for lx, ly in letras:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    halo_celdas.add((lx + dx, ly + dy))
        halo_celdas -= letras

        text = Text()
        for fy in range(ch):
            for fx in range(cw):
                if (fx, fy) in letras:
                    text.append(separado[fx - x], style=f"bold {color}")
                elif (fx, fy) in halo_celdas:
                    text.append("·", style=halo)
                else:
                    text.append(" ")
            text.append("\n")
        return text


class EdithApp(App):
    """Aplicación TUI de EDITH."""

    TITLE = "EDITH"
    SUB_TITLE = "Asistente de voz local"
    CSS = """
    Screen {
        layout: vertical;
        background: #04060e;
    }
    Header {
        background: #0a1226;
        color: #bfe8ff;
    }
    Footer {
        background: #0a1226;
    }
    #principal {
        height: 1fr;
        padding: 0 1;
    }
    #zona-superior {
        height: 1fr;
    }
    #columna-cerebro {
        width: 1fr;
        height: 1fr;
    }
    Cerebro {
        width: 1fr;
        height: 1fr;
        border: round #2a4a75;
    }
    #estado {
        height: 3;
        padding: 0 1;
        content-align: center middle;
        color: #9fd4ff;
    }
    #log {
        width: 1fr;
        height: 1fr;
        border: round #2a4a75;
        padding: 0 1;
        display: none;
    }
    #fila-entrada {
        height: 3;
        padding: 0 1 0 1;
        align: center middle;
    }
    #mic {
        min-width: 14;
    }
    #entrada {
        margin: 0 1;
    }
    #enviar {
        min-width: 10;
    }
    """

    BINDINGS = [
        Binding("ctrl+g", "toggle_mic", "Grabar"),
        Binding("ctrl+l", "toggle_chat", "Chat"),
        Binding("ctrl+c", "quit", "Salir"),
    ]

    def __init__(self):
        super().__init__()
        self._listo = False
        self._grabando = False
        self._stop: threading.Event | None = None
        self._hilo_captura: threading.Thread | None = None
        self._audio_capturado: np.ndarray | None = None
        self._vad = None
        self._stt = None
        self._tts = None
        self._tts_lock = threading.Lock()
        self._stt_lock = threading.Lock()
        self.historial: list[dict] = []
        if config.SYSTEM_PROMPT:
            self.historial.append({"role": "system", "content": config.SYSTEM_PROMPT})

    # --- montaje -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="principal"):
            with Horizontal(id="zona-superior"):
                with Vertical(id="columna-cerebro"):
                    yield Cerebro()
                    yield Label("Cargando modelos…", id="estado")
                yield RichLog(id="log", markup=True, wrap=True, highlight=True)
            with Horizontal(id="fila-entrada"):
                yield Button("💬 Chat", id="chat", variant="default")
                yield Button("🎤 Hablar", id="mic", variant="primary", disabled=True)
                yield Input(placeholder="Escribe un mensaje a EDITH…", id="entrada")
                yield Button("Enviar", id="enviar", variant="success", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._cargar_modelos(), name="carga", exit_on_error=False)

    # --- carga y estado -----------------------------------------------

    async def _cargar_modelos(self) -> None:
        def _inicializar():
            self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
            from faster_whisper import WhisperModel

            self._stt = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8")
            self._tts = tts_mod.crear_tts()

        try:
            await asyncio.to_thread(_inicializar)
        except Exception as e:
            import traceback as _tb

            _tb.print_exc()
            self._set_estado(f"Error cargando modelos: {e}")
            return
        self._listo = True
        self.query_one("#mic", Button).disabled = False
        self.query_one("#enviar", Button).disabled = False
        self.query_one("#entrada", Input).focus()
        if self._ollama_ok():
            self._set_estado("Escuchando… (🎤 hablar, Ctrl+G, o escribe)")
        else:
            self._set_estado("⚠ Ollama no responde en localhost:11434")

    def _ollama_ok(self) -> bool:
        try:
            httpx.get("http://localhost:11434/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def _set_estado(self, texto: str, intensidad: float = 0.4) -> None:
        self.query_one("#estado", Label).update(texto)
        self.query_one(Cerebro).set_estado(intensidad)

    def _log(self, quien: str, texto: str) -> None:
        log = self.query_one("#log", RichLog)
        if quien == "EDITH":
            log.write(f"[bold {AZUL_TITULO}]EDITH:[/] {texto}")
        else:
            log.write(f"[bold {AZUL_TEXTO}]Tú:[/] {texto}")

    # --- eventos de la interfaz ---------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._enviar_texto(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "enviar":
            entrada = self.query_one("#entrada", Input)
            self._enviar_texto(entrada.value)
        elif event.button.id == "mic":
            self._toggle_mic()
        elif event.button.id == "chat":
            self._toggle_chat()

    def action_toggle_mic(self) -> None:
        if self._listo:
            self._toggle_mic()

    def action_toggle_chat(self) -> None:
        self._toggle_chat()

    def _toggle_chat(self) -> None:
        log = self.query_one("#log", RichLog)
        columna = self.query_one("#columna-cerebro", Vertical)
        boton = self.query_one("#chat", Button)
        if log.display == "none":
            log.display = "block"
            columna.styles.width = "42%"
            boton.label = "🧠 Cerebro"
        else:
            log.display = "none"
            columna.styles.width = "1fr"
            boton.label = "💬 Chat"

    def _enviar_texto(self, texto: str) -> None:
        texto = texto.strip()
        if not texto or not self._listo:
            return
        self.query_one("#entrada", Input).value = ""
        self._log("Tú", texto)
        self.historial.append({"role": "user", "content": texto})
        self._set_estado("Pensando…", 1.5)
        self.run_worker(self._generar_respuesta(), group="ia", exit_on_error=False)

    # --- captura por micrófono (push-to-talk) -------------------------

    def _toggle_mic(self) -> None:
        boton = self.query_one("#mic", Button)
        if not self._grabando:
            self._grabando = True
            self._stop = threading.Event()
            self._audio_capturado = None
            self._hilo_captura = threading.Thread(target=self._capturar, daemon=True)
            self._hilo_captura.start()
            boton.label = "⏹ Parar"
            self._set_estado("Escuchándote… (pulsa 🎤 o Ctrl+G al terminar)", 0.9)
        else:
            self._grabando = False
            self._stop.set()
            boton.label = "🎤 Hablar"
            self._set_estado("Procesando…", 1.0)
            self.run_worker(self._terminar_grabacion(), group="ia", exit_on_error=False)

    def _capturar(self) -> None:
        frames = collections.deque()
        pre = collections.deque(maxlen=10)
        hablando = False
        inicio = None
        try:
            with sd.RawInputStream(
                samplerate=config.SAMPLE_RATE,
                blocksize=FRAME_SAMPLES,
                channels=1,
                dtype="int16",
            ) as stream:
                while not self._stop.is_set():
                    chunk, _ = stream.read(FRAME_SAMPLES)
                    voz = self._vad.is_speech(chunk, config.SAMPLE_RATE)
                    pre.append(chunk)
                    if not hablando:
                        if voz:
                            hablando = True
                            inicio = time.time()
                            frames.extend(pre)
                    else:
                        frames.append(chunk)
                        if time.time() - inicio >= config.MAX_RECORD_SECONDS:
                            break
        except Exception:
            pass
        if frames:
            audio = np.concatenate(frames) if len(frames) > 1 else frames[0]
            self._audio_capturado = audio

    async def _terminar_grabacion(self) -> None:
        await asyncio.to_thread(self._hilo_captura.join, 10)
        audio = self._audio_capturado
        self._audio_capturado = None
        if audio is None:
            self._set_estado("No he capturado audio, inténtalo de nuevo.")
            return
        await self._transcribir(audio)

    # --- cadena STT -> LLM -> TTS -------------------------------------

    async def _transcribir(self, audio: np.ndarray) -> None:
        self._set_estado("Transcribiendo…", 1.0)
        try:
            texto = await asyncio.to_thread(self._stt_transcribir, audio)
        except Exception as e:
            self._set_estado(f"Error transcribiendo: {e}")
            return
        if not texto:
            self._set_estado("No te he entendido, prueba otra vez.")
            return
        self._log("Tú", texto)
        self.historial.append({"role": "user", "content": texto})
        await self._generar_respuesta()

    def _stt_transcribir(self, audio: np.ndarray) -> str:
        with self._stt_lock:
            muestras = np.frombuffer(audio.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._stt.transcribe(
                muestras,
                language=config.STT_LANG,
                initial_prompt=config.STT_PROMPT,
                vad_filter=True,
            )
            return " ".join(s.text.strip() for s in segments).strip()

    async def _generar_respuesta(self) -> None:
        self._set_estado("Pensando…", 1.5)
        try:
            respuesta = await asyncio.to_thread(self._llm)
        except Exception as e:
            self._set_estado(f"Error con Ollama: {e}")
            return
        self.historial.append({"role": "assistant", "content": respuesta})
        del self.historial[1:-12]
        self._log("EDITH", respuesta)
        self._set_estado("Hablando…", 1.2)
        try:
            await asyncio.to_thread(self._hablar, respuesta)
        except Exception as e:
            self._set_estado(f"Error de voz: {e}")
            return
        self._set_estado("Escuchando… (🎤 hablar, Ctrl+G, o escribe)")

    def _llm(self) -> str:
        r = httpx.post(
            config.OLLAMA_URL,
            json={"model": config.OLLAMA_MODEL, "messages": self.historial, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _hablar(self, texto: str) -> None:
        with self._tts_lock:
            tts_mod.hablar(self._tts, texto)


if __name__ == "__main__":
    EdithApp().run()
