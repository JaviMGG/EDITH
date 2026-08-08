# EDITH — Asistente de voz local

Asistente de voz 100% local: escucha por micrófono, transcribe, piensa con un LLM (Ollama) y responde por altavoz con voz en español. Todo corre en tu máquina, sin servicios en la nube.

## Instalación rápida

Necesitas `python3`, `git`, `curl` y (para el LLM) [Ollama](https://ollama.com) instalado y en marcha.

```bash
git clone <URL_de_este_repositorio> edith
cd edith
./install.sh                 # venv + dependencias + voz Piper
# opcional:
./install.sh --with-ollama   # además comprueba el modelo LLM local 'edith'
./install.sh --full          # además descarga el motor TTS alternativo (Kokoro, ~380 MB)
```

Y a usarlo:

```bash
./.venv/bin/python assistant.py
```

El instalador lo deja todo listo: entorno aislado, dependencias y modelos. El modelo de transcripción (Whisper) se descarga solo la primera vez que ejecutas algo.

## Índice

- [Cómo funciona (flujo)](#cómo-funciona-flujo)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Librerías y su papel](#librerías-y-su-papel)
- [Dependencias del sistema](#dependencias-del-sistema)
- [Instalación desde cero](#instalación-desde-cero)
  - [1. Dependencias del sistema](#1-dependencias-del-sistema)
  - [2. Entorno Python (venv)](#2-entorno-python-venv)
  - [3. Descargar los modelos](#3-descargar-los-modelos)
  - [4. Modelo LLM en Ollama](#4-modelo-llm-en-ollama)
- [Configuración (config.py)](#configuración-configpy)
- [Uso](#uso)
  - [Asistente de voz](#asistente-de-voz)
  - [Texto a voz](#texto-a-voz)
  - [Transcripción de audio](#transcripción-de-audio)
- [Motores de voz](#motores-de-voz)
- [Solución de problemas](#solución-de-problemas)
- [Notas técnicas](#notas-técnicas)

---

## Cómo funciona (flujo)

```
micrófono → VAD → STT → Ollama → TTS → altavoz
 (audio)   (inicio)  (texto)  (LLM)  (voz)
```

1. **Captura de audio**: `sounddevice` graba desde el micrófono en mono a 16 kHz.
2. **VAD (Voice Activity Detection)**: `webrtcvad` detecta cuándo empiezas a hablar y conserva un poco de audio previo. La grabación continúa hasta que **pulsas Enter** (o hasta `MAX_RECORD_SECONDS`).
3. **STT**: `faster-whisper` (Whisper) transcribe el audio capturado a texto en español.
4. **LLM**: el texto se envía por HTTP (`httpx`) a la API local de **Ollama** (`http://localhost:11434/api/chat`) con el historial de la conversación. El modelo responde en texto.
5. **TTS**: la respuesta se convierte a voz con **Piper** (por defecto, voz femenina) y se reproduce por el altavoz con `sounddevice`.
6. El historial se mantiene (se recorta para no pasarse de contexto) y el bucle vuelve al punto 1.

---

## Estructura del proyecto

```
~/Documentos/EDITH/
├── assistant.py        # Loop completo: mic → STT → Ollama → TTS → altavoz
├── transcribe.py       # Transcribe archivos de audio a texto (por lotes)
├── tts.py              # Texto → voz (con 2 motores: piper y kokoro)
├── config.py           # Toda la configuración centralizada
├── requirements.txt    # Dependencias Python
├── README.md           # Este documento
├── .venv/              # Entorno Python aislado (no se toca a mano)
└── models/
    ├── kokoro-multilang.onnx      # Modelo TTS Kokoro (multilingüe, 311 MB)
    ├── kokoro-voices.bin          # Voces de Kokoro (52 MB)
    ├── kokoro-tokens.txt          # Vocabulario de Kokoro
    └── piper/
        ├── es_ES-sharvard-medium.onnx      # Voz femenina española activa (74 MB)
        ├── es_ES-sharvard-medium.onnx.json # Configuración (2 hablantes: M=0, F=1)
        ├── es_ES-davefx-medium.onnx        # Otra voz española (opcional)
        └── es_ES-davefx-medium.onnx.json
```

---

## Librerías y su papel

| Librería | Versión | Papel |
|---|---|---|
| `faster-whisper` | 1.2.1 | STT (habla → texto). Usa CTranslate2 (motor C++) por debajo |
| `ctranslate2` | 4.8.1 | Motor de inferencia de faster-whisper |
| `webrtcvad` | 2.0.10 | Detección de actividad de voz (detectar el inicio del habla para grabar) |
| `piper-tts` | 1.6.0 | TTS (texto → voz) con la voz española |
| `sherpa-onnx` | 1.13.4 | Motor TTS alternativo (Kokoro multilingüe) |
| `kokoro-onnx` | 0.4.7 | (dependencia, aporta `espeakng-loader`) |
| `espeakng-loader` | 0.2.4 | Carga `libespeak-ng.so` y los datos de fonemización (necesario para sherpa-onnx y piper) |
| `sounddevice` | 0.5.5 | Entrada/salida de audio (micrófono y altavoz) vía PortAudio |
| `soundfile` | 0.14.0 | Leer/escribir archivos de audio (wav, etc.) |
| `httpx` | 0.28.1 | Llamadas HTTP a la API local de Ollama |
| `onnxruntime` | 1.28.0 | Motor de inferencia ONNX (piper y kokoro) |
| `numpy` | 2.5.1 | Procesado de las muestras de audio |

Nota: no se usa `torch` a propósito (pesa ~2 GB y arrastra librerías CUDA de NVIDIA que no se necesitan). Todos los motores funcionan con CPU/ONNX/CTranslate2.

---

## Dependencias del sistema

- **Python 3.14** (u otro 3.10+). El proyecto se probó con 3.14.6.
- **PortAudio** — lo necesita `sounddevice`. En Arch/CachyOS:
  ```bash
  sudo pacman -S portaudio
  ```
- **Ollama** — el servidor local de LLM. Instalado y corriendo en `localhost:11434`.
- No se necesita `espeak-ng` del sistema: el venv trae `libespeak-ng` y `espeak-ng-data` dentro de `espeakng-loader` y del wheel de `piper-tts`.
- No se necesita GPU NVIDIA. Todo corre en CPU (Intel iGPU suficiente).

---

## Instalación desde cero

Estos pasos asumen una máquina nueva y reconstruyen el proyecto completo. **Si quieres lo fácil, ejecuta `./install.sh`** (ver [Instalación rápida](#instalación-rápida)): hace exactamente los pasos 1 a 4 por ti.

### 1. Dependencias del sistema

```bash
sudo pacman -S portaudio
```

(Ollama instalado aparte, ver paso 4.)

### 2. Entorno Python (venv)

```bash
cd ~/Documentos/EDITH
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` contiene las dependencias fijadas en las versiones probadas (más `setuptools<81`, necesario para `webrtcvad`):

```
faster-whisper==1.2.1
sounddevice==0.5.5
soundfile==0.14.0
webrtcvad==2.0.10
piper-tts==1.6.0
sherpa-onnx==1.13.4
kokoro-onnx==0.4.7
espeakng-loader==0.2.4
httpx==0.28.1
setuptools<81
```

**Importante**: `requirements.txt` ya fija `setuptools<81` (necesario para que `webrtcvad` funcione). Si lo instalaste a mano y Python da `ModuleNotFoundError: No module named 'pkg_resources'`:

```bash
.venv/bin/pip install 'setuptools<81'
```

### 3. Descargar los modelos

Todos dentro de `~/Documentos/EDITH/models/`.

**Voz Piper femenina (español, la que usa el asistente por defecto):**

```bash
mkdir -p ~/Documentos/EDITH/models/piper
cd ~/Documentos/EDITH/models/piper
curl -L -o es_ES-sharvard-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx"
curl -L -o es_ES-sharvard-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json"
```

Voz alternativa (`es_ES-davefx-medium`), por si quieres comparar:

```bash
cd ~/Documentos/EDITH/models/piper
curl -L -o es_ES-davefx-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
curl -L -o es_ES-davefx-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
```

**Modelo Kokoro multilingüe** (TTS alternativo, calidad muy alta; incluye soporte multilingüe vía espeak-ng). Se usa el export del equipo de sherpa-onnx, porque el modelo oficial de hexgrad requiere cuenta de HuggingFace (modelo "gated"):

```bash
cd ~/Documentos/EDITH/models
curl -L -o kokoro-multilang.onnx \
  "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/model.onnx"   # 325 MB
curl -L -o kokoro-voices.bin \
  "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/voices.bin"  # 51 MB
curl -L -o kokoro-tokens.txt \
  "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/tokens.txt"
```

> Si una descarga se corta, reanuda con `curl -L -C - -o ...`. Verifica los tamaños: `kokoro-multilang.onnx` debe pesar exactamente **325 631 784 bytes**; un archivo menor está incompleto.

**Modelo STT (Whisper)** — no hay que descargarlo a mano: `faster-whisper` lo baja automáticamente la primera vez que ejecutas `transcribe.py` o `assistant.py`. Usa `medium` (~1,5 GB) según `config.py`.

### 4. Modelo LLM en Ollama

Crea el modelo local `edith` a partir de su `Modelfile` (base `qwen2.5:3b`):

```bash
ollama create edith -f Modelfile
```

Verifica que el servidor responde y que el modelo está cargado:

```bash
curl http://localhost:11434/api/tags
ollama list | grep edith
```

---

## Configuración (config.py)

Todo está centralizado en `config.py`:

| Variable | Valor por defecto | Significado |
|---|---|---|
| `STT_MODEL` | `"medium"` | Tamaño del modelo Whisper (`tiny`/`base`/`small`/`medium`). Mayor = mejor calidad, más lento y RAM |
| `STT_LANG` | `"es"` | Idioma para transcribir |
| `STT_PROMPT` | `""` | "Ancla" opcional de Whisper (vacío = sin sesgo) |
| `TTS_ENGINE` | `"piper"` | Motor de voz: `"piper"` o `"kokoro"` |
| `PIPER_MODEL` / `PIPER_CONFIG` | rutas a `models/piper/` | Voz española Piper (sharvard-medium) |
| `PIPER_SPEAKER_ID` | `1` | Hablante de la voz Piper (1 = femenino en sharvard) |
| `TTS_SID` | `1` | Voz de Kokoro (solo si usas `--engine kokoro`) |
| `TTS_SPEED` | `1.0` | Velocidad de habla de Kokoro |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | API local de Ollama |
| `OLLAMA_MODEL` | `"edith"` | Modelo LLM local (creado con `ollama create`) |
| `SYSTEM_PROMPT` | `""` | Sistema extra opcional (vacío = se usa la personalidad embebida del modelo `edith`) |
| `SAMPLE_RATE` | `16000` | Tasa del micrófono (Whisper) |
| `VAD_AGGRESSIVENESS` | `1` | Sensibilidad del VAD para detectar el inicio del habla (0 = menos sensible, 3 = más; más alto corta arranques suaves) |
| `MAX_RECORD_SECONDS` | `20` | Tope de duración de una frase |

---

## Uso

Todos los comandos se ejecutan desde cualquier carpeta usando la ruta absoluta del intérprete del venv.

### Asistente de voz

```bash
~/Documentos/EDITH/.venv/bin/python ~/Documentos/EDITH/assistant.py
```

- Habla y **pulsa Enter** cuando termines la frase: se envía al instante (no espera silencio).
- EDITH responde con voz (femenina) y sigue escuchando.
- Escribe `salir` + Enter (o `Ctrl+C`) para salir.
- Requiere Ollama corriendo.

### Texto a voz

```bash
~/Documentos/EDITH/.venv/bin/python ~/Documentos/EDITH/tts.py "Hola, soy Edith"
```

Opciones:

```bash
# Guardar en un archivo sin reproducir
~/Documentos/EDITH/.venv/bin/python ~/Documentos/EDITH/tts.py "Hola" -o saludo.wav --no-play

# Usar Kokoro en vez de Piper, con otra voz (0=af_maple, 1=af_sol, 2=bf_vale)
~/Documentos/EDITH/.venv/bin/python ~/Documentos/EDITH/tts.py "Hola" --engine kokoro -s 1
```

### Transcripción de audio

```bash
~/Documentos/EDITH/.venv/bin/python ~/Documentos/EDITH/transcribe.py audio.wav
```

Admite wav, mp3, ogg, etc. Opción `-m tiny|base|small|medium` para elegir el modelo.

---

## Motores de voz

**Piper (por defecto)** — `es_ES-sharvard-medium`, voz **femenina española** (hablante F, `PIPER_SPEAKER_ID=1`; el modelo tiene 2 hablantes: M=0, F=1). Rápida, ligera, ONNX, 22 kHz. Se descarga pública desde `rhasspy/piper-voices`. `es_ES-davefx-medium` queda como alternativa (cambia las rutas en `config.py`).

**Kokoro** — calidad muy alta, pero el modelo multilingüe accesible públicamente **no tiene voces españolas**: solo `af_maple` (sid 0), `af_sol` (sid 1, femeninas americanas) y `bf_vale` (sid 2, británica). Pronuncian español correctamente pero con timbre anglosajón. Para obtener voces Kokoro 100% españolas haría falta el modelo oficial `hexgrad/Kokoro-82M-v1.1`, que es "gated" (requiere cuenta y token de HuggingFace).

---

## Solución de problemas

- **`ModuleNotFoundError: No module named 'pkg_resources'`** (al importar `webrtcvad`)
  Setuptools >= 81 eliminó `pkg_resources`. Solución:
  ```bash
  ~/Documentos/EDITH/.venv/bin/pip install 'setuptools<81'
  ```

- **`torch` / paquetes NVIDIA (CUDA)**
  No hace falta torch. Si `pip` intenta instalarlo (p. ej. al instalar `kokoro` o `silero-vad` originales), está arrastrando ~2 GB de CUDA. Usa `piper-tts` y `sherpa-onnx`, que no lo necesitan. Nunca uses `pip install torch` por defecto en esta máquina sin GPU NVIDIA.

- **Error `InvalidProtobuf: Protobuf parsing failed` al cargar el modelo Kokoro**
  El archivo `kokoro-multilang.onnx` está incompleto. Reanuda y verifica el tamaño (325 631 784 bytes):
  ```bash
  cd ~/Documentos/EDITH/models
  curl -L -C - -o kokoro-multilang.onnx "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/model.onnx"
  ```

- **`espeak-ng` no se encuentra**
  No lo instales con pacman: el venv ya lo trae. Los scripts llaman a `espeakng_loader.load_library()` antes de importar `sherpa_onnx`, que carga `libespeak-ng.so` en el proceso.

- **El asistente no te oye**
  Comprueba que `sounddevice` ve tu micrófono:
  ```bash
  ~/Documentos/EDITH/.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
  ```
  Y que el VAD no esté demasiado agresivo: baja `VAD_AGGRESSIVENESS` a 1 en `config.py`.

- **Ollama no responde**
  ```bash
  curl http://localhost:11434/api/tags
  ```
  Si no responde, arranca Ollama (`systemctl --user start ollama` o el servicio de tu distro) y comprueba que el modelo `edith` existe (`ollama list`).

---

## Notas técnicas

- **Por qué Python 3.14 no es un problema aquí**: `faster-whisper` (wheels ctranslate2), `sherpa-onnx`, `onnxruntime`, `piper-tts` (wheel `cp39-abi3`, ABI estable) tienen soporte. La pieza problemática sería `torch`, que se evita a propósito.
- **VAD de `webrtcvad`**: requiere que setuptools sea < 81 (ver arriba).
- **Formato de los modelos Kokoro**: el `.bin` de sherpa-onnx (`voices.bin`) es binario propio, no numpy; por eso Kokoro se ejecuta con `sherpa-onnx` y no con la librería `kokoro-onnx`. `kokoro-onnx` queda instalado solo como proveedor de `espeakng-loader`.
- **Rendimiento en esta máquina** (Intel iGPU, 15 GB RAM, sin GPU NVIDIA): STT con `small` ~ tiempo real; TTS Piper ~ instantáneo; respuesta LLM `edith` (base qwen2.5:3b) ~ 5-10 s en CPU.
- **Espacio en disco**: modelos ~ 425 MB (Piper 61 MB + Kokoro 362 MB) + modelo Whisper `small` ~ 466 MB en `~/.cache/huggingface`. La caché de pip y `~/.ollama` son aparte.

---

*EDITH: un asistente de voz local, reproducible y sin nube.*
