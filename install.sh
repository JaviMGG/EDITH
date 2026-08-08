#!/usr/bin/env bash
# EDITH - asistente de voz local.
# Instalador completo: crea el venv, instala dependencias y descarga los modelos.
#
# Uso:
#   ./install.sh                  instala lo básico (venv + voz Piper)
#   ./install.sh --full           además descarga los modelos de Kokoro (motor TTS alternativo, ~380 MB)
#   ./install.sh --with-ollama    además comprueba el modelo LLM local 'edith' (requiere Ollama instalado y en marcha)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY=python3

FULL=0
OLLAMA=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --with-ollama) OLLAMA=1 ;;
    *) echo "ERROR: opción desconocida '$arg' (usa --full y/o --with-ollama)" >&2; exit 1 ;;
  esac
done

echo "==> EDITH: instalando en $DIR"

# 1) Python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: no encuentro '$PY'. Instálalo primero (p. ej. 'sudo pacman -S python' o tu gestor de paquetes)." >&2
  exit 1
fi
ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "    Python detectado: $ver"

# 2) Entorno virtual
if [ ! -d "$VENV" ]; then
  echo "==> Creando entorno virtual (.venv)..."
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 3) Dependencias de Python
echo "==> Instalando dependencias de Python..."
pip install --upgrade pip
pip install -r "$DIR/requirements.txt"

# 4) Voz Piper femenina (motor TTS por defecto)
mkdir -p "$DIR/models/piper"
echo "==> Descargando voz Piper femenina (es_ES-sharvard-medium)..."
cd "$DIR/models/piper"
curl -L -C - -o es_ES-sharvard-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx"
curl -L -C - -o es_ES-sharvard-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json"
if [ "$(wc -c < es_ES-sharvard-medium.onnx)" -lt 1000000 ]; then
  echo "ERROR: la descarga de la voz Piper falló (archivo demasiado pequeño). Vuelve a ejecutar ./install.sh." >&2
  exit 1
fi

# 5) Modelos de Kokoro (motor TTS alternativo, opcional)
if [ "$FULL" = 1 ]; then
  echo "==> Descargando modelos de Kokoro (--full)..."
  cd "$DIR/models"
  curl -L -C - -o kokoro-multilang.onnx \
    "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/model.onnx"
  curl -L -C - -o kokoro-voices.bin \
    "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/voices.bin"
  curl -L -C - -o kokoro-tokens.txt \
    "https://huggingface.co/csukuangfj/kokoro-multi-lang-v1_1/resolve/main/tokens.txt"
  if [ "$(wc -c < kokoro-multilang.onnx)" -ne 325631784 ]; then
    echo "ERROR: kokoro-multilang.onnx incompleto (se esperan exactamente 325 631 784 bytes). Vuelve a ejecutar ./install.sh." >&2
    exit 1
  fi
fi

# 6) Modelo LLM en Ollama (opcional)
if [ "$OLLAMA" = 1 ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "AVISO: Ollama no está instalado. Descárgalo de https://ollama.com, arráncalo y crea el modelo 'edith'." >&2
  else
    echo "==> Comprobando el modelo LLM local 'edith' en Ollama..."
    if ! ollama list | grep -q '^edith[[:space:]]'; then
      echo "AVISO: no encuentro el modelo 'edith'. Créalo con 'ollama create edith -f Modelfile'." >&2
    else
      echo "    Modelo 'edith' encontrado."
    fi
  fi
fi

echo
echo "==> ¡EDITH instalada!"
echo
echo "    Asistente de voz:   $VENV/bin/python $DIR/assistant.py"
echo "    Probar la voz:      $VENV/bin/python $DIR/tts.py \"Hola, soy Edith\""
echo "    Transcribir audio:  $VENV/bin/python $DIR/transcribe.py audio.wav"
echo
echo "    (El modelo de transcripción de Whisper se descarga solo la primera vez.)"
echo "    Habla y pulsa Enter para enviar; escribe 'salir' + Enter para terminar."
