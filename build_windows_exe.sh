#!/usr/bin/env bash
# build_windows_exe.sh
# Script Windows-only (para Git Bash / MSYS / CI windows-latest) que crea TensorSuggestLite.exe
# Uso:
#   ./build_windows_exe.sh [--no-venv] [--onefile] [--windowed] [--icon path] [--clean] [--force]
# Ejemplos:
#   ./build_windows_exe.sh --onefile --icon icons/TensorSuggestLite.svg
#   ./build_windows_exe.sh --no-venv --onefile --icon build/TensorSuggestLite.ico

set -euo pipefail

# Configuración por defecto
VENV_DIR=.venv_win_pyinstaller
BUILD_VENV=1
ONEFILE=0
WINDOWED=1
CLEAN_BEFORE=0
FORCE=0
APP_NAME=TensorSuggestLite
ENTRY_POINT=tensorsuggestlite.py
DIST_DIR=dist
BUILD_DIR=build
PROJECT_ROOT=$(pwd)
ICON_INPUT=icons/TensorSuggestLite.svg
ICON_ICO=${BUILD_DIR}/TensorSuggestLite.ico
PYINSTALLER_MIN_VERSION=6.0

# Parsear argumentos
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv)
      BUILD_VENV=0; shift;;
    --onefile)
      ONEFILE=1; shift;;
    --windowed)
      WINDOWED=1; shift;;
    --no-windowed)
      WINDOWED=0; shift;;
    --icon)
      if [[ -z "${2:-}" ]]; then echo "--icon requiere argumento" >&2; exit 1; fi
      ICON_INPUT="$2"; shift 2;;
    --clean)
      CLEAN_BEFORE=1; shift;;
    --force)
      FORCE=1; shift;;
    --help|-h)
      sed -n '1,120p' "$0"; exit 0;;
    *) echo "Argumento desconocido: $1"; exit 1;;
  esac
done

# Detectar plataforma (esperamos Windows/Git Bash)
OS_NAME=$(uname -s 2>/dev/null || echo "Unknown")
if [[ "$OS_NAME" != MINGW* && "$OS_NAME" != MSYS* && "$OS_NAME" != CYGWIN* && "$OS_NAME" != "Windows_NT" ]]; then
  echo "Advertencia: este script está pensado para ejecutarse en Windows (Git Bash / CI windows-latest)." >&2
  if [[ $FORCE -ne 1 ]]; then
    echo "Si deseas continuar en este SO, vuelve a ejecutar con --force." >&2
    exit 1
  else
    echo "--force proporcionado: continuando de todas formas." >&2
  fi
fi

# Limpieza opcional
if [[ $CLEAN_BEFORE -eq 1 ]]; then
  echo "==> Limpieza previa: borrando $BUILD_DIR, $DIST_DIR y specs antiguas"
  rm -rf "$BUILD_DIR" || true
  rm -rf "$DIST_DIR" || true
  find . -maxdepth 1 -name "${APP_NAME}.spec" -exec rm -f {} + || true
fi

# Crear/activar venv
if [[ $BUILD_VENV -eq 1 ]]; then
  echo "==> Crear/activar venv en $VENV_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv "$VENV_DIR"
  fi
  if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Git Bash / MSYS
    # shellcheck disable=SC1090
    source "$VENV_DIR/Scripts/activate"
  elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # fallback (WSL/unix-like)
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  else
    echo "ERROR: no se encontró script de activación en $VENV_DIR" >&2
    echo "Si estás en PowerShell, activa manualmente: $VENV_DIR\\Scripts\\Activate.ps1" >&2
    exit 1
  fi
else
  echo "==> Usando entorno actual (no se creó venv)"
fi

# Actualizar pip y setuptools
echo "==> Actualizando pip y wheel"
python -m pip install --upgrade pip setuptools wheel

# Instalar PyInstaller
echo "==> Instalando PyInstaller (si falta)"
python -m pip install "pyinstaller>=$PYINSTALLER_MIN_VERSION"

# Instalar dependencias ligeras (para evitar paquetes muy pesados por defecto)
echo "==> Instalando dependencias base necesarias"
python -m pip install PyQt6 PyYAML toml || true
# Si prefieres instalar todo: python -m pip install -r requirements.txt

# Preparar directorios
mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# Preparar icono: soporta .ico o .svg -> .ico via ImageMagick (magick)
ICON_EXT="${ICON_INPUT##*.}"
if [[ -f "$ICON_INPUT" ]]; then
  if [[ "$ICON_EXT" == "ico" ]]; then
    echo "==> Icono .ico proporcionado: $ICON_INPUT"
    cp -f "$ICON_INPUT" "$ICON_ICO"
  elif [[ "$ICON_EXT" == "svg" ]]; then
    echo "==> SVG detectado: intentando convertir a .ico con ImageMagick (magick)"
    if command -v magick >/dev/null 2>&1; then
      TMPDIR="$BUILD_DIR/icon_pngs"
      mkdir -p "$TMPDIR"
      echo "==> Generando PNGs: 16,32,48,256"
      magick -background none -resize 16x16 "$ICON_INPUT" "$TMPDIR/icon_16.png"
      magick -background none -resize 32x32 "$ICON_INPUT" "$TMPDIR/icon_32.png"
      magick -background none -resize 48x48 "$ICON_INPUT" "$TMPDIR/icon_48.png"
      magick -background none -resize 256x256 "$ICON_INPUT" "$TMPDIR/icon_256.png"
      echo "==> Creando .ico"
      magick "$TMPDIR/icon_16.png" "$TMPDIR/icon_32.png" "$TMPDIR/icon_48.png" "$TMPDIR/icon_256.png" "$ICON_ICO"
      if [[ -f "$ICON_ICO" ]]; then
        echo "==> Icono .ico creado en $ICON_ICO"
      else
        echo "ERROR: falló la creación del .ico con ImageMagick" >&2
        exit 1
      fi
    else
      echo "ERROR: 'magick' no detectado en PATH. Instala ImageMagick (choco install imagemagick) y rerun." >&2
      exit 1
    fi
  else
    echo "ERROR: formato de icono no soportado: .$ICON_EXT" >&2
    exit 1
  fi
else
  echo "Aviso: no se encontró $ICON_INPUT. Se construirá sin icono personalizado." >&2
  ICON_ICO=""
fi

# Construir argumentos de PyInstaller
PYI_ARGS=(--noconfirm --clean)
if [[ $ONEFILE -eq 1 ]]; then
  PYI_ARGS+=(--onefile)
fi
if [[ $WINDOWED -eq 1 ]]; then
  PYI_ARGS+=(--windowed)
fi
PYI_ARGS+=(--name "$APP_NAME")
if [[ -n "$ICON_ICO" && -f "$ICON_ICO" ]]; then
  PYI_ARGS+=(--icon "$ICON_ICO")
fi
# Añadir recursos (Windows pyinstaller espera separador ';' en --add-data)
# Usamos rutas absolutas para evitar ambigüedades
PYI_ARGS+=(--add-data "${PROJECT_ROOT}/styles_dark.qss;.")
PYI_ARGS+=(--add-data "${PROJECT_ROOT}/styles_light.qss;.")
PYI_ARGS+=(--add-data "${PROJECT_ROOT}/icons;icons")

# Evitar conflictos previos raros
if [[ -e "$BUILD_DIR/${APP_NAME}.spec" ]]; then
  echo "==> Eliminando spec previo raro: $BUILD_DIR/${APP_NAME}.spec"
  rm -f "$BUILD_DIR/${APP_NAME}.spec" || true
fi

# Ejecutar PyInstaller
echo "==> Ejecutando PyInstaller: python -m PyInstaller ${PYI_ARGS[*]} $ENTRY_POINT"
# shellcheck disable=SC2086
python -m PyInstaller ${PYI_ARGS[*]} "$ENTRY_POINT"

# Verificar salida
EXE_PATH1="$DIST_DIR/${APP_NAME}.exe"
EXE_PATH2="$DIST_DIR/${APP_NAME}/${APP_NAME}.exe"
if [[ -f "$EXE_PATH1" ]]; then
  echo "==> Build finalizado. Ejecutable: $EXE_PATH1"
elif [[ -f "$EXE_PATH2" ]]; then
  echo "==> Build finalizado. Ejecutable: $EXE_PATH2"
else
  echo "ERROR: No se encontró ejecutable en dist/. Revisa la salida de PyInstaller." >&2
  exit 1
fi

# Mensaje final
echo "==> Listo. Comprueba $DIST_DIR para el exe. Si necesitas empaquetar, crea un ZIP del exe y la carpeta 'icons' si es necesaria."

# Mantener venv activo para depuración
if [[ $BUILD_VENV -eq 1 ]]; then
  echo "Venv activo en $VENV_DIR (usa 'deactivate' para salir)"
fi

