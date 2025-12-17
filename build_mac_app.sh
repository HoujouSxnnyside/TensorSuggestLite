#!/usr/bin/env zsh
# Script reproducible para construir TensorSuggestLite.app en macOS usando PyInstaller
# Uso:
#  ./build_mac_app.sh [--no-venv] [--with-tf]
# Opciones:
#  --no-venv   : no crear ni activar virtualenv (usar entorno actual)
#  --with-tf   : instalar tensorflow en el entorno de build (incrementa tamaño)

set -euo pipefail

# Evitar errores de globbing en zsh cuando no hay coincidencias
setopt nullglob 2>/dev/null || true

BUILD_VENV=1
BUILD_WITH_TF=0
VENV_DIR=.venv_build
PYINSTALLER_VER=5.11
APP_NAME=TensorSuggestLite
# Forzamos el uso de invocación CLI para PyInstaller (asegura control de datas/icon)

# Definir DIST_DIR y BUILD_DIR antes de usarlos
DIST_DIR=dist
BUILD_DIR=build
PROJECT_ROOT=$(pwd)
ICON_SVG=icons/TensorSuggestLite.svg
ICON_PNG=${BUILD_DIR}/icon_512x512.png
ICON_ICNS=${BUILD_DIR}/TensorSuggestLite.icns

# Parse args
for arg in "$@"; do
  case $arg in
    --no-venv)
      BUILD_VENV=0
      ;;
    --with-tf)
      BUILD_WITH_TF=1
      ;;
    *)
      echo "Unknown arg: $arg"
      exit 1
      ;;
  esac
done

# Limpieza explícita: borrar contenidos previos para evitar artefactos mezclados
rm -rf "$BUILD_DIR"/* || true
rm -rf "$DIST_DIR"/* || true

if [ "$BUILD_VENV" -eq 1 ]; then
  echo "==> Creando virtualenv en $VENV_DIR (si no existe)"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
else
  echo "==> Usando entorno actual (no se crea venv)"
fi

echo "==> Actualizando pip y setuptools"
pip install --upgrade pip setuptools wheel

echo "==> Instalando PyInstaller"
pip install "pyinstaller>=$PYINSTALLER_VER"

# Comprobar rsvg-convert (librsvg) — requerido para generar icns desde SVG
if ! command -v rsvg-convert >/dev/null 2>&1; then
  echo "ERROR: 'rsvg-convert' no está instalado o no está en PATH. Instálalo con Homebrew: 'brew install librsvg' y vuelve a ejecutar." >&2
  exit 1
fi

if [ "$BUILD_WITH_TF" -eq 1 ]; then
  echo "==> Instalando dependencias completas (incluye tensorflow)"
  pip install -r requirements.txt
else
  echo "==> Instalando dependencias ligeras (sin tensorflow)"
  pip install numpy scikit-learn PyQt6 PyYAML toml
fi

# Crear build dir
mkdir -p "$BUILD_DIR"

# Generar .icns desde svg usando rsvg-convert exclusivamente
if [ -f "$ICON_SVG" ]; then
  echo "==> Generando PNG desde SVG con rsvg-convert"
  if rsvg-convert -w 512 -h 512 "$ICON_SVG" -o "$ICON_PNG"; then
    echo "==> PNG creado en $ICON_PNG"
  else
    echo "ERROR: rsvg-convert falló al generar PNG desde $ICON_SVG" >&2
    exit 1
  fi

  ICONSET_DIR="$BUILD_DIR/icon.iconset"
  mkdir -p "$ICONSET_DIR"
  echo "==> Generando tamaños de icono con sips"
  sips -z 16 16    "$ICON_PNG" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null 2>&1 || true
  sips -z 32 32    "$ICON_PNG" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null 2>&1 || true
  sips -z 128 128  "$ICON_PNG" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null 2>&1 || true
  sips -z 256 256  "$ICON_PNG" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null 2>&1 || true
  sips -z 512 512  "$ICON_PNG" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null 2>&1 || true

  if command -v iconutil >/dev/null 2>&1; then
    iconutil -c icns "$ICONSET_DIR" -o "$ICON_ICNS" >/dev/null 2>&1 || true
    if [ -f "$ICON_ICNS" ]; then
      echo "==> .icns generado en $ICON_ICNS"
    else
      echo "ERROR: iconutil no generó .icns correctamente" >&2
      exit 1
    fi
  else
    echo "ERROR: 'iconutil' no disponible en este sistema; no se puede generar .icns." >&2
    exit 1
  fi
else
  echo "Aviso: $ICON_SVG no existe; no se generará icono .icns"
fi

# Ejecutar PyInstaller vía CLI asegurando que QSS queden en Resources root y icons/ incluido
PYINSTALLER_CMD=(pyinstaller --noconfirm --clean --windowed --name "$APP_NAME")
if [ -f "$ICON_ICNS" ]; then
  PYINSTALLER_CMD+=(--icon "$ICON_ICNS")
fi
# Añadir datas
PYINSTALLER_CMD+=(--add-data "${PROJECT_ROOT}/styles_dark.qss:.")
PYINSTALLER_CMD+=(--add-data "${PROJECT_ROOT}/styles_light.qss:.")
PYINSTALLER_CMD+=(--add-data "${PROJECT_ROOT}/icons:icons")
# Script entry
PYINSTALLER_CMD+=(tensorsuggestlite.py)

echo "==> Ejecutando: ${PYINSTALLER_CMD[*]}"
# Ejecutar el comando
eval "${PYINSTALLER_CMD[*]}"

# Post-build: copiar icns a Resources y actualizar Info.plist
APP_PATH="$DIST_DIR/${APP_NAME}.app"
if [ -d "$APP_PATH" ]; then
  RES="$APP_PATH/Contents/Resources"
  PLIST="$APP_PATH/Contents/Info.plist"
  mkdir -p "$RES"
  if [ -f "$ICON_ICNS" ]; then
    cp -f "$ICON_ICNS" "$RES/$(basename "$ICON_ICNS")"
    # Set CFBundleIconFile to basename without extension
    ICON_BASENAME=$(basename "$ICON_ICNS" .icns)
    if command -v /usr/libexec/PlistBuddy >/dev/null 2>&1; then
      /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile $ICON_BASENAME" "$PLIST" 2>/dev/null || \
      /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string $ICON_BASENAME" "$PLIST" 2>/dev/null || true
    else
      echo "Aviso: PlistBuddy no disponible; Info.plist no fue actualizado automáticamente." >&2
    fi
  fi
  # Forzar que qss e icons estén disponibles también en Resources
  for q in "styles_dark.qss" "styles_light.qss"; do
    if [ -f "$PROJECT_ROOT/$q" ]; then
      cp -f "$PROJECT_ROOT/$q" "$RES/"
    fi
  done
  if [ -d "$PROJECT_ROOT/icons" ]; then
    rm -rf "$RES/icons" || true
    cp -R "$PROJECT_ROOT/icons" "$RES/"
  fi

  # Si PyInstaller dejó también la carpeta one-folder en dist/, eliminarla para dejar sólo el .app
  if [ -d "$DIST_DIR/${APP_NAME}" ] && [ -d "$DIST_DIR/${APP_NAME}.app" ]; then
    echo "==> Eliminando carpeta redundante dist/${APP_NAME} dejando solo ${APP_NAME}.app"
    rm -rf "$DIST_DIR/${APP_NAME}"
  fi
fi

# Comprobar salida
if [ -d "$DIST_DIR/${APP_NAME}.app" ]; then
  echo "==> Build finalizado. App en: $DIST_DIR/${APP_NAME}.app"
  echo "Puedes abrirla con: open $DIST_DIR/${APP_NAME}.app"
else
  echo "==> Build finalizado, pero no se encontró ${APP_NAME}.app en dist/" >&2
  echo "Revisa la salida de PyInstaller y build/warn-*.txt para detalles." >&2
fi

# Dejar el venv activado si lo creamos
if [ "$BUILD_VENV" -eq 1 ]; then
  echo "Venv activo en $VENV_DIR (deactivate para salir)"
fi

