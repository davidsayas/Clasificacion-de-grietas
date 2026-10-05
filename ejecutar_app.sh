#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "============ Grietas y riesgo ============"
command -v python3 >/dev/null || { echo "No encuentro python3. Instala Python 3.11 desde https://www.python.org/downloads/"; exit 1; }
[ -d .venv ] || { echo "Creando el entorno (solo la primera vez)..."; python3 -m venv .venv; }
source .venv/bin/activate
python -c "import streamlit, tensorflow" >/dev/null 2>&1 || {
  echo "Instalando paquetes (solo la primera vez, tarda varios minutos)..."
  python -m pip install --upgrade pip && python -m pip install -r requirements-app.txt || { echo "FALLO la instalacion."; exit 1; }
}
streamlit run app/streamlit_app.py
