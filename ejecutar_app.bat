@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "LOG=%~dp0ejecutar_app.log"
echo ============ Grietas y riesgo ============
echo [%date% %time%] inicio > "%LOG%"
echo Carpeta: %~dp0 >> "%LOG%"

rem --- 1) buscar Python: primero "python", despues el lanzador "py" ---
set "PY="
python -c "import sys" >nul 2>&1 && set "PY=python"
if not defined PY ( py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11" )
if not defined PY ( py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10" )
if not defined PY ( py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12" )
if not defined PY (
  echo.
  echo No encuentro Python en este computador.
  echo Instala Python 3.11 desde https://www.python.org/downloads/
  echo y MARCA la casilla "Add python.exe to PATH". Despues vuelve a abrir este archivo.
  pause
  exit /b 1
)
echo Usando: %PY%
%PY% --version
%PY% --version >> "%LOG%" 2>&1

rem --- 2) entorno propio del proyecto (solo la primera vez) ---
if not exist ".venv\Scripts\activate.bat" (
  echo Creando el entorno ^(solo la primera vez^)...
  %PY% -m venv .venv >> "%LOG%" 2>&1
)
if not exist ".venv\Scripts\activate.bat" (
  echo No se pudo crear el entorno. Mira el archivo !LOG!
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"

rem --- 3) paquetes (solo la primera vez) ---
python -c "import streamlit, tensorflow" >nul 2>&1
if errorlevel 1 (
  echo Instalando paquetes ^(solo la primera vez, tarda varios minutos; no cierres esta ventana^)...
  python -m pip install --upgrade pip >> "%LOG%" 2>&1
  python -m pip install -r requirements-app.txt >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo.
    echo FALLO la instalacion. Estas son las ultimas lineas del registro:
    echo ------------------------------------------------------------
    powershell -NoProfile -Command "Get-Content -Tail 15 '!LOG!'"
    echo ------------------------------------------------------------
    echo El registro completo esta en: !LOG!
    echo Si el error habla de rutas largas, mueve esta carpeta a C:\grietas y vuelve a abrir este archivo.
    pause
    exit /b 1
  )
)

rem --- 4) la app ---
echo Abriendo la app en tu navegador ^(si no se abre, entra a http://localhost:8501^)...
python -m streamlit run app\streamlit_app.py
if errorlevel 1 (
  echo.
  echo La app termino con un error. Copia los mensajes de arriba.
)
pause