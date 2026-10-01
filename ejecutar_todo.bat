@echo off
REM Lanzador de doble clic: usa el python del .venv sin necesidad de activar scripts de PowerShell
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" ejecutar_todo.py %*
echo.
pause
