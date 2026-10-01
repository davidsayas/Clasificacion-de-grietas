@echo off
REM Abre la app web accesible desde el celular (misma Wi-Fi) e imprime la direccion a usar
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" app_celular.py %*
echo.
pause
