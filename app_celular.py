"""
app_celular.py
--------------
Lanza la app web de modo que se pueda abrir desde el CELULAR conectado a la MISMA red Wi-Fi
que el computador, e imprime la dirección que hay que escribir en el navegador del teléfono.

Uso (desde la raíz del proyecto, con el .venv activo):
    python app_celular.py                                   # usa mobilenet.tflite si existe
    python app_celular.py --modelo models/baseline.joblib   # forzar la línea base
    python app_celular.py --puerto 8502

Notas:
  - En el celular la opción "Cámara" del navegador solo funciona con HTTPS; por Wi-Fi (http://)
    usa "Subir archivo": en el teléfono ese botón permite tomar la foto en el momento.
  - Si el celular no conecta, Windows suele estar bloqueando el puerto: ver docs/EJECUCION_CELULAR.md.
"""
import argparse
import os
import socket
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)


def ip_local():
    """IP del computador en la red local (la que ve el celular)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no envía nada; solo elige la interfaz de salida
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    defecto = "models/mobilenet.tflite" if os.path.exists("models/mobilenet.tflite") else "models/baseline.joblib"
    ap.add_argument("--modelo", default=defecto)
    ap.add_argument("--puerto", type=int, default=8501)
    a = ap.parse_args()

    ip = ip_local()
    print("=" * 66)
    print("  App lista para el celular (misma red Wi-Fi que este computador)")
    print("=" * 66)
    print(f"  Modelo : {a.modelo}")
    print(f"  En el CELULAR abre:   http://{ip}:{a.puerto}")
    print(f"  En este PC:           http://localhost:{a.puerto}")
    print("  Para cerrar: Ctrl+C en esta ventana")
    print("=" * 66, flush=True)

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app_streamlit.py",
                    "--server.address", "0.0.0.0", "--server.port", str(a.puerto),
                    "--server.headless", "true",
                    "--", "--modelo", a.modelo])


if __name__ == "__main__":
    main()
