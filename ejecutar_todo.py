"""
ejecutar_todo.py
----------------
Ejecuta TODO el flujo del proyecto con el dataset real, de principio a fin:

  1. Prepara data/raw/{Positive,Negative} (si solo tienes el archive.zip de Kaggle, lo descomprime)
  2. Análisis exploratorio (src/dataset.py)
  3. Entrena la línea base con fotos reales (src/baseline.py) -> models/baseline.joblib + métricas
  4. Predice la foto de demostración (src/predict.py)
  5. Corre las pruebas automáticas (tests/test_pipeline.py)
  6. (opcional) Entrena MobileNetV2 en este PC si TensorFlow está instalado (--entrenar-mobilenet)
  7. Abre la app web con el mejor modelo disponible (mobilenet.tflite si existe, si no baseline.joblib)

Uso (desde la raíz del proyecto, con el .venv activo):
    python ejecutar_todo.py                          # flujo completo con 4000 fotos por clase
    python ejecutar_todo.py --max-por-clase 1000     # más rápido, para probar
    python ejecutar_todo.py --sin-app                # todo menos abrir Streamlit
    python ejecutar_todo.py --zip C:\\Descargas\\archive.zip   # descomprime el dataset por ti
    python ejecutar_todo.py --entrenar-mobilenet --epochs 3    # además entrena MobileNet en CPU (lento)
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)                      # todas las rutas relativas parten de la raíz del proyecto
PY = sys.executable                 # el python del .venv activo
RAW = os.path.join("data", "raw")
CLASES = ("Positive", "Negative")
DEMO = os.path.join("docs", "demo", "columna_grieta_diagonal_+2.5deg.jpg")


def titulo(n, total, texto):
    print("\n" + "=" * 70)
    print(f"[{n}/{total}] {texto}")
    print("=" * 70, flush=True)


def correr(argumentos, que):
    """Ejecuta un script del proyecto mostrando su salida en vivo; se detiene si falla."""
    print(f"$ {' '.join(argumentos)}\n", flush=True)
    r = subprocess.run([PY] + argumentos)
    if r.returncode != 0:
        raise SystemExit(f"\nERROR en '{que}' (código {r.returncode}). Revisa el mensaje de arriba.")


def contar(clase):
    return sum(len(glob.glob(os.path.join(RAW, clase, e))) for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG"))


def aplanar_dataset():
    """Si al descomprimir quedó una carpeta intermedia (data/raw/algo/Positive), la sube a data/raw."""
    for clase in CLASES:
        destino = os.path.join(RAW, clase)
        if os.path.isdir(destino) and contar(clase) > 0:
            continue
        hallados = [d for d in glob.glob(os.path.join(RAW, "**", clase), recursive=True)
                    if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(destino)]
        if hallados:
            if os.path.isdir(destino):
                shutil.rmtree(destino)
            shutil.move(hallados[0], destino)


def preparar_dataset(ruta_zip):
    os.makedirs(RAW, exist_ok=True)
    aplanar_dataset()
    if all(contar(c) > 0 for c in CLASES):
        return
    # no hay fotos: buscar un zip (el que pasó el usuario, o cualquiera dentro de data/)
    candidatos = [ruta_zip] if ruta_zip else []
    candidatos += glob.glob(os.path.join("data", "*.zip")) + glob.glob(os.path.join("data", "raw", "*.zip"))
    candidatos = [c for c in candidatos if c and os.path.isfile(c)]
    if not candidatos:
        raise SystemExit(
            "\nNo encuentro el dataset. Opciones:\n"
            "  a) Descomprime archive.zip (Kaggle: Surface Crack Detection) de modo que queden\n"
            f"     {os.path.join(RAW, 'Positive')} y {os.path.join(RAW, 'Negative')} con fotos .jpg, o\n"
            "  b) Copia archive.zip dentro de la carpeta data/ y vuelve a ejecutar, o\n"
            "  c) Ejecuta: python ejecutar_todo.py --zip <ruta\\a\\archive.zip>")
    z = candidatos[0]
    print(f"Descomprimiendo {z} en {RAW} (puede tardar 1-2 minutos)...", flush=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(RAW)
    aplanar_dataset()
    if not all(contar(c) > 0 for c in CLASES):
        raise SystemExit(f"El zip {z} no contiene carpetas Positive/ y Negative/ con imágenes.")


def hay_interprete_tflite():
    for mod in ("ai_edge_litert", "tensorflow"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def resumen_metricas(ruta):
    if not os.path.exists(ruta):
        return
    m = json.load(open(ruta, encoding="utf-8"))
    t = m.get("test", {})
    print(f"\nMétricas en el conjunto de PRUEBA ({os.path.basename(ruta)}):")
    for k in ("exactitud", "precision", "recall", "f1"):
        if k in t:
            print(f"  {k:10s} {t[k]:.4f}")
    if "matriz_confusion" in t:
        print(f"  matriz de confusión [[VN, FP], [FN, VP]] = {t['matriz_confusion']}")


def main():
    ap = argparse.ArgumentParser(description="Ejecuta todo el flujo con el dataset real.")
    ap.add_argument("--max-por-clase", type=int, default=4000, help="fotos por clase (default 4000)")
    ap.add_argument("--features", choices=["stats", "hog", "ambos"], default="stats")
    ap.add_argument("--zip", default=None, help="ruta al archive.zip de Kaggle si aún no está descomprimido")
    ap.add_argument("--sin-app", action="store_true", help="no abrir la app web al final")
    ap.add_argument("--sin-pruebas", action="store_true", help="saltar tests/test_pipeline.py")
    ap.add_argument("--entrenar-mobilenet", action="store_true",
                    help="entrenar MobileNetV2 aquí (requiere tensorflow; en CPU es lento)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=3)
    a = ap.parse_args()

    total = 7
    titulo(1, total, "Preparar el dataset en data/raw/{Positive,Negative}")
    preparar_dataset(a.zip)
    for c in CLASES:
        print(f"  {c:9s}: {contar(c):6d} imágenes")

    titulo(2, total, "Análisis exploratorio (EDA)")
    correr(["src/dataset.py", "--root", RAW, "--max-por-clase", str(a.max_por_clase)], "EDA")

    titulo(3, total, f"Entrenar línea base con {a.max_por_clase} fotos reales por clase")
    correr(["src/baseline.py", "--root", RAW, "--max-por-clase", str(a.max_por_clase),
            "--features", a.features, "--out", "models/baseline.joblib"], "línea base")
    resumen_metricas("models/baseline_metricas.json")

    titulo(4, total, "Predicción de la foto de demostración con la línea base")
    correr(["src/predict.py", DEMO, "--modelo", "models/baseline.joblib", "--elemento", "columna",
            "--debug", "docs/demo/salida_lineas.jpg"], "predicción")

    titulo(5, total, "Pruebas automáticas")
    if a.sin_pruebas:
        print("(saltadas con --sin-pruebas)")
    else:
        correr(["tests/test_pipeline.py"], "pruebas")

    titulo(6, total, "MobileNetV2 (transfer learning)")
    tflite = "models/mobilenet.tflite"
    if a.entrenar_mobilenet:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise SystemExit("--entrenar-mobilenet requiere TensorFlow: pip install tensorflow "
                             "(o entrena en Colab con entrenar_mobilenet_colab.ipynb).")
        correr(["src/train_transfer.py", "--root", RAW, "--max-por-clase", str(a.max_por_clase),
                "--epochs", str(a.epochs), "--epochs-finetune", str(a.epochs_finetune),
                "--out", "models/mobilenet"], "MobileNet")
        resumen_metricas("models/mobilenet_metricas.json")
    elif os.path.exists(tflite):
        print(f"Encontrado {tflite} (entrenado en Colab).")
        resumen_metricas("models/mobilenet_metricas.json")
    else:
        print("No hay models/mobilenet.tflite. Entrénalo en Colab con entrenar_mobilenet_colab.ipynb\n"
              "o aquí con: python ejecutar_todo.py --entrenar-mobilenet (necesita tensorflow).")

    if os.path.exists(tflite) and hay_interprete_tflite():
        modelo_app = tflite
        correr(["src/predict.py", DEMO, "--modelo", tflite, "--elemento", "columna"], "predicción tflite")
    else:
        modelo_app = "models/baseline.joblib"
        if os.path.exists(tflite):
            print("Existe mobilenet.tflite pero falta el intérprete: pip install ai-edge-litert "
                  "(o pip install tensorflow). Mientras tanto la app usará la línea base.")

    titulo(7, total, f"App web con el modelo {modelo_app}")
    if a.sin_app:
        print("(no se abre por --sin-app). Para abrirla:\n"
              f"  {PY} -m streamlit run src/app_streamlit.py -- --modelo {modelo_app}")
        return
    print("Se abrirá en el navegador (http://localhost:8501). Ctrl+C en esta terminal para cerrarla.\n")
    subprocess.run([PY, "-m", "streamlit", "run", "src/app_streamlit.py", "--", "--modelo", modelo_app])


if __name__ == "__main__":
    main()
