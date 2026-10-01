"""
INVENTARIO DEL PROYECTO
Revisa que tenes guardado. No modifica ni borra nada.

Uso:
    python inventario.py

Corrrelo desde la carpeta raiz de tu proyecto.
"""

import os
import sys

EXT_MODELO    = (".keras", ".h5", ".hdf5", ".tflite", ".pb")
EXT_HISTORIAL = (".json", ".csv", ".pkl", ".pickle", ".npy")
EXT_NOTEBOOK  = (".ipynb",)
IGNORAR       = {".venv", "venv", "__pycache__", ".git", "node_modules", "data"}


def bonito(n_bytes):
    for unidad in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unidad}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


print("=" * 64)
print("INVENTARIO DEL PROYECTO")
print("=" * 64)
print(f"Carpeta actual: {os.getcwd()}\n")

modelos, historiales, notebooks = [], [], []

for raiz, dirs, archivos in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORAR and not d.startswith(".")]
    for a in archivos:
        ruta = os.path.join(raiz, a)
        low = a.lower()
        if low.endswith(EXT_MODELO):
            modelos.append(ruta)
        elif low.endswith(EXT_HISTORIAL) and any(
            p in low for p in ("hist", "log", "train", "metric", "curva")
        ):
            historiales.append(ruta)
        elif low.endswith(EXT_NOTEBOOK):
            notebooks.append(ruta)

# ------------------------------------------------------------------
print("-" * 64)
print("1. MODELOS ENCONTRADOS")
print("-" * 64)
if modelos:
    for m in modelos:
        print(f"  {m}   ({bonito(os.path.getsize(m))})")
else:
    print("  NINGUNO.")
    print("  -> Si entrenaste en Colab, el modelo quedo alla.")
    print("     Hay que descargarlo antes de seguir.")

# ------------------------------------------------------------------
print("\n" + "-" * 64)
print("2. POSIBLES HISTORIALES DE ENTRENAMIENTO")
print("-" * 64)
if historiales:
    for h in historiales:
        print(f"  {h}   ({bonito(os.path.getsize(h))})")
    print("\n  -> Si alguno guarda loss y val_loss por epoca,")
    print("     las curvas salen sin reentrenar.")
else:
    print("  NINGUNO.")
    print("  -> Las curvas exigen reentrenar. Es la ruta lenta,")
    print("     pero con la base congelada son 15-30 min en Colab.")

# ------------------------------------------------------------------
print("\n" + "-" * 64)
print("3. NOTEBOOKS")
print("-" * 64)
if notebooks:
    for n in notebooks:
        print(f"  {n}   ({bonito(os.path.getsize(n))})")
    print("\n  -> Si alguno tiene las SALIDAS guardadas, puede que las")
    print("     curvas ya esten dibujadas adentro. Abrilo y revisa.")
else:
    print("  NINGUNO en esta carpeta.")

# ------------------------------------------------------------------
print("\n" + "-" * 64)
print("4. RESUMEN DEL MODELO (parametros)")
print("-" * 64)

candidatos = [m for m in modelos if m.lower().endswith((".keras", ".h5", ".hdf5"))]
if not candidatos:
    print("  No hay modelo Keras (.keras / .h5).")
    if any(m.lower().endswith(".tflite") for m in modelos):
        print("  Solo hay .tflite: sirve para inferencia, pero NO da")
        print("  el conteo de parametros entrenables que pide el profesor.")
else:
    ruta = candidatos[0]
    print(f"  Intentando cargar: {ruta}\n")
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}\n")
        modelo = tf.keras.models.load_model(ruta, compile=False)
        modelo.summary()

        total = modelo.count_params()
        entrenables = sum(
            int(tf.size(w)) for w in modelo.trainable_weights
        )
        print("\n  " + "=" * 50)
        print(f"  Parametros totales      : {total:,}")
        print(f"  Parametros entrenables  : {entrenables:,}")
        print(f"  Parametros congelados   : {total - entrenables:,}")
        print(f"  Tamano en disco         : {bonito(os.path.getsize(ruta))}")
        print("  " + "=" * 50)
        print("\n  -> ANOTA ESTOS NUMEROS. Son la respuesta a")
        print("     '¿que parametros?' de tu profesor.")

    except ImportError:
        print("  TensorFlow no esta instalado en este entorno.")
        print("  Instalalo con:  pip install tensorflow")
    except Exception as e:
        print(f"  No se pudo cargar el modelo: {type(e).__name__}")
        print(f"  {e}")

print("\n" + "=" * 64)
print("FIN DEL INVENTARIO")
print("=" * 64)
print("Copia TODA esta salida y pegala en el chat.\n")
