"""complejidad.py — Eficiencia y complejidad (15 % de la rúbrica).

El documento del reto pide, en la Etapa 4, reportar para cada modelo:
    * número de parámetros
    * tamaño del modelo en disco (MB)
    * tiempo de inferencia por imagen
y en la Entrega 3, comparar desempeño Y complejidad entre versiones.

CÓMO SE MIDE EL TIEMPO (para que sea honesto y comparable)
  * Se descartan las primeras llamadas (calentamiento): la primera siempre es lenta.
  * Se reporta la MEDIANA y el percentil 95, no solo el promedio: un promedio se
    contamina con un solo pico.
  * Se mide UNA imagen por llamada, que es el caso real de la app (el usuario sube
    una foto). En lote el costo por imagen es menor, pero no es lo que vive el usuario.
  * Una CPU de Colab NO es un celular. Estas cifras sirven para COMPARAR versiones
    entre sí; el tiempo en un celular real se mide en el celular y se reporta aparte.

Unidad: 1 MB = 10^6 bytes. Usá la misma unidad en todas las versiones.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _n_elementos(peso) -> int:
    forma = peso.shape
    if hasattr(forma, "num_elements"):
        return int(forma.num_elements())
    return int(np.prod(tuple(forma)))


def contar_parametros(modelo) -> dict:
    """Parámetros totales, entrenables y no entrenables de un modelo de Keras."""
    total = int(modelo.count_params())
    entrenables = int(sum(_n_elementos(w) for w in modelo.trainable_weights))
    return {"parametros": total, "entrenables": entrenables, "no_entrenables": total - entrenables}


def tamano_mb(ruta) -> float:
    """Tamaño de un archivo (o de una carpeta, para el formato SavedModel) en MB."""
    ruta = Path(ruta)
    if ruta.is_dir():
        return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file()) / 1e6
    return os.path.getsize(ruta) / 1e6


def medir_tiempo(fn, n: int = 100, calentamiento: int = 10) -> dict:
    """Milisegundos por llamada a `fn()`: mediana, p95, media y desvío."""
    for _ in range(calentamiento):
        fn()
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms = np.array(ms)
    return {"ms_mediana": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
            "ms_media": float(ms.mean()), "ms_desvio": float(ms.std())}


def imagen_de_prueba(semilla: int = config.SEMILLA) -> np.ndarray:
    """Una imagen aleatoria (1, 224, 224, 3) con valores 0-255. El contenido no cambia el
    tiempo de una red convolucional: hace las mismas operaciones con cualquier imagen."""
    rng = np.random.default_rng(semilla)
    return rng.uniform(0, 255, (1, config.IMG, config.IMG, 3)).astype(np.float32)


def tiempo_keras(modelo_inferencia, n: int = 100) -> dict:
    """Tiempo de UNA imagen con el modelo de Keras (sin aumentación ni dropout)."""
    x = imagen_de_prueba()
    return medir_tiempo(lambda: np.asarray(modelo_inferencia(x, training=False)), n=n)


def tflite_predictor(ruta, hilos: int = 1):
    """Función `predecir(lote) -> probabilidades` que ejecuta un .tflite imagen por imagen.

    Sirve para dos cosas: medir tiempo y comprobar que la conversión NO cambió las
    predicciones (se compara contra el modelo de Keras).
    """
    import tensorflow as tf

    interprete = tf.lite.Interpreter(model_path=str(ruta), num_threads=hilos)
    interprete.allocate_tensors()
    entrada, salida = interprete.get_input_details()[0], interprete.get_output_details()[0]

    def predecir(lote):
        lote = np.asarray(lote, dtype=np.float32)
        resultado = []
        for imagen in lote:
            interprete.set_tensor(entrada["index"], imagen[None])
            interprete.invoke()
            resultado.append(float(interprete.get_tensor(salida["index"]).ravel()[0]))
        return np.array(resultado)

    return predecir


def tiempo_tflite(ruta, n: int = 100, hilos: int = 1) -> dict:
    """Tiempo de UNA imagen con un modelo TFLite (CPU, `hilos` hilos)."""
    predecir, x = tflite_predictor(ruta, hilos), imagen_de_prueba()
    return medir_tiempo(lambda: predecir(x), n=n)


def fila_complejidad(nombre: str, modelo_keras=None, modelo_inf=None, ruta_keras=None,
                     ruta_tflite=None, n: int = 100, hilos: int = 1) -> dict:
    """Una fila de la tabla de complejidad. Cada dato se calcula solo si se dio lo necesario."""
    fila = {"modelo": nombre}
    if modelo_keras is not None:
        fila.update(contar_parametros(modelo_keras))
    if ruta_keras is not None:
        fila["mb_keras"] = round(tamano_mb(ruta_keras), 2)
    if modelo_inf is not None:
        t = tiempo_keras(modelo_inf, n)
        fila["ms_keras_1img"], fila["ms_keras_p95"] = round(t["ms_mediana"], 2), round(t["ms_p95"], 2)
    if ruta_tflite is not None:
        fila["mb_tflite"] = round(tamano_mb(ruta_tflite), 2)
        t = tiempo_tflite(ruta_tflite, n, hilos)
        fila["ms_tflite_1img"], fila["ms_tflite_p95"] = round(t["ms_mediana"], 2), round(t["ms_p95"], 2)
    return fila


def tabla_complejidad(filas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(filas)


def fidelidad_tflite(predecir_keras, predecir_tflite, imagenes) -> dict:
    """¿La conversión a TFLite cambió las respuestas?

    Compara las probabilidades de Keras y TFLite sobre las mismas imágenes. Se espera una
    diferencia máxima del orden de 1e-2 en float16 y NINGÚN cambio de clase.
    """
    a, b = np.asarray(predecir_keras(imagenes)).ravel(), np.asarray(predecir_tflite(imagenes)).ravel()
    return {"dif_max": float(np.abs(a - b).max()), "dif_media": float(np.abs(a - b).mean()),
            "cambian_de_clase": int(((a >= config.UMBRAL_DECISION) != (b >= config.UMBRAL_DECISION)).sum()),
            "n": int(len(a))}
