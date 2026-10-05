"""entrenar.py — Entrenamiento reproducible, paso a paso.

QUÉ HACE EL ENTRENAMIENTO
-------------------------
Mostrarle a la red miles de imágenes con su respuesta correcta y ajustar sus
pesos para que se equivoque menos. Cada vuelta completa sobre el entrenamiento
es una ÉPOCA. Después de cada época se mide el error en VALIDACIÓN (imágenes que
la red NO usa para aprender): sirve para decidir cuándo parar y qué versión guardar.

  entrenamiento : la red APRENDE con esto
  validación    : se usa para ELEGIR (cuándo parar, qué época guardar)
  prueba        : se usa UNA vez, al final, para informar el resultado

LAS DOS FASES
  1. base congelada, tasa de aprendizaje 1e-3 (aprende solo la cabeza)
  2. ajuste fino, últimas capas libres, tasa 1e-5 (adapta la base con delicadeza)

CORRECCIONES RESPECTO DE LA VERSIÓN ANTERIOR
  * División por GRUPOS (src/datos.py): sin fuga entre entrenamiento y prueba.
  * El historial se guarda con append=False y UN archivo por fase. Antes, un
    intento interrumpido quedaba pegado al principio del historial (107 filas
    para 100 épocas).
  * Se guarda TODO lo necesario para reproducir: particiones, hiperparámetros,
    versiones y semilla.

Uso desde terminal:
    python -m src.entrenar --fuentes fuentes.json --nombre v2_mobilenet
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, datos


# --------------------------------------------------------------------------
# Datos -> tf.data
# --------------------------------------------------------------------------
def crear_dataset(df: pd.DataFrame, lote: int = 32, entrenar: bool = False,
                  semilla: int = config.SEMILLA):
    """Dataset de TensorFlow: lee cada imagen del disco, la lleva a 224x224 y arma lotes.

    Las imágenes quedan como float32 con valores 0-255 (el preprocesado y la
    aumentación están DENTRO del modelo).
    """
    import tensorflow as tf

    rutas = tf.constant(df["ruta"].tolist())
    y = tf.constant(df["etiqueta"].to_numpy(dtype="float32").reshape(-1, 1))
    ds = tf.data.Dataset.from_tensor_slices((rutas, y))
    if entrenar:
        ds = ds.shuffle(len(df), seed=semilla, reshuffle_each_iteration=True)

    def cargar(ruta, etiqueta):
        img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
        img = tf.image.resize(img, (config.IMG, config.IMG))
        img = tf.ensure_shape(img, (config.IMG, config.IMG, 3))
        return img, etiqueta

    return ds.map(cargar, num_parallel_calls=tf.data.AUTOTUNE).batch(lote).prefetch(tf.data.AUTOTUNE)


def sobremuestrear(df: pd.DataFrame, fuente: str, veces: int) -> pd.DataFrame:
    """Repite `veces` las filas de una fuente pequeña (ej.: negativos difíciles).

    Sin esto, 300 negativos difíciles entre 40.000 imágenes casi no influyen.
    Con `veces` muy alto la red los MEMORIZA: se vigila con la validación por escenas.
    """
    extra = df[df["fuente"] == fuente]
    return pd.concat([df] + [extra] * (veces - 1), ignore_index=True) if veces > 1 else df


def pesos_de_clase(df: pd.DataFrame) -> dict:
    """Peso por clase para compensar desbalance: n / (2 · n_clase)."""
    n, n1 = len(df), int(df["etiqueta"].sum())
    return {0: n / (2 * max(1, n - n1)), 1: n / (2 * max(1, n1))}


def cargar_imagenes(rutas, lado: int = config.IMG) -> np.ndarray:
    """Imágenes como arreglo (N, lado, lado, 3) float32 0-255, en RGB (para las pruebas)."""
    from PIL import Image
    return np.stack([np.asarray(Image.open(r).convert("RGB").resize((lado, lado)), dtype=np.float32)
                     for r in rutas])


def predecir_rutas(funcion, rutas, trozo: int = 128):
    """Aplica `funcion` (un predictor o un extractor de embeddings) a MUCHAS imágenes sin llenar la RAM.

    PROBLEMA QUE RESUELVE: cargar_imagenes() deja cada imagen como 224x224x3 números decimales
    (~0,6 MB). 6.000 imágenes juntas son ~3,6 GB, y Colab se queda sin memoria. Acá se cargan
    `trozo` imágenes, se predicen y se descartan antes de cargar las siguientes: el consumo de
    memoria es el de UN trozo, sin importar cuántas imágenes haya en total.
    """
    rutas = list(rutas)
    partes = [np.asarray(funcion(cargar_imagenes(rutas[i:i + trozo]))) for i in range(0, len(rutas), trozo)]
    return np.concatenate(partes, axis=0) if partes else np.array([])


def predictor(modelo_inf, lote: int = 64):
    """Convierte un modelo de Keras en `predecir(lote_de_imagenes) -> probabilidades`."""
    return lambda x: modelo_inf.predict(np.asarray(x, dtype=np.float32), batch_size=lote, verbose=0).ravel()


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def _callbacks(carpeta: Path, fase: int, paciencia: int):
    from tensorflow import keras
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(str(carpeta / f"mejor_fase{fase}.keras"),
                                            monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(str(carpeta / f"historial_fase{fase}.csv"), append=False)]


def entrenar(nombre: str, tr: pd.DataFrame, va: pd.DataFrame, arquitectura: str = "mobilenetv2",
             epocas1: int = 60, epocas2: int = 40, paciencia: int = 50, lote: int = 32,
             lr1: float = 1e-3, lr2: float = 1e-5, brillo: float = 0.2, ruido: float = 0.0,
             capas_descongeladas: int = 30, salida: Path = config.MODELOS) -> dict:
    """Entrena las dos fases y guarda modelo, historiales y metadatos en models/<nombre>/."""
    import tensorflow as tf
    from tensorflow import keras
    from . import modelo as M

    keras.utils.set_random_seed(config.SEMILLA)
    carpeta = Path(salida) / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    modelo = M.construir_modelo(arquitectura, lr=lr1, brillo=brillo, ruido=ruido)
    ds_tr, ds_va = crear_dataset(tr, lote, True), crear_dataset(va, lote, False)
    pesos = pesos_de_clase(tr)

    modelo.fit(ds_tr, validation_data=ds_va, epochs=epocas1, class_weight=pesos,
               callbacks=_callbacks(carpeta, 1, paciencia))
    M.activar_ajuste_fino(modelo, capas_descongeladas, lr2)
    modelo.fit(ds_tr, validation_data=ds_va, epochs=epocas2, class_weight=pesos,
               callbacks=_callbacks(carpeta, 2, paciencia))

    modelo.save(carpeta / "modelo.keras")
    tr.to_csv(carpeta / "particion_entrenamiento.csv", index=False)
    va.to_csv(carpeta / "particion_validacion.csv", index=False)
    meta = {"nombre": nombre, "arquitectura": arquitectura, "fecha": datetime.now().isoformat(timespec="seconds"),
            "semilla": config.SEMILLA, "imagenes_entrenamiento": len(tr), "imagenes_validacion": len(va),
            "epocas_fase1": epocas1, "epocas_fase2": epocas2, "paciencia": paciencia, "lote": lote,
            "lr_fase1": lr1, "lr_fase2": lr2, "brillo_aumentacion": brillo, "ruido_aumentacion": ruido,
            "capas_descongeladas": capas_descongeladas, "fuentes": sorted(tr["fuente"].unique().tolist()),
            "tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version()}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def barrido_lr(tr: pd.DataFrame, va: pd.DataFrame, lrs=(1e-2, 1e-3, 1e-4), arquitectura: str = "mobilenetv2",
               epocas: int = 5, n_max: int = 8000, lote: int = 32) -> pd.DataFrame:
    """Prueba varias tasas de aprendizaje con pocas épocas y pocos datos.

    La tasa de aprendizaje es cuánto se mueven los pesos en cada paso: muy alta, la
    red se descontrola; muy baja, aprende demasiado lento. Se elige la de menor
    val_loss. Es una prueba RÁPIDA para elegir, no el entrenamiento definitivo.
    """
    from tensorflow import keras
    from . import modelo as M

    sub = tr.sample(min(n_max, len(tr)), random_state=config.SEMILLA)
    ds_tr, ds_va = crear_dataset(sub, lote, True), crear_dataset(va, lote, False)
    filas = []
    for lr in lrs:
        keras.utils.set_random_seed(config.SEMILLA)
        m = M.construir_modelo(arquitectura, lr=lr)
        h = m.fit(ds_tr, validation_data=ds_va, epochs=epocas, verbose=0).history
        filas.append({"lr": lr, "val_loss_final": h["val_loss"][-1], "val_loss_min": min(h["val_loss"]),
                      "val_recall_final": h["val_recall"][-1]})
    return pd.DataFrame(filas).sort_values("val_loss_min").reset_index(drop=True)


def ajustar_dominio(base: str, nombre: str, tr: pd.DataFrame, va: pd.DataFrame, epocas: int = 10, lr: float = 1e-4,
                    capas: int = 60, paciencia: int = 4, lote: int = 32, salida: Path = config.MODELOS) -> dict:
    """Continúa el entrenamiento de un modelo ya entrenado con fotos de OTRO dominio (celulares).

    Retoma `base` (no empieza de cero), libera las últimas `capas` de la red y entrena con una tasa de aprendizaje baja
    para adaptarla sin destruir lo aprendido. Se detiene cuando la validación (paredes que NO entrenan) deja de mejorar
    y se queda con la mejor época.
    """
    from tensorflow import keras
    from . import modelo as M

    keras.utils.set_random_seed(config.SEMILLA)
    carpeta = Path(salida) / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    m = keras.models.load_model(Path(salida) / base / "modelo.keras")
    M.activar_ajuste_fino(m, capas, lr)
    ds_tr, ds_va = crear_dataset(tr, lote, True), crear_dataset(va, lote, False)
    m.fit(ds_tr, validation_data=ds_va, epochs=epocas, class_weight=pesos_de_clase(tr),
          callbacks=_callbacks(carpeta, 1, paciencia))
    m.save(carpeta / "modelo.keras")
    tr.to_csv(carpeta / "particion_entrenamiento.csv", index=False)
    va.to_csv(carpeta / "particion_validacion.csv", index=False)
    meta = {"nombre": nombre, "base": base, "tipo": "ajuste de dominio", "fecha": datetime.now().isoformat(timespec="seconds"),
            "epocas_max": epocas, "lr": lr, "capas_liberadas": capas, "paciencia": paciencia,
            "imagenes_entrenamiento": len(tr), "imagenes_validacion": len(va)}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrena el clasificador de grietas.")
    ap.add_argument("--fuentes", required=True, help="JSON con las fuentes (ver fuentes.ejemplo.json)")
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--arquitectura", default="mobilenetv2", choices=("mobilenetv2", "efficientnetb0"))
    ap.add_argument("--reservar", default=None, help="fuente a apartar para la prueba fuera de dominio")
    ap.add_argument("--sobremuestrear", default=None, help="fuente a repetir en el entrenamiento")
    ap.add_argument("--veces", type=int, default=5)
    ap.add_argument("--epocas1", type=int, default=60)
    ap.add_argument("--epocas2", type=int, default=40)
    a = ap.parse_args()

    df = datos.construir_inventario(json.loads(Path(a.fuentes).read_text(encoding="utf-8")))
    if a.reservar:
        df, _ = datos.reservar_fuente(df, a.reservar)
    tr, va, _ = datos.dividir(df)
    if a.sobremuestrear:
        tr = sobremuestrear(tr, a.sobremuestrear, a.veces)
    print(datos.resumen_particiones(entrenamiento=tr, validacion=va))
    print(entrenar(a.nombre, tr, va, a.arquitectura, a.epocas1, a.epocas2))


if __name__ == "__main__":
    main()
