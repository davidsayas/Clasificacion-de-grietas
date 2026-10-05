"""gradcam.py — ¿EN QUÉ SE FIJA el modelo? (Grad-CAM)

Cuando el modelo dice "grieta" sobre una pared lisa con una mano, no basta saber
que se equivocó: hay que saber POR QUÉ. Grad-CAM pinta sobre la foto las zonas
que más empujaron la decisión hacia "grieta":

  * Si marca los bordes y sombras de los dedos -> el modelo confunde líneas
    oscuras con grietas: faltan ejemplos de ese tipo (datos).
  * Si marca el fondo liso o todo por igual   -> está usando el brillo o la
    textura global como atajo.
  * Si marca la grieta real                   -> está mirando lo correcto.

CÓMO FUNCIONA
  1. Se toma el mapa de rasgos de la última capa convolucional (7x7 para 224 px).
  2. Se calcula cuánto cambia la probabilidad de "grieta" si cada canal de ese
     mapa aumenta un poco (el gradiente).
  3. Se promedia el gradiente por canal (= importancia del canal) y se hace una
     suma ponderada de los canales. Solo se conserva lo positivo (ReLU).
  4. El mapa de 7x7 se agranda hasta 224x224 y se superpone a la foto.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config


def mapa_gradcam(modelo, imagen: np.ndarray, arquitectura: str = "mobilenetv2") -> np.ndarray:
    """Mapa 224x224 con valores 0-1 (1 = zona que más indica 'grieta').

    imagen: (224, 224, 3) float/uint8 con valores 0-255, en RGB.
    modelo: el modelo completo entrenado (con capas "base", "gap", "salida").
    """
    import tensorflow as tf
    from .modelo import preprocesar

    x = tf.convert_to_tensor(np.asarray(imagen, dtype=np.float32)[None])
    x = preprocesar(x, arquitectura)
    base = modelo.get_layer("base")
    cabeza = [modelo.get_layer(n) for n in ("gap", "salida")]   # dropout no actúa al predecir

    with tf.GradientTape() as cinta:
        conv = base(x, training=False)          # (1, 7, 7, 1280)
        cinta.watch(conv)
        y = conv
        for capa in cabeza:
            y = capa(y)
        puntaje = y[:, 0]                       # probabilidad de "con grieta"
    gradientes = cinta.gradient(puntaje, conv)
    pesos = tf.reduce_mean(gradientes, axis=(1, 2), keepdims=True)
    mapa = tf.nn.relu(tf.reduce_sum(pesos * conv, axis=-1))[0].numpy()
    mapa = mapa / (mapa.max() + 1e-8)
    return cv2.resize(mapa, (config.IMG, config.IMG), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def figura_gradcam(modelo, rutas, etiquetas=None, arquitectura: str = "mobilenetv2", ruta_salida=None):
    """Cuadrícula: cada foto, su mapa y la superposición, con la probabilidad."""
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(rutas)
    fig, ejes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i, ruta in enumerate(rutas):
        img = np.asarray(Image.open(ruta).convert("RGB").resize((config.IMG, config.IMG)), dtype=np.float32)
        mapa = mapa_gradcam(modelo, img, arquitectura)
        prob = float(modelo.predict(img[None], verbose=0)[0, 0])
        calor = cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        sobre = (0.55 * img + 0.45 * calor).astype(np.uint8)
        titulo = f"p(grieta) = {prob:.2f}" + (f" · real: {etiquetas[i]}" if etiquetas is not None else "")
        for eje, im, t in zip(ejes[i], (img.astype(np.uint8), calor, sobre), (titulo, "Grad-CAM", "superpuesto")):
            eje.imshow(im)
            eje.set_title(t, fontsize=9)
            eje.axis("off")
    fig.tight_layout()
    if ruta_salida:
        fig.savefig(ruta_salida, dpi=150)
    return fig
