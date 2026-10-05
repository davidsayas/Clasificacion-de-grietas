"""unet.py — Segmentación: dibujar la grieta PÍXEL A PÍXEL (modelo extra de la Entrega 3).

LA CADENA COMPLETA
------------------
    MobileNetV2  ->  "¿hay grieta en esta foto?"            (clasificación)
    U-Net        ->  "¿qué píxeles son la grieta?"          (segmentación)
    ancho.py     ->  "¿cuántos mm de ancho tiene?"          (medición propia)

POR QUÉ UNA U-NET
  El clasificador solo responde sí/no. Para medir el ancho hace falta saber exactamente
  qué píxeles son grieta. Una U-Net tiene dos mitades:
    - ENCODER: va reduciendo la imagen y extrayendo rasgos (acá, MobileNetV2 preentrenada)
    - DECODER: va agrandando de nuevo hasta el tamaño original, y en cada paso recibe un
      "atajo" (skip connection) del encoder para recuperar el detalle fino.

DATOS QUE NECESITA (esto NO viene en Surface Crack)
  Pares imagen + máscara (blanco = grieta) en:
      datos_seg/imagenes/  y  datos_seg/mascaras/   (mismo nombre de archivo)
  Conjuntos públicos de segmentación de grietas a buscar y verificar licencia/disponibilidad:
  DeepCrack, CrackForest, "Concrete Crack Segmentation". Sin ellos, ancho.segmentar_clasico
  sirve como alternativa sin red neuronal.

⚠ Este archivo no se pudo ejecutar donde se escribió (sin TensorFlow). Se revisó
  su sintaxis y las dimensiones a mano; la primera corrida en Colab puede pedir ajustes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

SALTOS = ("block_1_expand_relu",    # 112x112
          "block_3_expand_relu",    # 56x56
          "block_6_expand_relu",    # 28x28
          "block_13_expand_relu",   # 14x14
          "out_relu")               # 7x7


def construir_unet(congelar_encoder: bool = True, lr: float = 1e-3):
    from tensorflow import keras
    L = keras.layers

    base = keras.applications.MobileNetV2(input_shape=(config.IMG, config.IMG, 3),
                                          include_top=False, weights="imagenet")
    encoder = keras.Model(base.input, [base.get_layer(n).output for n in SALTOS], name="encoder")
    encoder.trainable = not congelar_encoder

    entrada = keras.Input((config.IMG, config.IMG, 3), name="imagen")
    x = keras.applications.mobilenet_v2.preprocess_input(entrada)
    s1, s2, s3, s4, s5 = encoder(x, training=False)

    def subir(x, salto, filtros):
        x = L.Conv2DTranspose(filtros, 3, strides=2, padding="same")(x)   # duplica el tamaño
        x = L.Concatenate()([x, salto])                                     # recupera detalle
        for _ in range(2):
            x = L.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)
        return x

    x = subir(s5, s4, 256)    # 7  -> 14
    x = subir(x, s3, 128)     # 14 -> 28
    x = subir(x, s2, 64)      # 28 -> 56
    x = subir(x, s1, 32)      # 56 -> 112
    x = L.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)   # 112 -> 224
    salida = L.Conv2D(1, 1, activation="sigmoid", name="mascara")(x)
    modelo = keras.Model(entrada, salida, name="unet_mobilenetv2")
    compilar_unet(modelo, lr)
    return modelo


def _dice(y, p, eps=1.0):
    import tensorflow as tf
    y, p = tf.reshape(y, (tf.shape(y)[0], -1)), tf.reshape(p, (tf.shape(p)[0], -1))
    inter = tf.reduce_sum(y * p, axis=1)
    return (2 * inter + eps) / (tf.reduce_sum(y, axis=1) + tf.reduce_sum(p, axis=1) + eps)


def perdida_bce_dice(y, p):
    """BCE + (1 - Dice). Las grietas ocupan pocos píxeles: solo BCE aprendería a
    decir 'nada' en todos lados; Dice penaliza no encontrarlas."""
    from tensorflow import keras
    return keras.losses.binary_crossentropy(y, p) + (1.0 - _dice(y, p))[:, None, None]


def dice_metrica(y, p):
    import tensorflow as tf
    return tf.reduce_mean(_dice(y, tf.cast(p > 0.5, tf.float32)))


def compilar_unet(modelo, lr: float) -> None:
    from tensorflow import keras
    modelo.compile(optimizer=keras.optimizers.Adam(lr), loss=perdida_bce_dice,
                   metrics=[dice_metrica, keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou")])


def dataset_segmentacion(carpeta_imagenes, carpeta_mascaras, lote: int = 8, entrenar: bool = False):
    """Pares (imagen, máscara) emparejados por nombre de archivo (sin extensión)."""
    import tensorflow as tf
    from .datos import listar_imagenes

    mascaras = {p.stem: p for p in listar_imagenes(carpeta_mascaras)}
    pares = [(str(p), str(mascaras[p.stem])) for p in listar_imagenes(carpeta_imagenes) if p.stem in mascaras]
    if not pares:
        raise FileNotFoundError("No hay pares imagen/máscara con el mismo nombre.")
    ds = tf.data.Dataset.from_tensor_slices(([a for a, _ in pares], [b for _, b in pares]))
    if entrenar:
        ds = ds.shuffle(len(pares), seed=config.SEMILLA)

    def cargar(ri, rm):
        img = tf.image.resize(tf.io.decode_image(tf.io.read_file(ri), channels=3, expand_animations=False),
                              (config.IMG, config.IMG))
        m = tf.image.resize(tf.io.decode_image(tf.io.read_file(rm), channels=1, expand_animations=False),
                            (config.IMG, config.IMG), method="nearest")
        m = tf.cast(m > 127, tf.float32)
        par = tf.concat([img, m], axis=-1)                          # se transforman JUNTAS
        if entrenar:
            par = tf.image.random_flip_left_right(par)
            par = tf.image.random_flip_up_down(par)
        par = tf.ensure_shape(par, (config.IMG, config.IMG, 4))
        return par[..., :3], par[..., 3:]

    return ds.map(cargar, num_parallel_calls=tf.data.AUTOTUNE).batch(lote).prefetch(tf.data.AUTOTUNE), len(pares)


def segmentar_con_unet(modelo, imagen_rgb: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    """Máscara 0/1 al tamaño de la imagen original, a partir de una U-Net entrenada."""
    import cv2
    h, w = imagen_rgb.shape[:2]
    entrada = cv2.resize(imagen_rgb, (config.IMG, config.IMG))[None].astype(np.float32)
    prob = modelo.predict(entrada, verbose=0)[0, ..., 0]
    return (cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= umbral).astype(np.uint8)
