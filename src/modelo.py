"""modelo.py — Arquitectura, aumentación, congelado/descongelado y exportación.

QUÉ ES TRANSFER LEARNING (lo que hace este archivo)
---------------------------------------------------
Una red entrenada con millones de fotos (ImageNet) ya sabe ver bordes, texturas y
formas. En vez de entrenar desde cero, se reutiliza esa "base" y se le pone una
cabeza nueva que decide "grieta / no grieta":

    imagen -> aumentación -> preprocesado -> BASE (congelada) -> pooling -> dropout -> sigmoide

FASE 1: la base está CONGELADA (sus pesos no cambian); solo aprende la cabeza.
FASE 2 (ajuste fino): se DESCONGELAN las últimas capas de la base con una tasa de
aprendizaje muy baja, para adaptarlas a las grietas sin destruir lo aprendido.

DETALLES QUE IMPORTAN (y que ya dieron problemas)
-------------------------------------------------
* La aumentación va ANTES del preprocesado: el brillo se varía sobre valores
  0-255. Si fuera después (valores de -1 a 1), el brillo quedaría roto.
* La aumentación solo está activa al entrenar; al predecir no hace nada.
* La base se llama SIEMPRE con training=False, incluso en la fase 2: así las
  capas BatchNorm conservan las estadísticas de ImageNet y no se desestabilizan
  con lotes chicos.
* Cada capa tiene NOMBRE ("base", "gap", "salida"...) porque Grad-CAM y la puerta
  de concreto las buscan por nombre.
"""
from __future__ import annotations

from . import config

ARQUITECTURAS = ("mobilenetv2", "efficientnetb0")


def _keras():
    from tensorflow import keras
    return keras


def aumentacion(brillo: float = 0.2, contraste: float = 0.2, rotacion: float = 0.15,
                zoom: float = 0.1, ruido: float = 0.0):
    """Transformaciones aleatorias SOLO en entrenamiento.

    brillo=0.2 sobre valores 0-255 suma un número al azar entre -51 y +51. La
    diferencia real de brillo entre clases en Surface Crack es de ~18 niveles, así
    que el brillo global deja de servir como pista (ver prueba_atajo_brillo).
    rotacion=0.15 es una fracción de vuelta completa: ±54°.
    """
    keras = _keras()
    L = keras.layers
    capas = [L.RandomFlip("horizontal_and_vertical"), L.RandomRotation(rotacion),
             L.RandomZoom(zoom), L.RandomBrightness(brillo, value_range=(0, 255)),
             L.RandomContrast(contraste)]
    if ruido:
        capas.append(L.GaussianNoise(ruido))
    return keras.Sequential(capas, name="aumentacion")


def preprocesar(x, arquitectura: str):
    """Preprocesado propio de cada arquitectura (recibe valores 0-255)."""
    keras = _keras()
    if arquitectura == "mobilenetv2":
        return keras.applications.mobilenet_v2.preprocess_input(x)       # a [-1, 1]
    if arquitectura == "efficientnetb0":
        return keras.applications.efficientnet.preprocess_input(x)       # no-op: la red ya reescala
    raise ValueError(f"Arquitectura desconocida: {arquitectura}. Opciones: {ARQUITECTURAS}")


def _base(arquitectura: str):
    keras = _keras()
    forma = (config.IMG, config.IMG, 3)
    if arquitectura == "mobilenetv2":
        return keras.applications.MobileNetV2(input_shape=forma, include_top=False,
                                              weights="imagenet", name="base")
    if arquitectura == "efficientnetb0":
        return keras.applications.EfficientNetB0(input_shape=forma, include_top=False,
                                                 weights="imagenet", name="base")
    raise ValueError(f"Arquitectura desconocida: {arquitectura}. Opciones: {ARQUITECTURAS}")


def compilar(modelo, lr: float) -> None:
    keras = _keras()
    modelo.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="exactitud"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")])


def construir_modelo(arquitectura: str = "mobilenetv2", lr: float = 1e-3, brillo: float = 0.2,
                     ruido: float = 0.0, dropout: float = 0.3):
    """Modelo completo para la FASE 1 (base congelada)."""
    keras = _keras()
    L = keras.layers
    base = _base(arquitectura)
    base.trainable = False

    entrada = keras.Input((config.IMG, config.IMG, 3), name="imagen")
    x = aumentacion(brillo=brillo, ruido=ruido)(entrada)       # 1) aumentación (0-255)
    x = preprocesar(x, arquitectura)                           # 2) preprocesado
    x = base(x, training=False)                                # 3) base en modo inferencia
    x = L.GlobalAveragePooling2D(name="gap")(x)                # 4) un vector de 1280 números
    x = L.Dropout(dropout, name="dropout")(x)
    salida = L.Dense(1, activation="sigmoid", name="salida")(x)  # P(con grieta)
    modelo = keras.Model(entrada, salida, name=f"{arquitectura}_grietas")
    compilar(modelo, lr)
    return modelo


def activar_ajuste_fino(modelo, capas: int = 30, lr: float = 1e-5) -> None:
    """FASE 2: descongela las últimas `capas` de la base (menos las BatchNorm)."""
    keras = _keras()
    base = modelo.get_layer("base")
    base.trainable = True
    for capa in base.layers[:-capas]:
        capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, keras.layers.BatchNormalization):
            capa.trainable = False
    compilar(modelo, lr)            # hay que recompilar para que el cambio surta efecto


def modelo_inferencia(modelo, arquitectura: str = "mobilenetv2"):
    """Mismo modelo SIN aumentación ni dropout: para predecir, exportar y Grad-CAM."""
    keras = _keras()
    entrada = keras.Input((config.IMG, config.IMG, 3), name="imagen")
    x = preprocesar(entrada, arquitectura)
    x = modelo.get_layer("base")(x, training=False)
    x = modelo.get_layer("gap")(x)
    salida = modelo.get_layer("salida")(x)
    return keras.Model(entrada, salida, name="inferencia")


def modelo_embeddings(modelo, arquitectura: str = "mobilenetv2"):
    """Devuelve el vector de 1280 números previo a la decisión (para la puerta de concreto)."""
    keras = _keras()
    entrada = keras.Input((config.IMG, config.IMG, 3), name="imagen")
    x = preprocesar(entrada, arquitectura)
    x = modelo.get_layer("base")(x, training=False)
    x = modelo.get_layer("gap")(x)
    return keras.Model(entrada, x, name="embeddings")


def exportar_tflite(modelo, ruta, arquitectura: str = "mobilenetv2", float16: bool = True) -> int:
    """Exporta a TFLite para el celular. float16 reduce el tamaño a la mitad.
    Devuelve el tamaño en bytes."""
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo_inferencia(modelo, arquitectura))
    if float16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    datos = conv.convert()
    with open(ruta, "wb") as f:
        f.write(datos)
    return len(datos)


def exportar_variantes(modelo, carpeta, arquitectura: str = "mobilenetv2") -> dict:
    """Exporta TRES versiones TFLite para comparar tamaño, velocidad y fidelidad:

      float32   sin comprimir (referencia)
      dinamico  pesos en int8, cálculo en float (cuantización dinámica): ~4x más chico
      float16   pesos en float16: ~2x más chico, casi sin pérdida

    Devuelve {nombre: (ruta, bytes)}. Después hay que comprobar con
    complejidad.fidelidad_tflite que la cuantización no cambió las predicciones.
    """
    import tensorflow as tf
    from pathlib import Path

    inf = modelo_inferencia(modelo, arquitectura)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    resultado = {}
    for nombre in ("float32", "dinamico", "float16"):
        conv = tf.lite.TFLiteConverter.from_keras_model(inf)
        if nombre in ("dinamico", "float16"):
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if nombre == "float16":
            conv.target_spec.supported_types = [tf.float16]
        datos = conv.convert()
        ruta = carpeta / f"modelo_{nombre}.tflite"
        ruta.write_bytes(datos)
        resultado[nombre] = (ruta, len(datos))
    return resultado
