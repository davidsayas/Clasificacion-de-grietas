"""Corre las funciones de TensorFlow con un TensorFlow SIMULADO (MagicMock).

NO valida que Keras funcione: valida que NUESTRO código no tenga errores de Python
(nombres mal escritos, argumentos que no existen en NUESTRAS funciones, rutas, etc.).
La validación real ocurre en Colab. Uso:  python tests/humo_tf_simulado.py
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# --- TensorFlow/Keras falsos ---
class _BN:                                        # para isinstance(capa, BatchNormalization)
    def __call__(self, *a, **k):
        return MagicMock()

tf = MagicMock(name="tf")
keras = MagicMock(name="keras")
keras.layers.BatchNormalization = _BN
keras.__version__ = "0.0-simulado"
tf.__version__ = "0.0-simulado"
tf.keras = keras
capas_base = [MagicMock(name=f"capa{i}") for i in range(10)] + [_BN()]
keras.applications.MobileNetV2.return_value.layers = capas_base
keras.applications.EfficientNetB0.return_value.layers = capas_base
keras.Model.return_value.fit.return_value.history = {"val_loss": [0.5, 0.4], "val_recall": [0.9, 0.95]}
sys.modules["tensorflow"] = tf
sys.modules["tensorflow.keras"] = keras

from src import config, datos, entrenar, gradcam, modelo, unet   # noqa: E402

# --- modelo.py ---
for arq in modelo.ARQUITECTURAS:
    m = modelo.construir_modelo(arq)
    modelo.activar_ajuste_fino(m, capas=3)
    modelo.modelo_inferencia(m, arq)
    modelo.modelo_embeddings(m, arq)
try:
    modelo.construir_modelo("inexistente")
except ValueError as e:
    print("arquitectura inválida -> ValueError correcto:", str(e)[:40])

# --- entrenar.py ---
df = pd.DataFrame({"ruta": [f"/x/{i}.jpg" for i in range(40)], "etiqueta": [0, 1] * 20,
                   "fuente": ["a"] * 30 + ["dificiles"] * 10, "grupo": [f"g{i}" for i in range(40)]})
assert len(entrenar.sobremuestrear(df, "dificiles", 5)) == 40 + 4 * 10
assert set(entrenar.pesos_de_clase(df)) == {0, 1}
entrenar.crear_dataset(df, 8, True)
with __import__("tempfile").TemporaryDirectory() as d:
    meta = entrenar.entrenar("humo", df.iloc[:30], df.iloc[30:], epocas1=1, epocas2=1, salida=Path(d))
    assert (Path(d) / "humo" / "meta.json").exists() and (Path(d) / "humo" / "particion_validacion.csv").exists()
    print("meta.json escrito con claves:", sorted(meta)[:5], "...")
entrenar.barrido_lr(df, df, lrs=(1e-3,), epocas=1)

# --- unet.py / gradcam.py (solo construcción y llamadas) ---
keras.Model.return_value.return_value = [MagicMock() for _ in range(5)]   # el encoder devuelve 5 saltos
unet.construir_unet()
assert len(unet.SALTOS) == 5
# --- variantes TFLite y predictor TFLite ---
tf.lite.TFLiteConverter.from_keras_model.return_value.convert.return_value = b"tflite"
with __import__("tempfile").TemporaryDirectory() as d:
    v = modelo.exportar_variantes(m, d)
    assert set(v) == {"float32", "dinamico", "float16"} and all(Path(r).exists() for r, _ in v.values())
from src import complejidad
assert len(complejidad.tflite_predictor("x.tflite")(np.zeros((2, 224, 224, 3)))) == 2
with __import__("tempfile").TemporaryDirectory() as d:
    (Path(d) / "base").mkdir()
    meta = entrenar.ajustar_dominio("base", "nuevo", df.iloc[:30], df.iloc[30:], epocas=1, salida=Path(d))
    assert (Path(d) / "nuevo" / "meta.json").exists() and meta["base"] == "base"
print("humo OK: no hay errores de Python en modelo/entrenar/unet/complejidad/mejora")
