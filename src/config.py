"""config.py — Todo número que usa más de un archivo vive acá, con su porqué.

Si cambiás algo (el tamaño de imagen, la semilla), se cambia en un solo lugar y
todo el proyecto queda consistente.
"""
import os
from pathlib import Path

# Semilla fija: hace que la división en train/val/test y el entrenamiento sean
# reproducibles. Con la misma semilla, dos corridas son comparables entre sí.
SEMILLA = 42

# Lado de la imagen que entra a la red. MobileNetV2 se preentrenó con 224x224.
IMG = 224

# Surface Crack trae recortes de 227x227. Se achican a 224 al cargar.
LADO_ORIGINAL = 227

# Probabilidad a partir de la cual se dice "con grieta". Se puede mover para
# priorizar no perder grietas (bajarlo) o evitar falsas alarmas (subirlo).
UMBRAL_DECISION = 0.5

# Banda de duda: entre estos dos valores el sistema dice "incierto" en vez de
# afirmar. Es parte de la corrección del falso positivo (ver README).
BANDA_INCIERTA = (0.20, 0.80)

# Fracciones de la división. La prueba NUNCA se toca hasta el final.
FRACCION_PRUEBA = 0.15
FRACCION_VALIDACION = 0.15

# Surface Crack son ~40.000 recortes salidos de 458 fotografías, o sea unos 44
# recortes por foto. HIPÓTESIS (se verifica con prueba_adyacencia): los archivos
# consecutivos salen de la misma foto. Si se confirma, partir por bloques de 44
# evita que recortes del mismo muro caigan en train y en test a la vez.
TAM_BLOQUE = 44

RAIZ_DATOS = Path(os.environ.get("GRIETAS_DATOS", "data"))
SALIDAS = Path(os.environ.get("GRIETAS_SALIDAS", "results"))
MODELOS = Path(os.environ.get("GRIETAS_MODELOS", "models"))
