PREPARACION = '''
# ===================== CELDA DE ARRANQUE: no hay que editar nada =====================
# Monta tu Drive, encuentra sola la carpeta del proyecto y Surface Crack, y deja todo listo.
from google.colab import drive
drive.mount("/content/drive")

import glob, os, sys
_raiz = "/content/drive/MyDrive"
_cands = sorted(glob.glob(f"{_raiz}/*/src/datos.py") + glob.glob(f"{_raiz}/*/*/src/datos.py"),
                key=os.path.getmtime, reverse=True)
if not _cands:
    raise FileNotFoundError("No encontré la carpeta del proyecto NUEVO en tu Drive (busco una carpeta que tenga src/datos.py). "
                            "Subí la carpeta del zip nuevo a tu Drive y volvé a ejecutar esta celda.")
PROYECTO = os.path.dirname(os.path.dirname(_cands[0]))
os.chdir(PROYECTO)
sys.path.insert(0, PROYECTO)
print("Proyecto:", PROYECTO)
!pip -q install scikit-image scikit-learn pandas matplotlib

from src import colab
RUTA_DATOS = colab.preparar(PROYECTO, _raiz)       # Surface Crack (zip -> disco local, o carpeta en Drive)

import json, shutil, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from src import config, datos, eda, evaluar
'''
