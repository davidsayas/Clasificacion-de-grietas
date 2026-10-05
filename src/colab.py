"""colab.py — Encuentra solo el proyecto y Surface Crack en tu Drive, para no escribir rutas a mano.

Lo usa la primera celda de cada cuaderno. Busca:
  * el PROYECTO: una carpeta (a lo sumo 2 niveles adentro de tu Drive) que tenga src/datos.py
  * SURFACE CRACK, en este orden:
      1) un .zip que adentro tenga una carpeta Positive/ (el de Kaggle suele llamarse archive.zip)
      2) una carpeta con Positive/ y Negative/ sueltas (hasta 4 niveles)

Si es un zip, se descomprime al disco LOCAL de Colab (leer 40.000 imágenes sueltas desde Drive es
muchísimo más lento). Si son carpetas, se usan donde están (sirve para explorar; para entrenar
conviene un zip).
"""
from __future__ import annotations

import glob
import os
import zipfile
from pathlib import Path

RAIZ_DRIVE = "/content/drive/MyDrive"
LOCAL = "/content/datos"


def buscar_proyecto(raiz: str = RAIZ_DRIVE) -> Path | None:
    """La carpeta del proyecto NUEVO (la que tiene src/datos.py). Si hay varias, la más reciente."""
    cands = glob.glob(f"{raiz}/*/src/datos.py") + glob.glob(f"{raiz}/*/*/src/datos.py")
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return Path(cands[0]).parent.parent


def _zip_tiene_surface(ruta) -> bool:
    try:
        with zipfile.ZipFile(ruta) as z:
            return any("Positive/" in n for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def _carpeta_surface(raiz: str, profundidad: int = 4) -> Path | None:
    """Carpeta que contiene Positive/ y Negative/ (sin entrar a ellas, para no listar 40.000 archivos)."""
    base = Path(raiz)
    for actual, carpetas, _ in os.walk(raiz):
        nivel = len(Path(actual).relative_to(base).parts)
        if "Positive" in carpetas and "Negative" in carpetas:
            return Path(actual)
        carpetas[:] = [c for c in carpetas if not c.startswith(".") and c not in ("Positive", "Negative")]
        if nivel >= profundidad:
            carpetas[:] = []
    return None


def buscar_surface(raiz: str = RAIZ_DRIVE, proyecto: Path | None = None):
    """('zip', ruta) | ('carpeta', ruta) | None."""
    zips = glob.glob(f"{raiz}/*.zip")
    if proyecto is not None:
        zips += glob.glob(f"{proyecto}/*.zip") + glob.glob(f"{proyecto}/data/*.zip")
    for z in sorted(zips, key=os.path.getmtime, reverse=True):
        if _zip_tiene_surface(z):
            return "zip", Path(z)
    carpeta = _carpeta_surface(raiz)
    return ("carpeta", carpeta) if carpeta else None


def preparar(proyecto, raiz: str = RAIZ_DRIVE, local: str = LOCAL) -> str:
    """Deja Surface Crack listo y devuelve la ruta que contiene Positive/ y Negative/."""
    encontrado = buscar_surface(raiz, Path(proyecto))
    if encontrado is None:
        raise FileNotFoundError(
            "No encontré Surface Crack en tu Drive. Subí a la raíz de tu Drive el .zip que bajaste de Kaggle "
            "(suele llamarse archive.zip y tiene adentro las carpetas Positive y Negative) y volvé a ejecutar esta celda.")
    tipo, ruta = encontrado
    if tipo == "zip":
        destino = Path(local)
        if not list(destino.glob("**/Positive")):
            print(f"Descomprimiendo {ruta.name} al disco local (1-3 minutos)...")
            destino.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(ruta).extractall(destino)
        datos = list(destino.glob("**/Positive"))[0].parent
    else:
        datos = ruta
        print("Surface Crack se usa DESDE Drive (sirve para explorar). Para entrenar, un .zip es mucho más rápido.")
    n_pos, n_neg = len(list((datos / "Positive").glob("*.jp*g"))), len(list((datos / "Negative").glob("*.jp*g")))
    print(f"Surface Crack: {datos}\n  Positive: {n_pos} imágenes\n  Negative: {n_neg} imágenes")
    return str(datos)
