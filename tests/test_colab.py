import io
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src import colab


def _jpg():
    return cv2.imencode(".jpg", np.full((16, 16, 3), 120, np.uint8))[1].tobytes()


def _zip(ruta, prefijo="", positive=True):
    with zipfile.ZipFile(ruta, "w") as z:
        for i in range(3):
            if positive:
                z.writestr(f"{prefijo}Positive/{i}.jpg", _jpg())
            z.writestr(f"{prefijo}Negative/{i}.jpg", _jpg())


def _drive(tmp):
    """Un Drive simulado con el proyecto VIEJO, el NUEVO y un zip señuelo."""
    raiz = Path(tmp) / "MyDrive"
    (raiz / "grietas-riesgo/src").mkdir(parents=True)
    (raiz / "grietas-riesgo/src/dataset.py").write_text("viejo")          # el proyecto viejo NO tiene datos.py
    (raiz / "mi_proyecto/src").mkdir(parents=True)
    (raiz / "mi_proyecto/src/datos.py").write_text("nuevo")
    return raiz


def test_encuentra_el_proyecto_nuevo_y_no_el_viejo(tmp):
    raiz = _drive(tmp)
    assert colab.buscar_proyecto(str(raiz)) == raiz / "mi_proyecto"


def test_sin_proyecto_nuevo_devuelve_none(tmp):
    raiz = Path(tmp) / "MyDrive"
    (raiz / "grietas-riesgo/src").mkdir(parents=True)
    (raiz / "grietas-riesgo/src/dataset.py").write_text("viejo")
    assert colab.buscar_proyecto(str(raiz)) is None


def test_zip_de_kaggle_con_cualquier_nombre_y_con_carpeta_extra(tmp):
    raiz = _drive(tmp)
    _zip(raiz / "archive.zip", prefijo="Surface Crack/")                 # el nombre típico de Kaggle
    _zip(raiz / "otro.zip", positive=False)                              # señuelo sin Positive
    tipo, ruta = colab.buscar_surface(str(raiz), raiz / "mi_proyecto")
    assert tipo == "zip" and ruta.name == "archive.zip"
    datos = colab.preparar(raiz / "mi_proyecto", str(raiz), local=str(Path(tmp) / "local"))
    assert Path(datos, "Positive/0.jpg").exists() and Path(datos).name == "Surface Crack"


def test_carpetas_sueltas_en_un_subdirectorio(tmp):
    raiz = _drive(tmp)
    for c in ("Positive", "Negative"):
        (raiz / "datasets/surface" / c).mkdir(parents=True)
        (raiz / "datasets/surface" / c / "a.jpg").write_bytes(_jpg())
    tipo, ruta = colab.buscar_surface(str(raiz), raiz / "mi_proyecto")
    assert tipo == "carpeta" and ruta == raiz / "datasets/surface"
    assert colab.preparar(raiz / "mi_proyecto", str(raiz), local=str(Path(tmp) / "l")) == str(ruta)


def test_si_hay_zip_y_carpeta_gana_el_zip(tmp):
    raiz = _drive(tmp)
    _zip(raiz / "archive.zip")
    for c in ("Positive", "Negative"):
        (raiz / "suelto" / c).mkdir(parents=True)
    assert colab.buscar_surface(str(raiz), raiz / "mi_proyecto")[0] == "zip"


def test_sin_surface_explica_que_hacer(tmp):
    raiz = _drive(tmp)
    try:
        colab.preparar(raiz / "mi_proyecto", str(raiz), local=str(Path(tmp) / "l"))
    except FileNotFoundError as e:
        assert "archive.zip" in str(e) and "Positive" in str(e)
        return
    raise AssertionError("debería explicar qué falta")


def test_el_zip_zip_corrupto_no_rompe_la_busqueda(tmp):
    raiz = _drive(tmp)
    (raiz / "roto.zip").write_bytes(b"esto no es un zip")
    _zip(raiz / "archive.zip")
    assert colab.buscar_surface(str(raiz), raiz / "mi_proyecto")[1].name == "archive.zip"
