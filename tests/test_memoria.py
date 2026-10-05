from pathlib import Path

import cv2
import numpy as np

from src import entrenar
from sinteticos import textura_concreto


def test_predecir_rutas_nunca_carga_mas_de_un_trozo_y_da_el_mismo_resultado(tmp):
    rutas = []
    for i in range(23):
        r = Path(tmp) / f"{i}.jpg"
        cv2.imwrite(str(r), textura_concreto(64, semilla=i, brillo=100 + 5 * i))
        rutas.append(str(r))
    tamanos = []

    def predictor(lote):
        tamanos.append(len(lote))
        return lote.mean(axis=(1, 2, 3)) / 255.0

    en_trozos = entrenar.predecir_rutas(predictor, rutas, trozo=5)
    assert max(tamanos) <= 5 and sum(tamanos) == 23, tamanos       # NUNCA más de 5 imágenes juntas
    todo_junto = predictor(entrenar.cargar_imagenes(rutas))
    assert np.allclose(en_trozos, todo_junto), "partir en trozos NO debe cambiar el resultado"
    assert en_trozos.shape == (23,)


def test_predecir_rutas_con_embeddings_2d_y_sin_rutas(tmp):
    r = Path(tmp) / "a.jpg"
    cv2.imwrite(str(r), textura_concreto(64))
    e = entrenar.predecir_rutas(lambda lote: np.ones((len(lote), 7)), [str(r)] * 6, trozo=4)
    assert e.shape == (6, 7)
    assert len(entrenar.predecir_rutas(lambda l: l, [], trozo=4)) == 0


def test_el_peso_de_cargar_todo_junto_explica_el_error():
    """6.028 imágenes de validación: lo que el Paso 7 cargaba de golpe."""
    gb = 6028 * 224 * 224 * 3 * 4 / 1e9
    assert 3.5 < gb < 3.8, gb
