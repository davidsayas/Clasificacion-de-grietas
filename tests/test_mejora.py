from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import mejora
from sinteticos import textura_concreto


def _proyecto(tmp, paredes_con=6, paredes_sin=6, fotos=3):
    raiz = Path(tmp)
    (raiz / "models/v2_mobilenet").mkdir(parents=True)
    filas = []
    for i in range(400):
        filas.append({"ruta": f"/sc/{i}.jpg", "etiqueta": i % 2, "fuente": "surface_crack", "grupo": f"b{i // 40}", "orden": i})
    for i in range(10):                                   # 5 fotos únicas de celular, repetidas x2 como hizo el entrenamiento
        for rep in range(2):
            filas.append({"ruta": f"/dif/{i % 5}.jpg", "etiqueta": (i % 5) % 2, "fuente": "negativos_dificiles",
                          "grupo": "dif|escena", "orden": i})
    pd.DataFrame(filas).to_csv(raiz / "models/v2_mobilenet/particion_entrenamiento.csv", index=False)
    for clase, n in (("con", paredes_con), ("sin", paredes_sin)):
        for p in range(n):
            carpeta = raiz / "data/campo" / clase / f"pared_{p}"
            carpeta.mkdir(parents=True)
            for k in range(fotos):
                cv2.imwrite(str(carpeta / f"{k}.jpg"), textura_concreto(32, semilla=p * 10 + k))
    return raiz


def test_reparte_por_pared_y_nunca_mezcla(tmp):
    from src import datos
    raiz = _proyecto(tmp, 8, 8)
    campo = datos.construir_inventario({"campo": {"con_grieta": raiz / "data/campo/con", "sin_grieta": raiz / "data/campo/sin",
                                                  "grupos": "subcarpeta"}})
    tr, va = mejora.dividir_por_paredes(campo, 0.25)
    assert not (set(tr["grupo"]) & set(va["grupo"])), "una pared no puede estar en las dos partes"
    assert set(va["etiqueta"]) == {0, 1}, "validación debe tener paredes de AMBAS clases"
    assert va["grupo"].nunique() == 4 and tr["grupo"].nunique() == 12      # 25 % de 8 = 2 por clase


def test_pocas_paredes_da_un_mensaje_claro(tmp):
    raiz = _proyecto(tmp, paredes_con=2, paredes_sin=6)
    try:
        mejora.preparar_mejora(raiz, n_surface=50)
    except ValueError as e:
        assert "2 pared" in str(e) and "con grieta" in str(e) and "al menos 4" in str(e)
        return
    raise AssertionError("debería exigir más paredes")


def test_preparar_mejora_arma_el_conjunto_y_no_toca_la_prueba(tmp):
    raiz = _proyecto(tmp, 8, 8)
    tr, va, resumen = mejora.preparar_mejora(raiz, n_surface=100, veces_campo=4, veces_dificiles=2)
    unicas = tr.drop_duplicates("ruta")
    assert len(unicas[unicas["fuente"] == "surface_crack"]) == 200                 # 100 por clase
    assert len(tr[tr["fuente"] == "negativos_dificiles"].drop_duplicates("ruta")) == 5
    assert (tr["fuente"] == "negativos_dificiles").sum() == 5 * 2                  # 5 únicas x 2 veces (no x2 del original)
    campo_tr = tr[tr["fuente"] == "campo"]
    assert len(campo_tr) == len(campo_tr.drop_duplicates("ruta")) * 4              # x4
    assert not (set(tr["ruta"]) & set(va["ruta"])), "validación no puede aparecer en entrenamiento"
    assert set(va["fuente"]) == {"campo"} and len(resumen) == 4
