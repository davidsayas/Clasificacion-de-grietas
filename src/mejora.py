"""mejora.py — Preparar el ajuste del modelo con FOTOS DE CELULAR (adaptación de dominio).

POR QUÉ ESTE CAMINO
-------------------
El análisis mostró que el modelo acierta el 100 % de las pocas fotos propias con las que entrenó y detecta 2 de 13 de las
nuevas: memoriza esas paredes en vez de aprender "grieta" en fotos de celular. Lo que le falta no es más épocas ni otra
arquitectura: son MUCHAS PAREDES DISTINTAS. Cambiar el modelo no arregla falta de variedad de datos.

QUÉ HACE ESTE ARCHIVO
---------------------
Arma el conjunto de ajuste con tres partes:
  * un ENSAYO de Surface Crack (para que el modelo no olvide lo que ya sabe)
  * las fotos de celular que el modelo ya usó (data/dificiles)
  * las fotos NUEVAS de celular (data/campo/con/<pared>/ y data/campo/sin/<pared>/)

y reparte las fotos nuevas POR PARED: algunas paredes enteras van a validación, para tener una señal honesta de si el
modelo generaliza a paredes que no vio MIENTRAS entrena. `fotos_propias` y `problemas` NO se tocan: siguen siendo la prueba.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, datos

MIN_PAREDES_POR_CLASE = 4          # menos que esto no alcanza ni para entrenar ni para validar
RECOMENDADAS_POR_CLASE = 10


def dividir_por_paredes(campo: pd.DataFrame, fraccion_val: float = 0.25, semilla: int = config.SEMILLA):
    """Reparte las fotos de campo en (entrenamiento, validación) por PARED entera.

    Garantiza al menos una pared de cada clase en validación, y que ninguna pared esté en las dos partes.
    """
    rng = np.random.default_rng(semilla)
    val_grupos = []
    for etiqueta in (0, 1):
        grupos = sorted(campo.loc[campo["etiqueta"] == etiqueta, "grupo"].unique())
        if len(grupos) < MIN_PAREDES_POR_CLASE:
            clase = "con grieta" if etiqueta == 1 else "sin grieta"
            raise ValueError(f"Hay solo {len(grupos)} pared(es) {clase} en data/campo. Hacen falta al menos "
                             f"{MIN_PAREDES_POR_CLASE} (lo recomendable: {RECOMENDADAS_POR_CLASE}), cada una en su subcarpeta.")
        rng.shuffle(grupos)
        n_val = max(1, int(round(fraccion_val * len(grupos))))
        val_grupos += grupos[:n_val]
    es_val = campo["grupo"].isin(val_grupos)
    tr, va = campo[~es_val].reset_index(drop=True), campo[es_val].reset_index(drop=True)
    datos.verificar_sin_fuga(tr, va)
    return tr, va


def preparar_mejora(proyecto, modelo_base: str = "v2_mobilenet", n_surface: int = 2000, veces_campo: int = 4,
                    veces_dificiles: int = 2, fraccion_val: float = 0.25, semilla: int = config.SEMILLA):
    """Devuelve (entrenamiento, validación, resumen) para ajustar el modelo base.

    n_surface:       imágenes de Surface Crack POR CLASE que se repasan (se toman del entrenamiento original, jamás de la prueba)
    veces_campo:     cuántas veces se repite cada foto nueva por época (son pocas y valen mucho)
    veces_dificiles: ídem para las fotos de celular que el modelo ya vio (menos, para no memorizarlas más)
    """
    proyecto = Path(proyecto)
    original = pd.read_csv(proyecto / "models" / modelo_base / "particion_entrenamiento.csv")
    surface = original[original["fuente"] == "surface_crack"]
    ensayo = pd.concat([surface[surface["etiqueta"] == e].sample(min(n_surface, int((surface["etiqueta"] == e).sum())),
                                                                  random_state=semilla) for e in (0, 1)])
    dificiles = original[original["fuente"] == "negativos_dificiles"].drop_duplicates("ruta")      # el entrenamiento los había repetido

    campo = datos.construir_inventario({"campo": {"con_grieta": proyecto / "data" / "campo" / "con",
                                                  "sin_grieta": proyecto / "data" / "campo" / "sin",
                                                  "grupos": "subcarpeta"}})
    campo_tr, campo_va = dividir_por_paredes(campo, fraccion_val, semilla)

    tr = pd.concat([ensayo] + [dificiles] * veces_dificiles + [campo_tr] * veces_campo, ignore_index=True)
    resumen = pd.DataFrame([
        {"parte": "ensayo Surface Crack", "fotos_unicas": len(ensayo), "paredes": "-", "veces": 1},
        {"parte": "fotos de celular ya vistas (dificiles)", "fotos_unicas": len(dificiles), "paredes": dificiles["grupo"].nunique(), "veces": veces_dificiles},
        {"parte": "fotos NUEVAS para entrenar (campo)", "fotos_unicas": len(campo_tr), "paredes": campo_tr["grupo"].nunique(), "veces": veces_campo},
        {"parte": "fotos NUEVAS para validar (paredes que NO entrenan)", "fotos_unicas": len(campo_va), "paredes": campo_va["grupo"].nunique(), "veces": 1},
    ])
    return tr, campo_va, resumen
