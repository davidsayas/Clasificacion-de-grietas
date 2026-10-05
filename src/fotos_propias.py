"""fotos_propias.py — Desempeño con las fotos que TOMA EL EQUIPO.

El documento del reto lo pide en tres lugares:
  * Etapa 1: el equipo captura fotos propias de grietas reales "para probar el modelo al final"
  * Entrega 1: primera estimación de inclinación sobre una foto propia
  * Rúbrica (30 %): "métricas en el conjunto de prueba Y con fotos propias"

Es la prueba más honesta del proyecto: son fotos que ningún conjunto de datos contiene, tomadas
con celulares y paredes reales. Por eso el rendimiento acá suele ser MENOR que en la prueba del
dataset, y reportarlo bien (con sus errores) vale más que ocultarlo.

CARPETA ESPERADA
    data/fotos_propias/foto01.jpg ...
    data/fotos_propias/etiquetas.csv      archivo,etiqueta      (1 = con grieta, 0 = sin grieta)

Dos formas de medir, y por qué las dos:
  FORZADA        el sistema SIEMPRE responde (prob ≥ umbral = grieta). Es la medición clásica.
  CON ABSTENCIÓN el sistema puede decir INDETERMINADO. Se reporta la cobertura (qué fracción
                 respondió) y los errores SOLO entre las respondidas. Lo importante: cuántas
                 grietas REALES quedaron en INDETERMINADO, porque eso es una grieta sin diagnóstico.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config, datos, evaluar
from . import etiquetas as et


def _leer_etiquetas(etiquetas) -> dict:
    return et.normalizar(etiquetas)


def analizar_carpeta(sistema, carpeta, etiquetas=None, parametros: dict | None = None,
                     salida_csv=None, **kwargs) -> pd.DataFrame:
    """Analiza cada foto de `carpeta` con el sistema completo y devuelve una fila por foto.

    etiquetas:  dict {nombre_archivo: 0/1} o ruta a un CSV con columnas archivo,etiqueta
    parametros: dict {nombre_archivo: {roll_deg:..., mm_por_px:..., elemento:...}} para los
                datos que cambian de foto en foto (el giro del celular, la escala)
    kwargs:     parámetros comunes a todas las fotos (p. ej. escalas=(1.0, 0.5))
    """
    etiquetas, parametros = _leer_etiquetas(etiquetas), parametros or {}
    if etiquetas:
        et.revisar(carpeta, etiquetas)
    filas = []
    for ruta in datos.listar_imagenes(carpeta):
        foto = np.asarray(Image.open(ruta).convert("RGB"))
        r = sistema.analizar(foto, **{**kwargs, **parametros.get(ruta.name, {})})
        med, orient, ev = r["medicion"] or {}, r["orientacion"] or {}, r["riesgo"]
        filas.append({
            "archivo": ruta.name, "etiqueta": etiquetas.get(et.clave(ruta.name)),
            "prob_grieta": round(r["prob_grieta"], 4), "es_concreto": r["es_concreto"],
            "riesgo": ev.nivel, "puntaje": ev.puntaje,
            "ancho_mm": med.get("ancho_p95_mm"), "ancho_px": med.get("ancho_p95_px"),
            "orientacion": orient.get("categoria"), "inclinacion_deg": r["inclinacion_deg"],
            "modo": r["modo"], "notas": " | ".join(r["notas"])})
    tabla = pd.DataFrame(filas)
    if salida_csv:
        Path(salida_csv).parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(salida_csv, index=False)
    return tabla


def metricas_fotos_propias(tabla: pd.DataFrame, umbral: float = config.UMBRAL_DECISION) -> dict:
    """Métricas FORZADAS y CON ABSTENCIÓN sobre las fotos con etiqueta."""
    t = tabla.dropna(subset=["etiqueta"]).copy()
    if t.empty:
        raise ValueError("Ninguna foto tiene etiqueta: agregá data/fotos_propias/etiquetas.csv")
    y = t["etiqueta"].astype(int).to_numpy()
    prob = t["prob_grieta"].to_numpy()
    forzada = evaluar.metricas(y, prob, umbral)

    respondio = (t["riesgo"] != "INDETERMINADO").to_numpy()
    pred = prob >= umbral
    resp = {"n_respondidas": int(respondio.sum()),
            "cobertura": float(respondio.mean()),
            "exactitud_en_respondidas": float((pred[respondio] == y[respondio].astype(bool)).mean())
            if respondio.any() else float("nan"),
            "falsos_neg_en_respondidas": int(((~pred) & (y == 1) & respondio).sum()),
            "falsos_pos_en_respondidas": int((pred & (y == 0) & respondio).sum()),
            "grietas_reales_sin_diagnostico": int(((y == 1) & ~respondio).sum()),
            "sin_grieta_sin_diagnostico": int(((y == 0) & ~respondio).sum())}
    return {"n": int(len(t)), "con_grieta": int(y.sum()), "sin_grieta": int((y == 0).sum()),
            "forzada": {k: forzada[k] for k in ("exactitud", "precision", "recall", "f1", "tp", "tn", "fp", "fn")},
            "con_abstencion": resp}
