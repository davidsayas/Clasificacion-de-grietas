"""linea_base.py — La tabla comparativa: ¿cuánto aporta la red neuronal?

Una red de 99 % no significa nada sin saber cuánto lograría un método SIMPLE.
Esta "línea base" usa tres métodos que no son redes neuronales:

  1. Umbral de brillo        "más oscura = grieta"
  2. Umbral de bordes        "más bordes (Canny) = grieta"
  3. Regresión logística     sobre 5 rasgos simples (brillo, contraste, bordes...)

Se ajustan SOLO con entrenamiento y se miden en prueba, con la misma división por
grupos que usará la red. La tabla final enfrenta: línea base vs. MobileNetV2 vs.
el segundo modelo.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import evaluar

NOMBRES_RASGOS = ["brillo", "contraste", "fraccion_bordes", "percentil5", "fraccion_oscura"]


def rasgos_linea_base(ruta, lado: int = 112) -> np.ndarray:
    """5 rasgos de una imagen, calculados en gris y a tamaño `lado`."""
    gris = np.asarray(Image.open(ruta).convert("L").resize((lado, lado)), dtype=np.uint8)
    g = gris.astype(np.float32)
    bordes = cv2.Canny(cv2.GaussianBlur(gris, (3, 3), 0), 40, 120)
    return np.array([g.mean(), g.std(), (bordes > 0).mean(), np.percentile(g, 5),
                     (g < np.median(g) - 25).mean()], dtype=np.float32)


def extraer(df: pd.DataFrame) -> np.ndarray:
    return np.stack([rasgos_linea_base(r) for r in df["ruta"]])


def mejor_umbral(valores: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Umbral y dirección (+1: 'grieta si valor > umbral'; -1: 'si valor < umbral')
    con la mayor exactitud en entrenamiento."""
    mejor = (0.0, 0.0, 1)
    cand = np.unique(np.quantile(valores, np.linspace(0.01, 0.99, 197)))
    for u in cand:
        for d in (1, -1):
            pred = (valores > u) if d == 1 else (valores < u)
            acc = float((pred == y.astype(bool)).mean())
            if acc > mejor[1]:
                mejor = (float(u), acc, d)
    return mejor[0], mejor[2]


def tabla_linea_base(entrenamiento: pd.DataFrame, prueba: pd.DataFrame,
                     x_ent: np.ndarray | None = None, x_prueba: np.ndarray | None = None) -> pd.DataFrame:
    """Métricas en PRUEBA de los tres métodos de línea base (ajustados en entrenamiento)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_ent = extraer(entrenamiento) if x_ent is None else x_ent
    x_prueba = extraer(prueba) if x_prueba is None else x_prueba
    y_ent, y_pru = entrenamiento["etiqueta"].to_numpy(), prueba["etiqueta"].to_numpy()
    filas = []

    def anotar(nombre, prob):
        m = evaluar.metricas(y_pru, prob)
        filas.append({"metodo": nombre, "exactitud": round(m["exactitud"], 4),
                      "recall": round(m["recall"], 4), "precision": round(m["precision"], 4), "f1": round(m["f1"], 4),
                      "falsos_neg": m["fn"], "falsos_pos": m["fp"]})

    for indice, nombre in ((0, "umbral de brillo"), (2, "umbral de bordes")):
        u, d = mejor_umbral(x_ent[:, indice], y_ent)
        anotar(nombre, ((x_prueba[:, indice] > u) if d == 1 else (x_prueba[:, indice] < u)).astype(float))

    esc = StandardScaler().fit(x_ent)
    lr = LogisticRegression(max_iter=1000).fit(esc.transform(x_ent), y_ent)
    anotar("regresión logística (5 rasgos)", lr.predict_proba(esc.transform(x_prueba))[:, 1])
    return pd.DataFrame(filas)
