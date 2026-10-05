"""eda.py — Análisis exploratorio. QUÉ HACE el EDA y POR QUÉ.

El EDA no es "ver gráficos bonitos". Responde una pregunta: ¿ESTOS DATOS ME
DEJAN APRENDER LO QUE QUIERO APRENDER? Cada función de acá contesta una parte:

  brillo_por_imagen         ¿las clases se distinguen por la luz? (atajo posible)
  clasificador_tonto_brillo ¿cuánto acertaría un "modelo" que solo mira el brillo?
  cohen_d                   ¿qué tan separadas están dos distribuciones?
  resumen_tamanos / color   ¿todas las imágenes son comparables entre sí?

Si un clasificador TONTO ya acierta mucho, el modelo puede estar usando ese
atajo y no la grieta. Si acierta poco (cerca de 50 %), el atajo es débil.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image


def brillo_por_imagen(rutas) -> np.ndarray:
    """Brillo promedio (0 a 255) de CADA imagen.

    Misma idea que tu función, pero en vez de un acumulador numérico que se
    resume en un solo número, usa una LISTA: una caja por imagen. Así se
    conserva cada valor y se puede hacer un histograma.
    """
    brillos = []                                   # nace UNA vez, antes del ciclo
    for ruta in rutas:
        dato = np.asarray(Image.open(ruta).convert("RGB"))
        brillos.append(float(dato.mean()))         # el brillo de ESTA imagen
    return np.array(brillos)                       # return afuera del for


def cohen_d(a, b) -> float:
    """Tamaño del efecto: diferencia de medias en unidades de desvío estándar.

    ~0.2 pequeño, ~0.5 mediano, ~0.8 grande. Con d=0.85 las dos montañas del
    histograma están claramente separadas, pero todavía se superponen mucho.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sd)


def exactitud_con_umbral(b_pos, b_neg, umbral: float) -> float:
    """Exactitud de decir "con grieta" si la imagen es más oscura que `umbral`."""
    b_pos, b_neg = np.asarray(b_pos), np.asarray(b_neg)
    aciertos = (b_pos < umbral).sum() + (b_neg >= umbral).sum()
    return float(aciertos / (len(b_pos) + len(b_neg)))


def clasificador_tonto_brillo(b_pos, b_neg) -> dict:
    """El MEJOR umbral posible si solo se mirara el brillo, y su exactitud.

    Es una COTA: un clasificador que solo mira el brillo no puede acertar más
    que esto. Si da ~65 %, el atajo existe pero es débil; si diera 95 %, el
    modelo seguramente lo estaría usando.
    """
    b_pos, b_neg = np.asarray(b_pos, float), np.asarray(b_neg, float)
    valores = np.concatenate([b_pos, b_neg])
    y = np.concatenate([np.ones(len(b_pos)), np.zeros(len(b_neg))])
    orden = np.argsort(valores, kind="stable")
    v, y = valores[orden], y[orden]
    total_neg = float((1 - y).sum())
    # Predecir "grieta" para los k más oscuros. Aciertan: los positivos entre
    # ellos y los negativos que quedaron afuera.
    pos_dentro = np.concatenate([[0.0], np.cumsum(y)])
    neg_dentro = np.concatenate([[0.0], np.cumsum(1 - y)])
    aciertos = pos_dentro + (total_neg - neg_dentro)
    mejor = int(np.argmax(aciertos))
    if mejor == 0:
        umbral = float(v[0] - 1)
    elif mejor == len(v):
        umbral = float(v[-1] + 1)
    else:
        umbral = float((v[mejor - 1] + v[mejor]) / 2)
    return {"umbral": umbral, "exactitud": float(aciertos[mejor] / len(v)),
            "azar": float(max(len(b_pos), len(b_neg)) / len(v))}


def resumen_tamanos(rutas, n: int = 500) -> Counter:
    """Cuenta (ancho, alto, modo) en una muestra. Todas deberían coincidir:
    si hay tamaños mezclados, el redimensionado afecta distinto a cada fuente."""
    cuenta: Counter = Counter()
    for r in list(rutas)[:n]:
        with Image.open(r) as im:
            cuenta[(im.size[0], im.size[1], im.mode)] += 1
    return cuenta


def resumen_color(rutas, n: int = 500) -> dict:
    """Media de cada canal RGB. En concreto gris, R≈G≈B; si una fuente tiene un
    tinte distinto (foto con filtro cálido, pared pintada), se nota acá."""
    medias = []
    for r in list(rutas)[:n]:
        a = np.asarray(Image.open(r).convert("RGB"), dtype=np.float32)
        medias.append(a.mean(axis=(0, 1)))
    m = np.mean(medias, axis=0)
    return {"R": float(m[0]), "G": float(m[1]), "B": float(m[2])}


def figura_brillo(b_pos, b_neg, ruta_salida=None, etiquetas=("Con grieta", "Sin grieta")):
    """Histograma superpuesto de los brillos de las dos clases."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(b_pos, bins=50, alpha=0.55, label=f"{etiquetas[0]} (media {np.mean(b_pos):.0f})")
    ax.hist(b_neg, bins=50, alpha=0.55, label=f"{etiquetas[1]} (media {np.mean(b_neg):.0f})")
    ax.set_xlabel("Brillo promedio de la imagen (0-255)")
    ax.set_ylabel("Cantidad de imágenes")
    ax.legend()
    fig.tight_layout()
    if ruta_salida:
        fig.savefig(ruta_salida, dpi=150)
    return fig
