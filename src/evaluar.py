"""evaluar.py — Métricas y PRUEBAS DE ROBUSTEZ.

Las curvas de entrenamiento NO detectan problemas de dominio: la validación sale
de las mismas fotos que el entrenamiento. Lo único que lo detecta son pruebas
que cambian UN factor a la vez y miden cuánto se mueve la respuesta del modelo:

  prueba_atajo_brillo   oscurecer/aclarar y contar cambios de clase
  prueba_robustez       giro, lejanía, perspectiva, desenfoque, ruido

Todas reciben una función `predecir(lote) -> probabilidades`, donde `lote` es un
arreglo (N, 224, 224, 3) con valores 0-255. Así se pueden probar con modelos
falsos (en tests/) y con el modelo real (en el cuaderno).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metricas(y_true, prob, umbral: float = config.UMBRAL_DECISION) -> dict:
    """Matriz de confusión y métricas derivadas.

    En este problema el error CARO es el falso negativo (grieta no vista), así
    que `recall` (sensibilidad) es la métrica a vigilar; `precision` dice cuántas
    alarmas son reales; `especificidad` es el complemento de las falsas alarmas.
    """
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "exactitud": div(tp + tn, len(y)), "precision": p, "recall": r,
            "especificidad": div(tn, tn + fp),
            "f1": div(2 * p * r, p + r) if not (np.isnan(p) or np.isnan(r)) else float("nan"),
            "matriz": [[tn, fp], [fn, tp]]}


def intervalo_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza (95 %) para una proporción k/n.

    "2 errores de 3.000" no es "0,07 % exacto": con pocos errores la incertidumbre
    es grande. Wilson es la forma honesta de reportarlo.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(centro - mitad), float(centro + mitad))


def umbral_para_recall(y_true, prob, recall_objetivo: float = 0.99) -> float:
    """Umbral MÁS ALTO que todavía alcanza el recall pedido (menos falsas alarmas)."""
    y, prob = np.asarray(y_true).astype(int), np.asarray(prob)
    pos = np.sort(prob[y == 1])
    if len(pos) == 0:
        return config.UMBRAL_DECISION
    k = int(np.floor((1 - recall_objetivo) * len(pos)))
    return float(pos[k])


# --------------------------------------------------------------------------
# Transformaciones (cada una cambia UN factor)
# --------------------------------------------------------------------------
def brillo(img: np.ndarray, delta: float) -> np.ndarray:
    """Suma `delta` a todos los píxeles (aclara o oscurece sin tocar la textura)."""
    return np.clip(img.astype(np.float32) + delta, 0, 255)


def giro(img: np.ndarray, grados: float) -> np.ndarray:
    """Gira la imagen. Los bordes vacíos se rellenan por reflexión (como la aumentación)."""
    if grados % 90 == 0:
        return np.rot90(img, k=int(grados // 90)).copy()
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def lejania(img: np.ndarray, escala: float) -> np.ndarray:
    """Simula una foto tomada más lejos: se achica (pierde detalle) y vuelve al tamaño
    de entrada. Con escala 0,25 una grieta de 4 px queda en 1 px: casi invisible."""
    h, w = img.shape[:2]
    chico = cv2.resize(img, (max(2, int(w * escala)), max(2, int(h * escala))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(chico, (w, h), interpolation=cv2.INTER_LINEAR)


def perspectiva(img: np.ndarray, grados: float) -> np.ndarray:
    """Simula una foto tomada en ángulo (el borde superior queda más lejos)."""
    h, w = img.shape[:2]
    recorte = 0.5 * w * (1 - np.cos(np.radians(grados)))
    origen = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    destino = np.float32([[recorte, 0], [w - recorte, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(origen, destino)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def desenfoque(img: np.ndarray, sigma: float) -> np.ndarray:
    """Foto movida o fuera de foco."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def ruido(img: np.ndarray, sd: float, semilla: int = config.SEMILLA) -> np.ndarray:
    """Ruido de sensor (poca luz)."""
    rng = np.random.default_rng(semilla)
    return np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape), 0, 255)


def _aplicar(imagenes: np.ndarray, fn, parametro) -> np.ndarray:
    return np.stack([fn(im, parametro) for im in imagenes]).astype(np.float32)


# --------------------------------------------------------------------------
# Prueba del atajo del brillo
# --------------------------------------------------------------------------
def prueba_atajo_brillo(predecir, imagenes_sin_grieta, imagenes_con_grieta,
                        deltas=(18, 30, 51), umbral: float = config.UMBRAL_DECISION) -> pd.DataFrame:
    """¿El modelo usa el brillo como atajo?

    Se cambia SOLO el brillo (misma textura, mismo concreto) y se cuenta cuántas
    imágenes cambian de respuesta:
      - sin grieta OSCURECIDAS que pasan a "con grieta"
      - con grieta ACLARADAS que pasan a "sin grieta"

    Si casi ninguna cambia -> el modelo NO depende del brillo.
    Si muchas cambian     -> el modelo usa el brillo como atajo.

    18 es la diferencia real entre clases en Surface Crack; 51 es el máximo que
    varía la aumentación RandomBrightness(0.2).
    """
    neg = np.asarray(imagenes_sin_grieta, dtype=np.float32)
    pos = np.asarray(imagenes_con_grieta, dtype=np.float32)
    base_neg = np.asarray(predecir(neg)) >= umbral
    base_pos = np.asarray(predecir(pos)) >= umbral
    filas = []
    for d in deltas:
        osc = np.asarray(predecir(_aplicar(neg, brillo, -d))) >= umbral
        acl = np.asarray(predecir(_aplicar(pos, brillo, +d))) >= umbral
        # solo se cuentan las que ANTES estaban bien clasificadas (aísla el efecto)
        n1, n2 = (~base_neg).sum(), base_pos.sum()
        filas.append({
            "delta_brillo": d,
            "sin_grieta_oscurecidas→con_grieta_%": round(100 * float((osc & ~base_neg).sum() / max(1, n1)), 1),
            "con_grieta_aclaradas→sin_grieta_%": round(100 * float((~acl & base_pos).sum() / max(1, n2)), 1),
        })
    return pd.DataFrame(filas)


def veredicto_atajo(tabla: pd.DataFrame, tolerancia_pct: float = 5.0) -> str:
    """Interpreta la tabla de prueba_atajo_brillo."""
    peor = max(tabla["sin_grieta_oscurecidas→con_grieta_%"].max(),
               tabla["con_grieta_aclaradas→sin_grieta_%"].max())
    if peor <= tolerancia_pct:
        return (f"El brillo NO es un atajo: como máximo {peor:.1f} % de las imágenes cambian de "
                f"clase. El falso positivo debe venir de otra causa (datos nunca vistos).")
    return (f"ATENCIÓN: hasta {peor:.1f} % de las imágenes cambian de clase solo por el brillo. "
            f"El modelo está usando el brillo como atajo: hay que corregirlo.")


# --------------------------------------------------------------------------
# Pruebas de robustez generales
# --------------------------------------------------------------------------
TRANSFORMACIONES = {
    "giro_grados": (giro, (5, 15, 30, 45, 54, 90)),
    "lejania_escala": (lejania, (0.75, 0.5, 0.35, 0.25)),
    "perspectiva_grados": (perspectiva, (10, 20, 30, 45)),
    "desenfoque_sigma": (desenfoque, (1, 2, 4)),
    "ruido_sd": (ruido, (5, 15, 30)),
}


def prueba_robustez(predecir, imagenes, etiquetas=None, transformaciones=None,
                    umbral: float = config.UMBRAL_DECISION) -> pd.DataFrame:
    """Para cada transformación y cada intensidad: % de imágenes que cambian de
    respuesta y, si se dan `etiquetas`, la exactitud y el recall resultantes.

    El giro hasta 54° es lo que la aumentación RandomRotation(0.15) cubre en
    teoría; acá se MIDE si el modelo realmente lo resiste.
    """
    transformaciones = transformaciones or TRANSFORMACIONES
    imgs = np.asarray(imagenes, dtype=np.float32)
    base = np.asarray(predecir(imgs)) >= umbral
    filas = []
    for nombre, (fn, parametros) in transformaciones.items():
        for p in parametros:
            pred = np.asarray(predecir(_aplicar(imgs, fn, p))) >= umbral
            fila = {"prueba": nombre, "parametro": p,
                    "cambian_de_clase_%": round(100 * float((pred != base).mean()), 1)}
            if etiquetas is not None:
                y = np.asarray(etiquetas).astype(bool)
                fila["exactitud_%"] = round(100 * float((pred == y).mean()), 1)
                fila["recall_%"] = round(100 * float(pred[y].mean()), 1) if y.any() else float("nan")
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_diagnostico(diag: pd.DataFrame, columnas=None, umbral: float = config.UMBRAL_DECISION) -> pd.DataFrame:
    """Para cada columna de probabilidad `p_*`: AUC y cuántas grietas se separan de las sanas.

    AUC: la probabilidad de que una foto CON grieta reciba una probabilidad MÁS ALTA que una foto SIN grieta
    (1,0 = separa perfecto · 0,5 = azar). Sirve cuando las probabilidades están desplazadas hacia abajo por un
    cambio de dominio: el AUC no depende del umbral, y muestra si la red distingue aunque no llegue a 0,5.

    EXPLORATORIO: se calcula sobre las mismas fotos con las que se prueba. No es un resultado final ni se debe usar
    para elegir un umbral y luego reportar el desempeño en esas mismas fotos.
    """
    from sklearn.metrics import roc_auc_score

    columnas = columnas or [c for c in diag.columns if c.startswith("p_")]
    y = diag["etiqueta"].to_numpy().astype(int)
    filas = []
    for c in columnas:
        p = diag[c].to_numpy(dtype=float)
        pos, neg = p[y == 1], p[y == 0]
        filas.append({"metodo": c, "AUC": round(float(roc_auc_score(y, p)), 3),
                      "grietas_detectadas": f"{int((pos >= umbral).sum())}/{len(pos)}",
                      "grietas_sobre_todas_las_sanas": f"{int((pos > neg.max()).sum())}/{len(pos)}",
                      "falsos_positivos": f"{int((neg >= umbral).sum())}/{len(neg)}",
                      "max_prob_en_sanas": round(float(neg.max()), 3)})
    return pd.DataFrame(filas)
