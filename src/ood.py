"""ood.py — Puerta "¿esto parece concreto?" (detección de datos fuera de dominio).

POR QUÉ HACE FALTA
------------------
El clasificador solo conoce dos respuestas: "con grieta" y "sin grieta". Cuando le
llega algo que nunca vio (una mano, una pared pintada lisa, una baldosa), NO tiene
cómo decir "esto no es concreto": fuerza una de las dos. Eso fue el falso positivo.

CÓMO FUNCIONA
-------------
Cada imagen se resume en un vector (los 1280 números que salen de la capa de
pooling de MobileNetV2: el "embedding"). Se aprende cómo se ve el embedding de
las imágenes de CONCRETO de entrenamiento (centro y forma de la nube). Una imagen
nueva se compara con esa nube con la DISTANCIA DE MAHALANOBIS: si queda muy lejos,
se dice "no parece concreto" y el sistema NO afirma nada sobre grietas.

El umbral se fija con un percentil de las distancias de VALIDACIÓN (ej.: 99 %):
así se sabe de antemano qué fracción de concreto legítimo se rechazaría (1 %).

HONESTIDAD: que esta puerta rechace bien una pared lisa depende de los datos.
Se MIDE con AUROC sobre los negativos difíciles (ver `auroc`); no se asume.
"""
from __future__ import annotations

import numpy as np


class PuertaConcreto:
    def __init__(self, percentil: float = 99.0):
        self.percentil = percentil
        self.media = None
        self.precision = None
        self.umbral = None

    def ajustar(self, emb_entrenamiento: np.ndarray, emb_validacion: np.ndarray | None = None):
        """Aprende la nube de concreto. `emb_*` tienen forma (N, D).

        Usa un estimador de covarianza con encogimiento (Ledoit-Wolf): con D=1280
        dimensiones la covarianza común es inestable; este la estabiliza.
        """
        from sklearn.covariance import LedoitWolf

        x = np.asarray(emb_entrenamiento, dtype=np.float64)
        self.media = x.mean(axis=0)
        lw = LedoitWolf().fit(x)
        self.precision = lw.precision_
        ref = np.asarray(emb_validacion if emb_validacion is not None else x, dtype=np.float64)
        self.umbral = float(np.percentile(self.distancia(ref), self.percentil))
        return self

    def distancia(self, emb: np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis de cada embedding a la nube de concreto."""
        d = np.atleast_2d(np.asarray(emb, dtype=np.float64)) - self.media
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))

    def es_concreto(self, emb: np.ndarray) -> np.ndarray:
        return self.distancia(emb) <= self.umbral

    def auroc(self, emb_concreto: np.ndarray, emb_otros: np.ndarray) -> float:
        """¿Qué tan bien separa concreto de lo que NO es concreto? 1.0 = perfecto, 0.5 = azar."""
        from sklearn.metrics import roc_auc_score

        d = np.concatenate([self.distancia(emb_concreto), self.distancia(emb_otros)])
        y = np.r_[np.zeros(len(emb_concreto)), np.ones(len(emb_otros))]
        return float(roc_auc_score(y, d))

    def guardar(self, ruta) -> None:
        np.savez(ruta, media=self.media, precision=self.precision, umbral=self.umbral,
                 percentil=self.percentil)

    @classmethod
    def cargar(cls, ruta) -> "PuertaConcreto":
        z = np.load(ruta)
        p = cls(float(z["percentil"]))
        p.media, p.precision, p.umbral = z["media"], z["precision"], float(z["umbral"])
        return p
