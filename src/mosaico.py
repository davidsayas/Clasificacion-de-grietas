"""mosaico.py — Fotos tomadas de LEJOS (o muy grandes): clasificar por teselas.

EL PROBLEMA
-----------
Si se achica una foto de 4000x3000 a 224x224 para la red, una grieta fina
desaparece: su ancho pasa de ~20 px a ~1 px. Se mide en la prueba de lejanía
(evaluar.prueba_robustez) y el recall cae.

LA SOLUCIÓN
-----------
Cortar la foto en pedazos (teselas) del tamaño de entrada de la red, con
solapamiento, y clasificar cada una. Beneficios:
  - la grieta conserva su detalle,
  - se sabe DÓNDE está (mapa de calor),
  - un objeto raro en una tesela no arruina la decisión de toda la foto.

ESCALAS
-------
Los recortes de entrenamiento (227 px, de fotos de 4032x3024) muestran la grieta
con cierto tamaño en píxeles. Si la foto se tomó más cerca o más lejos, ese tamaño
cambia. Con `escalas=(1.0, 0.5, 0.25)` se repite el mosaico a varios tamaños de
foto y se toma el MÁXIMO por zona: alguna escala coincide con la del entrenamiento.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config


def posiciones(largo: int, lado: int, paso: int) -> list[int]:
    """Posiciones de inicio de las teselas a lo largo de un eje, cubriendo hasta el borde."""
    if largo <= lado:
        return [0]
    pos = list(range(0, largo - lado + 1, paso))
    if pos[-1] != largo - lado:
        pos.append(largo - lado)                      # la última tesela llega justo al borde
    return pos


def _rellenar(imagen: np.ndarray, lado: int) -> np.ndarray:
    """Si la imagen es más chica que una tesela, se rellena por reflexión."""
    h, w = imagen.shape[:2]
    if h >= lado and w >= lado:
        return imagen
    return cv2.copyMakeBorder(imagen, 0, max(0, lado - h), 0, max(0, lado - w), cv2.BORDER_REFLECT_101)


def clasificar_mosaico(imagen: np.ndarray, predecir, lado: int = config.IMG, solape: float = 0.5,
                       escalas=(1.0,), umbral: float = config.UMBRAL_DECISION,
                       tam_lote: int = 64, vetar=None, margen_veto: int = 1) -> dict:
    """Clasifica una foto por teselas.

    imagen:  (alto, ancho, 3), en el MISMO orden de canales con que se entrenó (RGB).
    predecir: función que recibe (N, lado, lado, 3) con valores 0-255 y devuelve N
              probabilidades de "con grieta".
    vetar:    (opcional) función que recibe lo mismo y devuelve N booleanos: True = "esta
              tesela NO es concreto" (la puerta). Una tesela vetada no puede dar grieta.
    margen_veto: también se descartan las teselas VECINAS a una vetada (1 = las 8 de
              alrededor). Un objeto (una mano) en el borde de una tesela deja fragmentos en
              las vecinas que pasan la puerta y disparan falsas alarmas. Costo: un poco
              menos de sensibilidad junto a objetos. 0 desactiva el margen.

    Devuelve:
      mapa            (alto, ancho) con la probabilidad por zona (0-1)
      prob_max        probabilidad de la tesela más sospechosa
      fraccion        proporción de teselas con grieta
      cajas           lista de (x, y, lado, prob) en coordenadas de la foto ORIGINAL
      n_teselas       cuántas teselas se clasificaron
      fraccion_vetada proporción de teselas que la puerta rechazó DIRECTAMENTE
    """
    h0, w0 = imagen.shape[:2]
    mapas, cajas, n_total, n_pos, n_veto = [], [], 0, 0, 0
    paso = max(1, int(lado * (1 - solape)))

    for escala in escalas:
        if escala != 1.0:
            interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
            esc = cv2.resize(imagen, (max(1, int(w0 * escala)), max(1, int(h0 * escala))), interpolation=interp)
        else:
            esc = imagen
        esc = _rellenar(esc, lado)
        h, w = esc.shape[:2]
        xs, ys = posiciones(w, lado, paso), posiciones(h, lado, paso)
        coords = [(x, y) for y in ys for x in xs]
        teselas = np.stack([esc[y:y + lado, x:x + lado] for x, y in coords]).astype(np.float32)

        probs = np.concatenate([np.asarray(predecir(teselas[i:i + tam_lote])).ravel()
                                for i in range(0, len(teselas), tam_lote)])
        if vetar is not None:
            veto = np.concatenate([np.asarray(vetar(teselas[i:i + tam_lote])).ravel()
                                   for i in range(0, len(teselas), tam_lote)]).astype(bool)
            n_veto += int(veto.sum())
            if margen_veto > 0:
                k = 2 * margen_veto + 1
                grilla = veto.reshape(len(ys), len(xs)).astype(np.uint8)
                veto = cv2.dilate(grilla, np.ones((k, k), np.uint8)).ravel().astype(bool)
            probs = np.where(veto, 0.0, probs)
        acum, cuenta = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for (x, y), p in zip(coords, probs):
            acum[y:y + lado, x:x + lado] += p
            cuenta[y:y + lado, x:x + lado] += 1
            n_total += 1
            if p >= umbral:
                n_pos += 1
                cajas.append((int(x / escala), int(y / escala), int(lado / escala), float(p)))
        mapa = acum / np.maximum(cuenta, 1)
        mapa = cv2.resize(mapa[:int(round(h0 * escala)) or 1, :int(round(w0 * escala)) or 1],
                          (w0, h0), interpolation=cv2.INTER_LINEAR)
        mapas.append(mapa)

    final = np.maximum.reduce(mapas)
    todas = np.array([c[3] for c in cajas]) if cajas else np.array([0.0])
    return {"mapa": final, "prob_max": float(max(final.max(), todas.max())),
            "fraccion": n_pos / max(1, n_total), "cajas": cajas, "n_teselas": n_total,
            "fraccion_vetada": n_veto / max(1, n_total)}


def superponer_mapa(imagen: np.ndarray, mapa: np.ndarray, alfa: float = 0.5) -> np.ndarray:
    """Pinta el mapa de probabilidad sobre la foto (rojo = más probable que haya grieta)."""
    calor = cv2.applyColorMap((np.clip(mapa, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = imagen.astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alfa, calor, alfa, 0)
