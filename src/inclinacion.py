"""inclinacion.py — ¿Cuánto se desploma el muro respecto de la vertical?

PASOS
-----
1. LÍNEAS (Canny + Hough): segmentos rectos largos casi verticales. Muy preciso con rayas nítidas.
2. BORDES (histograma de orientación): si no hay líneas largas, que es lo normal en una foto real, se mide la dirección
   dominante de todos los bordes fuertes. Un medidor que solo usara líneas devolvía "nada" en columnas simuladas realistas.
3. Si ningún método encuentra un borde vertical dominante, no se devuelve un número.

EL PROBLEMA QUE ESTO RESUELVE
-----------------------------
Si el celular está girado (rolido) al tomar la foto, un muro perfectamente a
plomo sale inclinado en la imagen: el módulo confundiría el giro del celular
con el desplome del edificio. El sensor de gravedad del celular dice cuánto
estaba girado; `corregir_con_gravedad` lo descuenta.

CONVENCIONES (fijarlas evita errores de signo)
  angulo_imagen : grados medidos en la imagen. POSITIVO si la parte de ARRIBA del
                  muro se inclina hacia la DERECHA.
  roll_celular  : POSITIVO si el celular estaba girado en sentido horario
                  (visto de frente, desde quien lo sostiene).
  Con un muro a plomo y el celular girado `roll`, angulo_imagen = −roll.
  Por eso:  inclinacion_real = angulo_imagen + roll.

LIMITACIÓN IMPORTANTE: si el celular apunta hacia arriba o hacia abajo (pitch),
las verticales CONVERGEN (efecto "keystone") y ninguna corrección de roll
alcanza. Hay que tomar la foto con el celular vertical y de frente al muro.
"""
from __future__ import annotations

import cv2
import numpy as np


def _mediana_ponderada(valores: np.ndarray, pesos: np.ndarray) -> float:
    orden = np.argsort(valores)
    v, p = valores[orden], pesos[orden]
    acumulado = np.cumsum(p) / p.sum()
    return float(v[np.searchsorted(acumulado, 0.5)])


def _gris(imagen: np.ndarray) -> np.ndarray:
    g = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return g.astype(np.uint8)


def mascara_piel(rgb: np.ndarray) -> np.ndarray | None:
    """Regiones de color PIEL (manos, brazos) como máscara booleana, o None si no hay o si ocupan demasiado.

    POR QUÉ: un brazo o una mano son bordes largos y rectos, y el medidor los confunde con el eje de una columna (con confianza
    alta). Se detectan por color (Cr 142–175, Cb 80–122 en YCrCb: cubre pieles claras a oscuras y deja afuera el beige y el gris
    del concreto) y se borran antes de medir.

    LÍMITES: no ve brazos con manga ni guantes, y una pared de color salmón o terracota puede parecer piel. Por eso solo se
    aplica si la zona de piel ocupa entre 0,5 % y 35 % de la imagen: una pared entera de color cálido no se borra.
    """
    if rgb is None or rgb.ndim != 3:
        return None
    ycc = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    m = ((cr >= 142) & (cr <= 175) & (cb >= 80) & (cb <= 122)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    fraccion = float(m.mean())
    if fraccion < 0.005 or fraccion > 0.35:
        return None
    return m.astype(bool)


def borrar_piel(gris: np.ndarray, rgb: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Rellena las zonas de piel (y un margen) con el gris mediano del resto, para que no aporten bordes. Devuelve la imagen
    y la fracción de la foto que se borró."""
    piel = mascara_piel(rgb) if rgb is not None else None
    if piel is None:
        return gris, 0.0
    ancho = max(15, int(0.02 * max(gris.shape)))
    amplia = cv2.dilate(piel.astype(np.uint8), np.ones((ancho, ancho), np.uint8)).astype(bool)
    if amplia.all():
        return gris, 0.0
    sal = gris.copy()
    sal[amplia] = int(np.median(gris[~amplia]))
    return sal, float(amplia.mean())


def _por_lineas(gris: np.ndarray, votos: int, max_desvio: float, largo_min_frac: float) -> dict | None:
    """Método 1 — LÍNEAS (Canny + Hough). Muy preciso con rayas nítidas; en fotos reales casi nunca encuentra una
    línea larga e ininterrumpida (bordes suaves, ruido), y entonces devuelve None."""
    h, w = gris.shape
    bordes = cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0), 50, 150)
    largo_min = int(largo_min_frac * min(h, w))
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=votos, minLineLength=largo_min, maxLineGap=10)
    if lineas is None:
        return None
    angulos, largos = [], []
    for x1, y1, x2, y2 in lineas.reshape(-1, 4):
        if y1 > y2:                                   # orientar de arriba hacia abajo
            x1, y1, x2, y2 = x2, y2, x1, y1
        dy = y2 - y1
        largo = float(np.hypot(x2 - x1, dy))
        if dy < 0.5 * largo:                          # demasiado horizontal
            continue
        tilt = float(np.degrees(np.arctan2(x1 - x2, dy)))   # + si la punta de arriba va a la derecha
        if abs(tilt) <= max_desvio:
            angulos.append(tilt)
            largos.append(largo)
    if not angulos:
        return None
    a, p = np.array(angulos), np.array(largos)
    mediana = _mediana_ponderada(a, p)
    return {"angulo_imagen_deg": mediana, "n_lineas": len(a), "dispersion_deg": _mediana_ponderada(np.abs(a - mediana), p),
            "metodo": "líneas", "confianza": None}


SIGMA_BORDES = 3.0          # suavizado: apaga la textura fina y deja los bordes largos
PERCENTIL_BORDES = 90       # solo se consideran el 10 % de los píxeles con borde más fuerte
CONFIANZA_MINIMA = 0.15     # por debajo no hay un borde vertical dominante: se devuelve None
CONFIANZA_BUENA = 0.25      # por debajo de esto la medición se marca como poco confiable


def _orientacion_de_bordes(gris: np.ndarray, sigma: float = SIGMA_BORDES, percentil: float = PERCENTIL_BORDES):
    """Para cada píxel de borde fuerte: su inclinación respecto de la vertical (+ si la punta de arriba va a la derecha)
    y su fuerza. Trabaja a ≤ 800 px de lado largo. Devuelve (inclinaciones, pesos, máscara, factor_de_escala)."""
    h, w = gris.shape
    f = min(1.0, 800.0 / max(h, w))
    g = cv2.resize(gris, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gris
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mascara = mag > np.percentile(mag, percentil)
    dx, dy = -gy[mascara], gx[mascara]                 # el borde es perpendicular al gradiente
    signo = np.where(dy < 0, -1.0, 1.0)                # orientar hacia abajo
    dx, dy = dx * signo, dy * signo
    return np.degrees(np.arctan2(-dx, dy)), mag[mascara], mascara, f


def _por_bordes(gris: np.ndarray, max_desvio: float) -> dict | None:
    """Método 2 — BORDES. Histograma de la orientación de todos los bordes fuertes, ponderado por su fuerza: el pico es la
    inclinación dominante. No necesita líneas completas, así que aguanta bordes suaves, ruido y fotos reales.
    `confianza` = qué fracción del peso está cerca del pico (ruido o textura pura ≈ 0,09; una columna, 0,2 a 0,4)."""
    tilt, peso, _, _ = _orientacion_de_bordes(gris)
    ok = np.abs(tilt) <= max_desvio
    if ok.sum() < 50:
        return None
    tilt, peso = tilt[ok], peso[ok]
    bins = np.arange(-max_desvio, max_desvio + 0.25, 0.25)
    hist, _ = np.histogram(tilt, bins, weights=peso)
    suave = np.convolve(hist, cv2.getGaussianKernel(9, 4).ravel(), mode="same")      # suavizado de ~1°
    pico = float((bins[:-1] + 0.125)[int(np.argmax(suave))])
    for _ in range(3):                                                                 # afinado por media ponderada
        cerca = np.abs(tilt - pico) <= 2.5
        pico = float(np.average(tilt[cerca], weights=peso[cerca]))
    cerca = np.abs(tilt - pico) <= 2.5
    confianza = float(peso[cerca].sum() / peso.sum())
    if confianza < CONFIANZA_MINIMA:
        return None
    return {"angulo_imagen_deg": pico, "n_lineas": int(cerca.sum()),
            "dispersion_deg": float(np.sqrt(np.average((tilt[cerca] - pico) ** 2, weights=peso[cerca]))),
            "metodo": "bordes", "confianza": confianza}


def mapa_de_bordes_alineados(imagen: np.ndarray, angulo_deg: float, tolerancia_deg: float = 2.5) -> np.ndarray:
    """Máscara booleana, del tamaño de la imagen, con los bordes fuertes que apuntan en la dirección medida. Sirve para
    DIBUJARLOS sobre la foto y comprobar a ojo que se midió la columna y no otra cosa."""
    gris = _gris(imagen)
    tilt, _, mascara, f = _orientacion_de_bordes(gris)
    alineado = np.zeros(mascara.shape, np.uint8)
    alineado[mascara] = (np.abs(tilt - angulo_deg) <= tolerancia_deg).astype(np.uint8)
    alineado = cv2.dilate(alineado, np.ones((3, 3), np.uint8))
    if f < 1:
        alineado = cv2.resize(alineado, (gris.shape[1], gris.shape[0]), interpolation=cv2.INTER_NEAREST)
    return alineado.astype(bool)


def estimar_inclinacion(imagen: np.ndarray, votos: int = 60, max_desvio: float = 30.0,
                        largo_min_frac: float = 0.25, devolver_mapa: bool = False,
                        rgb: np.ndarray | None = None) -> dict | None:
    """Ángulo del muro respecto de la vertical, tal como se ve en la IMAGEN.

    Prueba dos métodos: primero las LÍNEAS (precisas si hay rayas nítidas) y, si no hay suficientes, los BORDES (robusto con
    fotos reales). Devuelve None si no hay un borde vertical dominante (pared lisa, textura, líneas horizontales): en ese caso
    no hay que inventar un número. El resultado dice con qué `metodo` se midió y, para bordes, su `confianza`.
    """
    gris, fraccion_piel = borrar_piel(_gris(imagen), rgb)           # manos y brazos no son el muro
    r = _por_lineas(gris, votos, max_desvio, largo_min_frac)
    if r is None or r["n_lineas"] < 3 or r["dispersion_deg"] > 2.0:
        r = _por_bordes(gris, max_desvio)
    if r is not None:
        r["piel_excluida"] = fraccion_piel
        if devolver_mapa:
            r["mapa_bordes"] = mapa_de_bordes_alineados(gris, r["angulo_imagen_deg"])
    return r


def corregir_con_gravedad(angulo_imagen_deg: float, roll_celular_deg: float,
                          pitch_celular_deg: float | None = None,
                          tolerancia_pitch_deg: float = 5.0) -> dict:
    """Inclinación REAL del muro, descontando el giro del celular.

    pitch_celular_deg: cuánto apuntaba el celular hacia arriba/abajo (0 = vertical).
    """
    real = angulo_imagen_deg + roll_celular_deg
    avisos = []
    if pitch_celular_deg is not None and abs(pitch_celular_deg) > tolerancia_pitch_deg:
        avisos.append(f"El celular apuntaba {pitch_celular_deg:+.0f}° hacia arriba/abajo: las "
                      f"verticales convergen y esta medición NO es confiable. Repetir con el "
                      f"celular vertical y de frente al muro.")
    return {"inclinacion_real_deg": real, "angulo_imagen_deg": angulo_imagen_deg,
            "roll_celular_deg": roll_celular_deg, "confiable": not avisos, "avisos": avisos}
