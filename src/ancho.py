"""ancho.py — Medir el ancho de una grieta (en píxeles y en milímetros).

IDEA EN CUATRO PASOS
--------------------
1. SEGMENTAR: separar los píxeles de la grieta del resto (máscara blanca/negra).
2. ESQUELETO: adelgazar la máscara hasta una línea de 1 píxel (el "eje" de la grieta).
3. DISTANCIA: para cada punto del eje, ¿qué tan lejos está el borde más cercano?
   Esa distancia es la MITAD del ancho en ese punto. Ancho ≈ 2·distancia − 0,5.
   (El 0,5 se calibró con rectángulos de ancho exacto en 12 ángulos y 6 anchos:
   sesgo medio −0,02 px, peor caso 1 px. Con −1 el sesgo era −0,5 px.)
4. ESCALA: pasar de píxeles a milímetros con un objeto de tamaño conocido que
   esté en el MISMO plano que la grieta (una moneda, una tarjeta, una regla).

LIMITACIONES (decirlas en el informe):
  - La escala solo vale si el objeto de referencia está en el mismo plano y la
    foto es casi frontal; para fotos en ángulo, usar rectificar_con_referencia.
  - Con la segmentación clásica, una sombra o una junta oscura se mide como grieta.
    El modelo de segmentación (unet.py) lo hace mejor.
  - Un ancho de 1-2 píxeles tiene un error relativo grande: conviene acercarse
    (por debajo de ~3 px de ancho el desenfoque de la propia foto domina).
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. Segmentación clásica (sin red neuronal)
# --------------------------------------------------------------------------
def a_gris(imagen: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises uint8 (acepta BGR, RGB o gris)."""
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return imagen.astype(np.uint8)


def segmentar_clasico(imagen: np.ndarray, ventana: int | None = None, contraste_min: int = 18,
                      area_min: int = 30, fraccion_profundidad: float = 0.5) -> np.ndarray:
    """Máscara (0/1) de las zonas OSCURAS y FINAS respecto a su entorno: las grietas.

    Usa "black-hat": cierra la imagen (rellena lo oscuro más fino que la ventana) y
    resta la original. Donde hay una línea oscura, la diferencia es grande.

    DOS UMBRALES, y por qué:
      1. `contraste_min` (bajo) decide DÓNDE hay algo: una zona candidata, que
         incluye el borde borroso de la grieta.
      2. Dentro de cada zona se vuelve a cortar a `fraccion_profundidad` (0,5 =
         MEDIA PROFUNDIDAD) de su profundidad típica. Medir a media profundidad es
         el estándar para anchos de líneas borrosas: no cambia aunque la foto esté
         desenfocada o ampliada. Con un solo umbral bajo, el ancho se inflaba
         2-3 px (una grieta de 2 mm salía de 2,6 mm tras ampliar la imagen).

    ventana: tamaño del elemento. Tiene que ser MAYOR que el ancho máximo esperado.
             Por defecto, 8 % del lado menor (mínimo 15 px).
    area_min: se descartan manchas más chicas que esto (ruido).
    """
    gris = a_gris(imagen)
    if ventana is None:
        ventana = max(15, int(0.08 * min(gris.shape)))
    ventana |= 1                                           # impar
    suave = cv2.GaussianBlur(gris, (3, 3), 0)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ventana, ventana))
    blackhat = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, nucleo)
    candidata = (blackhat > contraste_min).astype(np.uint8)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(candidata, connectivity=8)
    final = np.zeros_like(candidata)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min:
            continue
        zona = etiquetas[y:y + h, x:x + w] == i
        respuesta = blackhat[y:y + h, x:x + w]
        profundidad = np.percentile(respuesta[zona], 90)
        final[y:y + h, x:x + w][zona & (respuesta >= fraccion_profundidad * profundidad)] = 1
    return final


# --------------------------------------------------------------------------
# 2-3. Esqueleto y ancho
# --------------------------------------------------------------------------
def medir_ancho(mascara: np.ndarray, mm_por_px: float | None = None) -> dict | None:
    """Ancho de la grieta a partir de su máscara. Devuelve None si no hay grieta.

    Devuelve (en px, y en mm si se da la escala):
      ancho_medio, ancho_p95, ancho_max, largo
    `ancho_p95` es el valor recomendado para decidir: el máximo puro se infla con
    cualquier mancha; el p95 ignora el 5 % más extremo.
    """
    from skimage.morphology import skeletonize

    m = (np.asarray(mascara) > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    esqueleto = skeletonize(m.astype(bool))
    if not esqueleto.any():
        return None
    distancia = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    anchos = np.clip(2.0 * distancia[esqueleto] - 0.5, 1.0, None)
    res = {"ancho_medio_px": float(anchos.mean()),
           "ancho_p95_px": float(np.percentile(anchos, 95)),
           "ancho_max_px": float(anchos.max()),
           "largo_px": int(esqueleto.sum())}
    if mm_por_px is not None:
        res.update({"ancho_medio_mm": res["ancho_medio_px"] * mm_por_px,
                    "ancho_p95_mm": res["ancho_p95_px"] * mm_por_px,
                    "ancho_max_mm": res["ancho_max_px"] * mm_por_px,
                    "largo_mm": res["largo_px"] * mm_por_px})
    return res


def mm_por_px(referencia_mm: float, referencia_px: float) -> float:
    """Escala a partir de un objeto de tamaño conocido.

    Ej.: una tarjeta de 85,6 mm de largo que mide 428 px en la foto -> 0,2 mm/px.
    """
    if referencia_px <= 0:
        raise ValueError("La referencia en píxeles debe ser positiva.")
    return referencia_mm / referencia_px


def dibujar_medicion(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Superpone máscara (rojo) y esqueleto (amarillo) para REVISAR a ojo la medición."""
    from skimage.morphology import skeletonize

    base = imagen.copy() if imagen.ndim == 3 else cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    base = base.astype(np.uint8)
    base[mascara > 0] = (0.5 * base[mascara > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    base[skeletonize(mascara > 0)] = (0, 255, 255)
    return base


# --------------------------------------------------------------------------
# 4. Fotos en ángulo: rectificar con un rectángulo de referencia
# --------------------------------------------------------------------------
def rectificar_con_referencia(imagen: np.ndarray, esquinas, ancho_mm: float, alto_mm: float,
                              px_por_mm: float = 4.0, lado_max: int = 4000):
    """Corrige la perspectiva usando un RECTÁNGULO de tamaño conocido en el muro.

    esquinas: 4 puntos (x, y) en la foto, en orden: sup-izq, sup-der, inf-der, inf-izq,
              de una tarjeta, una hoja carta (216x279 mm) o un cuadro marcado con cinta.
    Devuelve (imagen_rectificada, mm_por_px). En la imagen rectificada el plano del
    muro se ve de frente y todos los píxeles miden lo mismo: `mm_por_px = 1/px_por_mm`.

    Ventaja doble: resuelve la perspectiva Y la escala con un solo objeto.
    Solo es exacta para lo que está en el MISMO plano que la referencia.
    """
    origen = np.float32(esquinas)
    ancho_px, alto_px = ancho_mm * px_por_mm, alto_mm * px_por_mm
    destino = np.float32([[0, 0], [ancho_px, 0], [ancho_px, alto_px], [0, alto_px]])
    H = cv2.getPerspectiveTransform(origen, destino)

    h, w = imagen.shape[:2]
    esq_img = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
    transf = cv2.perspectiveTransform(esq_img, H).reshape(-1, 2)
    minimo, maximo = transf.min(axis=0), transf.max(axis=0)
    tamano = maximo - minimo
    reduccion = min(1.0, lado_max / float(tamano.max()))       # evita imágenes gigantes
    T = np.array([[reduccion, 0, -minimo[0] * reduccion],
                  [0, reduccion, -minimo[1] * reduccion],
                  [0, 0, 1]], dtype=np.float64)
    salida = cv2.warpPerspective(imagen, T @ H, (int(np.ceil(tamano[0] * reduccion)),
                                                  int(np.ceil(tamano[1] * reduccion))),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return salida, 1.0 / (px_por_mm * reduccion)


# --------------------------------------------------------------------------
# 5. Orientación de la grieta (vertical / horizontal / diagonal)
# --------------------------------------------------------------------------
def orientacion_grieta(mascara: np.ndarray, roll_deg: float = 0.0, tolerancia_deg: float = 20.0) -> dict | None:
    """Dirección dominante de la grieta, a partir del eje principal de su esqueleto (PCA).

    Es el "nivel intermedio" del documento del reto: clasificar la orientación
    (vertical, horizontal, diagonal) SIN entrenar nada, con geometría.

    CÓMO: los píxeles del esqueleto forman una nube de puntos alargada; su dirección más
    larga (primer componente principal) es la dirección de la grieta.

    angulo_deg: 0° = horizontal, 90° = vertical (medido respecto del MUNDO), con el eje y hacia
                ARRIBA como en matemática: una grieta tipo "/" mide 45° y una tipo "\\" mide 135°.
                Para decidir vertical/horizontal/diagonal ese sentido no importa.
    roll_deg:   giro del celular al sacar la foto. Si el celular estaba girado, una grieta
                vertical se ve inclinada en la imagen; se descuenta con el sensor de gravedad
                (misma convención que inclinacion.py).
    elongacion: 0 = nube redonda (la orientación no significa nada) a 1 = línea perfecta.
    """
    from skimage.morphology import skeletonize

    ys, xs = np.nonzero(skeletonize(np.asarray(mascara) > 0))
    if len(xs) < 10:
        return None
    puntos = np.column_stack([xs.astype(float), -ys.astype(float)])     # y hacia arriba
    puntos -= puntos.mean(axis=0)
    valores, vectores = np.linalg.eigh(np.cov(puntos.T))
    principal = vectores[:, int(np.argmax(valores))]
    angulo = float(np.degrees(np.arctan2(principal[1], principal[0])) % 180.0)
    angulo = (angulo - roll_deg) % 180.0
    if abs(angulo - 90.0) <= tolerancia_deg:
        categoria = "vertical"
    elif angulo <= tolerancia_deg or angulo >= 180.0 - tolerancia_deg:
        categoria = "horizontal"
    else:
        categoria = "diagonal"
    elongacion = float(1.0 - np.sqrt(max(valores.min(), 0.0) / max(valores.max(), 1e-12)))
    return {"angulo_deg": angulo, "categoria": categoria, "elongacion": elongacion}
