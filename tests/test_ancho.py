import cv2
import numpy as np

from sinteticos import textura_concreto
from src import ancho


def rect_exacto(w, ang, n=500, largo=360, valor=1, base=None):
    """Rectángulo girado con ancho EXACTO w (vértices con subpíxel).
    cv2.line(thickness=w) NO sirve para esto: dibuja w+2 px de ancho."""
    S = 16
    c = n / 2
    a = np.radians(ang)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    pts = [c * np.array([1, 1]) + s1 * largo / 2 * u + s2 * w / 2 * v
           for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    m = np.zeros((n, n), np.uint8) if base is None else base
    cv2.fillPoly(m, [np.round(np.array(pts) * S).astype(np.int32)], valor, shift=4)
    return m


def test_ancho_sobre_mascara_exacta():
    """Máscara perfecta: el ancho medido debe estar a ~1 px de la verdad, sin sesgo."""
    errores = []
    for w in (3, 4, 5, 8, 12, 20):
        for ang in range(0, 180, 15):
            m = rect_exacto(w, ang)
            real = m.sum() / 360.0
            r = ancho.medir_ancho(m)
            errores.append(r["ancho_p95_px"] - real)
    errores = np.array(errores)
    print(f"      sesgo {errores.mean():+.2f} px, peor caso {np.abs(errores).max():.2f} px")
    assert abs(errores.mean()) < 0.15
    assert np.abs(errores).max() <= 1.15


def test_ancho_desde_imagen_con_ruido():
    """Pipeline completo: imagen con ruido -> segmentar -> medir."""
    rng = np.random.default_rng(0)
    for w in (4, 8, 12):
        fondo = cv2.GaussianBlur(rng.normal(180, 5, (500, 500)).astype(np.float32), (0, 0), 1.0)
        img = np.clip(fondo, 0, 255).astype(np.uint8)
        rect_exacto(w, 35, base=img, valor=70)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)               # bordes suaves, como una foto
        masc = ancho.segmentar_clasico(img)
        r = ancho.medir_ancho(masc)
        assert r is not None
        assert abs(r["ancho_p95_px"] - w) <= 2.0, (w, r)


def test_sin_grieta_devuelve_none():
    img = textura_concreto(224, brillo=180, ruido=4)
    masc = ancho.segmentar_clasico(img)
    assert ancho.medir_ancho(masc) is None or masc.sum() < 50


def test_mm_por_px():
    assert abs(ancho.mm_por_px(85.6, 428) - 0.2) < 1e-9
    m = rect_exacto(10, 0)
    real_mm = (m.sum() / 360.0) * 0.2                  # el dibujo ocupa 11 columnas: se mide de la máscara
    r = ancho.medir_ancho(m, mm_por_px=0.2)
    assert abs(r["ancho_p95_mm"] - real_mm) <= 0.25, (r["ancho_p95_mm"], real_mm)


def test_rectificar_corrige_perspectiva_y_da_la_escala():
    """Escena plana: tarjeta de 85.6x54 mm + grieta de 2 mm, fotografiada en ángulo.

    (a) medir ingenuamente con la escala de la tarjeta, (b) rectificar primero.
    """
    ppm = 8.0                                           # px por mm en la escena "real"
    W, H = int(260 * ppm), int(180 * ppm)
    escena = np.full((H, W), 175, np.uint8)
    x0, y0, tw, th = int(20 * ppm), int(20 * ppm), int(85.6 * ppm), int(54 * ppm)
    escena[y0:y0 + th, x0:x0 + tw] = 235                # tarjeta de referencia
    cx = int(190 * ppm)
    escena[int(10 * ppm):int(170 * ppm), cx - 8:cx + 8] = 60     # grieta vertical: 16 px = 2 mm exactos
    escena = cv2.GaussianBlur(escena, (0, 0), 1.0)

    # "tomar la foto": perspectiva fuerte (el lado derecho queda más lejos)
    origen = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    destino = np.float32([[0, 0], [W * 0.80, H * 0.12], [W * 0.80, H * 0.88], [0, H]])
    M = cv2.getPerspectiveTransform(origen, destino)
    foto = cv2.warpPerspective(escena, M, (int(W * 0.85), H), borderValue=175)

    # el usuario marca las 4 esquinas de la tarjeta EN LA FOTO
    esq = np.float32([[x0, y0], [x0 + tw, y0], [x0 + tw, y0 + th], [x0, y0 + th]]).reshape(-1, 1, 2)
    esq_foto = cv2.perspectiveTransform(esq, M).reshape(-1, 2)

    ref_px = float(np.linalg.norm(esq_foto[1] - esq_foto[0]))
    r_ing = ancho.medir_ancho(ancho.segmentar_clasico(foto, ventana=61),
                              mm_por_px=ancho.mm_por_px(85.6, ref_px))
    rect, mm_px = ancho.rectificar_con_referencia(foto, esq_foto, 85.6, 54.0, px_por_mm=6.0)
    r_rect = ancho.medir_ancho(ancho.segmentar_clasico(rect, ventana=61), mm_por_px=mm_px)

    err_ing = abs(r_ing["ancho_p95_mm"] - 2.0)
    err_rect = abs(r_rect["ancho_p95_mm"] - 2.0)
    print(f"      verdad 2.00 mm | sin rectificar: {r_ing['ancho_p95_mm']:.2f} (error {err_ing:.2f}) "
          f"| rectificado: {r_rect['ancho_p95_mm']:.2f} (error {err_rect:.2f})")
    assert err_rect <= 0.35, r_rect
    assert err_rect < err_ing, "rectificar debería mejorar la medición"
