"""Integración con modelos FALSOS: prueba el encadenado completo, y el arreglo del falso positivo."""
import cv2
import numpy as np

from src import ancho, predecir
from sinteticos import textura_concreto, con_grieta


def pred_oscuro(lote):
    """Falso clasificador con el defecto del modelo real: 'oscuro y con contraste => grieta'."""
    out = []
    for im in lote:
        g = im.mean(axis=2)
        out.append(float(((g < 100).mean() > 0.004)))
    return np.array(out)


class EmbFalso:
    """Embedding falso: (brillo medio, TEXTURA LOCAL).
    Textura local = percentil 25 del filtro pasa-altos: si más de un cuarto de la zona
    es lisa, vale ~0. (El desvío GLOBAL no sirve: un objeto oscuro lo sube aunque la
    pared sea lisa.)"""
    def __call__(self, lote):
        sal = []
        for im in lote:
            g = im.mean(axis=2).astype(np.float32)
            sal.append([im.mean(), float(np.percentile(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2)), 25))])
        return np.array(sal, dtype=float)


class PuertaFalsa:
    """Rechaza lo LISO: una pared pintada o una mano no tienen textura de concreto."""
    def es_concreto(self, emb):
        return emb[:, 1] > 0.3


def _muro_con_grieta(n=224):
    return con_grieta(textura_concreto(n, semilla=3, brillo=185, ruido=14), grosor=3, semilla=5, oscuridad=110)


def _pared_lisa_con_mano(n=224):
    img = np.full((n, n, 3), 205, np.uint8)               # pared pintada: lisa
    cv2.ellipse(img, (110, 120), (30, 70), 20, 0, 360, (45, 45, 45), -1)   # "mano": mancha oscura
    return img


def test_sin_puerta_el_modelo_defectuoso_da_falso_positivo():
    s = predecir.Sistema(pred_oscuro)
    r = s.analizar(_pared_lisa_con_mano())
    assert r["prob_grieta"] >= 0.5, "reproduce el falso positivo original"
    assert any("Sin puerta" in n for n in r["notas"])


def test_con_puerta_el_falso_positivo_se_corrige():
    s = predecir.Sistema(pred_oscuro, embeddings=EmbFalso(), puerta=PuertaFalsa())
    r = s.analizar(_pared_lisa_con_mano())
    assert r["es_concreto"] is False
    assert r["riesgo"].nivel == "INDETERMINADO", r["riesgo"]


def test_con_puerta_NO_pierde_una_grieta_real():
    s = predecir.Sistema(pred_oscuro, embeddings=EmbFalso(), puerta=PuertaFalsa())
    r = s.analizar(_muro_con_grieta(), mm_por_px=0.2)
    assert r["es_concreto"] is True and r["prob_grieta"] >= 0.5
    assert r["medicion"] is not None and r["medicion"]["ancho_p95_mm"] > 0
    assert r["riesgo"].nivel in ("BAJO", "MEDIO", "ALTO")


def test_foto_grande_usa_mosaico_y_localiza():
    foto = np.full((900, 1200, 3), 185, np.uint8)
    rng = np.random.default_rng(0)
    foto = np.clip(185 + rng.normal(0, 6, foto.shape), 0, 255).astype(np.uint8)
    cv2.line(foto, (800, 500), (870, 700), (40, 40, 40), 6)
    s = predecir.Sistema(pred_oscuro, embeddings=EmbFalso(), puerta=PuertaFalsa())
    r = s.analizar(foto, mm_por_px=0.5)
    assert r["modo"] == "mosaico" and r["prob_grieta"] >= 0.5
    assert r["mapa"][600, 830] >= 0.5 and r["mapa"][100, 100] < 0.5
    assert r["medicion"] is not None


def test_mano_en_una_esquina_de_una_foto_grande_no_arruina_la_foto():
    """La mano ocupa una tesela; el resto es concreto: la foto sigue siendo concreto
    y la tesela de la mano NO genera falsa alarma."""
    rng = np.random.default_rng(1)
    foto = np.clip(185 + rng.normal(0, 8, (900, 1200, 3)), 0, 255).astype(np.uint8)
    foto[600:850, 900:1150] = 205                          # zona lisa (pared pintada)
    cv2.ellipse(foto, (1000, 720), (30, 80), 15, 0, 360, (45, 45, 45), -1)   # la mano
    s = predecir.Sistema(pred_oscuro, embeddings=EmbFalso(), puerta=PuertaFalsa())
    r = s.analizar(foto)
    assert r["es_concreto"] is True
    assert r["prob_grieta"] < 0.5, "la mano no debe contar como grieta"


def test_inclinacion_solo_se_usa_con_el_giro_del_celular():
    muro = np.full((300, 300, 3), 200, np.uint8)
    for x in range(40, 280, 40):
        cv2.line(muro, (x, 299), (x + 20, 0), (40, 40, 40), 3)   # juntas inclinadas ~3.8°
    s = predecir.Sistema(lambda l: np.zeros(len(l)))
    sin_roll = s.analizar(muro)
    assert sin_roll["inclinacion_deg"] is None
    assert any("giro del celular" in n for n in sin_roll["notas"])
    con_roll = s.analizar(muro, roll_deg=0.0, pitch_deg=0.0)
    assert con_roll["inclinacion_deg"] is not None and abs(con_roll["inclinacion_deg"] - 3.8) < 1.0
