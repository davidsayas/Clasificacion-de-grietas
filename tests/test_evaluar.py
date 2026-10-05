import cv2
import numpy as np

from sinteticos import textura_concreto, con_grieta
from src import evaluar


# ---------- modelos FALSOS para probar las pruebas ----------
def pred_atajo_brillo(lote):
    """Un 'modelo' tramposo: solo mira el brillo. Más oscuro => grieta.
    Umbral 171: a mitad de camino entre las medias reales de Surface Crack (162 y 180)."""
    return 1 / (1 + np.exp((lote.mean(axis=(1, 2, 3)) - 171) / 4))


def pred_contraste_local(lote):
    """Un modelo 'bueno': busca líneas oscuras respecto a su entorno. Ignora el brillo global."""
    salida = []
    for im in lote:
        g = im.mean(axis=2).astype(np.float32)
        fondo = cv2.GaussianBlur(g, (0, 0), 6)
        frac = float(((g - fondo) < -25).mean())
        salida.append(1 / (1 + np.exp(-(frac - 0.006) / 0.002)))
    return np.array(salida)


def _lotes(n=30, grosor=3, oscuridad=90):
    """Imitan Surface Crack: sin grieta ~180 de brillo, con grieta ~162."""
    neg = np.stack([textura_concreto(semilla=i, brillo=180) for i in range(n)]).astype(np.float32)
    pos = np.stack([con_grieta(textura_concreto(semilla=100 + i, brillo=166), grosor=grosor,
                               semilla=i, oscuridad=oscuridad) for i in range(n)]).astype(np.float32)
    return neg, pos


def test_metricas_y_wilson():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([.9, .8, .2, .1, .6, .3, .2, .7])
    m = evaluar.metricas(y, p)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 3, 1, 1)
    assert m["recall"] == 0.75 and m["matriz"] == [[3, 1], [1, 3]]
    lo, hi = evaluar.intervalo_wilson(50, 100)
    assert abs(lo - 0.4038) < 0.002 and abs(hi - 0.5962) < 0.002
    lo, hi = evaluar.intervalo_wilson(2, 3000)       # "2 errores de 3000"
    assert hi > 0.0015, "con 2 errores la incertidumbre NO es cero"


def test_umbral_para_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(1000), np.zeros(1000)]
    p = np.r_[rng.beta(8, 2, 1000), rng.beta(2, 8, 1000)]
    u = evaluar.umbral_para_recall(y, p, 0.95)
    assert evaluar.metricas(y, p, u)["recall"] >= 0.95


def test_el_modelo_falso_funciona_antes_de_probar_nada():
    neg, pos = _lotes()
    for pred in (pred_atajo_brillo, pred_contraste_local):
        assert (pred(neg) < 0.5).mean() > 0.9 and (pred(pos) >= 0.5).mean() > 0.9, pred.__name__


def test_prueba_atajo_DETECTA_un_modelo_que_usa_el_brillo():
    neg, pos = _lotes()
    t = evaluar.prueba_atajo_brillo(pred_atajo_brillo, neg, pos, deltas=(18, 30, 51))
    # a 30 niveles de diferencia el modelo tramposo debe cambiar de opinión masivamente
    assert t["sin_grieta_oscurecidas→con_grieta_%"].iloc[1] > 80, t
    assert "ATENCIÓN" in evaluar.veredicto_atajo(t)


def test_prueba_atajo_NO_acusa_a_un_modelo_que_mira_la_textura():
    neg, pos = _lotes()
    t = evaluar.prueba_atajo_brillo(pred_contraste_local, neg, pos, deltas=(18, 30, 51))
    assert t["sin_grieta_oscurecidas→con_grieta_%"].max() <= 5, t
    assert t["con_grieta_aclaradas→sin_grieta_%"].max() <= 5, t
    assert "NO es un atajo" in evaluar.veredicto_atajo(t)


def test_prueba_robustez_tabla_completa():
    # grieta FINA y de poco contraste: el caso difícil de verdad
    neg, pos = _lotes(10, grosor=2, oscuridad=50)
    imgs = np.concatenate([neg, pos])
    y = np.r_[np.zeros(10), np.ones(10)]
    t = evaluar.prueba_robustez(pred_contraste_local, imgs, y)
    assert set(t["prueba"]) == set(evaluar.TRANSFORMACIONES)
    assert {"cambian_de_clase_%", "exactitud_%", "recall_%"} <= set(t.columns)
    # a 0,25 de escala la grieta fina se pierde: el recall debe caer respecto de 0,75
    r = t[t["prueba"] == "lejania_escala"].set_index("parametro")["recall_%"]
    assert r[0.25] < r[0.75], r


def test_transformaciones_conservan_forma_y_rango():
    im = textura_concreto(224, brillo=170).astype(np.float32)
    for fn, params in evaluar.TRANSFORMACIONES.values():
        for p in params:
            out = fn(im, p)
            assert out.shape == im.shape, (fn.__name__, p)
            assert out.min() >= 0 and out.max() <= 255
