import json
from pathlib import Path

import cv2
import numpy as np

from src import app_core, predecir, riesgo
from sinteticos import textura_concreto


def test_reducir_lado_largo_conserva_la_proporcion_y_nunca_agranda():
    grande = np.zeros((3000, 4000, 3), np.uint8)
    r = app_core.reducir_lado_largo(grande)
    assert max(r.shape[:2]) == 1280 and abs(r.shape[1] / r.shape[0] - 4000 / 3000) < 0.01
    chica = np.zeros((960, 1280, 3), np.uint8)
    assert app_core.reducir_lado_largo(chica) is chica                      # ya mide 1280: no se toca
    assert app_core.reducir_lado_largo(np.zeros((100, 200, 3), np.uint8)).shape == (100, 200, 3)


def test_el_aviso_usa_los_numeros_reales_si_existen(tmp):
    (Path(tmp) / "results").mkdir()
    (Path(tmp) / "results" / "metricas_fotos_propias.json").write_text(json.dumps(
        {"n": 29, "con_grieta": 13, "sin_grieta": 16, "forzada": {"tp": 3, "fp": 0}}), encoding="utf-8")
    t = app_core.aviso_de_limitaciones(tmp)
    assert "NO significa que no exista" in t and "3 de 13" in t and "0 falsas alarmas en 16" in t


def test_el_aviso_siempre_sale_aunque_no_haya_resultados(tmp):
    t = app_core.aviso_de_limitaciones(tmp)
    assert "NO significa que no exista" in t and "todavía no tiene resultados" in t


def _red_comprimida(p):
    """Una red que, como la real en fotos de celular, da probabilidades bajas pero constantes."""
    return lambda lote: np.full(len(lote), p)


def test_la_sensibilidad_alta_detecta_lo_que_la_estandar_deja_pasar():
    foto = textura_concreto(960, brillo=180)[:, :720]                       # 960x720: se analiza por teselas
    foto = np.stack([foto[..., 0]] * 3, axis=2) if foto.ndim == 3 else foto
    s = predecir.Sistema(_red_comprimida(0.12))                             # 0,12: por debajo de 0,5 y de 0,2
    estandar = app_core.analizar(s, foto, "Estándar (umbral 0,50)")
    alta = app_core.analizar(s, foto, "Alta sensibilidad (umbral 0,10)")
    assert estandar["prob_grieta"] >= 0.0 and estandar["prob_grieta"] < 0.5
    assert "No se detectó grieta" in " ".join(estandar["riesgo"].razones)
    assert "Se detectó una grieta" in " ".join(alta["riesgo"].razones)       # misma red, otro umbral


def test_con_el_umbral_alto_una_probabilidad_intermedia_sigue_siendo_duda():
    foto = textura_concreto(500)[:, :400]
    s = predecir.Sistema(_red_comprimida(0.5))
    r = app_core.analizar(s, np.stack([foto[..., 0]] * 3, axis=2), "Estándar (umbral 0,50)")
    assert r["riesgo"].nivel == "INDETERMINADO"                              # 0,5 cae en la banda 0,2–0,8


def test_sin_banda_en_sensibilidad_alta_nunca_hay_indeterminado_por_duda():
    foto = np.stack([textura_concreto(500)[:, :400, 0]] * 3, axis=2)
    for p in (0.15, 0.5, 0.9):
        r = app_core.analizar(predecir.Sistema(_red_comprimida(p)), foto, "Muy alta sensibilidad (umbral 0,02)")
        assert r["riesgo"].nivel != "INDETERMINADO", p


def test_la_escala_se_corrige_al_reducir_la_foto():
    """Una tarjeta de 85,6 mm que mide 800 px en una foto de 4000 px: al reducir a 1280, mide 256 px."""
    foto = np.full((3000, 4000, 3), 180, np.uint8)
    cv2.line(foto, (500, 100), (700, 2900), (30, 30, 30), 40)                # grieta de 40 px en la foto original
    s = predecir.Sistema(_red_comprimida(0.9))
    r = app_core.analizar(s, foto, "Estándar (umbral 0,50)", ref_mm=85.6, ref_px=800)
    mm_por_px_original = 85.6 / 800
    ancho_px_reducida = r["medicion"]["ancho_p95_px"]
    ancho_real_mm = 40 * mm_por_px_original                                  # ≈ 4,28 mm
    assert abs(r["medicion"]["ancho_p95_mm"] - ancho_real_mm) / ancho_real_mm < 0.25, (r["medicion"], ancho_real_mm)
    assert r["foto"].shape[1] == 1280 and ancho_px_reducida < 40            # la foto sí se redujo


def test_sin_ancho_conocido_una_grieta_detectada_ya_no_es_riesgo_bajo():
    ev = riesgo.evaluar_riesgo(0.97, ancho_mm=None, inclinacion_deg=None)
    assert ev.nivel == "MEDIO" and any("asume" in r for r in ev.razones)
    assert "Ancho desconocido" in riesgo.reglas_markdown()


def test_elemento_no_se_pasa_al_riesgo():
    foto = np.stack([textura_concreto(500)[:, :400, 0]] * 3, axis=2)
    s = predecir.Sistema(_red_comprimida(0.9))
    a = app_core.analizar(s, foto, "Estándar (umbral 0,50)", elemento="No sé")
    b = app_core.analizar(s, foto, "Estándar (umbral 0,50)", elemento="columna")
    assert a["riesgo"].puntaje < b["riesgo"].puntaje


def test_el_aviso_lee_el_diagnostico_de_6d_con_la_configuracion_de_la_app(tmp):
    """Con los números REALES del equipo: la columna p_esc_0.5 da 3 de 13 y 0 de 16 (no el 0 de 13 de la puerta)."""
    from test_entrega import _diag_real
    d = _diag_real().rename(columns={"p_esc_0_5": "p_esc_0.5", "p_esc_0_25": "p_esc_0.25"})
    (Path(tmp) / "results").mkdir()
    d.to_csv(Path(tmp) / "results" / "diagnostico_fotos_propias.csv", index=False)
    (Path(tmp) / "results" / "metricas_fotos_propias.json").write_text(json.dumps(       # el JSON viejo, con puerta: 0 de 13
        {"n": 29, "con_grieta": 13, "sin_grieta": 16, "forzada": {"tp": 0, "fp": 0}}), encoding="utf-8")
    t = app_core.aviso_de_limitaciones(tmp)
    assert "3 de 13" in t and "0 falsas alarmas en 16" in t and "29 fotos" in t, t
    assert "0 de 13" not in t, "no debe usar el resultado viejo con puerta"
