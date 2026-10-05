import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sinteticos import textura_concreto, con_grieta
from test_ancho import rect_exacto
from src import ancho, complejidad, fotos_propias, informe, predecir, riesgo, verificar


# ====================== COMPLEJIDAD ======================
class _Peso:
    def __init__(self, *forma): self.shape = forma

class _ModeloFalso:
    trainable_weights = [_Peso(3, 3, 3, 8), _Peso(8)]
    def count_params(self): return 3 * 3 * 3 * 8 + 8 + 100          # 100 no entrenables


def test_contar_parametros():
    p = complejidad.contar_parametros(_ModeloFalso())
    assert p == {"parametros": 324, "entrenables": 224, "no_entrenables": 100}


def test_medir_tiempo_mide_de_verdad_y_descarta_el_calentamiento():
    llamadas = []
    def fn():
        llamadas.append(1)
        time.sleep(0.004)
    t = complejidad.medir_tiempo(fn, n=20, calentamiento=5)
    assert len(llamadas) == 25, "5 de calentamiento + 20 medidas"
    assert 3.5 < t["ms_mediana"] < 12, t
    assert t["ms_p95"] >= t["ms_mediana"]


def test_el_tiempo_distingue_un_modelo_lento_de_uno_rapido():
    rapido = complejidad.medir_tiempo(lambda: time.sleep(0.002), n=15)["ms_mediana"]
    lento = complejidad.medir_tiempo(lambda: time.sleep(0.010), n=15)["ms_mediana"]
    assert lento > 2.5 * rapido


def test_tamano_mb(tmp):
    (tmp / "a.bin").write_bytes(b"x" * 2_500_000)
    assert abs(complejidad.tamano_mb(tmp / "a.bin") - 2.5) < 1e-9


def test_fidelidad_detecta_un_cambio_de_clase():
    x = np.zeros((4, 1))
    ok = complejidad.fidelidad_tflite(lambda _: np.array([.9, .8, .1, .2]), lambda _: np.array([.91, .79, .12, .2]), x)
    assert ok["cambian_de_clase"] == 0 and ok["dif_max"] < 0.03
    mal = complejidad.fidelidad_tflite(lambda _: np.array([.9, .8, .1, .2]), lambda _: np.array([.9, .4, .1, .2]), x)
    assert mal["cambian_de_clase"] == 1


def test_fila_y_tabla_de_complejidad(tmp):
    (tmp / "m.tflite").write_bytes(b"0" * 1_000_000)
    from unittest import mock
    with mock.patch.object(complejidad, "tiempo_tflite", return_value={"ms_mediana": 6.7, "ms_p95": 9.1}):
        fila = complejidad.fila_complejidad("v2", modelo_keras=_ModeloFalso(), ruta_tflite=tmp / "m.tflite")
    t = complejidad.tabla_complejidad([fila])
    assert {"parametros", "entrenables", "mb_tflite", "ms_tflite_1img"} <= set(t.columns)
    assert t.loc[0, "mb_tflite"] == 1.0 and t.loc[0, "ms_tflite_1img"] == 6.7


# ====================== ORIENTACIÓN ======================
def test_orientacion_vertical_horizontal_diagonal():
    esperado = {0: "horizontal", 90: "vertical", 45: "diagonal", 135: "diagonal", 5: "horizontal", 85: "vertical"}
    for ang, cat in esperado.items():
        r = ancho.orientacion_grieta(rect_exacto(5, ang))
        assert r["categoria"] == cat, (ang, r)
        # rect_exacto dibuja con el eje y hacia ABAJO (imagen); orientacion_grieta usa y hacia ARRIBA
        esperado_math = (180 - ang) % 180
        assert min(abs(r["angulo_deg"] - esperado_math), 180 - abs(r["angulo_deg"] - esperado_math)) < 4, (ang, r)
        assert r["elongacion"] > 0.9


def test_orientacion_con_grieta_ramificada_gana_la_dominante():
    m = rect_exacto(5, 90)                                   # tronco vertical largo
    cv2.line(m, (250, 250), (290, 290), 1, 3)                # rama corta
    assert ancho.orientacion_grieta(m)["categoria"] == "vertical"


def test_orientacion_se_corrige_con_el_giro_del_celular():
    """Grieta vertical en el mundo + celular girado 15°: en la imagen se ve inclinada; con el roll vuelve a vertical."""
    m = rect_exacto(5, 90).astype(np.uint8) * 255
    giro = cv2.warpAffine(m, cv2.getRotationMatrix2D((250, 250), 15, 1.0), (500, 500)) > 127
    en_imagen = ancho.orientacion_grieta(giro.astype(np.uint8))
    corregida = ancho.orientacion_grieta(giro.astype(np.uint8), roll_deg=15)
    assert en_imagen["categoria"] == "diagonal" or abs(en_imagen["angulo_deg"] - 90) > 10
    assert corregida["categoria"] == "vertical", corregida


def test_orientacion_sin_grieta_devuelve_none():
    assert ancho.orientacion_grieta(np.zeros((100, 100), np.uint8)) is None


# ====================== RIESGO AVANZADO ======================
def test_riesgo_diagonal_y_columna_suben_el_puntaje():
    base = riesgo.evaluar_riesgo(0.97, ancho_mm=0.5, inclinacion_deg=0.2)
    diag = riesgo.evaluar_riesgo(0.97, ancho_mm=0.5, inclinacion_deg=0.2, orientacion="diagonal")
    col = riesgo.evaluar_riesgo(0.97, ancho_mm=0.5, inclinacion_deg=0.2, orientacion="diagonal", elemento="columna")
    assert base.puntaje < diag.puntaje < col.puntaje
    assert any("columna" in r for r in col.razones) and any("diagonal" in r for r in col.razones)


def test_orientacion_y_elemento_no_cuentan_si_no_hay_grieta():
    a = riesgo.evaluar_riesgo(0.02, orientacion="diagonal", elemento="columna")
    assert a.nivel == "BAJO" and a.puntaje == 0


def test_reglas_markdown_refleja_las_constantes_del_codigo():
    md = riesgo.reglas_markdown()
    assert str(riesgo.ANCHO_BANDAS_MM[0]) in md and "diagonal" in md and "INDETERMINADO" in md
    assert str(riesgo.FACTOR_CRITICO_INCL_DEG) in md


# ====================== FOTOS PROPIAS ======================
def _pred_oscuro(lote):
    return np.array([float((im.mean(axis=2) < 100).mean() > 0.004) for im in lote])

class _EmbF:
    def __call__(self, lote):
        return np.array([[0, float(np.percentile(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2)), 25))]
                         for g in (im.mean(axis=2).astype(np.float32) for im in lote)])

class _PuertaF:
    def es_concreto(self, emb): return emb[:, 1] > 0.3


def _carpeta_fotos(tmp):
    etiquetas = {}
    for i in range(4):
        cv2.imwrite(str(tmp / f"grieta{i}.jpg"), con_grieta(textura_concreto(224, semilla=i, brillo=185, ruido=14),
                                                           grosor=3, semilla=i, oscuridad=110))
        etiquetas[f"grieta{i}.jpg"] = 1
        cv2.imwrite(str(tmp / f"sana{i}.jpg"), textura_concreto(224, semilla=50 + i, brillo=185, ruido=14))
        etiquetas[f"sana{i}.jpg"] = 0
    lisa = np.full((224, 224, 3), 205, np.uint8)                       # pared pintada: el caso difícil
    cv2.ellipse(lisa, (110, 120), (30, 70), 20, 0, 360, (45, 45, 45), -1)
    cv2.imwrite(str(tmp / "pared_con_mano.jpg"), lisa)
    etiquetas["pared_con_mano.jpg"] = 0
    return etiquetas


def test_fotos_propias_sin_puerta_el_falso_positivo_se_ve_en_las_metricas(tmp):
    et = _carpeta_fotos(tmp)
    s = predecir.Sistema(_pred_oscuro)
    t = fotos_propias.analizar_carpeta(s, tmp, etiquetas=et, salida_csv=tmp / "out" / "r.csv")
    m = fotos_propias.metricas_fotos_propias(t)
    assert (tmp / "out" / "r.csv").exists() and len(t) == 9
    assert m["forzada"]["recall"] == 1.0
    assert m["forzada"]["fp"] >= 1, "la pared con mano debe aparecer como falso positivo"


def test_fotos_propias_con_puerta_se_abstiene_y_lo_reporta(tmp):
    et = _carpeta_fotos(tmp)
    s = predecir.Sistema(_pred_oscuro, embeddings=_EmbF(), puerta=_PuertaF())
    t = fotos_propias.analizar_carpeta(s, tmp, etiquetas=et)
    m = fotos_propias.metricas_fotos_propias(t)
    assert t.set_index("archivo").loc["pared_con_mano.jpg", "riesgo"] == "INDETERMINADO"
    assert m["con_abstencion"]["falsos_pos_en_respondidas"] == 0
    assert m["con_abstencion"]["grietas_reales_sin_diagnostico"] == 0, "no debe esconder grietas reales"
    assert 0 < m["con_abstencion"]["cobertura"] < 1
    assert {"orientacion", "ancho_px", "inclinacion_deg", "riesgo"} <= set(t.columns)


def test_fotos_propias_etiquetas_desde_csv_y_parametros_por_foto(tmp):
    et = _carpeta_fotos(tmp)
    pd.DataFrame({"archivo": list(et), "etiqueta": list(et.values())}).to_csv(tmp / "etiquetas.csv", index=False)
    s = predecir.Sistema(_pred_oscuro)
    t = fotos_propias.analizar_carpeta(s, tmp, etiquetas=tmp / "etiquetas.csv",
                                       parametros={"grieta0.jpg": {"mm_por_px": 0.2, "elemento": "columna"}})
    fila = t.set_index("archivo").loc["grieta0.jpg"]
    assert fila["ancho_mm"] is not None and fila["ancho_mm"] > 0
    assert t["etiqueta"].notna().all()


def test_fotos_propias_sin_etiquetas_avisa(tmp):
    cv2.imwrite(str(tmp / "a.jpg"), textura_concreto(224))
    t = fotos_propias.analizar_carpeta(predecir.Sistema(_pred_oscuro), tmp)
    try:
        fotos_propias.metricas_fotos_propias(t)
    except ValueError as e:
        assert "etiquetas" in str(e)
        return
    raise AssertionError("debería exigir etiquetas")


# ====================== VERIFICADOR ======================
def _crear(raiz, ruta, texto="x"):
    p = Path(raiz) / ruta
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding="utf-8")


def test_verificador_no_marca_nada_en_un_repo_vacio(tmp):
    t = verificar.estado(tmp)
    assert (t["estado"] == "✅").sum() == 0
    assert set(t["estado"]) <= {"❌", "🧍"}
    assert len(t) == 23 and set(t["corte"]) == {"Entrega 1", "Entrega 2", "Entrega 3"}


def test_verificador_marca_solo_lo_que_tiene_evidencia(tmp):
    _crear(tmp, "results/linea_base.csv", "metodo,exactitud\na,0.6\n")
    _crear(tmp, "results/complejidad.csv", "modelo,parametros,mb_tflite,ms_tflite_1img\nv2,2259265,4.46,6.7\n")
    t = verificar.estado(tmp).set_index("id")
    assert t.loc["1.4", "estado"] == "✅" and t.loc["2.6", "estado"] == "✅"
    assert t.loc["2.3", "estado"] == "❌" and "modelo.keras" in t.loc["2.3", "detalle"]
    assert t.loc["1.7", "estado"] == "🧍" and t.loc["3.8", "estado"] == "🧍"


def test_verificador_rechaza_columnas_faltantes_y_marcadores_pendientes(tmp):
    _crear(tmp, "results/complejidad.csv", "modelo,parametros\nv2,1\n")                # faltan columnas
    _crear(tmp, "PLAN_DE_TRABAJO.md", "| `[COMPLETAR]` | rol |\n| `[COMPLETAR]` | rol |")
    t = verificar.estado(tmp).set_index("id")
    assert t.loc["2.6", "estado"] == "❌" and "mb_tflite" in t.loc["2.6", "detalle"]
    assert t.loc["1.6", "estado"] == "❌" and "2 marcador" in t.loc["1.6", "detalle"]
    _crear(tmp, "PLAN_DE_TRABAJO.md", "| Ana | Datos |")
    assert verificar.estado(tmp).set_index("id").loc["1.6", "estado"] == "✅"


def test_verificador_comodines_y_texto(tmp):
    _crear(tmp, "models/v2_mobilenet/meta.json", "{}")
    _crear(tmp, "INFORME_TECNICO.md", "## Estado del arte\n### Clases a predecir\n")
    t = verificar.estado(tmp).set_index("id")
    assert t.loc["2.1", "estado"] == "✅" and t.loc["1.3", "estado"] == "✅" and t.loc["1.1", "estado"] == "✅"


def test_verificador_escribe_el_resumen_y_la_rubrica(tmp):
    t = verificar.main(tmp)
    md = (Path(tmp) / "results" / "estado_entregas.md").read_text(encoding="utf-8")
    assert "Estado de las entregas" in md and "Rúbrica" in md
    r = verificar.resumen_rubrica(t)
    assert r["peso_%"].sum() == 100 and len(r) == 5


# ====================== INFORME ======================
def test_informe_sin_resultados_dice_pendiente_y_siempre_trae_las_reglas(tmp):
    texto = informe.generar(tmp)
    assert texto.count("PENDIENTE") >= 5
    assert "Reglas del motor de riesgo" in texto and "diagonal" in texto


def test_informe_toma_los_numeros_de_los_csv(tmp):
    _crear(tmp, "results/complejidad.csv", "modelo,parametros,mb_tflite\nmobilenetv2,2259265,4.46\n")
    pd.DataFrame({"delta_brillo": [18, 30], "sin_grieta_oscurecidas→con_grieta_%": [0.0, 1.0],
                  "con_grieta_aclaradas→sin_grieta_%": [0.5, 0.0]}).to_csv(tmp / "results" / "prueba_atajo_brillo.csv", index=False)
    _crear(tmp, "results/metricas_fotos_propias.json", json.dumps({
        "n": 10, "con_grieta": 5, "sin_grieta": 5,
        "forzada": {"exactitud": .9, "precision": .8, "recall": 1.0, "f1": .89, "tp": 5, "tn": 4, "fp": 1, "fn": 0},
        "con_abstencion": {"cobertura": .8, "exactitud_en_respondidas": .95, "grietas_reales_sin_diagnostico": 1}}))
    texto = informe.generar(tmp)
    assert "2,259,265" in texto or "2259265" in texto
    assert "NO es un atajo" in texto
    assert "grietas reales sin diagnóstico: 1" in texto
    informe.main(tmp)
    assert (Path(tmp) / "results" / "secciones_informe_v2.md").exists()


# ====================== DIAGNÓSTICO DE FOTOS PROPIAS (con los números REALES del equipo) ======================
def _diag_real():
    sanas = [(0, 0, 0, 0)] * 3 + [(0, .002, 0, 0), (.014, .005, .007, .002)] + [(0, 0, 0, 0)] * 4 + [(.001, .002, 0, 0)] + \
            [(0, 0, 0, 0)] * 4 + [(.001, 0, 0, 0), (.002, .004, .001, .004)]
    grietas = [(.070, .195, .163, .135), (.047, .146, .166, .146), (.261, .993, .553, 0), (.004, .002, .020, .002), (0, 0, 0, 0),
               (.642, .993, .941, 0), (0, .018, .001, 0), (0, .015, .002, .001), (0, .005, 0, .007), (0, 0, 0, 0),
               (.006, .021, .021, 0), (.008, .135, .011, .001), (.999, .999, 1.0, .002)]
    filas = [dict(etiqueta=0, p_entera=a, p_esc_0_5=b, p_esc_0_25=c, p_sistema=d) for a, b, c, d in sanas] + \
            [dict(etiqueta=1, p_entera=a, p_esc_0_5=b, p_esc_0_25=c, p_sistema=d) for a, b, c, d in grietas]
    return pd.DataFrame(filas)


def test_resumen_diagnostico_reproduce_los_numeros_del_equipo():
    from src import evaluar
    r = evaluar.resumen_diagnostico(_diag_real()).set_index("metodo")
    assert abs(r.loc["p_entera", "AUC"] - 0.75) < 0.01 and abs(r.loc["p_esc_0_5", "AUC"] - 0.89) < 0.01
    assert abs(r.loc["p_esc_0_25", "AUC"] - 0.86) < 0.01 and abs(r.loc["p_sistema", "AUC"] - 0.71) < 0.01
    assert [r.loc[m, "grietas_detectadas"] for m in ("p_entera", "p_esc_0_5", "p_esc_0_25", "p_sistema")] == ["2/13", "3/13", "3/13", "0/13"]
    assert r.loc["p_esc_0_5", "grietas_sobre_todas_las_sanas"] == "9/13" and r.loc["p_esc_0_5", "falsos_positivos"] == "0/16"
    # la puerta EMPEORA el resultado: el sistema con puerta detecta menos que la red sola
    assert r.loc["p_sistema", "AUC"] < r.loc["p_esc_0_5", "AUC"]


def test_el_informe_incluye_el_diagnostico_marcado_como_exploratorio(tmp):
    (Path(tmp) / "results").mkdir()
    _diag_real().to_csv(Path(tmp) / "results" / "diagnostico_fotos_propias.csv", index=False)
    texto = informe.generar(tmp)
    assert "EXPLORATORIO" in texto and "AUC" in texto and "0.887" in texto.replace(",", ".")
