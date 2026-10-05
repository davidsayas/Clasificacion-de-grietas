import cv2
import numpy as np
import pandas as pd

from sinteticos import textura_concreto, con_grieta
from src import inclinacion, mosaico, ood, riesgo, linea_base


# ====================== INCLINACIÓN ======================
def _muro(tilt_deg, roll_celular_deg, n=500):
    """Muro con N juntas verticales inclinadas `tilt_deg` (+ = arriba a la derecha),
    fotografiado con el celular girado `roll_celular_deg` (+ = horario)."""
    img = np.full((n, n), 200, np.uint8)
    t = np.tan(np.radians(tilt_deg))
    for xb in range(60, n - 40, 70):
        cv2.line(img, (int(xb), n - 1), (int(xb + (n - 1) * t), 0), 40, 3)
    # celular girado en sentido horario `roll` => el mundo se ve girado en sentido antihorario
    # (cv2: ángulo positivo = antihorario)
    m = cv2.getRotationMatrix2D((n / 2, n / 2), roll_celular_deg, 1.0)
    return cv2.warpAffine(img, m, (n, n), borderValue=200)


def test_inclinacion_muro_a_plomo():
    r = inclinacion.estimar_inclinacion(_muro(0, 0))
    assert abs(r["angulo_imagen_deg"]) < 0.6, r


def test_inclinacion_conocida():
    for tilt in (2.0, -3.0, 5.0):
        r = inclinacion.estimar_inclinacion(_muro(tilt, 0))
        assert abs(r["angulo_imagen_deg"] - tilt) < 0.7, (tilt, r)


def test_gravedad_separa_el_giro_del_celular_del_desplome_del_muro():
    """Caso clave: muro A PLOMO + celular girado 10°. Sin corrección parece un
    desplome de 10°; con el roll del sensor, ~0°."""
    foto = _muro(0, 10)
    medido = inclinacion.estimar_inclinacion(foto)["angulo_imagen_deg"]
    assert abs(medido) > 8, "el giro del celular SÍ aparece como falso desplome"
    real = inclinacion.corregir_con_gravedad(medido, roll_celular_deg=10)["inclinacion_real_deg"]
    assert abs(real) < 1.0, real
    # y con un desplome real de 2.5° y celular girado -7°
    foto2 = _muro(2.5, -7)
    m2 = inclinacion.estimar_inclinacion(foto2)["angulo_imagen_deg"]
    real2 = inclinacion.corregir_con_gravedad(m2, roll_celular_deg=-7)["inclinacion_real_deg"]
    assert abs(real2 - 2.5) < 1.0, (m2, real2)


def test_pitch_alto_advierte():
    r = inclinacion.corregir_con_gravedad(1.0, 0.0, pitch_celular_deg=20)
    assert not r["confiable"] and r["avisos"]


def test_pared_lisa_no_inventa_inclinacion():
    liso = np.full((300, 300), 190, np.uint8)
    assert inclinacion.estimar_inclinacion(liso) is None


# ====================== MOSAICO ======================
def _pred_oscuro(lote):
    """Modelo falso: 'grieta' si hay una banda oscura (>0.3 % de píxeles muy oscuros)."""
    return np.array([float((im.mean(axis=2) < 110).mean() > 0.003) for im in lote])


def test_mosaico_localiza_la_grieta_y_cubre_toda_la_foto():
    foto = np.full((700, 900, 3), 185, np.uint8)
    cv2.line(foto, (700, 420), (790, 560), (50, 50, 50), 5)       # grieta en la zona inferior derecha
    r = mosaico.clasificar_mosaico(foto, _pred_oscuro, lado=224, solape=0.5)
    assert r["mapa"].shape == (700, 900)
    assert r["prob_max"] >= 0.5 and r["fraccion"] > 0
    assert r["mapa"][500, 740] >= 0.5, "debe marcar la zona de la grieta"
    assert r["mapa"][100, 100] < 0.5, "y NO marcar una zona lisa"
    # toda caja positiva debe contener o tocar la grieta
    for x, y, lado, p in r["cajas"]:
        assert x <= 790 and x + lado >= 700 and y <= 560 and y + lado >= 420


def test_mosaico_posiciones_llegan_al_borde():
    assert mosaico.posiciones(900, 224, 112)[-1] == 900 - 224
    assert mosaico.posiciones(100, 224, 112) == [0]


def test_mosaico_foto_chica_no_falla():
    r = mosaico.clasificar_mosaico(np.full((100, 150, 3), 185, np.uint8), _pred_oscuro, lado=224)
    assert r["n_teselas"] == 1 and r["mapa"].shape == (100, 150)


def test_mosaico_multiescala_encuentra_grieta_gruesa_de_lejos():
    """Una foto grande donde la grieta es MUY ancha para una tesela: a escala 0.25 sí entra."""
    foto = np.full((1600, 2000, 3), 185, np.uint8)
    cv2.rectangle(foto, (900, 0), (1100, 1600), (60, 60, 60), -1)  # banda de 200 px de ancho
    pred_banda = lambda lote: np.array([float(0.05 < (im.mean(axis=2) < 110).mean() < 0.6) for im in lote])
    una = mosaico.clasificar_mosaico(foto, pred_banda, escalas=(1.0,))
    varias = mosaico.clasificar_mosaico(foto, pred_banda, escalas=(1.0, 0.25))
    assert varias["prob_max"] >= una["prob_max"]


# ====================== PUERTA (OOD) ======================
def test_puerta_rechaza_lo_que_no_es_concreto():
    rng = np.random.default_rng(0)
    d = 64
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    concreto = lambda n: rng.normal(size=(n, d)) @ A + 5.0
    otros = rng.normal(size=(500, d)) @ A * 1.0 + 5.0 + rng.normal(size=d) * 1.2   # nube desplazada
    p = ood.PuertaConcreto(percentil=99).ajustar(concreto(4000), concreto(1000))
    rechazo_legitimo = 1 - p.es_concreto(concreto(2000)).mean()
    rechazo_otros = 1 - p.es_concreto(otros).mean()
    assert rechazo_legitimo < 0.03, rechazo_legitimo           # ~1 % esperado
    assert rechazo_otros > 0.9, rechazo_otros
    assert p.auroc(concreto(500), otros) > 0.97


def test_puerta_guardar_y_cargar(tmp):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 16))
    p = ood.PuertaConcreto().ajustar(x)
    p.guardar(tmp / "puerta.npz")
    q = ood.PuertaConcreto.cargar(tmp / "puerta.npz")
    assert np.allclose(p.distancia(x[:5]), q.distancia(x[:5])) and p.umbral == q.umbral


# ====================== RIESGO ======================
def test_riesgo_no_afirma_si_no_es_concreto_o_duda():
    assert riesgo.evaluar_riesgo(0.99, es_concreto=False).nivel == "INDETERMINADO"
    assert riesgo.evaluar_riesgo(0.55).nivel == "INDETERMINADO"       # dentro de la banda incierta


def test_riesgo_es_monotono_con_el_ancho_y_la_inclinacion():
    orden = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}
    niveles = [orden[riesgo.evaluar_riesgo(0.97, ancho_mm=a, inclinacion_deg=0.2).nivel]
               for a in (0.1, 0.5, 1.5, 4.0)]
    assert niveles == sorted(niveles) and niveles[-1] == 2
    niveles = [orden[riesgo.evaluar_riesgo(0.02, inclinacion_deg=i).nivel] for i in (0.1, 1.0, 2.0, 4.0)]
    assert niveles == sorted(niveles) and niveles[0] == 0 and niveles[-1] == 2


def test_riesgo_sin_grieta_ni_inclinacion_es_bajo():
    r = riesgo.evaluar_riesgo(0.01, inclinacion_deg=0.1)
    assert r.nivel == "BAJO" and r.recomendacion


# ====================== LÍNEA BASE ======================
def test_linea_base_bate_al_azar_y_la_logistica_al_brillo(tmp):
    import tempfile
    from pathlib import Path
    rng = np.random.default_rng(0)
    filas = []
    for i in range(160):
        pos = i % 2
        base = textura_concreto(112, semilla=i, brillo=int(rng.normal(172 if pos else 176, 10)))
        img = con_grieta(base, grosor=2, semilla=i, oscuridad=60) if pos else base
        ruta = Path(tmp) / f"{i}.png"
        cv2.imwrite(str(ruta), img)
        filas.append({"ruta": str(ruta), "etiqueta": pos, "grupo": f"g{i}", "fuente": "s"})
    df = pd.DataFrame(filas)
    tr, te = df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)
    t = linea_base.tabla_linea_base(tr, te)
    ex = t.set_index("metodo")["exactitud"]
    # el brillo casi no separa (172 vs 176 con desvío 10); los rasgos de bordes sí
    assert ex["regresión logística (5 rasgos)"] > ex["umbral de brillo"], t
    assert ex["umbral de bordes"] > 0.8, t


def test_mosaico_veto_descarta_tambien_a_las_vecinas():
    """Un objeto en el borde de una tesela deja fragmentos en las vecinas: el margen las descarta."""
    foto = np.full((600, 900, 3), 185, np.uint8)
    foto[300:420, 400:520] = 50                       # objeto oscuro
    pred = lambda lote: np.array([float((im.mean(axis=2) < 110).mean() > 0.003) for im in lote])
    vetar = lambda lote: np.array([float((im.mean(axis=2) < 110).mean() > 0.2) for im in lote]).astype(bool)
    sin_margen = mosaico.clasificar_mosaico(foto, pred, vetar=vetar, margen_veto=0)
    con_margen = mosaico.clasificar_mosaico(foto, pred, vetar=vetar, margen_veto=1)
    sin_veto = mosaico.clasificar_mosaico(foto, pred)
    assert sin_veto["fraccion"] > sin_margen["fraccion"] > con_margen["fraccion"] - 1e-9
    assert con_margen["fraccion"] == 0.0, con_margen["fraccion"]
    assert con_margen["fraccion_vetada"] > 0
