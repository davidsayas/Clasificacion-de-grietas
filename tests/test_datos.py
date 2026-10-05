import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from sinteticos import textura_concreto, guardar_imagenes
from src import datos, eda


def _crear_dataset(tmp, n_fotos=30, por_foto=12):
    """Simula Surface Crack: cada 'foto' tiene un tinte propio; sus recortes son
    archivos CONSECUTIVOS. Así la hipótesis de los bloques se cumple a propósito."""
    rng = np.random.default_rng(0)
    for clase in ("Positive", "Negative"):
        carpeta = Path(tmp) / clase
        carpeta.mkdir(parents=True)
        k = 0
        for foto in range(n_fotos):
            tinte = tuple(int(x) for x in rng.integers(-25, 25, 3))
            brillo = int(rng.integers(140, 200))
            for _ in range(por_foto):
                k += 1
                img = textura_concreto(64, semilla=k, brillo=brillo, sesgo_color=tinte)
                cv2.imwrite(str(carpeta / f"{k:05d}.jpg"), img)
    return {"sc": {"con_grieta": Path(tmp) / "Positive", "sin_grieta": Path(tmp) / "Negative",
                   "grupos": "bloque", "tam_bloque": por_foto}}


def test_inventario_y_orden(tmp):
    df = datos.construir_inventario(_crear_dataset(tmp))
    assert len(df) == 2 * 30 * 12
    assert set(df["etiqueta"]) == {0, 1}
    pos = df[df["etiqueta"] == 1]["ruta"].tolist()
    assert pos == sorted(pos), "el inventario debe estar ordenado"
    assert df["grupo"].nunique() == 2 * 30


def test_division_sin_fuga_y_proporciones(tmp):
    df = datos.construir_inventario(_crear_dataset(tmp))
    tr, va, te = datos.dividir(df)
    assert len(tr) + len(va) + len(te) == len(df)
    datos.verificar_sin_fuga(tr, va, te)           # no debe lanzar
    assert abs(len(te) / len(df) - 0.15) < 0.04
    assert abs(len(va) / len(df) - 0.15) < 0.04


def test_verificador_detecta_fuga(tmp):
    df = datos.construir_inventario(_crear_dataset(tmp))
    a, b = df.iloc[:100], df.iloc[50:150]          # se solapan a propósito
    try:
        datos.verificar_sin_fuga(a, b)
    except AssertionError:
        return
    raise AssertionError("debería haber detectado la fuga")


def test_dividir_imagen_por_imagen_si_fuga_de_hermanos(tmp):
    """Demuestra POR QUÉ importa: con split al azar por imagen, hermanos caen en
    train y test; con split por grupo, nunca."""
    from sklearn.model_selection import train_test_split
    df = datos.construir_inventario(_crear_dataset(tmp))
    tr, te = train_test_split(df, test_size=0.15, random_state=42)
    compartidos = set(tr["grupo"]) & set(te["grupo"])
    assert len(compartidos) > 10, "el split al azar debería mezclar hermanos"
    tr2, va2, te2 = datos.dividir(df)
    assert not (set(tr2["grupo"]) & set(te2["grupo"]))


def test_prueba_adyacencia_confirma_bloques(tmp):
    df = datos.construir_inventario(_crear_dataset(tmp))
    rasgos = datos.extraer_rasgos(df)
    r = datos.prueba_adyacencia(df, rasgos)
    assert r["razon"] < 0.7, r


def test_prueba_adyacencia_rechaza_numeracion_aleatoria(tmp):
    """Si los archivos se barajan, la numeración NO sigue a la foto: la razón ~1."""
    df = datos.construir_inventario(_crear_dataset(tmp))
    rasgos = datos.extraer_rasgos(df)
    rng = np.random.default_rng(1)
    barajado = df.copy()
    for (f, e), sub in df.groupby(["fuente", "etiqueta"]):
        barajado.loc[sub.index, "orden"] = rng.permutation(len(sub))
    r = datos.prueba_adyacencia(barajado, rasgos)
    assert r["razon"] > 0.85, r


def test_cluster_agrupa_hermanos(tmp):
    df = datos.construir_inventario(_crear_dataset(tmp))
    rasgos = datos.extraer_rasgos(df)
    con_cluster = datos.asignar_grupos_cluster(df, rasgos, imagenes_por_grupo=12)
    # pureza: casi todos los recortes de un mismo cluster deberían ser de la misma foto
    foto = df["grupo"].to_numpy()
    pureza = []
    for c, sub in con_cluster.groupby("grupo"):
        vals, cuentas = np.unique(foto[sub.index], return_counts=True)
        pureza.append(cuentas.max() / cuentas.sum())
    assert np.mean(pureza) > 0.7, np.mean(pureza)


def test_auditoria_detecta_fuente_de_una_clase():
    df = pd.DataFrame({"fuente": ["a"] * 10 + ["b"] * 10 + ["dificiles"] * 5,
                       "etiqueta": [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5})
    tabla, avisos = datos.auditoria_fuentes(df)
    assert tabla.loc["dificiles", "con_grieta"] == 0
    assert len(avisos) == 1 and "dificiles" in avisos[0]


def test_reservar_fuente():
    df = pd.DataFrame({"fuente": ["a", "a", "sdnet"], "etiqueta": [0, 1, 1],
                       "ruta": list("xyz"), "grupo": list("xyz")})
    resto, fuera = datos.reservar_fuente(df, "sdnet")
    assert list(fuera["fuente"]) == ["sdnet"] and len(resto) == 2


def test_duplicados(tmp):
    # imagen CON estructura (degradado + bloque oscuro): la huella de 64 bits es estable
    base = np.tile(np.linspace(60, 200, 64, dtype=np.float32), (64, 1))
    base[10:30, 20:50] -= 50
    gris = np.clip(base, 0, 255).astype(np.uint8)
    cv2.imwrite(str(Path(tmp) / "a.png"), gris)
    cv2.imwrite(str(Path(tmp) / "a_mas_clara.png"), np.clip(gris.astype(int) + 20, 0, 255).astype(np.uint8))
    cv2.imwrite(str(Path(tmp) / "otra.png"), gris[::-1].copy())   # distinta: invertida en vertical
    dup = datos.encontrar_duplicados(sorted(Path(tmp).glob("*.png")))
    nombres = [sorted(Path(r).name for r in v) for v in dup.values()]
    assert ["a.png", "a_mas_clara.png"] in nombres, nombres


def test_clasificador_tonto_coincide_con_la_teoria():
    rng = np.random.default_rng(0)
    pos = rng.normal(162, 25, 20000)
    neg = rng.normal(180, 25, 20000)
    r = eda.clasificador_tonto_brillo(pos, neg)
    # teoría: acc = Phi(d/2) con d = 18/25 = 0.72  ->  Phi(0.36) ≈ 0.64
    assert 0.62 < r["exactitud"] < 0.67, r
    assert 168 < r["umbral"] < 175
    assert abs(eda.cohen_d(neg, pos) - 0.72) < 0.03


def test_exactitud_con_umbral_y_brillo_por_imagen(tmp):
    guardar_imagenes(Path(tmp) / "claras", 5, brillo=200)
    guardar_imagenes(Path(tmp) / "oscuras", 5, brillo=100)
    b_c = eda.brillo_por_imagen(sorted((Path(tmp) / "claras").glob("*.jpg")))
    b_o = eda.brillo_por_imagen(sorted((Path(tmp) / "oscuras").glob("*.jpg")))
    assert len(b_c) == 5 and b_c.mean() > 190 and b_o.mean() < 110
    assert eda.exactitud_con_umbral(b_o, b_c, 150) == 1.0
