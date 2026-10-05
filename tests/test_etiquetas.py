from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import datos, etiquetas as et, fotos_propias, predecir
from sinteticos import textura_concreto


def _csv(tmp, texto, codificacion="utf-8"):
    ruta = Path(tmp) / "etiquetas.csv"
    ruta.write_bytes(texto.encode(codificacion))
    return ruta


def test_excel_en_espanol_punto_y_coma_con_BOM():
    # lo que guarda Excel en español: ; como separador, BOM al inicio, saltos de línea de Windows
    texto = "\ufeffarchivo;etiqueta\r\nfoto (1).jpg;1\r\nfoto (2).jpg;0\r\n;\r\n"
    d = Path(__import__("tempfile").mkdtemp())
    assert et.leer_etiquetas(_csv(d, texto)) == {"foto (1)": 1, "foto (2)": 0}


def test_coma_normal_y_tabulacion(tmp):
    assert et.leer_etiquetas(_csv(tmp, "archivo,etiqueta\nfoto (1).jpg,1\n")) == {"foto (1)": 1}
    assert et.leer_etiquetas(_csv(tmp, "archivo\tetiqueta\nfoto (1).jpg\t0\n")) == {"foto (1)": 0}


def test_codificacion_windows_1252_con_ene_y_tildes(tmp):
    texto = "archivo;etiqueta\npared_baño.jpg;0\nfoto_grieta_ñ.jpg;1\n"
    assert et.leer_etiquetas(_csv(tmp, texto, "cp1252")) == {"pared_baño": 0, "foto_grieta_ñ": 1}


def test_nombres_sin_extension_mayusculas_y_espacios(tmp):
    r = et.leer_etiquetas(_csv(tmp, "archivo;etiqueta\nFoto (1);1\n  FOTO (2).JPG ;0\n"))
    assert r == {"foto (1)": 1, "foto (2)": 0}
    assert et.clave("foto (1).jpg") == et.clave("FOTO (1).JPG") == et.clave("foto (1)")


def test_etiquetas_en_texto(tmp):
    r = et.leer_etiquetas(_csv(tmp, "foto;clase\na.jpg;Con grieta\nb.jpg;sin grieta\nc.jpg;SI\nd.jpg;no\n"))
    assert r == {"a": 1, "b": 0, "c": 1, "d": 0}


def test_columnas_incorrectas_explican_que_vio(tmp):
    try:
        et.leer_etiquetas(_csv(tmp, "nombre_raro;valor_raro\na.jpg;1\n"))
    except ValueError as e:
        assert "nombre_raro" in str(e) and "etiqueta" in str(e)
        return
    raise AssertionError("debería explicar las columnas")


def test_etiqueta_incomprensible_se_rechaza_con_la_foto(tmp):
    try:
        et.leer_etiquetas(_csv(tmp, "archivo;etiqueta\na.jpg;2\nb.jpg;quizás\n"))
    except ValueError as e:
        assert "a.jpg" in str(e) and "1 = con grieta" in str(e)
        return
    raise AssertionError("debería rechazar etiquetas raras")


def test_misma_foto_con_dos_etiquetas_distintas(tmp):
    try:
        et.leer_etiquetas(_csv(tmp, "archivo;etiqueta\na.jpg;1\nA.JPG;0\n"))
    except ValueError as e:
        assert "distintas" in str(e)
        return
    raise AssertionError("debería detectar el conflicto")


def test_revisar_avisa_de_fotos_sin_etiqueta_y_etiquetas_sin_foto(tmp, capsys=None):
    for n in ("foto (1).jpg", "foto (2).jpg", "foto (3).jpg"):
        cv2.imwrite(str(Path(tmp) / n), textura_concreto(64))
    rep = et.revisar(tmp, {"foto (1)": 1, "foto (2)": 0, "foto (9)": 1})
    assert rep["fotos"] == 3 and rep["etiquetadas"] == 2
    assert rep["fotos_sin_etiqueta"] == ["foto (3).jpg"] and rep["etiquetas_sin_foto"] == ["foto (9)"]


def test_de_punta_a_punta_con_el_csv_de_excel_y_nombres_como_los_tuyos(tmp):
    """Fotos 'foto (1).jpg', 'foto (2).jpg'... y el CSV como lo exporta Excel en español."""
    pred = lambda lote: np.array([float((im.mean(axis=2) < 100).mean() > 0.004) for im in lote])
    filas = []
    for i in range(1, 7):
        img = textura_concreto(224, semilla=i, brillo=185, ruido=14)
        if i % 2:
            from sinteticos import con_grieta
            img = con_grieta(img, grosor=3, semilla=i, oscuridad=110)
        cv2.imwrite(str(Path(tmp) / f"foto ({i}).jpg"), img)
        filas.append(f"foto ({i}).jpg;{i % 2}")
    (Path(tmp) / "etiquetas.csv").write_bytes(("\ufeffarchivo;etiqueta\r\n" + "\r\n".join(filas) + "\r\n").encode("utf-8"))
    t = fotos_propias.analizar_carpeta(predecir.Sistema(pred), tmp, etiquetas=Path(tmp) / "etiquetas.csv")
    assert len(t) == 6 and t["etiqueta"].notna().all()
    m = fotos_propias.metricas_fotos_propias(t)
    assert m["n"] == 6 and m["con_grieta"] == 3


def test_auditoria_avisa_si_hay_pocas_escenas():
    df = pd.DataFrame({
        "fuente": ["surface_crack"] * 8 + ["negativos_dificiles"] * 4,
        "etiqueta": [0, 1] * 4 + [0] * 4,
        "grupo": [f"b{i}" for i in range(8)] + ["x|0|pared_habitaciones"] * 2 + ["x|0|pared_baño"] * 2})
    _, avisos = datos.auditoria_fuentes(df)
    textos = " ".join(avisos)
    assert "solo 2 escena" in textos and "negativos_dificiles" in textos
    # con 7 escenas ya no avisa de eso
    df2 = df.copy()
    df2.loc[df2.fuente == "negativos_dificiles", "grupo"] = [f"e{i}" for i in range(4)]
    df2 = pd.concat([df2, pd.DataFrame({"fuente": ["negativos_dificiles"] * 4, "etiqueta": [0] * 4,
                                        "grupo": [f"e{i}" for i in range(4, 8)]})], ignore_index=True)
    assert "escena" not in " ".join(datos.auditoria_fuentes(df2)[1])
