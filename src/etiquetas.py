"""etiquetas.py — Lee `etiquetas.csv` sin importar cómo lo haya guardado Excel.

Excel en español guarda el "CSV" con PUNTO Y COMA (;) en vez de coma, a veces con una marca
invisible al principio del archivo (BOM) y a veces en una codificación que rompe la "ñ" y las
tildes. Un lector estricto falla con un error confuso mucho después. Este lector acepta:

  * separador , ; o tabulación (lo detecta solo)
  * UTF-8 con o sin BOM, o Windows-1252 (la de Excel)
  * nombres de columna: archivo/foto/nombre/imagen  y  etiqueta/clase/label
  * nombres de foto con o sin extensión, en mayúsculas o minúsculas ("Foto (1).JPG" = "foto (1)")
  * etiquetas 1/0, o si/no, o "con grieta"/"sin grieta"

Y AVISA de lo que no cuadra: fotos sin etiqueta y etiquetas sin foto.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
COL_ARCHIVO = ("archivo", "foto", "nombre", "imagen", "file", "filename")
COL_ETIQUETA = ("etiqueta", "clase", "label", "class", "grieta")
VALORES = {"1": 1, "1.0": 1, "si": 1, "sí": 1, "true": 1, "verdadero": 1, "con grieta": 1,
           "con_grieta": 1, "grieta": 1,
           "0": 0, "0.0": 0, "no": 0, "false": 0, "falso": 0, "sin grieta": 0, "sin_grieta": 0}


def clave(nombre) -> str:
    """Nombre normalizado para comparar: minúsculas, sin espacios sobrantes y sin extensión de imagen."""
    n = Path(str(nombre).strip()).name.lower()
    p = Path(n)
    return p.stem if p.suffix in EXT else n


def _decodificar(ruta) -> str:
    crudo = Path(ruta).read_bytes()
    try:
        return crudo.decode("utf-8-sig")             # UTF-8, quitando la marca BOM si la hay
    except UnicodeDecodeError:
        return crudo.decode("cp1252")                # la que usa Excel en Windows


def leer_etiquetas(ruta) -> dict[str, int]:
    """{clave(nombre): 0 | 1}. Lanza un ValueError que EXPLICA qué está mal."""
    texto = _decodificar(ruta)
    primera = texto.splitlines()[0] if texto.strip() else ""
    sep = max((";", ",", "\t"), key=primera.count)
    tabla = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, skipinitialspace=True)
    tabla.columns = [str(c).strip().lower() for c in tabla.columns]

    col_a = next((c for c in COL_ARCHIVO if c in tabla.columns), None)
    col_e = next((c for c in COL_ETIQUETA if c in tabla.columns), None)
    if col_a is None or col_e is None:
        raise ValueError(f"No encuentro las columnas del archivo y de la etiqueta. Mi lectura del archivo "
                         f"dio estas columnas: {list(tabla.columns)}. Deben llamarse 'archivo' y 'etiqueta'.")

    resultado, raras, repetidas = {}, [], []
    for nombre, valor in zip(tabla[col_a], tabla[col_e]):
        if pd.isna(nombre) or str(nombre).strip() == "":
            continue                                   # fila vacía que deja Excel al final
        v = VALORES.get(str(valor).strip().lower())
        if v is None:
            raras.append(f"{nombre} → «{valor}»")
            continue
        k = clave(nombre)
        if k in resultado and resultado[k] != v:
            repetidas.append(str(nombre))
        resultado[k] = v
    if raras:
        raise ValueError("Hay etiquetas que no entiendo (usá 1 = con grieta, 0 = sin grieta): " + "; ".join(raras[:5]))
    if repetidas:
        raise ValueError("La misma foto aparece con etiquetas distintas: " + ", ".join(repetidas[:5]))
    return resultado


def normalizar(etiquetas) -> dict[str, int]:
    """Acepta None, una ruta a un CSV, o un diccionario {nombre: etiqueta}."""
    if etiquetas is None:
        return {}
    if isinstance(etiquetas, (str, Path)):
        return leer_etiquetas(etiquetas)
    return {clave(k): int(v) for k, v in etiquetas.items()}


def revisar(carpeta, etiquetas: dict[str, int]) -> dict:
    """Cruza las fotos de la carpeta con las etiquetas y IMPRIME lo que no cuadra."""
    from .datos import listar_imagenes

    fotos = {clave(p.name): p.name for p in listar_imagenes(carpeta)}
    sin_etiqueta = sorted(fotos[k] for k in fotos if k not in etiquetas)
    sin_foto = sorted(k for k in etiquetas if k not in fotos)
    informe = {"fotos": len(fotos), "etiquetadas": len(fotos) - len(sin_etiqueta),
               "con_grieta": sum(1 for k in fotos if etiquetas.get(k) == 1),
               "sin_grieta": sum(1 for k in fotos if etiquetas.get(k) == 0),
               "fotos_sin_etiqueta": sin_etiqueta, "etiquetas_sin_foto": sin_foto}
    print(f"Fotos propias: {informe['fotos']} · etiquetadas: {informe['etiquetadas']} "
          f"({informe['con_grieta']} con grieta, {informe['sin_grieta']} sin grieta)")
    if sin_etiqueta:
        print(f"  ⚠ {len(sin_etiqueta)} foto(s) SIN etiqueta (no cuentan en las métricas): {sin_etiqueta[:5]}")
    if sin_foto:
        print(f"  ⚠ {len(sin_foto)} etiqueta(s) sin foto (¿nombre mal escrito?): {sin_foto[:5]}")
    return informe
