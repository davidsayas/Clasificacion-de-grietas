"""datos.py — Inventario de imágenes, grupos, división SIN FUGA y auditoría de fuentes.

POR QUÉ EXISTE ESTE ARCHIVO
---------------------------
El 99,93 % de la corrida anterior estaba inflado: 40.000 recortes salen de solo
458 fotografías, y recortes del mismo muro caían en entrenamiento y en prueba.
El modelo "se examinaba con hermanos de lo que ya había visto".

La solución es dividir por GRUPOS (todas las imágenes de un mismo muro/escena
juntas), no imagen por imagen. Este archivo construye esos grupos y verifica
que ningún grupo aparezca en dos particiones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_ESCENAS = 6      # con menos escenas, ninguna cae en validación/prueba y no se puede medir en escenas no vistas


# --------------------------------------------------------------------------
# 1. Inventario
# --------------------------------------------------------------------------
def listar_imagenes(carpeta) -> list[Path]:
    """Todas las imágenes de una carpeta (y subcarpetas), SIEMPRE en orden.

    os.listdir() no garantiza el orden; sorted() sí. Sin orden fijo, los
    resultados cambian de una corrida a otra y no hay reproducibilidad.
    """
    carpeta = Path(carpeta)
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)


def construir_inventario(fuentes: dict) -> pd.DataFrame:
    """Una fila por imagen: ruta, etiqueta (1=con grieta), fuente, grupo, orden.

    `fuentes` describe de dónde sale cada conjunto de imágenes:

        {
          "surface_crack": {"con_grieta": "data/Positive", "sin_grieta": "data/Negative",
                            "grupos": "bloque", "tam_bloque": 44},
          "negativos_dificiles": {"sin_grieta": "data/dificiles", "grupos": "subcarpeta"},
        }

    `grupos` define qué imágenes se consideran "hermanas" y por lo tanto NO
    pueden separarse entre entrenamiento y prueba:
      - "bloque":     archivos consecutivos (hipótesis: misma foto original)
      - "subcarpeta": todas las imágenes de una misma subcarpeta (una escena)
      - "archivo":    cada imagen es su propio grupo (fotos independientes)
    """
    filas = []
    for fuente, cfg in fuentes.items():
        modo = cfg.get("grupos", "bloque")
        tam = int(cfg.get("tam_bloque", config.TAM_BLOQUE))
        for clave, etiqueta in (("con_grieta", 1), ("sin_grieta", 0)):
            carpeta = cfg.get(clave)
            if carpeta is None:
                continue
            carpeta = Path(carpeta)
            for orden, ruta in enumerate(listar_imagenes(carpeta)):
                if modo == "subcarpeta":
                    rel = ruta.parent.relative_to(carpeta)
                    grupo = f"{fuente}|{etiqueta}|{rel}"
                elif modo == "archivo":
                    grupo = f"{fuente}|{etiqueta}|{ruta.stem}"
                else:
                    grupo = f"{fuente}|{etiqueta}|b{orden // tam}"
                filas.append({"ruta": str(ruta), "etiqueta": etiqueta, "fuente": fuente,
                              "grupo": grupo, "orden": orden})
    if not filas:
        raise FileNotFoundError("No se encontró ninguna imagen. Revisá las rutas de `fuentes`.")
    return pd.DataFrame(filas).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Rasgos simples (sirven para agrupar, para detectar vecinos y para la línea base)
# --------------------------------------------------------------------------
def rasgos_simples(ruta, lado: int = 64) -> np.ndarray:
    """7 números por imagen: media RGB (3), desvío RGB (3), densidad de gradiente (1).

    Es una "huella de estilo": recortes de la misma foto comparten luz, color y
    textura, así que tienen rasgos parecidos.
    """
    img = Image.open(ruta).convert("RGB").resize((lado, lado))
    a = np.asarray(img, dtype=np.float32)
    gris = a.mean(axis=2)
    gy, gx = np.gradient(gris)
    borde = float(np.hypot(gx, gy).mean())
    return np.concatenate([a.mean(axis=(0, 1)), a.std(axis=(0, 1)), [borde]])


def extraer_rasgos(df: pd.DataFrame, n_max: int | None = None) -> np.ndarray:
    """Rasgos de todas las imágenes del inventario, en el mismo orden que `df`."""
    rutas = df["ruta"].tolist()
    if n_max:
        rutas = rutas[:n_max]
    return np.stack([rasgos_simples(r) for r in rutas])


# --------------------------------------------------------------------------
# 3. ¿Los archivos consecutivos son de la misma foto?  (verifica la hipótesis)
# --------------------------------------------------------------------------
COLUMNAS_COLOR = slice(0, 3)   # media R, G, B: el "estilo" de la foto (luz y color)


def prueba_adyacencia(df: pd.DataFrame, rasgos: np.ndarray, n_pares: int = 2000,
                      semilla: int = config.SEMILLA, columnas=COLUMNAS_COLOR) -> dict:
    """Compara la distancia entre archivos CONSECUTIVOS con la de pares AL AZAR.

    Si los consecutivos son claramente más parecidos que los pares al azar,
    probablemente salen de la misma foto y partir por bloques es válido.
    Si no hay diferencia, la numeración no sigue a la foto y hay que agrupar
    por otro método (ver asignar_grupos_cluster).

    Devuelve la razón mediana(consecutivos)/mediana(azar):
      ~1.0  -> sin relación (los bloques NO sirven)
      <0.7  -> hay relación (los bloques probablemente sirven)
    """
    rng = np.random.default_rng(semilla)
    rasgos = rasgos[:, columnas]
    z = (rasgos - rasgos.mean(axis=0)) / (rasgos.std(axis=0) + 1e-9)
    adyacentes, azar = [], []
    for _, sub in df.groupby(["fuente", "etiqueta"]):
        idx = sub.sort_values("orden").index.to_numpy()
        if len(idx) < 3:
            continue
        k = min(n_pares, len(idx) - 1)
        i = rng.choice(len(idx) - 1, size=k, replace=False)
        adyacentes.append(np.linalg.norm(z[idx[i]] - z[idx[i + 1]], axis=1))
        j = rng.integers(0, len(idx), size=k)
        l = rng.integers(0, len(idx), size=k)
        azar.append(np.linalg.norm(z[idx[j]] - z[idx[l]], axis=1))
    m_ady = float(np.median(np.concatenate(adyacentes)))
    m_azar = float(np.median(np.concatenate(azar)))
    razon = m_ady / m_azar
    if razon < 0.7:
        veredicto = "Los consecutivos se parecen: partir por BLOQUES es razonable."
    elif razon < 0.9:
        veredicto = "Relación débil: conviene reforzar con agrupación por similitud (cluster)."
    else:
        veredicto = "Sin relación: los bloques NO evitan la fuga. Usá asignar_grupos_cluster."
    return {"mediana_consecutivos": m_ady, "mediana_azar": m_azar, "razon": razon,
            "veredicto": veredicto}


def asignar_grupos_cluster(df: pd.DataFrame, rasgos: np.ndarray, imagenes_por_grupo: int = 44,
                           semilla: int = config.SEMILLA, columnas=COLUMNAS_COLOR) -> pd.DataFrame:
    """Plan B: agrupa por SIMILITUD de rasgos (no por número de archivo).

    Útil si la numeración no sigue a la foto original. Crea ~N/imagenes_por_grupo
    grupos con KMeans sobre los rasgos estandarizados. Sustituye la columna `grupo`.
    Es una aproximación: dos fotos muy parecidas pueden caer en el mismo grupo
    (eso no daña: solo las mantiene juntas).

    Solo se usa el COLOR MEDIO (luz y tinte de la foto). El desvío y el gradiente
    NO sirven para agrupar: cambian según haya grieta o no, no según la foto.
    """
    from sklearn.cluster import KMeans

    rasgos = rasgos[:, columnas]
    z = (rasgos - rasgos.mean(axis=0)) / (rasgos.std(axis=0) + 1e-9)
    k = max(2, len(df) // imagenes_por_grupo)
    etiquetas = KMeans(n_clusters=k, n_init=3, random_state=semilla).fit_predict(z)
    salida = df.copy()
    salida["grupo"] = [f"cluster|{c}" for c in etiquetas]
    return salida


# --------------------------------------------------------------------------
# 4. División sin fuga
# --------------------------------------------------------------------------
def dividir(df: pd.DataFrame, fraccion_prueba: float = config.FRACCION_PRUEBA,
            fraccion_val: float = config.FRACCION_VALIDACION,
            semilla: int = config.SEMILLA):
    """Divide en (entrenamiento, validación, prueba) manteniendo cada GRUPO entero.

    Primer corte: `fraccion_prueba` del total para prueba.
    Segundo corte: sobre lo que sobra, para que validación sea `fraccion_val`
    del TOTAL: val_rel = fraccion_val / (1 - fraccion_prueba).

    Lanza un error si algún grupo o ruta aparece en dos particiones.
    """
    from sklearn.model_selection import GroupShuffleSplit

    corte1 = GroupShuffleSplit(n_splits=1, test_size=fraccion_prueba, random_state=semilla)
    i_resto, i_prueba = next(corte1.split(df, groups=df["grupo"]))
    resto, prueba = df.iloc[i_resto], df.iloc[i_prueba]

    val_rel = fraccion_val / (1.0 - fraccion_prueba)
    corte2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=semilla)
    i_tr, i_va = next(corte2.split(resto, groups=resto["grupo"]))
    entrenamiento, validacion = resto.iloc[i_tr], resto.iloc[i_va]

    verificar_sin_fuga(entrenamiento, validacion, prueba)
    return (entrenamiento.reset_index(drop=True), validacion.reset_index(drop=True),
            prueba.reset_index(drop=True))


def verificar_sin_fuga(*particiones: pd.DataFrame) -> None:
    """Falla (con un mensaje claro) si dos particiones comparten un grupo o una ruta."""
    for a in range(len(particiones)):
        for b in range(a + 1, len(particiones)):
            comunes_g = set(particiones[a]["grupo"]) & set(particiones[b]["grupo"])
            comunes_r = set(particiones[a]["ruta"]) & set(particiones[b]["ruta"])
            if comunes_g or comunes_r:
                raise AssertionError(
                    f"FUGA entre las particiones {a} y {b}: {len(comunes_g)} grupos y "
                    f"{len(comunes_r)} imágenes repetidas.")


def reservar_fuente(df: pd.DataFrame, fuente: str):
    """Aparta una fuente ENTERA para la prueba fuera del dominio (ej.: SDNET2018).

    El modelo nunca la ve al entrenar. Devuelve (resto, fuera_de_dominio).
    """
    mascara = df["fuente"] == fuente
    if not mascara.any():
        raise ValueError(f"La fuente '{fuente}' no está en el inventario.")
    return df[~mascara].reset_index(drop=True), df[mascara].reset_index(drop=True)


def resumen_particiones(**particiones: pd.DataFrame) -> pd.DataFrame:
    """Tabla con tamaño, % del total, proporción de grietas y nº de grupos por partición."""
    total = sum(len(p) for p in particiones.values())
    filas = []
    for nombre, p in particiones.items():
        filas.append({"particion": nombre, "imagenes": len(p),
                      "porcentaje": round(100 * len(p) / total, 1),
                      "con_grieta_%": round(100 * p["etiqueta"].mean(), 1),
                      "grupos": p["grupo"].nunique()})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 5. Duplicados (hash perceptual)
# --------------------------------------------------------------------------
def huella_perceptual(ruta, lado: int = 8) -> str:
    """64 bits: imagen -> gris -> 8x8 -> cada píxel contra el promedio.

    No cambia si la imagen se aclara u oscurece de forma uniforme.
    """
    a = np.asarray(Image.open(ruta).convert("L").resize((lado, lado)), dtype=np.float32)
    return "".join("1" if v else "0" for v in (a > a.mean()).ravel())


def encontrar_duplicados(rutas) -> dict[str, list]:
    """Agrupa imágenes con la misma huella. Es un LÍMITE INFERIOR: solo detecta
    casi-copias, no recortes distintos de una misma foto."""
    grupos: dict[str, list] = {}
    for r in rutas:
        grupos.setdefault(huella_perceptual(r), []).append(r)
    return {h: rs for h, rs in grupos.items() if len(rs) > 1}


def huellas_compartidas(df_a: pd.DataFrame, df_b: pd.DataFrame) -> int:
    """Cuántas huellas aparecen en dos particiones a la vez (otra señal de fuga)."""
    ha = {huella_perceptual(r) for r in df_a["ruta"]}
    hb = {huella_perceptual(r) for r in df_b["ruta"]}
    return len(ha & hb)


# --------------------------------------------------------------------------
# 6. Auditoría de fuentes: el ATAJO DE LA FUENTE
# --------------------------------------------------------------------------
def auditoria_fuentes(df: pd.DataFrame):
    """Tabla fuente x clase y avisos.

    EL RIESGO: si una fuente tiene solo una clase (ej.: todos los negativos
    difíciles son "sin grieta"), el modelo puede aprender "foto de celular =
    sin grieta" en vez de aprender qué es una grieta. Es el atajo del brillo,
    pero en grande. Cada fuente debería tener ejemplos de las DOS clases, o
    quedar apartada solo para evaluar.
    """
    df = df.reset_index(drop=True)
    tabla = pd.crosstab(df["fuente"], df["etiqueta"]).reindex(columns=[0, 1], fill_value=0)
    tabla.columns = ["sin_grieta", "con_grieta"]
    avisos = []
    for fuente, fila in tabla.iterrows():
        if fila.min() == 0:
            clase = "sin_grieta" if fila["con_grieta"] == 0 else "con_grieta"
            avisos.append(f"La fuente '{fuente}' solo tiene '{clase}': el modelo podría aprender "
                          f"la FUENTE en vez de la grieta. Agregale ejemplos de la otra clase o "
                          f"usala solo para evaluar.")
        elif max(fila) > 10 * min(fila):
            avisos.append(f"La fuente '{fuente}' está muy desbalanceada ({fila.to_dict()}).")
    if "grupo" in df.columns:
        for fuente, n in df.groupby("fuente")["grupo"].nunique().items():
            if not str(fuente).startswith("surface") and n < MIN_ESCENAS:
                avisos.append(f"La fuente '{fuente}' tiene solo {n} escena(s). Con tan pocas, casi seguro ninguna cae en "
                              f"validación ni prueba y el falso positivo NO se podrá medir en escenas no vistas. "
                              f"Sumá escenas (meta: {MIN_ESCENAS} a 10, con 8-10 fotos cada una).")
    return tabla, avisos
