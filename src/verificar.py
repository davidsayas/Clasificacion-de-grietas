"""verificar.py — Checklist de las 3 entregas según EVIDENCIA, no según intención.

Cada requisito sale de la sección 8 del documento "Descripción del proyecto" del reto
(Algoritmos y Programación 2026-2). Un requisito se marca:

    ✅  si TODA su evidencia existe (archivos, columnas, texto sin "[COMPLETAR]")
    ❌  si falta algo (y dice qué)
    🧍  si no se puede comprobar desde el repositorio (GitHub, la demostración en vivo)

Uso:      python -m src.verificar
Escribe:  results/estado_entregas.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

INFORME = ("INFORME_TECNICO.md", "docs/INFORME_TECNICO.md")
PLAN = ("PLAN_DE_TRABAJO.md", "docs/PLAN_DE_TRABAJO.md")


# --- tipos de comprobación -------------------------------------------------
def archivos(*rutas):
    """Cada elemento es una ruta o una tupla de alternativas. Admite comodines (*)."""
    return ("archivos", [r if isinstance(r, tuple) else (r,) for r in rutas])


def contiene(alternativas, texto):
    return ("contiene", alternativas, texto)


def sin_marcador(alternativas, marcador="[COMPLETAR"):
    return ("sin_marcador", alternativas, marcador)


def columnas(archivo, *cols):
    return ("columnas", archivo, cols)


def manual(nota):
    return ("manual", nota)


# (id, corte, requisito, criterios de rúbrica, comprobaciones)
REQUISITOS = [
    ("1.1", "Entrega 1", "Problema, contexto y clases a predecir", ("comunicacion",),
     [contiene(INFORME, "Clases a predecir")]),
    ("1.2", "Entrega 1", "Dataset descrito (origen, nº, clases, balance) + análisis exploratorio", ("pipeline",),
     [archivos("results/inventario.csv", "results/histograma_brillo.png")]),
    ("1.3", "Entrega 1", "Revisión breve del estado del arte", ("pipeline",),
     [contiene(INFORME, "Estado del arte")]),
    ("1.4", "Entrega 1", "Clasificador de línea base con métricas", ("desempeno",),
     [archivos("results/linea_base.csv")]),
    ("1.5", "Entrega 1", "Primera estimación de inclinación sobre una foto propia", ("pipeline",),
     [archivos("results/inclinacion_fotos_propias.csv")]),
    ("1.6", "Entrega 1", "Plan de trabajo con roles del equipo (completo)", ("comunicacion",),
     [archivos(PLAN), sin_marcador(PLAN)]),
    ("1.7", "Entrega 1", "Código en GitHub", ("comunicacion",),
     [manual("Subir el repositorio (comandos al final del README) y verificar que clona y corre")]),

    ("2.1", "Entrega 2", "Preprocesamiento y aumentación documentados", ("pipeline",),
     [archivos("models/*/meta.json")]),
    ("2.2", "Entrega 2", "División entrenamiento / validación / prueba (sin fuga)", ("desempeno",),
     [archivos("results/particion_entrenamiento.csv", "results/particion_validacion.csv",
               "results/particion_prueba.csv")]),
    ("2.3", "Entrega 2", "Modelo por transfer learning, mejorado frente a la línea base", ("desempeno",),
     [archivos("models/*/modelo.keras", "results/tabla_comparativa.csv")]),
    ("2.4", "Entrega 2", "Métricas: exactitud, precisión, recall, F1 y matriz de confusión", ("desempeno",),
     [archivos("results/matriz_confusion.png"),
      columnas("results/tabla_comparativa.csv", "exactitud", "precision", "recall", "f1")]),
    ("2.5", "Entrega 2", "Curvas de entrenamiento registradas", ("pipeline",),
     [archivos("results/curvas.png")]),
    ("2.6", "Entrega 2", "Análisis de complejidad (parámetros, tamaño en MB, tiempo de inferencia)", ("eficiencia",),
     [columnas("results/complejidad.csv", "parametros", "mb_tflite", "ms_tflite_1img")]),
    ("2.7", "Entrega 2", "Integración grietas + inclinación + primer criterio de riesgo", ("pipeline",),
     [columnas("results/fotos_propias_analisis.csv", "prob_grieta", "inclinacion_deg", "riesgo")]),
    ("2.8", "Entrega 2", "Informe con resultados (sin marcadores [COMPLETAR])", ("comunicacion",),
     [archivos(INFORME), sin_marcador(INFORME)]),

    ("3.1", "Entrega 3", "Modelo desplegado (TFLite y/o Streamlit/Gradio)", ("despliegue",),
     [archivos("models/*/modelo*.tflite", "app/streamlit_app.py")]),
    ("3.2", "Entrega 3", "Probado con fotografías propias, con sus métricas", ("desempeno", "despliegue"),
     [archivos("results/fotos_propias_analisis.csv", "results/metricas_fotos_propias.json")]),
    ("3.3", "Entrega 3", "Mapeo a nivel de riesgo combinando grietas e inclinación", ("pipeline",),
     [columnas("results/fotos_propias_analisis.csv", "riesgo", "inclinacion_deg", "orientacion")]),
    ("3.4", "Entrega 3", "Comparación de desempeño y complejidad entre versiones", ("eficiencia", "desempeno"),
     [archivos("results/comparacion_versiones.csv")]),
    ("3.5", "Entrega 3", "Análisis de limitaciones e implicaciones éticas", ("pipeline",),
     [contiene(INFORME, "Consideraciones éticas"), contiene(INFORME, "Limitaciones")]),
    ("3.6", "Entrega 3", "Informe técnico final (completo, con complejidad y fotos propias)", ("comunicacion",),
     [contiene(INFORME, "Complejidad computacional"), contiene(INFORME, "fotos propias"), sin_marcador(INFORME)]),
    ("3.7", "Entrega 3", "Repositorio ordenado", ("comunicacion",),
     [archivos("README.md", "requirements.txt", ".gitignore", "tests/run_all.py")]),
    ("3.8", "Entrega 3", "Sustentación con demostración en vivo", ("comunicacion", "despliegue"),
     [manual("Probar la demo en el equipo y la red de la presentación; cada integrante explica todo el sistema")]),
]

RUBRICA = [("desempeno", "Desempeño del modelo (prueba y fotos propias)", 30),
           ("despliegue", "Despliegue funcional (computador o celular)", 25),
           ("eficiencia", "Eficiencia y complejidad", 15),
           ("pipeline", "Comprensión del pipeline y calidad del análisis (limitaciones y ética)", 20),
           ("comunicacion", "Comunicación, documentación y trabajo en equipo", 10)]


# --- evaluación ------------------------------------------------------------
def _buscar(raiz: Path, alternativas) -> Path | None:
    for a in alternativas:
        if "*" in a:
            hallados = sorted(raiz.glob(a))
            if hallados:
                return hallados[0]
        elif (raiz / a).exists():
            return raiz / a
    return None


def _comprobar(raiz: Path, c) -> tuple[bool | None, str]:
    """(True | False | None=manual, qué falta)."""
    tipo = c[0]
    if tipo == "manual":
        return None, c[1]
    if tipo == "archivos":
        faltan = [" o ".join(alt) for alt in c[1] if _buscar(raiz, alt) is None]
        return (not faltan), ("falta: " + ", ".join(faltan)) if faltan else ""
    ruta = _buscar(raiz, c[1])
    if ruta is None:
        return False, "falta: " + " o ".join(c[1])
    if tipo == "contiene":
        ok = c[2].lower() in ruta.read_text(encoding="utf-8", errors="ignore").lower()
        return ok, "" if ok else f"'{ruta.name}' no contiene «{c[2]}»"
    if tipo == "sin_marcador":
        n = ruta.read_text(encoding="utf-8", errors="ignore").count(c[2])
        return n == 0, "" if n == 0 else f"'{ruta.name}' todavía tiene {n} marcador(es) {c[2]}"
    if tipo == "columnas":
        r = _buscar(raiz, (c[1],))
        if r is None:
            return False, f"falta: {c[1]}"
        try:
            cols = set(pd.read_csv(r).columns)
        except Exception as e:                      # archivo vacío o dañado
            return False, f"no se pudo leer {c[1]}: {e}"
        faltan = [x for x in c[2] if x not in cols]
        return (not faltan), "" if not faltan else f"{c[1]} sin columnas: {', '.join(faltan)}"
    raise ValueError(f"tipo de comprobación desconocido: {tipo}")


def estado(raiz=".") -> pd.DataFrame:
    raiz = Path(raiz)
    filas = []
    for id_, corte, texto, criterios, comprobaciones in REQUISITOS:
        resultados = [_comprobar(raiz, c) for c in comprobaciones]
        if any(r[0] is False for r in resultados):
            marca = "❌"
        elif any(r[0] is None for r in resultados):
            marca = "🧍"
        else:
            marca = "✅"
        detalle = "; ".join(dict.fromkeys(r[1] for r in resultados if r[1]))       # sin repetidos
        filas.append({"id": id_, "corte": corte, "requisito": texto, "estado": marca,
                      "detalle": detalle, "criterios": criterios})
    return pd.DataFrame(filas)


def resumen_rubrica(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por criterio de la rúbrica: cuántos requisitos que lo alimentan están ✅."""
    filas = []
    for clave, nombre, peso in RUBRICA:
        sub = tabla[tabla["criterios"].apply(lambda c: clave in c)]
        filas.append({"criterio": nombre, "peso_%": peso, "requisitos": len(sub),
                      "cumplidos": int((sub["estado"] == "✅").sum()),
                      "manuales": int((sub["estado"] == "🧍").sum())})
    return pd.DataFrame(filas)


def a_markdown(tabla: pd.DataFrame, rubrica: pd.DataFrame) -> str:
    lineas = ["# Estado de las entregas (según evidencia)", "",
              "✅ evidencia completa · ❌ falta evidencia · 🧍 se comprueba a mano", ""]
    for corte in ("Entrega 1", "Entrega 2", "Entrega 3"):
        sub = tabla[tabla["corte"] == corte]
        lineas += [f"## {corte}  ({int((sub['estado'] == '✅').sum())}/{len(sub)} ✅)", "",
                   "| | Requisito | Qué falta |", "|---|---|---|"]
        lineas += [f"| {f.estado} | {f.id} {f.requisito} | {f.detalle} |" for f in sub.itertuples()]
        lineas.append("")
    lineas += ["## Rúbrica del reto", "", "| Criterio | Peso | Requisitos cumplidos |", "|---|---|---|"]
    lineas += [f"| {f.criterio} | {f._2} % | {f.cumplidos}/{f.requisitos}" + (f" (+{f.manuales} manual)" if f.manuales else "") + " |"
               for f in rubrica.itertuples()]
    return "\n".join(lineas)


def main(raiz="."):
    tabla = estado(raiz)
    texto = a_markdown(tabla, resumen_rubrica(tabla))
    salida = Path(raiz) / "results" / "estado_entregas.md"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(texto)
    return tabla


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
