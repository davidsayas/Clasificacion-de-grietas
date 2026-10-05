"""informe.py — Las secciones de RESULTADOS del informe, escritas desde los datos reales.

Los números del informe no se tipean: se leen de results/*.csv, que producen los cuadernos.
Así el informe no puede contradecir a los experimentos, ni quedar con cifras de una versión
anterior. Si falta un resultado, la sección dice PENDIENTE y en qué cuaderno se produce.

Uso:      python -m src.informe
Escribe:  results/secciones_informe_v2.md   (se copia a INFORME_TECNICO.md, secciones 6, 8, 9 y 10)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import evaluar, riesgo


def _md(tabla: pd.DataFrame, decimales: int = 4) -> str:
    def f(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else (f"{v:.{decimales}f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}")
        return "" if v is None else str(v)
    cab = "| " + " | ".join(map(str, tabla.columns)) + " |"
    sep = "|" + "|".join("---" for _ in tabla.columns) + "|"
    filas = ["| " + " | ".join(f(v) for v in fila) + " |" for fila in tabla.itertuples(index=False)]
    return "\n".join([cab, sep] + filas)


def _pendiente(cuaderno: str, archivo: str) -> str:
    return f"> **PENDIENTE:** falta `{archivo}`. Se produce en el cuaderno `{cuaderno}`."


def _csv(raiz: Path, nombre: str):
    ruta = raiz / "results" / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def generar(raiz=".") -> str:
    raiz = Path(raiz)
    out = ["# Secciones de resultados (v2) — generadas desde results/", ""]

    # --- Resultados en prueba ---
    out += ["## Resultados en el conjunto de prueba (división por grupos, sin fuga)", ""]
    t = _csv(raiz, "tabla_comparativa.csv")
    out += [_md(t) if t is not None else _pendiente("03_Robustez_y_falso_positivo", "tabla_comparativa.csv"), "",
            "Matriz de confusión: `results/matriz_confusion.png`. Curvas: `results/curvas.png`.", ""]

    # --- Atajo del brillo ---
    out += ["## Prueba del atajo del brillo", ""]
    t = _csv(raiz, "prueba_atajo_brillo.csv")
    if t is not None:
        out += [_md(t, 1), "", f"**Conclusión:** {evaluar.veredicto_atajo(t)}", ""]
    else:
        out += [_pendiente("03_Robustez_y_falso_positivo", "prueba_atajo_brillo.csv"), ""]

    # --- Robustez ---
    out += ["## Robustez a giro, lejanía, perspectiva, desenfoque y ruido", ""]
    t = _csv(raiz, "prueba_robustez.csv")
    out += [_md(t, 1) if t is not None else _pendiente("03_Robustez_y_falso_positivo", "prueba_robustez.csv"), ""]

    # --- Complejidad ---
    out += ["## 9. Complejidad computacional", "",
            "Tiempos medidos con UNA imagen por llamada (el caso de la app), mediana y p95 tras calentamiento, "
            "en CPU. Una CPU de Colab no es un celular: sirven para comparar versiones entre sí. "
            "1 MB = 10^6 bytes.", ""]
    t = _csv(raiz, "complejidad.csv")
    out += [_md(t, 2) if t is not None else _pendiente("05_Complejidad_y_despliegue", "complejidad.csv"), ""]
    ruta = raiz / "results" / "fidelidad_tflite.json"
    if ruta.exists():
        out += ["**Fidelidad de la conversión a TFLite** (¿cambió las respuestas?):", "", "```json",
                ruta.read_text(encoding="utf-8").strip(), "```", ""]

    # --- Versiones ---
    out += ["## Comparación de desempeño y complejidad entre versiones", ""]
    t = _csv(raiz, "comparacion_versiones.csv")
    out += [_md(t) if t is not None else _pendiente("05_Complejidad_y_despliegue", "comparacion_versiones.csv"), ""]

    # --- Fotos propias ---
    out += ["## Fotografías propias", ""]
    t, m = _csv(raiz, "fotos_propias_analisis.csv"), raiz / "results" / "metricas_fotos_propias.json"
    if t is not None:
        resumen = t["riesgo"].value_counts().rename_axis("riesgo").reset_index(name="fotos")
        out += [f"{len(t)} fotos analizadas. Distribución del nivel de riesgo:", "", _md(resumen), ""]
    else:
        out += [_pendiente("03_Robustez_y_falso_positivo", "fotos_propias_analisis.csv"), ""]
    if m.exists():
        d = json.loads(m.read_text(encoding="utf-8"))
        f, a = d["forzada"], d["con_abstencion"]
        out += [f"Con etiqueta: {d['n']} fotos ({d['con_grieta']} con grieta, {d['sin_grieta']} sin grieta).", "",
                "**Respuesta forzada** (el sistema siempre responde):", "",
                f"exactitud {f['exactitud']:.3f} · precisión {f['precision']:.3f} · recall {f['recall']:.3f} · "
                f"F1 {f['f1']:.3f} · falsos negativos {f['fn']} · falsos positivos {f['fp']}", "",
                "**Con abstención** (puede responder INDETERMINADO):", "",
                f"cobertura {a['cobertura']:.2f} · exactitud en las respondidas {a['exactitud_en_respondidas']:.3f} · "
                f"**grietas reales sin diagnóstico: {a['grietas_reales_sin_diagnostico']}**", ""]

    d = _csv(raiz, "diagnostico_fotos_propias.csv")
    if d is not None:
        out += ["### Diagnóstico por método (EXPLORATORIO)", "",
                "Cada método le pregunta a la misma red de forma distinta. `p_sistema` incluye la puerta de concreto del "
                "Paso 6b original; las demás columnas no. El AUC no depende del umbral. **Se calcula sobre las mismas fotos "
                "de prueba: es un dato exploratorio, no un resultado final.**", "",
                _md(evaluar.resumen_diagnostico(d), 3), ""]

    # --- Reglas ---
    out += ["## 8. Reglas del motor de riesgo (las que implementa el código)", "", riesgo.reglas_markdown(), ""]
    return "\n".join(out)


def main(raiz="."):
    texto = generar(raiz)
    salida = Path(raiz) / "results" / "secciones_informe_v2.md"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(f"Escrito: {salida}  ({texto.count('PENDIENTE')} secciones PENDIENTES)")
    return texto


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
