# Checklist de entregas (según el documento del reto)

Los requisitos salen de la sección 8 del documento *Descripción del proyecto* (Algoritmos y Programación 2026-2).
**El estado real no se escribe a mano:** `python -m src.verificar` marca cada fila ✅ solo si existe su evidencia
(archivos, columnas, texto sin `[COMPLETAR]`) y escribe `results/estado_entregas.md`. 🧍 = se comprueba a mano.

> Fechas del documento: Corte 1 **25/09/2026** (20 %) · Corte 2 **23/10/2026** (20 %) · Corte 3 **20/11/2026** (30 %).

## Entrega 1

| Requisito | Evidencia que lo marca ✅ | Se produce en |
|---|---|---|
| 1.1 Problema, contexto y clases a predecir | `INFORME_TECNICO.md` con «Clases a predecir» | `INFORME_TECNICO.md` (sección 1.3) |
| 1.2 Dataset descrito (origen, nº, clases, balance) + análisis exploratorio | `results/inventario.csv`; `results/histograma_brillo.png` | cuaderno 01, pasos 1–3 |
| 1.3 Revisión breve del estado del arte | `INFORME_TECNICO.md` con «Estado del arte» | `INFORME_TECNICO.md` (sección 3) |
| 1.4 Clasificador de línea base con métricas | `results/linea_base.csv` | cuaderno 01, paso 7 |
| 1.5 Primera estimación de inclinación sobre una foto propia | `results/inclinacion_fotos_propias.csv` | cuaderno 05, paso 3 · `docs/PROTOCOLO_FOTOS_PROPIAS.md` |
| 1.6 Plan de trabajo con roles del equipo (completo) | `PLAN_DE_TRABAJO.md o docs/PLAN_DE_TRABAJO.md`; `PLAN_DE_TRABAJO.md` sin `[COMPLETAR]` | completar `PLAN_DE_TRABAJO.md` |
| 1.7 Código en GitHub | 🧍 Subir el repositorio (comandos al final del README) y verificar que clona y corre | subir el repositorio (README, al final) |

## Entrega 2

| Requisito | Evidencia que lo marca ✅ | Se produce en |
|---|---|---|
| 2.1 Preprocesamiento y aumentación documentados | `models/*/meta.json` | cuaderno 02, paso 4 |
| 2.2 División entrenamiento / validación / prueba (sin fuga) | `results/particion_entrenamiento.csv`; `results/particion_validacion.csv`; `results/particion_prueba.csv` | cuaderno 01, paso 6 |
| 2.3 Modelo por transfer learning, mejorado frente a la línea base | `models/*/modelo.keras`; `results/tabla_comparativa.csv` | cuadernos 02 (pasos 4–5) y 03 (paso 8) |
| 2.4 Métricas: exactitud, precisión, recall, F1 y matriz de confusión | `results/matriz_confusion.png`; `results/tabla_comparativa.csv` con columnas exactitud, precision, recall, f1 | cuaderno 03, pasos 1 y 8 |
| 2.5 Curvas de entrenamiento registradas | `results/curvas.png` | cuaderno 02, paso 6 |
| 2.6 Análisis de complejidad (parámetros, tamaño en MB, tiempo de inferencia) | `results/complejidad.csv` con columnas parametros, mb_tflite, ms_tflite_1img | cuaderno 05, paso 1 |
| 2.7 Integración grietas + inclinación + primer criterio de riesgo | `results/fotos_propias_analisis.csv` con columnas prob_grieta, inclinacion_deg, riesgo | cuaderno 03, paso 6b |
| 2.8 Informe con resultados (sin marcadores [COMPLETAR]) | `INFORME_TECNICO.md o docs/INFORME_TECNICO.md`; `INFORME_TECNICO.md` sin `[COMPLETAR]` | `python -m src.informe` · `docs/ACTUALIZACION_INFORME_v2.md` |

## Entrega 3

| Requisito | Evidencia que lo marca ✅ | Se produce en |
|---|---|---|
| 3.1 Modelo desplegado (TFLite y/o Streamlit/Gradio) | `models/*/modelo*.tflite`; `app/streamlit_app.py` | cuaderno 05, paso 1 · `app/streamlit_app.py` |
| 3.2 Probado con fotografías propias, con sus métricas | `results/fotos_propias_analisis.csv`; `results/metricas_fotos_propias.json` | cuaderno 03, paso 6b |
| 3.3 Mapeo a nivel de riesgo combinando grietas e inclinación | `results/fotos_propias_analisis.csv` con columnas riesgo, inclinacion_deg, orientacion | cuaderno 03, paso 6b |
| 3.4 Comparación de desempeño y complejidad entre versiones | `results/comparacion_versiones.csv` | cuaderno 05, paso 2 |
| 3.5 Análisis de limitaciones e implicaciones éticas | `INFORME_TECNICO.md` con «Consideraciones éticas»; `INFORME_TECNICO.md` con «Limitaciones» | `INFORME_TECNICO.md` (secciones 10 y 11) · `docs/ACTUALIZACION_INFORME_v2.md` |
| 3.6 Informe técnico final (completo, con complejidad y fotos propias) | `INFORME_TECNICO.md` con «Complejidad computacional»; `INFORME_TECNICO.md` con «fotos propias»; `INFORME_TECNICO.md` sin `[COMPLETAR]` | `INFORME_TECNICO.md` completo |
| 3.7 Repositorio ordenado | `README.md`; `requirements.txt`; `.gitignore`; `tests/run_all.py` | ya incluido en el repositorio |
| 3.8 Sustentación con demostración en vivo | 🧍 Probar la demo en el equipo y la red de la presentación; cada integrante explica todo el sistema | ensayo con el equipo |

## Rúbrica del reto (sección 9)

| Criterio | Peso | Qué lo alimenta |
|---|---|---|
| Desempeño del modelo (prueba y fotos propias) | 30 % | requisitos 1.4, 2.2, 2.3, 2.4, 3.2, 3.4 |
| Despliegue funcional (computador o celular) | 25 % | requisitos 3.1, 3.2, 3.8 |
| Eficiencia y complejidad | 15 % | requisitos 2.6, 3.4 |
| Comprensión del pipeline y calidad del análisis (limitaciones y ética) | 20 % | requisitos 1.2, 1.3, 1.5, 2.1, 2.5, 2.7, 3.3, 3.5 |
| Comunicación, documentación y trabajo en equipo | 10 % | requisitos 1.1, 1.6, 1.7, 2.8, 3.6, 3.7, 3.8 |

## Extras NO exigidos por el documento

El documento solo menciona la segmentación como *"nivel avanzado/opcional"*. Estos módulos suman, pero **no deben desplazar** lo obligatorio:

* Grad-CAM (`src/gradcam.py`), puerta "¿parece concreto?" (`src/ood.py`), U-Net (`src/unet.py`, cuaderno 04, necesita máscaras).
