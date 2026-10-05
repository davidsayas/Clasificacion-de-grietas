# Secciones de resultados (v2) — generadas desde results/

## Resultados en el conjunto de prueba (división por grupos, sin fuga)

| modelo | exactitud | recall | precision | f1 | FN | FP |
|---|---|---|---|---|---|---|
| línea base: umbral de brillo | 0.7027 | 0.7392 | 0.6837 | 0.7104 | 386 | 506 |
| línea base: umbral de bordes | 0.8487 | 0.9669 | 0.7794 | 0.8631 | 49 | 405 |
| línea base: regresión logística (5 rasgos) | 0.9827 | 0.9818 | 0.9831 | 0.9824 | 27 | 25 |
| v2_mobilenet | 0.9995 | 0.9993 | 0.9997 | 0.9995 | 2 | 1 |

Matriz de confusión: `results/matriz_confusion.png`. Curvas: `results/curvas.png`.

## Prueba del atajo del brillo

| delta_brillo | sin_grieta_oscurecidas→con_grieta_% | con_grieta_aclaradas→sin_grieta_% |
|---|---|---|
| 18 | 0 | 0 |
| 30 | 0 | 0 |
| 51 | 0 | 0 |

**Conclusión:** El brillo NO es un atajo: como máximo 0.0 % de las imágenes cambian de clase. El falso positivo debe venir de otra causa (datos nunca vistos).

## Robustez a giro, lejanía, perspectiva, desenfoque y ruido

| prueba | parametro | cambian_de_clase_% | exactitud_% | recall_% |
|---|---|---|---|---|
| giro_grados | 5 | 0 | 100 | 100 |
| giro_grados | 15 | 0 | 100 | 100 |
| giro_grados | 30 | 0.3 | 99.7 | 99.3 |
| giro_grados | 45 | 0.3 | 99.7 | 99.3 |
| giro_grados | 54 | 0.7 | 99.3 | 98.7 |
| giro_grados | 90 | 0.3 | 99.7 | 99.3 |
| lejania_escala | 0.8 | 0 | 100 | 100 |
| lejania_escala | 0.5 | 0 | 100 | 100 |
| lejania_escala | 0.3 | 0 | 100 | 100 |
| lejania_escala | 0.2 | 1 | 99 | 98 |
| perspectiva_grados | 10 | 0 | 100 | 100 |
| perspectiva_grados | 20 | 0 | 100 | 100 |
| perspectiva_grados | 30 | 0 | 100 | 100 |
| perspectiva_grados | 45 | 0 | 100 | 100 |
| desenfoque_sigma | 1 | 0 | 100 | 100 |
| desenfoque_sigma | 2 | 1 | 99 | 98 |
| desenfoque_sigma | 4 | 8.7 | 91.3 | 82.7 |
| ruido_sd | 5 | 0.3 | 99.7 | 99.3 |
| ruido_sd | 15 | 3 | 97 | 94 |
| ruido_sd | 30 | 21.3 | 78.7 | 57.3 |

## 9. Complejidad computacional

Tiempos medidos con UNA imagen por llamada (el caso de la app), mediana y p95 tras calentamiento, en CPU. Una CPU de Colab no es un celular: sirven para comparar versiones entre sí. 1 MB = 10^6 bytes.

| modelo | parametros | entrenables | no_entrenables | mb_keras | ms_keras_1img | ms_keras_p95 | mb_tflite | ms_tflite_1img | ms_tflite_p95 | cambian_de_clase_vs_keras | dif_max_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_mobilenet/float32 | 2259265 | 1512001 | 747264 | 21.75 | 192.45 | 325.77 | 8.86 | 8.74 | 12.37 | 0 | 0 |
| v2_mobilenet/dinamico | 2259265 | 1512001 | 747264 | 21.75 | 195.37 | 295.38 | 2.5 | 14.59 | 18.94 | 0 | 0.02 |
| v2_mobilenet/float16 | 2259265 | 1512001 | 747264 | 21.75 | 178.75 | 282.86 | 4.46 | 9.33 | 11.52 | 0 | 0.01 |

**Fidelidad de la conversión a TFLite** (¿cambió las respuestas?):

```json
{
  "v2_mobilenet/float32": {
    "dif_max": 2.682209014892578e-06,
    "dif_media": 1.4522695247857579e-08,
    "cambian_de_clase": 0,
    "n": 300
  },
  "v2_mobilenet/dinamico": {
    "dif_max": 0.022817552089691162,
    "dif_media": 0.00018919228667688988,
    "cambian_de_clase": 0,
    "n": 300
  },
  "v2_mobilenet/float16": {
    "dif_max": 0.006658986210823059,
    "dif_media": 5.283737485816194e-05,
    "cambian_de_clase": 0,
    "n": 300
  }
}
```

## Comparación de desempeño y complejidad entre versiones

| version | exactitud_prueba | recall_prueba | mb_tflite | ms_tflite_1img | ms_tflite_p95 | parametros | exactitud_fotos_propias | recall_fotos_propias |
|---|---|---|---|---|---|---|---|---|
| línea base: regresión logística (5 rasgos) | 0.9827 | 0.9818 |  |  |  |  |  |  |
| v2 mobilenetv2 | 0.9993 | 0.9986 | 4.46 | 9.07 | 11.32 | 2,259,265 | 0.6207 | 0.1538 |

## Fotografías propias

29 fotos analizadas. Distribución del nivel de riesgo:

| riesgo | fotos |
|---|---|
| BAJO | 25 |
| INDETERMINADO | 4 |

Con etiqueta: 29 fotos (13 con grieta, 16 sin grieta).

**Respuesta forzada** (el sistema siempre responde):

exactitud 0.552 · precisión nan · recall 0.000 · F1 nan · falsos negativos 13 · falsos positivos 0

**Con abstención** (puede responder INDETERMINADO):

cobertura 0.86 · exactitud en las respondidas 0.640 · **grietas reales sin diagnóstico: 4**

## 8. Reglas del motor de riesgo (las que implementa el código)

| Factor | Regla | Puntos |
|---|---|---|
| Grieta detectada | probabilidad ≥ umbral | +1 |
| Ancho | < 0.3 mm / 0.3–1.0 / 1.0–3.0 / ≥ 3.0 mm | 0 / +1 / +2 / +3 |
| Orientación | diagonal +1 · vertical +0 · horizontal +0 | según orientación |
| Elemento afectado | columna +1 · viga +1 · muro +0 · placa +0 | según elemento |
| Inclinación | < 0.5° / 0.5–1.5 / 1.5–3.0 / ≥ 3.0° | 0 / +1 / +2 / +3 |

**Nivel:** ALTO si el total ≥ 5 **o** el ancho ≥ 3.0 mm **o** la inclinación ≥ 3.0°; MEDIO si el total ≥ 2; BAJO en otro caso.
**No se afirma (INDETERMINADO)** si la imagen no parece concreto o la probabilidad está entre 0.2 y 0.8.

> Reglas orientativas; no sustituyen la inspección de un ingeniero. Sesgadas a sobreestimar el riesgo.
