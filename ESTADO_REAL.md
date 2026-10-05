# Estado real del proyecto

Lo que dice cada fila viene de **algo que se ejecutó y se vio en la conversación**, o de **código/documentos que se
probaron con datos sintéticos**. Nada está marcado como hecho por intención.

## ✅ Ejecutado por el equipo en Colab (resultados vistos)

| Qué | Resultado |
|---|---|
| Cuaderno 01 completo | 40.044 imágenes · fuga demostrada (913 de 913 grupos) · división 70/15/15 sin fuga · razón de adyacencia 0,50 · duplicados 1,52 % |
| Cuaderno 02, pasos 1–4 y 7 | modelo `v2_mobilenet` entrenado · validación 99,87 % (5 FN, 3 FP) |
| Cuaderno 03, Paso 0 | prueba verificada: 0 imágenes y 0 grupos en común con el entrenamiento |
| Cuaderno 03, Paso 1 | **prueba 99,95 %** · matriz `[[3123,1],[2,2926]]` · 2 grietas no detectadas de 2.928 |
| Cuaderno 03, Paso 2 | el brillo **no** es un atajo |
| Cuaderno 03, Pasos 4, 5, 6b, 6c, 6d | Grad-CAM · puerta de concreto · 29 fotos propias · memorización (215/215 vs 2/13) · AUC 0,89 |

## ✅ Hecho y probado con datos sintéticos (105 pruebas)

División sin fuga · EDA · línea base · pruebas de robustez · mosaico · ancho (sesgo −0,02 px) · rectificación de perspectiva ·
orientación · inclinación con gravedad · motor de riesgo · fotos propias · complejidad · verificador · informe · lectura
de `etiquetas.csv` de Excel · carga por tandas · **lógica de la app**.

## ✅ Documentos terminados

`INFORME_TECNICO.md` (actualizado a la v2, con las reglas de riesgo tomadas del código) · `docs/HALLAZGOS_PARA_EL_INFORME.md` ·
`docs/GUIA_APP.md` · `docs/PROTOCOLO_FOTOS_PROPIAS.md` · `CHECKLIST.md`.

## ⚠ Escrito pero NO ejecutado donde se hizo (sin TensorFlow ni Streamlit)

La interfaz `app/streamlit_app.py`, la carga del modelo en la app, Grad-CAM, la U-Net y el cuaderno 06. Se revisaron con un
TensorFlow simulado; la primera corrida real puede pedir algún ajuste.

## 🔧 Falta ejecutar en TU Colab (en este orden)

1. **Cuaderno 03:** Paso 6b (ahora sin puerta) y Paso 6d → actualiza `metricas_fotos_propias.json` y el AUC.
2. **Cuaderno 03:** Paso 8 → `tabla_comparativa.csv`.
3. **Cuaderno 02:** Pasos 6 y 8 → `curvas.png` y `modelo.tflite`. (Saltar el 5.)
4. **Cuaderno 05** completo → complejidad, comparación de versiones, inclinación de fotos propias.
5. Una celda: `!python -m src.informe` y `!python -m src.verificar`.

## 🧍 Falta y solo lo podés hacer vos

| Qué | Dónde |
|---|---|
| Códigos y roles de cada integrante · día y hora de la reunión semanal · direcciones y licencias de las 43 fotos de internet | informe (equipo, roles, sección 10.4) · plan línea 144 |
| Capturas de la app (Figuras 5 y 6) y confirmar que la foto es propia | informe, secciones 7.3 y 10.1 |
| Instalar Python 3.11 y abrir `ejecutar_app.bat` | `docs/GUIA_APP.md` |
| Subir a GitHub | README, al final |
| Ensayar la demostración en vivo | — |

## Lo que el proyecto NO logra, dicho sin rodeos

El modelo detecta 2 a 3 de 13 grietas en fotos de celular. Está documentado como hallazgo, con su causa probable y su
camino de mejora (cuaderno 06, necesita fotos de 10 paredes con grieta y 10 sin ella).
