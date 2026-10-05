# Grietas y riesgo

Detección de grietas en concreto, medición de su ancho, inclinación del muro y nivel de riesgo.
Proyecto del reto de *Algoritmos y Programación* 2026-2 · Ingeniería en Inteligencia Artificial · UIS.

Repositorio: <https://github.com/davidsayas/Clasificacion-de-grietas>

> **Herramienta de tamizaje.** No reemplaza una inspección estructural. Los umbrales de riesgo de
> `src/riesgo.py` son orientativos y deben validarse con la norma aplicable y con tu docente.

## Cómo funciona

```
foto ─► ¿parece concreto? ─► ¿hay grieta?  ─► ¿qué ancho tiene? ─┐
        (puerta, ood.py)     (MobileNetV2,     (ancho.py,         ├─► nivel de riesgo
                              mosaico si la    escala con una      │   (riesgo.py)
                              foto es grande)  tarjeta)            │
        inclinación del muro (inclinacion.py, corregida con el sensor de gravedad) ┘
```

Si la imagen no parece concreto o el modelo duda, el resultado es **INDETERMINADO** y se pide otra foto.
Ante la duda, el sistema no afirma.

## Estructura

```
src/
  config.py        constantes (semilla, tamaño de imagen, umbrales)
  datos.py         inventario, grupos, división SIN FUGA, duplicados, auditoría de fuentes
  eda.py           análisis exploratorio (brillo, clasificador tonto, tamaño, color)
  linea_base.py    métodos sin red neuronal para la tabla comparativa
  modelo.py        MobileNetV2 / EfficientNetB0, aumentación, congelado, exportación TFLite
  entrenar.py      entrenamiento en dos fases, barrido de tasa de aprendizaje
  evaluar.py       métricas + pruebas de robustez (brillo, giro, lejanía, perspectiva...)
  gradcam.py       en qué se fija el modelo
  ood.py           puerta "¿parece concreto?"
  mosaico.py       fotos de lejos o grandes: clasificar por teselas
  unet.py          segmentación píxel a píxel (modelo extra)
  ancho.py         ancho de la grieta en mm + corrección de perspectiva
  inclinacion.py   inclinación del muro + corrección con el sensor de gravedad
  riesgo.py        motor de riesgo
  predecir.py      el sistema completo
  complejidad.py   parámetros, MB y tiempo de inferencia (15 % de la rúbrica)
  fotos_propias.py desempeño con las fotos del equipo (forzado y con abstención)
  informe.py       secciones de resultados del informe, desde results/*.csv
  verificar.py     checklist de las 3 entregas según la evidencia real
notebooks/         01_EDA · 02_Entrenamiento · 03_Robustez_y_falso_positivo · 04_Segmentacion_y_ancho · 05_Complejidad_y_despliegue · 06_Mejora_con_fotos_de_celular
tests/             105 pruebas con datos sintéticos
docs/              CORRECCION_FALSO_POSITIVO.md · PROTOCOLO_FOTOS_PROPIAS.md · ACTUALIZACION_INFORME_v2.md
INFORME_TECNICO.md · PLAN_DE_TRABAJO.md · CHECKLIST.md
app/               streamlit_app.py
```

## Flujo de trabajo

1. En tu Drive: subí el `.zip` de **Surface Crack** (con `Positive/` y `Negative/`) como `surface_crack.zip` y esta carpeta como
   `grietas-v2/`. Tus fotos van en `grietas-v2/data/` (`dificiles/`, `campo/`, `fotos_propias/`, `problemas/`).
   La **primera celda** de cada cuaderno monta Drive y pone los datos en el disco local; solo editás `PROYECTO` y `ZIP_SURFACE`.
2. Abrí los cuadernos **en orden** en Colab. Cada paso explica qué hace y por qué.
3. `01_EDA` → particiones sin fuga y tabla de línea base.
4. `02_Entrenamiento` → MobileNetV2 y EfficientNetB0 con la misma división.
5. `03_Robustez_y_falso_positivo` → resultados en prueba, atajo del brillo, Grad-CAM, puerta, fuera de dominio.
6. `04_Segmentacion_y_ancho` → U-Net y medición del ancho (opcional; necesita máscaras).
7. `05_Complejidad_y_despliegue` → parámetros, MB, tiempo, variantes TFLite, comparación entre versiones, informe y verificación.

```bash
pip install -r requirements.txt
python tests/run_all.py            # 105 pruebas, sin necesidad de pytest
python -m src.verificar            # estado de las 3 entregas según la evidencia
python -m src.informe              # secciones de resultados del informe
python tests/humo_tf_simulado.py   # revisa el código de TensorFlow con un TensorFlow simulado
streamlit run app/streamlit_app.py     # o doble clic en ejecutar_app.bat (ver docs/GUIA_APP.md)
```

## Qué se corrigió respecto de la versión 1

| Problema | Corrección |
|---|---|
| El 99,93 % estaba **inflado**: 40.000 recortes salen de 458 fotos y recortes del mismo muro caían en entrenamiento y prueba | División por **grupos** con verificación automática de que no hay fuga (`datos.dividir`) |
| Hipótesis "los archivos consecutivos son de la misma foto" sin verificar | `datos.prueba_adyacencia` la contrasta; si falla, `asignar_grupos_cluster` |
| Historial con 107 filas para 100 épocas (`CSVLogger append=True` + intento interrumpido) | Un historial por fase con `append=False` |
| Sin forma de saber si el brillo es un atajo | `evaluar.prueba_atajo_brillo`, validada con modelos falsos que sí y que no usan el brillo |
| Falso positivo con pared lisa y mano | Negativos difíciles + puerta de concreto + banda de duda (ver `docs/`) |
| Código del entrenamiento sin explicar | Docstrings y cuadernos con *qué hace / por qué / qué decidir* |

## Resultados

### Versión 1 — **no comparable**: la división tenía fuga

| Corrida | Imágenes | Épocas | val_loss | Exactitud | Recall | Falsos neg. |
|---|---|---|---|---|---|---|
| A | 8.000 | 8 | 0,0117 | 99,67 % | 99,50 % | 3 / 600 |
| B | 40.000 | 9 | 0,00599 | 99,87 % | 99,77 % | 7 / 3.000 |
| C | 40.000 | 100 | 0,00442 | 99,93 % | 99,93 % | 2 / 3.000 |

Estas cifras miden cuánto recuerda el modelo de muros que ya vio, no cuánto generaliza.

### Versión 2 — **pendiente**

Se completan al correr el cuaderno 03 sobre la partición de prueba sin fuga:

| Modelo | Exactitud | Recall | Precisión | FN | FP |
|---|---|---|---|---|---|
| Línea base: umbral de brillo | _pendiente_ | | | | |
| Línea base: umbral de bordes | _pendiente_ | | | | |
| Línea base: regresión logística | _pendiente_ | | | | |
| MobileNetV2 | _pendiente_ | | | | |
| EfficientNetB0 | _pendiente_ | | | | |

Se espera que queden **por debajo** del 99,93 %: la diferencia es la fuga eliminada, no un empeoramiento.

## Limitaciones conocidas

* Pared lisa pintada con objetos: falso positivo observado en la versión 1; se corrige con datos y la puerta, **y se mide**.
* El ancho en mm exige un objeto de tamaño conocido **en el mismo plano** que la grieta; por debajo de ~3 px de ancho el desenfoque domina.
* La inclinación no es confiable si el celular apunta hacia arriba o hacia abajo (las verticales convergen).
* Los umbrales de riesgo son orientativos.

## Subir a GitHub

```bash
git init
git remote add origin https://github.com/davidsayas/Clasificacion-de-grietas.git
git add .
git commit -m "Proyecto reorganizado: división sin fuga, pruebas de robustez, puerta de concreto"
git branch -M main
git pull origin main --allow-unrelated-histories   # trae el README que ya existe
git push -u origin main
```
