# 🏗️ Detección de grietas, inclinación y riesgo en edificaciones

**Foto → ¿hay grieta? → ¿qué orientación? → ¿está a plomo? → riesgo BAJO / MEDIO / ALTO**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2%20%2B%20TFLite-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-visi%C3%B3n%20cl%C3%A1sica-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app%20web-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Android](https://img.shields.io/badge/Android-Kotlin%20%2B%20LiteRT-3DDC84?logo=android&logoColor=white)](android/README_ANDROID.md)
[![Pruebas](https://img.shields.io/badge/pruebas-5%2F5%20OK-brightgreen)](tests/test_pipeline.py)

Reto de la mejor implementación · **Algoritmos y Programación 2026-2** · Ingeniería en Inteligencia Artificial · Universidad Industrial de Santander

> ⚠️ **Herramienta orientativa con fines académicos. No sustituye la inspección de un ingeniero civil.** Un nivel "BAJO" no certifica que una edificación sea segura; un nivel "ALTO" significa que vale la pena que la revise un profesional.

---

## 📌 Qué hace

A partir de una fotografía de un elemento de una edificación (columna, viga, muro, placa), el sistema responde tres preguntas y las combina en una recomendación:

| Módulo | Técnica | Salida |
|---|---|---|
| **¿Hay grieta?** | Línea base (descriptores + regresión logística) o **MobileNetV2** con transfer learning | `con grieta` / `sin grieta` + probabilidad |
| **¿Qué orientación tiene?** | Transformada Black-hat + Hough (OpenCV) | `vertical` / `horizontal` / `diagonal` |
| **¿El elemento está a plomo?** | Canny + Hough + mediana ponderada, o inclinómetro del celular | ángulo de desaplome (°) + categoría |
| **¿Qué riesgo hay?** | Reglas de ingeniería (sistema de puntos transparente) | `BAJO` / `MEDIO` / `ALTO` + recomendación + razones |

```mermaid
flowchart LR
    F([📷 foto]) --> C[Clasificador<br/>baseline.joblib · mobilenet.tflite]
    F --> O[Orientación de la grieta<br/>Black-hat + Hough]
    F --> I[Inclinación<br/>Canny + Hough · o sensor]
    E([🧱 elemento<br/>lo elige la persona]) --> R
    C --> R{Reglas de riesgo<br/>risk.py}
    O --> R
    I --> R
    R --> S([BAJO / MEDIO / ALTO<br/>+ recomendación])
```

Se puede usar de cuatro formas: **consola** (`predict.py` → JSON), **app web** (Streamlit), **desde el celular** por Wi-Fi y **app Android nativa** con el modelo `.tflite` corriendo en el teléfono.

---

## 📊 Resultados

### Modelo definitivo

| Métrica | Valor |
|---|---|
| Exactitud | **99,93 %** |
| Precisión | 99,93 % |
| Recall | **99,93 %** |
| F1 | 99,93 % |
| Falsos negativos | **2 de 3.000** (0,07 %) |
| Matriz de confusión `[[VN, FP], [FN, VP]]` | `[[2998, 2], [2, 2998]]` |

Evaluado sobre **6.000 imágenes de prueba** que el modelo nunca vio. Las cuatro métricas coinciden porque los errores quedaron simétricos: dos falsos positivos y dos falsos negativos.

> ⚠️ Este valor **no se sostiene fuera del dominio del dataset**. Ver [Hallazgos](#-hallazgos) y [Limitaciones](#️-limitaciones).

### Tres corridas, un factor a la vez

Para atribuir cada mejora a su causa, se cambió un solo factor por corrida. La partición usa siempre la misma semilla, así que las tres son comparables.

| Corrida | Imágenes | Paciencia | Épocas | Mejor `val_loss` | Exactitud | Recall | Falsos negativos |
|---|---|---|---|---|---|---|---|
| A | 8.000 | 2 | 8 | 0,0117 | 99,67 % | 99,50 % | 3 de 600 (0,50 %) |
| B | 40.000 | 2 | 9 | 0,00599 | 99,87 % | 99,77 % | 7 de 3.000 (0,23 %) |
| **C** | **40.000** | **50** | **100** | **0,00442** | **99,93 %** | **99,93 %** | **2 de 3.000 (0,07 %)** |

| Cambio | Tasa de falsos negativos |
|---|---|
| Más datos (A → B) | 0,50 % → 0,23 % |
| Más paciencia en la parada temprana (B → C) | 0,23 % → 0,07 % |

Las dos contribuciones fueron de magnitud parecida: **siete veces menos grietas no detectadas**, con los mismos 2.259.265 parámetros y 4,46 MB.

Los conjuntos de prueba tienen tamaños distintos (1.200 y 6.000), así que se comparan **tasas**, no conteos absolutos.

### Por qué el recall es la métrica que manda

| Error | Consecuencia |
|---|---|
| Falso positivo | Una inspección innecesaria. Cuesta dinero |
| **Falso negativo** | **Un elemento dañado declarado seguro. Puede costar vidas** |

El modelo devuelve una **probabilidad** entre 0 y 1. El umbral de 0,5 que la convierte en clase lo elige el diseñador, y puede bajarse para priorizar el recall.

### Comparación con la línea base

Ambos modelos se evalúan sobre **las mismas 1.200 fotos de prueba** (corrida A, 4.000 imágenes por clase), para que la comparación sea justa.

| | Línea base | MobileNetV2 (.tflite) |
|---|---|---|
| Descripción | 16 descriptores (Black-hat, Canny, histograma) + regresión logística | ImageNet preentrenado, aumentación, cabeza nueva, fine-tuning de 30 capas, float16 |
| Exactitud | *[pendiente]* | **0,9967** |
| Precisión | *[pendiente]* | 0,9983 |
| Recall | *[pendiente]* | 0,9950 |
| F1 | *[pendiente]* | 0,9967 |
| Matriz de confusión | *[pendiente]* | `[[599, 1], [3, 597]]` |
| Parámetros | 17 | 2.259.265 |
| Tamaño en disco | 1,9 kB | 4,46 MB |
| Interpretable | ✓ | ✗ |

---

## 🔍 Análisis exploratorio

**Dataset:** *Concrete Crack Images for Classification* (Özgenel y Gönenç Sorguç, METU, 2018). 40.000 imágenes de 227 × 227 px en RGB, 20.000 por clase. Balance exacto 1,00:1.

### El brillo separa las clases

Las imágenes con grieta son sistemáticamente más oscuras: la grieta es una línea oscura que baja el promedio. Se midió por dos caminos independientes:

| Medición | Con grieta | Sin grieta | Diferencia |
|---|---|---|---|
| Luminancia ITU-R BT.601, muestra de 2.000 imágenes | 163,4 | 181,8 | 18,4 |
| Promedio simple de RGB, **las 20.000 imágenes** | 162,05 | 180,14 | 18,1 |

Tamaño del efecto de Cohen: **d ≈ 0,85**, considerado grande.

**Consecuencia:** el brillo es un **atajo espurio**. Un clasificador que solo mirara el brillo acertaría bastante por encima del azar sin entender nada de grietas. Una estimación teórica sugiere cerca del 67 %; **está pendiente de verificarse experimentalmente**. Para neutralizarlo, el entrenamiento incluye aumentación de brillo (ver la sección del modelo).

### Estadísticos por canal

| Canal | Media (0-1) | Desviación |
|---|---|---|
| R | 0,6934 | 0,1320 |
| G | 0,6733 | 0,1281 |
| B | 0,6403 | 0,1266 |

Los tres canales son casi iguales porque el concreto es gris: **en este dataset el color aporta muy poca información**. Verificación cruzada: la fórmula de luminancia aplicada a estas medias da 172,3, y el promedio de las dos clases es 172,6. Dos mediciones independientes coinciden al 0,2 %.

### Origen de las imágenes

Las 40.000 imágenes son recortes de **solo 458 fotografías**, cerca de **87 recortes por foto**. Recortes del mismo muro comparten superficie, iluminación y cámara.

La detección de casi-duplicados por hash perceptual encontró **64 grupos redundantes** en una muestra de 4.000 (1,62 %). Es un límite inferior: el hash solo detecta recortes casi idénticos píxel a píxel.

### Nombres de archivo

En la carpeta `Positive`, **9.379 de las 20.000 imágenes terminan en `_1`**, y **ninguna** tiene una gemela con el mismo número sin el sufijo. No son copias: el `_1` forma parte del nombre. Una hipótesis es que marque un segundo lote numerado a continuación del primero; queda por comprobar.

---

## 🧠 El modelo

### Arquitectura

```
Aumentación (solo al entrenar)   volteos, rotación, zoom, brillo, contraste
    ↓
preprocess_input                 píxeles 0-255  →  [-1, 1]
    ↓
MobileNetV2 (ImageNet, include_top=False, en modo inferencia)
    ↓
GlobalAveragePooling2D           1.280 mapas de 7×7  →  1.280 números
    ↓
Dropout(0,3)
    ↓
Dense(1, sigmoide)               →  probabilidad entre 0 y 1
```

**Por qué MobileNetV2** (Sandler et al., Google, CVPR 2018): usa **convolución separable en profundidad**, que divide la convolución en un filtrado espacial por canal (*depthwise*) y una combinación de canales con filtros 1×1 (*pointwise*). Su costo relativo es `1/C_salida + 1/K²`, que con filtros de 3×3 tiende al **11 %** del de una convolución normal. Eso permite correr en celulares de gama baja sin conexión.

### Parámetros

| Grupo | Parámetros | Origen | Durante el entrenamiento |
|---|---|---|---|
| Capas iniciales de la base | 731.584 | ImageNet | Congeladas |
| Últimas 30 capas de la base | 1.526.400 | ImageNet | Afinadas en la fase 2 |
| Clasificador | 1.281 | Aleatorio | Entrenado desde cero |
| **Total** | **2.259.265** | | **1.527.681 entrenables** |

Los 1.281 del clasificador son 1.280 pesos, uno por cada número que entrega el pooling, más un sesgo. Lo que se congela no es la imagen, sino los **pesos** de las capas iniciales.

### Entrenamiento en dos fases

| | Fase 1 | Fase 2 |
|---|---|---|
| Base | Congelada | Últimas 30 capas descongeladas |
| Se entrena | Solo el clasificador | Capas finales y clasificador |
| Tasa de aprendizaje | 1e-3 | **1e-5** |

El clasificador arranca al azar: con la base libre, sus errores grandes destruirían los pesos preentrenados. La tasa cien veces menor en la fase 2 retoca esos pesos sin reescribirlos.

**Configuración común:** Adam, entropía cruzada binaria, lote de 32, Dropout de 0,3 antes de la capa final, y parada temprana sobre la pérdida de validación con `restore_best_weights=True`. La base corre en modo inferencia (`training=False`) incluso durante el ajuste fino, lo que mantiene fijas las estadísticas de sus capas de normalización.

### Aumentación de datos

Se aplica solo durante el entrenamiento, antes de normalizar, sobre píxeles de 0 a 255. En inferencia estas capas no actúan, y el `.tflite` exportado no las incluye.

| Transformación | Parámetro | Rango efectivo |
|---|---|---|
| Volteo | horizontal y vertical | — |
| Rotación | 0,15 | ±54° |
| Zoom | 0,1 | ±10 % |
| Brillo | 0,2 | ±51 niveles sobre 255 |
| Contraste | 0,2 | ±20 % |

**La aumentación de brillo neutraliza el atajo del análisis exploratorio.** La diferencia de brillo entre clases es de unos 18 niveles; la aumentación varía el brillo de cada imagen hasta ±51, casi tres veces más. Así el brillo global deja de servir como pista, mientras se conserva el contraste local: la grieta sigue siendo más oscura que el concreto que la rodea. Queda pendiente verificarlo experimentalmente.

### Partición

Función `dividir()` en `src/dataset.py`: `train_test_split` **aleatorio, estratificado y con semilla fija** (`random_state=42`), en dos cortes.

```
Corte 1:  100 %  →  15 % prueba  +  85 % resto
Corte 2:   85 %  →  17,65 % validación  +  resto entrenamiento
```

El segundo corte pide 17,65 % porque opera sobre el 85 % restante: `0,15 / 0,85 = 0,1765`, que equivale al 15 % del total.

| | 40.000 imágenes | 8.000 imágenes |
|---|---|---|
| Entrenamiento (70 %) | 28.000 | 5.600 |
| Validación (15 %) | 6.000 | 1.200 |
| Prueba (15 %) | 6.000 | 1.200 |

### Complejidad

| Métrica | Valor | Dónde se midió |
|---|---|---|
| Tamaño en Keras | 21,89 MB | — |
| Tamaño en TensorFlow Lite | **4,46 MB** | float16: 2.259.265 × 2 bytes |
| Inferencia por imagen, en lote | 1,57 ms | Colab |
| Inferencia de una imagen, TFLite | 6,73 ms | CPU de Colab |
| Inferencia de una imagen, TFLite | ~25 ms | Computador del equipo |

La conversión a TensorFlow Lite redujo el tamaño **4,9 veces**. Procesar de a una imagen es unas cuatro veces más lento que en lote, porque el costo fijo de iniciar la inferencia no se reparte.

---

## 💡 Hallazgos

**1. La parada temprana cortaba demasiado pronto.** Con paciencia 2, el entrenamiento se detenía en la época 9, durante una meseta temporal. Con paciencia 50 continuó hasta la 100 y la mejor época fue la **91**, con una pérdida de validación 26 % menor. Las diez mejores épocas están todas entre la 72 y la 100: la mejora real llegó después del inicio del ajuste fino.

**2. Cambio de dominio con una imagen sintética.** Una imagen dibujada de demostración es clasificada *con grieta* por la línea base (p = 1,00) y *sin grieta* por MobileNetV2 (p = 0,23). El modelo potente aprendió la textura del concreto real y no reconoce un trazo dibujado; el simple reacciona a cualquier línea oscura.

**3. Falso positivo en una pared real.** Una pared lisa y pintada, **sin grietas**, con una mano encima, fue clasificada *con grieta* con **probabilidad 1,00**. El modelo se especializó en concreto a la vista y no generaliza a superficies distintas. Además falla con máxima confianza, así que un umbral de confianza no lo habría filtrado. Como el modelo se entrenó con aumentación de brillo, la causa más probable no es el brillo, sino una textura que nunca vio y los bordes de la mano, que son líneas de alto contraste, como las grietas.

---

## ⚠️ Limitaciones

- **El 99,93 % está inflado.** Las 40.000 imágenes salen de 458 fotografías; con una partición aleatoria, recortes del mismo muro caen en entrenamiento y en prueba. La partición correcta sería por fotografía de origen, pero el dataset no publica esa correspondencia.
- **Las curvas de entrenamiento no detectan el cambio de dominio**, porque la validación también sale de las mismas 458 fotografías. El problema solo aparece al probar con imágenes externas.
- **Dominio estrecho.** El dataset es concreto liso en primer plano: ladrillo, pañete, pintura, pavimento o fachadas completas están fuera de lo aprendido.
- **Sin fisuras finas.** Todas las grietas del dataset son anchas y de alto contraste; el modelo será menos sensible al daño incipiente.
- **La inclinación por imagen** necesita una arista vertical visible y el celular nivelado; una foto torcida se interpreta como elemento torcido. Mitigación: el inclinómetro del celular.
- **Las reglas de riesgo** son una primera versión heurística, pendiente de revisión con criterio de ingeniería civil.
- **Las fotos de viviendas son datos personales:** la app no las almacena.

---

## 🚀 Inicio rápido

### 1. Instalar (una sola vez)

```bash
git clone https://github.com/davidsayas/Clasificacion-de-grietas.git
cd Clasificacion-de-grietas
python -m venv .venv
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate  (Linux/macOS)
pip install -r requirements.txt
pip install ai-edge-litert      # intérprete para el .tflite (alternativa: pip install tensorflow)
```

### 2. Probar sin dataset (5 minutos)

```bash
python src/make_synthetic_data.py                                  # 400 parches sintéticos
python src/baseline.py --root data/synthetic --out models/baseline.joblib
python src/predict.py docs/demo/columna_grieta_diagonal_+2.5deg.jpg --elemento columna
python tests/test_pipeline.py                                      # 5 × OK
streamlit run src/app_streamlit.py                                 # http://localhost:8501
```

### 3. Flujo completo con el dataset real

1. Descargá [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) de Kaggle (`archive.zip`, ~230 MB).
2. Copiá el zip dentro de `data/`, sin descomprimir.
3. Ejecutá:

```bash
python ejecutar_todo.py
```

Descomprime el dataset, hace el análisis exploratorio, entrena la línea base, predice, corre las pruebas y abre la app web con el mejor modelo disponible. Opciones: `--max-por-clase 1000` (rápido), `--sin-app`, `--zip <ruta>`.

En **VS Code**, la carpeta `.vscode/` trae 12 configuraciones listas (`Ctrl+Shift+D` → elegir → `F5`).

### 4. Entrenar MobileNetV2 (Google Colab, GPU T4)

Subí a Drive `archive.zip` y un zip de `src/`, abrí [`entrenar_mobilenet_colab.ipynb`](entrenar_mobilenet_colab.ipynb) en Colab y ejecutá las celdas. Los checkpoints y el historial por época se guardan directamente en Drive, así que una desconexión no pierde el progreso.

---

## 📱 En el celular

| Opción | Cómo | Cámara | Sin PC |
|---|---|---|---|
| Misma Wi-Fi | `python app_celular.py` → abrir en el teléfono la URL que imprime | vía "Subir archivo → tomar foto" | ✗ |
| Túnel HTTPS | `cloudflared tunnel --url http://localhost:8501` | ✓ | ✗ |
| Streamlit Community Cloud | desplegar `src/app_streamlit.py` desde este repo | ✓ | ✓ |
| **App Android nativa** | [`android/`](android/README_ANDROID.md): Kotlin + `.tflite` + acelerómetro | ✓ | ✓ (sin internet) |

En las opciones por Wi-Fi y túnel, la inferencia corre en el computador y el celular solo muestra la página. En la app nativa, el modelo corre dentro del teléfono.

Guía completa: [`docs/EJECUCION_CELULAR.md`](docs/EJECUCION_CELULAR.md).

---

## 🗂️ Estructura del repositorio

```
Clasificacion-de-grietas/
├── ejecutar_todo.py / .bat        flujo completo con el dataset real
├── app_celular.py / .bat          app web accesible desde el celular
├── entrenar_mobilenet_colab.ipynb cuaderno de entrenamiento para Colab
├── requirements.txt               dependencias
├── .vscode/                       12 configuraciones de ejecución
├── src/
│   ├── make_synthetic_data.py     parches sintéticos para probar el flujo
│   ├── dataset.py                 carga, análisis exploratorio, división 70/15/15
│   ├── features.py                descriptores de la línea base
│   ├── baseline.py                entrenamiento + métricas + complejidad
│   ├── train_transfer.py          MobileNetV2, aumentación, fine-tuning, exportación TFLite
│   ├── inclination.py             ángulo de desaplome (Hough o sensor)
│   ├── risk.py                    orientación de la grieta + reglas de riesgo
│   ├── predict.py                 CLI: foto → JSON
│   └── app_streamlit.py           app web
├── android/                       app Android nativa (Kotlin)
├── tests/                         test_pipeline.py (5 pruebas)
├── models/                        modelos y sus *_metricas.json
├── data/                          raw/ (no se versiona), synthetic/, propias/
└── docs/                          documentación, plan de sprints, demos
```

### Modelos disponibles

| Archivo | Corrida | Uso |
|---|---|---|
| `models/baseline.joblib` | Línea base | Comparación |
| `models/mobilenet.tflite` | A — 8.000 imágenes | Referencia |
| `models/mobilenet_100ep.tflite` | C — 40.000 imágenes, paciencia 50 | **Definitivo** |

---

## 📚 Documentación

| Documento | Contenido |
|---|---|
| [`docs/INFORME_TECNICO.md`](docs/INFORME_TECNICO.md) | Informe técnico de la Entrega 1: problema, estado del arte, datos, modelo, resultados, limitaciones y ética |
| [`docs/DOCUMENTACION.md`](docs/DOCUMENTACION.md) | Entorno, bitácora de problemas y correcciones, ejecución, pendientes |
| [`docs/documentacion_codigo.pdf`](docs/documentacion_codigo.pdf) | Explicación línea a línea del código y de cada decisión (34 páginas, fuente `.tex` incluida) |
| [`docs/EJECUCION_CELULAR.md`](docs/EJECUCION_CELULAR.md) | Las cuatro formas de usar el sistema en el celular |
| [`android/README_ANDROID.md`](android/README_ANDROID.md) | Compilar e instalar la app Android |
| [`docs/SPRINTS.md`](docs/SPRINTS.md) | Plan Scrum: 6 sprints alineados con los tres cortes (25/09, 23/10, 20/11) |
| [`docs/entrega1_formulacion.md`](docs/entrega1_formulacion.md) | Documento de formulación de la Entrega 1 |
| [`data/README.md`](data/README.md) | Cómo obtener el dataset y cómo nombrar las fotos propias |

---

## ⚖️ Reglas de riesgo

| Factor | Puntos |
|---|---|
| Orientación de la grieta | vertical 1 · horizontal 2 · diagonal 3 · desconocida 2 |
| Elemento | muro no estructural 0 · muro de carga 2 · placa 2 · columna 3 · viga 3 |
| Probabilidad ≥ 0,9 | +1 |
| Diagonal en columna / viga / muro de carga (posible cortante) | +2 |
| Horizontal en columna (posible flexión) | +2 |
| Desaplome ≥ 3° / ≥ 1° | +4 / +2 |
| Inclinación no medible en la foto | 0 (se informa) |
| **Nivel** | **ALTO ≥ 7 · MEDIO ≥ 3 · BAJO < 3** |

Reglas orientativas basadas en la práctica de inspección visual y la NSR-10, pendientes de revisión con criterio de ingeniería civil.

### Ejemplo de salida (`predict.py`)

```json
{
  "grieta": {"clase": "con grieta", "probabilidad": 1.0, "orientacion": "diagonal",
             "ms_inferencia": 24.65, "modelo": "tflite"},
  "inclinacion": {"angulo_desaplome": 2.48, "signo": "derecha", "categoria": "inclinado",
                  "n_lineas": 4, "referencia": "vertical", "metodo": "hough"},
  "riesgo": {"nivel": "ALTO", "puntaje": 11,
             "recomendacion": "Restringir el uso del área y solicitar inspección URGENTE de un ingeniero civil...",
             "detalle": {"razones": ["grieta detectada con alta confianza",
                                     "orientación diagonal en columna",
                                     "grieta diagonal en elemento portante: posible falla por cortante",
                                     "desaplome apreciable (2.5°)"]}}
}
```

---

## 🗺️ Estado y hoja de ruta

**Entrega 1 — 25/09/2026**

- [x] Definición del problema, contexto y clases
- [x] Dataset descrito y análisis exploratorio
- [x] Revisión del estado del arte
- [x] Línea base con descriptores hechos a mano
- [x] MobileNetV2 con transfer learning, fine-tuning y exportación a TFLite
- [x] Aumentación de datos: volteos, rotación, zoom, brillo y contraste
- [x] Tres corridas comparadas: datos y paciencia de la parada temprana
- [x] Orientación e inclinación con visión clásica; inclinómetro como alternativa
- [x] Reglas de riesgo y prototipo de extremo a extremo (CLI + app web)
- [x] Acceso desde el celular por Wi-Fi; código de la app Android
- [ ] Métricas de la línea base con datos reales en la tabla comparativa
- [ ] Estimación de inclinación sobre una fotografía propia

**Entrega 2 — 23/10/2026**

- [ ] Cuaderno documentado con el preprocesamiento y la aumentación
- [ ] Prueba del atajo: verificar si la aumentación de brillo neutralizó la dependencia del brillo
- [ ] Verificación experimental del clasificador por umbral de brillo
- [ ] Corrida sin aumentación, para medir su aporte (opcional)
- [ ] Salida de campo: fotografías propias en dos grupos, uno para entrenar y otro solo para evaluar
- [ ] Evaluación del modelo actual sobre SDNET2018, como prueba fuera del dominio
- [ ] Reentrenamiento con superficies lisas, pintadas y pañetadas etiquetadas como sin grieta
- [ ] Barrido de tasa de aprendizaje en el ajuste fino

**Entrega 3 — 20/11/2026**

- [ ] URL pública en Streamlit Community Cloud y APK compilado
- [ ] Validación final con fotografías propias
- [ ] Revisión de las reglas de riesgo con Ingeniería Civil (NSR-10)
- [ ] Sustentación con demostración en vivo

---

## 👥 Equipo

| Integrante | Rol |
|---|---|
| Juan David Sayas Hernández | Product owner, developer |
| Erick Fabian Cárdenas Bello | Developer |

Docente: Jheyston Omar Serrano Luna · Curso: Algoritmos y Programación 2026-2 · Programa: Ingeniería en Inteligencia Artificial, UIS.

## 📄 Licencia y créditos

Código bajo licencia MIT (ver `LICENSE`). Dataset *Concrete Crack Images for Classification* — Çağlar Fırat Özgenel, Middle East Technical University, licencia CC BY 4.0. MobileNetV2 — Sandler et al., 2018 (pesos de ImageNet vía Keras).
