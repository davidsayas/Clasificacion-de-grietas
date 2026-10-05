# Detección de Grietas e Inclinación de Elementos Estructurales para la Evaluación de Riesgo en Edificaciones

**Universidad Industrial de Santander — Facultad de Fisicomecánicas**
**Programa:** Ingeniería en Inteligencia Artificial
**Curso:** Algoritmos y Programación — Periodo 2026-2
**Docente:** Jheyston Omar Serrano
**Entrega:** 1 a 3 — versión 2 (corrección de la fuga, fotografías propias y análisis de generalización)
**Equipo:** Juan David Sayas Hernandez (código `[COMPLETAR]`) · Erick Fabian Cardenas Bello (código `[COMPLETAR]`)
**Repositorio:** https://github.com/davidsayas/Clasificacion-de-grietas

---

## Resumen

Se presenta un prototipo de extremo a extremo para el tamizaje del estado estructural de edificaciones a partir de fotografías. El sistema combina dos capacidades: un clasificador de imágenes basado en MobileNetV2, entrenado por transferencia de aprendizaje, que detecta la presencia de grietas en superficies de concreto; y un módulo de estimación del desaplome basado en detección de bordes (Canny) y transformada de Hough. Ambas salidas alimentan un motor de reglas que emite una recomendación orientativa de riesgo.

Se realizaron tres corridas de entrenamiento variando un factor a la vez. Ampliar el conjunto de 8.000 a 40.000 imágenes y corregir la paciencia de la parada temprana redujo la tasa de falsos negativos de 0,50 % a 0,07 %, con los mismos 2.259.265 parámetros y 4,46 MB en TensorFlow Lite. El modelo final alcanza 99,93 % de exactitud y recall en el conjunto de prueba. Se documenta además que ese desempeño no se sostiene fuera del dominio del dataset: sobre una pared lisa y pintada, el modelo produjo un falso positivo con probabilidad 1,00.

**Actualización de la versión 2.** La partición de la versión 1 repartía recortes de las mismas fotografías entre entrenamiento y prueba (con la división imagen por imagen, 913 de 913 grupos aparecían en ambos), por lo que sus cifras no son comparables con las de la versión 2, que divide por grupos y verifica que la prueba no comparte imágenes ni grupos con lo que el modelo vio. Sobre esa prueba limpia (6.052 imágenes de Surface Crack) el modelo alcanza 99,95 % de exactitud, con 2 grietas no detectadas de 2.928. Sin embargo, en 29 fotografías propias de celular (13 con grieta) detecta entre 2 y 3 grietas con el umbral habitual y no da ninguna falsa alarma: el modelo acierta todas las fotos propias con las que entrenó y casi ninguna de las nuevas, lo que indica memorización de pocas fotos, diferencia de dominio y una posible confusión de fuente (las fotos con grieta de entrenamiento provienen de internet y las sin grieta, de celular). Un componente diseñado para reducir falsas alarmas, la puerta de concreto, empeoró el resultado y se retiró. El mayor avance disponible es ampliar los datos con fotografías de celular de paredes distintas.

**Palabras clave:** visión por computador, redes convolucionales, transferencia de aprendizaje, MobileNetV2, convolución separable en profundidad, transformada de Hough, desaplome.

---

## 1. Introducción y planteamiento del problema

### 1.1 Contexto

Las grietas en muros, columnas, vigas y placas son el primer indicador visible del deterioro de una edificación. Algunas son superficiales y de origen estético; otras revelan asentamientos diferenciales, sobrecargas o corrosión del acero de refuerzo. A ellas se suma el desaplome, la pérdida de verticalidad de los elementos, asociado a asentamientos o pandeo.

Colombia tiene amenaza sísmica intermedia a alta en gran parte de su territorio, y ciudades como Barrancabermeja concentran un porcentaje elevado de construcción informal. En ese contexto el cuello de botella no es el conocimiento técnico sino la **cobertura**: no hay suficientes ingenieros para inspeccionar el parque construido, y una inspección profesional tiene un costo que muchas familias no asumen de forma preventiva.

### 1.2 El problema

Una herramienta de tamizaje automatizado no reemplaza al ingeniero, pero puede **priorizar**: identificar qué edificaciones merecen una visita profesional a partir de fotografías tomadas con un celular.

**Pregunta que guía el proyecto:** ¿es posible construir un sistema de bajo costo computacional que, a partir de una fotografía, detecte daño visible y estime la inclinación de un elemento, y traduzca ambas mediciones en una recomendación de riesgo?

### 1.3 Clases a predecir

El nivel implementado en esta entrega es **binario**:

| Clase | Descripción |
|---|---|
| `Negative` — sin grieta | Superficie sin discontinuidad visible |
| `Positive` — con grieta | Discontinuidad lineal visible en la superficie |

La orientación de la grieta (vertical, horizontal, diagonal) se estima como complemento en la aplicación, a partir del análisis de bordes.

### 1.4 Alcance

**Incluido:** clasificación de grietas, estimación del desaplome, motor de riesgo por reglas, despliegue funcional en computador.

**Excluido:** diagnóstico estructural certificado, análisis de esfuerzos, detección de daño no visible (corrosión interna, fallas de cimentación sin manifestación superficial).

---

## 2. Objetivos

### 2.1 Objetivo general

Diseñar, entrenar y desplegar un modelo de clasificación de imágenes que, combinado con la estimación de inclinación de elementos estructurales, apoye la estimación de un nivel de riesgo, comprendiendo todo el ciclo: captura, análisis, entrenamiento, evaluación y despliegue.

### 2.2 Objetivos específicos

1. Seleccionar y caracterizar un conjunto de datos libre de imágenes de grietas, documentando su origen y sus sesgos.
2. Analizar y preparar las imágenes mediante exploración, redimensionamiento y normalización.
3. Entrenar un clasificador por transferencia de aprendizaje y compararlo contra una línea base.
4. Estimar el desaplome de elementos verticales mediante detección de líneas.
5. Desplegar el modelo en una aplicación funcional y probarlo con fotografías externas al dataset.
6. Analizar la complejidad computacional y las limitaciones de la solución.

---

## 3. Estado del arte

### 3.1 Detección automática de grietas

La detección de grietas en infraestructura pasó por tres etapas metodológicas.

**Procesamiento clásico de imágenes.** Umbralización adaptativa, filtros morfológicos y detectores de bordes como Canny o Sobel. Funcionan en superficies homogéneas y bien iluminadas, pero no distinguen una grieta de otras discontinuidades de intensidad: manchas, sombras, juntas de construcción o humedad.

**Aprendizaje automático con características diseñadas a mano.** Extracción de descriptores (HOG, LBP, wavelets) seguida de un clasificador como SVM o Random Forest. Mejora la robustez, pero el desempeño queda limitado por la calidad de los descriptores elegidos.

**Aprendizaje profundo.** Redes convolucionales que aprenden las características directamente de los píxeles. Cha, Choi y Büyüköztürk (2017) demostraron con una CNN entrenada desde cero exactitudes superiores al 98 % en parches de concreto, superando ampliamente a Canny y Sobel en condiciones adversas. La aparición de conjuntos etiquetados de gran tamaño (Özgenel, 2018; SDNET2018) y la transferencia de aprendizaje desde ImageNet desplazaron el entrenamiento desde cero: hoy es práctica estándar partir de una red preentrenada.

La tendencia actual se mueve de la clasificación hacia la segmentación, que localiza la grieta píxel a píxel y permite estimar su ancho, y hacia la ejecución en el propio dispositivo mediante modelos compactos.

### 3.2 Redes ligeras para dispositivos móviles

**MobileNetV1** (Howard et al., 2017) introdujo la convolución separable en profundidad, que divide la convolución estándar en un filtrado espacial por canal seguido de una combinación de canales con filtros de 1×1. Alcanzó 70,6 % de exactitud top-1 en ImageNet con 4,2 millones de parámetros.

**MobileNetV2** (Sandler et al., 2018) añadió los bloques de residuales invertidos y los cuellos de botella lineales, llegando a 72,0 % con 3,4 millones de parámetros y cerca de la mitad de operaciones que su antecesora. **MobileNetV3** (Howard et al., 2019) mejoró la exactitud en 3,2 % y redujo la latencia en 20 % mediante búsqueda automática de arquitectura.

Otras alternativas ligeras son SqueezeNet, ShuffleNet, MnasNet y EfficientNet-Lite. Como contraste, VGG16 tiene cerca de 138 millones de parámetros: unas 60 veces el modelo de este proyecto.

### 3.3 Estimación del desaplome

En la práctica profesional el desaplome se mide con plomada, nivel digital o estación total. Las alternativas de bajo costo son dos: la detección de aristas verticales mediante transformada de Hough, midiendo su desviación respecto a la vertical de la imagen, y los sensores inerciales del celular, que estiman el vector gravedad.

### 3.4 Brecha que aborda el proyecto

La mayoría de trabajos evalúa solo el módulo de grietas, sobre concreto de infraestructura formal. Este proyecto integra grietas e inclinación en un único criterio de riesgo, y documenta explícitamente el comportamiento del modelo fuera del dominio del dataset de entrenamiento.

---

## 4. Datos

### 4.1 Selección y justificación

| Atributo | Valor |
|---|---|
| Nombre | Concrete Crack Images for Classification |
| Autores | Özgenel y Gönenç Sorguç (2018) |
| Fuente | Mendeley Data / Kaggle "Surface Crack Detection" |
| Volumen | 40.000 imágenes |
| Resolución | 227 × 227 px, RGB |
| Clases | `Positive` y `Negative`, 20.000 cada una |
| Origen | Recortes de **458 fotografías** de alta resolución de edificaciones del campus de la Universidad Técnica de Medio Oriente (METU), Ankara |

**Justificación.** Se eligió por cuatro razones: está balanceado, lo que simplifica el entrenamiento y la interpretación de métricas; su resolución es cercana a la entrada de MobileNetV2 (224×224), por lo que el redimensionado casi no pierde detalle; su volumen permite entrenar sin sobreajuste severo; y es el conjunto de referencia más citado en la literatura del área.

**Alternativas consideradas.** SDNET2018 ofrece más diversidad pero está fuertemente desbalanceado. CODEBRIM es multi-etiqueta y más cercano al riesgo real, pero más complejo para una primera implementación. CrackForest y DeepCrack son de segmentación.

### 4.2 Análisis exploratorio

El análisis se realizó con el script `eda_grietas.py`. Las figuras se encuentran en `results/figuras/`.

#### Balance y formato

El conjunto contiene 20.000 imágenes por clase, con razón de desbalance de **1,00:1**. Todas son RGB de 227×227 px, por lo que el redimensionado a 224×224 implica una pérdida del 1,3 % del lado.

*Figura 1 — `01_balance_clases.png`*

Con clases balanceadas la exactitud es una métrica interpretable directamente. Aun así se reportan precisión, recall y F1, que son robustas al desbalance.

#### Inspección visual

*Figura 2 — `02_ejemplos_por_clase.png`*

Dos hallazgos:

- **Todas las grietas son anchas y de alto contraste.** No hay fisuras finas ni incipientes, que son justamente las que interesan para el diagnóstico temprano.
- **En la clase `Negative` aparecen texturas lineales y manchas oscuras** susceptibles de generar falsos positivos.

#### Distribución de luminancia

*Figura 3 — `03_histograma_intensidad.png`*

| Clase | Luminancia media | Desviación |
|---|---|---|
| Sin grieta | 181,8 | 22,8 |
| Con grieta | 163,4 | 20,2 |

Las imágenes con grieta son sistemáticamente más oscuras. La diferencia de 18,4 puntos corresponde a un tamaño de efecto de Cohen de **d ≈ 0,85**, considerado grande.

La explicación es física: la grieta es una línea oscura que baja el brillo promedio. Pero tiene una consecuencia metodológica. Un clasificador que solo mire la intensidad media, con umbral en 172,6, acertaría bastante por encima del azar sin modelar la forma de la grieta; una estimación teórica sugiere cerca del 67 %, pendiente de verificar experimentalmente. Esto señala un **atajo espurio** que el modelo podría explotar. El entrenamiento lo neutraliza con aumentación de brillo (sección 5.6).

#### Estadísticos por canal

| Canal | Media (0-1) | Desviación |
|---|---|---|
| R | 0,6934 | 0,1320 |
| G | 0,6733 | 0,1281 |
| B | 0,6403 | 0,1266 |

**Verificación cruzada.** Aplicando la fórmula de luminancia ITU-R BT.601 a estas medias:

```
L = 0,299(0,6934) + 0,587(0,6733) + 0,114(0,6403) = 0,6755
0,6755 × 255 = 172,3
```

El promedio de las dos clases medidas por el otro camino es 172,6. La diferencia es de 0,3 sobre 255: dos mediciones independientes coinciden al 0,2 %.

Estos valores documentan el conjunto, pero **no se usan para normalizar**: MobileNetV2 exige su propia función `preprocess_input`.

#### Casi-duplicados

La detección por hash perceptual sobre 4.000 imágenes encontró **64 grupos redundantes** (1,62 %). Esta cifra es un límite inferior: el hash solo detecta recortes casi idénticos píxel a píxel.

El problema real es de mayor escala. Las 40.000 imágenes salen de 458 fotografías, cerca de **87 recortes por fotografía**, que comparten superficie, iluminación y cámara. Ese es el origen de la principal limitación del trabajo (sección 10).

### 4.3 Preprocesamiento

| Paso | Operación | Justificación |
|---|---|---|
| 1 | Redimensionado a 224 × 224 | Entrada de MobileNetV2 |
| 2 | Conversión a float32 | Requisito de la convolución |
| 3 | `preprocess_input`: `x / 127,5 − 1` | Lleva los píxeles a [-1, 1] |

**Por qué [-1, 1].** Dividir entre 127,5 ajusta el ancho del rango de 255 a 2; restar 1 lo centra en cero. Centrar en cero acelera la convergencia. Pero la razón decisiva es otra: MobileNetV2 fue preentrenada con ese rango, y entregarle otro invalida los pesos de ImageNet sin producir ningún error visible.

La normalización conserva la separación entre clases: en unidades de desviación estándar, la distancia es 0,87 antes y después.

### 4.4 Partición

La partición la realiza la función `dividir()` de `src/dataset.py`, en dos cortes aleatorios y estratificados con semilla fija (`random_state=42`):

```
Corte 1:  100 %  →  15 % prueba  +  85 % resto
Corte 2:   85 %  →  17,65 % validación  +  resto entrenamiento
```

El segundo corte pide 17,65 % y no 15 % porque opera sobre el 85 % restante: `0,15 / 0,85 = 0,1765`, que equivale al 15 % del total.

| Partición | Proporción | 40.000 img | 8.000 img |
|---|---|---|---|
| Entrenamiento | 70 % | 28.000 | 5.600 |
| Validación | 15 % | 6.000 | 1.200 |
| Prueba | 15 % | 6.000 | 1.200 |

**Tres propiedades y por qué importan:**

- **Aleatoria:** evita que el orden de los archivos sesgue la partición.
- **Estratificada:** cada conjunto conserva el balance 50/50.
- **Semilla fija:** la partición es idéntica en todas las corridas. Sin esto, las tres corridas del proyecto no serían comparables.

El conjunto de prueba se evalúa una sola vez, al final. Usarlo para tomar decisiones lo convertiría en un segundo conjunto de validación e inflaría las métricas.

### 4.4.1 Partición de la versión 2: por grupos

Los 40.000 recortes de Surface Crack salen de 458 fotografías, cerca de 87 recortes por foto. Una partición aleatoria imagen por imagen reparte recortes del mismo muro entre entrenamiento y prueba, de modo que el modelo se evalúa con "hermanos" de lo que ya vio y el resultado se infla. **Con la división de la versión 1, los 913 grupos del inventario aparecían a la vez en entrenamiento y en prueba.**

La versión 2 divide por **grupos**: todas las imágenes de un mismo grupo caen en la misma partición, y el código verifica automáticamente que ningún grupo ni imagen aparece en dos particiones (`datos.verificar_sin_fuga`). Los grupos son bloques de 44 archivos consecutivos por clase, bajo la hipótesis de que la numeración sigue a la fotografía de origen. La hipótesis se contrastó comparando la similitud de color entre archivos consecutivos (mediana 0,69) y entre pares al azar (mediana 1,36): la razón es **0,50**, es decir, los consecutivos se parecen el doble. Las fracciones son 70 / 15 / 15 con semilla 42.

**Control de integridad de la prueba.** Como `results/particion_prueba.csv` cambia si se repite el análisis exploratorio después de entrenar, antes de evaluar se compara la prueba con las particiones con las que el modelo realmente se entrenó (`models/<modelo>/particion_*.csv`). Resultado: **0 imágenes y 0 grupos en común**. La prueba verificada tiene 6.052 imágenes, todas de Surface Crack (3.124 sin grieta y 2.928 con grieta).

**Límites.** Los grupos aproximan las fotografías de origen, que el dataset no identifica. Además, los bloques se arman por clase, así que recortes de una misma foto con grieta y sin grieta podrían caer en particiones distintas. El resultado sigue siendo optimista respecto de muros nunca vistos; por eso se evalúa además con fotografías propias (sección 10.4). Duplicados: la huella perceptual halla 1,52 % de imágenes casi repetidas en una muestra de 4.000 (cota inferior).

---

## 5. Módulo A — Clasificación de grietas

### 5.1 Línea base

Descriptores de imagen con regresión logística.

**Resultados:** | Método (ajustado con 6.000 imágenes de entrenamiento) | Exactitud | Recall | Precisión | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|
| Umbral de brillo | 70,27 % | 73,92 % | 68,37 % | 71,04 % | 386 | 506 |
| Umbral de bordes (Canny) | 84,87 % | 96,69 % | 77,94 % | 86,31 % | 49 | 405 |
| Regresión logística (5 rasgos simples) | **98,27 %** | 98,18 % | 98,31 % | 98,24 % | 27 | 25 |

Medidas sobre una muestra de 3.000 imágenes de la partición de prueba. **Lectura:** un clasificador clásico con solo cinco rasgos calculados a mano ya alcanza 98,27 %, así que Surface Crack es un conjunto fácil. La red reduce la tasa de error de 1,73 % a 0,05 % (unas 35 veces menos errores; comparación aproximada porque la línea base se midió en una muestra). El brillo solo acierta 70 %: existe una diferencia entre clases, pero está lejos de explicar la clasificación. Esto sitúa el 99,95 % de la red: su mérito en el dominio del dataset es real pero acotado, y lo que decide la utilidad del sistema es el comportamiento fuera de ese dominio.

### 5.2 Arquitectura: MobileNetV2 por transferencia de aprendizaje

**Por qué MobileNetV2.** El usuario objetivo es construcción informal. Un modelo que exige un servidor con GPU excluye a quien se quiere atender. MobileNetV2 corre sin conexión en celulares de gama baja, y tiene el soporte más maduro en TensorFlow Lite, lo que reduce el riesgo del despliegue. Para una tarea binaria, la ganancia de exactitud de modelos más recientes sería marginal.

**Configuración:**

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

### 5.3 La convolución separable en profundidad

Una convolución estándar filtra espacialmente y combina canales en una sola operación. MobileNet la separa en dos pasos:

- **Depthwise:** un filtro de 3×3 por cada canal de entrada, trabajando aislado. Mira vecinos, no mezcla canales.
- **Pointwise:** filtros de 1×1 que combinan los canales. Mezclan canales, no miran vecinos.

**Costo en parámetros**, con K el tamaño del filtro:

| Tipo | Fórmula |
|---|---|
| Normal | K² · C_entrada · C_salida |
| Depthwise | K² · C_entrada |
| Pointwise | C_entrada · C_salida |

La razón entre ambas, al cancelarse `C_entrada`:

```
separable / normal  =  1/C_salida + 1/K²
```

Con filtros de 3×3, la razón tiende a **1/9 = 11,1 %** a medida que crecen los canales de salida. En bloques reales de MobileNetV2:

| Bloque | Separable | Normal equivalente | Costo |
|---|---|---|---|
| 24 → 32 | 5.904 | 41.472 | 14,2 % |
| 96 → 160 | 97.344 | 829.440 | 11,7 % |
| 160 → 320 | 315.840 | 2.764.800 | 11,4 % |

*Figuras de apoyo en `results/figuras/costo/` y animaciones en `results/figuras/`, generadas con los scripts `animacion_normal.py`, `animacion_separable.py` y `comparacion_costo.py`.*

### 5.4 Parámetros del modelo

**Total: 2.259.265 parámetros**, en tres grupos:

| Grupo | Parámetros | Origen | Durante el entrenamiento |
|---|---|---|---|
| Capas iniciales de la base | 731.584 | ImageNet | Congeladas |
| Últimas 30 capas de la base | 1.526.400 | ImageNet | Afinadas en la fase 2 |
| Clasificador | 1.281 | Aleatorio | Entrenado desde cero |

**Entrenables: 1.527.681.** Los 1.281 del clasificador son 1.280 pesos —uno por cada característica que entrega el pooling— más un sesgo.

**Qué se congela y qué no.** No se congela nada de la imagen: la foto atraviesa todas las capas. Se congelan los **pesos** de las capas iniciales, que detectan bordes y texturas genéricas y sirven igual para cualquier imagen. Las capas finales, que combinan esas formas en conceptos, sí se afinan porque ahí "grieta" es distinto de otras categorías.

*Figuras de apoyo en `results/figuras/congelado/`.*

### 5.5 Entrenamiento en dos fases

| | Fase 1 | Fase 2 |
|---|---|---|
| Base | Congelada | Últimas 30 capas descongeladas |
| Se entrena | Solo el clasificador | Capas finales y clasificador |
| Tasa de aprendizaje | 1e-3 | **1e-5** |

**Por qué ese orden.** El clasificador arranca con valores aleatorios y se equivoca mucho. Si la base estuviera libre desde el inicio, esos errores grandes se propagarían y destruirían los pesos preentrenados.

**Por qué la tasa cien veces menor en la fase 2.** Con 1e-3 los pesos preentrenados se alterarían demasiado en las primeras iteraciones. El ajuste fino debe ser un retoque, no un reentrenamiento.

**Configuración común:** optimizador Adam, entropía cruzada binaria, lote de 32, Dropout de 0,3 antes de la capa final, y parada temprana sobre la pérdida de validación con restauración de los mejores pesos (`restore_best_weights=True`).

**La base corre en modo inferencia** (`training=False`) incluso durante el ajuste fino. Así sus capas de normalización conservan las estadísticas aprendidas en ImageNet, que con lotes chicos se estropearían. Es parte de la razón por la que el pico del ajuste fino se recupera rápido.

### 5.6 Aumentación de datos

Se aplica solo durante el entrenamiento, **antes** de normalizar, sobre píxeles de 0 a 255. En inferencia estas capas no actúan, y el modelo exportado a TensorFlow Lite no las incluye.

| Transformación | Parámetro | Rango efectivo | Justificación |
|---|---|---|---|
| Volteo | horizontal y vertical | — | Una grieta no tiene lado preferido |
| Rotación | 0,15 | ±54° | Las grietas pueden ir en cualquier dirección |
| Zoom | 0,1 | ±10 % | Distancias de captura distintas |
| Brillo | 0,2 | ±51 niveles sobre 255 | Condiciones de luz distintas al dataset |
| Contraste | 0,2 | ±20 % | Cámaras y exposiciones distintas |

**La aumentación de brillo neutraliza el atajo del análisis exploratorio.** La diferencia de brillo entre clases es de unos 18 niveles; la aumentación desplaza el brillo de cada imagen hasta ±51, casi tres veces más. Durante el entrenamiento el modelo ve imágenes con grieta más claras que una sin grieta típica, y al revés: el brillo global deja de servir como pista.

Lo que se conserva es el **contraste local**. Sumar la misma cantidad a todos los píxeles aclara la imagen entera, pero la grieta sigue siendo más oscura que el concreto que la rodea. Se pierde la pista espuria y se mantiene la señal real: la forma de la grieta.

Queda pendiente verificarlo experimentalmente con la prueba del atajo (sección 13).

---

## 6. Resultados

> **Aviso.** Las cifras de las secciones 6.1 a 6.5 son de la **versión 1**, cuya partición tenía fuga (ver 4.4.1); no son comparables con la versión 2. Los resultados de la versión 2 están en la sección 6.6.

### 6.1 Tres corridas, un factor a la vez

| Corrida | Imágenes | Paciencia | Épocas | Mejor val_loss |
|---|---|---|---|---|
| A | 8.000 | 2 | 8 | 0,0117 |
| B | 40.000 | 2 | 9 | 0,00599 |
| C | 40.000 | 50 | 100 | **0,00442** |

**Métricas sobre el conjunto de prueba:**

| Corrida | Exactitud | Precisión | Recall | F1 | Falsos negativos |
|---|---|---|---|---|---|
| A | 99,67 % | 99,83 % | 99,50 % | 99,67 % | 3 de 600 (0,50 %) |
| B | 99,87 % | 99,97 % | 99,77 % | 99,87 % | 7 de 3.000 (0,23 %) |
| **C** | **99,93 %** | **99,93 %** | **99,93 %** | **99,93 %** | **2 de 3.000 (0,07 %)** |

**Atribución de cada mejora:**

| Cambio | Tasa de falsos negativos |
|---|---|
| Más datos (A → B) | 0,50 % → 0,23 % |
| Más paciencia (B → C) | 0,23 % → 0,07 % |

Las dos contribuciones fueron de magnitud parecida: ninguna sola habría bastado. En conjunto, **siete veces menos grietas no detectadas**.

**Nota sobre la comparación.** Los conjuntos de prueba tienen tamaños distintos (1.200 y 6.000), por lo que se comparan las **tasas**, no los conteos absolutos.

### 6.2 La parada temprana cortaba demasiado pronto

Con paciencia 2, el entrenamiento se detenía tras dos épocas sin mejora. En la corrida B cortó en la época 9 de 50 pedidas.

Al aumentar la paciencia a 50 en la corrida C, el entrenamiento completó las 100 épocas configuradas (60 + 40) y **la mejor época fue la 91**, con pérdida de validación de 0,00442: una mejora del 26 % respecto a la corrida B. Las diez mejores épocas están todas entre la 72 y la 100.

La explicación está en la curva: durante casi toda la fase 1 la pérdida de validación oscila en una meseta. Con paciencia 2, dos épocas malas seguidas bastaban para cortar. La mejora real llegó después del inicio del ajuste fino, y la paciencia corta nunca dejaba llegar ahí.

### 6.3 Curvas de entrenamiento

*Figura 4 — curvas de la corrida C*

**Lectura:**

- **Pico en la época 61.** Corresponde al inicio del ajuste fino: al descongelar capas, los pesos liberados se reacomodan y la pérdida de entrenamiento sube de forma transitoria. No es un error.
- **La validación no se ve afectada por ese pico** y continúa descendiendo hasta su mínimo en la época 91.
- **Al final aparece una brecha** entre entrenamiento (0,0006) y validación (0,005): el modelo empieza a ajustarse de más a los datos de entrenamiento. La validación, sin embargo, no empeora de forma sostenida.

### 6.4 Matriz de confusión del modelo final

```
                        PREDIJO
                  sin grieta   con grieta
REAL  sin grieta     2.998          2
      con grieta         2      2.998
```

**Cómo se calculan las métricas:**

| Métrica | Fórmula | Valor |
|---|---|---|
| Exactitud | (TP + TN) / total | 5.996 / 6.000 = 99,93 % |
| Precisión | TP / (TP + FP) | 2.998 / 3.000 = 99,93 % |
| Recall | TP / (TP + FN) | 2.998 / 3.000 = 99,93 % |

Las cuatro métricas coinciden porque los errores quedaron simétricos: dos falsos positivos y dos falsos negativos. Es una coincidencia de esta matriz, no una regla.

### 6.5 Por qué el recall es la métrica que manda

Los dos errores no cuestan lo mismo:

| Error | Consecuencia |
|---|---|
| Falso positivo | Una inspección innecesaria. Cuesta dinero |
| **Falso negativo** | **Un elemento dañado declarado seguro. Puede costar vidas** |

La precisión mira la columna de "predijo con grieta" y solo la castiga el falso positivo. El recall mira la fila de "real con grieta" y solo lo castiga el falso negativo.

**El modelo devuelve una probabilidad, no una etiqueta.** El umbral de 0,5 que la convierte en clase lo elige el diseñador y puede moverse según qué error se quiera minimizar.

### 6.6 Resultados de la versión 2 (partición por grupos)

| Conjunto | Imágenes | Exactitud | Recall | Errores |
|---|---|---|---|---|
| Validación (elige la mejor época: algo optimista) | 6.028 | 99,87 % | 99,84 % | 5 grietas no detectadas (0,16 %; IC95 0,07 a 0,38 %) y 3 falsas alarmas |
| **Prueba verificada** | 6.052 | **99,95 %** | **99,93 %** | 2 grietas no detectadas (0,07 %; IC95 0,02 a 0,25 %) y 1 falsa alarma |

La matriz de confusión de la prueba es `[[3123, 1], [2, 2926]]` (filas: real sin grieta y con grieta; columnas: predicho sin grieta y con grieta); precisión 99,97 %.

Con 3 errores no se puede afirmar que la versión 2 sea mejor ni peor que la 1: los intervalos se solapan. Dos explicaciones son compatibles con estos números y este experimento no las separa: Surface Crack es un conjunto fácil, o los grupos aproximados no separan todos los muros parecidos.

**Comparación con la línea base (conjunto de prueba verificado).**

| Modelo | Exactitud | Recall | Precisión | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|
| Línea base: umbral de brillo | 70,27 % | 73,92 % | 68,37 % | 71,04 % | 386 | 506 |
| Línea base: umbral de bordes | 84,87 % | 96,69 % | 77,94 % | 86,31 % | 49 | 405 |
| Línea base: regresión logística | 98,27 % | 98,18 % | 98,31 % | 98,24 % | 27 | 25 |
| **v2 MobileNetV2** | **99,95 %** | **99,93 %** | **99,97 %** | **99,95 %** | **2** | **1** |

Las filas de la línea base se midieron en una muestra de 3.000 imágenes de la prueba; la red, en las 6.052.

**La misma red en dos escenarios** (cuaderno 05, modelo TFLite float16, evaluando con la foto entera achicada a 224×224):

| | Prueba de Surface Crack (muestra de 1.500) | Fotografías propias (29) |
|---|---|---|
| Exactitud | 99,93 % | **62,07 %** |
| Recall | 99,86 % | **15,4 %** (2 de 13) |

La caída de 99,9 % a 62 % entre un escenario y otro es el resultado central del proyecto. Un clasificador que respondiera siempre «sin grieta» tendría 55,2 % de exactitud en las fotos propias (16 de 29): la red apenas lo supera. El recall de las fotos propias coincide con el de la sección 10.4 (foto entera: 2 de 13); con teselas a escala 0,5 sube a 3 de 13.

**Curvas de entrenamiento.** (INSERTAR AQUÍ la imagen `results/curvas.png`: *Figura 7 — Pérdida y recall de entrenamiento y validación de la versión 2, fases 1 y 2.*) El eje vertical se ajusta solo, y por eso variaciones de unas milésimas se ven como saltos. En las seis primeras épocas de la fase 1, la pérdida de validación bajó de 0,0057 a 0,0034 con un pico aislado de 0,0070, y fue siempre menor que la de entrenamiento (de 0,033 a 0,009), lo esperable porque el entrenamiento tiene dropout y aumentación activos y la validación no.

**El brillo no es un atajo.** Se tomaron imágenes de la prueba, se oscurecieron y aclararon hasta ±51 niveles (la diferencia real entre clases es de 18) y se contó cuántas cambiaban de clase: **0 %** en las tres intensidades (18, 30 y 51 niveles) y en las dos direcciones (imágenes sin grieta oscurecidas que pasan a «con grieta», e imágenes con grieta aclaradas que pasan a «sin grieta»). El resultado descarta que la falta de variación de brillo global explique el falso positivo de la sección 10.1; no descarta un efecto del brillo local (una mano oscura sobre una pared clara), que esta prueba no cubre. La prueba se validó con dos modelos sintéticos: detecta al que usa el brillo y no acusa al que mira la textura.

---

## 7. Módulo B — Estimación del desaplome

### 7.1 Por qué visión clásica

La inclinación es geometría: no hay un patrón visual complejo que aprender, sino un ángulo que medir. Entrenar un modelo para eso sería un uso innecesario de recursos.

### 7.2 Algoritmo

1. Conversión a escala de grises.
2. Suavizado gaussiano para reducir el ruido del sensor.
3. **Detección de bordes con Canny.**
4. **Transformada de Hough probabilística**, que convierte los bordes en segmentos de recta.
5. Filtrado de segmentos cercanos a la vertical.
6. Cálculo del ángulo de cada segmento respecto a la vertical.
7. **Estimación con la mediana**, no con el promedio: un segmento espurio —un cable, un objeto en primer plano— desplaza el promedio de forma desproporcionada, pero no la mediana.

### 7.3 Resultado sobre fotografía

*Figura 5 — `[COMPLETAR: captura de la aplicación con las líneas detectadas]`*

Sobre la fotografía de una columna, el módulo detectó 11 líneas y estimó un desaplome de **0,0°**: elemento a plomo.

`[COMPLETAR: confirmar que la fotografía es propia, como exige la entrega]`

### 7.4 Limitación

La transformada de Hough mide la inclinación **en el plano de la imagen**, no en el mundo real. Si la fotografía no se toma de frente, la perspectiva introduce una inclinación aparente que no existe. El protocolo de captura exige fotografía frontal, con el celular nivelado.

---

## 8. Motor de riesgo

La aplicación combina las dos salidas en una recomendación de riesgo **bajo, medio o alto**, considerando la presencia de grieta, su orientación, el tipo de elemento fotografiado y el desaplome.

**Reglas de la versión 2** (las que implementa `src/riesgo.py`; el informe las toma del código):

| Factor | Regla | Puntos |
|---|---|---|
| Grieta detectada | probabilidad ≥ umbral | +1 |
| Ancho desconocido (sin objeto de referencia) | no se puede medir en mm | +1 (prudencia) |
| Ancho | < 0.3 mm / 0.3–1.0 / 1.0–3.0 / ≥ 3.0 mm | 0 / +1 / +2 / +3 |
| Orientación | diagonal +1 · vertical +0 · horizontal +0 | según orientación |
| Elemento afectado | columna +1 · viga +1 · muro +0 · placa +0 | según elemento |
| Inclinación | < 0.5° / 0.5–1.5 / 1.5–3.0 / ≥ 3.0° | 0 / +1 / +2 / +3 |

**Nivel:** ALTO si el total ≥ 5 **o** el ancho ≥ 3.0 mm **o** la inclinación ≥ 3.0°; MEDIO si el total ≥ 2; BAJO en otro caso.
**No se afirma (INDETERMINADO)** si la imagen no parece concreto o la probabilidad está entre 0.2 y 0.8.

> Reglas orientativas; no sustituyen la inspección de un ingeniero. Sesgadas a sobreestimar el riesgo.

**Casos calculados con el código:**

| Caso | Nivel | Puntaje |
|---|---|---|
| Grieta diagonal en columna, 1,5 mm, desaplome 0,2° | **ALTO** | 5 |
| Grieta vertical en muro, 0,2 mm, desaplome 0,2° | **BAJO** | 1 |
| Grieta detectada, sin objeto de referencia (ancho desconocido) | **MEDIO** | 2 |
| Sin grieta, pero desaplome de 4° | **ALTO** | 3 |
| Sin grieta y sin desaplome | **BAJO** | 0 |
| La imagen no parece concreto | **INDETERMINADO** | — |
| Probabilidad 0,50 (zona de duda) | **INDETERMINADO** | — |

El esquema está deliberadamente sesgado hacia sobreestimar el riesgo, por el costo asimétrico de los errores. Toda salida incluye el aviso de que la herramienta es orientativa y no sustituye la inspección de un ingeniero.

Los umbrales son orientativos y no están validados contra dictámenes de ingenieros (limitación 8).

---

## 9. Complejidad computacional

### 9.1 Mediciones

*Las mediciones de esta sección corresponden a la arquitectura de la versión 1; MobileNetV2 con la misma cabeza tiene los mismos parámetros en la versión 2. Los tiempos de la versión 2, las variantes TFLite (float32, float16 y dinámico) y su fidelidad respecto de Keras se generan en el cuaderno 05 (`results/complejidad.csv`). Los tiempos se miden con una imagen por llamada, descartando el calentamiento; una CPU de Colab no es un celular.*

**Versión 2:** `models/v2_mobilenet/modelo.tflite` (pesos en float16) pesa **4,46 MB**, igual que el de la versión 1, como corresponde a la misma arquitectura y al mismo formato de exportación.

**Complejidad de la versión 2** (cuaderno 05; una imagen por llamada, 100 repeticiones tras el calentamiento, CPU de Colab):

| Variante TFLite | Tamaño | Mediana por imagen | p95 | Cambian de clase respecto de Keras (300 imágenes) | Diferencia máxima de probabilidad |
|---|---|---|---|---|---|
| float32 | 8,86 MB | 8,74 ms | 12,37 ms | 0 | 0,0000 |
| float16 (la exportada) | 4,46 MB | 9,33 ms | 11,52 ms | 0 | 0,0060 |
| dinámica (pesos en 8 bits) | **2,50 MB** | 14,59 ms | 18,94 ms | 0 | 0,0228 |

El modelo tiene 2.259.265 parámetros: 1.512.001 entrenables tras el ajuste fino y 747.264 congelados. En formato Keras pesa 21,75 MB, y una imagen en modo eager sobre CPU tarda entre 179 y 195 ms; esa cifra no es representativa del despliegue, que usa TFLite.

**Observaciones.** (1) float16 reduce el tamaño a la mitad sin cambiar la velocidad: en CPU los pesos se convierten a float32 al ejecutar. (2) La cuantización dinámica es la más pequeña (2,50 MB, cuatro veces menos que float32) pero la más lenta en esta CPU. (3) Ninguna de las tres variantes cambió la clase de una sola de las 300 imágenes de prueba, pero esas imágenes son solo de Surface Crack: la fidelidad no se verificó con fotografías de celular. (4) Los tiempos son de una CPU de Colab; el tiempo en un celular real debe medirse en el dispositivo.

| Métrica | Valor |
|---|---|
| Parámetros totales | 2.259.265 |
| Parámetros entrenables | 1.527.681 |
| Tamaño en Keras | 21,89 MB |
| Tamaño en TensorFlow Lite | **4,46 MB** |
| Inferencia por imagen, en lote | 1,57 ms |
| Inferencia de una imagen, TFLite, CPU | 6,73 ms |

### 9.2 Observaciones

**La conversión a TensorFlow Lite redujo el tamaño 4,9 veces** con las métricas prácticamente intactas.

**Procesar en lote es 4 veces más rápido por imagen.** El costo fijo de iniciar la inferencia se reparte entre las imágenes del lote. En la aplicación el usuario sube una foto a la vez y vive el caso lento; 6,73 ms sigue siendo imperceptible.

**Las tres corridas tienen los mismos 2.259.265 parámetros.** La mejora de siete veces en falsos negativos no costó un byte adicional.

### 9.3 Escalamiento

El costo de una capa convolucional crece con el cuadrado de la resolución. Duplicar el lado de la imagen cuadruplica las operaciones. Por eso se trabaja a 224×224 aunque una resolución mayor ayudaría a detectar grietas finas: es el compromiso central de diseño del proyecto.

### 9.4 Despliegue

El despliegue implementado es una **interfaz web con Streamlit**, que corre en el computador y es accesible desde cualquier dispositivo de la red local, incluido el celular. La inferencia ocurre en el computador.

Por separado, el modelo está exportado a TensorFlow Lite, que es el formato para ejecución dentro del dispositivo sin conexión. La aplicación Android que lo ejecuta está armada y pendiente de compilación.

---

## 10. Validación fuera del dataset y limitaciones

### 10.1 El falso positivo encontrado en campo

Al probar la aplicación con una fotografía de una **pared lisa y pintada**, sin grietas, con una mano limpiándola, el modelo respondió **"con grieta" con probabilidad 1,00**.

*Figura 6 — `[COMPLETAR: captura del resultado en la aplicación]`*

**Lo que este caso muestra:**

- **El 99,93 % no se sostiene fuera del dominio.** El modelo se entrenó con concreto a la vista y nunca vio una pared pintada.
- **El modelo no sabe cuándo está fuera de su dominio.** Falla con máxima confianza, así que el umbral de confianza no lo habría filtrado.
- **Causa probable.** El modelo se entrenó con aumentación de brillo de ±51 niveles, casi tres veces la diferencia entre clases, así que es poco probable que el brillo global sea la causa principal. Lo más probable es la combinación de una textura que nunca vio con los bordes de la mano y los pliegues del paño, que son líneas de alto contraste, como las grietas. Queda pendiente verificarlo con la prueba del atajo.

### 10.2 Por qué las curvas no lo detectan

La validación también sale de las mismas 458 fotografías: comparte superficie, iluminación y cámara con el entrenamiento. Las curvas miden si el modelo funciona con **más imágenes del mismo tipo**, no con superficies distintas.

| Problema | Dónde se observa |
|---|---|
| Sobreajuste dentro del dataset | Brecha entre las curvas de entrenamiento y validación |
| Cambio de dominio | Solo al probar con imágenes externas |

**El modelo generaliza bien dentro de su dominio** —99,93 % en imágenes de concreto que nunca vio— **pero no generaliza a superficies distintas.**

### 10.3 Limitaciones

1. **Fuga de información por recortes.** 40.000 recortes de 458 fotografías, cerca de 87 por foto. La partición aleatoria reparte recortes del mismo muro entre entrenamiento y prueba. La partición correcta sería por fotografía de origen, pero el dataset no publica esa correspondencia. **Actualización v2:** corregida con la partición por grupos (sección 4.4.1); los grupos siguen siendo una aproximación de las fotografías de origen.
2. **Cambio de dominio.** Entrenado con concreto a la vista de un campus turco; se aplica a mampostería pañetada y pintada de construcción informal colombiana.
3. **Sin fisuras finas.** Todas las grietas del dataset son anchas y de alto contraste. El modelo será menos sensible al daño incipiente.
4. **Solo daño visible.** No detecta corrosión interna, degradación del núcleo del concreto ni fallas de cimentación sin manifestación superficial.
5. **Sin estimación del ancho.** Sin referencia de escala ni calibración de cámara no se pueden convertir píxeles a milímetros. **Actualización v2:** con un objeto de tamaño conocido en el mismo plano que la grieta, el ancho se mide con esqueleto y transformada de distancia (sesgo de −0,02 px con anchos exactos, peor caso 1 px); sin referencia solo se da en píxeles.
6. **La inclinación depende de la perspectiva.** Sin rectificación es una estimación, no una medida metrológica. **Actualización v2:** el giro del celular se descuenta con el sensor de gravedad; si el celular apunta hacia arriba o abajo (más de 5°) la medición se marca como no confiable.
7. **Una fotografía no caracteriza una edificación.** El riesgo real depende de la configuración global, la cimentación y el suelo.
8. **Umbrales de riesgo no validados** contra dictámenes de ingenieros estructurales.
9. **Confusión de fuente en `dificiles`.** Las fotos sanas son de celular y las agrietadas son de internet: el origen separa las clases y el modelo pudo aprenderlo en vez de la grieta (sección 10.4).
10. **Fotografías de internet.** Deben citarse con su dirección y licencia, y ninguna se usó como prueba.

### 10.4 Resultados con fotografías propias (versión 2)

Se tomaron 29 fotografías con celular (960×1280 o 1280×960 px; según el equipo, las grietas miden unos 13 px de ancho): **13 con grieta y 16 sin grieta**. Ninguna se usó para entrenar ni para ajustar el sistema.

| Método (umbral 0,5) | Grietas detectadas | Falsas alarmas | AUC (exploratorio) |
|---|---|---|---|
| Foto entera achicada a 224×224 | 2 / 13 | 0 / 16 | 0,75 |
| Teselas a escala 0,5, sin puerta | 3 / 13 | 0 / 16 | 0,89 |
| Teselas a escala 0,25, sin puerta | 3 / 13 | 0 / 16 | 0,86 |
| Sistema con puerta de concreto | **0 / 13** | 0 / 16 | 0,71 |

**Lectura.** (1) Con 0 de 13, el recall real podría ser como máximo del 23 % (IC95): no es un efecto del tamaño de la muestra. (2) La escala no es la causa: tres escalas dieron entre 2 y 3 de 13. (3) **La puerta de concreto empeora el resultado:** las tres grietas que la red detecta con probabilidad mayor a 0,99 quedaron en 0,00 con la puerta, porque se calibró solo con Surface Crack y trata las fotos de celular como "algo raro"; se retiró del sistema final. (4) La red distingue algo: en las 16 fotos sanas la probabilidad máxima es 0,005, y en 9 de las 13 grietas es más alta que en todas las sanas (AUC 0,89), pero con probabilidades muy por debajo de 0,5. (5) Bajar el umbral a 0,01 da 9 de 13 sin falsas alarmas, pero elegirlo con estas mismas fotos las deja de ser prueba: se reporta como dato exploratorio, no como resultado. (6) 4 de las 13 grietas tienen probabilidad casi nula con todos los métodos.

**Memorización frente a transferencia.** Usando la partición con la que el modelo realmente se entrenó:

| Grupo | Fotos | Dice "con grieta" | Ideal |
|---|---|---|---|
| Vistas, sin grieta (`dificiles`) | 220 | 0 | 0 % |
| Vistas, con grieta (`dificiles`) | 215 | **215** | 100 % |
| Nuevas, sin grieta (`fotos_propias`) | 16 | 0 | 0 % |
| **Nuevas, con grieta (`fotos_propias`)** | 13 | **2** (15 %; IC95 4 a 42 %) | 100 % |

Las filas "vistas" son 44 y 43 fotografías únicas repetidas 5 veces por el sobremuestreo, por lo que sus intervalos no son válidos. El modelo acierta todo lo que vio y casi nada de lo nuevo: la lectura más fuerte es **memorización de pocas fotos**, sumada a una **diferencia de dominio** entre el concreto de Surface Crack y las paredes fotografiadas con celular. Las 44 fotografías sin grieta de esa carpeta son fotos de celular del equipo (3 escenas); **las 43 con grieta provienen de internet** (`[COMPLETAR: dirección y licencia de cada una]`).

**Confusión de fuente (hipótesis principal).** Dentro de `dificiles`, el origen separa las clases: lo sano es de celular y lo agrietado es de internet. El modelo pudo aprender «foto de celular = sin grieta» y «foto de internet = con grieta» en lugar de aprender la grieta. Es coherente con lo observado: acierta las 215 fotos con grieta de internet que vio, no da ninguna falsa alarma en las 16 fotos sanas de celular nuevas y detecta solo 2 de las 13 grietas de celular nuevas; en fotos de celular responde casi siempre «sin grieta». **No está demostrado.** Se comprobaría de dos maneras: fotografiando paredes sanas de internet (si el modelo las marca como grieta, la hipótesis se confirma) o reentrenando con grietas de celular. Es un error de diseño del equipo, que ya había identificado el riesgo de mezclar fuentes por clase.

**El falso positivo de la sección 10.1.** La pared con la mano, que no está en ningún conjunto de entrenamiento, da ahora p = 0,02. **No se cuenta como corrección**: el modelo también dice "sin grieta" en la mayoría de las grietas de celular, y un "no" generalizado no demuestra que distinga una pared lisa de una grieta. La carpeta `dificiles` tiene solo 3 escenas, todas en entrenamiento, así que el falso positivo no pudo medirse en escenas no vistas.

**Valoración.** El sistema final no usa la puerta de concreto y analiza las fotos a 1.280 px de lado largo y escala 0,5. La aplicación ofrece tres niveles de sensibilidad (umbral 0,50, 0,10 y 0,02); los dos últimos se eligieron mirando las mismas fotos de prueba y son exploratorios. Un aviso permanente recuerda que no detectar una grieta no significa que no exista.

---

## 11. Consideraciones éticas

**Falsos negativos y responsabilidad.** Un usuario que recibe "riesgo bajo" para una edificación dañada puede omitir una inspección necesaria. Por eso el motor se sesga hacia la sobreestimación y el aviso de limitación aparece en toda salida.

**Falsos positivos y daño socioeconómico.** Declarar "riesgo alto" sobre una vivienda genera angustia, costos y posible afectación del valor del inmueble. La herramienta no debe usarse para decisiones administrativas, catastrales o de aseguramiento.

**Equidad.** El modelo funciona mejor sobre construcción formal, donde el riesgo es menor, y peor sobre construcción informal, donde el riesgo se concentra. **Es menos confiable justamente donde más se necesita.** El falso positivo de la sección 10 lo confirma.

**Privacidad.** Las fotografías publicadas no incluyen metadatos de geolocalización ni permiten identificar direcciones.

**Abstención y grietas sin diagnóstico.** Cuando la imagen no parece concreto o el modelo duda (probabilidad entre 0,20 y 0,80 con el umbral estándar), el sistema responde INDETERMINADO en lugar de afirmar. Reduce las falsas alarmas pero introduce un riesgo propio: una grieta real sin diagnóstico. Por eso se reporta el número de grietas reales sin diagnóstico en las fotos propias, y la recomendación asociada pide repetir la foto o una revisión visual, nunca "sin riesgo".

**Transparencia sobre el desempeño en campo.** El modelo alcanza 99,95 % en Surface Crack pero detecta 2 a 3 de 13 grietas en fotos de celular. Mostrar solo la primera cifra sería engañoso para un usuario que confía en la herramienta. Por eso la aplicación muestra, sobre cada resultado, el desempeño real medido por el equipo con fotos propias y la advertencia de que puede dejar pasar grietas.

**Datos personales en las fotografías.** Las fotos pueden mostrar rostros, placas, números de casa o la ubicación en los metadatos. Antes de publicarlas se eliminan los metadatos EXIF y se evitan personas y direcciones identificables. Las fotografías tomadas de internet se usan solo con una licencia que lo permita y citando su origen.

---

## 12. Conclusiones

1. **Se construyó un prototipo funcional de extremo a extremo** que clasifica grietas, estima el desaplome y emite una recomendación de riesgo, desplegado en una aplicación web.

2. **La transferencia de aprendizaje sobre MobileNetV2 permitió alcanzar 99,93 %** de exactitud y recall en prueba (versión 1, con partición con fuga; en la versión 2, sin fuga, 99,95 % en una prueba verificada), con 2.259.265 parámetros, de los cuales el 99,94 % proviene de ImageNet.

3. **El estudio de ablación atribuyó cada mejora.** Ampliar los datos y corregir la paciencia de la parada temprana contribuyeron en proporciones parecidas, reduciendo siete veces la tasa de falsos negativos sin aumentar el tamaño del modelo.

4. **La parada temprana con paciencia corta es un riesgo real.** Cortaba durante una meseta temporal, antes de que el ajuste fino produjera la mejora. Con paciencia 50 la mejor época pasó de la 9 a la 91.

5. **El desempeño medido no se sostiene fuera del dominio del dataset.** Un falso positivo con probabilidad 1,00 sobre una pared pintada confirma lo que el análisis exploratorio anticipaba: el modelo generaliza dentro de su dominio y falla fuera de él.

6. **El modelo es viable en móvil:** 4,46 MB y 6,73 ms por imagen, compatible con celulares de gama baja sin conexión. En la versión 2 se midió de nuevo: 4,46 MB y 9,33 ms por imagen en una CPU de Colab (la variante dinámica pesa 2,50 MB), sin cambios de clase en 300 imágenes de prueba.

7. **Se encontró y corrigió una fuga de información.** Con la división de la versión 1 los 913 grupos aparecían en entrenamiento y prueba. La versión 2 divide por grupos y verifica que la prueba no comparte imágenes ni grupos con el entrenamiento.

8. **En Surface Crack la exactitud se mantiene:** 99,95 % en una prueba verificada de 6.052 imágenes, con 2 grietas no detectadas de 2.928. Con 3 errores no es posible afirmar una mejora o un empeoramiento respecto de la versión 1.

9. **El modelo no transfiere a fotografías de celular.** Detecta 2 a 3 de 13 grietas propias sin falsas alarmas, aunque acierta el 100 % de las 87 fotos propias con las que entrenó. La evidencia indica memorización de pocas fotos, diferencia de dominio y una posible confusión de fuente (grietas de internet contra paredes sanas de celular), no un problema de brillo ni de escala.

10. **Un componente que parecía una mejora la empeoró.** La puerta de concreto, calibrada solo con Surface Crack, descartó las tres grietas que la red sí detectaba y se retiró. Medir con fotos propias fue lo que permitió verlo.

11. **El mayor avance disponible es de datos, no de modelo:** fotografías de celular de muchas paredes distintas, con y sin grieta, de la misma fuente para las dos clases.

---

## 13. Trabajo futuro

| Prioridad | Tarea |
|---|---|
| Alta | Captura sistemática de fotografías propias y validación con métricas fuera del dataset |
| Alta | Reentrenar incluyendo superficies lisas, pintadas y pañetadas, con objetos delante, etiquetadas como sin grieta |
| Alta | Prueba del atajo: oscurecer las imágenes sin grieta del conjunto de prueba y medir cuántas cambian de clase, para verificar si la aumentación de brillo funcionó |
| Media | Corrida sin aumentación, cambiando solo ese factor, para medir su aporte |
| Media | Barrido de tasa de aprendizaje en la fase de ajuste fino |
| Media | Ajustar la paciencia de la parada temprana a un valor intermedio (10 a 15) |
| Media | Compilar la aplicación Android con el modelo TensorFlow Lite |
| Baja | Comparar contra MobileNetV3 y EfficientNet-Lite |
| Baja | Tercera clase "no aplica" para superficies fuera del dominio |
| Baja | Segmentación para estimar el ancho de la grieta (U-Net) |
| Alta | Ajustar el modelo con fotografías de celular de al menos 10 paredes con grieta y 10 sin grieta (cuaderno 06), sin tocar las fotos de prueba |
| Alta | Entrenar con fotos con grieta y sin grieta **del mismo origen** (celular) y comprobar la confusión de fuente fotografiando paredes sanas de internet |
| Alta | Calibrar el umbral de decisión sobre un conjunto de calibración distinto del de prueba |
| Media | Calibrar la puerta de concreto con fotos de celular, o descartarla |
| Media | Incorporar SDNET2018 y reservar parte como prueba fuera de dominio |

*Ya realizado en la versión 2:* prueba del atajo del brillo (el modelo no lo usa), estimación del ancho por esqueleto y transformada de distancia, rectificación de perspectiva, orientación de la grieta, y partición por grupos.

---

## 14. Referencias

1. Ç. F. Özgenel y A. Gönenç Sorguç, "Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings", *Proc. ISARC*, 2018. Dataset: https://data.mendeley.com/datasets/5y9wdsg2zt/2
2. Y.-J. Cha, W. Choi y O. Büyüköztürk, "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks", *Computer-Aided Civil and Infrastructure Engineering*, vol. 32, n.º 5, pp. 361-378, 2017.
3. S. Dorafshan, R. J. Thomas y M. Maguire, "SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks", *Data in Brief*, vol. 21, 2018.
4. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", arXiv:1704.04861, 2017.
5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks", *Proc. IEEE/CVF CVPR*, 2018.
6. A. Howard et al., "Searching for MobileNetV3", *Proc. IEEE/CVF ICCV*, 2019.
7. J. Canny, "A Computational Approach to Edge Detection", *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. PAMI-8, n.º 6, 1986.
8. Asociación Colombiana de Ingeniería Sísmica, *Reglamento Colombiano de Construcción Sismo Resistente NSR-10*, 2010.
9. OpenCV, "Hough Line Transform". https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
10. TensorFlow, "LiteRT". https://ai.google.dev/edge/litert

---

## Anexo A — Plan de trabajo y roles del equipo

| Integrante | Rol principal | Responsabilidades |
|---|---|---|
| Juan David Sayas Hernandez | `[COMPLETAR]` | `[COMPLETAR]` |
| Erick Fabian Cardenas Bello | `[COMPLETAR]` | `[COMPLETAR]` |

**Cronograma** (ver `docs/SPRINTS.md`):

| Entrega | Fecha | Contenido |
|---|---|---|
| 1 | 25/09/2026 | Formulación, EDA, línea base, primera inclinación |
| 2 | 23/10/2026 | Aumentación, fotos propias, reentrenamiento, métricas fuera del dominio |
| 3 | 20/11/2026 | Despliegue móvil, validación final, sustentación |

## Anexo B — Reproducibilidad

- Semilla fija `random_state=42` en la partición.
- Entrenamiento en Google Colab con GPU T4, TensorFlow 2.20.
- Checkpoints y registro por época (`ModelCheckpoint`, `CSVLogger`) guardados en Google Drive.
- **Nota sobre el historial de la corrida C.** El archivo `mobilenet_100ep_historial.csv` tiene 107 filas: las 7 primeras corresponden a un intento interrumpido, que quedó registrado porque el `CSVLogger` está configurado con `append=True`. La corrida completa ocupa las filas 8 a 107. Toda la numeración de épocas de este informe se refiere a la corrida completa. Para evitarlo en el futuro, borrar el archivo de historial antes de relanzar una corrida.
- Scripts: `src/dataset.py`, `src/train_transfer.py`, `eda_grietas.py`, `animacion_normal.py`, `animacion_separable.py`, `comparacion_costo.py`, `comparacion_visual.py`, `figura_congelado.py`.

- **Versión 2.** Partición por grupos con semilla 42, guardada en `results/particion_*.csv`; entrenamiento y evaluación leen las mismas particiones. El historial se guarda con un archivo por fase y `append=False`, lo que corrige el defecto de la corrida C. `models/<nombre>/meta.json` registra hiperparámetros y versiones de TensorFlow y Keras. `python tests/run_all.py` ejecuta las pruebas del repositorio y `python -m src.verificar` marca cada requisito de las entregas según la evidencia existente.
- **Código de la versión 2:** `src/datos.py`, `src/eda.py`, `src/linea_base.py`, `src/modelo.py`, `src/entrenar.py`, `src/evaluar.py`, `src/mosaico.py`, `src/ancho.py`, `src/inclinacion.py`, `src/riesgo.py`, `src/predecir.py`, `src/complejidad.py`, `src/fotos_propias.py`, `src/app_core.py` y `app/streamlit_app.py`.

## Anexo C — Modelos disponibles

| Archivo | Corrida | Uso |
|---|---|---|
| `models/mobilenet.tflite` | A — 8.000 imágenes | Referencia |
| `models/mobilenet_100ep.tflite` | C — 40.000 imágenes, paciencia 50 | Modelo de la versión 1 (partición con fuga) |
| `models/v2_mobilenet/modelo.tflite` | v2 — partición por grupos, float16 | **Modelo de la versión 2** (4,46 MB) |
| `models/v2_mobilenet/modelo_dinamico.tflite` | v2 — cuantización dinámica | Variante más pequeña (2,50 MB) |
