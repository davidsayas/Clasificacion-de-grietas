# Hallazgos de la v2, redactados para el informe

Todo lo que sigue sale de resultados que el equipo ya obtuvo. Cada cifra dice de dónde viene. **Lo que depende de
una corrida que todavía no hiciste está marcado con ▶** y se toma de `results/` (no se copia de acá).

## Resumen en cinco frases (para el Resumen y las Conclusiones)

> 1. La versión 1 reportaba 99,93 % de exactitud, pero la partición imagen por imagen mezclaba recortes de los
>    mismos muros entre entrenamiento y prueba (913 de 913 grupos aparecían en ambos).
> 2. La versión 2 divide por grupos y verifica por imagen y por grupo que la prueba no comparte nada con lo que el
>    modelo vio; sobre esa prueba limpia (6.052 imágenes de Surface Crack) alcanza 99,95 % de exactitud, 2 grietas no
>    detectadas de 2.928 y 1 falsa alarma de 3.124.
> 3. En 29 fotografías propias de celular (13 con grieta, 16 sin grieta), la red sola detecta entre 2 y 3 de las 13
>    grietas sin ninguna falsa alarma, y el sistema con la puerta de concreto detecta 0 de 13.
> 4. El modelo acierta el 100 % de las 87 fotos propias con las que se entrenó y detecta 2 de las 13 nuevas: señal de
>    memorización de pocas fotos y de diferencia de dominio entre concreto de Surface Crack y paredes de celular.
> 5. La red no es ciega a esas grietas: su AUC exploratorio es 0,89, pero con probabilidades muy por debajo de 0,5.

## 1. La fuga del 99,93 %

| Dato | Valor | Fuente |
|---|---|---|
| Grupos que aparecían en entrenamiento **y** prueba con la división imagen por imagen | 913 de 913 | cuaderno 01, paso 6 |
| Grupos compartidos con la división por grupos | 0 (verificado con `verificar_sin_fuga`) | cuaderno 01, paso 6 |
| Hipótesis "los archivos consecutivos son de la misma foto" | razón 0,50 (consecutivos 0,69 / azar 1,36) | cuaderno 01, paso 5 |
| Imágenes casi duplicadas | 1,52 % de una muestra de 4.000 (cota inferior) | cuaderno 01, paso 4 |

**Límite a declarar:** los grupos son bloques de 44 archivos consecutivos; el dataset no identifica las 458 fotos de
origen. Además, los bloques se arman por clase, así que recortes de una misma foto con grieta y sin grieta podrían caer
en particiones distintas. El resultado sigue siendo optimista respecto de muros nunca vistos.

## 2. Control de integridad de la prueba

El cuaderno 03 (Paso 0) compara la prueba con las particiones **con las que el modelo realmente se entrenó**
(`models/v2_mobilenet/particion_*.csv`): **0 imágenes y 0 grupos en común**. La prueba tiene 6.052 imágenes, todas de
Surface Crack (3.124 sin grieta, 2.928 con grieta). Esto importa porque `results/particion_prueba.csv` cambia si se
repite el cuaderno 01 después de entrenar.

## 3. Resultados en Surface Crack

| Conjunto | n | Exactitud | Recall | Errores |
|---|---|---|---|---|
| Validación (elige la mejor época: algo optimista) | 6.028 | 99,87 % | 99,84 % | 5 grietas no vistas (0,16 %, IC95 0,07–0,38 %) y 3 falsas alarmas |
| **Prueba verificada** | 6.052 | **99,95 %** | **99,93 %** | 2 grietas no vistas (0,07 %, IC95 0,02–0,25 %) y 1 falsa alarma |

Con 3 errores no se puede afirmar que la v2 sea mejor ni peor que la v1: los intervalos se solapan. La partición sin fuga
**no degradó** el resultado en Surface Crack. Eso es coherente con dos explicaciones que este experimento no separa:
Surface Crack es un conjunto fácil, o los grupos aproximados no separan todos los muros parecidos.

## 4. El brillo no es un atajo

▶ La tabla `results/prueba_atajo_brillo.csv` (cuaderno 03, Paso 2) cuenta cuántas imágenes cambian de clase al oscurecer
o aclarar hasta ±51 niveles. El cuaderno concluyó que **el modelo no usa el brillo como atajo**. El porcentaje máximo es **0 %** (a 18, 30 y 51 niveles, en las dos direcciones).

**Qué descarta y qué no:** descarta que el falso positivo de la pared lisa se deba a falta de variación de brillo
global. No descarta un efecto del brillo **local** (una mano oscura sobre una pared clara), que esta prueba no cubre.

## 5. Fotos propias: lo que se midió

Las 29 fotos son de celular (960×1280 o 1280×960 px); según el equipo, las grietas miden unos 13 px de ancho.
Ninguna foto propia de prueba se usó para entrenar ni para ajustar el sistema.

| Método (umbral 0,5) | Grietas detectadas | Falsos positivos | AUC (exploratorio) |
|---|---|---|---|
| Foto entera achicada a 224×224 | 2 / 13 | 0 / 16 | 0,75 |
| Teselas a escala 0,5, sin puerta | 3 / 13 | 0 / 16 | 0,89 |
| Teselas a escala 0,25, sin puerta | 3 / 13 | 0 / 16 | 0,86 |
| Sistema con puerta de concreto | **0 / 13** | 0 / 16 | 0,71 |

Fuente: cuaderno 03, Paso 6d (`results/diagnostico_fotos_propias.csv`).

▶ **Sistema final (sin puerta):** el Paso 6b se repite con el sistema sin puerta y escribe
`results/metricas_fotos_propias.json`. **Esa es la fila que se reporta como resultado del sistema**; las demás son
diagnóstico.

**Cómo leerlo:**
* Con 0 de 13, el recall real podría ser como máximo del 23 % (IC95). No es un efecto del tamaño de la muestra.
* **La escala no es la causa:** tres escalas dieron entre 2 y 3 de 13.
* **La puerta de concreto empeora el resultado.** Las tres grietas que la red detecta con probabilidad ≥ 0,99
  (fotos 2, 9 y 11) quedaron en 0,00 con la puerta. Se calibró solo con Surface Crack y trata las fotos de celular como
  "algo raro". **Se retiró del sistema final.**
* **La red distingue algo:** en las 16 fotos sanas la probabilidad máxima es 0,005; en 9 de las 13 grietas es más alta que
  en todas las sanas. Pero las probabilidades están muy por debajo de 0,5, y el umbral habitual casi nunca se supera.
* **Reajustar el umbral con estas mismas fotos no es válido**: bajar el umbral a 0,01 da 9 de 13 sin falsos positivos, pero
  elegirlo con las fotos de prueba la deja de ser prueba. Se reporta como dato exploratorio.
* 4 de las 13 grietas (fotos 5, 6, 12 y 13) tienen probabilidad casi nula en todos los métodos.

## 6. Memorización frente a transferencia

Cuaderno 03, Paso 6c, usando la partición con la que el modelo realmente entrenó:

| Grupo | Fotos | Dice "con grieta" | Ideal |
|---|---|---|---|
| Vistas, sin grieta (`dificiles`) | 220 | 0 | 0 % |
| Vistas, con grieta (`dificiles`) | 215 | **215** | 100 % |
| Nuevas, sin grieta (`fotos_propias`) | 16 | 0 | 0 % |
| **Nuevas, con grieta (`fotos_propias`)** | 13 | **2** (15 %, IC95 4–42 %) | 100 % |

Las filas "vistas" son 44 y 43 fotos únicas repetidas 5 veces por el sobremuestreo; sus intervalos no son válidos.
El modelo acierta todo lo que vio y casi nada de lo nuevo. La lectura más fuerte es **memorización de pocas fotos**; a eso se
suma una **diferencia de dominio**.

**Origen:** las 44 fotos sin grieta de `dificiles` son de celular del equipo; las **43 con grieta son de internet**. Eso separa las clases por
origen: el modelo pudo aprender «celular = sin grieta» y «internet = con grieta». Es la hipótesis principal y no está demostrada.

La pared con la mano (`pared_con_mano.jpg`, no usada en ningún entrenamiento) dio p = 0,02. **No se cuenta como
corrección del falso positivo**, porque el modelo también dice "sin grieta" en la mayoría de las grietas de celular: un "no"
generalizado no demuestra que distinga una pared lisa de una grieta.

## 7. Limitaciones (para la sección 10.3)

1. La partición por grupos es una aproximación de las 458 fotos de origen (ver sección 1).
2. 13 grietas y 16 fotos sanas son muestras muy chicas: un error mueve el recall unos 8 puntos.
3. El dominio de entrenamiento (concreto de Surface Crack) difiere del dominio de uso (paredes de celular).
4. `dificiles` tiene solo 3 escenas, todas en entrenamiento: el falso positivo no se pudo medir en escenas no vistas.
5. La puerta de concreto, calibrada solo con Surface Crack, no es utilizable con fotos de celular.
6. No se evaluó el efecto del acabado de la pared (pañete, pintura, concreto a la vista) sobre la detección.
7. No se usó ningún conjunto externo (SDNET2018 ni otro).
8. Las mediciones de ancho, inclinación y riesgo se validaron con datos sintéticos, no con mediciones de campo.

## 8. Trabajo futuro (con prioridad)

1. **Más fotos de celular, de paredes distintas, con y sin grieta, de las mismas condiciones de uso.** Es la acción de
   mayor impacto, según el análisis de memorización (sección 6).
2. Entrenar con ellas, sin tocar `fotos_propias`, y volver a medir.
3. Calibrar la puerta con fotos de celular, o descartarla.
4. Calibrar el umbral de decisión sobre un conjunto **de calibración distinto** del de prueba.
5. Incorporar un conjunto externo (SDNET2018) y reservar parte como prueba fuera de dominio.

## 9. Conclusión sugerida

> *El análisis encontró y corrigió una fuga de información que inflaba la exactitud, y mostró con una prueba verificada
> que la exactitud en Surface Crack se mantiene en 99,95 %. Pero el modelo no transfiere a fotos propias de celular:
> detecta entre 2 y 3 de 13 grietas, y un componente diseñado para reducir falsas alarmas (la puerta de concreto)
> empeoró el resultado. La evidencia indica memorización de las pocas fotos propias de entrenamiento y diferencia de
> dominio. El mayor avance disponible no es cambiar el modelo, sino ampliar los datos de entrenamiento con fotos de
> celular de paredes distintas.*
