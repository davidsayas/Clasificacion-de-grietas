# Cómo actualizar `INFORME_TECNICO.md` a la v2

Tu informe ya tiene casi todo lo que pide el documento del reto: estado del arte, EDA, línea base,
arquitectura, complejidad medida, validación fuera del dataset, ética y referencias. **No se rehace.**
Lo que cambia es que ahora hay una división sin fuga, un módulo de ancho, orientación, riesgo con más
factores, fotos propias y una puerta de concreto. Esto es sección por sección.

> Regla de oro: **ningún número del informe se tipea a mano.** Los resultados salen de
> `results/secciones_informe_v2.md`, que `python -m src.informe` escribe desde los CSV de los cuadernos.

## Qué reemplazar y con qué

| Sección actual | Qué hacer | De dónde sale |
|---|---|---|
| **Resumen** | No dejar el 99,93 % como titular. Decir "99,93 % en la v1 (con fuga) → X % en la v2 (sin fuga)" | cifra X de `tabla_comparativa.csv` |
| **4.4 Partición** | Reemplazar por el texto de abajo | `prueba_adyacencia` (cuaderno 01, paso 5) |
| **6 Resultados** | Conservar la v1 pero rotularla **"v1 — división con fuga, no comparable"**; agregar la v2 | `secciones_informe_v2.md` |
| **8 Riesgo** | Reemplazar el `[COMPLETAR: reglas exactas]` por la tabla de reglas | `secciones_informe_v2.md` (sección 8) |
| **9 Complejidad** | Reemplazar la tabla 9.1 por la nueva; agregar variantes TFLite y fidelidad | `complejidad.csv`, `fidelidad_tflite.json` |
| **10.1 Falso positivo** | Pasar de "problema abierto" a "corregido y medido" (ver cuidado abajo) | `fotos_propias`, cuaderno 03 pasos 5–6b |
| **10.3 Limitaciones** | Actualizar los puntos 1, 5 y 6 (ver abajo) | — |
| **11 Ética** | Agregar el párrafo de abstención | texto abajo |
| **12 Conclusiones** | Reescribir con los números de la v2 | `tabla_comparativa.csv` |
| **Anexo A** | Completar los `[COMPLETAR]` de integrantes y roles | tu equipo |
| **Anexo B** | Agregar las correcciones de reproducibilidad | texto abajo |

## Texto para pegar (no depende de resultados)

### 4.4 Partición

> Los 40.000 recortes de Surface Crack salen de 458 fotografías, cerca de 87 recortes por foto. Una partición
> aleatoria imagen por imagen reparte recortes del mismo muro entre entrenamiento y prueba, de modo que el
> modelo se evalúa con "hermanos" de lo que ya vio y el resultado se infla. La versión 2 divide por **grupos**:
> todas las imágenes de un mismo grupo caen en la misma partición, y el código verifica automáticamente que
> ningún grupo ni imagen aparece en dos particiones (`datos.verificar_sin_fuga`). Los grupos son bloques de
> archivos consecutivos, bajo la hipótesis de que la numeración sigue a la foto de origen; la hipótesis se
> contrastó comparando la similitud de color entre archivos consecutivos y entre pares al azar
> (`datos.prueba_adyacencia`: razón = **[valor del cuaderno 01]**). Si la hipótesis no se sostenía, los grupos
> se forman por similitud de color (`asignar_grupos_cluster`). Fracciones: 70 / 15 / 15, semilla 42.

### 10.3 Limitaciones — puntos que cambian

> **1. Fuga de información por recortes.** *Corregida en la v2* con la partición por grupos. Aun así, los
> grupos son una aproximación de las 458 fotos originales, que el dataset no identifica; el resultado sigue
> siendo optimista respecto de muros nunca vistos, y por eso se evalúa además en una fuente fuera de dominio y
> en fotos propias.
>
> **5. Estimación del ancho.** *Disponible con referencia de escala.* Con un objeto de tamaño conocido en el
> mismo plano que la grieta, el ancho se mide con esqueleto y transformada de distancia (sesgo de −0,02 px en
> pruebas con anchos exactos, peor caso 1 px). Por debajo de ~3 px de ancho el desenfoque domina; sin referencia
> solo se da en píxeles.
>
> **6. La inclinación depende de la perspectiva.** *Parcialmente corregida.* El giro del celular se descuenta
> con el sensor de gravedad; si el celular apunta hacia arriba o abajo (pitch > 5°) las verticales convergen y la
> medición se marca como no confiable.

### 11 Ética — párrafo nuevo

> **Abstención y grietas sin diagnóstico.** Cuando la imagen no parece concreto o el modelo duda
> (probabilidad entre 0,20 y 0,80), el sistema responde INDETERMINADO en lugar de afirmar. Esto reduce las
> falsas alarmas, pero introduce un riesgo propio: una grieta real que queda sin diagnóstico. Por eso se
> reporta, junto con las métricas, el número de **grietas reales sin diagnóstico** en las fotos propias, y la
> recomendación asociada pide repetir la foto o una revisión visual, nunca "sin riesgo".
>
> **Datos personales en las fotos.** Las fotografías propias pueden mostrar rostros, placas, números de casa o
> la ubicación en los metadatos. Antes de publicarlas se eliminan los metadatos EXIF y se evitan personas y
> direcciones identificables.

### Anexo B — reproducibilidad (agregar)

> - Partición por grupos con semilla 42, guardada en `results/particion_*.csv`; entrenamiento y evaluación leen
>   las mismas particiones.
> - El historial se guarda con un archivo por fase y `append=False`. Corrige la v1, cuyo historial de la corrida C
>   tenía 107 filas por un intento interrumpido registrado con `append=True`.
> - `models/<nombre>/meta.json` registra hiperparámetros, versiones de TensorFlow/Keras y semilla.
> - `python tests/run_all.py` ejecuta las pruebas del repositorio; `python -m src.verificar` marca cada requisito
>   de las tres entregas según la evidencia existente.

## Cuidado con el falso positivo (10.1)

**No escribas "el falso positivo fue corregido" salvo que lo midas.** El criterio se fija *antes* de entrenar
(`docs/CORRECCION_FALSO_POSITIVO.md`): por ejemplo, menos de 5 % de falsos positivos en negativos difíciles de
escenas no vistas. Si no se alcanza, la redacción honesta es *"se redujo de X % a Y %; no alcanzó la meta Z %"*,
con las causas probables. Eso se valora más que una afirmación que el cuaderno 03 no respalda.

Y la hipótesis del brillo se decide con `prueba_atajo_brillo`, no con intuición: en el informe va el resultado
de esa tabla, sea cual sea.

## Pendientes tuyos en el informe

* Figura 6 (`[COMPLETAR: captura del resultado en la aplicación]`): captura de la app con la pared lisa, antes y después.
* Anexo A: integrantes y roles.
* Conclusiones y resumen con los números definitivos.

`python -m src.verificar` cuenta cuántos `[COMPLETAR]` quedan y lo muestra en los requisitos 1.6, 2.8 y 3.6.
