# Plan de Trabajo — Proyecto Grietas, Inclinación y Riesgo

**Curso:** Algoritmos y Programación 2026-2 · UIS
**Equipo:** Juan David Sayas Hernandez · Erick Fabian Cardenas Bello

---

## 1. Roles del equipo

Cada integrante tiene un rol principal y uno de respaldo, de forma que ninguna tarea dependa de una sola persona.

| Rol | Responsable | Respaldo | Responsabilidades |
|---|---|---|---|
| **Datos y EDA** | `[ ]` | `[ ]` | Descarga y verificación del conjunto; análisis exploratorio; detección de duplicados; documentación de sesgos; partición estratificada |
| **Modelado** | `[ ]` | `[ ]` | Línea base; transfer learning; ajuste de hiperparámetros; registro de experimentos |
| **Inclinación (Módulo B)** | `[ ]` | `[ ]` | Canny + Hough; calibración de umbrales; medición con sensor; validación cruzada entre métodos |
| **Despliegue** | `[ ]` | `[ ]` | Exportación TFLite o app Streamlit; verificación de paridad entre cuaderno y despliegue; interfaz |
| **Documentación y evaluación** | `[ ]` | `[ ]` | Informe técnico; métricas y figuras; análisis de errores; presentación y sustentación |

**Responsabilidades compartidas por todo el equipo:**
- Captura de fotografías en campo (mínimo 8 imágenes por integrante).
- Revisión cruzada de código antes de fusionar a `main`.
- Asistencia a la reunión semanal de sincronización.

### Convenciones de trabajo en Git

- Rama `main` protegida; nadie sube directo.
- Una rama por funcionalidad: `feat/eda`, `feat/baseline`, `feat/hough`, `feat/app`.
- *Pull request* con revisión de al menos un compañero antes de fusionar.
- Mensajes de *commit* en español, en imperativo: `agrega módulo de preprocesamiento`.
- Cuadernos: limpiar salidas pesadas antes de subir (`jupyter nbconvert --clear-output`).

---

## 2. Cronograma

### Entrega 1 — Formulación y primera implementación funcional
**Fecha límite: 25/09/2026 (antes del 28/09) · 20 %**

| Semana | Actividad | Responsable | Estado |
|---|---|---|---|
| 1 | Conformación del equipo, creación del repositorio, definición de roles | Todos | ☐ |
| 2 | Selección y descarga del conjunto de datos; verificación de integridad | Datos | ☐ |
| 2 | Revisión del estado del arte (mínimo 6 fuentes) | Documentación | ☐ |
| 3 | EDA completo: conteos, ejemplos, histogramas, duplicados | Datos | ☐ |
| 3 | Definición formal del problema y de las clases | Todos | ☐ |
| 4 | Partición train/val/test estratificada y reproducible | Datos | ☐ |
| 4-5 | Módulo de preprocesamiento (`src/preprocessing.py`) | Modelado | ☐ |
| 5 | Línea base entrenada y funcional de extremo a extremo | Modelado | ☐ |
| 5 | Primera estimación de inclinación con Hough sobre foto propia | Inclinación | ☐ |
| 6 | Primera salida de campo: captura de fotografías propias | Todos | ☐ |
| 6-7 | Redacción del informe de Entrega 1 | Documentación | ☐ |
| 7 | Revisión interna completa y limpieza del repositorio | Todos | ☐ |
| 8 | **Entrega** | — | ☐ |

**Criterio de "listo" para la Entrega 1:** una persona ajena al equipo puede clonar el repositorio, seguir el README y obtener una predicción sobre una imagen nueva sin ayuda adicional.

---

### Entrega 2 — Mejora del modelo e integración
**Fecha límite: 23/10/2026 · 20 %**

| Semana | Actividad | Responsable | Estado |
|---|---|---|---|
| 9 | Implementación de la aumentación de datos | Modelado | ☐ |
| 9 | Experimento A/B: con aumentación vs. sin aumentación | Modelado | ☐ |
| 10 | Transfer learning fase 1 (base congelada) | Modelado | ☐ |
| 10 | Transfer learning fase 2 (ajuste fino con LR = 1e-5) | Modelado | ☐ |
| 10 | Calibración de umbrales de Canny sobre fotos propias | Inclinación | ☐ |
| 11 | Validación cruzada Hough vs. sensor (mínimo 10 elementos) | Inclinación | ☐ |
| 11 | Métricas completas: precisión, recall, F1, matriz de confusión | Evaluación | ☐ |
| 11 | Análisis de complejidad: parámetros, MB, tiempo de inferencia | Modelado | ☐ |
| 11 | Implementación del motor de riesgo (`src/riesgo.py`) | Todos | ☐ |
| 12 | Integración de los tres módulos y prueba de extremo a extremo | Despliegue | ☐ |
| 12 | Informe de Entrega 2 | Documentación | ☐ |
| 12 | **Entrega** | — | ☐ |

---

### Entrega 3 — Despliegue, validación y sustentación
**Fecha límite: 20/11/2026 · 30 %**

| Semana | Actividad | Responsable | Estado |
|---|---|---|---|
| 13 | Exportación a TFLite con cuantización int8 | Despliegue | ☐ |
| 13 | Verificación de paridad Keras vs. TFLite (100 imágenes) | Despliegue | ☐ |
| 14 | Aplicación funcional (Streamlit o Android) con los 7 requisitos funcionales | Despliegue | ☐ |
| 14 | Segunda salida de campo: validación con fotos nuevas | Todos | ☐ |
| 14 | Análisis de errores: 10 peores casos, categorizados | Evaluación | ☐ |
| 15 | Comparación entre versiones (desempeño y complejidad) | Evaluación | ☐ |
| 15 | Secciones de limitaciones y ética | Documentación | ☐ |
| 15 | Informe técnico final | Documentación | ☐ |
| 15 | Preparación y ensayo de la sustentación | Todos | ☐ |
| 16 | **Entrega y sustentación con demostración en vivo** | — | ☐ |

---

## 3. Riesgos del proyecto y mitigación

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Colab agota la cuota de GPU en pleno entrenamiento | Alta | Medio | Guardar *checkpoints* cada época en Drive; entrenar de noche; tener listo un plan de entrenamiento en CPU con conjunto reducido |
| Modelo con 99 % en prueba y desempeño malo en fotos propias | Alta | Alto | Es el resultado esperado, no un fallo: documentarlo como hallazgo de cambio de distribución y ampliar el conjunto propio |
| El módulo de Hough detecta líneas espurias | Alta | Medio | Filtrado por longitud y rango angular; uso de mediana en lugar de media; protocolo de captura frontal |
| Discrepancia entre predicciones del cuaderno y de la app | Media | Alto | Un único módulo de preprocesamiento importado por ambos; prueba de paridad automatizada |
| Integrante no disponible cerca de la entrega | Media | Alto | Rol de respaldo asignado; todo el trabajo en el repositorio, nada en máquinas locales |
| Desbalance del conjunto de datos mal manejado | Media | Alto | Reportar F1 y recall, nunca exactitud aislada; ponderación de clases |
| Sobreajuste al conjunto de prueba por evaluaciones repetidas | Media | Medio | El conjunto de prueba se toca una sola vez, al final; todas las decisiones se toman con validación |
| Fotografías de campo insuficientes o de mala calidad | Media | Medio | Protocolo de captura escrito; cuota mínima por integrante; revisión de calidad tras cada salida |

---

## 4. Lista de verificación antes de cada entrega

**Código**
- ☐ El repositorio clona y ejecuta desde cero siguiendo solo el README
- ☐ `requirements.txt` con versiones fijadas
- ☐ Sin credenciales, rutas absolutas ni datos pesados versionados
- ☐ Cuadernos ejecutados en orden, de arriba abajo, sin errores
- ☐ Semillas fijadas y documentadas

**Resultados**
- ☐ Todas las métricas reportadas provienen de ejecuciones reales, no de estimaciones
- ☐ Cada figura tiene título, ejes rotulados y leyenda
- ☐ El conjunto de prueba se evaluó una sola vez
- ☐ Se reportan tanto los resultados buenos como los malos

**Informe**
- ☐ Sin marcadores de pendiente en este documento
- ☐ Todas las referencias verificadas y citadas en el texto
- ☐ Sección de limitaciones escrita sin atenuantes
- ☐ El aviso de "no sustituye inspección profesional" aparece en informe y en la app

**Sustentación**
- ☐ Demostración probada en el equipo y la red que se usarán el día de la presentación
- ☐ Video de respaldo grabado por si la demo en vivo falla
- ☐ Cada integrante puede explicar cualquier parte del proyecto, no solo la suya
- ☐ Ensayo cronometrado dentro del tiempo asignado

---

## 5. Reuniones

**Sincronización semanal** — `[COMPLETAR: día y hora]`, 30 minutos:
1. Qué avanzó cada quien desde la última reunión.
2. Qué está bloqueado y quién puede desbloquearlo.
3. Compromisos de la semana, con responsable y fecha.

Acta breve en `docs/actas/AAAA-MM-DD.md`. Sirve como evidencia del criterio de "trabajo en equipo" de la rúbrica.
