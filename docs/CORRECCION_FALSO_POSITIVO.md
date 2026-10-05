# Corrección del falso positivo (pared lisa con mano → "con grieta")

## Diagnóstico

El clasificador tiene **solo dos respuestas**. Cuando ve algo que nunca vio (una mano, una
pared pintada lisa, una baldosa), no puede decir "esto no es concreto": fuerza una de las dos.

Dos hipótesis, y **se miden, no se discuten**:

| Hipótesis | Cómo se prueba | Si se confirma |
|---|---|---|
| El brillo es un atajo | `evaluar.prueba_atajo_brillo` (cuaderno 03, paso 2) | corregir la aumentación o los datos |
| Faltan datos de ese tipo | Grad-CAM sobre la foto (cuaderno 03, paso 4) | agregar negativos difíciles |

La aumentación `RandomBrightness(0.2)` **sí** se aplicó, y antes de `preprocess_input` (orden correcto).
Por eso la segunda hipótesis es la candidata fuerte; el cuaderno la confirma o la descarta.

## Qué NO arregla el falso positivo

* **Grad-CAM**: diagnostica. No cambia el modelo.
* **Entrenar con todo SDNET2018**: añade diversidad, pero gasta la única prueba fuera de dominio.
  Si se usa, que sea **una parte** (paredes) y se aparte otra categoría entera para evaluar.

## Qué SÍ lo arregla (tres capas)

1. **Datos** — negativos difíciles reales en el entrenamiento (este documento).
2. **Puerta "¿parece concreto?"** — `src/ood.py`. Responde INDETERMINADO si la imagen queda lejos de la nube de concreto.
3. **Banda de duda** — entre 0,20 y 0,80 de probabilidad el sistema no afirma (`config.BANDA_INCIERTA`).

## Capa 1 · Cómo juntar los negativos difíciles

**Meta:** unas **300–500 fotos** (más es mejor), de **al menos 25–30 escenas distintas**.

Qué fotografiar (todo SIN grieta):

* Paredes lisas pintadas de varios colores, con y sin mancha de humedad.
* La misma pared con una **mano**, un objeto, una sombra o una persona delante.
* Baldosas, madera, puertas, marcos, cables y tubos (líneas que *parecen* grietas).
* Juntas de construcción y bordes de ladrillo que **no** son grietas.

Cómo organizarlas — **una subcarpeta por escena**:

```
data/dificiles/
├── pared_sala_blanca/        ← 8-15 fotos de la MISMA pared (con y sin mano, distintos ángulos)
├── pared_cocina_verde/
├── baldosa_bano/
└── ...
```

La subcarpeta importa: `grupos: "subcarpeta"` mantiene **todas las fotos de una escena juntas**
en entrenamiento *o* en prueba. Si fotos casi idénticas de una pared cayeran en las dos, el modelo
"se examinaría con hermanos" y se repetiría la fuga del 99,93 %.

Reglas para que no se cree un atajo nuevo:

1. **Las dos clases en cada fuente.** Si todas las fotos de celular son "sin grieta", el modelo aprende
   *"foto de celular = sin grieta"*. Fotografiá también **grietas reales con el mismo celular** (fuente `campo`).
2. **Variá luz, distancia y ángulo** dentro de cada escena.
3. **Tamaño:** cualquiera. Se reduce a 224 px; si la foto es enorme, recortá el área de interés primero.

## Cómo se usa

```python
fuentes = {
  "surface_crack":       {"con_grieta": "data/Positive", "sin_grieta": "data/Negative", "grupos": "bloque"},
  "negativos_dificiles": {"sin_grieta": "data/dificiles", "grupos": "subcarpeta"},
  "campo":               {"con_grieta": "data/campo/con", "sin_grieta": "data/campo/sin", "grupos": "subcarpeta"},
}
```

Cuaderno 01 (EDA, con la auditoría de fuentes) → 02 (sobremuestrea los negativos difíciles ×5) → 03 (medición).

## Criterios de éxito — decidirlos ANTES de entrenar

Elegí los números **antes** de ver resultados (así no se ajustan a lo que salió). Propuesta inicial,
para que la valides con tu docente:

| Métrica | Dónde se mide | Meta propuesta |
|---|---|---|
| Falsos positivos en negativos difíciles de **escenas no vistas** | partición de prueba, fuente `negativos_dificiles` | < 5 % |
| Recall de grietas en la fuente fuera de dominio | `sdnet_fuera` y `campo` | ≥ 90 % |
| Concreto legítimo rechazado por la puerta | validación | ≈ 1 % (por diseño) |
| AUROC de la puerta (concreto vs. no concreto) | negativos difíciles de prueba | ≥ 0,95 |

Si el recall en fuera de dominio queda por debajo de la meta, **el modelo no generaliza** y la respuesta es
ampliar los datos, no ajustar el umbral.

## Límite honesto

Una puerta basada en embeddings puede dejar pasar una pared lisa si el embedding de esa pared cae dentro de la
nube de concreto. **Por eso se mide el AUROC**, no se asume. Y por eso `mosaico` descarta también las teselas
**vecinas** de una rechazada: un objeto en el borde de una tesela deja fragmentos que pasan la puerta.
