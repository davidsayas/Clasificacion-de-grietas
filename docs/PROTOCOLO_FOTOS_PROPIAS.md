# Protocolo de fotografías propias

El documento del reto lo exige en tres lugares: **Etapa 1** (el equipo captura fotos de grietas reales "para
probar el modelo al final"), **Entrega 1** (primera inclinación sobre una foto propia) y **rúbrica** (desempeño
"en el conjunto de prueba y con fotos propias", 30 % de la nota). Es la única prueba con datos que ningún
dataset contiene, y por eso el desempeño acá suele ser **menor** que en la prueba: reportarlo con sus errores
vale más que ocultarlo.

## Cuántas y cuáles

Cifras **sugeridas** (el plan del equipo pide al menos 8 fotos por integrante; confirmá con tu docente):

| Tipo | Mínimo | Para qué |
|---|---|---|
| Con grieta | 20 | recall y ancho |
| Sin grieta, concreto sano | 10 | falsas alarmas "normales" |
| Sin grieta, **casos difíciles** (pared pintada lisa, mano o sombra delante, baldosa, juntas, cables) | 10 | el falso positivo |

Con pocas fotos, cada error pesa mucho: con 30 fotos, un error es 3,3 %. Por eso las métricas de este conjunto
se reportan **con el número de fotos al lado** y sin extrapolar.

Variá: superficie (muro, andén, columna, viga), luz (sol, sombra, interior), distancia y orientación de la grieta
(vertical, horizontal, diagonal).

## Cómo tomar cada foto

1. **De frente** al muro, con el celular lo más vertical posible. Si apuntás hacia arriba o abajo, las verticales
   convergen y la inclinación no es confiable.
2. A **50–100 cm** para fotos de la grieta; una más **lejana** (2–3 m) de la misma zona si querés probar el mosaico.
3. **Referencia de escala en el mismo plano que la grieta:** una tarjeta bancaria (85,60 × 53,98 mm) pegada con
   cinta junto a la grieta, o una regla. Sin referencia no hay ancho en mm.
4. Si podés, anotá el **giro del celular** (roll) con una app de nivel/inclinómetro. Sin ese dato la inclinación
   del muro no se usa, porque no se distingue del giro del celular.
5. Foto nítida: sin zoom digital, con la grieta enfocada.

## Cómo organizarlas

```
data/fotos_propias/
├── foto01.jpg
├── foto02.jpg
├── ...
└── etiquetas.csv
```

`etiquetas.csv` (1 = con grieta, 0 = sin grieta):

```csv
archivo,etiqueta
foto01.jpg,1
foto02.jpg,0
pared_con_mano.jpg,0
```

Datos que cambian de foto en foto (escala, giro, elemento) se pasan en el cuaderno 03, paso 6b:

```python
parametros = {
    "foto01.jpg": {"mm_por_px": 0.21, "roll_deg": 3.0, "pitch_deg": 0.0, "elemento": "columna"},
}
```

`mm_por_px` = largo real de la referencia (mm) ÷ su largo en la foto (px). Con una tarjeta de 85,6 mm que mide
428 px: 0,2 mm/px. Para fotos tomadas en ángulo, el sistema puede rectificar la perspectiva con las cuatro
esquinas de la tarjeta (`esquinas_ref`, `ref_mm`).

## Reglas para que la prueba sea honesta

* **Estas fotos NUNCA se usan para entrenar.** Si el modelo las ve, la prueba deja de medir generalización.
  (La fuente `campo` del cuaderno 01 es otra cosa: fotos *adicionales*, separadas de estas.)
* **Decidí las etiquetas antes de correr el modelo**, mirando la foto, no la respuesta del sistema.
* **Guardá todos los errores.** El análisis de falsos negativos (grietas peligrosas no detectadas) es parte de
  lo que pide la Etapa 4 del documento.

## Privacidad

* Evitá **personas, placas y números de casa** en el encuadre.
* **Eliminá los metadatos EXIF** (incluida la ubicación GPS) antes de subir nada al repositorio.
* No fotografíes viviendas ajenas sin permiso. El campus y los espacios públicos son más simples.
* El `.gitignore` del proyecto excluye `data/`: las fotos no se suben por accidente. Para el informe usá solo
  las que puedas mostrar.
