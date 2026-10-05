"""Genera los cuadernos .ipynb (JSON puro, sin nbformat)."""
import json
from pathlib import Path


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(True)}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.strip("\n").splitlines(True)}


def guardar(nombre, celdas):
    nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}}}
    Path(nombre).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


import sys
sys.path.insert(0, str(Path(__file__).parent))
from _preparacion import PREPARACION

FUENTES = '''
fuentes = {
    "surface_crack": {"con_grieta": f"{RUTA_DATOS}/Positive", "sin_grieta": f"{RUTA_DATOS}/Negative",
                      "grupos": "bloque", "tam_bloque": config.TAM_BLOQUE},
    # --- Descomentá cada una cuando tengas las imágenes (ver docs/CORRECCION_FALSO_POSITIVO.md) ---
    # "negativos_dificiles": {"sin_grieta": f"{PROYECTO}/data/dificiles", "grupos": "subcarpeta"},
    # "campo":               {"con_grieta": f"{PROYECTO}/data/campo/con", "sin_grieta": f"{PROYECTO}/data/campo/sin", "grupos": "subcarpeta"},
    # SDNET2018: REPARTILO, no lo uses entero para entrenar. Verificá los nombres de carpeta al descargarlo.
    #   para ENTRENAR (paredes, lo más parecido a tu caso):
    # "sdnet_paredes":       {"con_grieta": f"{PROYECTO}/data/sdnet/W/CW", "sin_grieta": f"{PROYECTO}/data/sdnet/W/UW", "grupos": "archivo"},
    #   para PROBAR fuera de dominio (otra categoría entera; se aparta con reservar_fuente):
    # "sdnet_fuera":         {"con_grieta": f"{PROYECTO}/data/sdnet/P/CP", "sin_grieta": f"{PROYECTO}/data/sdnet/P/UP", "grupos": "archivo"},
}
'''

# ============================================================== 01 EDA
eda_nb = [
    md("""
# 01 · Análisis exploratorio (EDA)

## ¿Qué hace el EDA y por qué se hace ANTES de entrenar?
El EDA responde una sola pregunta: **¿estos datos me dejan aprender lo que quiero aprender?**
No es "mirar gráficos": cada paso busca una trampa concreta que, si no se detecta ahora, infla los resultados o
produce un modelo que falla en la calle.

| Paso | Pregunta | Trampa que busca |
|---|---|---|
| 1. Inventario | ¿qué tengo y de dónde viene? | fuentes con una sola clase (atajo de la fuente) |
| 2. Tamaño y color | ¿las imágenes son comparables? | tamaños mezclados, tintes distintos por fuente |
| 3. Brillo | ¿las clases se distinguen por la luz? | **atajo del brillo** |
| 4. Duplicados | ¿hay imágenes repetidas? | fuga por copias |
| 5. Adyacencia | ¿archivos consecutivos son de la misma foto? | decide cómo agrupar |
| 6. División | ¿cómo separo entrenamiento/validación/prueba? | **fuga por hermanos** (el 99,93 % inflado) |
| 7. Línea base | ¿cuánto logra un método simple? | un 99 % sin referencia no significa nada |
"""),
    code(PREPARACION),
    md("""
## Paso 1 · Inventario
**Qué hace:** recorre las carpetas y arma una tabla con una fila por imagen: ruta, etiqueta (1 = con grieta),
fuente, **grupo** y orden. El *grupo* es la idea clave: las imágenes "hermanas" (recortes de la misma foto, fotos
de la misma pared) comparten grupo y **nunca podrán separarse** entre entrenamiento y prueba.

**Qué mirar:** el balance entre clases y la **auditoría de fuentes**. Si una fuente tiene solo una clase, el modelo
podría aprender "foto de celular = sin grieta" en vez de aprender qué es una grieta.
"""),
    code(FUENTES + '''
df = datos.construir_inventario(fuentes)
print(f"{len(df):,} imágenes · {df['grupo'].nunique():,} grupos")
tabla, avisos = datos.auditoria_fuentes(df)
display(tabla)
for a in avisos:
    print("⚠", a)
'''),
    md("""
## Paso 1b · Apartar lo que NO se usa para entrenar
**Qué hace:** saca del inventario las fuentes reservadas para evaluar (SDNET fuera de dominio y tus fotos de campo) y
las guarda aparte. **Se hace antes de dividir**, para que ninguna imagen de evaluación llegue al entrenamiento ni a la
validación. Si entrenaras con ellas, la prueba fuera de dominio dejaría de ser honesta.
"""),
    code('''
EVALUACION = ["sdnet_fuera", "campo"]          # fuentes que el modelo NUNCA ve al entrenar
fuera = df[df["fuente"].isin(EVALUACION)]
df = df[~df["fuente"].isin(EVALUACION)].reset_index(drop=True)
Path("results").mkdir(exist_ok=True)
fuera.to_csv("results/fuera_de_dominio.csv", index=False)
print(f"Para entrenar/validar/probar: {len(df):,} imágenes · Apartadas para evaluar fuera de dominio: {len(fuera):,}")
'''),
    md("""
## Paso 2 · Tamaño y color
**Qué hace:** mide el tamaño (ancho × alto × modo) y la media de cada canal RGB en una muestra, **por fuente**.

**Qué mirar:** todas las imágenes de una fuente deberían tener el mismo tamaño. Concreto gris → R ≈ G ≈ B; una
fuente con un tinte distinto (pared pintada, filtro cálido) se nota aquí.
"""),
    code('''
for fuente, sub in df.groupby("fuente"):
    rutas = sub["ruta"].tolist()
    print(f"\\n== {fuente} ==")
    print("tamaños:", dict(eda.resumen_tamanos(rutas, n=300)))
    print("color medio:", {k: round(v, 1) for k, v in eda.resumen_color(rutas, n=300).items()})
'''),
    md("""
## Paso 3 · Brillo por imagen (el atajo del brillo)
**Qué hace:** calcula el brillo promedio (0–255) de **cada** imagen y compara las dos clases. Es tu función
`brillo_promedio`, pero guardando **una lista** (un valor por imagen) en vez de un acumulador (un solo número).

**Cómo leerlo**
* **Cohen d**: separación entre clases en desvíos estándar (0,2 chico · 0,5 mediano · 0,8 grande).
* **Clasificador tonto**: la exactitud MÁXIMA que lograría alguien que *solo* mirara el brillo. Si da ~65 %, el
  atajo es débil. Si diera 95 %, el modelo seguramente lo estaría usando.

**Qué decidir:** si el atajo es débil, la aumentación de brillo basta; si es fuerte, hay que corregir los datos.
"""),
    code('''
N = 4000   # muestra por clase para ir rápido; poné None para usar todas
pos = df[df.etiqueta == 1]["ruta"].tolist()[:N]
neg = df[df.etiqueta == 0]["ruta"].tolist()[:N]
b_pos, b_neg = eda.brillo_por_imagen(pos), eda.brillo_por_imagen(neg)
print(f"brillo medio  con grieta: {b_pos.mean():.1f} | sin grieta: {b_neg.mean():.1f}")
print(f"Cohen d = {eda.cohen_d(b_neg, b_pos):.2f}")
tonto = eda.clasificador_tonto_brillo(b_pos, b_neg)
print(f"Un clasificador que SOLO mira el brillo acierta como máximo {100*tonto['exactitud']:.1f} % (azar = {100*tonto['azar']:.0f} %)")
Path("results").mkdir(exist_ok=True)
eda.figura_brillo(b_pos, b_neg, "results/histograma_brillo.png");
'''),
    md("""
## Paso 4 · Duplicados (huella perceptual)
**Qué hace:** reduce cada imagen a una huella de 64 bits (gris → 8×8 → cada píxel contra el promedio). Dos
imágenes con la misma huella son casi-copias. La huella **no cambia** si se aclara la imagen uniformemente.

**Límite:** es una cota inferior. No detecta *recortes distintos de la misma foto*: eso lo resuelve el paso 5.
"""),
    code('''
muestra = df.sample(min(4000, len(df)), random_state=config.SEMILLA)
dup = datos.encontrar_duplicados(muestra["ruta"].tolist())
redundantes = sum(len(v) - 1 for v in dup.values())
print(f"{len(dup)} grupos de duplicados · {redundantes} imágenes redundantes ({100*redundantes/len(muestra):.2f} % de la muestra)")
'''),
    md("""
## Paso 5 · ¿Los archivos consecutivos son de la misma foto?
**Por qué importa:** Surface Crack son ~40.000 recortes salidos de 458 fotos. Si los archivos consecutivos salen de
la misma foto, partir por **bloques de ~44** mantiene juntos a los hermanos. Pero eso es una **hipótesis**.

**Qué hace:** compara la distancia (en color medio) entre archivos consecutivos con la de pares al azar.
* razón **< 0,7** → los consecutivos se parecen → los bloques sirven.
* razón **≈ 1** → no hay relación → usar agrupación por similitud (`asignar_grupos_cluster`).
"""),
    code('''
# se calculan sobre el inventario ordenado (la prueba necesita el orden de los archivos)
sub = df.groupby(["fuente", "etiqueta"], group_keys=False).apply(lambda g: g.head(3000)).reset_index(drop=True)
rasgos = datos.extraer_rasgos(sub)
adyacencia = datos.prueba_adyacencia(sub, rasgos)
print(adyacencia)
'''),
    md("""
**Decisión.** Mirá `razon` y `veredicto`. Si dice que los bloques NO sirven, ejecutá la celda siguiente (agrupar
por similitud). Si sirven, saltala.
"""),
    code('''
USAR_CLUSTER = adyacencia["razon"] >= 0.9      # ← cambialo a mano si querés forzarlo
if USAR_CLUSTER:
    rasgos_todos = datos.extraer_rasgos(df)
    df = datos.asignar_grupos_cluster(df, rasgos_todos, imagenes_por_grupo=config.TAM_BLOQUE)
    print("Grupos reasignados por similitud:", df["grupo"].nunique())
else:
    print("Se mantienen los bloques:", df["grupo"].nunique(), "grupos")
'''),
    md("""
## Paso 6 · División sin fuga
**Qué hace:** separa en entrenamiento / validación / prueba **por grupos enteros**. Verifica con un `assert` que
ningún grupo ni imagen aparece en dos particiones.

**Por qué:** dividir imagen por imagen mezclaría recortes del mismo muro en entrenamiento y prueba. El modelo
"se examinaría con hermanos de lo que ya vio" y el resultado saldría inflado. La celda siguiente **lo demuestra**
con tus datos.
"""),
    code('''
from sklearn.model_selection import train_test_split
tr0, te0 = train_test_split(df, test_size=0.15, random_state=config.SEMILLA)
mezclados = len(set(tr0["grupo"]) & set(te0["grupo"]))
print(f"División imagen por imagen (la anterior): {mezclados} grupos aparecen en entrenamiento Y en prueba  ← FUGA")

tr, va, te = datos.dividir(df)
print("División por grupos: sin fuga (verificada)")
display(datos.resumen_particiones(entrenamiento=tr, validacion=va, prueba=te))
print("huellas compartidas train↔test:", datos.huellas_compartidas(tr.sample(min(2000, len(tr)), random_state=1), te.sample(min(2000, len(te)), random_state=1)))
'''),
    md("""
## Paso 7 · Línea base (¿cuánto logra un método simple?)
**Qué hace:** ajusta tres métodos **sin red neuronal** con el entrenamiento y los mide en prueba: umbral de brillo,
umbral de bordes y regresión logística sobre 5 rasgos. Esta es la tabla que el proyecto tenía marcada
como *[pendiente]*.

**Para qué:** sin una línea base, "99 %" no significa nada. La red debe superar claramente a estos métodos
y, sobre todo, **medirse con la misma división sin fuga**.
"""),
    code('''
from src import linea_base
sub_tr = tr.sample(min(6000, len(tr)), random_state=config.SEMILLA).reset_index(drop=True)
sub_te = te.sample(min(3000, len(te)), random_state=config.SEMILLA).reset_index(drop=True)
tabla_lb = linea_base.tabla_linea_base(sub_tr, sub_te)
display(tabla_lb)
tabla_lb.to_csv("results/linea_base.csv", index=False)
'''),
    md("""
## Guardar las particiones
Se guardan para que **entrenamiento y evaluación usen exactamente las mismas** (reproducibilidad).
"""),
    code('''
tr.to_csv("results/particion_entrenamiento.csv", index=False)
va.to_csv("results/particion_validacion.csv", index=False)
te.to_csv("results/particion_prueba.csv", index=False)
df.to_csv("results/inventario.csv", index=False)
print("Guardado en results/. Siguiente: 02_Entrenamiento.ipynb")
'''),
]

# ============================================================== 02 ENTRENAMIENTO
ent_nb = [
    md("""
# 02 · Entrenamiento

## ¿Qué hace el entrenamiento?
Mostrarle a la red miles de imágenes con su respuesta correcta y ajustar sus pesos para que se equivoque menos.
Cada vuelta completa sobre el entrenamiento es una **época**. Tras cada época se mide el error en **validación**
(imágenes que la red no usa para aprender) para decidir cuándo parar y qué versión guardar.

| Conjunto | Para qué sirve | Quién lo usa |
|---|---|---|
| entrenamiento | la red **aprende** con esto | el entrenamiento |
| validación | **elegir** (cuándo parar, qué época guardar, qué lr) | tú, durante el desarrollo |
| prueba | **informar** el resultado final, UNA sola vez | solo al terminar |

## Las dos fases (transfer learning)
1. **Fase 1:** la base MobileNetV2 (ya entrenada con ImageNet) está *congelada*; solo aprende la cabeza nueva (lr 1e-3).
2. **Fase 2 (ajuste fino):** se liberan las últimas 30 capas de la base con lr muy bajo (1e-5) para adaptarlas.

## Normalización
La red espera valores en un rango concreto. MobileNetV2 usa `[-1, 1]`: `x/127.5 − 1`. Está **dentro del modelo**, justo
*después* de la aumentación (que trabaja en 0–255). Si fuera al revés, `RandomBrightness` quedaría roto.
"""),
    code(PREPARACION.replace("from src import config, datos, eda, evaluar", "from src import config, datos, evaluar, entrenar, modelo") + '''
import tensorflow as tf
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
'''),
    md("""
## Paso 1 · Cargar las particiones del EDA
Mismas particiones que se guardaron en el cuaderno 01. Nada se vuelve a dividir aquí.
"""),
    code('''
tr = pd.read_csv("results/particion_entrenamiento.csv")
va = pd.read_csv("results/particion_validacion.csv")
datos.verificar_sin_fuga(tr, va)
print(datos.resumen_particiones(entrenamiento=tr, validacion=va))
'''),
    md("""
## Paso 2 · (Corrección del falso positivo) Sobremuestrear los negativos difíciles
**Qué hace:** repite varias veces las filas de la fuente `negativos_dificiles` en el entrenamiento. Sin esto, 300
imágenes entre 40.000 casi no influyen.

**Cuidado:** si se repiten demasiado, la red las **memoriza**. Por eso la validación debe tener *escenas distintas*
(se logra con `grupos: "subcarpeta"`). Si no tenés aún esa fuente, saltá la celda.
"""),
    code('''
if "negativos_dificiles" in set(tr["fuente"]):
    tr = entrenar.sobremuestrear(tr, "negativos_dificiles", veces=5)
    print("Entrenamiento tras sobremuestrear:", len(tr))
else:
    print("Todavía no hay negativos difíciles: se entrena solo con Surface Crack.")
'''),
    md("""
## Paso 3 · Barrido de la tasa de aprendizaje (rápido)
**Qué hace:** entrena 5 épocas con 8.000 imágenes para cada lr y mira el `val_loss`. La tasa de aprendizaje es
*cuánto se mueven los pesos en cada paso*: muy alta, la red se descontrola; muy baja, aprende lentísimo.

**Qué decidir:** usar el lr de menor `val_loss_min` como `lr1`. Es una prueba para **elegir**, no el entrenamiento final.
"""),
    code('''
barrido = entrenar.barrido_lr(tr, va, lrs=(1e-2, 1e-3, 1e-4))
display(barrido)
LR1 = float(barrido.loc[0, "lr"])
print("lr elegido para la fase 1:", LR1)
'''),
    md("""
## Paso 4 · Entrenar MobileNetV2
**Qué hace:** fase 1 + fase 2, guardando el mejor modelo por `val_loss`, **un historial por fase** (`append=False`: se
acabó el historial con filas de un intento interrumpido) y un `meta.json` con todo lo necesario para reproducir.

**Paciencia:** `paciencia=50` en la corrida anterior significó, en la práctica, *no parar antes*. La parada temprana con
paciencia 2 cortaba durante una meseta temporal. Se deja igual para que sea comparable.
"""),
    code('''
meta = entrenar.entrenar("v2_mobilenet", tr, va, arquitectura="mobilenetv2", lr1=LR1,
                         epocas1=60, epocas2=40, paciencia=50)
print(meta)
'''),
    md("""
## Paso 5 · Entrenar el segundo modelo (EfficientNetB0)
**Por qué:** el profesor advirtió que si todos usan el mismo modelo la nota se reparte. Comparar **MobileNetV2 vs
EfficientNetB0** con la misma división, semilla y aumentación es un experimento limpio (cambia **un** factor).
EfficientNetB0 es más precisa pero más pesada: la comparación debe incluir **tamaño y tiempo de inferencia**, no solo exactitud.
"""),
    code('''
meta2 = entrenar.entrenar("v2_efficientnet", tr, va, arquitectura="efficientnetb0", lr1=LR1,
                          epocas1=60, epocas2=40, paciencia=50)
'''),
    md("""
## Paso 6 · Curvas de entrenamiento
**Qué mirar:** `val_loss` bajando y luego estable. Si `loss` baja pero `val_loss` sube → sobreajuste.
**Recordá:** estas curvas **no detectan** problemas de dominio (la validación sale de las mismas fotos). Eso lo
hace el cuaderno 03.
"""),
    code('''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for nombre in ("v2_mobilenet", "v2_efficientnet"):
    for fase in (1, 2):
        ruta = Path("models") / nombre / f"historial_fase{fase}.csv"
        if ruta.exists():
            h = pd.read_csv(ruta)
            ax[0].plot(h["val_loss"], label=f"{nombre} f{fase}")
            ax[1].plot(h["val_recall"], label=f"{nombre} f{fase}")
ax[0].set_title("val_loss"); ax[1].set_title("val_recall")
for a in ax: a.legend()
fig.tight_layout(); fig.savefig("results/curvas.png", dpi=150)
'''),
    md("""
## Paso 7 · Evaluación en validación con intervalo de confianza
**Qué hace:** matriz de confusión y métricas. El **recall** (sensibilidad) importa más que la exactitud: un falso
negativo es una grieta que no se vio. Con el intervalo de Wilson se reporta la **incertidumbre**: "2 errores de
3.000" no es "0,07 % exacto".

La **prueba** (test) se evalúa en el cuaderno 03, una sola vez.
"""),
    code('''
from tensorflow import keras
for nombre, arq in [(n, a) for n, a in (("v2_mobilenet", "mobilenetv2"), ("v2_efficientnet", "efficientnetb0")) if Path(f"models/{n}/modelo.keras").exists()]:
    m = keras.models.load_model(f"models/{nombre}/modelo.keras")
    inf = modelo.modelo_inferencia(m, arq)
    prob = entrenar.predecir_rutas(entrenar.predictor(inf), va["ruta"].tolist())
    r = evaluar.metricas(va["etiqueta"].to_numpy(), prob)
    lo, hi = evaluar.intervalo_wilson(r["fn"], r["fn"] + r["tp"])
    print(f"{nombre}: exactitud {r['exactitud']:.4f} | recall {r['recall']:.4f} | falsos neg. {r['fn']} "
          f"(tasa {100*r['fn']/(r['fn']+r['tp']):.2f} %, IC95 {100*lo:.2f}–{100*hi:.2f} %) | falsos pos. {r['fp']}")
'''),
    md("""
## Paso 8 · Exportar para el celular (TFLite, float16)
**Qué hace:** convierte el modelo (sin aumentación ni dropout) a TFLite con pesos en float16 (mitad de tamaño).
"""),
    code('''
m = keras.models.load_model("models/v2_mobilenet/modelo.keras")
tam = modelo.exportar_tflite(m, "models/v2_mobilenet/modelo.tflite", "mobilenetv2")
print(f"modelo.tflite: {tam/1e6:.2f} MB")
'''),
]

# ============================================================== 03 ROBUSTEZ
rob_nb = [
    md("""
# 03 · Robustez, Grad-CAM y corrección del falso positivo

## Por qué existe este cuaderno
Las curvas de entrenamiento **no detectan** cambios de dominio: la validación sale de las mismas fotos. Lo único que
los detecta son pruebas que cambian **un factor a la vez** y miden cuánto se mueve la respuesta del modelo.

## Hipótesis a contrastar con evidencia
> *"El falso positivo de la pared lisa con la mano se debe a que el modelo no aprendió con variación de brillo."*

El código de `construir_modelo` aplica la aumentación (incluida `RandomBrightness`) **antes** de `preprocess_input`,
así que la aumentación sí estuvo activa. Esta hipótesis se decide con la **prueba del atajo del brillo** (paso 2), no con
discusión: si casi ninguna imagen cambia de clase al oscurecerla, el brillo no es la causa.
"""),
    code(PREPARACION.replace("from src import config, datos, eda, evaluar", "from src import config, datos, evaluar, entrenar, modelo, gradcam, ood, mosaico, predecir") + '''
from tensorflow import keras
NOMBRE, ARQ = "v2_mobilenet", "mobilenetv2"
m = keras.models.load_model(f"models/{NOMBRE}/modelo.keras")
inf = modelo.modelo_inferencia(m, ARQ)
predecir_fn = entrenar.predictor(inf)
te = pd.read_csv("results/particion_prueba.csv")
va = pd.read_csv("results/particion_validacion.csv")
'''),
    md("""
## Paso 1 · Resultado en PRUEBA (se mira UNA vez)
Es el resultado que se informa. Con división por grupos, **se espera que sea menor** que el 99,93 % anterior: esa
diferencia es la fuga que se eliminó, no un empeoramiento.
"""),
    code('''
prob_te = entrenar.predecir_rutas(predecir_fn, te["ruta"].tolist())
r = evaluar.metricas(te["etiqueta"].to_numpy(), prob_te)
lo, hi = evaluar.intervalo_wilson(r["fn"], r["fn"] + r["tp"])
print(r["matriz"], f"\\nexactitud {r['exactitud']:.4f} · recall {r['recall']:.4f} · precisión {r['precision']:.4f}")
print(f"tasa de falsos negativos {100*r['fn']/(r['fn']+r['tp']):.2f} % (IC95 {100*lo:.2f}–{100*hi:.2f} %)")
'''),
    md("""
## Paso 2 · Prueba del atajo del brillo
**Qué hace:** toma imágenes **sin grieta** y las **oscurece** (−18, −30, −51 niveles) y cuenta cuántas pasan a *con
grieta*; y al revés, aclara las **con grieta**. Solo cambia el brillo: misma textura, mismo concreto.

* 18 = la diferencia real entre clases · 51 = el máximo de la aumentación.
* **≈ 0 % cambia → el modelo NO usa el brillo.** Hasta ~5 % es tolerable.
* **Mucho % → el modelo usa el brillo como atajo.**

Se probó con modelos falsos (`tests/test_evaluar.py`): detecta a uno que usa el brillo y NO acusa a uno que mira la textura.
"""),
    code('''
n = 300
neg = entrenar.cargar_imagenes(te[te.etiqueta == 0]["ruta"].tolist()[:n])
pos = entrenar.cargar_imagenes(te[te.etiqueta == 1]["ruta"].tolist()[:n])
t_brillo = evaluar.prueba_atajo_brillo(predecir_fn, neg, pos, deltas=(18, 30, 51))
display(t_brillo)
print(evaluar.veredicto_atajo(t_brillo))
t_brillo.to_csv("results/prueba_atajo_brillo.csv", index=False)
'''),
    md("""
## Paso 3 · Robustez: giro, lejanía, perspectiva, desenfoque, ruido
**Qué hace:** aplica cada transformación con varias intensidades y mide **% que cambia de clase**, exactitud y recall.

* **Giro hasta 54°:** lo que `RandomRotation(0.15)` cubre *en teoría*; aquí se mide *en la práctica*.
* **Lejanía:** simula una foto tomada de lejos (achicar y volver). Con 0,25 una grieta de 4 px queda en 1 px.
  Si el recall cae acá, **la solución es el mosaico** (paso 5).
* **Perspectiva:** foto en ángulo. Afecta más a la medición del ancho que a la clasificación.
"""),
    code('''
mezcla = pd.concat([te[te.etiqueta == 0].head(150), te[te.etiqueta == 1].head(150)])
imgs = entrenar.cargar_imagenes(mezcla["ruta"].tolist())
t_rob = evaluar.prueba_robustez(predecir_fn, imgs, mezcla["etiqueta"].to_numpy())
display(t_rob)
t_rob.to_csv("results/prueba_robustez.csv", index=False)
'''),
    md("""
## Paso 4 · Grad-CAM: ¿en qué se fija el modelo?
**Qué hace:** pinta sobre la foto las zonas que más empujaron la decisión hacia *grieta*.

Subí a `data/problemas/` tu foto de la **pared lisa con la mano** y otras fallas.

| Si marca… | Significa |
|---|---|
| bordes y sombras de los dedos | confunde líneas oscuras con grietas → faltan **datos** de ese tipo |
| el fondo liso / todo parejo | usa brillo o textura global como atajo |
| la grieta real | mira lo correcto |
"""),
    code('''
problemas = sorted(Path("data/problemas").glob("*.jp*g")) + sorted(Path("data/problemas").glob("*.png"))
if problemas:
    gradcam.figura_gradcam(m, problemas[:6], arquitectura=ARQ, ruta_salida="results/gradcam_problemas.png")
else:
    print("Poné fotos en data/problemas/ (la pared con la mano, etc.)")
'''),
    md("""
## Paso 5 · Corrección del falso positivo: la puerta "¿esto parece concreto?"
**El problema de fondo:** el clasificador solo conoce *dos* respuestas. Ante algo que nunca vio (una mano, una pared
pintada) no puede decir "esto no es concreto": fuerza una de las dos.

**La corrección (en tres capas, de más a menos importante):**
1. **Datos:** negativos difíciles reales en el entrenamiento (cuaderno 02, paso 2).
2. **Puerta de concreto:** distancia de Mahalanobis sobre el embedding de 1.280 números. Si la imagen queda lejos
   de la nube de concreto, el sistema responde *INDETERMINADO* en vez de afirmar.
3. **Banda de duda:** entre 0,20 y 0,80 de probabilidad el sistema no afirma.

**Qué medir:** el **AUROC** de la puerta (1,0 = separa perfecto · 0,5 = azar) y la **tasa de rechazo de concreto legítimo**
(por diseño ~1 %). No se asume que funciona: se mide con negativos difíciles de **escenas que no vio**.
"""),
    code('''
emb_modelo = modelo.modelo_embeddings(m, ARQ)
emb = lambda x: emb_modelo.predict(np.asarray(x, dtype=np.float32), batch_size=64, verbose=0)
tr = pd.read_csv("results/particion_entrenamiento.csv")
conc_tr = tr[tr.fuente == "surface_crack"].sample(4000, random_state=1)
conc_va = va[va.fuente == "surface_crack"].sample(1500, random_state=1)
puerta = ood.PuertaConcreto(percentil=99).ajustar(entrenar.predecir_rutas(emb, conc_tr["ruta"].tolist()),
                                                  entrenar.predecir_rutas(emb, conc_va["ruta"].tolist()))
puerta.guardar(f"models/{NOMBRE}/puerta.npz")
print("umbral de distancia:", round(puerta.umbral, 2))
'''),
    code('''
# Con los negativos difíciles de escenas NO vistas (los que están en la partición de prueba):
dificiles = te[te.fuente == "negativos_dificiles"]
if len(dificiles):
    e_dif = entrenar.predecir_rutas(emb, dificiles["ruta"].tolist())
    e_con = entrenar.predecir_rutas(emb, te[te.fuente == "surface_crack"].head(1000)["ruta"].tolist())
    print("AUROC de la puerta:", round(puerta.auroc(e_con, e_dif), 3))
    print("concreto legítimo rechazado:", f"{100*(1-puerta.es_concreto(e_con).mean()):.1f} %")
    print("negativos difíciles rechazados (lo que queremos):", f"{100*(1-puerta.es_concreto(e_dif).mean()):.1f} %")
else:
    print("Falta la fuente negativos_dificiles en la partición de prueba (docs/CORRECCION_FALSO_POSITIVO.md).")
'''),
    md("""
## Paso 6 · El sistema completo sobre tus fotos
Encadena: puerta → clasificación (mosaico si la foto es grande) → ancho → inclinación → riesgo. Para el ancho en mm hay
que dar la escala (`mm_por_px`) o las 4 esquinas de un objeto de referencia (`esquinas_ref`, `ref_mm`).
"""),
    code('''
from PIL import Image
sistema = predecir.Sistema(predecir_fn)   # SIN puerta de concreto: calibrada solo con Surface Crack, descartaba las grietas de celular (Paso 6d)
for ruta in problemas[:6]:
    foto = np.asarray(Image.open(ruta).convert("RGB"))
    r = sistema.analizar(foto, roll_deg=0.0, pitch_deg=0.0, escalas=(1.0, 0.5))
    print(f"{ruta.name}: p(grieta)={r['prob_grieta']:.2f} · concreto={r['es_concreto']} · "
          f"riesgo={r['riesgo'].nivel} · {r['riesgo'].razones}")
'''),
    md("""
## Paso 7 · Fuera de dominio: SDNET2018 y fotos de campo (solo evaluación)
Estas fuentes **no se usaron para entrenar** (se apartaron en el cuaderno 01, paso 1b). Es la prueba honesta de generalización:
si el recall aquí es mucho menor que en `surface_crack`, el modelo no generaliza y hay que ampliar los datos.
"""),
    code('''
fuera = pd.read_csv("results/fuera_de_dominio.csv")
for fuente in ("sdnet_fuera", "campo"):
    sub = fuera[fuera.fuente == fuente]
    if len(sub):
        sub = sub.sample(min(1500, len(sub)), random_state=1)
        p = entrenar.predecir_rutas(predecir_fn, sub["ruta"].tolist())
        r = evaluar.metricas(sub["etiqueta"].to_numpy(), p)
        print(f"{fuente}: recall {r['recall']:.3f} · precisión {r['precision']:.3f} · exactitud {r['exactitud']:.3f} · FN {r['fn']} · FP {r['fp']}")
    else:
        print(f"{fuente}: sin imágenes todavía")
'''),
    md("""
## Paso 8 · Tabla comparativa final (para el informe)
Une la línea base (cuaderno 01), MobileNetV2 y EfficientNetB0 en **la misma partición de prueba**. Agregá el tamaño
del `.tflite` y el tiempo de inferencia si vas a justificar la elección del modelo.
"""),
    code('''
filas = []
lb = pd.read_csv("results/linea_base.csv")
for _, f in lb.iterrows():
    filas.append({"modelo": "línea base: " + f["metodo"], "exactitud": f["exactitud"], "recall": f["recall"], "precision": f["precision"], "FN": f["falsos_neg"], "FP": f["falsos_pos"]})
for nombre, arq in [(n, a) for n, a in (("v2_mobilenet", "mobilenetv2"), ("v2_efficientnet", "efficientnetb0")) if Path(f"models/{n}/modelo.keras").exists()]:
    mm = keras.models.load_model(f"models/{nombre}/modelo.keras")
    p = entrenar.predecir_rutas(entrenar.predictor(modelo.modelo_inferencia(mm, arq)), te["ruta"].tolist())
    r = evaluar.metricas(te["etiqueta"].to_numpy(), p)
    filas.append({"modelo": nombre, "exactitud": round(r["exactitud"], 4), "recall": round(r["recall"], 4), "precision": round(r["precision"], 4), "FN": r["fn"], "FP": r["fp"]})
tabla_final = pd.DataFrame(filas)
display(tabla_final)
tabla_final.to_csv("results/tabla_comparativa.csv", index=False)
'''),
]

# ============================================================== 04 SEGMENTACIÓN Y ANCHO
seg_nb = [
    md("""
# 04 · Segmentación (U-Net) y medición del ancho

## La cadena completa
`MobileNetV2` decide **si** hay grieta → `U-Net` dice **dónde** (píxel a píxel) → `ancho.py` mide **cuántos mm**.

## Qué necesitás
Pares imagen + máscara en `datos_seg/imagenes/` y `datos_seg/mascaras/` (mismo nombre de archivo, máscara blanca = grieta).
Surface Crack **no trae máscaras**. Conjuntos públicos de segmentación de grietas a buscar y verificar (licencia y disponibilidad):
DeepCrack, CrackForest, "Concrete Crack Segmentation". Mientras tanto, `ancho.segmentar_clasico` funciona **sin red**.
"""),
    code(PREPARACION.replace("from src import config, datos, eda, evaluar", "from src import config, ancho, unet") + '''
import cv2
from tensorflow import keras
'''),
    md("""
## Paso 1 · Entrenar la U-Net
Pérdida **BCE + Dice**: las grietas ocupan pocos píxeles; solo BCE aprendería a decir "nada" en todos lados, y Dice
penaliza no encontrarlas. Métricas: **IoU** y **Dice** (cuánto se solapa la máscara predicha con la real).
"""),
    code('''
ds, n = unet.dataset_segmentacion("datos_seg/imagenes", "datos_seg/mascaras", lote=8, entrenar=True)
n_val = max(1, int(0.2 * n) // 8)
ds_va, ds_tr = ds.take(n_val), ds.skip(n_val)     # ⚠ al repartir por lotes, agrupá por escena si hay varias fotos de un mismo muro
modelo_seg = unet.construir_unet(congelar_encoder=True)
modelo_seg.fit(ds_tr, validation_data=ds_va, epochs=30)
modelo_seg.save("models/unet.keras")
'''),
    md("""
## Paso 2 · Medir el ancho: clásico vs. U-Net
Se mide con **los mismos** pasos (esqueleto + distancia al borde): solo cambia cómo se obtiene la máscara. Se compara
contra el ancho real si lo conocés (con una regla en la foto).

**Escala:** `mm_por_px` desde un objeto de tamaño conocido **en el mismo plano que la grieta**. Para fotos en ángulo,
`rectificar_con_referencia` (4 esquinas de una tarjeta u hoja carta) corrige la perspectiva **y** da la escala.
"""),
    code('''
foto = cv2.cvtColor(cv2.imread("data/campo/ejemplo.jpg"), cv2.COLOR_BGR2RGB)
mascara_clasica = ancho.segmentar_clasico(foto)
mascara_unet = unet.segmentar_con_unet(modelo_seg, foto)
MM_POR_PX = ancho.mm_por_px(85.6, 428)      # ej.: tarjeta de 85,6 mm que mide 428 px en la foto
for nombre, mascara in (("clásica", mascara_clasica), ("U-Net", mascara_unet)):
    print(nombre, ancho.medir_ancho(mascara, MM_POR_PX))
cv2.imwrite("results/medicion.png", ancho.dibujar_medicion(cv2.cvtColor(foto, cv2.COLOR_RGB2BGR), mascara_unet))
'''),
]

import sys
sys.path.insert(0, str(Path(__file__).parent))
import _extras
comp_nb = _extras.extender(md, code, PREPARACION, rob_nb)
comp6 = _extras.mejora(md, code, PREPARACION)

for nombre, celdas in (("01_EDA.ipynb", eda_nb), ("02_Entrenamiento.ipynb", ent_nb),
                       ("03_Robustez_y_falso_positivo.ipynb", rob_nb), ("04_Segmentacion_y_ancho.ipynb", seg_nb),
                       ("05_Complejidad_y_despliegue.ipynb", comp_nb),
                       ("06_Mejora_con_fotos_de_celular.ipynb", comp6)):
    guardar(Path(__file__).parent / nombre, celdas)
    print("generado", nombre, f"({len(celdas)} celdas)")
