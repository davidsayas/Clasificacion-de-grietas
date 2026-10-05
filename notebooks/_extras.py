"""Celdas agregadas para cubrir el documento del reto: F1 y matriz de confusión, fotos propias,
complejidad (15 % de la rúbrica), comparación entre versiones, informe y verificación."""


def _reemplazar_en_celdas(celdas, viejo, nuevo):
    for c in celdas:
        texto = "".join(c["source"])
        if viejo in texto:
            c["source"] = texto.replace(viejo, nuevo).splitlines(True)


def _posicion(celdas, fragmento):
    for i, c in enumerate(celdas):
        if fragmento in "".join(c["source"]):
            return i
    raise ValueError(f"no se encontró la celda con: {fragmento}")


def extender(md, code, PREPARACION, rob_nb):
    """Modifica el cuaderno 03 (rob_nb) y devuelve las celdas del cuaderno 05."""

    # ---------- cuaderno 03 ----------
    _reemplazar_en_celdas(rob_nb, "ood, mosaico, predecir", "ood, mosaico, predecir, fotos_propias, inclinacion")
    _reemplazar_en_celdas(rob_nb, '"precision": f["precision"], "FN"', '"precision": f["precision"], "f1": f["f1"], "FN"')
    _reemplazar_en_celdas(rob_nb, '"precision": round(r["precision"], 4), "FN"',
                          '"precision": round(r["precision"], 4), "f1": round(r["f1"], 4), "FN"')

    integridad = [
        md("""
## Paso 0 · Control de integridad: ¿la prueba está limpia?
**Qué hace:** compara la partición de prueba actual con las particiones **con las que este modelo realmente se entrenó**
(las guarda `models/<nombre>/particion_*.csv`). Cuenta cuántas imágenes y cuántos grupos de la prueba el modelo ya vio.

**Por qué:** si el cuaderno 01 se vuelve a ejecutar DESPUÉS de entrenar (porque cambiaron las carpetas de fotos, por ejemplo),
`results/particion_prueba.csv` cambia, pero el modelo sigue siendo el viejo. Parte de la "prueba" nueva podría haber sido
entrenamiento, y el resultado saldría inflado sin ningún aviso.

**Qué hace si encuentra solapamiento:** descarta de la prueba lo que el modelo ya vio y sigue con la parte limpia.
Esa versión se guarda en `results/particion_prueba_verificada.csv`, que es la que usa el cuaderno 05.
"""),
        code('''
tr_modelo = pd.read_csv(f"models/{NOMBRE}/particion_entrenamiento.csv")
va_modelo = pd.read_csv(f"models/{NOMBRE}/particion_validacion.csv")
solapa_img = te["ruta"].isin(set(tr_modelo["ruta"]) | set(va_modelo["ruta"]))
solapa_grp = te["grupo"].isin(set(tr_modelo["grupo"]) | set(va_modelo["grupo"]))
print(f"Prueba actual: {len(te):,} imágenes · {te['grupo'].nunique()} grupos")
print(f"  imágenes que el modelo YA VIO al entrenar o validar: {int(solapa_img.sum()):,}")
print(f"  imágenes cuyo GRUPO ya vio:                          {int(solapa_grp.sum()):,}")
if solapa_img.any() or solapa_grp.any():
    te = te[~solapa_img & ~solapa_grp].reset_index(drop=True)
    print(f"\\n⚠ LA PRUEBA ESTABA CONTAMINADA. Se descartó lo ya visto. Prueba limpia: {len(te):,} imágenes.")
    print("  No uses los resultados de antes: repetí el Paso 1 con esta prueba limpia.")
else:
    print("\\n✓ La prueba está limpia: ninguna imagen ni grupo coincide con el entrenamiento del modelo.")
te.to_csv("results/particion_prueba_verificada.csv", index=False)
print("Composición de la prueba verificada:", te["etiqueta"].value_counts().to_dict(), "| fuentes:", te["fuente"].value_counts().to_dict())
'''),
    ]
    _i0 = _posicion(rob_nb, "## Paso 1 · Resultado en PRUEBA")
    rob_nb[_i0:_i0] = integridad

    matriz = code('''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.imshow(r["matriz"], cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, r["matriz"][i][j], ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1], ["sin grieta", "con grieta"]); ax.set_yticks([0, 1], ["sin grieta", "con grieta"])
ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.set_title(f"Matriz de confusión (prueba, n={len(te):,})")
fig.tight_layout(); fig.savefig("results/matriz_confusion.png", dpi=150)
print("F1 =", round(r["f1"], 4), "| los falsos negativos (abajo a la izquierda) son las grietas NO detectadas")
''')
    i = _posicion(rob_nb, "## Paso 2 · Prueba del atajo del brillo")
    rob_nb.insert(i, matriz)

    fotos = [
        md("""
## Paso 6b · Fotos propias (lo pide el documento del reto, dentro del 30 % del desempeño)
**Qué hace:** analiza con el sistema completo cada foto de `data/fotos_propias/` y mide el desempeño de dos formas:

* **Forzada:** el sistema siempre responde. Es la medición clásica (exactitud, precisión, recall, F1).
* **Con abstención:** puede responder INDETERMINADO. Se reporta la **cobertura** y, sobre todo, cuántas **grietas reales
  quedaron sin diagnóstico**: un INDETERMINADO sobre una grieta real es un dato de seguridad que no se debe esconder.

⚠ **Decisión tomada con evidencia (Paso 6d):** este paso usa el sistema **SIN** la puerta de concreto. La puerta, calibrada solo con
Surface Crack, descartó las 3 grietas que la red sí detecta (0 de 13 con puerta, 3 de 13 sin ella).

**Preparación:** `data/fotos_propias/` con las fotos y `etiquetas.csv` (columnas `archivo,etiqueta`; 1 = con grieta).
Si sacaste la foto con una tarjeta o regla de referencia, anotá su escala en `parametros` (mm por píxel) para obtener el
ancho en mm. El giro del celular (`roll_deg`) lo da el sensor de gravedad; sin él, la inclinación no se usa.
"""),
        code('''
sistema = predecir.Sistema(predecir_fn)   # SIN puerta de concreto: calibrada solo con Surface Crack, descartaba las grietas de celular (Paso 6d)
parametros = {   # opcional, por foto: {"foto03.jpg": {"mm_por_px": 0.21, "roll_deg": 3.0, "pitch_deg": 0.0, "elemento": "columna"}}
}
tabla_fp = fotos_propias.analizar_carpeta(sistema, "data/fotos_propias", etiquetas="data/fotos_propias/etiquetas.csv",
                                          parametros=parametros, salida_csv="results/fotos_propias_analisis.csv",
                                          escalas=(1.0, 0.5))
display(tabla_fp)
met = fotos_propias.metricas_fotos_propias(tabla_fp)
Path("results/metricas_fotos_propias.json").write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(met, indent=2, ensure_ascii=False))
'''),
    ]
    fotos += [
        md("""
## Paso 6c · Paredes VISTAS contra paredes NUEVAS (¿memoriza o no generaliza?)
**La pregunta:** ¿el modelo funciona igual con fotos que **vio al entrenar** (tus escenas de `dificiles`) que con fotos **nuevas** (`fotos_propias`)?
Se separa por etiqueta, porque `dificiles` puede tener fotos con y sin grieta, y se usan las particiones con las que el modelo REALMENTE entrenó.

| Vistas | Nuevas | Lectura |
|---|---|---|
| acierta en las vistas (0 % sin grieta, ~100 % con grieta) | acierta en las nuevas | generaliza bien |
| acierta en las vistas | **falla en las nuevas** | **memoriza** y no generaliza |
| muchos | muchos | no aprendió a distinguir pared lisa de grieta (los datos no alcanzaron) |
| muchos | pocos | raro: revisá que las carpetas no estén mezcladas |

Se usa la **probabilidad cruda del modelo** en las dos columnas (sin la puerta), para que la comparación sea justa.
Con pocas fotos, mirá el intervalo de confianza: un solo error mueve mucho el porcentaje.
"""),
        code('''
tr_m = pd.read_csv(f"models/{NOMBRE}/particion_entrenamiento.csv")      # lo que ESTE modelo realmente vio al entrenar
print("Lo que el modelo vio al entrenar (fuente x etiqueta):")
print(tr_m.groupby(["fuente", "etiqueta"]).size().unstack(fill_value=0).rename(columns={0: "sin_grieta", 1: "con_grieta"}))
vistas = tr_m[tr_m.fuente == "negativos_dificiles"]
fp_tabla = pd.read_csv("results/fotos_propias_analisis.csv").dropna(subset=["etiqueta"])
conjuntos = [("VISTAS sin grieta (dificiles)", vistas[vistas.etiqueta == 0]["ruta"].tolist(), "deberia dar 0 %"),
             ("VISTAS con grieta (dificiles)", vistas[vistas.etiqueta == 1]["ruta"].tolist(), "deberia dar 100 %"),
             ("NUEVAS sin grieta (fotos_propias)", [f"data/fotos_propias/{a}" for a in fp_tabla[fp_tabla.etiqueta == 0]["archivo"]], "deberia dar 0 %"),
             ("NUEVAS con grieta (fotos_propias)", [f"data/fotos_propias/{a}" for a in fp_tabla[fp_tabla.etiqueta == 1]["archivo"]], "deberia dar 100 %")]
filas = []
for nombre, rutas, ideal in conjuntos:
    if not rutas:
        continue
    p = entrenar.predecir_rutas(predecir_fn, rutas)
    k, n = int((p >= config.UMBRAL_DECISION).sum()), len(p)
    lo, hi = evaluar.intervalo_wilson(k, n)
    filas.append({"grupo": nombre, "fotos": n, "dice_con_grieta": k, "%": round(100 * k / n, 1),
                  "IC95_%": f"{100*lo:.0f}-{100*hi:.0f}", "ideal": ideal})
display(pd.DataFrame(filas))
problemas = sorted(Path("data/problemas").glob("*.jp*g")) + sorted(Path("data/problemas").glob("*.png"))
if problemas:
    pp = predecir_fn(entrenar.cargar_imagenes([str(x) for x in problemas]))
    for x, v in zip(problemas, pp):
        print(f"{x.name}: p(grieta) = {v:.2f}")
'''),
    ]
    fotos += [
        md("""
## Paso 6d · Diagnóstico: ¿falla la RED o falla el SISTEMA?
Si el Paso 6b detecta pocas grietas en tus fotos, hay dos sospechosos distintos y no se arreglan igual:

* **La red** no reconoce grietas de fotos de celular (dominio distinto al de Surface Crack).
* **El sistema** (mosaico, escala o puerta de concreto) arruina una red que sí funcionaría.

Esta celda prueba **cuatro formas** de preguntarle a la misma red, sobre tus fotos con etiqueta:

| Columna | Qué hace |
|---|---|
| `p_entera` | achica TODA la foto a 224×224 y pregunta una vez |
| `p_esc_1.0` | corta la foto en teselas a **tamaño original**, SIN puerta, y toma la más sospechosa |
| `p_esc_0.75` | lo mismo a tres cuartos de tamaño |
| `p_esc_0.5` | lo mismo a mitad de tamaño |
| `p_esc_0.25` | lo mismo a un cuarto de tamaño (la grieta se ve más fina) |
| `p_sistema` | lo que dijo el sistema completo en el Paso 6b (con puerta) |

**AUC:** mide si el método *ordena bien* las fotos (las con grieta con puntaje más alto que las sanas), sin importar el umbral. 0,5 = azar; 1,0 = perfecto.
Si el AUC es alto pero detecta pocas con el umbral 0,5, la red "ve" algo pero sus probabilidades están mal calibradas para este tipo de foto.

**Cómo leerla:** si alguna columna detecta muchas de tus grietas sin disparar falsos positivos, la red SÍ sirve y el problema
es de escala o del sistema. Si ninguna las detecta, el problema es la red y los datos.

No hace falta repetir el Paso 6b: lee `results/fotos_propias_analisis.csv`. Tarda unos minutos en CPU (corta cada foto en cientos de teselas).
"""),
        code('''
from PIL import Image
from sklearn.metrics import roc_auc_score
tabla_fp = pd.read_csv("results/fotos_propias_analisis.csv")
sis = tabla_fp.set_index("archivo")
filas = []
for ruta in datos.listar_imagenes("data/fotos_propias"):
    if ruta.name not in sis.index or pd.isna(sis.loc[ruta.name, "etiqueta"]):
        continue
    foto = np.asarray(Image.open(ruta).convert("RGB"))
    entera = np.asarray(Image.fromarray(foto).resize((config.IMG, config.IMG)), dtype=np.float32)[None]
    fila = {"archivo": ruta.name, "etiqueta": int(sis.loc[ruta.name, "etiqueta"]),
            "tamano": f"{foto.shape[1]}x{foto.shape[0]}", "p_entera": round(float(predecir_fn(entera)[0]), 3)}
    for esc in (1.0, 0.75, 0.5, 0.25):
        fila[f"p_esc_{esc}"] = round(float(mosaico.clasificar_mosaico(foto, predecir_fn, escalas=(esc,))["prob_max"]), 3)
    fila["p_sistema"] = round(float(sis.loc[ruta.name, "prob_grieta"]), 3)
    filas.append(fila)
diag = pd.DataFrame(filas).sort_values(["etiqueta", "archivo"]).reset_index(drop=True)
display(diag)
print("\\nCuántas se detectan con cada método (umbral 0,5):")
for col in [c for c in diag.columns if c.startswith("p_")]:
    pos, neg = diag[diag.etiqueta == 1], diag[diag.etiqueta == 0]
    print(f"  {col:11s} grietas detectadas: {(pos[col] >= 0.5).sum():2d}/{len(pos)} · falsos positivos: {(neg[col] >= 0.5).sum():2d}/{len(neg)} · AUC {roc_auc_score(diag.etiqueta, diag[col]):.2f}")
diag.to_csv("results/diagnostico_fotos_propias.csv", index=False)
resumen = evaluar.resumen_diagnostico(diag)
print("\\nAUC y separación (EXPLORATORIO: mismas fotos de prueba):")
display(resumen)
resumen.to_csv("results/diagnostico_auc.csv", index=False)
'''),
    ]
    j = _posicion(rob_nb, "## Paso 7 · Fuera de dominio")
    rob_nb[j:j] = fotos

    # ---------- cuaderno 05 ----------
    prep = PREPARACION.replace("from src import config, datos, eda, evaluar",
                               "from src import config, datos, evaluar, entrenar, modelo, complejidad, inclinacion, informe, verificar")
    return [
        md("""
# 05 · Complejidad, despliegue y cierre de entregas

## Por qué existe este cuaderno
El documento del reto reparte la nota así: **desempeño 30 %** · **despliegue 25 %** · **eficiencia y complejidad 15 %** ·
pipeline y análisis 20 % · comunicación 10 %. La **Etapa 4** pide reportar, para cada modelo: **número de parámetros,
tamaño en disco (MB) y tiempo de inferencia por imagen**, y la **Entrega 3** pide **comparar desempeño y complejidad entre versiones**.

## Cómo se mide para que sea honesto
* Se descartan las primeras llamadas (calentamiento) y se reporta **mediana y p95**, no solo el promedio.
* Se mide **una imagen por llamada**: es lo que vive el usuario de la app. En lote es más rápido por imagen, pero no es el caso real.
* **Una CPU de Colab no es un celular.** Estas cifras comparan versiones entre sí. El tiempo en un celular real se mide
  en el celular y se reporta aparte (paso 4).
"""),
        code(prep + '''
from tensorflow import keras
te = pd.read_csv("results/particion_prueba_verificada.csv" if Path("results/particion_prueba_verificada.csv").exists() else "results/particion_prueba.csv")
'''),
        md("""
## Paso 1 · Tres variantes TFLite por modelo
**Qué hace:** exporta cada modelo en tres formatos y mide tamaño, velocidad y **fidelidad** (si la conversión cambió las respuestas).

| Variante | Qué hace | Esperable |
|---|---|---|
| float32 | sin comprimir | referencia |
| float16 | pesos en 16 bits | ~2× más chico, casi sin pérdida |
| dinámico | pesos en 8 bits | ~4× más chico; **verificar** que no cambie respuestas |

**Qué decidir:** la variante más chica y rápida cuya fidelidad sea de **0 cambios de clase**.
"""),
        code('''
MODELOS = {n: a for n, a in {"v2_mobilenet": "mobilenetv2", "v2_efficientnet": "efficientnetb0"}.items() if Path(f"models/{n}/modelo.keras").exists()}
muestra = entrenar.cargar_imagenes(te.sample(min(300, len(te)), random_state=1)["ruta"].tolist())
filas, fidelidad = [], {}
for nombre, arq in MODELOS.items():
    m = keras.models.load_model(f"models/{nombre}/modelo.keras")
    inf = modelo.modelo_inferencia(m, arq)
    variantes = modelo.exportar_variantes(m, f"models/{nombre}", arq)
    for var, (ruta, _) in variantes.items():
        fila = complejidad.fila_complejidad(f"{nombre}/{var}", modelo_keras=m, modelo_inf=inf,
                                            ruta_keras=f"models/{nombre}/modelo.keras", ruta_tflite=ruta)
        fid = complejidad.fidelidad_tflite(entrenar.predictor(inf), complejidad.tflite_predictor(ruta), muestra)
        fila["cambian_de_clase_vs_keras"] = fid["cambian_de_clase"]
        fila["dif_max_prob"] = round(fid["dif_max"], 4)
        filas.append(fila)
        fidelidad[f"{nombre}/{var}"] = fid
tabla_comp = complejidad.tabla_complejidad(filas)
display(tabla_comp)
tabla_comp.to_csv("results/complejidad.csv", index=False)
Path("results/fidelidad_tflite.json").write_text(json.dumps(fidelidad, indent=2), encoding="utf-8")
'''),
        md("""
## Paso 2 · Comparación entre versiones (Entrega 3)
Une, por versión: complejidad y desempeño en **tres conjuntos**: la prueba sin fuga, la fuente fuera de dominio y las fotos propias.

⚠ **La v1 (modelo anterior) NO se puede evaluar en la prueba nueva:** se entrenó con una división imagen por imagen, así que
vio muchas de esas imágenes. Compararla ahí sería injusto. Se compara **solo** en fuera de dominio y fotos propias, que ninguna
versión vio al entrenar. Si no tenés `models/mobilenet_100ep.tflite`, la fila de v1 se omite.
"""),
        code('''
fuera = pd.read_csv("results/fuera_de_dominio.csv") if Path("results/fuera_de_dominio.csv").exists() else pd.DataFrame()
fp = pd.read_csv("results/fotos_propias_analisis.csv") if Path("results/fotos_propias_analisis.csv").exists() else pd.DataFrame()
fp_et = fp.dropna(subset=["etiqueta"]) if len(fp) else fp

def en_conjunto(predictor, df, col_ruta="ruta", carpeta=None, n=1500):
    if len(df) == 0:
        return {}
    df = df.sample(min(n, len(df)), random_state=1)
    rutas = [f"{carpeta}/{a}" for a in df[col_ruta]] if carpeta else df[col_ruta].tolist()
    return evaluar.metricas(df["etiqueta"].to_numpy(), entrenar.predecir_rutas(predictor, rutas))

lb = pd.read_csv("results/linea_base.csv")
mejor_lb = lb.sort_values("recall", ascending=False).iloc[0]
versiones = [{"version": "línea base: " + mejor_lb["metodo"], "exactitud_prueba": mejor_lb["exactitud"], "recall_prueba": mejor_lb["recall"]}]

candidatos = []
if Path("models/mobilenet_100ep.tflite").exists():
    candidatos.append(("v1 MobileNetV2 (división con fuga)", "models/mobilenet_100ep.tflite", None))
for nombre, arq in MODELOS.items():
    candidatos.append((f"v2 {arq}", f"models/{nombre}/modelo_float16.tflite", nombre))

for etiqueta, ruta_tflite, nombre in candidatos:
    pred = complejidad.tflite_predictor(ruta_tflite)
    fila = {"version": etiqueta}
    fila.update({k: v for k, v in complejidad.fila_complejidad(etiqueta, ruta_tflite=ruta_tflite, n=50).items() if k != "modelo"})
    if nombre:   # solo las v2 se evalúan en la prueba sin fuga
        r = en_conjunto(pred, te)
        fila.update({"exactitud_prueba": round(r["exactitud"], 4), "recall_prueba": round(r["recall"], 4)})
        fila["parametros"] = complejidad.contar_parametros(keras.models.load_model(f"models/{nombre}/modelo.keras"))["parametros"]
    ro = en_conjunto(pred, fuera)
    if ro:
        fila.update({"exactitud_fuera_dominio": round(ro["exactitud"], 4), "recall_fuera_dominio": round(ro["recall"], 4)})
    rf = en_conjunto(pred, fp_et, col_ruta="archivo", carpeta="data/fotos_propias")
    if rf:
        fila.update({"exactitud_fotos_propias": round(rf["exactitud"], 4), "recall_fotos_propias": round(rf["recall"], 4)})
    versiones.append(fila)
comparacion = pd.DataFrame(versiones)
display(comparacion)
comparacion.to_csv("results/comparacion_versiones.csv", index=False)
'''),
        md("""
## Paso 3 · Inclinación sobre fotos propias (Entrega 1)
**Qué hace:** estima el desaplome de cada foto con Canny + Hough. Si diste el giro del celular (`roll`) lo descuenta; sin él, la medición
queda como *ángulo en la imagen* y no como inclinación real del muro.

**Cómo verificarlo a ojo:** para un muro que sabés a plomo, el ángulo corregido debe ser ≈ 0°.
"""),
        code('''
import cv2
roll = {}   # {"foto03.jpg": 3.0}  giro del celular (°) por foto, del sensor de gravedad
filas = []
fotos = sorted(Path("data/fotos_propias").glob("*.jp*g")) + sorted(Path("data/fotos_propias").glob("*.png"))
for ruta in fotos:
    est = inclinacion.estimar_inclinacion(cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE))
    fila = {"archivo": ruta.name, "angulo_imagen_deg": None, "n_lineas": 0, "inclinacion_real_deg": None, "roll_deg": roll.get(ruta.name)}
    if est:
        fila.update({"angulo_imagen_deg": round(est["angulo_imagen_deg"], 2), "n_lineas": est["n_lineas"]})
        if ruta.name in roll:
            fila["inclinacion_real_deg"] = round(inclinacion.corregir_con_gravedad(est["angulo_imagen_deg"], roll[ruta.name])["inclinacion_real_deg"], 2)
    filas.append(fila)
incl = pd.DataFrame(filas)
display(incl)
incl.to_csv("results/inclinacion_fotos_propias.csv", index=False)
'''),
        md("""
## Paso 4 · Medir en el celular (lo que la CPU de Colab no puede decir)
Para el criterio de eficiencia conviene un número **del celular**. Dos caminos:
1. En la app Android, medir con `System.nanoTime()` alrededor de `interpreter.run(...)`, descartando las primeras 10 llamadas y promediando 100.
2. Con la herramienta *TFLite benchmark model* sobre el `.tflite` en el dispositivo.

Reportá el **modelo del celular**, la **cantidad de hilos** y si usó GPU/NNAPI. Anotalo en `results/tiempo_celular.txt`.
"""),
        md("""
## Paso 5 · Cerrar: informe y estado de las entregas
* `informe.main()` escribe `results/secciones_informe_v2.md` **desde los CSV reales**: tablas de resultados, complejidad, versiones y fotos
  propias, y las reglas EXACTAS del motor de riesgo. Se copia a `INFORME_TECNICO.md` (secciones 6, 8, 9 y 10).
* `verificar.main()` marca cada requisito del documento del reto **según la evidencia que exista**. Lo que no tiene evidencia queda en ❌
  y dice qué falta.
"""),
        code('''
informe.main(".")
estado = verificar.main(".")
'''),
    ]


def mejora(md, code, PREPARACION):
    """Cuaderno 06: ajustar el modelo con fotos de celular y medir sobre las fotos de prueba (que no se tocan)."""
    prep = PREPARACION.replace("from src import config, datos, eda, evaluar",
                               "from src import config, datos, evaluar, entrenar, modelo, mejora, mosaico")
    return [
        md("""
# 06 · Mejorar el modelo con fotos de celular

## Por qué este cuaderno
El análisis mostró que el modelo **acierta el 100 % de las fotos propias con las que entrenó y detecta 2 de 13 de las nuevas**:
memoriza esas paredes. Lo que le falta no es más épocas ni otra arquitectura, son **muchas paredes distintas** en el formato en
que se va a usar (fotos de celular). Este cuaderno retoma tu modelo y lo ajusta con eso.

## Qué tenés que haber preparado
```
data/campo/con/<pared_1>/ foto.jpg ...    ← paredes CON grieta, una subcarpeta por pared
data/campo/sin/<pared_1>/ foto.jpg ...    ← paredes SIN grieta, una subcarpeta por pared
```
* **Mínimo 4 paredes por clase; lo recomendable, 10 o más**, con 5 a 8 fotos de cada una (distinto ángulo y distancia).
* Paredes **distintas** a las de `fotos_propias` y `problemas`. Esas dos carpetas son la PRUEBA y no se tocan.

## Cómo se evita engañarse
Las paredes nuevas se reparten **por pared entera**: algunas quedan para validar mientras entrena. Así hay una señal honesta de si
generaliza a paredes que no vio, y el entrenamiento se detiene cuando esa señal deja de mejorar.
"""),
        code(prep),
        md("""
## Paso 1 · Armar los datos
Imprime cuántas fotos y paredes hay en cada parte. Si faltan paredes, el mensaje dice cuántas hacen falta.
"""),
        code('''
tr, va, resumen = mejora.preparar_mejora(PROYECTO, modelo_base="v2_mobilenet")
display(resumen)
print("Imágenes por época:", len(tr), "| validación (paredes nuevas):", len(va))
'''),
        md("""
## Paso 2 · Ajustar el modelo
Retoma `v2_mobilenet`, libera las últimas 60 capas con tasa baja (1e-4) y entrena hasta 10 épocas, deteniéndose cuando la
validación en paredes nuevas deja de mejorar. **Necesita GPU** (*Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*).
Si no hay GPU disponible, tarda mucho: no lo corras en CPU.
"""),
        code('''
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices("GPU"))
meta = entrenar.ajustar_dominio("v2_mobilenet", "v3_celular", tr, va, epocas=10)
print(meta)
'''),
        md("""
## Paso 3 · Medirlo sobre tus fotos de prueba (que el modelo NUNCA vio)
Compara el modelo anterior (v2) con el ajustado (v3) sobre tus 29 fotos, con las mismas cuatro formas de preguntar. Lo que
importa: **¿cuántas de las 13 grietas detecta, con cuántas falsas alarmas?**
"""),
        code('''
from tensorflow import keras
from PIL import Image
fp = pd.read_csv("results/fotos_propias_analisis.csv").dropna(subset=["etiqueta"])
filas_resumen = []
for nombre in ("v2_mobilenet", "v3_celular"):
    inf = modelo.modelo_inferencia(keras.models.load_model(f"models/{nombre}/modelo.keras"), "mobilenetv2")
    pred = entrenar.predictor(inf)
    filas = []
    for _, f in fp.iterrows():
        foto = np.asarray(Image.open(f"data/fotos_propias/{f.archivo}").convert("RGB"))
        entera = np.asarray(Image.fromarray(foto).resize((config.IMG, config.IMG)), dtype=np.float32)[None]
        fila = {"archivo": f.archivo, "etiqueta": int(f.etiqueta), "p_entera": float(pred(entera)[0])}
        for esc in (0.5, 0.25):
            fila[f"p_esc_{esc}"] = float(mosaico.clasificar_mosaico(foto, pred, escalas=(esc,))["prob_max"])
        filas.append(fila)
    diag = pd.DataFrame(filas)
    diag.to_csv(f"results/diagnostico_{nombre}.csv", index=False)
    r = evaluar.resumen_diagnostico(diag)
    r.insert(0, "modelo", nombre)
    filas_resumen.append(r)
comparacion = pd.concat(filas_resumen)
display(comparacion)
comparacion.to_csv("results/comparacion_v2_v3_fotos_propias.csv", index=False)
'''),
        md("""
## Paso 4 · ¿Olvidó lo que sabía? (Surface Crack)
Si el ajuste destruyó lo aprendido, la exactitud en Surface Crack cae. Se mide con la validación **original** del modelo.
Una caída pequeña es aceptable si en las fotos de celular se detectan muchas más grietas.
"""),
        code('''
va_sc = pd.read_csv("models/v2_mobilenet/particion_validacion.csv")
va_sc = pd.concat([va_sc[va_sc.etiqueta == e].sample(500, random_state=1) for e in (0, 1)])
for nombre in ("v2_mobilenet", "v3_celular"):
    inf = modelo.modelo_inferencia(keras.models.load_model(f"models/{nombre}/modelo.keras"), "mobilenetv2")
    p = entrenar.predecir_rutas(entrenar.predictor(inf), va_sc["ruta"].tolist())
    r = evaluar.metricas(va_sc["etiqueta"].to_numpy(), p)
    print(f"{nombre}: exactitud {r['exactitud']:.4f} · recall {r['recall']:.4f} · falsos neg. {r['fn']} · falsos pos. {r['fp']}")
'''),
        md("""
## Paso 5 · Exportar para el celular
"""),
        code('''
tam = modelo.exportar_tflite(keras.models.load_model("models/v3_celular/modelo.keras"), "models/v3_celular/modelo.tflite", "mobilenetv2")
print(f"modelo.tflite: {tam/1e6:.2f} MB")
'''),
    ]
