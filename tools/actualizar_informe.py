"""Actualiza INFORME_TECNICO.md (versión 1) a la versión 2, con los resultados REALES del equipo.

Es idempotente: si ya se aplicó, no hace nada. Las reglas de riesgo y los casos de ejemplo se calculan
llamando al código (src/riesgo.py), para que el informe no pueda contradecirlo.
Lo que solo el equipo conoce (nombres, capturas de pantalla, origen de fotos) queda marcado con [COMPLETAR].

Uso:   python tools/actualizar_informe.py
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import riesgo  # noqa: E402

ARCHIVO = RAIZ / "INFORME_TECNICO.md"
MARCA = "Actualización de la versión 2."


def antes_de_seccion(texto: str, titulo: str, bloque: str) -> str:
    """Inserta `bloque` justo antes del separador (---) que precede a la sección `titulo`."""
    patron = f"\n---\n\n{titulo}"
    i = texto.index(patron)
    return texto[:i] + "\n" + bloque.strip("\n") + "\n" + texto[i:]


def casos_de_riesgo() -> str:
    casos = [
        ("Grieta diagonal en columna, 1,5 mm, desaplome 0,2°", dict(prob_grieta=0.97, ancho_mm=1.5, inclinacion_deg=0.2, orientacion="diagonal", elemento="columna")),
        ("Grieta vertical en muro, 0,2 mm, desaplome 0,2°", dict(prob_grieta=0.97, ancho_mm=0.2, inclinacion_deg=0.2, orientacion="vertical", elemento="muro")),
        ("Grieta detectada, sin objeto de referencia (ancho desconocido)", dict(prob_grieta=0.97)),
        ("Sin grieta, pero desaplome de 4°", dict(prob_grieta=0.02, inclinacion_deg=4.0)),
        ("Sin grieta y sin desaplome", dict(prob_grieta=0.02, inclinacion_deg=0.1)),
        ("La imagen no parece concreto", dict(prob_grieta=0.99, es_concreto=False)),
        ("Probabilidad 0,50 (zona de duda)", dict(prob_grieta=0.50)),
    ]
    filas = ["| Caso | Nivel | Puntaje |", "|---|---|---|"]
    for nombre, kw in casos:
        ev = riesgo.evaluar_riesgo(**kw)
        filas.append(f"| {nombre} | **{ev.nivel}** | {'—' if ev.puntaje is None else ev.puntaje} |")
    return "\n".join(filas)


RESUMEN = """**Actualización de la versión 2.** La partición de la versión 1 repartía recortes de las mismas fotografías entre entrenamiento y prueba (con la división imagen por imagen, 913 de 913 grupos aparecían en ambos), por lo que sus cifras no son comparables con las de la versión 2, que divide por grupos y verifica que la prueba no comparte imágenes ni grupos con lo que el modelo vio. Sobre esa prueba limpia (6.052 imágenes de Surface Crack) el modelo alcanza 99,95 % de exactitud, con 2 grietas no detectadas de 2.928. Sin embargo, en 29 fotografías propias de celular (13 con grieta) detecta entre 2 y 3 grietas con el umbral habitual y no da ninguna falsa alarma: el modelo acierta todas las fotos propias con las que entrenó y casi ninguna de las nuevas, lo que indica memorización de pocas fotos y diferencia de dominio. Un componente diseñado para reducir falsas alarmas, la puerta de concreto, empeoró el resultado y se retiró. El mayor avance disponible es ampliar los datos con fotografías de celular de paredes distintas.
"""

PARTICION_V2 = """
### 4.4.1 Partición de la versión 2: por grupos

Los 40.000 recortes de Surface Crack salen de 458 fotografías, cerca de 87 recortes por foto. Una partición aleatoria imagen por imagen reparte recortes del mismo muro entre entrenamiento y prueba, de modo que el modelo se evalúa con "hermanos" de lo que ya vio y el resultado se infla. **Con la división de la versión 1, los 913 grupos del inventario aparecían a la vez en entrenamiento y en prueba.**

La versión 2 divide por **grupos**: todas las imágenes de un mismo grupo caen en la misma partición, y el código verifica automáticamente que ningún grupo ni imagen aparece en dos particiones (`datos.verificar_sin_fuga`). Los grupos son bloques de 44 archivos consecutivos por clase, bajo la hipótesis de que la numeración sigue a la fotografía de origen. La hipótesis se contrastó comparando la similitud de color entre archivos consecutivos (mediana 0,69) y entre pares al azar (mediana 1,36): la razón es **0,50**, es decir, los consecutivos se parecen el doble. Las fracciones son 70 / 15 / 15 con semilla 42.

**Control de integridad de la prueba.** Como `results/particion_prueba.csv` cambia si se repite el análisis exploratorio después de entrenar, antes de evaluar se compara la prueba con las particiones con las que el modelo realmente se entrenó (`models/<modelo>/particion_*.csv`). Resultado: **0 imágenes y 0 grupos en común**. La prueba verificada tiene 6.052 imágenes, todas de Surface Crack (3.124 sin grieta y 2.928 con grieta).

**Límites.** Los grupos aproximan las fotografías de origen, que el dataset no identifica. Además, los bloques se arman por clase, así que recortes de una misma foto con grieta y sin grieta podrían caer en particiones distintas. El resultado sigue siendo optimista respecto de muros nunca vistos; por eso se evalúa además con fotografías propias (sección 10.4). Duplicados: la huella perceptual halla 1,52 % de imágenes casi repetidas en una muestra de 4.000 (cota inferior).
"""

BANNER_6 = """> **Aviso.** Las cifras de las secciones 6.1 a 6.5 son de la **versión 1**, cuya partición tenía fuga (ver 4.4.1); no son comparables con la versión 2. Los resultados de la versión 2 están en la sección 6.6.

"""

RESULTADOS_V2 = """
### 6.6 Resultados de la versión 2 (partición por grupos)

| Conjunto | Imágenes | Exactitud | Recall | Errores |
|---|---|---|---|---|
| Validación (elige la mejor época: algo optimista) | 6.028 | 99,87 % | 99,84 % | 5 grietas no detectadas (0,16 %; IC95 0,07 a 0,38 %) y 3 falsas alarmas |
| **Prueba verificada** | 6.052 | **99,95 %** | **99,93 %** | 2 grietas no detectadas (0,07 %; IC95 0,02 a 0,25 %) y 1 falsa alarma |

La matriz de confusión de la prueba es `[[3123, 1], [2, 2926]]` (filas: real sin grieta y con grieta; columnas: predicho sin grieta y con grieta); precisión 99,97 %.

Con 3 errores no se puede afirmar que la versión 2 sea mejor ni peor que la 1: los intervalos se solapan. Dos explicaciones son compatibles con estos números y este experimento no las separa: Surface Crack es un conjunto fácil, o los grupos aproximados no separan todos los muros parecidos.

**El brillo no es un atajo.** Se tomaron imágenes de la prueba, se oscurecieron y aclararon hasta ±51 niveles (la diferencia real entre clases es de 18) y se contó cuántas cambiaban de clase: `[COMPLETAR: porcentaje máximo de results/prueba_atajo_brillo.csv]`. El resultado descarta que la falta de variación de brillo global explique el falso positivo de la sección 10.1; no descarta un efecto del brillo local (una mano oscura sobre una pared clara), que esta prueba no cubre. La prueba se validó con dos modelos sintéticos: detecta al que usa el brillo y no acusa al que mira la textura.
"""

SECCION_8 = """**Reglas de la versión 2** (las que implementa `src/riesgo.py`; el informe las toma del código):

{reglas}

**Casos calculados con el código:**

{casos}

"""

NOTA_9 = """*Las mediciones de esta sección corresponden a la arquitectura de la versión 1; MobileNetV2 con la misma cabeza tiene los mismos parámetros en la versión 2. Los tiempos de la versión 2, las variantes TFLite (float32, float16 y dinámico) y su fidelidad respecto de Keras se generan en el cuaderno 05 (`results/complejidad.csv`). Los tiempos se miden con una imagen por llamada, descartando el calentamiento; una CPU de Colab no es un celular.*

"""

SECCION_10_4 = """
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

Las filas "vistas" son 44 y 43 fotografías únicas repetidas 5 veces por el sobremuestreo, por lo que sus intervalos no son válidos. El modelo acierta todo lo que vio y casi nada de lo nuevo: la lectura más fuerte es **memorización de pocas fotos**, sumada a una **diferencia de dominio** entre el concreto de Surface Crack y las paredes fotografiadas con celular. `[COMPLETAR: origen de las 43 fotografías con grieta de la carpeta dificiles: propias, internet y bajo qué licencia]`

**El falso positivo de la sección 10.1.** La pared con la mano, que no está en ningún conjunto de entrenamiento, da ahora p = 0,02. **No se cuenta como corrección**: el modelo también dice "sin grieta" en la mayoría de las grietas de celular, y un "no" generalizado no demuestra que distinga una pared lisa de una grieta. La carpeta `dificiles` tiene solo 3 escenas, todas en entrenamiento, así que el falso positivo no pudo medirse en escenas no vistas.

**Valoración.** El sistema final no usa la puerta de concreto y analiza las fotos a 1.280 px de lado largo y escala 0,5. La aplicación ofrece tres niveles de sensibilidad (umbral 0,50, 0,10 y 0,02); los dos últimos se eligieron mirando las mismas fotos de prueba y son exploratorios. Un aviso permanente recuerda que no detectar una grieta no significa que no exista.
"""

ETICA = """
**Abstención y grietas sin diagnóstico.** Cuando la imagen no parece concreto o el modelo duda (probabilidad entre 0,20 y 0,80 con el umbral estándar), el sistema responde INDETERMINADO en lugar de afirmar. Reduce las falsas alarmas pero introduce un riesgo propio: una grieta real sin diagnóstico. Por eso se reporta el número de grietas reales sin diagnóstico en las fotos propias, y la recomendación asociada pide repetir la foto o una revisión visual, nunca "sin riesgo".

**Transparencia sobre el desempeño en campo.** El modelo alcanza 99,95 % en Surface Crack pero detecta 2 a 3 de 13 grietas en fotos de celular. Mostrar solo la primera cifra sería engañoso para un usuario que confía en la herramienta. Por eso la aplicación muestra, sobre cada resultado, el desempeño real medido por el equipo con fotos propias y la advertencia de que puede dejar pasar grietas.

**Datos personales en las fotografías.** Las fotos pueden mostrar rostros, placas, números de casa o la ubicación en los metadatos. Antes de publicarlas se eliminan los metadatos EXIF y se evitan personas y direcciones identificables.
"""

CONCLUSIONES = """
7. **Se encontró y corrigió una fuga de información.** Con la división de la versión 1 los 913 grupos aparecían en entrenamiento y prueba. La versión 2 divide por grupos y verifica que la prueba no comparte imágenes ni grupos con el entrenamiento.

8. **En Surface Crack la exactitud se mantiene:** 99,95 % en una prueba verificada de 6.052 imágenes, con 2 grietas no detectadas de 2.928. Con 3 errores no es posible afirmar una mejora o un empeoramiento respecto de la versión 1.

9. **El modelo no transfiere a fotografías de celular.** Detecta 2 a 3 de 13 grietas propias sin falsas alarmas, aunque acierta el 100 % de las 87 fotos propias con las que entrenó. La evidencia indica memorización de pocas fotos y diferencia de dominio, no un problema de brillo ni de escala.

10. **Un componente que parecía una mejora la empeoró.** La puerta de concreto, calibrada solo con Surface Crack, descartó las tres grietas que la red sí detectaba y se retiró. Medir con fotos propias fue lo que permitió verlo.

11. **El mayor avance disponible es de datos, no de modelo:** fotografías de celular de muchas paredes distintas, con y sin grieta.
"""

FUTURO = """| Alta | Ajustar el modelo con fotografías de celular de al menos 10 paredes con grieta y 10 sin grieta (cuaderno 06), sin tocar las fotos de prueba |
| Alta | Calibrar el umbral de decisión sobre un conjunto de calibración distinto del de prueba |
| Media | Calibrar la puerta de concreto con fotos de celular, o descartarla |
| Media | Incorporar SDNET2018 y reservar parte como prueba fuera de dominio |

*Ya realizado en la versión 2:* prueba del atajo del brillo (el modelo no lo usa), estimación del ancho por esqueleto y transformada de distancia, rectificación de perspectiva, orientación de la grieta, y partición por grupos.
"""

ANEXO_B = """- **Versión 2.** Partición por grupos con semilla 42, guardada en `results/particion_*.csv`; entrenamiento y evaluación leen las mismas particiones. El historial se guarda con un archivo por fase y `append=False`, lo que corrige el defecto de la corrida C. `models/<nombre>/meta.json` registra hiperparámetros y versiones de TensorFlow y Keras. `python tests/run_all.py` ejecuta las pruebas del repositorio y `python -m src.verificar` marca cada requisito de las entregas según la evidencia existente.
- **Código de la versión 2:** `src/datos.py`, `src/eda.py`, `src/linea_base.py`, `src/modelo.py`, `src/entrenar.py`, `src/evaluar.py`, `src/mosaico.py`, `src/ancho.py`, `src/inclinacion.py`, `src/riesgo.py`, `src/predecir.py`, `src/complejidad.py`, `src/fotos_propias.py`, `src/app_core.py` y `app/streamlit_app.py`.
"""


def aplicar() -> None:
    t = ARCHIVO.read_text(encoding="utf-8")
    if MARCA in t:
        print("El informe ya estaba actualizado: no se cambia nada.")
        return

    t = t.replace("**Entrega:** 1 — Formulación y primera implementación funcional",
                  "**Entrega:** 1 a 3 — versión 2 (corrección de la fuga, fotografías propias y análisis de generalización)")
    t = t.replace("`[COMPLETAR: URL de GitHub]`", "https://github.com/davidsayas/Clasificacion-de-grietas")
    t = t.replace("`[COMPLETAR desde models/baseline_metricas.json]`",
                  "`[COMPLETAR: pegar la tabla de results/linea_base.csv, cuaderno 01 paso 7]`")
    t = t.replace("**Palabras clave:**", RESUMEN + "\n**Palabras clave:**", 1)
    t = antes_de_seccion(t, "## 5. Módulo A", PARTICION_V2)
    t = t.replace("## 6. Resultados\n\n", "## 6. Resultados\n\n" + BANNER_6, 1)
    t = antes_de_seccion(t, "## 7. Módulo B", RESULTADOS_V2)

    ini = t.index("**Comportamiento observado:**")
    fin = t.index("El esquema está deliberadamente sesgado")
    t = t[:ini] + SECCION_8.format(reglas=riesgo.reglas_markdown(), casos=casos_de_riesgo()) + t[fin:]
    t = t.replace("`[COMPLETAR: describir las reglas exactas implementadas en la aplicación]`",
                  "Los umbrales son orientativos y no están validados contra dictámenes de ingenieros (limitación 8).")

    t = t.replace("### 9.1 Mediciones\n\n", "### 9.1 Mediciones\n\n" + NOTA_9, 1)

    lineas = t.split("\n")
    for i, l in enumerate(lineas):
        if l.startswith("1. **Fuga de información por recortes.**"):
            lineas[i] = l + " **Actualización v2:** corregida con la partición por grupos (sección 4.4.1); los grupos siguen siendo una aproximación de las fotografías de origen."
        elif l.startswith("5. **Sin estimación del ancho.**"):
            lineas[i] = l + " **Actualización v2:** con un objeto de tamaño conocido en el mismo plano que la grieta, el ancho se mide con esqueleto y transformada de distancia (sesgo de −0,02 px con anchos exactos, peor caso 1 px); sin referencia solo se da en píxeles."
        elif l.startswith("6. **La inclinación depende de la perspectiva.**"):
            lineas[i] = l + " **Actualización v2:** el giro del celular se descuenta con el sensor de gravedad; si el celular apunta hacia arriba o abajo (más de 5°) la medición se marca como no confiable."
    t = "\n".join(lineas)

    t = antes_de_seccion(t, "## 11. Consideraciones éticas", SECCION_10_4)
    t = antes_de_seccion(t, "## 12. Conclusiones", ETICA)
    t = t.replace("permitió alcanzar 99,93 %** de exactitud y recall en prueba",
                  "permitió alcanzar 99,93 %** de exactitud y recall en prueba (versión 1, con partición con fuga; en la versión 2, sin fuga, 99,95 % en una prueba verificada)", 1)
    t = antes_de_seccion(t, "## 13. Trabajo futuro", CONCLUSIONES)
    t = t.replace("| Baja | Segmentación para estimar el ancho de la grieta |",
                  "| Baja | Segmentación para estimar el ancho de la grieta (U-Net) |\n" + FUTURO.rstrip("\n"), 1)
    i = t.index("\n## Anexo C")
    t = t[:i] + "\n" + ANEXO_B.rstrip("\n") + "\n" + t[i:]

    ARCHIVO.write_text(t, encoding="utf-8")
    print("Informe actualizado.")


if __name__ == "__main__":
    aplicar()
