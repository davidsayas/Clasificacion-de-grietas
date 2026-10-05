"""app_core.py — La lógica de la app, separada de la interfaz para poder PROBARLA sin Streamlit ni TensorFlow.

La interfaz (app/streamlit_app.py) solo dibuja widgets y llama a estas funciones.

DECISIONES DE DISEÑO, con el porqué (todas salen de las pruebas del equipo con fotos propias):
  * SIN puerta de concreto: calibrada solo con Surface Crack, descartó las 3 grietas que la red sí detecta.
  * La foto se reduce a 1280 px de lado largo antes de analizar: las pruebas se hicieron con fotos de 960x1280 / 1280x960.
    Una foto de 4000 px mostraría la grieta a otra escala que la que se probó.
  * Se analiza a escala 0,5 (la de mejor AUC en las pruebas: 0,89, aunque el AUC es exploratorio).
  * Tres niveles de sensibilidad: con el umbral habitual (0,50) la red casi no se activa en fotos de celular, porque sus
    probabilidades quedan muy por debajo; bajar el umbral detecta más grietas a costa de más falsas alarmas. Los valores 0,10
    y 0,02 se eligieron mirando las mismas fotos de prueba: son EXPLORATORIOS, no un resultado validado.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import ancho, config

LADO_LARGO = 1280
ESCALAS = (0.5,)
SIN_BANDA = (0.0, 0.0)           # sin zona de duda: con probabilidades comprimidas, "dudar" taparía las grietas

NIVELES = {
    "Estándar (umbral 0,50)": (0.50, config.BANDA_INCIERTA),
    "Alta sensibilidad (umbral 0,10)": (0.10, SIN_BANDA),
    "Muy alta sensibilidad (umbral 0,02)": (0.02, SIN_BANDA),
}
ELEMENTOS = ("No sé", "muro", "columna", "viga", "placa")

AVISO_FIJO = ("**No detectar una grieta NO significa que no exista.** Esta herramienta es de tamizaje, no sustituye la "
              "inspección de un ingeniero, y puede dejar pasar grietas reales.")


def reducir_lado_largo(foto: np.ndarray, lado: int = LADO_LARGO) -> np.ndarray:
    """Reduce la foto (sin deformarla) para que su lado largo mida `lado` px. Nunca agranda."""
    h, w = foto.shape[:2]
    if max(h, w) <= lado:
        return foto
    f = lado / max(h, w)
    return cv2.resize(foto, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def aviso_de_limitaciones(raiz=".") -> str:
    """Texto fijo de advertencia, con los números REALES de las pruebas del equipo si existen.

    Lee `results/diagnostico_fotos_propias.csv` (cuaderno 03, Paso 6d), columna `p_esc_0.5`: es EXACTAMENTE la configuración
    del sistema de la app (sin puerta, escala 0,5, umbral estándar). Si no existe, usa `metricas_fotos_propias.json`.
    """
    import csv

    base = AVISO_FIJO
    diag = Path(raiz) / "results" / "diagnostico_fotos_propias.csv"
    if diag.exists():
        with open(diag, encoding="utf-8", newline="") as f:
            filas = [r for r in csv.DictReader(f) if r.get("etiqueta") not in (None, "")]
        con = [float(r["p_esc_0.5"]) for r in filas if int(float(r["etiqueta"])) == 1]
        sin = [float(r["p_esc_0.5"]) for r in filas if int(float(r["etiqueta"])) == 0]
        if con and sin:
            tp = sum(p >= config.UMBRAL_DECISION for p in con)
            fp = sum(p >= config.UMBRAL_DECISION for p in sin)
            return (base + f" En las pruebas del equipo con {len(con) + len(sin)} fotos de celular (umbral estándar), el sistema "
                    f"detectó {tp} de {len(con)} grietas y dio {fp} falsas alarmas en {len(sin)} fotos sin grieta.")
    ruta = Path(raiz) / "results" / "metricas_fotos_propias.json"
    if not ruta.exists():
        return base + " Esta carpeta todavía no tiene resultados de pruebas con fotos propias."
    d = json.loads(ruta.read_text(encoding="utf-8"))
    f = d["forzada"]
    return (base + f" En las pruebas del equipo con {d['n']} fotos de celular, el sistema detectó {f['tp']} de "
            f"{d['con_grieta']} grietas y dio {f['fp']} falsas alarmas en {d['sin_grieta']} fotos sin grieta.")


def superponer_bordes(foto: np.ndarray, mapa) -> np.ndarray | None:
    """La foto con los bordes usados para medir la inclinación pintados de VERDE: para comprobar a ojo que se midió la
    columna y no otra cosa."""
    if mapa is None:
        return None
    sal = foto.copy().astype(np.uint8)
    sal[np.asarray(mapa, dtype=bool)] = (0, 255, 0)
    return sal


def descripcion_inclinacion(r: dict) -> str:
    """Una línea que dice qué ángulo se midió, con qué método, qué tan confiable es y si se usó para el riesgo."""
    a = r.get("inclinacion_imagen_deg")
    if not r.get("inclinacion_pedida", True):
        if a is None:
            return "No se pidió medir la inclinación del muro."
        return (f"Borde vertical dominante en la imagen: **{a:.1f}°**. **NO se usó para el riesgo**: la casilla «Medir la inclinación» "
                f"está apagada. Activala solo si la foto es de una columna o un muro, sin manos ni objetos delante.")
    if a is None:
        return "No se encontró un borde vertical dominante para medir la inclinación del muro."
    metodo, conf = r.get("inclinacion_metodo"), r.get("inclinacion_confianza")
    detalle = f"método de {metodo}" + (f", confianza {conf:.2f}" if conf is not None else "")
    texto = f"Inclinación de las líneas verticales en la imagen: **{a:.1f}°** ({detalle})"
    if conf is not None and conf < 0.25:
        texto += ". Medición POCO confiable: revisá que los bordes verdes sean los de la columna"
    if r.get("inclinacion_piel"):
        texto += f". Se ignoró el {100 * r['inclinacion_piel']:.0f} % de la foto por parecer piel (mano o brazo)"
    if r.get("inclinacion_deg") is None:
        texto += ". NO se usa para el riesgo (ángulo chico y falta el giro del celular); si la foto salió derecha, marcá la casilla «celular derecho»"
    elif r.get("inclinacion_verificada") is False:
        texto += ". Se usa SIN verificar (no se conoce el giro del celular): confirmala con un nivel o una plomada, o marcá «celular derecho» si la foto salió derecha"
    return texto


def listar_modelos(raiz=".") -> list[str]:
    """Carpetas de models/ que tienen un modelo.keras o un modelo.tflite."""
    base = Path(raiz) / "models"
    return sorted({p.parent.name for patron in ("*/modelo.keras", "*/modelo.tflite") for p in base.glob(patron)})


def _predictor_keras(carpeta: Path, nombre: str):
    from tensorflow import keras

    from . import entrenar, modelo

    meta = carpeta / "meta.json"
    arq = json.loads(meta.read_text(encoding="utf-8")).get("arquitectura", "mobilenetv2") if meta.exists() else "mobilenetv2"
    m = keras.models.load_model(carpeta / "modelo.keras")
    return entrenar.predictor(modelo.modelo_inferencia(m, arq))


def _predictor_tflite(ruta: Path):
    from . import complejidad

    return complejidad.tflite_predictor(ruta)


AYUDA_VERSIONES = ("Tu versión de TensorFlow/Keras es más vieja que la que guardó el modelo. Probá:  "
                   "py -3.10 -m pip install --upgrade tensorflow keras   "
                   "o copiá `modelo.tflite` a la misma carpeta del modelo: no depende de la versión.")


def cargar_sistema(nombre: str, raiz=".", cargador_keras=None, cargador_tflite=None):
    """Carga el modelo y arma el sistema. Es lo único que necesita TensorFlow.

    Primero intenta con `modelo.keras`. Si Keras no puede leerlo (casi siempre: el modelo se guardó en Colab con un Keras más
    nuevo que el instalado en este computador), usa `modelo.tflite`, que no depende de la versión de Keras.
    Los argumentos `cargador_*` existen para poder probar esta lógica sin TensorFlow.
    """
    from . import predecir

    carpeta = Path(raiz) / "models" / nombre
    keras_ruta, tflite_ruta = carpeta / "modelo.keras", carpeta / "modelo.tflite"
    motivo = ""
    if keras_ruta.exists():
        try:
            return predecir.Sistema((cargador_keras or _predictor_keras)(carpeta, nombre))
        except Exception as e:                                   # versión de Keras incompatible, archivo dañado, etc.
            motivo = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
    if tflite_ruta.exists():
        return predecir.Sistema((cargador_tflite or _predictor_tflite)(tflite_ruta))
    detalle = f" (Keras dijo: {motivo})" if motivo else ""
    raise RuntimeError(f"No pude cargar el modelo '{nombre}'{detalle}. {AYUDA_VERSIONES}")


def analizar(sistema, foto: np.ndarray, nivel: str, elemento: str | None = None, ref_mm: float | None = None,
             ref_px: float | None = None, roll: float | None = None, pitch: float | None = None,
             asumir_derecho: bool = False, medir_inclinacion: bool = False) -> dict:
    """Analiza una foto RGB con la sensibilidad elegida. Devuelve el resultado del sistema más la foto reducida.

    ref_mm / ref_px: largo real de un objeto de referencia (mm) y cuánto mide en la foto ORIGINAL (px). La escala se
    ajusta sola al reducir la foto.
    """
    umbral, banda = NIVELES[nivel]
    sistema.umbral, sistema.banda = umbral, banda
    h0, w0 = foto.shape[:2]
    chica = reducir_lado_largo(foto)
    mm = None
    if ref_mm and ref_px:
        mm = ancho.mm_por_px(ref_mm, ref_px) * (max(h0, w0) / max(chica.shape[:2]))
    elemento = None if elemento in (None, "No sé") else elemento
    nota = None
    if medir_inclinacion and asumir_derecho and roll is None:        # sin sensor: el usuario afirma que sacó la foto derecha, de frente al muro
        roll, pitch = 0.0, 0.0
        nota = "Se asumió que el celular estaba derecho (giro 0°): la inclinación medida se usa como inclinación del muro."
    r = sistema.analizar(chica, mm_por_px=mm, roll_deg=roll, pitch_deg=pitch, escalas=ESCALAS, elemento=elemento,
                         usar_inclinacion=medir_inclinacion)
    if nota:
        r["notas"] = r["notas"] + [nota]
    return {**r, "foto": chica, "umbral": umbral}
