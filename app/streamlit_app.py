"""App: foto -> clase, probabilidad y nivel de riesgo.

Uso:   doble clic en ejecutar_app.bat (Windows) o ./ejecutar_app.sh (Mac/Linux)
       o:   streamlit run app/streamlit_app.py

⚠ La lógica (src/app_core.py y src/predecir.py) está probada con datos sintéticos. Esta interfaz y la carga del modelo con
TensorFlow NO se pudieron ejecutar donde se escribió: la primera corrida en tu computador puede pedir algún ajuste.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import streamlit as st
from PIL import Image

from src import app_core, mosaico

st.set_page_config(page_title="Grietas y riesgo", layout="wide")
st.title("Detección de grietas y nivel de riesgo")
st.warning(app_core.aviso_de_limitaciones(RAIZ))


@st.cache_resource(show_spinner="Cargando el modelo (la primera vez tarda un poco)...")
def cargar(nombre):
    return app_core.cargar_sistema(nombre, RAIZ)


modelos = app_core.listar_modelos(RAIZ)
if not modelos:
    st.error("No encuentro ningún modelo. Copiá la carpeta `models/v2_mobilenet` (con `modelo.keras` y `meta.json`) "
             "desde tu Drive a la carpeta `models` de este proyecto y recargá la página.")
    st.stop()

with st.sidebar:
    st.header("Ajustes")
    nombre = st.selectbox("Modelo", modelos)
    nivel = st.radio("Sensibilidad", list(app_core.NIVELES),
                     help="Con el umbral estándar el modelo casi no se activa en fotos de celular. Subir la sensibilidad "
                          "detecta más grietas pero también da más falsas alarmas. Los valores altos son exploratorios.")
    elemento = st.selectbox("Elemento fotografiado", app_core.ELEMENTOS)

    st.subheader("Escala (opcional)")
    st.caption("Para dar el ancho en milímetros: poné una tarjeta o regla pegada junto a la grieta, en el mismo plano.")
    usar_ref = st.checkbox("Tengo un objeto de referencia", False)
    ref_mm = st.number_input("Largo real del objeto (mm)", 1.0, 1000.0, 85.6) if usar_ref else None
    ref_px = st.number_input("Cuánto mide en la foto (px)", 1.0, 20000.0, 400.0) if usar_ref else None

    st.subheader("Inclinación del muro (opcional)")
    medir = st.checkbox("Medir la inclinación (solo si la foto es de una columna o un muro, sin manos ni objetos delante)", False,
                        help="El programa mide el borde vertical más marcado de la foto, pero no sabe si es una columna o un brazo. "
                             "Por eso solo se usa si vos lo pedís.")
    st.subheader("Sensor del celular (opcional)")
    st.caption("Sin estos datos la inclinación del muro NO se usa, porque no se distingue del giro del celular.")
    usar_giro = st.checkbox("Tengo el giro del celular", False)
    roll = st.slider("Giro (roll, °)", -45.0, 45.0, 0.0, 0.5) if usar_giro else None
    pitch = st.slider("Inclinación arriba/abajo (pitch, °)", -45.0, 45.0, 0.0, 0.5) if usar_giro else None
    asumir = st.checkbox("Saqué la foto con el celular derecho, de frente al muro (asumir giro 0°)", False,
                         help="Si no tenés el giro del sensor pero la foto la tomaste derecha, marcalo: así se usa la inclinación medida. "
                              "Si la foto salió girada, el resultado de inclinación será falso.")

archivo = st.file_uploader("Foto del muro (de frente, con buena luz)", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.info("Subí una foto para analizarla.")
    st.stop()

foto = np.asarray(Image.open(archivo).convert("RGB"))
try:
    sistema = cargar(nombre)
except Exception as e:                                      # el error más probable: TensorFlow o el modelo
    st.error(f"No pude cargar el modelo: {e}")
    st.stop()

with st.spinner("Analizando..."):
    r = app_core.analizar(sistema, foto, nivel, elemento, ref_mm, ref_px, roll, pitch, asumir_derecho=asumir and not usar_giro, medir_inclinacion=medir)

ev = r["riesgo"]
col1, col2 = st.columns(2)
with col1:
    if r["mapa"] is None:
        st.image(r["foto"], caption="Foto analizada", use_container_width=True)
    else:
        sobre = mosaico.superponer_mapa(r["foto"][..., ::-1], r["mapa"])[..., ::-1]
        st.image(sobre, caption="Mapa de probabilidad de grieta (rojo = más probable)", use_container_width=True)
    bordes = app_core.superponer_bordes(r["foto"], r.get("mapa_inclinacion"))
    if bordes is not None:
        st.image(bordes, caption="Bordes usados para medir la inclinación (verde): tienen que ser los de la columna o el muro", use_container_width=True)
with col2:
    color = {"BAJO": "green", "MEDIO": "orange", "ALTO": "red", "INDETERMINADO": "gray"}[ev.nivel]
    detectada = r["prob_grieta"] >= r["umbral"]
    st.markdown(f"## Clase: {'CON GRIETA' if detectada else 'SIN GRIETA detectada'}")
    st.markdown(f"### Riesgo: :{color}[{ev.nivel}]" + (" (a confirmar)" if ev.a_confirmar else ""))
    st.write(f"Probabilidad de grieta: **{r['prob_grieta']:.3f}** (umbral {r['umbral']:.2f})")
    for linea in ev.razones:
        st.write("•", linea)
    st.info(ev.recomendacion)
    for nota in r["notas"]:
        st.caption("ℹ " + nota)
    if r["medicion"]:
        m = r["medicion"]
        if "ancho_p95_mm" in m:
            st.write(f"Ancho aproximado: **{m['ancho_p95_mm']:.2f} mm**")
        else:
            st.write(f"Ancho aproximado: **{m['ancho_p95_px']:.1f} px** (sin escala no se puede dar en mm)")
    st.write(app_core.descripcion_inclinacion(r))
    if r["orientacion"]:
        st.write(f"Orientación: **{r['orientacion']['categoria']}**")
    if not detectada:
        st.caption("No detectar una grieta no significa que no exista.")
