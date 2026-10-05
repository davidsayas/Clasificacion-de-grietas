"""app_gradio.py — La misma app, con Gradio, para correrla en GOOGLE COLAB sin instalar nada en tu computador.

Se usa desde el cuaderno 07. La lógica (`analizar`, `resultado_a_texto`) no necesita Gradio y está probada con
datos sintéticos; solo `lanzar` importa Gradio.

Gradio abre un enlace público temporal (algo como https://xxxx.gradio.live). Funciona mientras el cuaderno de Colab
siga abierto y conectado; para una demostración, abrilo ANTES de empezar.
"""
from __future__ import annotations

import numpy as np

from . import app_core, mosaico


def imagen_resultado(r: dict) -> np.ndarray:
    """La foto analizada; si se analizó por teselas, con el mapa de probabilidad encima (rojo = más probable)."""
    foto = r["foto"]
    if r["mapa"] is None:
        return foto
    return mosaico.superponer_mapa(foto[..., ::-1], r["mapa"])[..., ::-1]


def resultado_a_texto(r: dict, raiz=".") -> str:
    """El resultado en Markdown: aviso, clase, riesgo, razones, recomendación, medidas y notas."""
    ev = r["riesgo"]
    detectada = r["prob_grieta"] >= r["umbral"]
    lineas = [f"> ⚠ {app_core.aviso_de_limitaciones(raiz)}", "",
              f"## Clase: {'CON GRIETA' if detectada else 'SIN GRIETA detectada'}",
              f"### Riesgo: {ev.nivel}" + (" (a confirmar)" if ev.a_confirmar else ""),
              f"Probabilidad de grieta: **{r['prob_grieta']:.3f}** (umbral {r['umbral']:.2f})", ""]
    lineas += [f"- {x}" for x in ev.razones]
    lineas += ["", f"**Recomendación:** {ev.recomendacion}"]
    m = r.get("medicion")
    if m:
        if "ancho_p95_mm" in m:
            lineas.append(f"\nAncho aproximado: **{m['ancho_p95_mm']:.2f} mm**")
        else:
            lineas.append(f"\nAncho aproximado: **{m['ancho_p95_px']:.1f} px** (sin escala no se puede dar en mm)")
    lineas.append("\n" + app_core.descripcion_inclinacion(r))
    if r.get("orientacion"):
        lineas.append(f"\nOrientación: **{r['orientacion']['categoria']}**")
    lineas += [f"\n*{n}*" for n in r["notas"]]
    if not detectada:
        lineas.append("\n*No detectar una grieta no significa que no exista.*")
    return "\n".join(lineas)


def analizar(sistema, foto, nivel, elemento="No sé", ref_mm=None, ref_px=None, raiz=".", asumir_derecho=False, medir_inclinacion=False):
    """Devuelve (imagen, texto, bordes). `foto` es el arreglo RGB que entrega Gradio; None si no se subió nada."""
    if foto is None:
        return None, "Subí una foto del muro para analizarla.", None
    foto = np.asarray(foto)
    if foto.ndim == 3 and foto.shape[2] == 4:          # PNG con transparencia
        foto = foto[..., :3]
    r = app_core.analizar(sistema, foto.astype(np.uint8), nivel, elemento,
                          ref_mm or None, ref_px or None, asumir_derecho=bool(asumir_derecho), medir_inclinacion=bool(medir_inclinacion))     # 0 o vacío -> sin referencia
    return imagen_resultado(r), resultado_a_texto(r, raiz), app_core.superponer_bordes(r["foto"], r.get("mapa_inclinacion"))


def lanzar(sistema, raiz=".", compartir=True):
    """Arma la interfaz y la abre. `compartir=True` genera el enlace público (necesario en Colab)."""
    import gradio as gr

    niveles = list(app_core.NIVELES)
    with gr.Blocks(title="Grietas y riesgo") as demo:
        gr.Markdown("# Detección de grietas y nivel de riesgo")
        gr.Markdown(app_core.aviso_de_limitaciones(raiz))
        with gr.Row():
            with gr.Column():
                foto = gr.Image(type="numpy", label="Foto del muro (de frente, con buena luz)")
                nivel = gr.Radio(niveles, value=niveles[0], label="Sensibilidad",
                                 info="Con el umbral estándar el modelo casi no se activa en fotos de celular. "
                                      "Subirla detecta más grietas, con más falsas alarmas (valores exploratorios).")
                elemento = gr.Dropdown(list(app_core.ELEMENTOS), value="No sé", label="Elemento fotografiado")
                ref_mm = gr.Number(value=0, label="Largo real del objeto de referencia, en mm (0 si no hay)")
                ref_px = gr.Number(value=0, label="Cuánto mide ese objeto en la foto, en px (0 si no hay)")
                medir = gr.Checkbox(value=False, label="Medir la inclinación (solo si la foto es de una columna o un muro, sin manos ni objetos delante)")
                asumir = gr.Checkbox(value=False, label="Saqué la foto con el celular derecho, de frente al muro (asumir giro 0°)")
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column():
                salida_imagen = gr.Image(label="Resultado")
                salida_bordes = gr.Image(label="Bordes usados para medir la inclinación (verde)")
                salida_texto = gr.Markdown()
        boton.click(lambda f, n, e, m, p, a, i: analizar(sistema, f, n, e, m, p, raiz, a, i),
                    [foto, nivel, elemento, ref_mm, ref_px, asumir, medir], [salida_imagen, salida_texto, salida_bordes])
    demo.launch(share=compartir)
    return demo
