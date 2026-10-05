"""riesgo.py — Motor de riesgo: combina grieta, ancho de la grieta e inclinación.

⚠ LOS UMBRALES DE ESTE ARCHIVO SON ORIENTATIVOS. Están arriba, con nombre, para
que los reemplaces por los de la norma que corresponda (NSR-10 u otra) o por los
que defina tu docente. El resultado es un TAMIZAJE, no un dictamen estructural.

PRINCIPIO: ante la duda, NO afirmar. Si la imagen no parece concreto, o el modelo
duda, el resultado es INDETERMINADO y se pide otra foto. Mejor eso que un falso
positivo (o peor, un falso negativo) con cara de certeza.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config

# --- Umbrales ORIENTATIVOS (editables) -------------------------------------
ANCHO_BANDAS_MM = (0.3, 1.0, 3.0)        # puntos 0 | 1 | 2 | 3 según el ancho de la grieta
INCLINACION_BANDAS_DEG = (0.5, 1.5, 3.0)  # puntos 0 | 1 | 2 | 3 según |inclinación|
FACTOR_CRITICO_ANCHO_MM = 3.0            # un solo factor así de grande fuerza ALTO
FACTOR_CRITICO_INCL_DEG = 3.0
# Una inclinación medida en la imagen SIN saber cuánto giró el celular puede incluir ese giro (una mano suele girar unos pocos
# grados). Por eso solo cuenta si es grande; y, como el costo de ignorar una columna realmente inclinada es alto, cuando cuenta
# el nivel es ALTO "a confirmar" con un nivel o una plomada, en vez de callarlo.
INCLINACION_SIN_VERIFICAR_MIN_DEG = 5.0

# Heurísticas generales (el documento del reto pide considerar "orientación de la grieta,
# elemento afectado, ancho aproximado y desaplome"). ORIENTATIVAS: confirmalas con tu docente.
#   - Una grieta DIAGONAL suele asociarse a cortante o a asentamientos diferenciales; las
#     verticales u horizontales finas son con más frecuencia de retracción o de juntas.
#   - Columnas y vigas cargan la estructura: la misma grieta se trata con más cautela que en
#     un muro divisorio o una placa.
ORIENTACION_PUNTOS = {"diagonal": 1, "vertical": 0, "horizontal": 0}
ELEMENTO_PUNTOS = {"columna": 1, "viga": 1, "muro": 0, "placa": 0}
ELEMENTOS = tuple(ELEMENTO_PUNTOS)


@dataclass
class Evaluacion:
    nivel: str                            # BAJO | MEDIO | ALTO | INDETERMINADO
    puntaje: int | None
    razones: list[str] = field(default_factory=list)
    recomendacion: str = ""
    a_confirmar: bool = False         # ALTO por una inclinación sin verificar


def _puntos(valor: float, bandas: tuple) -> int:
    return sum(valor >= b for b in bandas)


def evaluar_riesgo(prob_grieta: float, ancho_mm: float | None = None,
                   inclinacion_deg: float | None = None, es_concreto: bool = True,
                   umbral: float = config.UMBRAL_DECISION,
                   banda_incierta: tuple = config.BANDA_INCIERTA,
                   orientacion: str | None = None, elemento: str | None = None,
                   inclinacion_verificada: bool = True) -> Evaluacion:
    """Nivel de riesgo a partir de lo medido. Cualquier dato puede faltar (None)."""
    if not es_concreto:
        return Evaluacion("INDETERMINADO", None,
                          ["La imagen no parece una superficie de concreto."],
                          "Repetir la foto acercándose a la zona de concreto, con buena luz y "
                          "sin objetos delante.")
    en_duda = banda_incierta[0] < prob_grieta < banda_incierta[1]
    inclinacion_critica = inclinacion_deg is not None and abs(inclinacion_deg) >= FACTOR_CRITICO_INCL_DEG
    if en_duda and not inclinacion_critica:       # una inclinación crítica NO se pierde porque el clasificador dude
        return Evaluacion("INDETERMINADO", None,
                          [f"El modelo duda (probabilidad {prob_grieta:.2f})."],
                          "Tomar otra foto, más cerca y con luz uniforme, o pedir revisión visual.")

    razones, puntos = [], 0
    hay_grieta = prob_grieta >= umbral and not en_duda
    critico = False
    critico_ancho = critico_incl = False

    if hay_grieta:
        puntos += 1
        razones.append(f"Se detectó una grieta (probabilidad {prob_grieta:.2f}).")
        if ancho_mm is None:
            puntos += 1                      # prudencia: sin medir el ancho, no se puede decir que sea una grieta fina
            razones.append("Ancho desconocido (falta un objeto de referencia): se asume una grieta de ancho intermedio.")
        else:
            puntos += _puntos(ancho_mm, ANCHO_BANDAS_MM)
            razones.append(f"Ancho de la grieta ≈ {ancho_mm:.2f} mm.")
            critico_ancho = ancho_mm >= FACTOR_CRITICO_ANCHO_MM
            critico |= critico_ancho
        if orientacion in ORIENTACION_PUNTOS:
            puntos += ORIENTACION_PUNTOS[orientacion]
            razones.append(f"Orientación {orientacion}.")
        if elemento in ELEMENTO_PUNTOS:
            puntos += ELEMENTO_PUNTOS[elemento]
            razones.append(f"Elemento afectado: {elemento}.")
    elif en_duda:
        razones.append(f"El modelo duda sobre la grieta (probabilidad {prob_grieta:.2f}); solo se evalúa el desaplome.")
    else:
        razones.append(f"No se detectó grieta (probabilidad {prob_grieta:.2f}).")

    if inclinacion_deg is not None:
        puntos += _puntos(abs(inclinacion_deg), INCLINACION_BANDAS_DEG)
        sufijo = "" if inclinacion_verificada else " (SIN verificar: no se conoce el giro del celular y el ángulo puede incluirlo)"
        razones.append(f"Inclinación respecto de la vertical ≈ {abs(inclinacion_deg):.1f}°{sufijo}.")
        critico_incl = abs(inclinacion_deg) >= FACTOR_CRITICO_INCL_DEG
        critico |= critico_incl

    a_confirmar = (not inclinacion_verificada) and critico_incl and not critico_ancho
    if critico or puntos >= 5:
        nivel, rec = "ALTO", "Solicitar inspección profesional con urgencia."
        if a_confirmar:
            rec = ("Inclinación importante SIN verificar. Confirmala con un nivel o una plomada, o repetí la foto con el celular "
                   "derecho, antes de decidir; si se confirma, solicitar inspección profesional con urgencia.")
    elif puntos >= 2:
        nivel, rec = "MEDIO", "Monitorear (fotos periódicas) y programar una revisión."
    else:
        nivel, rec = "BAJO", "Sin señales relevantes; mantener seguimiento rutinario."
    return Evaluacion(nivel, puntos, razones, rec, a_confirmar and nivel == "ALTO")


def reglas_markdown() -> str:
    """Las reglas EXACTAS que implementa este archivo, en Markdown (para el informe)."""
    a, i = ANCHO_BANDAS_MM, INCLINACION_BANDAS_DEG
    return "\n".join([
        "| Factor | Regla | Puntos |", "|---|---|---|",
        "| Grieta detectada | probabilidad ≥ umbral | +1 |",
        "| Ancho desconocido (sin objeto de referencia) | no se puede medir en mm | +1 (prudencia) |",
        f"| Ancho | < {a[0]} mm / {a[0]}–{a[1]} / {a[1]}–{a[2]} / ≥ {a[2]} mm | 0 / +1 / +2 / +3 |",
        "| Orientación | " + " · ".join(f"{k} +{v}" for k, v in ORIENTACION_PUNTOS.items()) + " | según orientación |",
        "| Elemento afectado | " + " · ".join(f"{k} +{v}" for k, v in ELEMENTO_PUNTOS.items()) + " | según elemento |",
        f"| Inclinación | < {i[0]}° / {i[0]}–{i[1]} / {i[1]}–{i[2]} / ≥ {i[2]}° | 0 / +1 / +2 / +3 |",
        f"| Inclinación sin verificar (no se conoce el giro del celular) | solo cuenta si es ≥ {INCLINACION_SIN_VERIFICAR_MIN_DEG}°; el nivel es ALTO «a confirmar» | — |",
        "",
        f"**Nivel:** ALTO si el total ≥ 5 **o** el ancho ≥ {FACTOR_CRITICO_ANCHO_MM} mm **o** la inclinación ≥ "
        f"{FACTOR_CRITICO_INCL_DEG}°; MEDIO si el total ≥ 2; BAJO en otro caso.",
        f"**No se afirma (INDETERMINADO)** si la imagen no parece concreto o la probabilidad está entre "
        f"{config.BANDA_INCIERTA[0]} y {config.BANDA_INCIERTA[1]}.",
        "",
        "> Reglas orientativas; no sustituyen la inspección de un ingeniero. Sesgadas a sobreestimar el riesgo.",
    ])
