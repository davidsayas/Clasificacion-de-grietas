"""
FIGURA - Que se congela y que se descongela
Proyecto: Deteccion de grietas - UIS 2026-2

Explica, para alguien que no sabe del tema, que partes del modelo
conservan lo que venia de ImageNet y cuales se modificaron al
entrenar con las fotos de grietas.

Uso:
    python figura_congelado.py

Genera:
    results/figuras/congelado/que_se_congela.png
    results/figuras/congelado/antes_despues.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrow

CARPETA = "results/figuras/congelado"
DPI = 150

COL_FONDO   = "#F7F6F2"
COL_TEXTO   = "#16191F"
COL_GRIS    = "#6E7480"

COL_HIELO   = "#4A7FB5"   # congelado
COL_HIELO_F = "#DCE8F4"
COL_FUEGO   = "#C6552B"   # se ajusto
COL_FUEGO_F = "#F7E2D8"
COL_NUEVO   = "#1D7A5F"   # aprendio desde cero
COL_NUEVO_F = "#DAEDE5"

# Los numeros reales del modelo
P_CONGELADO = 731_584
P_AJUSTADO  = 1_526_400
P_NUEVO     = 1_281
P_TOTAL     = P_CONGELADO + P_AJUSTADO + P_NUEVO


def candado(ax, x, y, s=1.0, color=COL_HIELO):
    """Dibuja un candado cerrado."""
    ax.add_patch(Rectangle((x - 0.16 * s, y - 0.20 * s), 0.32 * s, 0.26 * s,
                           facecolor=color, edgecolor=color, lw=1))
    t = np.linspace(np.pi, 2 * np.pi, 40)
    ax.plot(x + 0.10 * s * np.cos(t), y + 0.06 * s + 0.11 * s * np.sin(t),
            color=color, lw=2.4 * s, solid_capstyle="round")


def lapiz(ax, x, y, s=1.0, color=COL_FUEGO):
    """Dibuja un lapiz inclinado."""
    ax.plot([x - 0.16 * s, x + 0.10 * s], [y - 0.16 * s, y + 0.16 * s],
            color=color, lw=4.2 * s, solid_capstyle="butt")
    ax.plot([x + 0.10 * s, x + 0.18 * s], [y + 0.16 * s, y + 0.24 * s],
            color=color, lw=2.0 * s, solid_capstyle="round")


def chispa(ax, x, y, s=1.0, color=COL_NUEVO):
    """Dibuja una estrella de cuatro puntas."""
    for ang in (0, 90):
        r = np.deg2rad(ang)
        ax.plot([x - 0.19 * s * np.cos(r), x + 0.19 * s * np.cos(r)],
                [y - 0.19 * s * np.sin(r), y + 0.19 * s * np.sin(r)],
                color=color, lw=3.0 * s, solid_capstyle="round")
    for ang in (45, 135):
        r = np.deg2rad(ang)
        ax.plot([x - 0.11 * s * np.cos(r), x + 0.11 * s * np.cos(r)],
                [y - 0.11 * s * np.sin(r), y + 0.11 * s * np.sin(r)],
                color=color, lw=2.0 * s, solid_capstyle="round")


def bloque(ax, x, y, w, h, borde, relleno, titulo, estado, detalle,
           params, icono):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor=relleno, edgecolor=borde, lw=2.6))
    icono(ax, x + w / 2, y + h - 0.52, s=1.15)
    ax.text(x + w / 2, y + h - 1.02, titulo, ha="center", fontsize=13,
            color=COL_TEXTO, weight="bold")
    ax.text(x + w / 2, y + h - 1.42, estado, ha="center", fontsize=13.5,
            color=borde, weight="bold")
    ax.text(x + w / 2, y + h - 2.05, detalle, ha="center", fontsize=10.8,
            color=COL_GRIS, va="top", linespacing=1.6)
    ax.text(x + w / 2, y + 0.22, params, ha="center", fontsize=12.5,
            color=borde, weight="bold", family="monospace")


# ==================================================================
# FIGURA 1 - El esquema principal
# ==================================================================
def figura_principal():
    fig, ax = plt.subplots(figsize=(15, 8.4), facecolor=COL_FONDO)
    ax.set_facecolor(COL_FONDO)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.4, 8.45, "¿Que se congela y que se descongela?",
            fontsize=21, color=COL_TEXTO, weight="bold")
    ax.text(0.4, 7.95,
            "La foto pasa entera por todo el modelo. Lo que se congela no es la imagen: "
            "son los numeros que el modelo usa para analizarla.",
            fontsize=12.5, color=COL_GRIS)

    # --- la foto de entrada ---
    ax.add_patch(FancyBboxPatch((0.4, 3.9), 1.5, 2.3,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#E4E2DA", edgecolor="#B9B7AF", lw=2))
    rng = np.random.default_rng(3)
    trozo = rng.normal(0.72, 0.06, (14, 14))
    for i in range(14):
        j = int(i * 0.6) + 3
        if 0 <= j < 14:
            trozo[i, j] = 0.18
    ax.imshow(trozo, cmap="gray", vmin=0, vmax=1,
              extent=(0.62, 1.68, 4.55, 5.98), aspect="auto", zorder=3)
    ax.text(1.15, 4.20, "La foto", ha="center", fontsize=12,
            color=COL_TEXTO, weight="bold")
    ax.text(1.15, 3.45, "entra completa\ny la atraviesa toda",
            ha="center", fontsize=10.3, color=COL_GRIS, va="top")

    ancho, alto, y0 = 4.0, 3.9, 2.5
    xs = [2.45, 6.75, 11.05]

    bloque(ax, xs[0], y0, ancho, alto, COL_HIELO, COL_HIELO_F,
           "Capas iniciales", "NO CAMBIARON",
           "Buscan bordes, colores\ny texturas basicas.\n\n"
           "Eso sirve igual para una\ngrieta, un carro o un gato:\n"
           "no habia nada que adaptar.",
           f"{P_CONGELADO:,} numeros", candado)

    bloque(ax, xs[1], y0, ancho, alto, COL_FUEGO, COL_FUEGO_F,
           "Capas finales", "SE AJUSTARON",
           "Combinan esos bordes en\nformas mas complejas.\n\n"
           "Aqui SI importa si es una\ngrieta o un gato, asi que\n"
           "se afinaron con mis fotos.",
           f"{P_AJUSTADO:,} numeros", lapiz)

    bloque(ax, xs[2], y0, ancho, alto, COL_NUEVO, COL_NUEVO_F,
           "El clasificador", "APRENDIO DESDE CERO",
           "Decide: con grieta o sin\ngrieta.\n\n"
           "Lo agregue yo. Arranco con\nnumeros al azar porque\n"
           "nadie le habia enseñado.",
           f"{P_NUEVO:,} numeros", chispa)

    for x in (2.05, 6.35, 10.65):
        ax.add_patch(FancyArrow(x, 4.45, 0.30, 0, width=0.05,
                                head_width=0.26, head_length=0.20,
                                length_includes_head=True, color="#9AA0A8"))
    ax.add_patch(FancyArrow(15.15, 4.45, 0.45, 0, width=0.05,
                            head_width=0.26, head_length=0.22,
                            length_includes_head=True, color="#9AA0A8"))

    ax.text(0.4, 1.72,
            "De los 2.259.265 numeros del modelo, 731.584 quedaron intactos "
            "y 1.527.681 se modificaron al entrenar.",
            fontsize=13, color=COL_TEXTO)
    ax.text(0.4, 1.20,
            "Congelar las primeras capas ahorra tiempo de computo y evita "
            "estropear lo que ya estaba bien aprendido.",
            fontsize=12.5, color=COL_GRIS)

    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, "que_se_congela.png")
    fig.savefig(ruta, dpi=DPI, facecolor=COL_FONDO, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


# ==================================================================
# FIGURA 2 - Antes y despues, con numeros
# ==================================================================
def figura_antes_despues():
    fig, ax = plt.subplots(figsize=(14, 7.6), facecolor=COL_FONDO)
    ax.set_facecolor(COL_FONDO)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(0.4, 7.5, "Los mismos numeros, antes y despues de entrenar",
            fontsize=20, color=COL_TEXTO, weight="bold")
    ax.text(0.4, 7.02,
            "Cada capa guarda una tabla de numeros. Esto es un trocito de "
            "tres de esas tablas.",
            fontsize=12.5, color=COL_GRIS)

    rng = np.random.default_rng(99)
    antes_c = np.round(rng.normal(0, 0.3, 3), 3)
    antes_a = np.round(rng.normal(0, 0.3, 3), 3)
    desp_a = np.round(antes_a + rng.normal(0, 0.09, 3), 3)
    antes_n = np.round(rng.normal(0, 0.02, 3), 3)
    desp_n = np.round(rng.normal(0, 0.9, 3), 3)

    ax.text(4.9, 6.30, "ANTES de entrenar", fontsize=13.5,
            color=COL_TEXTO, weight="bold", ha="center")
    ax.text(9.8, 6.30, "DESPUES de entrenar", fontsize=13.5,
            color=COL_TEXTO, weight="bold", ha="center")

    filas = [
        ("Capas iniciales", COL_HIELO, COL_HIELO_F, antes_c, antes_c,
         "identicos", candado),
        ("Capas finales", COL_FUEGO, COL_FUEGO_F, antes_a, desp_a,
         "cambiaron un poco", lapiz),
        ("El clasificador", COL_NUEVO, COL_NUEVO_F, antes_n, desp_n,
         "cambiaron del todo", chispa),
    ]

    y = 5.10
    for nombre, borde, relleno, a, b, nota, icono in filas:
        icono(ax, 0.75, y + 0.30, s=1.0, color=borde)
        ax.text(1.25, y + 0.42, nombre, fontsize=12.5, color=COL_TEXTO,
                weight="bold", va="center")
        ax.text(1.25, y - 0.02, nota, fontsize=10.8, color=borde, va="center")

        for cx, vals in ((3.9, a), (8.8, b)):
            ax.add_patch(FancyBboxPatch((cx, y - 0.30), 2.05, 0.92,
                                        boxstyle="round,pad=0.02,rounding_size=0.10",
                                        facecolor=relleno, edgecolor=borde, lw=1.8))
            ax.text(cx + 1.02, y + 0.16,
                    "  ".join(f"{v:+.3f}" for v in vals),
                    ha="center", va="center", fontsize=11.5,
                    color=COL_TEXTO, family="monospace")

        ax.add_patch(FancyArrow(6.20, y + 0.16, 2.35, 0, width=0.03,
                                head_width=0.18, head_length=0.20,
                                length_includes_head=True, color="#B9B7AF"))
        y -= 1.55

    ax.text(0.4, 0.95,
            "Las capas congeladas conservan exactamente los valores que "
            "venian entrenados con un millon de fotos de ImageNet.",
            fontsize=12.5, color=COL_TEXTO)
    ax.text(0.4, 0.48,
            "El clasificador arranco con numeros casi en cero, al azar, y "
            "todo lo que sabe lo aprendio con mis 40.000 imagenes de grietas.",
            fontsize=12.5, color=COL_GRIS)

    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, "antes_despues.png")
    fig.savefig(ruta, dpi=DPI, facecolor=COL_FONDO, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


def main():
    print("Generando figuras...\n")
    figura_principal()
    figura_antes_despues()
    print(f"\nCongelados  : {P_CONGELADO:>10,}")
    print(f"Ajustados   : {P_AJUSTADO:>10,}")
    print(f"Nuevos      : {P_NUEVO:>10,}")
    print(f"TOTAL       : {P_TOTAL:>10,}")
    print(f"\nEntrenables : {P_AJUSTADO + P_NUEVO:,}")


if __name__ == "__main__":
    main()