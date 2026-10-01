"""
ANIMACION 1 - La convolucion, con dos filtros a la vez
Proyecto: Deteccion de grietas - UIS 2026-2

Un filtro VERTICAL y uno HORIZONTAL recorren la misma imagen al mismo
tiempo. Se ve que cada uno responde a un tipo de borde y es ciego al otro.

Uso:
    python animacion_convolucion.py
    python animacion_convolucion.py --imagen data/raw/Positive/00001.jpg
    python animacion_convolucion.py --imagen mis_fotos/columna.jpg --lado 11

Genera: results/figuras/anim_convolucion.gif
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

LADO_DEFECTO = 9
SALIDA = "results/figuras/anim_convolucion.gif"

FILTRO_V = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=float)

FILTRO_H = np.array([[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]], dtype=float)

COL_FONDO  = "#F7F6F2"
COL_TEXTO  = "#16191F"
COL_ACENTO = "#C6552B"
COL_VERDE  = "#1D7A5F"
COL_GRIS   = "#6E7480"
UMBRAL = 100


def imagen_sintetica(lado):
    """Concreto claro con una grieta oscura casi vertical."""
    rng = np.random.default_rng(42)
    img = rng.normal(190, 6, size=(lado, lado))
    for i in range(lado):
        j = int(i * 0.35) + lado // 3
        if 0 <= j < lado:
            img[i, j] = rng.normal(50, 5)
    return np.clip(img, 0, 255)


def imagen_desde_archivo(ruta, lado):
    """Recorta el centro de una foto y lo reduce a lado x lado, en gris."""
    from PIL import Image
    with Image.open(ruta) as im:
        g = im.convert("L")
        w, h = g.size
        c = min(w, h)
        g = g.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
        g = g.resize((lado, lado), Image.BILINEAR)
    return np.asarray(g, dtype=float)


def convolucionar(img, filtro):
    k = filtro.shape[0]
    n = img.shape[0] - k + 1
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.sum(img[i:i + k, j:j + k] * filtro)
    return out


def pintar_matriz(ax, datos, titulo, cmap, vmin, vmax, fuente=8,
                  fmt="{:.0f}", solo_hasta=None, n=None):
    ax.clear()
    ax.set_facecolor(COL_FONDO)
    if solo_hasta is None:
        vista = datos
    else:
        vista = np.full_like(datos, np.nan)
        for kk in range(solo_hasta + 1):
            ii, jj = divmod(kk, n)
            vista[ii, jj] = datos[ii, jj]

    ax.imshow(vista, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=10.5, color=COL_TEXTO, pad=7)
    ax.set_xticks([]); ax.set_yticks([])

    medio = (vmin + vmax) / 2
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = vista[i, j]
            if np.isnan(v):
                continue
            color = "#F7F6F2" if (cmap == "gray" and v < medio) else "#16191F"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fuente, color=color)

    for x in range(datos.shape[1] + 1):
        ax.axvline(x - 0.5, color="#B9B7AF", linewidth=0.6)
    for y in range(datos.shape[0] + 1):
        ax.axhline(y - 0.5, color="#B9B7AF", linewidth=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", default=None,
                    help="Ruta a una foto propia. Sin esto usa una sintetica.")
    ap.add_argument("--lado", type=int, default=LADO_DEFECTO,
                    help="Tamano de la rejilla (9 a 13 se lee bien)")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    img = (imagen_desde_archivo(args.imagen, args.lado) if args.imagen
           else imagen_sintetica(args.lado))

    mapa_v = convolucionar(img, FILTRO_V)
    mapa_h = convolucionar(img, FILTRO_H)
    n = mapa_v.shape[0]
    total = n * n
    lim = max(np.abs(mapa_v).max(), np.abs(mapa_h).max(), 1)

    fig = plt.figure(figsize=(14, 7.6), facecolor=COL_FONDO)
    gs = fig.add_gridspec(3, 3,
                          height_ratios=[1, 1, 0.62],
                          width_ratios=[1.35, 0.5, 1.1],
                          hspace=0.30, wspace=0.22)

    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_fv  = fig.add_subplot(gs[0, 1])
    ax_mv  = fig.add_subplot(gs[0, 2])
    ax_fh  = fig.add_subplot(gs[1, 1])
    ax_mh  = fig.add_subplot(gs[1, 2])
    ax_txt = fig.add_subplot(gs[2, :]); ax_txt.axis("off")

    fuente_img = 7 if args.lado <= 10 else 5.5

    fig.suptitle("Dos filtros recorren la misma imagen a la vez",
                 fontsize=15.5, color=COL_TEXTO, y=0.975)

    def cuadro(k):
        i, j = divmod(k, n)
        ven = img[i:i + 3, j:j + 3]
        sv = float(np.sum(ven * FILTRO_V))
        sh = float(np.sum(ven * FILTRO_H))

        pintar_matriz(ax_img, img, "Imagen (valores de pixel)",
                      "gray", 0, 255, fuente=fuente_img)
        ax_img.add_patch(Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False,
                                   edgecolor=COL_ACENTO, linewidth=3))

        pintar_matriz(ax_fv, FILTRO_V, "Filtro VERTICAL",
                      "coolwarm", -1.6, 1.6, fuente=12, fmt="{:+.0f}")
        pintar_matriz(ax_fh, FILTRO_H, "Filtro HORIZONTAL",
                      "coolwarm", -1.6, 1.6, fuente=12, fmt="{:+.0f}")

        pintar_matriz(ax_mv, mapa_v, "Mapa: bordes verticales",
                      "RdYlGn", -lim, lim, fuente=7, solo_hasta=k, n=n)
        ax_mv.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=COL_ACENTO, linewidth=2.6))

        pintar_matriz(ax_mh, mapa_h, "Mapa: bordes horizontales",
                      "RdYlGn", -lim, lim, fuente=7, solo_hasta=k, n=n)
        ax_mh.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=COL_ACENTO, linewidth=2.6))

        ax_txt.clear(); ax_txt.axis("off")

        col_izq = "\n".join(
            "  ".join(f"({ven[f, c]:5.0f} x {FILTRO_V[f, c]:+.0f})" for c in range(3))
            for f in range(3))
        col_der = "\n".join(
            "  ".join(f"({ven[f, c]:5.0f} x {FILTRO_H[f, c]:+.0f})" for c in range(3))
            for f in range(3))

        ax_txt.text(0.02, 0.95, "VERTICAL", fontsize=11, color=COL_TEXTO,
                    weight="bold", va="top")
        ax_txt.text(0.02, 0.76, col_izq, fontsize=9.5, color=COL_TEXTO,
                    family="monospace", va="top")
        ax_txt.text(0.02, 0.06, f"suma = {sv:+8.0f}", fontsize=13.5,
                    color=COL_TEXTO, family="monospace", weight="bold")
        ax_txt.text(0.21, 0.06, "BORDE" if abs(sv) > UMBRAL else "nada",
                    fontsize=13, weight="bold",
                    color=COL_ACENTO if abs(sv) > UMBRAL else COL_GRIS)

        ax_txt.text(0.52, 0.95, "HORIZONTAL", fontsize=11, color=COL_TEXTO,
                    weight="bold", va="top")
        ax_txt.text(0.52, 0.76, col_der, fontsize=9.5, color=COL_TEXTO,
                    family="monospace", va="top")
        ax_txt.text(0.52, 0.06, f"suma = {sh:+8.0f}", fontsize=13.5,
                    color=COL_TEXTO, family="monospace", weight="bold")
        ax_txt.text(0.71, 0.06, "BORDE" if abs(sh) > UMBRAL else "nada",
                    fontsize=13, weight="bold",
                    color=COL_VERDE if abs(sh) > UMBRAL else COL_GRIS)

        ax_txt.text(0.99, 0.06, f"posicion {k + 1} de {total}",
                    fontsize=10, color=COL_GRIS, ha="right")

    anim = FuncAnimation(fig, cuadro, frames=total, repeat=True)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    anim.save(args.salida, writer=PillowWriter(fps=1))
    print(f"Guardado: {args.salida}")
    print(f"\n{total} posiciones x 9 multiplicaciones x 2 filtros "
          f"= {total * 18} operaciones")
    print("Parametros: 9 por filtro, 18 en total, iguales en toda la imagen.")
    print("\nFijate: donde el filtro vertical se dispara, el horizontal da")
    print("casi cero, y al reves. Cada filtro es ciego al borde del otro.")


if __name__ == "__main__":
    main()