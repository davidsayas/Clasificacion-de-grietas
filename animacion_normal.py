"""
ANIMACION 3 - La convolucion NORMAL, sobre los tres canales a la vez
Proyecto: Deteccion de grietas - UIS 2026-2

Un filtro de 3x3x3 recorre los canales R, G y B al mismo tiempo.
Hace 27 multiplicaciones por posicion y las suma TODAS en un solo
numero de salida: por eso se dice que "mezcla canales".

Comparar con animacion_separable.py, donde ese trabajo se parte en dos.

Uso:
    python animacion_normal.py
    python animacion_normal.py --imagen data/raw/Positive/00001.jpg
    python animacion_normal.py --imagen mis_fotos/grieta.jpg --lado 7

Genera: results/figuras/anim_normal.gif
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

LADO_DEFECTO = 7
SALIDA = "results/figuras/anim_normal.gif"

# UN filtro de 3x3x3: una capa para cada canal de entrada.
# Las tres capas son distintas, pero pertenecen al MISMO filtro.
FILTRO_3D = np.stack([
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),   # capa para R
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),   # capa para G
    np.array([[0, -1, 0], [-1, 2, -1], [0, -1, 0]], float),  # capa para B
], axis=-1)

CANALES = ["R", "G", "B"]
COL_FONDO  = "#F7F6F2"
COL_TEXTO  = "#16191F"
COL_ACENTO = "#C6552B"
COL_GRIS   = "#6E7480"


def imagen_sintetica(lado):
    rng = np.random.default_rng(11)
    base = rng.normal(180, 10, size=(lado, lado))
    for i in range(lado):
        j = int(i * 0.5) + 1
        if 0 <= j < lado:
            base[i, j] = rng.normal(60, 8)
    img = np.stack([base * 1.05, base * 1.00, base * 0.92], axis=-1)
    return np.clip(img, 0, 255)


def imagen_desde_archivo(ruta, lado):
    from PIL import Image
    with Image.open(ruta) as im:
        c = im.convert("RGB")
        w, h = c.size
        m = min(w, h)
        c = c.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        c = c.resize((lado, lado), Image.BILINEAR)
    return np.asarray(c, dtype=float)


def conv_normal(img, filtro3d):
    """Una salida: suma el aporte de los tres canales."""
    k = filtro3d.shape[0]
    n = img.shape[0] - k + 1
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.sum(img[i:i + k, j:j + k, :] * filtro3d)
    return out


def rejilla(ax, datos, titulo, cmap, vmin, vmax, fuente=7,
            fmt="{:.0f}", solo_hasta=None, n=None, cero_claro=False):
    ax.clear(); ax.set_facecolor(COL_FONDO)
    if solo_hasta is None:
        vista = datos
    else:
        vista = np.full_like(datos, np.nan)
        for kk in range(solo_hasta + 1):
            ii, jj = divmod(kk, n)
            vista[ii, jj] = datos[ii, jj]

    ax.imshow(vista, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=10, color=COL_TEXTO, pad=6)
    ax.set_xticks([]); ax.set_yticks([])

    medio = (vmin + vmax) / 2
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = vista[i, j]
            if np.isnan(v):
                continue
            oscuro = (cmap == "gray" and v < medio)
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fuente, color="#F7F6F2" if oscuro else "#16191F")

    for x in range(datos.shape[1] + 1):
        ax.axvline(x - 0.5, color="#B9B7AF", linewidth=0.6)
    for y in range(datos.shape[0] + 1):
        ax.axhline(y - 0.5, color="#B9B7AF", linewidth=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", default=None)
    ap.add_argument("--lado", type=int, default=LADO_DEFECTO)
    ap.add_argument("--segundos", type=float, default=2.0)
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    img = (imagen_desde_archivo(args.imagen, args.lado) if args.imagen
           else imagen_sintetica(args.lado))

    mapa = conv_normal(img, FILTRO_3D)
    n = mapa.shape[0]
    total = n * n
    lim = max(np.abs(mapa).max(), 1)

    fig = plt.figure(figsize=(14.5, 8.4), facecolor=COL_FONDO)
    gs = fig.add_gridspec(3, 4,
                          height_ratios=[1.05, 0.62, 0.85],
                          width_ratios=[1, 1, 1, 1.15],
                          hspace=0.34, wspace=0.24)

    ax_can = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_fil = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_map = fig.add_subplot(gs[0:2, 3])
    ax_txt = fig.add_subplot(gs[2, :]); ax_txt.axis("off")

    fig.suptitle("Convolucion NORMAL: un filtro de 3x3x3 toca los tres canales a la vez",
                 fontsize=15, color=COL_TEXTO, y=0.972, weight="bold")

    def cuadro(k):
        i, j = divmod(k, n)
        parciales = []

        for c in range(3):
            canal = img[:, :, c]
            rejilla(ax_can[c], canal, f"Canal {CANALES[c]}",
                    "gray", 0, 255, fuente=7)
            ax_can[c].add_patch(Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False,
                                          edgecolor=COL_ACENTO, linewidth=2.8))

            capa = FILTRO_3D[:, :, c]
            rejilla(ax_fil[c], capa, f"Capa {CANALES[c]} del filtro",
                    "coolwarm", -2.2, 2.2, fuente=12, fmt="{:+.0f}")

            ven = canal[i:i + 3, j:j + 3]
            parciales.append(float(np.sum(ven * capa)))

        suma = sum(parciales)

        rejilla(ax_map, mapa, "UNA sola salida",
                "RdYlGn", -lim, lim, fuente=7, solo_hasta=k, n=n)
        ax_map.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=COL_ACENTO, linewidth=2.8))

        # ---------- la cuenta ----------
        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.text(0.01, 0.97, "27 multiplicaciones en esta posicion:",
                    fontsize=12, color=COL_TEXTO, weight="bold", va="top")

        for c in range(3):
            canal = img[:, :, c]
            capa = FILTRO_3D[:, :, c]
            ven = canal[i:i + 3, j:j + 3]
            cuenta = "  ".join(
                f"({ven[f, cc]:4.0f}x{capa[f, cc]:+.0f})"
                for f in range(3) for cc in range(3))
            ax_txt.text(0.01, 0.80 - c * 0.15,
                        f"{CANALES[c]}:  {cuenta}   =  {parciales[c]:+8.0f}",
                        fontsize=9.5, color=COL_TEXTO, family="monospace", va="top")

        ax_txt.plot([0.01, 0.72], [0.30, 0.30], color=COL_GRIS, lw=1.1,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.01, 0.20,
                    f"SUMA DE LOS TRES  =  {parciales[0]:+.0f} "
                    f"{parciales[1]:+.0f} {parciales[2]:+.0f}  =  {suma:+.0f}",
                    fontsize=13.5, color=COL_ACENTO, family="monospace",
                    weight="bold", va="top")
        ax_txt.text(0.01, 0.04,
                    "Los tres canales se funden en UN numero: eso es mezclar canales.",
                    fontsize=11.5, color=COL_TEXTO, va="top")
        ax_txt.text(0.99, 0.04, f"posicion {k + 1} de {total}",
                    fontsize=10, color=COL_GRIS, ha="right", va="top")

    anim = FuncAnimation(fig, cuadro, frames=total, repeat=True)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    anim.save(args.salida, writer=PillowWriter(fps=1.0 / args.segundos))

    print(f"Guardado: {args.salida}")
    print(f"\n{total} posiciones x 27 multiplicaciones = {total*27} operaciones")
    print("Parametros del filtro: 3 x 3 x 3 = 27 (para UNA sola salida)")
    print("\nPara 4 salidas harian falta 4 filtros como este: 27 x 4 = 108 parametros.")
    print("La separable logra lo mismo con 39. Ver comparacion_costo.py")


if __name__ == "__main__":
    main()