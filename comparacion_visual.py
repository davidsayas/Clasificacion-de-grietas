"""
COMPARACION VISUAL - Normal contra separable, misma imagen
Proyecto: Deteccion de grietas - UIS 2026-2

Aplica los DOS metodos a la misma foto y pone los resultados lado a lado,
con el conteo de parametros de cada uno. Es la figura que demuestra el
argumento: salidas comparables, costo muy distinto.

Uso:
    python comparacion_visual.py
    python comparacion_visual.py --imagen data/raw/Positive/00001.jpg

Genera:
    results/figuras/costo/comparacion_visual.png
    results/figuras/costo/comparacion_resumen.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

LADO = 180
K = 3
C_IN = 3
C_OUT = 4
CARPETA = "results/figuras/costo"
DPI = 145

COL_FONDO  = "#F7F6F2"
COL_TEXTO  = "#16191F"
COL_NORMAL = "#6E7480"
COL_SEPAR  = "#C6552B"
COL_VERDE  = "#1D7A5F"

P_NORMAL    = K * K * C_IN * C_OUT      # 108
P_DEPTHWISE = K * K * C_IN              # 27
P_POINTWISE = C_IN * C_OUT              # 12
P_SEPARABLE = P_DEPTHWISE + P_POINTWISE # 39


def imagen_sintetica(lado):
    rng = np.random.default_rng(7)
    base = rng.normal(185, 9, size=(lado, lado))
    for i in range(lado):
        j = int(i * 0.55) + lado // 5
        for d in (-1, 0, 1):
            if 0 <= j + d < lado:
                base[i, j + d] = rng.normal(55, 8)
    return np.clip(np.stack([base * 1.03, base * 0.99, base * 0.94], -1), 0, 255)


def imagen_desde_archivo(ruta, lado):
    from PIL import Image
    with Image.open(ruta) as im:
        c = im.convert("RGB")
        w, h = c.size
        m = min(w, h)
        c = c.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        c = c.resize((lado, lado), Image.BILINEAR)
    return np.asarray(c, dtype=float)


def conv2d(canal, filtro):
    ven = sliding_window_view(canal, filtro.shape)
    return np.einsum("ijkl,kl->ij", ven, filtro)


def normalizar(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


# ------------------------------------------------------------------
# LOS FILTROS
# ------------------------------------------------------------------
rng = np.random.default_rng(2026)

# NORMAL: C_OUT filtros, cada uno de KxKxC_IN. Cada filtro toca los 3 canales.
FILTROS_NORMAL = rng.normal(0, 0.6, size=(C_OUT, K, K, C_IN))

# SEPARABLE: un filtro plano por canal + una matriz de mezcla 1x1
FILTROS_DW = np.stack([
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),    # vertical
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),    # horizontal
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float),   # detalle
], axis=0)
PESOS_PW = rng.normal(0, 0.6, size=(C_OUT, C_IN))


def aplicar_normal(img):
    """Cada filtro cubre los 3 canales y los suma en UNA salida."""
    salidas = []
    for o in range(C_OUT):
        acum = None
        for c in range(C_IN):
            r = conv2d(img[:, :, c], FILTROS_NORMAL[o, :, :, c])
            acum = r if acum is None else acum + r
        salidas.append(acum)
    return salidas


def aplicar_separable(img):
    """Paso 1 depthwise (sin mezclar), paso 2 pointwise (mezcla 1x1)."""
    filtrados = [conv2d(img[:, :, c], FILTROS_DW[c]) for c in range(C_IN)]
    pila = np.stack(filtrados, axis=-1)
    salidas = [np.einsum("ijc,c->ij", pila, PESOS_PW[o]) for o in range(C_OUT)]
    return salidas, filtrados


def energia(m):
    """Cuanta 'estructura' capta el mapa: desviacion estandar normalizada."""
    return float(np.std(normalizar(m)))


# ==================================================================
# FIGURA A - Los mapas, lado a lado
# ==================================================================
def figura_visual(img, sal_n, sal_s, filtrados, ruta):
    fig = plt.figure(figsize=(15, 8.6), facecolor=COL_FONDO)
    gs = fig.add_gridspec(3, C_OUT + 1,
                          height_ratios=[1, 1, 0.42],
                          hspace=0.30, wspace=0.16)

    fig.suptitle("Misma imagen, los dos metodos: salidas comparables, costo distinto",
                 fontsize=16, color=COL_TEXTO, y=0.975, weight="bold")

    def pintar(ax, datos, titulo, cmap=None, sub=""):
        ax.clear(); ax.set_facecolor(COL_FONDO)
        if cmap:
            ax.imshow(normalizar(datos), cmap=cmap, vmin=-1, vmax=1)
        else:
            ax.imshow(datos.astype(np.uint8))
        ax.set_title(titulo, fontsize=10.5, color=COL_TEXTO, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#C4C2BA")
        if sub:
            ax.set_xlabel(sub, fontsize=9, color="#6E7480", labelpad=4)

    # Columna 0: la foto, centrada entre las dos filas
    ax0 = fig.add_subplot(gs[0:2, 0])
    pintar(ax0, img, "La imagen", sub=f"{LADO}x{LADO}, {C_IN} canales")

    # Fila 1: NORMAL
    for o in range(C_OUT):
        ax = fig.add_subplot(gs[0, o + 1])
        pintar(ax, sal_n[o], f"NORMAL · salida {o+1}", cmap="RdYlGn")
        if o == 0:
            ax.text(-0.10, 0.5, "NORMAL", transform=ax.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=13, color=COL_NORMAL, weight="bold")

    # Fila 2: SEPARABLE
    for o in range(C_OUT):
        ax = fig.add_subplot(gs[1, o + 1])
        pintar(ax, sal_s[o], f"SEPARABLE · salida {o+1}", cmap="RdYlGn")
        if o == 0:
            ax.text(-0.10, 0.5, "SEPARABLE", transform=ax.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=13, color=COL_SEPAR, weight="bold")

    # Pie: el costo
    axt = fig.add_subplot(gs[2, :]); axt.axis("off")
    axt.text(0.01, 0.92,
             f"NORMAL      {C_OUT} filtros de {K}x{K}x{C_IN}"
             f"                     {P_NORMAL} parametros",
             fontsize=13, color=COL_NORMAL, family="monospace",
             weight="bold", va="top")
    axt.text(0.01, 0.62,
             f"SEPARABLE   depthwise {P_DEPTHWISE} + pointwise {P_POINTWISE}"
             f"          {P_SEPARABLE} parametros",
             fontsize=13, color=COL_SEPAR, family="monospace",
             weight="bold", va="top")
    axt.text(0.01, 0.26,
             f"La separable cuesta el {100*P_SEPARABLE/P_NORMAL:.0f} % y produce "
             f"mapas del mismo tipo: bordes, texturas y contrastes.",
             fontsize=12.5, color=COL_TEXTO, va="top")
    axt.text(0.01, 0.02,
             "No son identicos: la separable es una aproximacion mas barata. "
             "Por eso MobileNet apila 53 bloques en vez de unas pocas capas normales.",
             fontsize=11, color="#6E7480", va="top")

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=DPI, facecolor=COL_FONDO, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


# ==================================================================
# FIGURA B - El resumen numerico
# ==================================================================
def figura_resumen(sal_n, sal_s, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   facecolor=COL_FONDO)

    # --- izquierda: parametros ---
    ax1.set_facecolor(COL_FONDO)
    ax1.set_title("Parametros para el mismo trabajo", fontsize=13.5,
                  color=COL_TEXTO, pad=12, weight="bold")
    ax1.barh(1.4, P_NORMAL, height=0.45, color=COL_NORMAL)
    ax1.barh(0.5, P_DEPTHWISE, height=0.45, color=COL_SEPAR)
    ax1.barh(0.5, P_POINTWISE, height=0.45, left=P_DEPTHWISE, color=COL_VERDE)
    ax1.text(P_NORMAL * 1.03, 1.4, f"{P_NORMAL}", va="center",
             fontsize=16, color=COL_TEXTO, weight="bold")
    ax1.text(P_SEPARABLE + P_NORMAL * 0.03, 0.5, f"{P_SEPARABLE}", va="center",
             fontsize=16, color=COL_SEPAR, weight="bold")
    ax1.text(0, 1.78, "normal", fontsize=12, color=COL_TEXTO)
    ax1.text(0, 0.88, "separable (dw + pw)", fontsize=12, color=COL_TEXTO)
    ax1.set_xlim(0, P_NORMAL * 1.25); ax1.set_ylim(-0.3, 2.2)
    ax1.set_yticks([]); ax1.set_xlabel("parametros", fontsize=11, color=COL_TEXTO)
    ax1.tick_params(colors=COL_TEXTO)
    for s in ax1.spines.values():
        s.set_edgecolor("#C4C2BA")

    # --- derecha: estructura captada por cada mapa ---
    ax2.set_facecolor(COL_FONDO)
    ax2.set_title("Estructura captada en cada salida", fontsize=13.5,
                  color=COL_TEXTO, pad=12, weight="bold")
    x = np.arange(C_OUT)
    en = [energia(m) for m in sal_n]
    es = [energia(m) for m in sal_s]
    ax2.bar(x - 0.19, en, 0.38, color=COL_NORMAL, label="normal")
    ax2.bar(x + 0.19, es, 0.38, color=COL_SEPAR, label="separable")
    ax2.set_xticks(x); ax2.set_xticklabels([f"salida {i+1}" for i in x])
    ax2.set_ylabel("desviacion estandar del mapa", fontsize=11, color=COL_TEXTO)
    ax2.legend(fontsize=11, facecolor=COL_FONDO, edgecolor="#C4C2BA")
    ax2.grid(alpha=0.3, axis="y", color="#D8D6CE")
    ax2.tick_params(colors=COL_TEXTO)
    for s in ax2.spines.values():
        s.set_edgecolor("#C4C2BA")

    fig.text(0.5, -0.02,
             "Los dos metodos captan una cantidad de estructura parecida; "
             "la separable lo hace con el 36 % de los parametros.",
             fontsize=12, color=COL_TEXTO, ha="center")

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=DPI, facecolor=COL_FONDO, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", default=None)
    args = ap.parse_args()

    img = (imagen_desde_archivo(args.imagen, LADO) if args.imagen
           else imagen_sintetica(LADO))

    sal_n = aplicar_normal(img)
    sal_s, filtrados = aplicar_separable(img)

    print("Generando comparacion visual...\n")
    figura_visual(img, sal_n, sal_s, filtrados,
                  os.path.join(CARPETA, "comparacion_visual.png"))
    figura_resumen(sal_n, sal_s,
                   os.path.join(CARPETA, "comparacion_resumen.png"))

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Normal      : {C_OUT} filtros de {K}x{K}x{C_IN}   = {P_NORMAL} parametros")
    print(f"Depthwise   : {C_IN} filtros de {K}x{K}         = {P_DEPTHWISE}")
    print(f"Pointwise   : {C_OUT} filtros de 1x1x{C_IN}     = {P_POINTWISE}")
    print(f"Separable   : {P_DEPTHWISE} + {P_POINTWISE}"
          f"                      = {P_SEPARABLE} parametros")
    print(f"\nCosto relativo: {100*P_SEPARABLE/P_NORMAL:.1f} %")
    print("\nEstructura captada (desviacion estandar del mapa normalizado):")
    for o in range(C_OUT):
        print(f"  salida {o+1}:  normal {energia(sal_n[o]):.3f}   "
              f"separable {energia(sal_s[o]):.3f}")
    print("\nADVERTENCIA para el informe: las salidas NO son identicas.")
    print("La separable es una aproximacion mas barata, no un reemplazo exacto.")
    print("MobileNet compensa apilando muchos bloques.")


if __name__ == "__main__":
    main()
