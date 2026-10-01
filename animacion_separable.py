"""
ANIMACION 2 - Convolucion separable sobre una imagen real
Proyecto: Deteccion de grietas - UIS 2026-2

Aplica de verdad el depthwise y el pointwise sobre una foto, canal por
canal, para que se vea el resultado visual de cada paso. Once escenas,
lentas, con el conteo de parametros en cada una.

Uso:
    python animacion_separable.py
    python animacion_separable.py --imagen data/raw/Positive/00001.jpg
    python animacion_separable.py --imagen mis_fotos/grieta.jpg --segundos 10
    python animacion_separable.py --solo-imagenes

Genera:
    results/figuras/anim_separable.gif          (la animacion)
    results/figuras/separable/paso_01.png ...   (cada escena por separado,
                                                 para pegar en diapositivas)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
LADO = 160          # la foto se reduce a LADO x LADO
C_ENTRADA = 3       # R, G, B
C_SALIDA = 4        # canales de salida del ejemplo
K = 3
SALIDA = "results/figuras/anim_separable.gif"

P_NORMAL    = K * K * C_ENTRADA * C_SALIDA     # 108
P_DEPTHWISE = K * K * C_ENTRADA                # 27
P_POINTWISE = C_ENTRADA * C_SALIDA             # 12
P_SEPARABLE = P_DEPTHWISE + P_POINTWISE        # 39

COL_FONDO  = "#F7F6F2"
COL_TEXTO  = "#16191F"
COL_ACENTO = "#C6552B"
COL_VERDE  = "#1D7A5F"
COL_GRIS   = "#6E7480"

# Un filtro distinto por canal: eso es lo que hace el depthwise
FILTROS_DW = {
    "R": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),   # vertical
    "G": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),   # horizontal
    "B": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float),  # detalle
}
NOMBRE_DW = {"R": "vertical", "G": "horizontal", "B": "detalle"}

# Pesos 1x1: cada salida es una mezcla distinta de los 3 canales filtrados
PESOS_PW = np.array([
    [1.0,  0.2, 0.1],
    [0.1,  1.0, 0.2],
    [0.5,  0.5, 0.8],
    [0.8, -0.4, 0.3],
])


def imagen_sintetica(lado):
    """Concreto con una grieta diagonal, en tres canales."""
    rng = np.random.default_rng(7)
    base = rng.normal(185, 9, size=(lado, lado))
    for i in range(lado):
        j = int(i * 0.55) + lado // 5
        for d in (-1, 0, 1):
            if 0 <= j + d < lado:
                base[i, j + d] = rng.normal(55, 8)
    img = np.stack([base * 1.03, base * 0.99, base * 0.94], axis=-1)
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


def conv2d(canal, filtro):
    """Convolucion valida, vectorizada."""
    from numpy.lib.stride_tricks import sliding_window_view
    ven = sliding_window_view(canal, filtro.shape)
    return np.einsum("ijkl,kl->ij", ven, filtro)


def normalizar(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


def panel(ax, datos, titulo, cmap=None, sub=""):
    ax.clear(); ax.set_facecolor(COL_FONDO)
    ax.imshow(datos, cmap=cmap) if cmap else ax.imshow(datos.astype(np.uint8))
    ax.set_title(titulo, fontsize=10.5, color=COL_TEXTO, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#C4C2BA")
    if sub:
        ax.set_xlabel(sub, fontsize=9, color=COL_GRIS, labelpad=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", default=None,
                    help="Ruta a una foto propia. Sin esto usa una sintetica.")
    ap.add_argument("--segundos", type=float, default=7.0,
                    help="Segundos por escena en el GIF")
    ap.add_argument("--solo-imagenes", action="store_true",
                    help="Genera solo los PNG, sin el GIF (mucho mas rapido)")
    ap.add_argument("--dpi", type=int, default=130,
                    help="Resolucion de los PNG")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    img = (imagen_desde_archivo(args.imagen, LADO) if args.imagen
           else imagen_sintetica(LADO))

    canales = {"R": img[:, :, 0], "G": img[:, :, 1], "B": img[:, :, 2]}
    filtrados = {c: conv2d(canales[c], FILTROS_DW[c]) for c in "RGB"}
    pila_dw = np.stack([filtrados[c] for c in "RGB"], axis=-1)

    salidas_pw = [np.einsum("ijc,c->ij", pila_dw, PESOS_PW[o])
                  for o in range(C_SALIDA)]

    fig = plt.figure(figsize=(14, 7.4), facecolor=COL_FONDO)

    def limpiar():
        fig.clf()
        fig.patch.set_facecolor(COL_FONDO)

    def encabezado(titulo, sub):
        fig.suptitle(titulo, fontsize=17, color=COL_TEXTO, y=0.965, weight="bold")
        fig.text(0.5, 0.905, sub, fontsize=12, color=COL_GRIS, ha="center")

    def pie(texto, cuenta="", color=COL_ACENTO):
        fig.text(0.06, 0.09, texto, fontsize=12.5, color=COL_TEXTO, va="top")
        if cuenta:
            fig.text(0.06, 0.025, cuenta, fontsize=14, color=color,
                     family="monospace", weight="bold")

    # ---------------- LAS ESCENAS ----------------
    def esc_0():
        encabezado("La imagen de entrada",
                   "Para el computador no es una foto: son tres tablas de numeros")
        ax = fig.add_subplot(1, 2, 1); panel(ax, img, "La foto")
        ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off"); ax2.set_facecolor(COL_FONDO)
        ax2.text(0.05, 0.75, f"{LADO} x {LADO} pixeles\n{C_ENTRADA} canales de color",
                 fontsize=15, color=COL_TEXTO, va="top")
        ax2.text(0.05, 0.40, f"{LADO*LADO*3:,} numeros en total",
                 fontsize=15, color=COL_ACENTO, weight="bold")
        pie("Cada canal guarda la intensidad de un color en cada pixel.")

    def esc_1():
        encabezado("Los tres canales, por separado",
                   "Rojo, verde y azul: tres tablas independientes")
        for i, c in enumerate("RGB"):
            ax = fig.add_subplot(1, 3, i + 1)
            panel(ax, canales[c], f"Canal {c}", cmap="gray",
                  sub=f"{LADO} x {LADO} = {LADO*LADO:,} numeros")
        pie("Una convolucion normal mira los tres a la vez. MobileNet no.")

    def esc_2():
        encabezado("Convolucion normal",
                   "Un filtro cubre los tres canales: mira vecinos Y mezcla canales")
        ax = fig.add_subplot(1, 2, 1); ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        for i in range(C_ENTRADA):
            ax.add_patch(Rectangle((1 + i * 0.5, 5 - i * 0.5), 3.4, 3.4,
                                   facecolor="#5A6270", edgecolor="#3A3E45", lw=2))
        ax.text(2.9, 4.0, "3 canales", fontsize=12, color=COL_TEXTO, ha="center")
        ax.annotate("", xy=(7.6, 6.4), xytext=(5.4, 6.4),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=COL_GRIS))
        for i in range(C_SALIDA):
            ax.add_patch(Rectangle((7.7 + i * 0.28, 5.6 - i * 0.28), 1.6, 1.6,
                                   facecolor=COL_ACENTO, edgecolor="#8E3A1C", lw=1.8))
        ax.text(8.6, 4.4, f"{C_SALIDA} filtros\nde {K}x{K}x{C_ENTRADA}",
                fontsize=12, color=COL_TEXTO, ha="center")

        ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off")
        ax2.text(0.02, 0.80, "Cada filtro tiene que cubrir\nlos tres canales de entrada.",
                 fontsize=14, color=COL_TEXTO, va="top")
        ax2.text(0.02, 0.42, f"{K} x {K} x {C_ENTRADA} = {K*K*C_ENTRADA} numeros por filtro",
                 fontsize=13, color=COL_GRIS, family="monospace")
        ax2.text(0.02, 0.25, f"x {C_SALIDA} filtros",
                 fontsize=13, color=COL_GRIS, family="monospace")
        pie("Este es el costo que MobileNet quiere reducir.",
            f"{K} x {K} x {C_ENTRADA} x {C_SALIDA}  =  {P_NORMAL} parametros")

    def esc_dw(idx):
        c = "RGB"[idx]
        encabezado(f"DEPTHWISE - paso {idx+1} de 3: canal {c}",
                   f"Un filtro de {K}x{K} solo para este canal. No toca los otros.")
        ax = fig.add_subplot(1, 3, 1)
        panel(ax, canales[c], f"Canal {c}", cmap="gray")

        axf = fig.add_subplot(1, 3, 2)
        f = FILTROS_DW[c]
        axf.clear(); axf.set_facecolor(COL_FONDO)
        axf.imshow(f, cmap="coolwarm", vmin=-4, vmax=4)
        axf.set_title(f"Filtro {NOMBRE_DW[c]}", fontsize=10.5, color=COL_TEXTO, pad=6)
        axf.set_xticks([]); axf.set_yticks([])
        for a in range(3):
            for b in range(3):
                axf.text(b, a, f"{f[a,b]:+.0f}", ha="center", va="center",
                         fontsize=17, color="#16191F", weight="bold")
        axf.set_xlabel(f"{K*K} parametros", fontsize=9, color=COL_GRIS, labelpad=5)

        axr = fig.add_subplot(1, 3, 3)
        panel(axr, normalizar(filtrados[c]), f"Canal {c} filtrado", cmap="RdYlGn")

        acum = K * K * (idx + 1)
        pie(f"Canal {c} entra, canal {c} sale. Los canales NO se mezclan.",
            f"acumulado: {K} x {K} x {idx+1} = {acum} parametros")

    def esc_6():
        encabezado("Resultado del DEPTHWISE",
                   "Tres canales filtrados, todavia sin mezclar entre si")
        for i, c in enumerate("RGB"):
            ax = fig.add_subplot(1, 3, i + 1)
            panel(ax, normalizar(filtrados[c]), f"{c} → {NOMBRE_DW[c]}", cmap="RdYlGn")
        pie("Cada canal vio a sus vecinos. Ninguno vio a los otros canales.",
            f"{K} x {K} x {C_ENTRADA}  =  {P_DEPTHWISE} parametros")

    def esc_7():
        encabezado("POINTWISE",
                   "Filtros de 1x1: mezclan los canales sin mirar vecinos")
        for i, c in enumerate("RGB"):
            ax = fig.add_subplot(2, 4, i + 1)
            panel(ax, normalizar(filtrados[c]), f"{c} filtrado", cmap="RdYlGn")
        axp = fig.add_subplot(2, 4, 4); axp.axis("off")
        p = PESOS_PW[0]
        axp.text(0.0, 0.62, "pesos 1x1", fontsize=12, color=COL_TEXTO, weight="bold")
        axp.text(0.0, 0.34, f"{p[0]:+.1f} R\n{p[1]:+.1f} G\n{p[2]:+.1f} B",
                 fontsize=14, color=COL_ACENTO, family="monospace", va="top")
        axo = fig.add_subplot(2, 1, 2)
        panel(axo, normalizar(salidas_pw[0]), "Salida 1: combinacion de los tres",
              cmap="RdYlGn")
        pie("Un solo pixel a la vez, pero de los tres canales juntos.",
            f"1 x 1 x {C_ENTRADA} = {C_ENTRADA} parametros por salida")

    def esc_8():
        encabezado("Varias salidas, varias mezclas",
                   f"Cada una de las {C_SALIDA} salidas usa pesos distintos")
        for o in range(C_SALIDA):
            ax = fig.add_subplot(1, C_SALIDA, o + 1)
            p = PESOS_PW[o]
            panel(ax, normalizar(salidas_pw[o]), f"Salida {o+1}", cmap="RdYlGn",
                  sub=f"{p[0]:+.1f}R {p[1]:+.1f}G {p[2]:+.1f}B")
        pie("Mismos canales filtrados, distintas combinaciones.",
            f"1 x 1 x {C_ENTRADA} x {C_SALIDA}  =  {P_POINTWISE} parametros")

    def esc_9():
        encabezado("La comparacion",
                   "Mismo tipo de resultado, muchisimo menos costo")
        ax = fig.add_subplot(1, 1, 1); ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ancho_n = 7.6
        ancho_s = ancho_n * P_SEPARABLE / P_NORMAL

        ax.text(0.3, 7.9, "Convolucion normal", fontsize=13, color=COL_TEXTO)
        ax.add_patch(Rectangle((0.3, 6.6), ancho_n, 1.0,
                               facecolor=COL_GRIS, edgecolor="#4A4E54", lw=1.5))
        ax.text(0.3 + ancho_n + 0.2, 7.1, f"{P_NORMAL}", fontsize=17,
                color=COL_TEXTO, va="center", weight="bold")

        ax.text(0.3, 5.1, "Convolucion separable", fontsize=13, color=COL_TEXTO)
        ax.add_patch(Rectangle((0.3, 3.8), ancho_s, 1.0,
                               facecolor=COL_ACENTO, edgecolor="#8E3A1C", lw=1.5))
        ax.text(0.3 + ancho_s + 0.2, 4.3, f"{P_SEPARABLE}", fontsize=17,
                color=COL_ACENTO, va="center", weight="bold")
        ax.text(0.3, 2.5,
                f"depthwise {P_DEPTHWISE}  +  pointwise {P_POINTWISE}  =  {P_SEPARABLE}",
                fontsize=13, color=COL_GRIS, family="monospace")
        pie(f"Con estos numeros el ahorro es del {100-100*P_SEPARABLE/P_NORMAL:.0f} %.",
            f"1/C_salida + 1/K²  =  1/{C_SALIDA} + 1/{K*K}  =  "
            f"{P_SEPARABLE/P_NORMAL:.3f}")

    def esc_10():
        encabezado("En capas reales el ahorro llega al 11 %",
                   "El termino 1/C_salida se achica cuando hay muchos canales")
        ax = fig.add_subplot(1, 1, 1); ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        filas = [("C_salida", "1/C_sal", "1/K²", "total"),
                 ("4",        "0,250",  "0,111", "36,1 %"),
                 ("32",       "0,031",  "0,111", "14,2 %"),
                 ("160",      "0,006",  "0,111", "11,7 %"),
                 ("320",      "0,003",  "0,111", "11,4 %"),
                 ("muy alto", "0,000",  "0,111", "11,1 %")]
        y = 8.2
        for r, fila in enumerate(filas):
            peso = "bold" if r == 0 else "normal"
            color = COL_TEXTO if r < len(filas) - 1 else COL_ACENTO
            for c, celda in enumerate(fila):
                ax.text(1.0 + c * 2.1, y, celda, fontsize=14, color=color,
                        weight=peso, family="monospace")
            if r == 0:
                ax.plot([0.9, 8.4], [y - 0.32, y - 0.32], color=COL_TEXTO, lw=1.3)
            y -= 1.05
        pie("El piso lo fija el tamano del filtro, no la cantidad de canales.",
            "con filtros de 3x3:  1/9 = 11,1 %", color=COL_VERDE)

    escenas = [esc_0, esc_1, esc_2,
               lambda: esc_dw(0), lambda: esc_dw(1), lambda: esc_dw(2),
               esc_6, esc_7, esc_8, esc_9, esc_10]

    # ---------- 1. Cada escena como PNG suelto ----------
    carpeta_png = os.path.join(os.path.dirname(args.salida) or ".", "separable")
    os.makedirs(carpeta_png, exist_ok=True)

    for i, escena in enumerate(escenas):
        limpiar()
        escena()
        ruta = os.path.join(carpeta_png, f"paso_{i+1:02d}.png")
        fig.savefig(ruta, dpi=args.dpi, facecolor=COL_FONDO)
        print(f"  {ruta}")

    print(f"\n{len(escenas)} imagenes en {carpeta_png}/")

    # ---------- 2. El GIF ----------
    if args.solo_imagenes:
        print("\n(GIF omitido por --solo-imagenes)")
    else:
        def cuadro(t):
            limpiar()
            escenas[t]()

        anim = FuncAnimation(fig, cuadro, frames=len(escenas), repeat=True)
        os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
        # fps fraccionario: cada cuadro se muestra args.segundos completos
        anim.save(args.salida, writer=PillowWriter(fps=1.0 / args.segundos))
        print(f"\nGIF: {args.salida}")
        print(f"{len(escenas)} escenas a {args.segundos:.0f} s cada una "
              f"= {len(escenas)*args.segundos:.0f} s en total\n")
    print(f"Normal    : {P_NORMAL} parametros")
    print(f"Depthwise : {P_DEPTHWISE}")
    print(f"Pointwise : {P_POINTWISE}")
    print(f"Separable : {P_SEPARABLE}  ({100*P_SEPARABLE/P_NORMAL:.1f} % del normal)")


if __name__ == "__main__":
    main()