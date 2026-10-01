"""
EDA - Surface Crack Detection (Ozgenel / METU)
Proyecto: Deteccion de grietas e inclinacion - UIS 2026-2

Genera todas las figuras y estadisticas del analisis exploratorio
y las guarda en results/figuras/.

Uso:
    python eda_grietas.py

Ajusta RUTA_DATASET a la carpeta que contiene Positive/ y Negative/.
"""

import os
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")           # guarda figuras sin abrir ventanas
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------------
RUTA_DATASET = r"C:\Users\David\Desktop\grietas-riesgo\data\raw"
CARPETA_SALIDA = "results/figuras"
SEMILLA = 42
N_MUESTRA = 2000        # imagenes por clase para los calculos pesados

random.seed(SEMILLA)
np.random.seed(SEMILLA)
os.makedirs(CARPETA_SALIDA, exist_ok=True)

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")


def listar_clases(ruta):
    """Devuelve {clase: [rutas de archivo]} leyendo las subcarpetas."""
    datos = {}
    for clase in sorted(os.listdir(ruta)):
        sub = os.path.join(ruta, clase)
        if not os.path.isdir(sub):
            continue
        archivos = [os.path.join(sub, f) for f in os.listdir(sub)
                    if f.lower().endswith(EXTENSIONES)]
        if archivos:
            datos[clase] = archivos
    return datos


# ==================================================================
# 1. CONTEO POR CLASE Y BALANCE
# ==================================================================
def conteo_por_clase(datos):
    print("\n" + "=" * 60)
    print("1. CONTEO POR CLASE")
    print("=" * 60)

    clases = list(datos.keys())
    conteos = [len(datos[c]) for c in clases]
    total = sum(conteos)

    for c, n in zip(clases, conteos):
        print(f"  {c:<12} {n:>7} imagenes  ({100*n/total:.1f} %)")
    print(f"  {'TOTAL':<12} {total:>7}")

    razon = max(conteos) / min(conteos)
    print(f"\n  Razon de desbalance: {razon:.2f} : 1")
    if razon < 1.2:
        print("  -> Dataset balanceado. La exactitud es una metrica valida.")
    else:
        print("  -> Dataset desbalanceado. Reportar F1 y recall, no solo exactitud.")
        print(f"  -> Un clasificador trivial alcanzaria {100*max(conteos)/total:.1f} % de exactitud.")

    plt.figure(figsize=(6, 4))
    plt.bar(clases, conteos, color=["#1D9E75", "#E24B4A"][:len(clases)])
    plt.ylabel("Numero de imagenes")
    plt.title("Balance de clases")
    for i, n in enumerate(conteos):
        plt.text(i, n, f"{n:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/01_balance_clases.png", dpi=150)
    plt.close()
    print(f"\n  Figura guardada: {CARPETA_SALIDA}/01_balance_clases.png")


# ==================================================================
# 2. REJILLA DE EJEMPLOS
# ==================================================================
def rejilla_ejemplos(datos, n_por_clase=8):
    print("\n" + "=" * 60)
    print("2. EJEMPLOS POR CLASE")
    print("=" * 60)

    clases = list(datos.keys())
    fig, ejes = plt.subplots(len(clases), n_por_clase,
                             figsize=(2 * n_por_clase, 2.3 * len(clases)))
    if len(clases) == 1:
        ejes = np.array([ejes])

    for i, clase in enumerate(clases):
        muestra = random.sample(datos[clase], min(n_por_clase, len(datos[clase])))
        for j, ruta in enumerate(muestra):
            ax = ejes[i, j]
            ax.imshow(Image.open(ruta))
            ax.axis("off")
            if j == 0:
                ax.set_title(clase, loc="left", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/02_ejemplos_por_clase.png", dpi=150)
    plt.close()
    print(f"  Figura guardada: {CARPETA_SALIDA}/02_ejemplos_por_clase.png")
    print("  -> Revisala a ojo: buscá juntas, manchas de humedad o sombras")
    print("     que un modelo podria confundir con grietas.")


# ==================================================================
# 3. RESOLUCION Y FORMATO
# ==================================================================
def revisar_resolucion(datos, n=500):
    print("\n" + "=" * 60)
    print("3. RESOLUCION Y FORMATO")
    print("=" * 60)

    tamanos, modos = Counter(), Counter()
    for clase, archivos in datos.items():
        for ruta in random.sample(archivos, min(n, len(archivos))):
            with Image.open(ruta) as im:
                tamanos[im.size] += 1
                modos[im.mode] += 1

    print("  Resoluciones encontradas:")
    for t, c in tamanos.most_common(5):
        print(f"    {t[0]} x {t[1]} px  ->  {c} imagenes")
    print("  Modos de color:")
    for m, c in modos.most_common():
        print(f"    {m}  ->  {c} imagenes")

    if len(tamanos) == 1:
        print("\n  -> Resolucion uniforme. El redimensionado a 224x224 es minimo")
        print("     y casi no pierde detalle fino.")
    else:
        print("\n  -> Resoluciones mixtas: documentar el efecto del redimensionado.")


# ==================================================================
# 4. HISTOGRAMA DE INTENSIDAD POR CLASE
# ==================================================================
def histograma_intensidad(datos, n=N_MUESTRA):
    print("\n" + "=" * 60)
    print("4. DISTRIBUCION DE INTENSIDAD")
    print("=" * 60)

    plt.figure(figsize=(7, 4))
    colores = ["#1D9E75", "#E24B4A", "#7B6CE0", "#E0A33B"]

    for k, (clase, archivos) in enumerate(datos.items()):
        medias = []
        for ruta in random.sample(archivos, min(n, len(archivos))):
            with Image.open(ruta) as im:
                medias.append(np.asarray(im.convert("L")).mean())
        medias = np.array(medias)
        print(f"  {clase:<12} media={medias.mean():6.1f}  desv={medias.std():5.1f}")
        plt.hist(medias, bins=50, alpha=0.6, label=clase,
                 color=colores[k % len(colores)])

    plt.xlabel("Intensidad media (0-255)")
    plt.ylabel("Numero de imagenes")
    plt.title("Distribucion de luminancia por clase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/03_histograma_intensidad.png", dpi=150)
    plt.close()

    print(f"\n  Figura guardada: {CARPETA_SALIDA}/03_histograma_intensidad.png")
    print("  -> Si las distribuciones se separan mucho, un clasificador basado")
    print("     solo en brillo daria exactitud alta sin aprender la forma de")
    print("     la grieta. Eso es una fuga de informacion que hay que reportar.")


# ==================================================================
# 5. ESTADISTICOS POR CANAL (para la normalizacion)
# ==================================================================
def estadisticos_canal(datos, n=N_MUESTRA):
    print("\n" + "=" * 60)
    print("5. ESTADISTICOS POR CANAL RGB")
    print("=" * 60)

    todas = [r for archivos in datos.values() for r in archivos]
    muestra = random.sample(todas, min(n, len(todas)))

    suma = np.zeros(3)
    suma_cuad = np.zeros(3)
    for ruta in muestra:
        with Image.open(ruta) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        suma += arr.mean(axis=(0, 1))
        suma_cuad += (arr ** 2).mean(axis=(0, 1))

    media = suma / len(muestra)
    desv = np.sqrt(np.maximum(suma_cuad / len(muestra) - media ** 2, 0))

    for i, c in enumerate("RGB"):
        print(f"  Canal {c}:  media={media[i]:.4f}   desv={desv[i]:.4f}")
    print(f"\n  (calculado sobre {len(muestra)} imagenes, escala 0-1)")


# ==================================================================
# 6. DETECCION DE DUPLICADOS Y CASI-DUPLICADOS
# ==================================================================
def hash_promedio(ruta, lado=8):
    """Average hash: 64 bits que resumen la imagen."""
    with Image.open(ruta) as im:
        arr = np.asarray(im.convert("L").resize((lado, lado), Image.BILINEAR))
    return (arr > arr.mean()).flatten()


def detectar_duplicados(datos, n=N_MUESTRA):
    print("\n" + "=" * 60)
    print("6. DUPLICADOS Y CASI-DUPLICADOS")
    print("=" * 60)

    hashes = defaultdict(list)
    total = 0
    for clase, archivos in datos.items():
        for ruta in random.sample(archivos, min(n, len(archivos))):
            h = hash_promedio(ruta)
            hashes[h.tobytes()].append(os.path.basename(ruta))
            total += 1

    grupos = [v for v in hashes.values() if len(v) > 1]
    n_dup = sum(len(g) - 1 for g in grupos)

    print(f"  Imagenes analizadas : {total}")
    print(f"  Grupos duplicados   : {len(grupos)}")
    print(f"  Imagenes redundantes: {n_dup}  ({100*n_dup/total:.2f} %)")

    if grupos:
        print("\n  Ejemplos de grupos duplicados:")
        for g in grupos[:5]:
            print(f"    {g[:4]}")

    print("\n  IMPORTANTE para el informe:")
    print("  Las imagenes de este dataset son recortes de 458 fotografias")
    print("  originales. Recortes casi identicos pueden quedar repartidos")
    print("  entre entrenamiento y prueba, inflando la metrica de prueba.")
    print("  La particion ideal seria POR FOTOGRAFIA DE ORIGEN, pero el")
    print("  dataset no publica esa correspondencia. Declararlo como")
    print("  limitacion explicita del trabajo.")


# ==================================================================
# 7. MUESTRA PARA REVISION MANUAL DE ETIQUETAS
# ==================================================================
def muestra_revision(datos, n=100):
    print("\n" + "=" * 60)
    print("7. MUESTRA PARA REVISION MANUAL DE ETIQUETAS")
    print("=" * 60)

    filas = ["archivo,clase_original,tu_juicio,comentario"]
    for clase, archivos in datos.items():
        for ruta in random.sample(archivos, min(n // len(datos), len(archivos))):
            filas.append(f"{os.path.basename(ruta)},{clase},,")

    salida = "results/revision_etiquetas.csv"
    os.makedirs("results", exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(filas))

    print(f"  Archivo generado: {salida}")
    print(f"  Contiene {len(filas)-1} imagenes para revisar a mano.")
    print("  Abrilo en Excel, mirá cada imagen y llená 'tu_juicio'.")
    print("  Reportá el porcentaje de discrepancia con la etiqueta original:")
    print("  es tu estimacion del ruido de etiqueta del dataset.")


# ==================================================================
def main():
    if not os.path.isdir(RUTA_DATASET):
        print(f"ERROR: no existe la ruta {RUTA_DATASET}")
        print("Editá la variable RUTA_DATASET al inicio del archivo.")
        return

    datos = listar_clases(RUTA_DATASET)
    if not datos:
        print(f"ERROR: no se encontraron subcarpetas con imagenes en {RUTA_DATASET}")
        print("Se espera una estructura tipo:  data/raw/Positive/  y  data/raw/Negative/")
        return

    print(f"\nDataset: {RUTA_DATASET}")
    print(f"Clases detectadas: {list(datos.keys())}")

    conteo_por_clase(datos)
    rejilla_ejemplos(datos)
    revisar_resolucion(datos)
    histograma_intensidad(datos)
    estadisticos_canal(datos)
    detectar_duplicados(datos)
    muestra_revision(datos)

    print("\n" + "=" * 60)
    print("EDA COMPLETO")
    print("=" * 60)
    print(f"Figuras en: {CARPETA_SALIDA}/")
    print("Ahora escribí los hallazgos en prosa en el informe.")
    print("Las graficas solas no son el entregable: la interpretacion si.\n")


if __name__ == "__main__":
    main()
