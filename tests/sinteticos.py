"""Imágenes sintéticas para probar el código SIN el dataset real."""
import cv2
import numpy as np


def textura_concreto(n=224, semilla=0, brillo=170, ruido=18, sesgo_color=(0, 0, 0)):
    """Fondo tipo concreto: ruido suavizado alrededor de un brillo medio."""
    rng = np.random.default_rng(semilla)
    base = rng.normal(brillo, ruido, (n, n)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    img = np.stack([base + s for s in sesgo_color[::-1]], axis=2) if any(sesgo_color) \
        else np.stack([base] * 3, axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def con_grieta(img, grosor=3, semilla=0, oscuridad=90):
    """Dibuja una grieta zigzagueante oscura de un lado al otro."""
    rng = np.random.default_rng(semilla + 1000)
    salida = img.copy()
    n = img.shape[0]
    x, y = int(rng.integers(0, n // 3)), 0
    valor = int(max(0, img.mean() - oscuridad))
    while y < n - 1:
        nx = int(np.clip(x + rng.integers(-12, 20), 0, n - 1))
        ny = min(n - 1, y + int(rng.integers(8, 20)))
        cv2.line(salida, (x, y), (nx, ny), (valor, valor, valor), grosor)
        x, y = nx, ny
    return salida


def guardar_imagenes(carpeta, n, **kw):
    from pathlib import Path
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(carpeta / f"{i + 1:05d}.jpg"), textura_concreto(semilla=i, **kw))
