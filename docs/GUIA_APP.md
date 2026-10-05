# Correr la app en tu computador

## Una sola vez (unos 10 minutos)

1. **Python:** instalá Python 3.11 desde <https://www.python.org/downloads/>. En Windows, marcá **"Add python.exe to PATH"**.
2. **El proyecto:** bajá de tu Drive la carpeta completa del proyecto a tu computador.
3. **El modelo:** comprobá que existe `models/v2_mobilenet/modelo.keras` y `models/v2_mobilenet/meta.json`.
   Si no, copiá esa carpeta desde tu Drive.

## Cada vez

* **Windows:** doble clic en `ejecutar_app.bat`.
* **Mac/Linux:** `./ejecutar_app.sh`.

La primera vez instala los paquetes (tarda varios minutos). Después se abre en el navegador, en `http://localhost:8501`.

## Si algo falla

| Mensaje | Qué hacer |
|---|---|
| "No encuentro Python" | Instalá Python 3.11 marcando "Add to PATH" |
| Falla la instalación de TensorFlow | Casi siempre es la versión de Python: usá la 3.10, 3.11 o 3.12 |
| "No encuentro ningún modelo" | Copiá `models/v2_mobilenet` desde Drive |
| "No pude cargar el modelo" | Mandá el texto del error completo |

## Cómo usarla en la demostración

1. Subí una foto de un muro **de frente**.
2. Elegí la **sensibilidad**. Con *Estándar* el modelo casi no detecta grietas de celular; con *Alta* detecta más a costa de
   falsas alarmas. Mostrá las dos y explicá por qué: es parte del análisis del proyecto.
3. Si pegaste una tarjeta de 85,6 mm junto a la grieta, marcá "Tengo un objeto de referencia" y poné cuánto mide en la foto.
4. El aviso amarillo de arriba es permanente a propósito: *no detectar una grieta no significa que no exista*.
