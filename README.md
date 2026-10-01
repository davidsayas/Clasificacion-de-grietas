# Detección de grietas, inclinación y riesgo en edificaciones

Reto de la mejor implementación · Algoritmos y Programación 2026-2 · Ingeniería en IA · UIS

Prototipo de extremo a extremo: **foto → ¿hay grieta? (probabilidad y orientación) → ¿está a plomo el elemento? (ángulo)
→ nivel de riesgo BAJO / MEDIO / ALTO con recomendación**. Corre en el computador (CLI y app web Streamlit) y el modelo
de transfer learning se exporta a TensorFlow Lite para el celular.

> Herramienta orientativa con fines académicos. No sustituye la inspección de un ingeniero civil.

## Estructura

```
src/
  make_synthetic_data.py  genera datos sintéticos para probar el pipeline sin el dataset
  dataset.py              carga, EDA y división train/val/test
  features.py             descriptores de la línea base (Black-hat, bordes, HOG)
  baseline.py             entrena la línea base (Reg. Logística) + métricas + complejidad
  train_transfer.py       MobileNetV2 con aumentación, fine-tuning y exportación a TFLite (Colab)
  inclination.py          ángulo de desaplome con Canny + Hough (o sensor del celular)
  risk.py                 orientación de la grieta y reglas de riesgo
  predict.py              CLI que une todo: foto -> JSON con grieta, inclinación y riesgo
  app_streamlit.py        demo web (subir foto / cámara)
tests/                    pruebas automáticas e imágenes de demostración
docs/                     SPRINTS.md (plan Scrum), entrega1_formulacion.md, demos
data/                     ver data/README.md (dataset + fotos propias)
models/                   modelos entrenados (los .keras/.tflite grandes van en Drive)
```

## Ejecución automática con el dataset real

```bash
python ejecutar_todo.py        # dataset -> EDA -> línea base -> predicción -> pruebas -> MobileNet (si existe) -> app
python app_celular.py          # la misma app, accesible desde el celular en la misma Wi-Fi
```

Documentación completa de lo hecho: **`docs/DOCUMENTACION.md`** · Ejecución en el celular: **`docs/EJECUCION_CELULAR.md`** ·
Entrenamiento de MobileNetV2 en Colab: **`entrenar_mobilenet_colab.ipynb`**.

## Inicio rápido (5 minutos, sin dataset)

```bash
pip install -r requirements.txt          # TensorFlow es opcional; solo lo necesita train_transfer.py
python src/make_synthetic_data.py        # 400 imágenes sintéticas en data/synthetic
python src/baseline.py --root data/synthetic --out models/baseline.joblib
python src/predict.py docs/demo/columna_grieta_diagonal_+2.5deg.jpg --elemento columna --debug docs/demo/salida.jpg
python src/inclination.py docs/demo/columna_+4.0deg.jpg --debug docs/demo/lineas.jpg
streamlit run src/app_streamlit.py
python tests/test_pipeline.py            # todas las pruebas deben decir OK
```

## Con el dataset real
1. Descargar *Concrete Crack Images* (ver `data/README.md`) en `data/raw/{Positive,Negative}`.
2. `python src/dataset.py --root data/raw` (EDA) y `python src/baseline.py --root data/raw --max-por-clase 4000`.
3. En Google Colab (GPU): `python src/train_transfer.py --root data/raw --max-por-clase 4000 --epochs 5`
   → `models/mobilenet.keras`, `models/mobilenet.tflite`, `models/mobilenet_metricas.json`, `models/curvas.png`.
4. `python src/predict.py foto.jpg --modelo models/mobilenet.tflite --elemento columna` o
   `streamlit run src/app_streamlit.py -- --modelo models/mobilenet.tflite`.

## Salida de ejemplo

```json
{"grieta": {"clase": "con grieta", "probabilidad": 1.0, "orientacion": "diagonal", "ms_inferencia": 5.5, "modelo": "baseline"},
 "inclinacion": {"angulo_desaplome": 1.97, "signo": "derecha", "categoria": "inclinado", "n_lineas": 3, "referencia": "vertical", "metodo": "hough"},
 "riesgo": {"nivel": "ALTO", "puntaje": 11,
            "recomendacion": "Restringir el uso del área y solicitar inspección URGENTE de un ingeniero civil...",
            "detalle": {"razones": ["grieta detectada con alta confianza", "orientación diagonal en columna",
                                    "grieta diagonal en elemento portante: posible falla por cortante", "desaplome apreciable (2.0°)"]}}}
```

## Plan de trabajo
Seis sprints de dos semanas alineados con los tres cortes (25/09, 23/10, 20/11): ver **`docs/SPRINTS.md`**.
El documento de la Entrega 1 está en `docs/entrega1_formulacion.md` (faltan solo los datos reales del equipo).
