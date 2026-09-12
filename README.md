<div align="center">

  # 🚀 Nombre de tu Proyecto

  **Una descripción corta, directa e impactante de lo que hace tu aplicación o biblioteca.**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](#)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  [📑 Documentación](#-visión-general) •
  [✨ Características](#-características-principales) •
  [🛠️ Instalación](#-guía-de-instalación) •
  [🚀 Uso](#-ejemplo-de-uso)

</div>
> ⚠️ **Aviso:** Herramienta orientativa con fines académicos. No sustituye la inspección de un ingeniero civil.

---

## Índice
1. [Qué hace el sistema](#1-qué-hace-el-sistema)
2. [Estado frente al plan de sprints](#2-estado-frente-al-plan-de-sprints)
3. [Entorno de trabajo](#3-entorno-de-trabajo)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Bitácora de problemas y correcciones](#5-bitácora-de-problemas-y-correcciones)
6. [Cómo ejecutar el proyecto en el PC](#6-cómo-ejecutar-el-proyecto-en-el-pc)
7. [Dataset](#7-dataset)
8. [Modelo 1: línea base](#8-modelo-1-línea-base)
9. [Modelo 2: MobileNetV2 con transfer learning](#9-modelo-2-mobilenetv2-con-transfer-learning)
10. [Comparación de los dos modelos](#10-comparación-de-los-dos-modelos)
11. [Módulos de visión clásica: orientación e inclinación](#11-módulos-de-visión-clásica-orientación-e-inclinación)
12. [Reglas de riesgo](#12-reglas-de-riesgo)
13. [Pruebas con fotos reales y hallazgos](#13-pruebas-con-fotos-reales-y-hallazgos)
14. [Limitaciones conocidas](#14-limitaciones-conocidas)
15. [Trabajo pendiente](#15-trabajo-pendiente)
16. [Anexo: comandos de referencia](#16-anexo-comandos-de-referencia)

---

## 1. Qué hace el sistema

Prototipo que apartir de una fotografia (de una edificación), se hace tres preguntas con las cuales determina el nivel de riesgo de aquella edificacion a nivel arquitectonico. ⚠️ **Aviso:** Esta aplicación, no es 100$ segura para su implementacion en la vida real.

```text
foto (jpg/png)
   ├─► ¿Hay grieta?          clasificador (línea base o MobileNetV2) ─► clase + probabilidad  Primera pregunta
   ├─► ¿Qué orientación?     Hough sobre máscara Black-hat            ─► vertical / horizontal / diagonal # Segunda pregunta
   ├─► ¿Está a plomo?        Canny + HoughLinesP (o inclinómetro)     ─► ángulo (°) + categoría # Tercera pregunta
   └─► Riesgo                reglas de ingeniería (risk.py)           ─► BAJO / MEDIO / ALTO + recomendación
```

## Explicacion de cada una de las preguntas:
1) ¿Hay grieta?
2) ¿Qué orientación?
3) ¿Está a plomo?


## Características: 
* **⚡ Alto Rendimiento:** Optimizado para procesar datos de forma rápida y eficiente.
* **🎯 Precisión Integrada:** Algoritmos diseñados para minimizar el margen de error.
* **📊 Interfaz Visual Clara:** Salidas estructuradas con gráficas e indicadores interactivos.
* **🔧 Modular y Extensible:** Arquitectura limpia diseñada para facilitar la adición de nuevos módulos.

---

## 🛠️ Guía de Instalación

Sigue estos pasos para clonar el repositorio y configurar el entorno de ejecución local.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
   cd tu-repositorio
