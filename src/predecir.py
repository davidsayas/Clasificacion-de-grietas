"""predecir.py — El sistema completo, de la foto al nivel de riesgo.

    foto ─► puerta "¿es concreto?" ─► clasificación (o mosaico si la foto es grande)
         ─► [si hay grieta] medir ancho ─► inclinación (con gravedad) ─► riesgo

Las piezas pesadas (red neuronal, puerta, U-Net) se INYECTAN como funciones, así
este archivo se puede probar entero con modelos falsos (tests/test_predecir.py) y
funciona igual con los reales.

  predecir(lote)    -> probabilidades de "con grieta"          (obligatorio)
  embeddings(lote)  -> vectores para la puerta                 (opcional)
  puerta            -> objeto PuertaConcreto ya ajustado       (opcional)
  segmentar(imagen) -> máscara 0/1 de la grieta                (opcional; si no, clásica)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ancho, config, inclinacion, mosaico, riesgo


class Sistema:
    def __init__(self, predecir, embeddings=None, puerta=None, segmentar=None,
                 umbral: float = config.UMBRAL_DECISION, lado: int = config.IMG,
                 banda_incierta: tuple = config.BANDA_INCIERTA):
        self.predecir, self.embeddings, self.puerta = predecir, embeddings, puerta
        self.banda = banda_incierta
        self.segmentar = segmentar or ancho.segmentar_clasico
        self.umbral, self.lado = umbral, lado

    def _vetar(self, lote: np.ndarray) -> np.ndarray:
        """True = la puerta dice que esa tesela NO es concreto."""
        return ~np.asarray(self.puerta.es_concreto(self.embeddings(lote)))

    @property
    def _hay_puerta(self) -> bool:
        return self.puerta is not None and self.embeddings is not None

    def analizar(self, imagen_rgb: np.ndarray, mm_por_px: float | None = None,
                 roll_deg: float | None = None, pitch_deg: float | None = None,
                 esquinas_ref=None, ref_mm: tuple | None = None, escalas=(1.0,),
                 elemento: str | None = None, usar_inclinacion: bool = True) -> dict:
        """Analiza una foto RGB (alto, ancho, 3) con valores 0-255.

        mm_por_px:    escala conocida (si no se rectifica)
        esquinas_ref: 4 esquinas (x,y) de un rectángulo de referencia en la foto, y
        ref_mm:       su tamaño real (ancho_mm, alto_mm): rectifica la perspectiva y fija la escala
        roll_deg / pitch_deg: del sensor de gravedad del celular
        elemento:     'muro' | 'columna' | 'viga' | 'placa' (lo indica quien toma la foto)
        """
        h, w = imagen_rgb.shape[:2]
        notas = []

        # --- 1) clasificación ---
        vetada = 0.0
        if max(h, w) > 1.5 * self.lado:
            r = mosaico.clasificar_mosaico(imagen_rgb, self.predecir, lado=self.lado, escalas=escalas,
                                           umbral=self.umbral,
                                           vetar=self._vetar if self._hay_puerta else None)
            prob, modo, cajas, mapa, vetada = r["prob_max"], "mosaico", r["cajas"], r["mapa"], r["fraccion_vetada"]
        else:
            chica = cv2.resize(imagen_rgb, (self.lado, self.lado))[None].astype(np.float32)
            prob = float(np.asarray(self.predecir(chica)).ravel()[0])
            if self._hay_puerta and self._vetar(chica)[0]:
                prob, vetada = 0.0, 1.0
            modo, cajas, mapa = "imagen", [], None

        # --- 2) ¿parece concreto? (fracción de teselas que pasaron la puerta) ---
        es_concreto = True
        if self._hay_puerta:
            es_concreto = (1.0 - vetada) >= 0.5
            if vetada > 0:
                notas.append(f"{100 * vetada:.0f} % de la foto no parece concreto.")
        else:
            notas.append("Sin puerta de concreto: no se puede descartar que la foto no sea concreto.")

        # --- 3) ancho (solo si hay grieta y es concreto) ---
        medicion, mm_usado, orient = None, mm_por_px, None
        if es_concreto and prob >= self.umbral:
            img_ancho = imagen_rgb
            if esquinas_ref is not None and ref_mm is not None:
                img_ancho, mm_usado = ancho.rectificar_con_referencia(imagen_rgb, esquinas_ref, *ref_mm)
            elif cajas and mm_por_px is None:
                notas.append("Sin referencia de escala: el ancho no se puede dar en mm.")
            if modo == "mosaico" and esquinas_ref is None and cajas:      # recortar a la zona sospechosa
                x0 = min(c[0] for c in cajas); y0 = min(c[1] for c in cajas)
                x1 = max(c[0] + c[2] for c in cajas); y1 = max(c[1] + c[2] for c in cajas)
                img_ancho = img_ancho[y0:min(y1, h), x0:min(x1, w)]
            mascara = self.segmentar(img_ancho)
            medicion = ancho.medir_ancho(mascara, mm_usado)
            orient = ancho.orientacion_grieta(mascara, roll_deg or 0.0)
            if mm_usado is None and medicion is not None:
                notas.append("Ancho solo en píxeles (falta la escala).")

        # --- 4) inclinación ---
        gris = cv2.cvtColor(imagen_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        incl = inclinacion.estimar_inclinacion(gris, devolver_mapa=True, rgb=imagen_rgb)
        incl_deg, incl_verificada = None, True
        if incl is not None and not usar_inclinacion:
            notas.append("La inclinación del muro NO se usó para el riesgo porque no se pidió medirla.")
        elif incl is not None:
            if roll_deg is not None:
                c = inclinacion.corregir_con_gravedad(incl["angulo_imagen_deg"], roll_deg, pitch_deg)
                notas.extend(c["avisos"])
                incl_deg = c["inclinacion_real_deg"] if c["confiable"] else None
            else:
                angulo = incl["angulo_imagen_deg"]
                if abs(angulo) >= riesgo.INCLINACION_SIN_VERIFICAR_MIN_DEG:
                    incl_deg, incl_verificada = angulo, False
                    notas.append(f"Inclinación medida en la imagen ({abs(angulo):.1f}°) SIN verificar: no se conoce el giro del "
                                 f"celular y el ángulo puede incluirlo. Confirmala con un nivel o una plomada.")
                else:
                    notas.append("Sin dato de giro del celular: la inclinación NO se usa (a ángulos chicos podría ser el "
                                 "giro de la mano y no el desplome del muro).")

        # --- 5) riesgo ---
        ancho_mm = medicion.get("ancho_p95_mm") if medicion else None
        ev = riesgo.evaluar_riesgo(prob, ancho_mm, incl_deg, es_concreto, self.umbral, banda_incierta=self.banda,
                                   orientacion=orient["categoria"] if orient else None, elemento=elemento,
                                   inclinacion_verificada=incl_verificada)
        return {"modo": modo, "prob_grieta": prob, "es_concreto": es_concreto, "medicion": medicion,
                "inclinacion_deg": incl_deg, "inclinacion_verificada": incl_verificada, "inclinacion_imagen_deg": (incl or {}).get("angulo_imagen_deg"), "mapa_inclinacion": (incl or {}).get("mapa_bordes"),
                "inclinacion_pedida": usar_inclinacion, "inclinacion_piel": (incl or {}).get("piel_excluida"),
                "inclinacion_confianza": (incl or {}).get("confianza"), "inclinacion_metodo": (incl or {}).get("metodo"), "orientacion": orient, "riesgo": ev, "mapa": mapa, "cajas": cajas, "notas": notas}
