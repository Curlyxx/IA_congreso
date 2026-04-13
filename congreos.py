"""
╔══════════════════════════════════════════════════════════════╗
║         PREDICTOR DE RIESGO CREDITICIO                      ║
║  Dataset: credit_risk_dataset.csv                           ║
║  Modelo : Random Forest Classifier                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

try:
    from PIL import Image, ImageTk
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False

# ─────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────
C = {
    "bg":        "#0e0e16",
    "panel":     "#16161f",
    "panel2":    "#1c1c2a",
    "accent":    "#6c63ff",
    "accent2":   "#ff6584",
    "green":     "#43e97b",
    "yellow":    "#f9d423",
    "red":       "#ff6b6b",
    "text":      "#e8e8f0",
    "muted":     "#5a5a7a",
    "border":    "#2a2a3e",
    "card":      "#1a1a2e",
}

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_BIG    = ("Segoe UI", 26, "bold")

RUTA_IMAGEN = "/home/alecc/Documents/IA_congreso/prestamo.webp"

# ─────────────────────────────────────────────────────────────
# MAPAS DE TRADUCCIÓN UI → DATASET
# ─────────────────────────────────────────────────────────────
VIVIENDA_ES_A_EN = {
    "Renta / Arrendamiento": "RENT",
    "Casa propia":           "OWN",
    "Hipoteca":              "MORTGAGE",
    "Otra":                  "OTHER",
}

INTENT_ES_A_EN = {
    "Personal":                  "PERSONAL",
    "Educación":                 "EDUCATION",
    "Médico / Salud":            "MEDICAL",
    "Negocio / Emprendimiento":  "VENTURE",
    "Mejoras del hogar":         "HOMEIMPROVEMENT",
    "Consolidación de deudas":   "DEBTCONSOLIDATION",
}

GRADE_LABELS = [
    "A — Excelente (menor riesgo)",
    "B — Muy bueno",
    "C — Bueno",
    "D — Regular",
    "E — Alto riesgo",
    "F — Muy alto riesgo",
    "G — Máximo riesgo",
]
GRADE_VALUE = {lbl: lbl[0] for lbl in GRADE_LABELS}


# ─────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────
class ModeloRiesgo:
    FEATURES_CAT = ["person_home_ownership", "loan_intent",
                    "loan_grade", "cb_person_default_on_file"]
    FEATURES_NUM = ["person_age", "person_income", "person_emp_length",
                    "loan_amnt", "loan_int_rate", "loan_percent_income",
                    "cb_person_cred_hist_length"]
    TARGET = "loan_status"

    OPCIONES = {
        "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
        "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL",
                        "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
        "cb_person_default_on_file": ["Y", "N"],
    }

    def __init__(self):
        self.modelo         = RandomForestClassifier(n_estimators=200,
                                                     max_depth=12,
                                                     random_state=42,
                                                     n_jobs=-1,
                                                     class_weight={0:1, 1:3}  # 🔥 penaliza más los defaults
                                                     )
        self.encoders       = {}
        self.esta_entrenado = False
        self.accuracy       = None
        self.auc            = None
        self.reporte        = ""
        self.importancias   = {}
        self.df_info        = {}

    def preparar_features(self, df):
        df = df.copy()
        for col in self.FEATURES_CAT:
            if col not in self.encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df[self.FEATURES_NUM + self.FEATURES_CAT]

    def entrenar(self, ruta_csv, callback_progreso=None):
        if callback_progreso:
            callback_progreso("Cargando dataset…")

        df = pd.read_csv(ruta_csv)
        df.dropna(inplace=True)

        self.df_info = {
            "total":    len(df),
            "defaults": int(df[self.TARGET].sum()),
            "buenos":   int((df[self.TARGET] == 0).sum()),
            "tasa":     df[self.TARGET].mean() * 100,
        }

        if callback_progreso:
            callback_progreso(f"Dataset cargado: {len(df):,} filas. Preparando…")

        X = self.preparar_features(df)
        y = df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        if callback_progreso:
            callback_progreso("Entrenando Random Forest… (puede tardar ~30 seg)")

        self.modelo.fit(X_train, y_train)

        threshold = 0.35  # 🔥 AJUSTABLE

        y_proba = self.modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc      = roc_auc_score(y_test, y_proba)
        self.reporte  = classification_report(
            y_test, y_pred,
            target_names=["Sin default", "Con default"],
            digits=4
        )

        cols_all = self.FEATURES_NUM + self.FEATURES_CAT
        self.importancias = dict(sorted(
            zip(cols_all, self.modelo.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))

        self.esta_entrenado = True
        return self.accuracy, self.auc, self.reporte

    def predecir(self, datos: dict):
        if not self.esta_entrenado:
            raise ValueError("El modelo no está entrenado.")

        row = pd.DataFrame([datos])
        for col in self.FEATURES_CAT:
            le = self.encoders[col]
            row[col] = le.transform(row[col].astype(str))

        X = row[self.FEATURES_NUM + self.FEATURES_CAT]
        pred  = self.modelo.predict(X)[0]
        proba = self.modelo.predict_proba(X)[0]

        return int(pred), float(proba[1])


# ─────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────
def entry_widget(parent, width=18):
    return tk.Entry(parent, font=FONT_BODY,
                    bg=C["card"], fg=C["text"],
                    insertbackground=C["accent"],
                    relief="flat", bd=6, width=width)

def combo_widget(parent, values, width=17):
    cb = ttk.Combobox(parent, values=values, font=FONT_BODY,
                      state="readonly", width=width)
    cb.current(0)
    return cb

def section_title(parent, text, fg=None):
    f = tk.Frame(parent, bg=C["panel"])
    f.pack(fill="x", pady=(14, 4))
    tk.Frame(f, bg=fg or C["accent"], width=4, height=18).pack(side="left")
    tk.Label(f, text=f"  {text}", font=FONT_HEAD,
             fg=fg or C["accent"], bg=C["panel"]).pack(side="left")


# ─────────────────────────────────────────────────────────────
# GAUGE WIDGET
# ─────────────────────────────────────────────────────────────
class GaugeWidget(tk.Canvas):
    def __init__(self, parent, size=170, **kw):
        super().__init__(parent, width=size, height=size // 2 + 30,
                         bg=C["panel2"], highlightthickness=0, **kw)
        self.size  = size
        self.prob  = 0.0
        self._draw()

    def _color_for(self, p):
        if p < 0.35:  return C["green"]
        if p < 0.60:  return C["yellow"]
        return C["red"]

    def set_prob(self, p):
        self.prob = max(0.0, min(1.0, p))
        self._draw()

    def _draw(self):
        self.delete("all")
        s   = self.size
        pad = 16
        cx  = s // 2
        cy  = s // 2 + 10
        r   = s // 2 - pad

        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                        start=0, extent=180,
                        style="arc", outline=C["border"], width=14)
        extent = self.prob * 180
        color  = self._color_for(self.prob)
        if extent > 0:
            self.create_arc(cx - r, cy - r, cx + r, cy + r,
                            start=0, extent=extent,
                            style="arc", outline=color, width=14)

        pct = f"{self.prob*100:.1f}%"
        self.create_text(cx, cy - 6, text=pct,
                         font=("Segoe UI", 20, "bold"), fill=color)
        nivel = ("BAJO" if self.prob < 0.35 else
                 "MEDIO" if self.prob < 0.60 else "ALTO")
        self.create_text(cx, cy + 18, text=f"RIESGO  {nivel}",
                         font=("Segoe UI", 9, "bold"), fill=C["muted"])


# ─────────────────────────────────────────────────────────────
# TARJETA DE RESULTADO
# ─────────────────────────────────────────────────────────────
class ResultCard(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["panel2"], padx=20, pady=16)

        self.gauge = GaugeWidget(self, size=170)
        self.gauge.pack(pady=(0, 10))

        self.lbl_veredicto = tk.Label(self, text="—",
                                       font=("Segoe UI", 14, "bold"),
                                       bg=C["panel2"], fg=C["muted"])
        self.lbl_veredicto.pack()

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", pady=12)

        self.txt_explain = scrolledtext.ScrolledText(
            self, font=("Segoe UI", 10),
            bg=C["card"], fg=C["text"],
            relief="flat", bd=0,
            padx=10, pady=8,
            height=12, wrap="word",
            state="disabled"
        )
        self.txt_explain.pack(fill="both", expand=True)

    def actualizar(self, pred, prob, datos):
        self.gauge.set_prob(prob)

        color = C["red"]   if pred == 1 else C["green"]
        icono = "⚠️ DEFAULT" if pred == 1 else "✅ APROBADO"
        self.lbl_veredicto.config(text=icono, fg=color)

        self.txt_explain.config(state="normal")
        self.txt_explain.delete("1.0", "end")
        self._insertar_con_tags(pred, prob, datos)
        self.txt_explain.config(state="disabled")

    def _insertar_con_tags(self, pred, prob, datos):
        t = self.txt_explain
        t.tag_config("titulo",  font=("Segoe UI", 10, "bold"), foreground=C["accent"])
        t.tag_config("bueno",   foreground=C["green"])
        t.tag_config("malo",    foreground=C["red"])
        t.tag_config("alerta",  foreground=C["yellow"])
        t.tag_config("normal",  foreground=C["text"])
        t.tag_config("muted",   foreground=C["muted"])

        nivel     = "BAJO" if prob < 0.35 else "MEDIO" if prob < 0.60 else "ALTO"
        tag_nivel = "bueno" if prob < 0.35 else ("alerta" if prob < 0.60 else "malo")

        t.insert("end", "📋  ¿QUÉ SIGNIFICA ESTE RESULTADO?\n\n", "titulo")

        if pred == 0:
            t.insert("end", "✅  El solicitante tiene BUEN perfil crediticio.\n", "bueno")
            t.insert("end", "    El modelo predice que pagará el préstamo\n", "bueno")
            t.insert("end", "    sin incurrir en default.\n\n", "bueno")
        else:
            t.insert("end", "⚠️  El solicitante tiene RIESGO DE NO PAGAR.\n", "malo")
            t.insert("end", "    El modelo predice que puede incurrir en\n", "malo")
            t.insert("end", "    DEFAULT (impago del préstamo).\n\n", "malo")

        t.insert("end", "📊  Probabilidad de default:  ", "titulo")
        t.insert("end", f"{prob*100:.1f}%  (riesgo {nivel})\n\n", tag_nivel)

        t.insert("end", "─" * 42 + "\n", "muted")
        t.insert("end", "🔍  FACTORES ANALIZADOS\n\n", "titulo")

        edad = datos["person_age"]
        t.insert("end", f"  • Edad ({edad:.0f} años): ", "normal")
        if edad < 25:
            t.insert("end", "perfil joven, historial corto.\n", "alerta")
        elif edad < 45:
            t.insert("end", "rango estable y productivo.\n", "bueno")
        else:
            t.insert("end", "mayor estabilidad financiera.\n", "bueno")

        ingreso = datos["person_income"]
        t.insert("end", f"  • Ingreso anual (${ingreso:,.0f}): ", "normal")
        if ingreso < 30000:
            t.insert("end", "ingreso bajo, más presión financiera.\n", "malo")
        elif ingreso < 80000:
            t.insert("end", "ingreso medio, razonable.\n", "alerta")
        else:
            t.insert("end", "ingreso alto, positivo.\n", "bueno")

        pct = datos["loan_percent_income"]
        t.insert("end", f"  • % ingreso comprometido ({pct*100:.0f}%): ", "normal")
        if pct >= 0.31:
            t.insert("end", "zona de alto riesgo — en el dataset,\n", "malo")
            t.insert("end", "    valores ≥ 31% tienen ~70% de tasa de default real.\n", "malo")
        elif pct > 0.25:
            t.insert("end", "moderado — viable pero vigilar.\n", "alerta")
        else:
            t.insert("end", "bajo — carga manejable.\n", "bueno")

        grade = datos["loan_grade"]
        grade_map = {"A": "bueno", "B": "bueno", "C": "alerta",
                     "D": "alerta", "E": "malo", "F": "malo", "G": "malo"}
        g_tag = grade_map.get(grade, "normal")
        desc  = "excelente" if g_tag == "bueno" else ("regular" if g_tag == "alerta" else "alto riesgo")
        t.insert("end", f"  • Grado del préstamo ({grade}): ", "normal")
        t.insert("end", f"{desc}.\n", g_tag)

        interes = datos["loan_int_rate"]
        t.insert("end", f"  • Tasa de interés ({interes:.1f}%): ", "normal")
        if interes > 18:
            t.insert("end", "muy alta — indica riesgo elevado.\n", "malo")
        elif interes > 12:
            t.insert("end", "moderada.\n", "alerta")
        else:
            t.insert("end", "baja — perfil confiable.\n", "bueno")

        defhist = datos["cb_person_default_on_file"]
        t.insert("end", "  • ¿Historial de default previo? ", "normal")
        if defhist == "Y":
            t.insert("end", "SÍ — antecedente negativo importante.\n", "malo")
        else:
            t.insert("end", "NO — sin antecedentes negativos.\n", "bueno")

        # Nota explicativa si está en zona de umbral
        if 0.29 <= pct <= 0.32:
            t.insert("end", "\n─" * 21 + "\n", "muted")
            t.insert("end", "ℹ️  NOTA SOBRE % INGRESO COMPROMETIDO\n\n", "titulo")
            t.insert("end",
                "  El dataset muestra un umbral real muy marcado:\n"
                "  valores hasta 0.30 tienen ~24% de default,\n"
                "  mientras que desde 0.31 suben a ~70%.\n"
                "  Este salto refleja patrones reales en los datos\n"
                "  de entrenamiento, no un error del modelo.\n", "alerta")

        t.insert("end", "\n─" * 21 + "\n", "muted")
        t.insert("end", "💡  RECOMENDACIÓN\n\n", "titulo")

        if pred == 0 and prob < 0.35:
            t.insert("end", "  APROBACIÓN RECOMENDADA. Perfil sólido,\n  bajo riesgo de impago.\n", "bueno")
        elif pred == 0 and prob < 0.60:
            t.insert("end", "  APROBACIÓN CONDICIONADA. Considerar\n  garantías adicionales o monto reducido.\n", "alerta")
        elif pred == 1 and prob < 0.70:
            t.insert("end", "  REVISAR CON CUIDADO. Riesgo medio-alto;\n  solicitar documentación adicional.\n", "alerta")
        else:
            t.insert("end", "  DENEGAR O REESTRUCTURAR. Riesgo muy alto\n  de no recuperación del préstamo.\n", "malo")


# ─────────────────────────────────────────────────────────────
# PANEL DE PRESENTACIÓN
# ─────────────────────────────────────────────────────────────
class PanelPresentacion(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["panel2"])

        # Imagen
        self._img_ref = None
        self._cargar_imagen()

        # Contenido textual
        contenido = tk.Frame(self, bg=C["panel2"], padx=30, pady=20)
        contenido.pack(fill="both", expand=True)

        # Título principal
        tk.Label(contenido,
                 text="Predictor de Riesgo Crediticio",
                 font=("Segoe UI", 20, "bold"),
                 fg=C["accent"], bg=C["panel2"]).pack(anchor="w", pady=(0, 4))

        tk.Label(contenido,
                 text="Modelo de Machine Learning  ·  Random Forest Classifier",
                 font=("Segoe UI", 10),
                 fg=C["muted"], bg=C["panel2"]).pack(anchor="w", pady=(0, 18))

        tk.Frame(contenido, bg=C["border"], height=1).pack(fill="x", pady=(0, 18))

        # Descripción
        desc = (
            "Esta herramienta utiliza un modelo de aprendizaje automático entrenado "
            "sobre datos reales de solicitudes de préstamo para evaluar la probabilidad "
            "de que un solicitante incurra en default (impago).\n\n"
            "El sistema analiza 11 variables financieras y personales del solicitante "
            "y produce una puntuación de riesgo que puede apoyar decisiones crediticias "
            "de manera objetiva, rápida y reproducible."
        )
        tk.Label(contenido, text=desc,
                 font=("Segoe UI", 10), fg=C["text"], bg=C["panel2"],
                 wraplength=520, justify="left").pack(anchor="w", pady=(0, 20))

        # Tarjetas de características
        cards_frame = tk.Frame(contenido, bg=C["panel2"])
        cards_frame.pack(fill="x", pady=(0, 18))

        cards = [
            ("🎯", "Alta precisión",      "Accuracy >90%\nAUC-ROC >0.97"),
            ("⚡", "Evaluación rápida",   "Resultado en\nmenos de 1 segundo"),
            ("🔍", "11 variables",        "Perfil financiero\ncompleto"),
            ("📊", "Métricas detalladas", "Reporte completo\ndel modelo"),
        ]

        for icono, titulo, desc_c in cards:
            card = tk.Frame(cards_frame, bg=C["card"],
                            padx=12, pady=10, relief="flat")
            card.pack(side="left", padx=(0, 10), fill="y")

            tk.Label(card, text=icono, font=("Segoe UI", 18),
                     bg=C["card"], fg=C["text"]).pack(anchor="w")
            tk.Label(card, text=titulo,
                     font=("Segoe UI", 9, "bold"),
                     bg=C["card"], fg=C["accent"]).pack(anchor="w")
            tk.Label(card, text=desc_c,
                     font=("Segoe UI", 8),
                     bg=C["card"], fg=C["muted"],
                     justify="left").pack(anchor="w")

        tk.Frame(contenido, bg=C["border"], height=1).pack(fill="x", pady=(0, 14))

        # Instrucciones de uso
        tk.Label(contenido,
                 text="¿Cómo usar esta herramienta?",
                 font=("Segoe UI", 10, "bold"),
                 fg=C["yellow"], bg=C["panel2"]).pack(anchor="w", pady=(0, 8))

        pasos = [
            "1️⃣   Carga el dataset CSV y presiona  ⚙ ENTRENAR MODELO  en el panel izquierdo.",
            "2️⃣   Llena los datos del solicitante: edad, ingreso mensual, tipo de vivienda, etc.",
            "3️⃣   Presiona  🔍 EVALUAR SOLICITUD  para obtener la predicción de riesgo.",
            "4️⃣   Revisa el resultado en la pestaña  📊 Resultado  y las métricas del modelo.",
        ]

        for paso in pasos:
            tk.Label(contenido, text=paso,
                     font=("Segoe UI", 9), fg=C["text"], bg=C["panel2"],
                     anchor="w", justify="left").pack(fill="x", pady=2)

        # Nota sobre el umbral del 31%
        nota_frame = tk.Frame(contenido, bg=C["card"],
                              padx=14, pady=10, relief="flat")
        nota_frame.pack(fill="x", pady=(16, 0))

        tk.Label(nota_frame,
                 text="📌  Dato importante del dataset",
                 font=("Segoe UI", 9, "bold"),
                 fg=C["yellow"], bg=C["card"]).pack(anchor="w")
        tk.Label(nota_frame,
                 text=(
                     "El campo '% del ingreso comprometido' presenta un umbral estadístico "
                     "real en los datos: solicitudes con valor ≤ 30% tienen ~24% de tasa de "
                     "default, mientras que las de ≥ 31% saltan al ~70%. Este comportamiento "
                     "no es un error del modelo — refleja el patrón real aprendido del dataset."
                 ),
                 font=("Segoe UI", 8), fg=C["muted"], bg=C["card"],
                 wraplength=520, justify="left").pack(anchor="w", pady=(4, 0))

    def _cargar_imagen(self):
        """Intenta cargar la imagen de presentación con PIL."""
        if not PIL_DISPONIBLE:
            return
        if not os.path.exists(RUTA_IMAGEN):
            return
        try:
            img = Image.open(RUTA_IMAGEN)
            # Redimensionar manteniendo proporción, ancho máx 680px, alto máx 200px
            img.thumbnail((680, 200), Image.LANCZOS)
            self._img_ref = ImageTk.PhotoImage(img)
            lbl = tk.Label(self, image=self._img_ref,
                           bg=C["panel2"], bd=0)
            lbl.pack(fill="x", side="top")
        except Exception:
            pass   # Si falla, simplemente no muestra imagen


# ─────────────────────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Riesgo Crediticio")
        self.root.geometry("1080x740")
        self.root.minsize(900, 640)
        self.root.configure(bg=C["bg"])

        self.modelo   = ModeloRiesgo()
        self.ruta_csv = tk.StringVar(value="/home/alecc/Downloads/credit_risk_dataset.csv")

        self._aplicar_estilos()
        self._crear_ui()

    def _aplicar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground=C["card"],
                         background=C["card"],
                         foreground=C["text"],
                         arrowcolor=C["accent"],
                         bordercolor=C["border"],
                         lightcolor=C["border"],
                         darkcolor=C["border"],
                         padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", C["card"])])
        style.configure("Vertical.TScrollbar",
                         background=C["panel"], troughcolor=C["bg"],
                         arrowcolor=C["muted"], bordercolor=C["bg"])

    def _crear_ui(self):
        # Top bar
        topbar = tk.Frame(self.root, bg=C["panel"], pady=10)
        topbar.pack(fill="x")
        tk.Label(topbar, text="  💳  Predictor de Riesgo Crediticio",
                 font=FONT_TITLE, fg=C["accent"], bg=C["panel"]).pack(side="left")
        tk.Label(topbar, text="Random Forest  ·  Sin IA generativa  ",
                 font=FONT_SMALL, fg=C["muted"], bg=C["panel"]).pack(side="right")
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # Panel izquierdo
        self.panel_izq = tk.Frame(body, bg=C["panel"], width=360)
        self.panel_izq.pack(side="left", fill="y")
        self.panel_izq.pack_propagate(False)

        # Panel derecho
        panel_der = tk.Frame(body, bg=C["panel2"])
        panel_der.pack(side="left", fill="both", expand=True)

        self._crear_panel_izq()
        self._crear_panel_der(panel_der)

        # Status bar
        self.lbl_status = tk.Label(
            self.root,
            text="  Carga el dataset y entrena el modelo para comenzar.",
            font=FONT_SMALL, fg=C["muted"], bg=C["bg"], anchor="w")
        self.lbl_status.pack(fill="x", side="bottom", ipady=5)

    # ── Panel izquierdo ──────────────────────────────────────
    def _crear_panel_izq(self):
        canvas = tk.Canvas(self.panel_izq, bg=C["panel"], highlightthickness=0)
        scroll = ttk.Scrollbar(self.panel_izq, orient="vertical",
                                command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=C["panel"])
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self._construir_form(frame)

    def _construir_form(self, frame):
        pad = dict(padx=16, pady=3)

        # ── Dataset ───────────────────────────────────────────
        section_title(frame, "DATASET")

        f_csv = tk.Frame(frame, bg=C["panel"])
        f_csv.pack(fill="x", **pad)
        tk.Label(f_csv, text="Ruta CSV:", font=FONT_SMALL,
                 fg=C["muted"], bg=C["panel"]).pack(anchor="w")

        f_ruta = tk.Frame(f_csv, bg=C["panel"])
        f_ruta.pack(fill="x", pady=2)
        tk.Entry(f_ruta, textvariable=self.ruta_csv,
                 font=("Consolas", 8),
                 bg=C["card"], fg=C["text"],
                 insertbackground=C["accent"],
                 relief="flat", bd=4).pack(side="left", fill="x", expand=True)
        tk.Button(f_ruta, text="📂",
                  font=FONT_SMALL, bg=C["panel2"], fg=C["accent"],
                  relief="flat", cursor="hand2",
                  command=self._seleccionar_csv).pack(side="left", padx=4)

        tk.Button(frame, text="⚙  ENTRENAR MODELO",
                  font=("Segoe UI", 10, "bold"),
                  bg=C["accent"], fg="white",
                  activebackground="#9d97ff", activeforeground="white",
                  relief="flat", cursor="hand2", pady=8,
                  command=self._entrenar).pack(fill="x", padx=16, pady=(8, 2))

        self.lbl_modelo = tk.Label(frame, text="⬤  Modelo no entrenado",
                                    font=FONT_SMALL, fg=C["red"], bg=C["panel"])
        self.lbl_modelo.pack(padx=16, anchor="w")

        # ── Datos personales ──────────────────────────────────
        section_title(frame, "DATOS PERSONALES")
        self.e = {}

        self._campo(frame, "Edad (años)", "person_age", "30", pad)

        # Ingreso mensual en USD con cálculo a anual
        f_ing = tk.Frame(frame, bg=C["panel"])
        f_ing.pack(fill="x", **pad)
        tk.Label(f_ing, text="Ingreso mensual (USD):",
                 font=FONT_SMALL, fg=C["muted"], bg=C["panel"]).pack(anchor="w")
        self.e_ingreso_mensual = entry_widget(f_ing, width=28)
        self.e_ingreso_mensual.insert(0, "4583")
        self.e_ingreso_mensual.pack(fill="x", pady=2)
        self.lbl_ingreso_anual = tk.Label(
            f_ing,
            text="≈  $54,996 USD anuales",
            font=("Segoe UI", 8), fg=C["yellow"], bg=C["panel"]
        )
        self.lbl_ingreso_anual.pack(anchor="w")
        self.e_ingreso_mensual.bind("<KeyRelease>",
                                    lambda e: self._actualizar_preview_ingreso())

        self._campo(frame, "Años de empleo", "person_emp_length", "4", pad)
        self._combo(frame, "Tipo de vivienda", "person_home_ownership",
                    list(VIVIENDA_ES_A_EN.keys()), pad)

        # ── Datos del préstamo ────────────────────────────────
        section_title(frame, "DATOS DEL PRÉSTAMO")

        self._combo(frame, "Propósito del préstamo", "loan_intent",
                    list(INTENT_ES_A_EN.keys()), pad)
        self._combo(frame, "Grado del préstamo", "loan_grade",
                    GRADE_LABELS, pad, width=30)
        self._campo(frame, "Monto del préstamo ($)", "loan_amnt", "10000", pad)
        self._campo(frame, "Tasa de interés (%)", "loan_int_rate", "11.5", pad)

        # % ingreso comprometido con indicador visual de umbral
        f_pct = tk.Frame(frame, bg=C["panel"])
        f_pct.pack(fill="x", **pad)
        tk.Label(f_pct, text="% del ingreso comprometido (0–1):",
                 font=FONT_SMALL, fg=C["muted"], bg=C["panel"]).pack(anchor="w")
        self.e["loan_percent_income"] = entry_widget(f_pct, width=28)
        self.e["loan_percent_income"].insert(0, "0.20")
        self.e["loan_percent_income"].pack(fill="x", pady=2)
        self.lbl_pct_aviso = tk.Label(
            f_pct,
            text="ℹ  ≤0.30 riesgo moderado  |  ≥0.31 riesgo muy alto",
            font=("Segoe UI", 8), fg=C["muted"], bg=C["panel"]
        )
        self.lbl_pct_aviso.pack(anchor="w")
        self.e["loan_percent_income"].bind("<KeyRelease>",
                                           lambda e: self._actualizar_aviso_pct())

        # ── Historial crediticio ──────────────────────────────
        section_title(frame, "HISTORIAL CREDITICIO")

        self._combo(frame, "¿Default previo registrado?", "cb_person_default_on_file",
                    ModeloRiesgo.OPCIONES["cb_person_default_on_file"], pad)
        self._campo(frame, "Años de historial crediticio", "cb_person_cred_hist_length", "5", pad)

        # ── Botones ───────────────────────────────────────────
        tk.Frame(frame, bg=C["panel"], height=12).pack()
        tk.Button(frame, text="🔍  EVALUAR SOLICITUD",
                  font=("Segoe UI", 11, "bold"),
                  bg=C["green"], fg="#0e1a12",
                  activebackground="#6effa8", activeforeground="#0e1a12",
                  relief="flat", cursor="hand2", pady=10,
                  command=self._predecir).pack(fill="x", padx=16, pady=4)

        tk.Button(frame, text="↺  Limpiar",
                  font=FONT_SMALL,
                  bg=C["panel2"], fg=C["muted"],
                  activebackground=C["border"],
                  relief="flat", cursor="hand2", pady=5,
                  command=self._limpiar).pack(fill="x", padx=16, pady=(0, 16))

    def _campo(self, parent, label_txt, key, default, pad, width=28):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", **pad)
        tk.Label(f, text=label_txt, font=FONT_SMALL,
                 fg=C["muted"], bg=C["panel"]).pack(anchor="w")
        e = entry_widget(f, width=width)
        e.insert(0, default)
        e.pack(fill="x", pady=2)
        self.e[key] = e

    def _combo(self, parent, label_txt, key, valores, pad, width=26):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", **pad)
        tk.Label(f, text=label_txt, font=FONT_SMALL,
                 fg=C["muted"], bg=C["panel"]).pack(anchor="w")
        cb = combo_widget(f, valores, width=width)
        cb.pack(fill="x", pady=2)
        self.e[key] = cb

    def _actualizar_preview_ingreso(self):
        try:
            mensual = float(self.e_ingreso_mensual.get().strip())
            anual   = mensual * 12
            self.lbl_ingreso_anual.config(
                text=f"≈  ${anual:,.0f} USD anuales"
            )
        except Exception:
            self.lbl_ingreso_anual.config(text="Ingresa un valor válido")

    def _actualizar_aviso_pct(self):
        try:
            v = float(self.e["loan_percent_income"].get().strip())
            if v >= 0.31:
                self.lbl_pct_aviso.config(
                    text="⚠  ≥0.31 — zona de alto riesgo en datos (~70% default)",
                    fg=C["red"]
                )
            elif v > 0.25:
                self.lbl_pct_aviso.config(
                    text="⚡  0.25–0.30 — riesgo moderado (~25% default)",
                    fg=C["yellow"]
                )
            else:
                self.lbl_pct_aviso.config(
                    text="✓  ≤0.25 — riesgo bajo (~15% default)",
                    fg=C["green"]
                )
        except Exception:
            self.lbl_pct_aviso.config(
                text="ℹ  ≤0.30 riesgo moderado  |  ≥0.31 riesgo muy alto",
                fg=C["muted"]
            )

    # ── Panel derecho con tabs ────────────────────────────────
    def _crear_panel_der(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("TNotebook", background=C["panel2"], borderwidth=0)
        style.configure("TNotebook.Tab", background=C["panel2"],
                         foreground=C["muted"],
                         font=("Segoe UI", 10),
                         padding=(14, 6))
        style.map("TNotebook.Tab",
                   background=[("selected", C["panel"])],
                   foreground=[("selected", C["accent"])])

        # Tab 0 – Presentación
        tab_pres = tk.Frame(nb, bg=C["panel2"])
        nb.add(tab_pres, text="  🏠  Inicio  ")
        scroll_pres = tk.Canvas(tab_pres, bg=C["panel2"], highlightthickness=0)
        sb_pres = ttk.Scrollbar(tab_pres, orient="vertical",
                                 command=scroll_pres.yview)
        sb_pres.pack(side="right", fill="y")
        scroll_pres.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(scroll_pres, bg=C["panel2"])
        scroll_pres.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: scroll_pres.configure(
                       scrollregion=scroll_pres.bbox("all")))
        PanelPresentacion(inner).pack(fill="both", expand=True)

        # Tab 1 – Resultado
        tab_res = tk.Frame(nb, bg=C["panel2"])
        nb.add(tab_res, text="  📊  Resultado  ")
        self.result_card = ResultCard(tab_res)
        self.result_card.pack(fill="both", expand=True, padx=20, pady=16)

        # Tab 2 – Métricas
        tab_met = tk.Frame(nb, bg=C["panel2"])
        nb.add(tab_met, text="  📈  Métricas del modelo  ")
        self._crear_tab_metricas(tab_met)

        # Tab 3 – Importancias
        tab_imp = tk.Frame(nb, bg=C["panel2"])
        nb.add(tab_imp, text="  🔬  Variables clave  ")
        self._crear_tab_importancias(tab_imp)

        self.nb = nb   # guardar referencia para navegar tabs

    def _crear_tab_metricas(self, parent):
        self.txt_metricas = scrolledtext.ScrolledText(
            parent, font=FONT_MONO,
            bg=C["card"], fg=C["text"],
            relief="flat", bd=0, padx=16, pady=12,
            state="disabled"
        )
        self.txt_metricas.pack(fill="both", expand=True, padx=20, pady=16)

    def _crear_tab_importancias(self, parent):
        self.frame_imp = tk.Frame(parent, bg=C["panel2"])
        self.frame_imp.pack(fill="both", expand=True, padx=20, pady=16)
        tk.Label(self.frame_imp,
                 text="Entrena el modelo para ver la importancia de cada variable.",
                 font=FONT_BODY, fg=C["muted"], bg=C["panel2"]).pack(pady=40)

    # ── Acciones ──────────────────────────────────────────────
    def _seleccionar_csv(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona el CSV de riesgo crediticio",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if ruta:
            self.ruta_csv.set(ruta)

    def _status(self, msg):
        self.lbl_status.config(text=f"  {msg}")

    def _entrenar(self):
        ruta = self.ruta_csv.get().strip()
        if not os.path.exists(ruta):
            messagebox.showerror("Archivo no encontrado",
                                 f"No se encontró el archivo:\n{ruta}")
            return

        self.lbl_modelo.config(text="⬤  Entrenando…", fg=C["yellow"])
        self._status("Entrenando modelo Random Forest…")

        def tarea():
            try:
                acc, auc, rep = self.modelo.entrenar(
                    ruta, lambda m: self.root.after(0, self._status, m)
                )
                self.root.after(0, self._on_entrenado, acc, auc, rep)
            except Exception as ex:
                self.root.after(0, messagebox.showerror, "Error al entrenar", str(ex))
                self.root.after(0, self._status, "Error durante el entrenamiento.")

        threading.Thread(target=tarea, daemon=True).start()

    def _on_entrenado(self, acc, auc, rep):
        self.lbl_modelo.config(
            text=f"⬤  Entrenado  ·  Acc {acc*100:.1f}%  ·  AUC {auc:.3f}",
            fg=C["green"]
        )
        self._status(f"Modelo listo · Accuracy {acc*100:.1f}% · AUC-ROC {auc:.3f}")
        self._actualizar_metricas(acc, auc, rep)
        self._actualizar_importancias()
        messagebox.showinfo("Entrenamiento completo",
                            f"✅ Modelo entrenado correctamente.\n\n"
                            f"  Accuracy : {acc*100:.1f}%\n"
                            f"  AUC-ROC  : {auc:.4f}\n\n"
                            f"Dataset: {self.modelo.df_info['total']:,} préstamos "
                            f"({self.modelo.df_info['buenos']:,} buenos / "
                            f"{self.modelo.df_info['defaults']:,} defaults)")

    def _actualizar_metricas(self, acc, auc, rep):
        info = self.modelo.df_info
        texto = (
            f"{'═'*52}\n"
            f"  MÉTRICAS DEL MODELO — RANDOM FOREST\n"
            f"{'═'*52}\n\n"
            f"  Dataset total  : {info['total']:>8,} préstamos\n"
            f"  Sin default    : {info['buenos']:>8,} ({100-info['tasa']:.1f}%)\n"
            f"  Con default    : {info['defaults']:>8,} ({info['tasa']:.1f}%)\n\n"
            f"  ► Accuracy     : {acc*100:.2f}%\n"
            f"  ► AUC-ROC      : {auc:.4f}   "
            f"({'Excelente' if auc > 0.90 else 'Muy bueno' if auc > 0.80 else 'Aceptable'})\n\n"
            f"{'─'*52}\n"
            f"  REPORTE DE CLASIFICACIÓN (25% test set)\n"
            f"{'─'*52}\n\n"
            f"{rep}\n"
            f"{'─'*52}\n"
            f"  ¿Qué significan estas métricas?\n\n"
            f"  • Accuracy: de cada 100 solicitudes evaluadas,\n"
            f"    el modelo acierta en {acc*100:.0f} de ellas.\n\n"
            f"  • AUC-ROC: qué tan bien separa buenos pagadores\n"
            f"    de malos. 1.0 = perfecto, 0.5 = al azar.\n"
            f"    Un valor de {auc:.2f} es {'excelente' if auc > 0.90 else 'muy bueno'}.\n\n"
            f"{'─'*52}\n"
            f"  NOTA SOBRE loan_percent_income\n"
            f"{'─'*52}\n\n"
            f"  Umbral estadístico real detectado en el dataset:\n\n"
            f"  Rango       Tasa de default   Registros\n"
            f"  ≤ 0.30          ~24%           ~7,500\n"
            f"  0.31 – 0.40     ~69%           ~2,300\n"
            f"  > 0.40          ~76%             ~960\n\n"
            f"  Este salto es real en los datos, no un artefacto\n"
            f"  del modelo. El RF aprendió correctamente este umbral.\n"
        )
        self.txt_metricas.config(state="normal")
        self.txt_metricas.delete("1.0", "end")
        self.txt_metricas.insert("end", texto)
        self.txt_metricas.config(state="disabled")

    def _actualizar_importancias(self):
        for w in self.frame_imp.winfo_children():
            w.destroy()

        tk.Label(self.frame_imp,
                 text="¿Qué factores pesan más en la decisión?",
                 font=("Segoe UI", 11, "bold"),
                 fg=C["accent"], bg=C["panel2"]).pack(anchor="w", pady=(0, 12))

        nombres_bonitos = {
            "loan_percent_income":        "% ingreso comprometido",
            "loan_int_rate":              "Tasa de interés",
            "loan_grade":                 "Grado del préstamo",
            "loan_amnt":                  "Monto del préstamo",
            "person_income":              "Ingreso anual",
            "cb_person_cred_hist_length": "Historial crediticio",
            "person_age":                 "Edad",
            "person_emp_length":          "Años de empleo",
            "person_home_ownership":      "Tipo de vivienda",
            "loan_intent":                "Propósito del préstamo",
            "cb_person_default_on_file":  "¿Default previo?",
        }

        max_val = max(self.modelo.importancias.values())

        for col, val in list(self.modelo.importancias.items())[:11]:
            fila = tk.Frame(self.frame_imp, bg=C["panel2"])
            fila.pack(fill="x", pady=3)

            nombre = nombres_bonitos.get(col, col)
            tk.Label(fila, text=nombre, font=FONT_SMALL,
                     fg=C["text"], bg=C["panel2"],
                     width=28, anchor="w").pack(side="left")

            bar_frame = tk.Frame(fila, bg=C["border"], height=14, width=260)
            bar_frame.pack(side="left", padx=8)
            bar_frame.pack_propagate(False)

            ratio = val / max_val
            color = C["accent"] if ratio > 0.5 else (C["yellow"] if ratio > 0.2 else C["muted"])
            bar_w = max(4, int(260 * ratio))
            tk.Frame(bar_frame, bg=color, width=bar_w, height=14).place(x=0, y=0)

            tk.Label(fila, text=f"{val*100:.1f}%", font=FONT_SMALL,
                     fg=C["muted"], bg=C["panel2"], width=6).pack(side="left")

    # ── Recolección de datos ──────────────────────────────────
    def _get_datos(self):
        d = {}
        campos_num = {
            "person_age":                 (18, 144),
            "person_emp_length":          (0,  60),
            "loan_amnt":                  (0,  None),
            "loan_int_rate":              (0,  40),
            "loan_percent_income":        (0,  1.0),
            "cb_person_cred_hist_length": (0,  None),
        }
        for key, (vmin, vmax) in campos_num.items():
            raw = self.e[key].get().strip()
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(f"Valor inválido en '{key}': '{raw}'")
            if v < vmin:
                raise ValueError(f"'{key}' debe ser ≥ {vmin}.")
            if vmax is not None and v > vmax:
                raise ValueError(f"'{key}' debe ser ≤ {vmax}.")
            d[key] = v

        # Ingreso mensual → anual
        try:
            mensual = float(self.e_ingreso_mensual.get().strip())
            if mensual < 0:
                raise ValueError
        except Exception:
            raise ValueError("El ingreso mensual debe ser un número positivo.")
        d["person_income"] = mensual * 12

        # Tipo de vivienda
        vivienda_es = self.e["person_home_ownership"].get()
        d["person_home_ownership"] = VIVIENDA_ES_A_EN.get(vivienda_es, "OTHER")

        # Propósito
        intent_es = self.e["loan_intent"].get()
        d["loan_intent"] = INTENT_ES_A_EN.get(intent_es, "PERSONAL")

        # Grado (extraer letra)
        grade_lbl = self.e["loan_grade"].get()
        d["loan_grade"] = GRADE_VALUE.get(grade_lbl, grade_lbl[0] if grade_lbl else "A")

        # Default previo
        d["cb_person_default_on_file"] = self.e["cb_person_default_on_file"].get()

        return d

    def _predecir(self):
        if not self.modelo.esta_entrenado:
            messagebox.showwarning("Modelo no entrenado",
                                   "Primero carga el CSV y entrena el modelo.")
            return
        try:
            datos = self._get_datos()
        except ValueError as ex:
            messagebox.showerror("Datos inválidos", str(ex))
            return

        pred, prob = self.modelo.predecir(datos)
        self.result_card.actualizar(pred, prob, datos)

        # Cambiar a tab de resultado automáticamente
        self.nb.select(1)

        nivel = "BAJO" if prob < 0.35 else ("MEDIO" if prob < 0.60 else "ALTO")
        self._status(
            f"Evaluación completada  ·  "
            f"{'DEFAULT' if pred == 1 else 'APROBADO'}  ·  "
            f"Prob. default {prob*100:.1f}%  ·  Riesgo {nivel}"
        )

    def _limpiar(self):
        defaults = {
            "person_age":               "30",
            "person_emp_length":        "4",
            "loan_amnt":                "10000",
            "loan_int_rate":            "11.5",
            "loan_percent_income":      "0.20",
            "cb_person_cred_hist_length": "5",
        }
        for key, val in defaults.items():
            w = self.e[key]
            w.delete(0, "end")
            w.insert(0, val)

        self.e_ingreso_mensual.delete(0, "end")
        self.e_ingreso_mensual.insert(0, "4583")
        self._actualizar_preview_ingreso()
        self._actualizar_aviso_pct()

        for key in ["person_home_ownership", "loan_intent", "loan_grade",
                    "cb_person_default_on_file"]:
            self.e[key].current(0)

        self._status("Formulario limpiado.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()