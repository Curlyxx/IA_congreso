# =============================================================
#  ANÁLISIS DE ENFERMEDADES - ÁRBOL DE DECISIÓN
#  Archivo: Pacientes2.csv
# =============================================================

# ── LIBRERÍAS ────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# ── 1. CARGA DEL ARCHIVO ─────────────────────────────────────
RUTA = "/home/alecc/Downloads/Pacientes2.csv"

df = pd.read_csv(RUTA)

# ── 2. PRIMERAS 5 FILAS ──────────────────────────────────────
print("=" * 60)
print("  PRIMERAS 5 FILAS")
print("=" * 60)
print(df.head())

# ── 3. INFORMACIÓN DEL DATASET ───────────────────────────────
print("\n" + "=" * 60)
print("  INFORMACIÓN DEL DATASET")
print("=" * 60)
print(df.info())

# ── 4. CONTEO DE CLASES SI / NO ──────────────────────────────
print("\n" + "=" * 60)
print("  CANTIDAD DE PACIENTES POR CLASE (Enfermedad)")
print("=" * 60)
print(df["Enfermedad"].value_counts())

# ── 5. COUNTPLOT DE LA VARIABLE ENFERMEDAD ───────────────────
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df,
    x="Enfermedad",
    palette={"SI": "#E74C3C", "NO": "#2ECC71"},
    order=["SI", "NO"]
)
ax.bar_label(ax.containers[0], fontsize=12, fontweight="bold")
plt.title("Distribución de la Variable Enfermedad", fontsize=14, fontweight="bold")
plt.xlabel("Enfermedad")
plt.ylabel("Cantidad de Pacientes")
plt.tight_layout()
plt.savefig("/home/alecc/Downloads/countplot_enfermedad.png", dpi=150)
plt.show()
print("\n  [Countplot guardado en: /home/alecc/Downloads/countplot_enfermedad.png]")

# ── 6. DESCRIPCIÓN ESTADÍSTICA ───────────────────────────────
print("\n" + "=" * 60)
print("  DESCRIPCIÓN ESTADÍSTICA")
print("=" * 60)
print(df.describe(include="all"))

# ── 7. REVISIÓN DE DATOS NULOS ───────────────────────────────
print("\n" + "=" * 60)
print("  DATOS NULOS POR COLUMNA")
print("=" * 60)
print(df.isnull().sum())

# ── 8. CODIFICACIÓN DE VARIABLES CATEGÓRICAS ─────────────────
# Se copian para no alterar el df original
df_model = df.copy()

le = LabelEncoder()

# Columnas categóricas (SI/NO y MASCULINO/FEMENINO, etc.)
cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ["NOEXPED", "Enfermedad"]]

for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Variable objetivo
df_model["Enfermedad"] = le.fit_transform(df_model["Enfermedad"].astype(str))
# Guardar mapeo: 0=NO, 1=SI (puede variar según LabelEncoder)
print("\n  Clases codificadas (Enfermedad):", dict(zip(le.classes_, le.transform(le.classes_))))

# ── 9. SEPARACIÓN X / Y ──────────────────────────────────────
X = df_model.drop(columns=["Enfermedad", "NOEXPED"])
Y = df_model["Enfermedad"]

print("\n" + "=" * 60)
print("  VARIABLES INDEPENDIENTES (X):", list(X.columns))
print("  VARIABLE OBJETIVO       (Y): Enfermedad")
print("=" * 60)

# ── 10. SPLIT ENTRENAMIENTO / PRUEBA (80/20) ─────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)
print(f"\n  Registros de entrenamiento : {len(X_train)}")
print(f"  Registros de prueba        : {len(X_test)}")

# ── 11. CREACIÓN DEL MODELO (máximo 4 niveles) ───────────────
modelo = DecisionTreeClassifier(
    max_depth=4,          # Solo 4 niveles
    criterion="gini",
    random_state=42
)

# ── 12. ENTRENAMIENTO ────────────────────────────────────────
modelo.fit(X_train, Y_train)
print("\n  Modelo entrenado correctamente ✓")

# ── 13. GRÁFICO DEL ÁRBOL DE DECISIÓN ───────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    modelo,
    feature_names=X.columns.tolist(),
    class_names=["NO", "SI"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False
)
plt.title("Árbol de Decisión – Predicción de Enfermedad (max_depth=4)",
          fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/alecc/Downloads/arbol_decision.png", dpi=150, bbox_inches="tight")
plt.show()
print("  [Árbol guardado en: /home/alecc/Downloads/arbol_decision.png]")

# Reglas en texto
print("\n  REGLAS DEL ÁRBOL (texto):")
print(export_text(modelo, feature_names=list(X.columns)))

# ── 14. PREDICCIONES ─────────────────────────────────────────
Y_pred = modelo.predict(X_test)

print("\n" + "=" * 60)
print("  PREDICCIONES vs VALORES REALES (primeros 15)")
print("=" * 60)
comparacion = pd.DataFrame({
    "Real"     : Y_test.values[:15],
    "Predicho" : Y_pred[:15]
})
comparacion["Correcto"] = comparacion["Real"] == comparacion["Predicho"]
print(comparacion.to_string(index=False))

# ── 15. MÉTRICAS ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ACCURACY DEL MODELO")
print("=" * 60)
acc = accuracy_score(Y_test, Y_pred)
print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

print("\n" + "=" * 60)
print("  REPORTE DE CLASIFICACIÓN")
print("=" * 60)
print(classification_report(Y_test, Y_pred, target_names=["NO", "SI"]))

# ── 16. MATRIZ DE CONFUSIÓN ──────────────────────────────────
cm = confusion_matrix(Y_test, Y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NO", "SI"])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
plt.title("Matriz de Confusión – Árbol de Decisión", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/alecc/Downloads/matriz_confusion.png", dpi=150)
plt.show()
print("  [Matriz guardada en: /home/alecc/Downloads/matriz_confusion.png]\n")

print("  VN (Verdaderos Negativos) :", cm[0, 0])
print("  FP (Falsos Positivos)     :", cm[0, 1])
print("  FN (Falsos Negativos)     :", cm[1, 0])
print("  VP (Verdaderos Positivos) :", cm[1, 1])

print("\n  ✅  Análisis completado.")