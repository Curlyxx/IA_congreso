# Predict One — Predictor de Riesgo Crediticio

Sistema de predicción de riesgo crediticio basado en Machine Learning,
desarrollado en Python con una interfaz de escritorio construida en Tkinter.

## Descripción

Predict One analiza el perfil financiero de un solicitante de préstamo y
predice la probabilidad de que incurra en default (impago), apoyando
decisiones crediticias de forma objetiva y reproducible.

El modelo utilizado es un Random Forest Classifier entrenado sobre el
dataset público credit_risk_dataset.csv, con 28,638 registros válidos
de solicitudes reales de préstamo.

## Características

- Interfaz gráfica de escritorio con diseño oscuro profesional
- Formulario de captura con validación de datos en tiempo real
- Gauge visual de probabilidad de riesgo (bajo / medio / alto)
- Explicación detallada del resultado por variable
- 4 visualizaciones automáticas del modelo tras el entrenamiento
- Métricas completas: Accuracy, AUC-ROC y reporte de clasificación

## Métricas del modelo

| Métrica    | Valor  |
|------------|--------|
| Accuracy   | 92.35% |
| AUC-ROC    | 0.9254 |
| Registros  | 28,638 |
| Default    | 21.7%  |

## Requisitos

- Python 3.8 o superior
- Las siguientes librerías:

pip install -r requirements.txt

Contenido de requirements.txt:

numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow

## Cómo ejecutar

1. Clona el repositorio:
   git clone https://github.com/tu-usuario/predict-one.git
   cd predict-one

2. Instala las dependencias:
   pip install -r requirements.txt

3. Ejecuta la aplicación:
   python predictor.py

4. Dentro de la app:
   - Carga el archivo credit_risk_dataset.csv usando el botón de carpeta
   - Presiona ENTRENAR MODELO y espera ~30 segundos
   - Llena el formulario con los datos del solicitante
   - Presiona EVALUAR SOLICITUD para obtener el resultado

## Dataset

El dataset credit_risk_dataset.csv debe colocarse en la misma carpeta
que el archivo predictor.py, o bien seleccionarse manualmente desde
la interfaz.

Registros originales : 32,581
Registros tras limpieza : 28,638
Variables utilizadas : 11

## Variables del modelo

| Variable                    | Tipo        |
|-----------------------------|-------------|
| Edad                        | Numérica    |
| Ingreso anual               | Numérica    |
| Años de empleo              | Numérica    |
| Monto del préstamo          | Numérica    |
| Tasa de interés             | Numérica    |
| % ingreso comprometido      | Numérica    |
| Tipo de vivienda            | Categórica  |
| Propósito del préstamo      | Categórica  |
| Grado del préstamo          | Categórica  |
| Default previo registrado   | Categórica  |
| Años de historial crediticio| Numérica    |

## Estructura del proyecto

predict-one/
├── predictor.py
├── credit_risk_dataset.csv
├── requirements.txt
└── README.md

## Autores

Davila Mendez Harold Alexander
Gomez Solis Isaac Efrain
