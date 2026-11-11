# Actividad 2.12 - Laboratorio: Implementación de Redes Neuronales (MLP + MLflow)

Este repositorio contiene la solución completa y la evidencia del laboratorio de clasificación supervisada utilizando un **Perceptrón Multicapa (Multi-Layer Perceptron - MLP)** sobre el conjunto de datos *Breast Cancer Wisconsin (Diagnostic)*, integrado con **MLflow** para el registro, trazabilidad y observabilidad del ciclo de vida de los experimentos de Machine Learning.

---

## 📌 Descripción del Proyecto

El objetivo principal de este laboratorio es implementar y evaluar un modelo de clasificación MLP con `scikit-learn`, registrar de forma sistemática cada experimento en un servidor de seguimiento local de **MLflow** (desplegado en Google Colab mediante un túnel público de `ngrok`), y analizar cómo impactan los siguientes factores en el rendimiento del modelo:

1. **Preprocesamiento de datos:** Comparación directa entre datos en su escala original vs. estandarizados (`StandardScaler` dentro de un `Pipeline`).
2. **Arquitectura de la red:** Variación en el número de capas ocultas y cantidad de neuronas `(30,)` vs. `(64, 32)`.
3. **Funciones de activación:** Evaluación de `relu` vs. `tanh`.
4. **Reproducibilidad y variabilidad de semillas:** Evaluación del efecto del estado pseudoaleatorio (`random_state`) en la partición y la inicialización de pesos del modelo.

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python 3.13+
* **Entorno de Ejecución:** Google Colab
* **ML & Preprocesamiento:** `scikit-learn` (1.6.1), `pandas` (2.2.3), `numpy`
* **Observabilidad y Registro:** `mlflow` (3.18.0)
* **Despliegue de Túneles:** `pyngrok`
* **Visualización:** `matplotlib`, `seaborn`
* **Fuente de Datos:** `kagglehub` (*uciml/breast-cancer-wisconsin-data*)

---

## 📂 Estructura del Repositorio

```text
.
├── IEI097_Lab2.12_MLP_MLflow.ipynb  # Notebook principal ejecutado en Google Colab
├── runs.csv                         # Exportación de las métricas y parámetros de MLflow
├── README.md                        # Documentación del proyecto
└── evidencias/                      # Capturas de pantalla de la interfaz gráfica de MLflow
    ├── 01_tabla_runs_ngrok.png      # Tabla global de corridas con URL de ngrok
    ├── 02_comparacion_train_loss.png# Gráfico de curvas de pérdida (train_loss)
    └── 03_artefacto_matriz_cm.png   # Artefacto registrado con la matriz de confusión# Laboratorio_2.12_Multi_Layer_Perceptron
