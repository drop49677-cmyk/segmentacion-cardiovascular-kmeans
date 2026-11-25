# Segmentación de Pacientes Mediante Técnicas de Clustering Basadas en Indicadores Clínicos Cardiovasculares

Proyecto de aula de la asignatura **Minería de Datos** (Departamento de Ciencias de la Computación y Electrónica, Universidad de la Costa – CUC).

**Integrantes:** Fernando Barbosa, Daliany Rojas León, Kevin Muñoz, Jose Díaz.

## Descripción

Este repositorio contiene el notebook reproducible, el conjunto de datos (`Heart Disease UCI`, 303 registros iniciales / 302 registros únicos tras eliminar 1 duplicado), el *pipeline* serializado de scikit-learn y la aplicación web interactiva en **Streamlit**.

El *pipeline* definitivo integra:
- Preprocesamiento con `ColumnTransformer`: `StandardScaler` para variables numéricas (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`), `passthrough` para binarias (`sex`, `fbs`, `exang`) y `OneHotEncoder` para variables categóricas nominales (`cp`, `restecg`, `slope`, `thal`), generando 23 características transformadas.
- Agrupamiento no supervisado con `KMeans(n_clusters=2, n_init=10, random_state=42)`.

> **Aviso académico:** Esta aplicación tiene fines exclusivamente académicos y exploratorios. No constituye un diagnóstico médico ni sustituye la valoración de un profesional de la salud.

## Estructura del repositorio

```text
├── app.py                              # Aplicación web en Streamlit
├── Proyecto_Clustering_Final.ipynb     # Notebook final ejecutado y comentado
├── requirements.txt                    # Dependencias del proyecto
├── data/
│   └── heart.csv                       # Conjunto de datos Heart Disease UCI
├── models/
│   ├── clustering_pipeline.joblib      # Pipeline entrenado (ColumnTransformer + KMeans)
│   ├── pca_2d.joblib                   # Modelo PCA 2D ajustado sobre el espacio transformado
│   └── cluster_metadata.json           # Métricas, rangos y perfiles promedio por clúster
└── figures/                            # Gráficas generadas por el notebook
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para volver a ejecutar el notebook completo:

```bash
jupyter execute --inplace Proyecto_Clustering_Final.ipynb
```
