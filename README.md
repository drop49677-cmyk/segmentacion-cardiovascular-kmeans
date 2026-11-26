# Segmentación de Pacientes Mediante Técnicas de Clustering Basadas en Indicadores Clínicos Cardiovasculares

Proyecto de aula de la asignatura **Minería de Datos** (Tercer Corte — Actividad 1: Integración y despliegue del modelo con Streamlit).  
**Institución:** Universidad de la Costa (CUC) — Departamento de Ciencias de la Computación y Electrónica.

**Integrantes:**
- Fernando Barbosa
- Daliany Rojas León
- Kevin Muñoz
- Jose Díaz

---

## 1. Descripción y Objetivo del Proyecto

Las enfermedades cardiovasculares presentan una alta heterogeneidad clínica entre pacientes. El objetivo de este proyecto es revisar el desempeño del modelo no supervisado desarrollado en el segundo corte sobre el conjunto de datos **Heart Disease UCI** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) / Kaggle), corregir las asunciones de preprocesamiento mediante un `Pipeline` único serializado con `joblib`, y desplegar una aplicación web interactiva en **Streamlit** que reproduzca fielmente las asignaciones del notebook.

> **Aviso académico:** Esta aplicación es un **prototipo académico exploratorio**. No realiza diagnósticos médicos ni sustituye el criterio de un profesional de la salud.

---

## 2. Revisión del Rendimiento y Pipeline Definitivo

En el segundo corte se aplicó `StandardScaler` sobre las 13 variables clínicas tratándolas como numéricas ordinales, obteniendo un coeficiente de silueta de `0.1681` (< `0.70`). Siguiendo la revisión técnica para métricas inferiores a `0.70`:

1. **Corrección del preprocesamiento:** Se implementó un `ColumnTransformer` que aplica:
   - `StandardScaler` a las 6 variables numéricas (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`);
   - `passthrough` a las 3 variables binarias (`sex`, `fbs`, `exang`);
   - `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` a las 4 variables categóricas nominales (`cp`, `restecg`, `slope`, `thal`), generando **23 características transformadas**.
2. **Modelo final:** `KMeans(n_clusters=2, n_init=10, random_state=42)` empaquetado junto al `ColumnTransformer` en `models/clustering_pipeline.joblib` (la aplicación carga el archivo serializado sin reentrenar).

### Comparación Corte 2 vs. Corte 3 y Alternativas ($k=2$, $n=302$ pacientes únicos)

| Configuración evaluada | Dim. | Silueta | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| Corte 2: `StandardScaler` (13 ordinales) + `KMeans` | 13 | 0.1681 | 2.1900 | 54.99 |
| **Corte 3 (Seleccionado): `StandardScaler` + `OneHotEncoder` + `KMeans`** | **23** | **0.1722** | **2.0650** | **66.61** |
| Alternativa: `RobustScaler` + `OneHotEncoder` + `KMeans` | 23 | 0.1727 | 2.1197 | 59.55 |
| Alternativa: `StandardScaler` + `OneHotEncoder` + `Ward` | 23 | 0.1019 | 2.2109 | 45.29 |
| Alternativa: `StandardScaler` + `OneHotEncoder` + `GMM` | 23 | 0.0791 | 3.7787 | 11.71 |

### Auditoría Técnica del Pipeline (Alerta < 0.70)
- **Fuga de información:** Nula (`target` excluido del entrenamiento, inferencia y artefactos).
- **Colinealidad:** Controlada (`|r| máx. = 0.40`, `VIF máx. = 1.407 < 5` en las 6 variables numéricas).
- **Validación fuera de muestra (5-fold) y estabilidad:** Silueta en entrenamiento `0.1705 ± 0.0051`, silueta en prueba `0.1687 ± 0.0215`, `ARI vs solución completa = 0.9733`, estabilidad multi-semilla `ARI = 1.0000` y convergencia idéntica de inercia (`2186.73`) para `n_init ∈ {10, 20, 50}`.
- **Balance de clústeres:** `137` vs `165` pacientes (razón `1.2:1`, equilibrado).

### Limitaciones del Modelo
- **Silueta moderada-baja (`0.1722 < 0.70`):** Los indicadores clínicos continuos presentan solapamiento fisiológico gradual; los clústeres representan perfiles estadísticos orientativos y no fronteras clínicas aisladas.
- **Ausencia de `predict_proba()` y matriz de confusión supervisada:** `KMeans` es un algoritmo no supervisado basado en distancias euclidianas; por ello no produce probabilidades calibradas ni debe evaluarse con *accuracy* o matriz de confusión. La columna `target` se excluyó del entrenamiento para evitar fuga de información y solo se muestra en la app como cruce descriptivo exploratorio.
- **Alcance muestral y sensibilidad a atípicos:** Cohorte de `302` pacientes únicos (tras eliminar `1` duplicado de los `303` registros iniciales) con predominio masculino (`68.2%`) y 20 valores atípicos IQR conservados por plausibilidad clínica.

---

## 3. Estructura del Repositorio

```text
├── app.py                              # Aplicación web interactiva en Streamlit
├── Proyecto_Clustering_Final.ipynb     # Notebook actualizado y ejecutado íntegramente
├── requirements.txt                    # Dependencias para ejecución local y Streamlit Cloud
├── README.md                           # Documentación del proyecto y evaluación de uso de IA
├── data/
│   └── heart.csv                       # Conjunto de datos Heart Disease UCI
├── models/
│   ├── clustering_pipeline.joblib      # Pipeline serializado (ColumnTransformer + KMeans)
│   ├── pca_2d.joblib                   # Modelo PCA 2D ajustado sobre el espacio transformado
│   └── cluster_metadata.json           # Métricas, rangos, perfiles y casos de prueba verificados
├── figures/                            # Visualizaciones EDA y de evaluación del modelo
│   ├── eda_distribuciones_clinicas.png
│   ├── eda_categoricas_clinicas.png
│   ├── correlacion_eda.png
│   ├── seleccion_k.png
│   ├── perfil_clusters_heatmap.png
│   └── pca_tsne_clusters.png
└── evidencias_pruebas/                 # Capturas de verificación automatizada en Playwright
```

---

## 4. Enlaces del Proyecto

- **Repositorio público en GitHub:** `https://github.com/drop49677-cmyk/segmentacion-cardiovascular-kmeans`
- **Aplicación pública en Streamlit Community Cloud:** `https://segmentacion-cardiovascular-kmeans-r8sd9m4yjfhvefwzfdjeef.streamlit.app/`

---

## 5. Ejecución Local y Verificación

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para reejecutar el notebook completo en orden:

```bash
jupyter execute --inplace Proyecto_Clustering_Final.ipynb
```

### Casos de Prueba Verificados (Notebook ↔ Streamlit)

| Caso de prueba | Clúster asignado | Distancia C0 | Distancia C1 | Estado |
|---|---|---:|---:|---|
| Caso normal (Perfil mayor carga) | Clúster 0 | 1.823 | 4.098 | Coincidencia 100% |
| Caso normal (Perfil menor esfuerzo) | Clúster 1 | 4.007 | 1.706 | Coincidencia 100% |
| Registro conocido: Paciente #0 (`heart.csv`) | Clúster 0 | 2.819 | 3.451 | Coincidencia 100% |
| Registro conocido: Paciente #165 (`heart.csv`) | Clúster 0 | 2.074 | 4.071 | Coincidencia 100% |
| Valores límite observados (`age=77, chol=564...`) | Clúster 0 | 9.277 | 11.249 | Advertencias de límite activas |
| Entrada inválida (`age=12, trestbps=280, chol=750`) | N/A | N/A | N/A | Bloqueado sin fallos con guía clara |
| Entrada incompleta (`age=None, chol=None`) | N/A | N/A | N/A | Bloqueado sin fallos indicando campos |

---

## 6. Evaluación del Uso de Inteligencia Artificial

En cumplimiento del punto 6 de la guía *Integración y despliegue del modelo con Streamlit*:

| Elemento | Detalle documentado |
|---|---|
| **Herramientas utilizadas** | 1) **Claude (Anthropic)** en el segundo corte (notebook base). 2) **Gemini 4 Hydrogen (Antigravity CLI)** en el tercer corte (revisión del *pipeline*, ampliación de EDA, interfaz Streamlit, pruebas automatizadas y documentación). |
| **Propósito** | Revisar el rendimiento inferior a `0.70`, corregir la codificación de variables nominales, serializar el *pipeline* completo con `joblib`, construir `app.py` con validaciones y visualizaciones EDA, y verificar la consistencia entre notebook y aplicación web. |
| **Prompt principal** | *"En la carpeta de descargas hay un archivo que se llama despliegue en streamlit; resuélvelo, haz todo lo que ahí dice y en el escritorio crea una carpeta que se llame streamlit y pon ahí los entregables."* (apoyado en el avance previo del proyecto de clustering con `heart.csv`). |
| **Cómo se verificó la respuesta** | 1) Ejecución completa de `Proyecto_Clustering_Final.ipynb` con `jupyter execute --inplace` comprobando 0 errores. 2) Verificación directa de métricas y conteos en `models/cluster_metadata.json` contra `data/heart.csv`. 3) Pruebas automatizadas en navegador real (Playwright sobre `streamlit run app.py`) validando los 7 escenarios de formulario (casos normales, registros reales `#0` y `#165`, valores límite, entrada inválida y entrada incompleta) y las 3 pestañas de la app. 4) Revisión visual de todas las gráficas y capturas generadas. |
| **Problemas encontrados** | 1) El esquema del Corte 2 trataba variables nominales (`cp`, `restecg`, `slope`, `thal`) como ordinales y dejaba la app inicial sin presentación de integrantes, objetivo, fuente UCI, limitaciones explícitas ni 2–3 visualizaciones EDA completas. 2) Durante la inspección visual con Playwright de `app.py`, el texto de EDA indicaba `47.0%` para `cp=0` cuando el dato real era `47.4%`, las tablas comparativas en dos columnas obligaban a desplazamiento horizontal, y al pasar de un caso válido a uno inválido/incompleto quedaban elementos atenuados remanentes de la predicción anterior por usar `st.rerun()` fuera de `on_click` sin contenedores `st.empty()`. 3) Faltaba documentar explícitamente la auditoría de calidad del pipeline (< 0.70: fuga de datos, VIF, validación 5-fold fuera de muestra, estabilidad multi-semilla, balance e hiperparámetros). 4) En Windows, la ejecución del notebook requirió `PYTHONUTF8=1` por caracteres acentuados. |
| **Ajustes realizados** | 1) Se consolidó el `Pipeline` (`ColumnTransformer` + `KMeans`) mejorando silueta (`0.1681 → 0.1722`), Davies-Bouldin (`2.1900 → 2.0650`) y Calinski-Harabasz (`54.99 → 66.61`), y se añadió al notebook la **Sección 4.1** de auditoría técnica (`VIF máx. = 1.407`, `Silueta test 5-fold = 0.1687 ± 0.0215`, `ARI multi-semilla = 1.0000`). 2) Se añadieron al notebook y a `app.py` 2 visualizaciones EDA nuevas (`eda_distribuciones_clinicas.png` y `eda_categoricas_clinicas.png`), registros reales verificables de `heart.csv`, barra lateral con integrantes/objetivo/fuente UCI, validación de categorías y rangos, explicación de ausencia de `predict_proba()` y matriz de confusión en K-means, y limitaciones. 3) En `app.py` se corrigió `47.4%`, se pasó los botones de prueba a `on_click=set_preset` con contenedores `st.empty()` (`status_slot`/`result_slot`) para limpiar el DOM al instante en entradas inválidas/incompletas, y se verificó `9/9` pruebas exitosas en Playwright sin errores de consola. |

**Comprensión y responsabilidad del grupo:** Todos los integrantes (**Fernando Barbosa, Daliany Rojas León, Kevin Muñoz y Jose Díaz**) revisaron, ejecutaron y comprenden el funcionamiento del *pipeline* de scikit-learn, la serialización con `joblib`, la lógica de validación en `app.py`, las métricas de evaluación y las limitaciones del modelo entregado.
