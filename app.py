from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Segmentación Cardiovascular — Minería de Datos CUC",
    page_icon="🫀",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "clustering_pipeline.joblib"
PCA_PATH = BASE_DIR / "models" / "pca_2d.joblib"
META_PATH = BASE_DIR / "models" / "cluster_metadata.json"
DATA_PATH = BASE_DIR / "data" / "heart.csv"
FIG_DIR = BASE_DIR / "figures"

CP_LABELS = {
    0: "0 — Angina típica",
    1: "1 — Angina atípica",
    2: "2 — Dolor no anginoso",
    3: "3 — Asintomático",
}
RESTECG_LABELS = {
    0: "0 — Normal",
    1: "1 — Anomalía onda ST-T",
    2: "2 — Hipertrofia ventricular izquierda",
}
SLOPE_LABELS = {
    0: "0 — Ascendente",
    1: "1 — Plana",
    2: "2 — Descendente",
}
THAL_LABELS = {
    0: "0 — Nulo / indeterminado",
    1: "1 — Defecto fijo",
    2: "2 — Normal",
    3: "3 — Defecto reversible",
}

ALLOWED_CATEGORIES = {
    "ca": {0, 1, 2, 3, 4},
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {0, 1, 2, 3},
    "restecg": {0, 1, 2},
    "slope": {0, 1, 2},
    "thal": {0, 1, 2, 3},
}

HARD_BOUNDS = {
    "age": (18.0, 110.0, "Edad (años)"),
    "trestbps": (60.0, 250.0, "Presión arterial en reposo (mm Hg)"),
    "chol": (80.0, 700.0, "Colesterol sérico (mg/dl)"),
    "thalach": (50.0, 230.0, "Frecuencia cardíaca máxima (lpm)"),
    "oldpeak": (0.0, 10.0, "Depresión del segmento ST (oldpeak)"),
    "ca": (0.0, 4.0, "Número de vasos coloreados (ca)"),
}


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    pca_2d = joblib.load(PCA_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    df_clean = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    return pipeline, pca_2d, metadata, df_clean


def validate_inputs(record: dict, feature_ranges: dict):
    """Valida completitud, tipos de datos, categorías permitidas y rangos clínicos."""
    errors = []
    warnings = []

    missing_fields = [
        k for k, v in record.items()
        if v is None or (isinstance(v, float) and np.isnan(v))
    ]
    if missing_fields:
        errors.append(
            f"Faltan campos requeridos por el pipeline: {', '.join(missing_fields)}. "
            "Complete todos los indicadores clínicos antes de ejecutar la segmentación."
        )
        return errors, warnings

    for col, val in record.items():
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(float(val)):
            errors.append(f"El campo '{col}' debe contener un valor numérico válido.")

    if errors:
        return errors, warnings

    for col, allowed in ALLOWED_CATEGORIES.items():
        val = record[col]
        if float(val) != int(val) or int(val) not in allowed:
            errors.append(
                f"La categoría '{col}' = {val} no pertenece al conjunto válido {sorted(allowed)}."
            )

    for col, (lo, hi, label) in HARD_BOUNDS.items():
        val = float(record[col])
        if val < lo or val > hi:
            errors.append(
                f"{label} = {val} está fuera del dominio clínico aceptado ({lo:g} a {hi:g})."
            )

    if errors:
        return errors, warnings

    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        val = float(record[col])
        cmin = feature_ranges[col]["min"]
        cmax = feature_ranges[col]["max"]
        if val < cmin or val > cmax:
            warnings.append(
                f"'{col}' ({val}) supera el rango observado en el dataset de entrenamiento [{cmin}, {cmax}]; "
                "la asignación es una extrapolación del pipeline."
            )
        elif val <= cmin + 0.03 * (cmax - cmin) or val >= cmax - 0.03 * (cmax - cmin):
            warnings.append(
                f"'{col}' ({val}) se encuentra muy próximo a los límites observados del dataset [{cmin}, {cmax}]."
            )

    return errors, warnings


def set_preset(values: dict):
    for k, v in values.items():
        st.session_state[f"inp_{k}"] = v


def find_matching_verified_case(record: dict, verified_cases: list, features: list):
    for item in verified_cases:
        inputs = item["inputs"]
        if all(abs(float(record[k]) - float(inputs[k])) < 1e-6 for k in features):
            return item
    return None


def render_sidebar():
    with st.sidebar:
        st.header("Proyecto de Aula — Corte 3")
        st.markdown(
            "**Título:** Segmentación de Pacientes Mediante Técnicas de Clustering "
            "Basadas en Indicadores Clínicos Cardiovasculares"
        )
        st.markdown(
            "**Objetivo:** Revisar el modelo no supervisado del segundo corte, "
            "optimizar su preprocesamiento mixto en un *pipeline* serializado con `joblib` "
            "y desplegar una interfaz web reproducible para consultar perfiles clínicos."
        )
        st.markdown(
            "**Integrantes:**\n"
            "- Fernando Barbosa\n"
            "- Daliany Rojas León\n"
            "- Kevin Muñoz\n"
            "- Jose Díaz"
        )
        st.markdown("**Institución:** Universidad de la Costa (CUC) · Minería de Datos")
        st.markdown(
            "**Fuente de los datos:** *Heart Disease UCI* (Cleveland Clinic Foundation) — "
            "[UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) / Kaggle "
            "(303 registros iniciales → 302 únicos tras eliminar 1 duplicado; 13 variables clínicas, "
            "`target` excluida del entrenamiento)."
        )
        st.info(
            "Prototipo académico exploratorio. No realiza diagnósticos médicos ni reemplaza "
            "la valoración de un profesional de la salud."
        )


def main():
    pipeline, pca_2d, metadata, df_clean = load_artifacts()
    features = metadata["features"]
    verified_cases = metadata.get("verified_test_cases", [])

    render_sidebar()

    st.title("🫀 Segmentación de Pacientes con Indicadores Cardiovasculares")
    st.caption(
        "Universidad de la Costa (CUC) · Minería de Datos · Integrantes: Fernando Barbosa, "
        "Daliany Rojas León, Kevin Muñoz, Jose Díaz · Pipeline `ColumnTransformer` + `KMeans(k=2)`"
    )

    st.warning(
        "**Aviso académico obligatorio:** Esta aplicación es un **prototipo académico exploratorio**. "
        "Los perfiles mostrados reflejan patrones estadísticos en la cohorte histórica *Heart Disease UCI* "
        "y **no constituyen un diagnóstico clínico ni sustituyen una decisión médica profesional**."
    )

    defaults = {
        "age": 63, "trestbps": 145, "chol": 260, "thalach": 125, "oldpeak": 2.4, "ca": 2,
        "sex": 1, "fbs": 1, "exang": 1, "cp": 0, "restecg": 0, "slope": 1, "thal": 3,
    }
    for k, v in defaults.items():
        if f"inp_{k}" not in st.session_state:
            st.session_state[f"inp_{k}"] = v

    tab_pred, tab_eda, tab_metrics = st.tabs([
        "🔎 Segmentación Interactiva",
        "📈 Análisis Exploratorio (EDA)",
        "📊 Métricas, Contraste y Limitaciones",
    ])

    with tab_pred:
        st.subheader("Casos rápidos de verificación (consistencia, límites e invalidez)")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            st.button(
                "Caso normal: Perfil mayor carga",
                use_container_width=True,
                on_click=set_preset,
                args=(verified_cases[0]["inputs"],),
            )
        with r1_c2:
            st.button(
                "Caso normal: Perfil menor esfuerzo",
                use_container_width=True,
                on_click=set_preset,
                args=(verified_cases[1]["inputs"],),
            )
        with r1_c3:
            st.button(
                "Registro conocido: Paciente #0 (heart.csv)",
                use_container_width=True,
                on_click=set_preset,
                args=(verified_cases[2]["inputs"],),
            )

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            st.button(
                "Registro conocido: Paciente #165 (heart.csv)",
                use_container_width=True,
                on_click=set_preset,
                args=(verified_cases[3]["inputs"],),
            )
        with r2_c2:
            st.button(
                "Prueba: Valores límite observados",
                use_container_width=True,
                on_click=set_preset,
                args=(verified_cases[4]["inputs"],),
            )
        with r2_c3:
            st.button(
                "Prueba: Entrada inválida (fuera de rango)",
                use_container_width=True,
                on_click=set_preset,
                args=({
                    "age": 12, "trestbps": 280, "chol": 750, "thalach": 150, "oldpeak": 1.0, "ca": 0,
                    "sex": 1, "fbs": 0, "exang": 0, "cp": 1, "restecg": 1, "slope": 1, "thal": 2,
                },),
            )
        with r2_c4:
            st.button(
                "Prueba: Entrada incompleta (campos vacíos)",
                use_container_width=True,
                on_click=set_preset,
                args=({
                    "age": None, "trestbps": 130, "chol": None, "thalach": 150, "oldpeak": 1.0, "ca": 0,
                    "sex": 1, "fbs": 0, "exang": 0, "cp": 1, "restecg": 1, "slope": 1, "thal": 2,
                },),
            )

        st.divider()

        with st.form("form_clustering"):
            st.markdown("### Ingreso de las 13 variables clínicas del paciente")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Edad (`age`, años)", min_value=0, max_value=130, step=1, key="inp_age")
                trestbps = st.number_input("Presión arterial en reposo (`trestbps`, mm Hg)", min_value=0, max_value=320, step=1, key="inp_trestbps")
                chol = st.number_input("Colesterol sérico (`chol`, mg/dl)", min_value=0, max_value=900, step=1, key="inp_chol")
                thalach = st.number_input("Frecuencia cardíaca máxima (`thalach`, lpm)", min_value=0, max_value=260, step=1, key="inp_thalach")
                oldpeak = st.number_input("Depresión ST por ejercicio (`oldpeak`, mm)", min_value=0.0, max_value=12.0, step=0.1, format="%.1f", key="inp_oldpeak")

            with col2:
                ca = st.selectbox("Vasos coloreados por fluoroscopía (`ca`, 0–4)", options=[0, 1, 2, 3, 4], key="inp_ca")
                sex = st.selectbox("Sexo biológico (`sex`)", options=[0, 1], format_func=lambda x: "0 — Mujer" if x == 0 else "1 — Hombre", key="inp_sex")
                fbs = st.selectbox("Glucosa en ayunas > 120 mg/dl (`fbs`)", options=[0, 1], format_func=lambda x: "0 — No" if x == 0 else "1 — Sí", key="inp_fbs")
                exang = st.selectbox("Angina inducida por ejercicio (`exang`)", options=[0, 1], format_func=lambda x: "0 — No" if x == 0 else "1 — Sí", key="inp_exang")

            with col3:
                cp = st.selectbox("Tipo de dolor torácico (`cp`)", options=[0, 1, 2, 3], format_func=lambda x: CP_LABELS[x], key="inp_cp")
                restecg = st.selectbox("Electrocardiograma en reposo (`restecg`)", options=[0, 1, 2], format_func=lambda x: RESTECG_LABELS[x], key="inp_restecg")
                slope = st.selectbox("Pendiente del segmento ST (`slope`)", options=[0, 1, 2], format_func=lambda x: SLOPE_LABELS[x], key="inp_slope")
                thal = st.selectbox("Resultado de talasemia (`thal`)", options=[0, 1, 2, 3], format_func=lambda x: THAL_LABELS[x], key="inp_thal")

            submitted = st.form_submit_button("Asignar Clúster con el Pipeline", type="primary", use_container_width=True)

        record = {
            "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak, "ca": ca,
            "sex": sex, "fbs": fbs, "exang": exang, "cp": cp, "restecg": restecg, "slope": slope, "thal": thal,
        }

        errors, warnings = validate_inputs(record, metadata["feature_ranges"])
        status_slot = st.empty()
        result_slot = st.empty()

        if errors:
            result_slot.empty()
            with status_slot.container():
                for err in errors:
                    st.error(err)
        else:
            if warnings:
                with status_slot.container():
                    for warn in warnings:
                        st.warning(warn)
            else:
                status_slot.empty()

            with result_slot.container():
                df_input = pd.DataFrame([{k: record[k] for k in features}])
                cluster_id = int(pipeline.predict(df_input)[0])
                dists = pipeline.transform(df_input)[0]
                label = metadata["cluster_labels"][str(cluster_id)]

                # Proximidad relativa geométrica (K-means no dispone de predict_proba)
                inv_d = 1.0 / (dists + 1e-9)
                rel_prox = inv_d / inv_d.sum() * 100.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Clúster Asignado", f"Clúster {cluster_id}")
                m2.metric("Distancia al Centroide 0", f"{dists[0]:.3f}")
                m3.metric("Distancia al Centroide 1", f"{dists[1]:.3f}")
                m4.metric(
                    "Proximidad geométrica relativa*",
                    f"{rel_prox[cluster_id]:.1f}%",
                    help="Calculada como inverso normalizado de la distancia euclidiana. K-means no implementa predict_proba().",
                )

                if cluster_id == 0:
                    st.info(f"**Perfil asignado:** Clúster 0 — *{label}*")
                else:
                    st.success(f"**Perfil asignado:** Clúster 1 — *{label}*")

                st.caption(
                    "*Nota sobre probabilidad:* Al tratarse de un modelo no supervisado (`KMeans`), el algoritmo no "
                    "dispone de `predict_proba()` ni calcula probabilidades diagnósticas; la asignación se determina "
                    "exclusivamente por la menor distancia euclidiana a los centroides en el espacio de 23 dimensiones."
                )

                matched_case = find_matching_verified_case(record, verified_cases, features)
                if matched_case:
                    st.success(
                        f"**Verificación Notebook ↔ Streamlit ({matched_case['caso']}):** "
                        f"Coincidencia exacta confirmada — Notebook: Clúster {matched_case['cluster']} "
                        f"(c0={matched_case['dist_c0']:.3f}, c1={matched_case['dist_c1']:.3f}) | "
                        f"Streamlit: Clúster {cluster_id} (c0={dists[0]:.3f}, c1={dists[1]:.3f}). "
                        f"Origen: {matched_case['origen']}."
                    )

                col_left, col_right = st.columns([1.1, 1.0])

                with col_left:
                    st.markdown("#### Comparación con los perfiles promedio de la cohorte (n = 302)")
                    p0 = metadata["cluster_profiles_mean"]["0"]
                    p1 = metadata["cluster_profiles_mean"]["1"]
                    n0 = metadata["cluster_sizes"]["0"]
                    n1 = metadata["cluster_sizes"]["1"]
                    comp_table = pd.DataFrame({
                        "Variable": features,
                        "Registro ingresado": [record[f] for f in features],
                        f"Media Clúster 0 (n={n0})": [p0[f] for f in features],
                        f"Media Clúster 1 (n={n1})": [p1[f] for f in features],
                    })
                    st.dataframe(comp_table, use_container_width=True, hide_index=True)

                with col_right:
                    st.markdown("#### Proyección PCA 2D respecto a la cohorte")
                    X_cohort = df_clean[features]
                    cohort_labels = pipeline.predict(X_cohort)
                    cohort_pca = pca_2d.transform(pipeline.named_steps["preprocessor"].transform(X_cohort))
                    input_pca = pca_2d.transform(pipeline.named_steps["preprocessor"].transform(df_input))

                    fig, ax = plt.subplots(figsize=(6.2, 4.8))
                    for cid, color, name in [
                        (0, "#d62728", "Clúster 0: Mayor carga aparente"),
                        (1, "#1f77b4", "Clúster 1: Menor esfuerzo aparente"),
                    ]:
                        mask = cohort_labels == cid
                        ax.scatter(
                            cohort_pca[mask, 0], cohort_pca[mask, 1],
                            c=color, label=name, alpha=0.45, edgecolors="white", s=40,
                        )
                    ax.scatter(
                        input_pca[0, 0], input_pca[0, 1],
                        c="#ffd700", edgecolors="black", marker="*", s=260, linewidths=1.4,
                        label="Registro evaluado", zorder=5,
                    )
                    var_ratio = metadata["metrics"]["pca_variance_ratio"]
                    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var.)")
                    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var.)")
                    ax.set_title("Ubicación del registro en el plano PCA", fontweight="bold")
                    ax.legend(fontsize=8, loc="best")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    st.pyplot(fig)
                    plt.close(fig)

    with tab_eda:
        st.subheader("Análisis Exploratorio de Datos (Heart Disease UCI)")
        st.markdown(
            f"El conjunto de datos original cuenta con **{metadata.get('n_records_raw', 303)} registros** "
            f"y **14 columnas**. Tras verificar ausencia de valores nulos (`0` faltantes) y eliminar "
            f"**{metadata.get('n_duplicates_removed', 1)} registro duplicado**, la cohorte analizada se "
            f"consolida en **{metadata['n_records']} pacientes únicos** y **13 variables clínicas**."
        )

        eda1, eda2 = st.columns(2)
        with eda1:
            if (FIG_DIR / "eda_distribuciones_clinicas.png").exists():
                st.image(
                    str(FIG_DIR / "eda_distribuciones_clinicas.png"),
                    caption="Figura EDA 1. Distribución de las 6 variables numéricas clínicas (n = 302).",
                    use_container_width=True,
                )
            st.markdown(
                "**Interpretación:** La edad presenta mediana de `55.5` años, la presión en reposo de "
                "`130.0 mm Hg` y el colesterol de `240.5 mg/dl`. Los valores extremos detectados por IQR "
                "(`trestbps`: 9, `chol`: 5, `oldpeak`: 5, `thalach`: 1) se conservaron por ser clínicamente plausibles."
            )

        with eda2:
            if (FIG_DIR / "eda_categoricas_clinicas.png").exists():
                st.image(
                    str(FIG_DIR / "eda_categoricas_clinicas.png"),
                    caption="Figura EDA 2. Frecuencia de variables categóricas y binarias clave (n = 302).",
                    use_container_width=True,
                )
            st.markdown(
                "**Interpretación:** Predomina el sexo masculino (`68.2%`), el dolor torácico tipo 0 (`47.4%`) "
                "y la ausencia de angina inducida por ejercicio (`67.2%`). Las variables nominales `cp`, `restecg`, "
                "`slope` y `thal` se codificaron con `OneHotEncoder` para evitar órdenes numéricos artificiales."
            )

        st.divider()
        col_corr, col_desc = st.columns([1.15, 1.0])
        with col_corr:
            if (FIG_DIR / "correlacion_eda.png").exists():
                st.image(
                    str(FIG_DIR / "correlacion_eda.png"),
                    caption="Figura EDA 3. Matriz de correlación de Pearson entre variables numéricas.",
                    use_container_width=True,
                )
        with col_desc:
            st.markdown("#### Resumen estadístico de variables clínicas (n = 302)")
            desc_df = df_clean[features].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
            desc_df.columns = ["Media", "Desv. Est.", "Mínimo", "Mediana", "Máximo"]
            st.dataframe(desc_df, use_container_width=True)
            st.markdown(
                "**Hallazgo clave:** `age` y `thalach` exhiben la correlación negativa más marcada (`r = -0.40`), "
                "coherente con la reducción esperada de frecuencia cardíaca máxima con la edad."
            )

    with tab_metrics:
        st.subheader("Evaluación Final del Pipeline, Contraste y Limitaciones")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Pacientes únicos", metadata["n_records"])
        k2.metric("Variables transformadas", metadata["n_transformed_features"])
        k3.metric("Coef. de Silueta (k=2)", f"{metadata['metrics']['silhouette']:.4f}")
        k4.metric("Índice Davies-Bouldin", f"{metadata['metrics']['davies_bouldin']:.4f}")
        k5.metric("Índice Calinski-Harabasz", f"{metadata['metrics']['calinski_harabasz']:.2f}")

        st.markdown(
            "El *pipeline* definitivo (`models/clustering_pipeline.joblib`) integra `ColumnTransformer` "
            "(`StandardScaler` en `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`; `passthrough` en `sex`, "
            "`fbs`, `exang`; `OneHotEncoder` en `cp`, `restecg`, `slope`, `thal`) y `KMeans(n_clusters=2, n_init=10, random_state=42)`."
        )

        st.markdown("#### Comparación Corte 2 vs. Corte 3 y alternativas (k = 2)")
        if "comparison_approaches" in metadata:
            comp_view = pd.DataFrame(metadata["comparison_approaches"]).rename(columns={
                "Configuracion": "Configuración",
                "Dimensiones": "Dim.",
                "Davies_Bouldin": "Davies-Bouldin",
                "Calinski_Harabasz": "Calinski-Harabasz",
            })
            st.dataframe(comp_view, use_container_width=True, hide_index=True)

        st.markdown("#### Evaluación de métricas internas para k = 2 a 8")
        if "metrics_by_k" in metadata:
            k_view = pd.DataFrame(metadata["metrics_by_k"]).rename(columns={
                "Davies_Bouldin": "Davies-Bouldin",
                "Calinski_Harabasz": "Calinski-Harabasz",
            })
            st.dataframe(k_view, use_container_width=True, hide_index=True)

        g1, g2 = st.columns(2)
        if (FIG_DIR / "seleccion_k.png").exists():
            g1.image(str(FIG_DIR / "seleccion_k.png"), caption="Selección del número óptimo de clústeres (k=2 a 8)", use_container_width=True)
        if (FIG_DIR / "pca_tsne_clusters.png").exists():
            g2.image(str(FIG_DIR / "pca_tsne_clusters.png"), caption="Proyección bidimensional mediante PCA y t-SNE", use_container_width=True)

        g3, g4 = st.columns([1.05, 1.0])
        with g3:
            if (FIG_DIR / "perfil_clusters_heatmap.png").exists():
                st.image(str(FIG_DIR / "perfil_clusters_heatmap.png"), caption="Perfil promedio comparativo por clúster", use_container_width=True)
        with g4:
            st.markdown("#### Contraste descriptivo con `target` (sin matriz de confusión)")
            st.info(
                "Dado que **K-means es aprendizaje no supervisado**, las etiquetas de clúster (`0` y `1`) son "
                "identificadores geométricos arbitrarios y **no deben evaluarse con accuracy ni matriz de confusión** "
                "como un clasificador supervisado."
            )
            if "target_crosstab" in metadata:
                tc = metadata["target_crosstab"]
                cruce_df = pd.DataFrame([
                    {"Clúster asignado": "Clúster 0 (Mayor carga aparente)", "target = 0": tc["cluster_0"]["target_0"], "target = 1": tc["cluster_0"]["target_1"], "Total": tc["cluster_0"]["total"]},
                    {"Clúster asignado": "Clúster 1 (Menor esfuerzo aparente)", "target = 0": tc["cluster_1"]["target_0"], "target = 1": tc["cluster_1"]["target_1"], "Total": tc["cluster_1"]["total"]},
                ])
                st.dataframe(cruce_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Tabla estrictamente exploratoria: la columna `target` se excluyó por completo del entrenamiento "
                    "y de la inferencia para prevenir fugas de información."
                )

        st.divider()
        st.markdown("### Limitaciones del modelo y revisión técnica (métrica < 0.70)")
        st.markdown(
            "- **Coeficiente de silueta moderado-bajo (`0.1722 < 0.70`):** Tras revisar el *pipeline* del segundo corte, "
            "reemplazar la codificación ordinal por `OneHotEncoder` mejoró silueta (`0.1681 → 0.1722`), Davies-Bouldin (`2.1900 → 2.0650`) "
            "y Calinski-Harabasz (`54.99 → 66.61`), superando en equilibrio global a `RobustScaler`, agrupamiento jerárquico `Ward` (`0.1019`) "
            "y `GMM` (`0.0791`). Que ninguna técnica alcance `0.70` refleja el solapamiento fisiológico continuo de los datos clínicos.\n"
            "- **Auditoría de calidad del pipeline (Ficha técnica < 0.70):** Sin fuga de información (`target` excluido); "
            "colinealidad controlada (`|r| máx. = 0.40`, `VIF máx. = 1.407 < 5`); estabilidad fuera de muestra en 5 particiones "
            "(`Silueta test = 0.1687 ± 0.0215`, `ARI vs full = 0.9733`); convergencia idéntica multi-semilla (`ARI = 1.0000`) y con "
            "`n_init ∈ {10, 20, 50}` (`inercia = 2186.73`); y clústeres equilibrados (`137` vs `165`, razón `1.2:1`).\n"
            "- **Alcance muestral y sesgo de cohorte:** La muestra de `302` pacientes proviene de una única cohorte histórica de Cleveland "
            "con predominio masculino (`68.2%`), por lo que no es generalizable sin validación multicéntrica externa.\n"
            "- **Sensibilidad a atípicos y geometría esférica:** Se conservaron outliers clínicamente plausibles (`chol` hasta `564 mg/dl`, "
            "`trestbps` hasta `200 mm Hg`), los cuales pueden desplazar los centroides de K-means.\n"
            "- **Aproximación bidimensional y ausencia de causalidad:** PCA 2D captura el `39.5%` de la varianza total (complementado con t-SNE), "
            "y los clústeres describen únicamente asociaciones estadísticas multivariadas sin implicar causalidad ni diagnóstico médico."
        )


if __name__ == "__main__":
    main()
