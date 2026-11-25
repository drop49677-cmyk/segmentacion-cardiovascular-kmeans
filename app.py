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


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    pca_2d = joblib.load(PCA_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    df_raw = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    return pipeline, pca_2d, metadata, df_raw


def validate_inputs(record: dict, feature_ranges: dict):
    """Valida completitud, dominio fisiológico básico y cercanía a los límites observados."""
    errors = []
    warnings = []

    missing_fields = [k for k, v in record.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    if missing_fields:
        errors.append(
            f"Faltan campos requeridos por el pipeline: {', '.join(missing_fields)}. "
            "Complete todos los indicadores clínicos antes de ejecutar la segmentación."
        )
        return errors, warnings

    hard_bounds = {
        "age": (18, 110, "Edad (años)"),
        "trestbps": (60, 250, "Presión arterial en reposo (mm Hg)"),
        "chol": (80, 700, "Colesterol sérico (mg/dl)"),
        "thalach": (50, 230, "Frecuencia cardíaca máxima (lpm)"),
        "oldpeak": (0.0, 10.0, "Depresión del segmento ST (oldpeak)"),
        "ca": (0, 4, "Número de vasos coloreados (ca)"),
    }
    for col, (lo, hi, label) in hard_bounds.items():
        val = float(record[col])
        if val < lo or val > hi:
            errors.append(f"{label} = {val} está fuera del dominio clínico aceptado ({lo} a {hi}).")

    if errors:
        return errors, warnings

    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        val = float(record[col])
        cmin = feature_ranges[col]["min"]
        cmax = feature_ranges[col]["max"]
        if val < cmin or val > cmax:
            warnings.append(
                f"'{col}' ({val}) supera el rango observado en el dataset de entrenamiento [{cmin}, {cmax}]; "
                "el resultado es una extrapolación del pipeline."
            )
        elif val <= cmin + 0.03 * (cmax - cmin) or val >= cmax - 0.03 * (cmax - cmin):
            warnings.append(
                f"'{col}' ({val}) se encuentra muy próximo a los límites observados del dataset [{cmin}, {cmax}]."
            )

    return errors, warnings


def set_preset(values: dict):
    for k, v in values.items():
        st.session_state[f"inp_{k}"] = v


def main():
    pipeline, pca_2d, metadata, df_clean = load_artifacts()
    features = metadata["features"]

    st.title("🫀 Segmentación de Pacientes con Indicadores Cardiovasculares")
    st.caption(
        "Universidad de la Costa (CUC) · Minería de Datos · Modelo K-means (k=2) "
        "con Pipeline ColumnTransformer (StandardScaler + OneHotEncoder)"
    )

    st.warning(
        "**Aviso académico obligatorio:** Esta aplicación es una herramienta exploratoria de apoyo académico. "
        "Los perfiles mostrados reflejan patrones estadísticos en el conjunto histórico *Heart Disease UCI* "
        "y **no constituyen un diagnóstico clínico ni sustituyen la valoración de un profesional de la salud**."
    )

    # Inicializar estado por defecto con la Prueba 1 si no existe
    defaults = {
        "age": 63, "trestbps": 145, "chol": 260, "thalach": 125, "oldpeak": 2.4, "ca": 2,
        "sex": 1, "fbs": 1, "exang": 1, "cp": 0, "restecg": 0, "slope": 1, "thal": 3,
    }
    for k, v in defaults.items():
        if f"inp_{k}" not in st.session_state:
            st.session_state[f"inp_{k}"] = v

    tab_pred, tab_eda = st.tabs(["🔎 Segmentación de Registro", "📊 Métricas y Análisis del Modelo"])

    with tab_pred:
        st.subheader("Casos rápidos de verificación")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            if st.button("Prueba 1: Perfil mayor carga", use_container_width=True):
                set_preset({
                    "age": 63, "trestbps": 145, "chol": 260, "thalach": 125, "oldpeak": 2.4, "ca": 2,
                    "sex": 1, "fbs": 1, "exang": 1, "cp": 0, "restecg": 0, "slope": 1, "thal": 3,
                })
                st.rerun()
        with c_p2:
            if st.button("Prueba 2: Perfil menor esfuerzo", use_container_width=True):
                set_preset({
                    "age": 44, "trestbps": 120, "chol": 210, "thalach": 172, "oldpeak": 0.0, "ca": 0,
                    "sex": 0, "fbs": 0, "exang": 0, "cp": 2, "restecg": 1, "slope": 2, "thal": 2,
                })
                st.rerun()
        with c_p3:
            if st.button("Prueba 3: Valores límite", use_container_width=True):
                set_preset({
                    "age": 77, "trestbps": 200, "chol": 564, "thalach": 71, "oldpeak": 6.2, "ca": 4,
                    "sex": 1, "fbs": 1, "exang": 1, "cp": 0, "restecg": 2, "slope": 0, "thal": 3,
                })
                st.rerun()
        with c_p4:
            if st.button("Prueba 4: Entrada incompleta", use_container_width=True):
                set_preset({
                    "age": None, "trestbps": 130, "chol": None, "thalach": 150, "oldpeak": 1.0, "ca": 0,
                    "sex": 1, "fbs": 0, "exang": 0, "cp": 1, "restecg": 1, "slope": 1, "thal": 2,
                })
                st.rerun()

        st.divider()

        with st.form("form_clustering"):
            st.markdown("### Ingreso de las 13 variables clínicas del paciente")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Edad (`age`, años)", min_value=0, max_value=120, step=1, key="inp_age")
                trestbps = st.number_input("Presión arterial reposo (`trestbps`, mm Hg)", min_value=0, max_value=300, step=1, key="inp_trestbps")
                chol = st.number_input("Colesterol sérico (`chol`, mg/dl)", min_value=0, max_value=800, step=1, key="inp_chol")
                thalach = st.number_input("Frecuencia cardíaca máx. (`thalach`, lpm)", min_value=0, max_value=250, step=1, key="inp_thalach")
                oldpeak = st.number_input("Depresión ST (`oldpeak`)", min_value=0.0, max_value=12.0, step=0.1, format="%.1f", key="inp_oldpeak")

            with col2:
                ca = st.selectbox("Vasos coloreados (`ca`, 0–4)", options=[0, 1, 2, 3, 4], key="inp_ca")
                sex = st.selectbox("Sexo biológico (`sex`)", options=[0, 1], format_func=lambda x: "0 — Mujer" if x == 0 else "1 — Hombre", key="inp_sex")
                fbs = st.selectbox("Glucosa ayunas > 120 mg/dl (`fbs`)", options=[0, 1], format_func=lambda x: "0 — No" if x == 0 else "1 — Sí", key="inp_fbs")
                exang = st.selectbox("Angina inducida por ejercicio (`exang`)", options=[0, 1], format_func=lambda x: "0 — No" if x == 0 else "1 — Sí", key="inp_exang")

            with col3:
                cp = st.selectbox("Tipo de dolor torácico (`cp`)", options=[0, 1, 2, 3], format_func=lambda x: CP_LABELS[x], key="inp_cp")
                restecg = st.selectbox("ECG en reposo (`restecg`)", options=[0, 1, 2], format_func=lambda x: RESTECG_LABELS[x], key="inp_restecg")
                slope = st.selectbox("Pendiente segmento ST (`slope`)", options=[0, 1, 2], format_func=lambda x: SLOPE_LABELS[x], key="inp_slope")
                thal = st.selectbox("Talasemia (`thal`)", options=[0, 1, 2, 3], format_func=lambda x: THAL_LABELS[x], key="inp_thal")

            submitted = st.form_submit_button("Asignar Clúster con el Pipeline", type="primary", use_container_width=True)

        record = {
            "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak, "ca": ca,
            "sex": sex, "fbs": fbs, "exang": exang, "cp": cp, "restecg": restecg, "slope": slope, "thal": thal,
        }

        if submitted or all(record[k] is not None for k in features):
            errors, warnings = validate_inputs(record, metadata["feature_ranges"])

            if errors:
                for err in errors:
                    st.error(err)
            else:
                for warn in warnings:
                    st.warning(warn)

                df_input = pd.DataFrame([{k: record[k] for k in features}])
                cluster_id = int(pipeline.predict(df_input)[0])
                dists = pipeline.transform(df_input)[0]
                label = metadata["cluster_labels"][str(cluster_id)]

                m1, m2, m3 = st.columns(3)
                m1.metric("Clúster Asignado", f"Clúster {cluster_id}")
                m2.metric("Distancia al Centroide 0", f"{dists[0]:.3f}")
                m3.metric("Distancia al Centroide 1", f"{dists[1]:.3f}")

                if cluster_id == 0:
                    st.info(f"**Perfil asignado:** Clúster 0 — *{label}*")
                else:
                    st.success(f"**Perfil asignado:** Clúster 1 — *{label}*")

                col_left, col_right = st.columns([1.1, 1.0])

                with col_left:
                    st.markdown("#### Comparación con los perfiles promedio de la cohorte (n = 302)")
                    p0 = metadata["cluster_profiles_mean"]["0"]
                    p1 = metadata["cluster_profiles_mean"]["1"]
                    comp_table = pd.DataFrame({
                        "Variable": features,
                        "Registro ingresado": [record[f] for f in features],
                        "Media Clúster 0 (n=137)": [p0[f] for f in features],
                        "Media Clúster 1 (n=165)": [p1[f] for f in features],
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
                            c=color, label=name, alpha=0.45, edgecolors="white", s=40
                        )
                    ax.scatter(
                        input_pca[0, 0], input_pca[0, 1],
                        c="#ffd700", edgecolors="black", marker="*", s=260, linewidths=1.4,
                        label="Registro evaluado", zorder=5
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
        st.subheader("Resumen cuantitativo del modelo definitivo")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Pacientes únicos", metadata["n_records"])
        k2.metric("Variables transformadas", metadata["n_transformed_features"])
        k3.metric("Coef. de Silueta (k=2)", f"{metadata['metrics']['silhouette']:.4f}")
        k4.metric("Índice Davies-Bouldin", f"{metadata['metrics']['davies_bouldin']:.4f}")
        k5.metric("Índice Calinski-Harabasz", f"{metadata['metrics']['calinski_harabasz']:.2f}")

        st.markdown(
            "El *pipeline* final empaqueta un `ColumnTransformer` (`StandardScaler` sobre `age`, `trestbps`, "
            "`chol`, `thalach`, `oldpeak`, `ca`; `passthrough` en `sex`, `fbs`, `exang`; y `OneHotEncoder` "
            "en `cp`, `restecg`, `slope`, `thal`) seguido de `KMeans(n_clusters=2, n_init=10, random_state=42)`."
        )

        g1, g2 = st.columns(2)
        if (FIG_DIR / "seleccion_k.png").exists():
            g1.image(str(FIG_DIR / "seleccion_k.png"), caption="Selección del número óptimo de clústeres (k=2 a 8)", use_container_width=True)
        if (FIG_DIR / "pca_tsne_clusters.png").exists():
            g2.image(str(FIG_DIR / "pca_tsne_clusters.png"), caption="Proyección bidimensional mediante PCA y t-SNE", use_container_width=True)

        g3, g4 = st.columns(2)
        if (FIG_DIR / "perfil_clusters_heatmap.png").exists():
            g3.image(str(FIG_DIR / "perfil_clusters_heatmap.png"), caption="Perfil promedio comparativo por clúster", use_container_width=True)
        if (FIG_DIR / "correlacion_eda.png").exists():
            g4.image(str(FIG_DIR / "correlacion_eda.png"), caption="Matriz de correlación de variables numéricas", use_container_width=True)


if __name__ == "__main__":
    main()
