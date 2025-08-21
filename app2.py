import io
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="🌱 EDA interactivo - Agricultura", page_icon="🌱", layout="wide")
st.title("🌱 EDA interactivo para Dataset de Agricultura")
st.caption("Carga tu CSV, limpia los datos y explora con visualizaciones interactivas.")

# =========================
# Utilidades de limpieza
# =========================
MOJIBAKE_FIXES = {
    "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
    "Ã±": "ñ", "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó",
    "Ãš": "Ú", "Ã‘": "Ñ", "CafÃ©": "Café"
}

def fix_mojibake(s: str) -> str:
    if not isinstance(s, str):
        return s
    for wrong, right in MOJIBAKE_FIXES.items():
        s = s.replace(wrong, right)
    # Normaliza acentos compuestos
    return unicodedata.normalize("NFC", s)

def sanitize_columns(cols):
    return (
        pd.Index(cols)
        .map(lambda c: fix_mojibake(str(c)))
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

def coerce_numeric(df, num_cols):
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def infer_types(df: pd.DataFrame):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (df[c].dtype == "object") or pd.api.types.is_categorical_dtype(df[c])]
    dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return num_cols, cat_cols, dt_cols

def add_numpy_ols_line(fig, df_local, x_col, y_col):
    # Añade recta de ajuste OLS sin statsmodels
    if pd.api.types.is_numeric_dtype(df_local[x_col]) and pd.api.types.is_numeric_dtype(df_local[y_col]) and len(df_local) >= 2:
        x_vals = df_local[x_col].to_numpy()
        y_vals = df_local[y_col].to_numpy()
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
        y_line = m * x_line + b
        fig.add_scatter(x=x_line, y=y_line, mode="lines", name="Ajuste (numpy)")

# =========================
# Carga de archivo
# =========================
st.sidebar.header("📂 Cargar archivo")
uploaded = st.sidebar.file_uploader("Sube tu CSV (.csv)", type=["csv"])

read_opts = st.sidebar.expander("Opciones de lectura")
with read_opts:
    sep = st.text_input("Separador", value=",")
    decimal = st.text_input("Decimal", value=".")
    header = st.number_input("Fila de encabezado (0-index)", min_value=0, value=0, step=1)
    try_encodings = st.multiselect("Intentar codificaciones", ["utf-8", "latin1", "cp1252"], default=["utf-8","latin1"])

df_raw = None
read_error = None

if uploaded:
    # Intento de lectura con varias codificaciones
    content = uploaded.read()
    for enc in try_encodings:
        try:
            df_raw = pd.read_csv(io.BytesIO(content), sep=sep, decimal=decimal, header=header, encoding=enc)
            break
        except Exception as e:
            read_error = str(e)

    if df_raw is None:
        st.error(f"No se pudo leer el CSV. Último error: {read_error}")

if df_raw is None:
    st.info("👆 Carga un archivo CSV para comenzar el análisis.")
    st.stop()

# =========================
# Limpieza inicial
# =========================
df = df_raw.copy()

# Corrige posibles mojibake en nombres de columnas
df.columns = sanitize_columns(df.columns)

# Corrige mojibake en celdas de texto
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].map(fix_mojibake)

# Reemplaza cadenas problemáticas comunes
df = df.replace({"error": np.nan, "NA": np.nan, "na": np.nan, "": np.nan, " ": np.nan})

# Sugerencia de columnas típicas del dataset planteado
sugeridas_numericas = ["pH_suelo","Humedad","Temperatura","Precipitacion","RadiacionSolar","Nutrientes"]
sugerida_categoria = "Cultivo"

presentes_num = [c for c in sugeridas_numericas if c in df.columns]
present_cat = sugerida_categoria if sugerida_categoria in df.columns else None

# Coerción numérica
df = coerce_numeric(df, presentes_num)

# Intento de conversión de fecha si existe
posibles_fechas = [c for c in df.columns if c.lower() in ["fecha","date","dia","diam edicion".replace(" ", ""), "timestamp"]]
for c in posibles_fechas:
    try:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    except Exception:
        pass

# =========================
# Panel de limpieza
# =========================
st.sidebar.header("🧹 Limpieza de datos")

# Elegir columnas numéricas y categóricas (auto-inferidas)
num_cols, cat_cols, dt_cols = infer_types(df)
sel_num_cols = st.sidebar.multiselect("Columnas numéricas", options=num_cols, default=presentes_num or num_cols)
sel_cat_cols = st.sidebar.multiselect("Columnas categóricas", options=cat_cols, default=([present_cat] if present_cat else cat_cols))

# Imputación de faltantes
st.sidebar.subheader("Imputación de faltantes")
how_num = st.sidebar.selectbox("Numéricas", ["(no imputar)","media","mediana","cero","eliminar filas con NaN"], index=1)
how_cat = st.sidebar.selectbox("Categóricas", ["(no imputar)","moda","Desconocido","eliminar filas con NaN"], index=1)

df_clean = df.copy()

# Imputación numérica
if sel_num_cols:
    if how_num == "media":
        for c in sel_num_cols:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
    elif how_num == "mediana":
        for c in sel_num_cols:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    elif how_num == "cero":
        df_clean[sel_num_cols] = df_clean[sel_num_cols].fillna(0)
    elif how_num == "eliminar filas con NaN":
        df_clean = df_clean.dropna(subset=sel_num_cols)

# Imputación categórica
if sel_cat_cols:
    if how_cat == "moda":
        for c in sel_cat_cols:
            moda = df_clean[c].mode(dropna=True)
            if not moda.empty:
                df_clean[c] = df_clean[c].fillna(moda.iloc[0])
    elif how_cat == "Desconocido":
        for c in sel_cat_cols:
            df_clean[c] = df_clean[c].fillna("Desconocido")
    elif how_cat == "eliminar filas con NaN":
        df_clean = df_clean.dropna(subset=sel_cat_cols)

# =========================
# Vista previa y descarga
# =========================
st.subheader("👀 Vista previa")
st.write("**Dimensiones:** ", df_clean.shape)
st.dataframe(df_clean.head(20), use_container_width=True)

csv_clean = df_clean.to_csv(index=False).encode("utf-8")
st.download_button("💾 Descargar CSV limpio", data=csv_clean, file_name="dataset_agricultura_limpio.csv", mime="text/csv")

# =========================
# Resumen EDA
# =========================
st.subheader("📊 Resumen (EDA)")
colA, colB, colC, colD = st.columns(4)
with colA: st.metric("Filas", len(df_clean))
with colB: st.metric("Columnas", len(df_clean.columns))
with colC: st.metric("Nº numéricas", len(sel_num_cols))
with colD: st.metric("Nº categóricas", len(sel_cat_cols))

with st.expander("Valores faltantes (post-limpieza)"):
    st.dataframe(df_clean.isna().sum().to_frame("faltantes"))

with st.expander("Estadísticas descriptivas (numéricas)"):
    if sel_num_cols:
        st.dataframe(df_clean[sel_num_cols].describe().T)
    else:
        st.info("No hay columnas numéricas seleccionadas.")

with st.expander("Frecuencias (categóricas)"):
    if sel_cat_cols:
        for c in sel_cat_cols:
            st.write(f"**{c}**")
            st.dataframe(df_clean[c].value_counts(dropna=False).rename("conteo").to_frame())
    else:
        st.info("No hay columnas categóricas seleccionadas.")

# =========================
# Filtros por categoría
# =========================
st.subheader("🔎 Filtros")
cat_for_filters = sel_cat_cols
if cat_for_filters:
    cols_filt = st.columns(min(len(cat_for_filters), 4))
    df_viz = df_clean.copy()
    for i, c in enumerate(cat_for_filters):
        with cols_filt[i % len(cols_filt)]:
            vals = sorted([v for v in df_viz[c].dropna().unique().tolist()])
            sel = st.multiselect(f"Filtrar **{c}**", options=vals, default=vals)
            if sel:
                df_viz = df_viz[df_viz[c].isin(sel)]
    st.caption(f"Filtrado aplicado. Filas visibles: **{len(df_viz)}**")
else:
    df_viz = df_clean

# =========================
# Visualizaciones
# =========================
st.subheader("📈 Visualizaciones interactivas")

tab_line, tab_bar, tab_scatter, tab_pie, tab_hist, tab_box, tab_corr = st.tabs(
    ["Tendencia (línea)", "Barras", "Dispersión", "Pastel", "Histograma", "Caja (Boxplot)", "Correlación"]
)

# Tendencia (línea)
with tab_line:
    st.markdown("Usa una columna temporal o numérica en el eje X.")
    posibles_x = [c for c in df_viz.columns if pd.api.types.is_datetime64_any_dtype(df_viz[c]) or pd.api.types.is_numeric_dtype(df_viz[c])]
    if not posibles_x:
        st.info("No hay columna temporal/numérica para eje X.")
    else:
        x = st.selectbox("Eje X", options=posibles_x, index=0, key="line_x")
        y_opts = [c for c in sel_num_cols if c != x] if sel_num_cols else []
        if not y_opts:
            st.warning("Selecciona columnas numéricas en la barra lateral.")
        else:
            y = st.selectbox("Variable Y", options=y_opts, index=0, key="line_y")
            color_opt = st.selectbox("Color (opcional)", options=["(ninguno)"] + [c for c in df_viz.columns if c not in [x, y]], index=0, key="line_color")
            df_sorted = df_viz.sort_values(by=x)
            fig = px.line(df_sorted, x=x, y=y, color=None if color_opt == "(ninguno)" else color_opt, markers=True)
            st.plotly_chart(fig, use_container_width=True)

# Barras (ARREGLADO: nombres consistentes)
with tab_bar:
    st.markdown("Requiere 1 categórica y (opcional) 1 métrica.")
    cats = sel_cat_cols
    nums = sel_num_cols
    if not cats:
        st.info("Selecciona al menos una categórica en la barra lateral.")
    else:
        ccat = st.selectbox("Categoría", options=cats, index=0, key="bar_cat")
        agg_mode = st.selectbox("Agregación", options=["conteo","suma","media","mediana"], index=0)
        df_bar = None

        if agg_mode == "conteo" or not nums:
            tmp = df_viz[ccat].value_counts(dropna=False).reset_index()
            tmp.columns = ["categoria", "valor"]
            df_bar = tmp
        else:
            mnum = st.selectbox("Métrica (numérica)", options=nums, index=0, key="bar_num")
            if agg_mode == "suma":
                tmp = df_viz.groupby(ccat, dropna=False)[mnum].sum().reset_index()
            elif agg_mode == "media":
                tmp = df_viz.groupby(ccat, dropna=False)[mnum].mean().reset_index()
            else:
                tmp = df_viz.groupby(ccat, dropna=False)[mnum].median().reset_index()
            # Renombrar a nombres estándar
            tmp = tmp.rename(columns={ccat: "categoria", mnum: "valor"})
            df_bar = tmp

        if df_bar is None or df_bar.empty:
            st.warning("No hay datos para graficar.")
        else:
            # Convertimos NaN de categoría a string para evitar errores en Plotly
            df_bar["categoria"] = df_bar["categoria"].astype(str)
            fig = px.bar(df_bar, x="categoria", y="valor", text="valor")
            st.plotly_chart(fig, use_container_width=True)

# Dispersión
with tab_scatter:
    if len(sel_num_cols) < 2:
        st.info("Selecciona al menos 2 columnas numéricas en la barra lateral.")
    else:
        x_sc = st.selectbox("X", options=sel_num_cols, index=0, key="sc_x")
        y_sc = st.selectbox("Y", options=[c for c in sel_num_cols if c != x_sc], index=0, key="sc_y")
        color_opt = st.selectbox("Color (opcional)", options=["(ninguno)"] + [c for c in df_viz.columns if c not in [x_sc, y_sc]], index=0, key="sc_color")
        size_opt = st.selectbox("Tamaño (opcional, numérica)", options=["(ninguno)"] + sel_num_cols, index=0, key="sc_size")
        add_trend = st.checkbox("Añadir recta OLS (numpy)", value=False)

        fig = px.scatter(
            df_viz, x=x_sc, y=y_sc,
            color=None if color_opt == "(ninguno)" else color_opt,
            size=None if size_opt == "(ninguno)" else size_opt,
        )
        if add_trend:
            add_numpy_ols_line(fig, df_viz.dropna(subset=[x_sc, y_sc]), x_sc, y_sc)
        st.plotly_chart(fig, use_container_width=True)

# Pastel
with tab_pie:
    cats = sel_cat_cols
    if not cats:
        st.info("Selecciona al menos una categórica.")
    else:
        ccat = st.selectbox("Categoría", options=cats, index=0, key="pie_cat")
        weight_opt = st.selectbox("Ponderar por (opcional, numérica)", options=["(conteo)"] + sel_num_cols, index=0, key="pie_weight")
        if weight_opt == "(conteo)":
            df_pie = df_viz[ccat].value_counts(dropna=False).reset_index()
            df_pie.columns = ["categoria", "valor"]
        else:
            df_pie = df_viz.groupby(ccat, dropna=False)[weight_opt].sum().reset_index().rename(columns={ccat: "categoria", weight_opt: "valor"})
        df_pie["categoria"] = df_pie["categoria"].astype(str)
        fig = px.pie(df_pie, names="categoria", values="valor", hole=0.2)
        st.plotly_chart(fig, use_container_width=True)

# Histograma
with tab_hist:
    if not sel_num_cols:
        st.info("Selecciona columnas numéricas en la barra lateral.")
    else:
        varh = st.selectbox("Variable", options=sel_num_cols, index=0, key="hist_var")
        bins = st.slider("Bins", 5, 60, 20, key="hist_bins")
        fig = px.histogram(df_viz, x=varh, nbins=bins, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

# Boxplot
with tab_box:
    if not sel_num_cols:
        st.info("Selecciona columnas numéricas en la barra lateral.")
    else:
        y_b = st.selectbox("Variable numérica", options=sel_num_cols, index=0, key="box_y")
        cat_b = st.selectbox("Categoría (opcional)", options=["(ninguna)"] + sel_cat_cols, index=0, key="box_cat")
        if cat_b == "(ninguna)":
            fig = px.box(df_viz, y=y_b, points="suspectedoutliers")
        else:
            fig = px.box(df_viz, x=cat_b, y=y_b, points="suspectedoutliers")
        st.plotly_chart(fig, use_container_width=True)

# Correlación
with tab_corr:
    if len(sel_num_cols) < 2:
        st.info("Se requieren al menos dos columnas numéricas.")
    else:
        corr = df_viz[sel_num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Consejos de uso
# =========================
with st.expander("💡 Consejos de uso"):
    st.markdown("""
- Si ves nombres raros como **CafÃ©**, ya se corrigen automáticamente a **Café**.
- Marca *Añadir recta OLS (numpy)* en **Dispersión** para ver tendencia lineal sin instalar `statsmodels`.
- Usa la barra lateral para **elegir columnas** y **cómo imputar** faltantes.
- En **Tendencia (línea)**, si no tienes fechas, puedes usar un índice numérico o cualquier variable numérica como eje X.
- Puedes **descargar** el dataset limpio en CSV para otros análisis.
""")

