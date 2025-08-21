import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO

# --------------------------
# Configuración de la página
# --------------------------
st.set_page_config(
    page_title="EDA Deportivo Sintético",
    page_icon="🏟️",
    layout="wide"
)

st.title("🏟️ EDA Interactivo con Datos Sintéticos de Deportes")
st.caption("Genera un dataset sintético (cuantitativo, cualitativo o mixto), elige columnas (máx. 6) y explora visualmente con gráficos y estadísticas.")

# -----------------------------------------------------
# Definición del "esquema" de columnas y generadores
# -----------------------------------------------------
SPORTS = ["Fútbol", "Baloncesto", "Tenis", "Atletismo", "Natación", "Ciclismo"]
TEAMS = ["Leones", "Tiburones", "Águilas", "Tigres", "Toros", "Pumas"]
POSITIONS = ["Portero", "Defensa", "Mediocampo", "Delantero", "Base", "Alero", "Pívot"]
GENDERS = ["M", "F"]
COUNTRIES = ["Colombia", "Argentina", "Brasil", "España", "Francia", "USA", "México"]

# Definimos un catálogo de posibles columnas con su tipo y función generadora
def gen_fecha(n, seed=None):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    # Serie temporal diaria con algún jitter de días
    return [start + timedelta(days=int(i)) for i in range(n)]

def gen_deporte(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice(SPORTS, size=n)

def gen_equipo(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice(TEAMS, size=n)

def gen_posicion(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice(POSITIONS, size=n)

def gen_genero(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice(GENDERS, size=n, p=[0.6, 0.4])

def gen_pais(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice(COUNTRIES, size=n)

def gen_edad(n, seed=None):
    rng = np.random.default_rng(seed); return rng.integers(16, 40, size=n)

def gen_minutos(n, seed=None):
    rng = np.random.default_rng(seed); return np.clip(rng.normal(70, 15, size=n).round(0), 0, 120)

def gen_velocidad(n, seed=None):
    rng = np.random.default_rng(seed); return np.clip(rng.normal(28, 4, size=n).round(2), 10, 40)  # km/h

def gen_fc(n, seed=None):
    rng = np.random.default_rng(seed); return np.clip(rng.normal(145, 12, size=n).round(0), 90, 200)  # frecuencia cardiaca

def gen_goles(n, seed=None):
    rng = np.random.default_rng(seed); base = rng.poisson(0.6, size=n); return base

def gen_asistencias(n, seed=None):
    rng = np.random.default_rng(seed); return rng.poisson(0.8, size=n)

def gen_injury(n, seed=None):
    rng = np.random.default_rng(seed); return rng.choice([0, 1], size=n, p=[0.85, 0.15])  # 1=lesionado

def gen_costo_ficha(n, seed=None):
    rng = np.random.default_rng(seed); return (np.exp(rng.normal(10, 0.5, size=n)) / 1e5).round(2)  # valor sintético

def gen_indice_rendimiento(n, seed=None):
    rng = np.random.default_rng(seed)
    # Índice que depende de minutos, velocidad, fc, goles y asistencias (se ajusta luego si faltan)
    minutos = gen_minutos(n, seed)
    vel = gen_velocidad(n, seed)
    fc = gen_fc(n, seed)
    goles = gen_goles(n, seed)
    asist = gen_asistencias(n, seed)
    score = (0.4*minutos/90) + (0.3*vel/35) + (0.1*(180-fc)/80) + (0.15*goles) + (0.05*asist)
    score = np.clip(score, 0, None)
    return np.round(100 * score / score.max(), 2)

COLUMN_CATALOG = {
    # Mixtas / temporales
    "fecha": {"tipo": "temporal", "gen": gen_fecha},
    # Cualitativas
    "deporte": {"tipo": "categorica", "gen": gen_deporte},
    "equipo": {"tipo": "categorica", "gen": gen_equipo},
    "posicion": {"tipo": "categorica", "gen": gen_posicion},
    "genero": {"tipo": "categorica", "gen": gen_genero},
    "pais": {"tipo": "categorica", "gen": gen_pais},
    # Cuantitativas
    "edad": {"tipo": "numerica", "gen": gen_edad},
    "minutos": {"tipo": "numerica", "gen": gen_minutos},
    "velocidad_kmh": {"tipo": "numerica", "gen": gen_velocidad},
    "fc_promedio": {"tipo": "numerica", "gen": gen_fc},
    "goles": {"tipo": "numerica", "gen": gen_goles},
    "asistencias": {"tipo": "numerica", "gen": gen_asistencias},
    "lesion": {"tipo": "binaria", "gen": gen_injury},
    "costo_ficha_musd": {"tipo": "numerica", "gen": gen_costo_ficha},
    "indice_rendimiento": {"tipo": "numerica", "gen": gen_indice_rendimiento},
}

def tipo_columna(nombre):
    return COLUMN_CATALOG[nombre]["tipo"]

# --------------------------
# Sidebar: controles
# --------------------------
st.sidebar.header("⚙️ Configuración de datos")
seed = st.sidebar.number_input("Semilla (opcional)", min_value=0, step=1, value=42)
n_muestras = st.sidebar.slider("Número de filas", 50, 500, 200, step=10)

todas_cols = list(COLUMN_CATALOG.keys())
cols_seleccionadas = st.sidebar.multiselect(
    "Selecciona hasta 6 columnas",
    options=todas_cols,
    default=["fecha", "deporte", "edad", "minutos", "goles", "indice_rendimiento"],
    max_selections=6
)

generar = st.sidebar.button("🎲 Generar/Actualizar datos")

# Guardamos en sesión
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

def generar_df(n, cols, seed):
    df = pd.DataFrame()
    for c in cols:
        df[c] = COLUMN_CATALOG[c]["gen"](n, seed)
    # Si no está 'fecha', creamos un índice temporal opcional para tendencias
    if "fecha" not in df.columns:
        df["fecha"] = gen_fecha(n, seed)
    # Mezcla aleatoria consistente
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if generar or st.session_state.df.empty:
    st.session_state.df = generar_df(n_muestras, cols_seleccionadas, seed)

df = st.session_state.df.copy()

# --------------------------
# Filtros dinámicos (categorías)
# --------------------------
st.subheader("🔎 Filtros")
cat_cols = [c for c in df.columns if tipo_columna(c) in ("categorica", "binaria")]
cols_filtros = st.columns(min(len(cat_cols), 4) if cat_cols else 1)
aplicar_filtro = False
for i, c in enumerate(cat_cols):
    with cols_filtros[i % len(cols_filtros)]:
        valores = sorted(df[c].unique().tolist())
        sel = st.multiselect(f"Filtrar **{c}**", options=valores, default=valores)
        if len(sel) < len(valores):
            aplicar_filtro = True
        df = df[df[c].isin(sel)]

if aplicar_filtro:
    st.caption(f"Se aplicaron filtros. Filas restantes: **{len(df)}**")

# --------------------------
# Sección EDA rápida
# --------------------------
st.subheader("📊 Resumen Rápido (EDA)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filas", len(df))
with col2:
    st.metric("Columnas", len(df.columns))
with col3:
    n_null = int(df.isna().sum().sum())
    st.metric("Valores faltantes", n_null)

with st.expander("Ver estadísticas descriptivas (numéricas)"):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        st.dataframe(df[num_cols].describe().T)
    else:
        st.info("No hay columnas numéricas seleccionadas.")

with st.expander("Ver frecuencia por variables categóricas"):
    if cat_cols:
        for c in cat_cols:
            st.write(f"**{c}**")
            st.dataframe(df[c].value_counts().rename("conteo").to_frame())
    else:
        st.info("No hay columnas categóricas seleccionadas.")

# --------------------------
# Tabla y descarga
# --------------------------
st.subheader("🧾 Tabla de datos")
st.dataframe(df, use_container_width=True)
csv = df.to_csv(index=False)
st.download_button("💾 Descargar CSV", data=csv, file_name="datos_deportivos_sinteticos.csv", mime="text/csv")

# --------------------------
# Visualizaciones
# --------------------------
st.subheader("📈 Visualizaciones Interactivas")

tab_line, tab_bar, tab_scatter, tab_pie, tab_hist, tab_box = st.tabs(
    ["Tendencia (línea)", "Barras", "Dispersión", "Pastel", "Histograma", "Caja (Boxplot)"]
)

# Tendencia (línea)
with tab_line:
    st.markdown("Gráfico de **tendencia** (ideal usar `fecha` en el eje X).")
    posibles_x = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c])]
    x = st.selectbox("Eje X", options=posibles_x, index=posibles_x.index("fecha") if "fecha" in posibles_x else 0)
    y_candidatas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x]
    if not y_candidatas:
        st.warning("No hay columnas numéricas para Y.")
    else:
        y = st.selectbox("Variable Y (numérica)", options=y_candidatas, index=0)
        color_opt = st.selectbox("Color (opcional)", options=["(ninguno)"] + [c for c in df.columns if c != x and c != y], index=0)
        df_sorted = df.sort_values(by=x)
        fig = px.line(df_sorted, x=x, y=y, color=None if color_opt == "(ninguno)" else color_opt, markers=True)
        st.plotly_chart(fig, use_container_width=True)

# Barras
with tab_bar:
    st.markdown("Gráfico de **barras** por categoría con agregación de una métrica.")
    cat_for_bar = [c for c in df.columns if df[c].dtype == "object" or tipo_columna(c) in ("categorica", "binaria")]
    num_for_bar = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cat_for_bar or not num_for_bar:
        st.warning("Necesitas al menos 1 categórica y 1 numérica.")
    else:
        ccat = st.selectbox("Categoría", options=cat_for_bar, index=0)
        mnum = st.selectbox("Métrica (numérica)", options=num_for_bar, index=0)
        agg = st.selectbox("Agregación", options=["count", "sum", "mean", "median"], index=2)
        df_bar = df.groupby(ccat)[mnum].agg(agg).reset_index()
        fig = px.bar(df_bar, x=ccat, y=mnum, text=mnum)
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# Dispersión
with tab_scatter:
    st.markdown("Gráfico de **dispersión** para ver relaciones entre dos numéricas.")
    num_cols_sc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols_sc) < 2:
        st.warning("Se requieren al menos 2 columnas numéricas.")
    else:
        x = st.selectbox("X (numérica)", options=num_cols_sc, index=0, key="scx")
        y = st.selectbox("Y (numérica)", options=[c for c in num_cols_sc if c != x], index=0, key="scy")
        color_opt = st.selectbox("Color (opcional)", options=["(ninguno)"] + [c for c in df.columns if c not in [x, y]], index=0, key="sccolor")
        size_opt = st.selectbox("Tamaño (opcional, numérica)", options=["(ninguno)"] + num_cols_sc, index=0, key="scsize")
        fig = px.scatter(
            df, x=x, y=y,
            color=None if color_opt == "(ninguno)" else color_opt,
            size=None if size_opt == "(ninguno)" else size_opt,
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

# Pastel
with tab_pie:
    st.markdown("Gráfico de **pastel** para distribución de una categoría.")
    cats = [c for c in df.columns if df[c].dtype == "object" or tipo_columna(c) in ("categorica", "binaria")]
    if not cats:
        st.warning("Se necesita al menos una columna categórica.")
    else:
        ccat = st.selectbox("Categoría", options=cats, index=0, key="piecat")
        peso_opt = st.selectbox("Valores (opcional, numérico para ponderar)", options=["(conteo)"] + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], index=0)
        if peso_opt == "(conteo)":
            df_pie = df[ccat].value_counts().reset_index()
            df_pie.columns = [ccat, "valor"]
        else:
            df_pie = df.groupby(ccat)[peso_opt].sum().reset_index().rename(columns={peso_opt: "valor"})
        fig = px.pie(df_pie, names=ccat, values="valor", hole=0.2)
        st.plotly_chart(fig, use_container_width=True)

# Histograma
with tab_hist:
    st.markdown("**Histograma** para ver la distribución de una variable numérica.")
    num_cols_h = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols_h:
        st.warning("No hay columnas numéricas.")
    else:
        varh = st.selectbox("Variable", options=num_cols_h, index=0)
        bins = st.slider("Número de bins", 5, 60, 20)
        fig = px.histogram(df, x=varh, nbins=bins, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

# Boxplot
with tab_box:
    st.markdown("Gráfico de **caja (boxplot)** para comparar distribución por categoría.")
    num_cols_b = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_b = [c for c in df.columns if df[c].dtype == "object" or tipo_columna(c) in ("categorica", "binaria")]
    if not num_cols_b:
        st.warning("No hay variables numéricas.")
    else:
        y = st.selectbox("Variable numérica", options=num_cols_b, index=0, key="boxy")
        cat = st.selectbox("Categoría (opcional)", options=["(ninguna)"] + cat_cols_b, index=0, key="boxcat")
        if cat == "(ninguna)":
            fig = px.box(df, y=y, points="suspectedoutliers")
        else:
            fig = px.box(df, x=cat, y=y, points="suspectedoutliers")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Tips de uso
# --------------------------
with st.expander("💡 Consejos de uso"):
    st.markdown("""
- Usa **`fecha`** en el eje X para ver **tendencias**.
- En **barras**, combina una **categoría** con una **métrica** y elige la **agregación**.
- En **dispersión**, prueba **color** por `deporte` o `equipo` para descubrir patrones.
- **Pastel** muestra la proporción de categorías (o ponderada por una métrica).
- Ajusta **filtros** por `deporte`, `equipo`, `genero`, etc. para enfocarte en subgrupos.
- Descarga el CSV para continuar tu análisis en R, Python o Excel.
""")

