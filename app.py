import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="Futuro Laboral Puebla: IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Carga y Caché de Datos ---
@st.cache_data # <-- Usamos caché para no recargar datos en cada interacción
def load_data():
    """
    Carga y genera datos simulados de matrícula y riesgo IA.
    En un proyecto real, aquí leerías desde tu BBDD (MySQL/MongoDB) o CSV.
    """
    years = list(range(2005, 2025))
    carreras = ['Derecho', 'Contaduría', 'Comunicación', 'Diseño Gráfico',
                'Ing. Software', 'Medicina', 'Arquitectura', 'Psicología']
    universidades = ['BUAP', 'UDLAP', 'UPAEP', 'Ibero Puebla', 'Tec Puebla']
    areas = {'Derecho': 'Ciencias Sociales', 'Contaduría': 'Económico-Admin.',
             'Comunicación': 'Ciencias Sociales', 'Diseño Gráfico': 'Artes y Humanidades',
             'Ing. Software': 'Ingeniería y Tecnología', 'Medicina': 'Ciencias de la Salud',
             'Arquitectura': 'Ingeniería y Tecnología', 'Psicología': 'Ciencias de la Salud'}

    data = []

    np.random.seed(42)

    base_matricula = {
        'Derecho': 250, 'Contaduría': 230, 'Comunicación': 180, 'Diseño Gráfico': 150,
        'Ing. Software': 100, 'Medicina': 200, 'Arquitectura': 160, 'Psicología': 170
    }
    tendencia = {
        'Derecho': 0.01, 'Contaduría': -0.015, 'Comunicación': 0.005, 'Diseño Gráfico': -0.02,
        'Ing. Software': 0.09, 'Medicina': 0.03, 'Arquitectura': 0.00, 'Psicología': 0.02
    }
    riesgo = {
        'Derecho': 0.60, 'Contaduría': 0.85, 'Comunicación': 0.50, 'Diseño Gráfico': 0.75,
        'Ing. Software': 0.25, 'Medicina': 0.40, 'Arquitectura': 0.55, 'Psicología': 0.45
    }

    for carrera in carreras:
        for uni in universidades:
            mat = base_matricula[carrera] / len(universidades) # Distribuir base
            for year in years:
                # Aplicar tendencia y ruido aleatorio
                mat = mat * (1 + tendencia[carrera] + np.random.uniform(-0.03, 0.03)) + np.random.randint(-5, 5)
                # Simular menor matrícula en algunas unis
                factor_uni = 0.8 if uni in ['Ibero Puebla', 'Tec Puebla'] else 1.0
                mat_final = int(max(mat * factor_uni, 10))

                data.append({
                    'Año': year,
                    'Carrera': carrera,
                    'Universidad': uni,
                    'Area': areas[carrera],
                    'Matricula': mat_final,
                    'Riesgo_IA': riesgo[carrera]
                })

    df = pd.DataFrame(data)
    return df

# --- 3. Funciones de Análisis y Visualización ---

def plot_historical_trends(df, group_by='Carrera'):
    """Genera un gráfico de líneas con las tendencias históricas."""
    df_trend = df.groupby(['Año', group_by])['Matricula'].sum().reset_index()
    fig = px.line(
        df_trend,
        x='Año',
        y='Matricula',
        color=group_by,
        title=f'Evolución Histórica de la Matrícula por {group_by}',
        markers=True,
        labels={'Matricula': 'Número de Estudiantes'}
    )
    fig.update_layout(hovermode="x unified")
    return fig

def plot_latest_enrollment(df):
    """Genera un gráfico de barras con la matrícula del último año y el riesgo."""
    df_latest = df[df['Año'] == df['Año'].max()]
    df_latest_grouped = df_latest.groupby('Carrera').agg(
        Matricula_Total=('Matricula', 'sum'),
        Riesgo_IA=('Riesgo_IA', 'first')
    ).reset_index()

    fig = px.bar(
        df_latest_grouped.sort_values('Matricula_Total', ascending=False),
        x='Carrera',
        y='Matricula_Total',
        color='Riesgo_IA',
        title=f'Matrícula en {df["Año"].max()} y Nivel de Riesgo IA',
        labels={'Matricula_Total': f'Matrícula Total ({df["Año"].max()})', 'Riesgo_IA': 'Riesgo IA (0-1)'},
        color_continuous_scale=px.colors.sequential.Reds,
        hover_data={'Carrera': True, 'Matricula_Total': True, 'Riesgo_IA': ':.2f'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_risk_vs_growth(df):
    """Genera un gráfico de dispersión: Riesgo IA vs Crecimiento."""
    df_agg = df.groupby('Carrera').agg(
        Matricula_Inicio=('Matricula', lambda x: df.loc[x.index[df.loc[x.index, 'Año'] == df['Año'].min()], 'Matricula'].sum()),
        Matricula_Fin=('Matricula', lambda x: df.loc[x.index[df.loc[x.index, 'Año'] == df['Año'].max()], 'Matricula'].sum()),
        Riesgo_IA=('Riesgo_IA', 'first')
    ).reset_index()

    # Calcular Tasa de Crecimiento Anual Compuesta (CAGR)
    num_years = df['Año'].max() - df['Año'].min()
    df_agg['CAGR'] = ((df_agg['Matricula_Fin'] / df_agg['Matricula_Inicio'])**(1/num_years) - 1) * 100
    df_agg = df_agg[df_agg['Matricula_Inicio'] > 0] # Evitar división por cero

    fig = px.scatter(
        df_agg,
        x='Riesgo_IA',
        y='CAGR',
        size='Matricula_Fin',
        color='Carrera',
        title='Riesgo IA vs. Crecimiento Histórico (CAGR %)',
        labels={'Riesgo_IA': 'Nivel de Riesgo IA (0=Bajo, 1=Alto)', 'CAGR': 'Crecimiento Anual (%)', 'Matricula_Fin': 'Matrícula Actual'},
        hover_name='Carrera',
        size_max=60
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=df_agg['Riesgo_IA'].mean(), line_dash="dash", line_color="red", annotation_text="Riesgo Promedio")
    return fig

def generate_projection(df):
    """Genera una proyección SIMPLIFICADA a 10 años."""
    proyeccion_list = []
    df_riesgo = df[['Carrera', 'Riesgo_IA']].drop_duplicates().set_index('Carrera')
    df_grouped = df.groupby(['Año', 'Carrera'])['Matricula'].sum().reset_index()

    for carrera in df_grouped['Carrera'].unique():
        hist_data = df_grouped[df_grouped['Carrera'] == carrera]
        if len(hist_data) < 2: continue # Necesita al menos 2 puntos

        # Regresión lineal simple para la tendencia
        z = np.polyfit(hist_data['Año'], hist_data['Matricula'], 1)
        tendencia_anual = z[0]

        matricula_actual = hist_data['Matricula'].iloc[-1]
        riesgo = df_riesgo.loc[carrera, 'Riesgo_IA']

        # Factor de ajuste basado en riesgo (más riesgo = más frena la tendencia)
        # Si riesgo es alto (0.85), factor es bajo (0.15). Si riesgo es bajo (0.25), factor es alto (0.75)
        # Se aplica un factor 'amortiguador' (0.5) para no ser tan extremos
        factor_ia = (1 - riesgo) * 0.5 + 0.5

        # Proyección: (Matricula + Tendencia * Años) * Factor IA
        # Si la tendencia es negativa, el factor IA la hace 'más negativa'
        # Usamos una forma más estable: (Matricula_Actual + Tendencia_Ajustada * Años)
        tendencia_ajustada = tendencia_anual * factor_ia if tendencia_anual > 0 else tendencia_anual * (1 + riesgo * 0.5)

        matricula_proyectada = matricula_actual + (tendencia_ajustada * 11) # 2025 a 2035 son 11 años
        matricula_proyectada = max(0, int(matricula_proyectada))

        proyeccion_list.append({
            'Carrera': carrera,
            f'Matrícula {df["Año"].max()}': matricula_actual,
            'Riesgo IA': riesgo,
            'Tendencia Hist. (Alumnos/Año)': round(tendencia_anual, 1),
            'Matrícula Estimada 2035': matricula_proyectada,
            'Cambio Estimado (%)': round(((matricula_proyectada / matricula_actual) - 1) * 100, 1) if matricula_actual > 0 else 0
        })

    df_proyeccion = pd.DataFrame(proyeccion_list).sort_values('Cambio Estimado (%)')
    return df_proyeccion


# --- 4. Carga de Datos y Aplicación ---
df_data = load_data()

# --- 5. Barra Lateral de Filtros ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Escudo_del_Estado_de_Puebla.svg/1200px-Escudo_del_Estado_de_Puebla.svg.png", width=100)
st.sidebar.title("Panel de Control ⚙️")
st.sidebar.markdown("Filtra los datos para tu análisis:")

# Filtro de Años
min_year, max_year = int(df_data['Año'].min()), int(df_data['Año'].max())
selected_years = st.sidebar.slider(
    'Selecciona Rango de Años:',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filtro de Universidades
all_universities = df_data['Universidad'].unique()
selected_universities = st.sidebar.multiselect(
    'Selecciona Universidad(es):',
    options=all_universities,
    default=all_universities
)

# Filtro de Áreas
all_areas = df_data['Area'].unique()
selected_areas = st.sidebar.multiselect(
    'Selecciona Área(s) de Conocimiento:',
    options=all_areas,
    default=all_areas
)

# Filtro de Carreras (dependiente de las áreas seleccionadas)
carreras_in_areas = df_data[df_data['Area'].isin(selected_areas)]['Carrera'].unique()
selected_carreras = st.sidebar.multiselect(
    'Selecciona Carrera(s):',
    options=carreras_in_areas,
    default=carreras_in_areas
)

# Aplicar filtros
df_filtered = df_data[
    (df_data['Año'] >= selected_years[0]) &
    (df_data['Año'] <= selected_years[1]) &
    (df_data['Universidad'].isin(selected_universities)) &
    (df_data['Area'].isin(selected_areas)) &
    (df_data['Carrera'].isin(selected_carreras))
].copy()

# --- 6. Cuerpo Principal de la Aplicación ---
st.title("🎓🤖 Futuro Laboral Puebla: IA vs Carreras Universitarias")
st.markdown(f"""
Bienvenido al tablero de análisis del impacto de la Inteligencia Artificial en las carreras universitarias del estado de Puebla.
Explora las tendencias históricas (`{selected_years[0]}-{selected_years[1]}`) y las proyecciones hacia 2035.
**Recuerda: Los datos actuales son *simulados* y las proyecciones son *ejemplificativas*.**
""")

# Mostrar KPIs si hay datos
if not df_filtered.empty:
    total_students_latest = df_filtered[df_filtered['Año'] == selected_years[1]]['Matricula'].sum()
    num_carreras = len(df_filtered['Carrera'].unique())
    avg_risk = df_filtered.drop_duplicates(subset=['Carrera'])['Riesgo_IA'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Estudiantes (Últ. Año)", f"{total_students_latest:,}")
    col2.metric("Carreras Analizadas", f"{num_carreras}")
    col3.metric("Riesgo IA Promedio", f"{avg_risk:.2f}", help="Promedio del riesgo IA (0=Bajo, 1=Alto) de las carreras seleccionadas.")
else:
    st.warning("No hay datos para los filtros seleccionados. Por favor, ajusta tu selección.")


# --- 7. Pestañas de Visualización ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Tendencias Históricas",
    "📊 Snapshot Actual y Riesgo",
    "🗺️ Riesgo vs Crecimiento",
    "🔮 Proyecciones 2035",
    "📄 Datos Detallados"
])

with tab1:
    st.header("Evolución de la Matrícula")
    if not df_filtered.empty:
        group_option = st.radio("Agrupar por:", ('Carrera', 'Area', 'Universidad'), horizontal=True)
        st.plotly_chart(plot_historical_trends(df_filtered, group_by=group_option), use_container_width=True)
    else:
        st.info("Selecciona datos para ver las tendencias.")

with tab2:
    st.header(f"Panorama Actual ({selected_years[1]}) y Riesgo IA")
    if not df_filtered.empty:
        st.plotly_chart(plot_latest_enrollment(df_filtered), use_container_width=True)
        st.markdown("""
        Este gráfico muestra la matrícula total por carrera en el último año seleccionado.
        El color indica el **nivel de riesgo asignado a cada carrera** (rojo = más alto).
        Carreras con barras altas y color rojo intenso podrían ser focos de atención.
        """)
    else:
        st.info("Selecciona datos para ver el snapshot.")

with tab3:
    st.header("Mapa de Riesgo vs. Crecimiento Histórico")
    if not df_filtered.empty and len(df_filtered['Año'].unique()) > 1:
        st.plotly_chart(plot_risk_vs_growth(df_filtered), use_container_width=True)
        st.markdown("""
        Este gráfico posiciona cada carrera según dos ejes:
        * **Eje X (Riesgo IA):** Más a la derecha, mayor riesgo de automatización/transformación.
        * **Eje Y (Crecimiento %):** Más arriba, mayor crecimiento histórico de matrícula.
        * **Tamaño de la burbuja:** Matrícula actual.

        **Cuadrantes clave:**
        * **Superior Izquierda:** Bajo riesgo, alto crecimiento (Ideal).
        * **Inferior Derecha:** Alto riesgo, bajo crecimiento/decrecimiento (¡Alerta!).
        * **Superior Derecha:** Alto riesgo, pero aún con crecimiento (Requiere adaptación).
        * **Inferior Izquierda:** Bajo riesgo, pero con decrecimiento (Otras causas?).
        """)
    else:
        st.info("Selecciona datos (con más de un año) para ver el mapa de riesgo.")

with tab4:
    st.header("Proyecciones Simplificadas hacia 2035")
    st.warning("⚠️ **¡Modelo Simplificado!** Estas proyecciones son *ilustrativas* y se basan en tendencias lineales ajustadas por un factor de riesgo IA. **NO son predicciones precisas** y no consideran factores complejos como cambios curriculares, políticas públicas o nuevas tecnologías.")
    if not df_filtered.empty and len(df_filtered['Año'].unique()) > 1:
        df_proj = generate_projection(df_filtered)
        st.dataframe(df_proj.style.format({
            'Riesgo IA': '{:.2f}',
            'Tendencia Hist. (Alumnos/Año)': '{:+.1f}',
            'Cambio Estimado (%)': '{:+.1f}%'
        }).background_gradient(cmap='Reds', subset=['Riesgo IA'])
          .background_gradient(cmap='RdYlGn', subset=['Cambio Estimado (%)'], vmin=-100, vmax=100)
          .background_gradient(cmap='coolwarm', subset=['Tendencia Hist. (Alumnos/Año)']),
          use_container_width=True
        )
    else:
        st.info("Selecciona datos (con más de un año) para ver las proyecciones.")


with tab5:
    st.header("Explorador de Datos Detallados")
    if not df_filtered.empty:
        st.dataframe(df_filtered)
        # Opción para descargar datos
        @st.cache_data
        def convert_df_to_csv(df):
           return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_filtered)
        st.download_button(
           label="📥 Descargar Datos Filtrados (CSV)",
           data=csv,
           file_name='matricula_puebla_filtrada.csv',
           mime='text/csv',
        )
    else:
        st.info("Selecciona datos para ver la tabla.")

# --- 8. Pie de Página ---
st.sidebar.markdown("---")
st.sidebar.info("""
    **Proyecto Estadístico - v0.2**
    Análisis del Futuro Laboral en Puebla.
    *Basado en datos simulados.*
    **(Necesita conexión a BBDD real)**
""")