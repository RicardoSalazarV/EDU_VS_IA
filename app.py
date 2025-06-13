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

# --- 2. LÓGICA INTERNA DE CLASIFICACIÓN (Basada en Palabras Clave) ---
AREA_KEYWORDS = {
    'Económico-Admin.': ['administración', 'contaduría', 'finanzas', 'economía', 'negocios', 'comercio', 'mercadotecnia', 'auditoría', 'empresas', 'aduanas', 'banca'],
    'Ciencias Sociales': ['derecho', 'sociales', 'sociología', 'antropología', 'políticas', 'relaciones internacionales', 'historia', 'comunicación', 'periodismo', 'trabajo social'],
    'Ingeniería y Tecnología': ['ingeniería', 'sistemas', 'software', 'computación', 'mecatrónica', 'electrónica', 'civil', 'industrial', 'tecnologías', 'mecánica', 'química', 'arquitectura', 'robótica', 'telecomunicaciones'],
    'Ciencias de la Salud': ['medicina', 'enfermería', 'odontología', 'estomatología', 'psicología', 'nutrición', 'terapia', 'fisioterapia', 'farmacia', 'químico', 'gerontología'],
    'Artes y Humanidades': ['diseño', 'arte', 'música', 'danza', 'filosofía', 'letras', 'literatura', 'humanidades', 'idiomas', 'lenguas', 'traducción', 'teatro', 'cinematografía', 'fotografía'],
    'Educación': ['educación', 'pedagogía', 'enseñanza', 'docencia', 'psicopedagogía', 'procesos educativos', 'puericultura'],
    'Ciencias Exactas y Naturales': ['física', 'matemáticas', 'biología', 'química', 'actuaria', 'geofísica', 'biotecnología', 'nanotecnología'],
    'Ciencias Agropecuarias': ['agronomía', 'agroindustrial', 'zootecnista', 'forestal', 'rural', 'agrícola', 'veterinaria']
}

CAREER_KEYWORDS = {
    'Administración': ['administración', 'empresas', 'negocios', 'gestión'], 'Contaduría': ['contaduría', 'contabilidad', 'auditoría', 'fiscal', 'finanzas'],
    'Derecho': ['derecho', 'jurídica', 'abogado', 'leyes', 'penales', 'notario'], 'Comunicación': ['comunicación', 'periodismo', 'medios'],
    'Psicología': ['psicología'], 'Arquitectura': ['arquitectura', 'urbanismo'], 'Medicina': ['medicina', 'médico', 'cirujano'], 'Enfermería': ['enfermería'],
    'Ingeniería en Software': ['software', 'sistemas', 'computación', 'informática', 'tecnologías de información'], 'Ingeniería Civil': ['civil'],
    'Ingeniería Industrial': ['industrial'], 'Ingeniería Mecatrónica': ['mecatrónica'], 'Ingeniería Mecánica': ['mecánica'], 'Ingeniería Química': ['química'],
    'Diseño Gráfico': ['diseño gráfico', 'diseño y comunicación', 'diseño multimedia'], 'Gastronomía': ['gastronomía', 'culinarias', 'chef'],
    'Turismo': ['turismo', 'hotelería', 'hospitalidad']
}

def classify_career(name):
    if not isinstance(name, str): return 'Otra', 'Otras'
    clean_name = name.lower().strip()
    
    found_career, found_area = name.capitalize(), 'Otras'
    
    for standard_name, keywords in CAREER_KEYWORDS.items():
        if any(keyword in clean_name for keyword in keywords):
            found_career = standard_name
            break

    for area, keywords in AREA_KEYWORDS.items():
        if any(keyword in clean_name for keyword in keywords):
            found_area = area
            break
            
    return found_career, found_area

# --- 3. Funciones de Carga y Procesamiento de Datos ---
@st.cache_data
def load_and_process_data(matricula_file, habilidades_file):
    try:
        df_matricula = pd.read_csv(matricula_file)
        df_habilidades = pd.read_csv(habilidades_file)
    except FileNotFoundError as e:
        st.error(f"Error: No se pudo encontrar el archivo {e.filename}. Asegúrate de que los archivos están en la misma carpeta.")
        return pd.DataFrame()

    df_matricula.columns = [col.strip().capitalize() for col in df_matricula.columns]
    if not all(col in df_matricula.columns for col in ['Año', 'Universidad', 'Carrera', 'Matricula']): return pd.DataFrame()
    df_matricula.dropna(subset=['Carrera', 'Universidad'], inplace=True)
    df_matricula['Universidad'] = df_matricula['Universidad'].astype(str).str.strip()
    df_matricula[['Carrera_Std', 'Area']] = df_matricula['Carrera'].apply(lambda x: pd.Series(classify_career(x)))
    df_matricula['Area'] = df_matricula['Area'].astype(str)
    df_matricula['Matricula'] = pd.to_numeric(df_matricula['Matricula'], errors='coerce').fillna(0).astype(int)
    df_matricula = df_matricula[df_matricula['Matricula'] > 0]

    df_habilidades.columns = [col.strip().upper() for col in df_habilidades.columns]
    df_habilidades.rename(columns={'LICENCIATURA': 'Carrera_Original', 'HABILIDADES': 'Habilidad'}, inplace=True)
    df_habilidades.dropna(subset=['Carrera_Original'], inplace=True)
    df_habilidades['Carrera_Std'] = df_habilidades['Carrera_Original'].apply(lambda x: classify_career(x)[0])

    skill_risk_map = {
        'Conciliaciones y Auditorías Básicas': 0.9, 'Preparación de Informes Estándar': 0.85, 'Diseño basado en plantillas': 0.8,
        'Tareas Transaccionales': 0.9, 'Diseño Técnico (BIM)': 0.7, 'Investigación de Mercados': 0.6, 'Análisis de Datos': 0.5,
        'Investigación Social': 0.5, 'Análisis Financiero': 0.55, 'Interpretación de Complejidades Normativas': 0.3,
        'Pensamiento Crítico Sistémico': 0.2, 'Análisis de Escenarios Complejos': 0.25, 'Comunicación Efectiva y Persuasiva': 0.15,
        'Ética y Juicio': 0.1, 'Resolución de Problemas No Estructurados': 0.2, 'Creatividad Original e Innovación': 0.1,
        'Empatía Cultural': 0.15, 'Comunicación Estratégica': 0.2, 'Visión Holística': 0.2, 'Negociación': 0.15, 'Liderazgo': 0.1,
        'Creatividad espacial': 0.15, 'Pensamiento de diseño': 0.2
    }
    df_habilidades['Riesgo_Habilidad'] = df_habilidades['Habilidad'].map(skill_risk_map).fillna(0.5)
    df_risk = df_habilidades.groupby('Carrera_Std')['Riesgo_Habilidad'].mean().reset_index()
    df_risk.rename(columns={'Riesgo_Habilidad': 'Riesgo_IA'}, inplace=True)

    df_final = pd.merge(df_matricula, df_risk, on='Carrera_Std', how='left')
    df_final['Riesgo_IA'] = df_final['Riesgo_IA'].fillna(0.5)
    return df_final


# --- 4. Funciones de Visualización y Proyección ---
def plot_historical_trends(df, group_by='Carrera_Std'):
    df_trend = df.groupby(['Año', group_by])['Matricula'].sum().reset_index()
    fig = px.line(df_trend, x='Año', y='Matricula', color=group_by, title=f'Evolución Histórica de la Matrícula por {group_by}', markers=True, labels={'Matricula': 'Número de Estudiantes', 'Carrera_Std': 'Carrera'})
    fig.update_layout(hovermode="x unified")
    return fig

def plot_latest_enrollment(df):
    df_latest = df[df['Año'] == df['Año'].max()]
    df_latest_grouped = df_latest.groupby('Carrera_Std').agg(Matricula_Total=('Matricula', 'sum'), Riesgo_IA=('Riesgo_IA', 'first')).reset_index()
    fig = px.bar(df_latest_grouped.sort_values('Matricula_Total', ascending=False), x='Carrera_Std', y='Matricula_Total', color='Riesgo_IA', title=f'Matrícula en {df["Año"].max()} y Nivel de Riesgo IA', labels={'Matricula_Total': 'Matrícula Total', 'Riesgo_IA': 'Riesgo IA (0-1)', 'Carrera_Std': 'Carrera'}, color_continuous_scale=px.colors.sequential.Reds, hover_data={'Riesgo_IA': ':.2f'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_risk_vs_growth(df):
    if df['Año'].nunique() < 2: return go.Figure(layout_title_text="Se necesita más de un año de datos para este gráfico.")
    start_year, end_year = df['Año'].min(), df['Año'].max()
    df_start = df[df['Año'] == start_year].groupby('Carrera_Std')['Matricula'].sum().reset_index().rename(columns={'Matricula': 'Matricula_Inicio'})
    df_end = df[df['Año'] == end_year].groupby('Carrera_Std')['Matricula'].sum().reset_index().rename(columns={'Matricula': 'Matricula_Fin'})
    df_agg = pd.merge(df_start, df_end, on='Carrera_Std')
    df_risk_unique = df[['Carrera_Std', 'Riesgo_IA']].drop_duplicates()
    df_agg = pd.merge(df_agg, df_risk_unique, on='Carrera_Std')
    num_years = end_year - start_year
    df_agg['CAGR'] = ((df_agg['Matricula_Fin'] / df_agg['Matricula_Inicio'])**(1/num_years) - 1) * 100 if num_years > 0 else 0
    df_agg = df_agg[(df_agg['Matricula_Inicio'] > 0) & (df_agg['Matricula_Fin'] > 0)]
    fig = px.scatter(df_agg, x='Riesgo_IA', y='CAGR', size='Matricula_Fin', color='Carrera_Std', title='Riesgo IA vs. Crecimiento Histórico (CAGR %)', labels={'Riesgo_IA': 'Nivel de Riesgo IA', 'CAGR': 'Crecimiento Anual Promedio (%)', 'Matricula_Fin': 'Matrícula Actual', 'Carrera_Std': 'Carrera'}, hover_name='Carrera_Std', size_max=60)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    if not df_agg.empty: fig.add_vline(x=df_agg['Riesgo_IA'].mean(), line_dash="dash", line_color="red", annotation_text="Riesgo Promedio")
    return fig

def generate_projection(df):
    if df['Año'].nunique() < 2: return pd.DataFrame()
    proyeccion_list = []
    df_riesgo = df[['Carrera_Std', 'Riesgo_IA']].drop_duplicates().set_index('Carrera_Std')
    df_grouped = df.groupby(['Año', 'Carrera_Std'])['Matricula'].sum().reset_index()
    for carrera in df_grouped['Carrera_Std'].unique():
        hist_data = df_grouped[df_grouped['Carrera_Std'] == carrera]
        if len(hist_data) < 2: continue
        z = np.polyfit(hist_data['Año'], hist_data['Matricula'], 1)
        tendencia_anual, matricula_actual = z[0], hist_data['Matricula'].iloc[-1]
        riesgo = df_riesgo.loc[carrera, 'Riesgo_IA']
        factor_ia = (1 - riesgo) * 0.5 + 0.5
        tendencia_ajustada = tendencia_anual * factor_ia if tendencia_anual > 0 else tendencia_anual * (1 + riesgo * 0.5)
        matricula_proyectada = max(0, int(matricula_actual + (tendencia_ajustada * 10)))
        proyeccion_list.append({'Carrera': carrera, f'Matrícula {df["Año"].max()}': matricula_actual, 'Riesgo IA': riesgo, 'Tendencia Hist. (Alumnos/Año)': round(tendencia_anual, 1), 'Matrícula Estimada 2035': matricula_proyectada, 'Cambio Estimado (%)': round(((matricula_proyectada / matricula_actual) - 1) * 100, 1) if matricula_actual > 0 else 0})
    return pd.DataFrame(proyeccion_list).sort_values('Cambio Estimado (%)')

# --- 5. Flujo Principal de la Aplicación ---
st.title("🎓🤖 Futuro Laboral Puebla: Análisis de Datos")
st.markdown("""
Análisis del impacto de la IA en carreras universitarias de Puebla. Según el Foro Económico Mundial, se proyecta una **disrupción del 40% de las habilidades laborales**, lo que exige una constante adaptación.
""")

HABILIDADES_FILE = "LICENCIATURAS - HABILIDADES.xlsx - Hoja1.csv"
MATRICULA_FILE = "matricula_puebla_tidy.csv"

df_data = load_and_process_data(MATRICULA_FILE, HABILIDADES_FILE)

if not df_data.empty:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Escudo_del_Estado_de_Puebla.svg/1200px-Escudo_del_Estado_de_Puebla.svg.png", width=100)
    st.sidebar.title("Panel de Control ⚙️")
    min_year, max_year = int(df_data['Año'].min()), int(df_data['Año'].max())
    if min_year >= max_year:
        st.sidebar.info(f"Mostrando datos para el único año disponible: {min_year}")
        selected_years = (min_year, max_year)
    else:
        selected_years = st.sidebar.slider('Rango de Años:', min_year, max_year, (min_year, max_year))
    
    all_universities = sorted(df_data['Universidad'].unique())
    selected_universities = st.sidebar.multiselect('Universidad(es):', all_universities, default=all_universities)
    all_areas = sorted(df_data['Area'].unique())
    selected_areas = st.sidebar.multiselect('Área(s) de Conocimiento:', all_areas, default=all_areas)
    carreras_in_areas = sorted(df_data[df_data['Area'].isin(selected_areas)]['Carrera_Std'].unique())
    selected_carreras = st.sidebar.multiselect('Carrera(s) Estandarizadas:', carreras_in_areas, default=carreras_in_areas)

    df_filtered = df_data[(df_data['Año'] >= selected_years[0]) & (df_data['Año'] <= selected_years[1]) & (df_data['Universidad'].isin(selected_universities)) & (df_data['Area'].isin(selected_areas)) & (df_data['Carrera_Std'].isin(selected_carreras))].copy()
    
    if not df_filtered.empty:
        total_students_latest = int(df_filtered[df_filtered['Año'] == selected_years[1]]['Matricula'].sum()) if selected_years[1] in df_filtered['Año'].values else 0
        num_carreras = len(df_filtered['Carrera_Std'].unique())
        avg_risk = df_filtered.drop_duplicates(subset=['Carrera_Std'])['Riesgo_IA'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Estudiantes (Últ. Año)", f"{total_students_latest:,}")
        col2.metric("Carreras Analizadas", f"{num_carreras}")
        col3.metric("Riesgo IA Promedio", f"{avg_risk:.2f}")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Tendencias", "📊 Snapshot Actual", "🗺️ Riesgo vs Crecimiento", "🔮 Proyecciones", "📊 Estadísticas", "📄 Datos Detallados"])
        
        with tab1:
            st.header("Evolución de la Matrícula")
            if min_year < max_year:
                group_option = st.radio("Agrupar por:", ('Carrera_Std', 'Area', 'Universidad'), horizontal=True, key='radio_trends')
                st.plotly_chart(plot_historical_trends(df_filtered, group_by=group_option), use_container_width=True)
            else:
                st.info("Se necesita más de un año de datos para mostrar tendencias.")
        
        with tab2:
            st.header(f"Panorama Actual ({selected_years[1]}) y Riesgo IA")
            st.plotly_chart(plot_latest_enrollment(df_filtered), use_container_width=True)
            st.markdown("---")
            st.subheader("Insights Clave por Área")
            col1, col2 = st.columns(2)
            with col1:
                st.info("#### Económico-Administrativas\nEl rol evoluciona de transaccional a **asesor estratégico**. La IA automatiza tareas rutinarias, demandando profesionales que interpreten datos complejos y asesoren a la alta dirección. El pensamiento crítico y la ética son cruciales.")
            with col2:
                st.success("#### Humanidades y Ciencias Sociales\nSon cada vez más valoradas para roles de **ética de la IA, análisis cultural y comunicación estratégica**. Aportan el pensamiento crítico y la visión holística que la IA no posee para entender y mitigar el impacto social de la tecnología.")

        with tab3:
            st.header("Mapa de Riesgo vs. Crecimiento Histórico")
            if min_year < max_year:
                st.plotly_chart(plot_risk_vs_growth(df_filtered), use_container_width=True)
                st.markdown("""
                **Interpretación de Cuadrantes:**
                - **Superior Izquierda (Zona Ideal):** Bajo riesgo y alto crecimiento. Carreras con futuro prometedor.
                - **Inferior Derecha (Zona de Alerta):** Alto riesgo y bajo crecimiento/decrecimiento. Requieren una reinvención urgente.
                - **Superior Derecha (Zona de Adaptación):** Alto riesgo, pero aún con crecimiento. Deben adaptar sus planes de estudio para incorporar la IA como herramienta.
                - **Inferior Izquierda (Zona de Observación):** Bajo riesgo, pero con decrecimiento. Su declive puede deberse a otros factores del mercado.
                """)
            else:
                st.info("Se necesita más de un año de datos para calcular el crecimiento.")
        
        with tab4:
            st.header("Proyecciones Simplificadas hacia 2035")
            st.warning("⚠️ **Modelo Ilustrativo:** Estas proyecciones se basan en tendencias lineales ajustadas por el factor de riesgo IA. No son predicciones precisas.")
            if min_year < max_year:
                df_proj = generate_projection(df_filtered)
                st.dataframe(df_proj.style.format({'Riesgo IA': '{:.2f}', 'Tendencia Hist. (Alumnos/Año)': '{:+.1f}', 'Cambio Estimado (%)': '{:+.1f}%'}).background_gradient(cmap='Reds', subset=['Riesgo IA']).background_gradient(cmap='RdYlGn', subset=['Cambio Estimado (%)'], vmin=-100, vmax=100), use_container_width=True)
            else:
                st.info("Se necesita más de un año de datos para generar proyecciones.")
        
        with tab5:
            st.header("Estadísticas Descriptivas y Distribuciones")
            st.markdown("Analiza las características numéricas de los datos filtrados.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Resumen Estadístico")
                stat_selection = st.selectbox("Elegir variable para describir:", ['Matricula', 'Riesgo_IA'])
                st.dataframe(df_filtered[stat_selection].describe())
            
            with col2:
                st.subheader("Distribución del Riesgo IA")
                df_unique_risk = df_filtered.drop_duplicates(subset=['Carrera_Std'])
                fig_hist_risk = px.histogram(df_unique_risk, x='Riesgo_IA', nbins=20, title="Frecuencia de Niveles de Riesgo")
                st.plotly_chart(fig_hist_risk, use_container_width=True)
                st.info("Una concentración a la derecha (valores > 0.6) sugiere un panorama de mayor disrupción para el conjunto de carreras seleccionado.")

            st.markdown("---")
            st.subheader("Distribución de la Matrícula")
            fig_hist_mat = px.histogram(df_filtered, x='Matricula', nbins=50, title="Frecuencia del Tamaño de Matrículas")
            st.plotly_chart(fig_hist_mat, use_container_width=True)
            st.info("Este histograma muestra si la mayoría de las entradas de matrícula son pequeñas o grandes. Una cola larga a la derecha indica la presencia de carreras con matrículas muy altas en comparación con el resto.")

        with tab6:
            st.header("Explorador de Datos Detallados")
            st.info("Aquí puedes ver la tabla de datos completa con las clasificaciones y filtros aplicados.")
            st.dataframe(df_filtered[['Año', 'Universidad', 'Carrera', 'Carrera_Std', 'Area', 'Matricula', 'Riesgo_IA']])
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar Datos Filtrados (CSV)", data=csv, file_name='matricula_puebla_filtrada.csv', mime='text/csv')
    else:
        st.warning("No se encontraron datos para los filtros seleccionados.")
else:
    st.error("No se pudieron cargar o procesar los archivos de datos. Verifica los nombres y el formato de los archivos CSV.")