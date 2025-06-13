import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Futuro Laboral Puebla: IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LÓGICA DE CLASIFICACIÓN Y PROCESAMIENTO ---

# Diccionarios para estandarización
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

@st.cache_data
def load_and_process_data(matricula_file, tareas_file):
    try:
        df_matricula = pd.read_csv(matricula_file)
        df_tareas = pd.read_csv(tareas_file, encoding='utf-8-sig')
    except FileNotFoundError as e:
        st.error(f"Error: No se pudo encontrar el archivo {e.filename}. Asegúrate de que los archivos están en la misma carpeta.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Procesamiento Matrícula ---
    df_matricula.columns = [col.strip().capitalize() for col in df_matricula.columns]
    if not all(col in df_matricula.columns for col in ['Año', 'Universidad', 'Carrera', 'Matricula']):
        st.error("El archivo de matrícula debe contener las columnas: Año, Universidad, Carrera, Matricula")
        return pd.DataFrame(), pd.DataFrame()
    df_matricula.dropna(subset=['Carrera', 'Universidad'], inplace=True)
    df_matricula['Universidad'] = df_matricula['Universidad'].astype(str).str.strip()
    df_matricula[['Carrera_Std', 'Area']] = df_matricula['Carrera'].apply(lambda x: pd.Series(classify_career(x)))
    df_matricula['Area'] = df_matricula['Area'].astype(str)
    df_matricula['Matricula'] = pd.to_numeric(df_matricula['Matricula'], errors='coerce').fillna(0).astype(int)
    df_matricula = df_matricula[df_matricula['Matricula'] > 0]

    # --- Procesamiento Tareas ---
    df_tareas.columns = [col.strip().upper() for col in df_tareas.columns]
    df_tareas.rename(columns={'LICENCIATURA': 'CARRERA_ORIGINAL'}, inplace=True)
    required_cols = ['CARRERA_ORIGINAL', 'TAREA', 'PESO', 'RIESGO_TAREA']
    if not all(col in df_tareas.columns for col in required_cols):
        st.error(f"El archivo de tareas debe contener las columnas: {', '.join(required_cols)}. Columnas encontradas: {list(df_tareas.columns)}")
        return pd.DataFrame(), pd.DataFrame()
    df_tareas.dropna(subset=required_cols, inplace=True)
    df_tareas['Carrera_Std'] = df_tareas['CARRERA_ORIGINAL'].apply(lambda x: classify_career(x)[0])

    # --- Calcular Riesgo IA Ponderado ---
    df_tareas['RIESGO_TAREA'] = pd.to_numeric(df_tareas['RIESGO_TAREA'], errors='coerce')
    df_tareas['PESO'] = pd.to_numeric(df_tareas['PESO'], errors='coerce')
    df_tareas.dropna(subset=['RIESGO_TAREA', 'PESO'], inplace=True)
    def weighted_average(x):
        try:
            return np.average(x['RIESGO_TAREA'], weights=x['PESO'])
        except ZeroDivisionError:
            return np.nan
    df_risk = df_tareas.groupby('Carrera_Std').apply(weighted_average).reset_index(name='Riesgo_IA')
    
    # --- Unión final ---
    df_final = pd.merge(df_matricula, df_risk, on='Carrera_Std', how='left')
    df_final['Riesgo_IA'] = df_final['Riesgo_IA'].fillna(df_final['Riesgo_IA'].median())
    
    return df_final, df_tareas

# --- 3. FUNCIONES DE VISUALIZACIÓN Y CÁLCULO ---
def plot_historical_trends(df, group_by='Carrera_Std'):
    df_trend = df.groupby(['Año', group_by])['Matricula'].sum().reset_index()
    fig = px.line(df_trend, x='Año', y='Matricula', color=group_by, 
                  title=f'Evolución Histórica de la Matrícula por {group_by.replace("_Std", "")}', 
                  markers=True, labels={'Matricula': 'Número de Estudiantes', 'Carrera_Std': 'Carrera'})
    fig.update_layout(hovermode="x unified", legend_title_text=group_by.replace("_Std", "").capitalize())
    return fig

def plot_latest_enrollment(df):
    latest_year = df['Año'].max()
    df_latest = df[df['Año'] == latest_year]
    df_latest_grouped = df_latest.groupby('Carrera_Std').agg(Matricula_Total=('Matricula', 'sum'), Riesgo_IA=('Riesgo_IA', 'first')).reset_index()
    fig = px.bar(df_latest_grouped.sort_values('Matricula_Total', ascending=False), 
                 x='Carrera_Std', y='Matricula_Total', color='Riesgo_IA', 
                 title=f'Matrícula en {latest_year} y Nivel de Riesgo IA Ponderado', 
                 labels={'Matricula_Total': 'Matrícula Total', 'Riesgo_IA': 'Riesgo IA (0-1)', 'Carrera_Std': 'Carrera'},
                 color_continuous_scale=px.colors.sequential.Reds, hover_name='Carrera_Std',
                 hover_data={'Riesgo_IA': ':.2f'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Total de Estudiantes")
    return fig

def calculate_growth_data(df):
    if df['Año'].nunique() < 2: return pd.DataFrame()
    start_year, end_year = df['Año'].min(), df['Año'].max()
    df_start = df[df['Año'] == start_year].groupby('Carrera_Std')['Matricula'].sum().reset_index().rename(columns={'Matricula': 'Matricula_Inicio'})
    df_end = df[df['Año'] == end_year].groupby('Carrera_Std')['Matricula'].sum().reset_index().rename(columns={'Matricula': 'Matricula_Fin'})
    df_agg = pd.merge(df_start, df_end, on='Carrera_Std')
    df_risk_unique = df[['Carrera_Std', 'Riesgo_IA']].drop_duplicates()
    df_agg = pd.merge(df_agg, df_risk_unique, on='Carrera_Std')
    
    num_years = end_year - start_year
    if num_years > 0:
        df_agg['CAGR'] = ((df_agg['Matricula_Fin'] / df_agg['Matricula_Inicio'])**(1/num_years) - 1) * 100
    else:
        df_agg['CAGR'] = 0
    df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_agg.dropna(subset=['CAGR'], inplace=True)
    return df_agg

# --- 4. FLUJO PRINCIPAL DE LA APLICACIÓN ---
st.title("🎓🤖 Futuro Laboral en Puebla: Un Análisis Basado en Datos")
st.markdown("""
Esta plataforma interactiva analiza datos de **matrículas universitarias en Puebla** y los cruza con un **índice de riesgo de automatización por IA** para generar insights sobre el futuro del panorama profesional en el estado.
El objetivo es servir como una herramienta para estudiantes, educadores y responsables de políticas públicas en la toma de decisiones estratégicas.
""")
st.info("""
**Metodología de Riesgo:** El 'Riesgo IA' (de 0 a 1) no mide la probabilidad de que una carrera "desaparezca", sino el grado en que sus **tareas constitutivas** son susceptibles a la automatización. Se calcula usando un **promedio ponderado**, dando más importancia a las tareas que son centrales para cada profesión. Un riesgo alto no implica obsolescencia, sino una necesidad imperativa de **transformación y adaptación de habilidades**.
""")

# Carga de datos
TAREAS_FILE = "tareas_riesgo_ia.csv" 
MATRICULA_FILE = "matricula_puebla_tidy.csv"
df_data, df_tareas = load_and_process_data(MATRICULA_FILE, TAREAS_FILE)

if not df_data.empty and not df_tareas.empty:
    # --- PANEL LATERAL DE FILTROS ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Escudo_del_Estado_de_Puebla.svg/1200px-Escudo_del_Estado_de_Puebla.svg.png", width=100)
    st.sidebar.title("Panel de Control ⚙️")

    min_year, max_year = int(df_data['Año'].min()), int(df_data['Año'].max())
    if min_year >= max_year:
        st.sidebar.info(f"Mostrando datos para el único año disponible: {min_year}")
        selected_years = (min_year, max_year)
    else:
        selected_years = st.sidebar.slider('Selecciona el Rango de Años a Analizar:', min_year, max_year, (min_year, max_year))
    
    all_universities = sorted(df_data['Universidad'].unique())
    selected_universities = st.sidebar.multiselect('Filtra por Universidad(es):', all_universities, default=all_universities)
    
    all_areas = sorted(df_data['Area'].unique())
    selected_areas = st.sidebar.multiselect('Filtra por Área(s) de Conocimiento:', all_areas, default=all_areas)
    
    carreras_in_areas = sorted(df_data[df_data['Area'].isin(selected_areas)]['Carrera_Std'].unique())
    selected_carreras = st.sidebar.multiselect('Filtra por Carrera(s):', carreras_in_areas, default=carreras_in_areas)

    # Filtrado del DataFrame principal
    df_filtered = df_data[(df_data['Año'] >= selected_years[0]) & (df_data['Año'] <= selected_years[1]) & (df_data['Universidad'].isin(selected_universities)) & (df_data['Area'].isin(selected_areas)) & (df_data['Carrera_Std'].isin(selected_carreras))].copy()
    
    if not df_filtered.empty:
        # --- MÉTRICAS PRINCIPALES ---
        last_year_selected = selected_years[1]
        df_latest_year_risk = df_filtered[df_filtered['Año'] == last_year_selected]
        total_students_latest = int(df_latest_year_risk['Matricula'].sum())
        num_carreras = len(df_filtered['Carrera_Std'].unique())
        
        weighted_avg_risk = np.average(df_latest_year_risk['Riesgo_IA'], weights=df_latest_year_risk['Matricula']) if not df_latest_year_risk.empty else 0

        st.markdown("### Métricas Clave para la Selección Actual")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Estudiantes ({last_year_selected})", f"{total_students_latest:,}")
        col2.metric("Carreras Analizadas", f"{num_carreras}")
        col3.metric("Riesgo IA Ponderado", f"{weighted_avg_risk:.2f}", help="El riesgo promedio de la selección, ponderado por el número de estudiantes en cada carrera. Un valor más alto indica mayor exposición a la automatización.")

        # --- PANELES DE NAVEGACIÓN ---
        tab_list = ["📈 Panorama General", "🗺️ Riesgo vs. Crecimiento", "🎓 Perfil por Universidad", "🏛️ Cartera Universitaria", "🛠️ Planificador de Carrera", "📊 Análisis Estadístico", "📄 Datos Detallados"]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)
        
        # --- PESTAÑA 1: PANORAMA GENERAL ---
        with tab1:
            st.header("📈 Panorama General y Tendencias de Matrícula")
            st.markdown("""
            **Descripción Académica:** Esta sección presenta una visión macro del panorama educativo. El gráfico de barras ofrece una "fotografía" del último año, permitiendo identificar rápidamente las carreras con mayor población estudiantil y su correspondiente nivel de riesgo de automatización. El gráfico de líneas, por otro lado, analiza la evolución temporal de la matrícula, lo cual es fundamental para la planificación estratégica al identificar patrones de crecimiento, estancamiento o declive que pueden (o no) estar correlacionados con la percepción del futuro laboral de dichas profesiones.
            """)
            st.plotly_chart(plot_latest_enrollment(df_filtered), use_container_width=True)
            st.markdown("---")
            if min_year < max_year:
                st.subheader("Evolución Histórica de la Matrícula")
                group_option = st.radio("Agrupar tendencias por:", ('Carrera_Std', 'Area', 'Universidad'), horizontal=True, key='radio_trends')
                st.plotly_chart(plot_historical_trends(df_filtered, group_by=group_option), use_container_width=True)

        # --- PESTAÑA 2: RIESGO VS. CRECIMIENTO ---
        with tab2:
            st.header("🗺️ Mapa de Riesgo vs. Crecimiento (Matriz de Posicionamiento Estratégico)")
            st.markdown("""
            **Descripción Académica:** Este es uno de los análisis más importantes para la toma de decisiones estratégicas. El gráfico posiciona cada carrera en una matriz de dos ejes: el **Riesgo de Automatización por IA** (eje X) y la **Tasa de Crecimiento Anual Compuesta (TCAC/CAGR)** de su matrícula (eje Y). La TCAC es una métrica estándar en finanzas y economía para determinar la tasa de retorno de una inversión a lo largo de un período. Aquí, la usamos para medir el "interés" sostenido en una carrera.
            
            **El Camino a este Resultado:** La TCAC se calcula usando la matrícula del primer y último año del rango seleccionado. La fórmula es:
            $$ TCAC = \\left( \\frac{\\text{Matrícula Final}}{\\text{Matrícula Inicial}} \\right)^{\\frac{1}{\\text{N° de Años}}} - 1 $$
            Este valor se grafica contra el 'Riesgo_IA' de cada carrera para generar la matriz de posicionamiento.
            """)
            if df_filtered['Año'].nunique() > 1:
                df_growth = calculate_growth_data(df_filtered)
                if not df_growth.empty:
                    view_type = st.radio("Seleccionar tipo de vista:", ("Burbujas (Dispersión)", "Mapa de Calor (Densidad)"), horizontal=True, key='risk_view')
                    if view_type == "Burbujas (Dispersión)":
                        st.markdown("**Interpretación:** Cada burbuja es una carrera. El tamaño representa la matrícula actual. Las líneas de promedio dividen el gráfico en cuatro cuadrantes estratégicos.")
                        fig = px.scatter(df_growth, x='Riesgo_IA', y='CAGR', size='Matricula_Fin', color='Carrera_Std',
                                       title='Riesgo IA vs. Tasa de Crecimiento Anual Compuesta (TCAC)',
                                       labels={'Riesgo_IA': 'Nivel de Riesgo IA (0=Bajo, 1=Alto)', 'CAGR': 'Crecimiento Anual Promedio (%)', 'Matricula_Fin': 'Matrícula Actual'},
                                       hover_name='Carrera_Std', size_max=60)
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        mean_risk = df_growth['Riesgo_IA'].mean()
                        fig.add_vline(x=mean_risk, line_dash="dash", line_color="red", annotation_text=f"Riesgo Promedio ({mean_risk:.2f})")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("**Interpretación:** Este mapa de calor muestra la concentración de carreras. Las áreas más 'calientes' (amarillas) indican que un gran número de carreras comparten esa combinación específica de riesgo y crecimiento. Es útil para ver tendencias generales en lugar de casos individuales.")
                        fig = px.density_heatmap(df_growth, x="Riesgo_IA", y="CAGR", marginal_x="histogram", marginal_y="histogram",
                                                 title="Mapa de Densidad de Carreras por Riesgo y Crecimiento",
                                                 labels={'Riesgo_IA': 'Nivel de Riesgo IA (0=Bajo, 1=Alto)', 'CAGR': 'Crecimiento Anual Promedio (%)'})
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                    **Análisis de Cuadrantes:**
                    - **🟢 Superior Izquierda (Zona de Fortaleza):** Bajo riesgo y alto crecimiento. Carreras idealmente posicionadas para el futuro inmediato.
                    - **🟡 Superior Derecha (Zona de Adaptación Urgente):** Alto riesgo, pero con crecimiento en la demanda. Requieren una transformación curricular para incorporar habilidades humanas complementarias a la IA.
                    - **🔵 Inferior Izquierda (Zona de Nicho o Reinvención):** Bajo riesgo, pero con decrecimiento. Pueden ser carreras de nicho o que necesitan una mejor propuesta de valor.
                    - **🔴 Inferior Derecha (Zona de Alerta Crítica):** Alto riesgo y bajo (o negativo) crecimiento. Requieren una reinvención fundamental o podrían enfrentar obsolescencia.
                    """)
                else:
                    st.info("No hay suficientes datos para calcular el crecimiento en la selección actual.")
            else:
                st.info("Se necesita un rango de más de un año para calcular la tasa de crecimiento.")

        # --- PESTAÑA 3: PERFIL POR UNIVERSIDAD ---
        with tab3:
            st.header("🎓 Perfil Individual por Universidad")
            st.markdown("""
            **Descripción Académica:** Este panel permite un análisis a nivel micro, enfocándose en la oferta académica y el perfil de riesgo de una sola institución. Es una herramienta de autoevaluación para que las universidades analicen la composición de su portafolio de carreras, identifiquen sus áreas de mayor riesgo y concentración de matrícula, y fundamenten decisiones de inversión, rediseño curricular o lanzamiento de nuevos programas.
            """)
            univ_selection = st.selectbox("Selecciona una Universidad para un Análisis Detallado:", options=all_universities, key='uni_select')
            if univ_selection:
                df_univ = df_filtered[df_filtered['Universidad'] == univ_selection].copy()
                st.subheader(f"Análisis para: {univ_selection}")
                if not df_univ.empty:
                    df_latest_univ = df_univ[df_univ['Año'] == df_univ['Año'].max()]
                    univ_students = int(df_latest_univ['Matricula'].sum())
                    univ_risk = np.average(df_latest_univ['Riesgo_IA'], weights=df_latest_univ['Matricula']) if not df_latest_univ.empty else 0
                    c1, c2 = st.columns(2)
                    c1.metric("Total de Estudiantes (Últ. Año)", f"{univ_students:,}")
                    c2.metric("Riesgo IA Ponderado de su Oferta", f"{univ_risk:.2f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Distribución de Matrícula por Área")
                        df_area_dist = df_latest_univ.groupby('Area')['Matricula'].sum().reset_index()
                        fig_pie = px.pie(df_area_dist, names='Area', values='Matricula', hole=0.3, title="Proporción de Alumnos por Área")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.markdown("##### Cartera de Carreras por Riesgo IA")
                        df_risk_dist = df_latest_univ.groupby('Carrera_Std')['Riesgo_IA'].mean().reset_index().sort_values('Riesgo_IA', ascending=True)
                        fig_bar_risk = px.bar(df_risk_dist, x='Riesgo_IA', y='Carrera_Std', orientation='h', color='Riesgo_IA', 
                                              color_continuous_scale=px.colors.sequential.Reds, title="Riesgo IA de la Oferta Académica")
                        fig_bar_risk.update_layout(yaxis_title="", xaxis_title="Nivel de Riesgo IA")
                        st.plotly_chart(fig_bar_risk, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para '{univ_selection}' con los filtros actuales.")

        # --- PESTAÑA 4: CARTERA UNIVERSITARIA ---
        with tab4:
            st.header("🏛️ Análisis Comparativo de Carteras Universitarias")
            st.markdown("""
            **Descripción Académica:** Este panel realiza un análisis a nivel macro, comparando el perfil de riesgo agregado de las universidades seleccionadas. Se calcula el **Riesgo de Cartera Ponderado por Matrícula** para cada institución, una métrica que representa la exposición general de su cuerpo estudiantil a la automatización. Una universidad con un riesgo de cartera más bajo puede considerarse, en teoría, como una que posiciona mejor a sus estudiantes para un futuro colaborativo con la IA. Este análisis es vital para responsables de políticas públicas y para la evaluación comparativa (benchmarking) entre instituciones.
            """)
            df_latest_year = df_filtered[df_filtered['Año'] == df_filtered['Año'].max()].copy()
            if not df_latest_year.empty:
                df_latest_year['Matricula_x_Riesgo'] = df_latest_year['Matricula'] * df_latest_year['Riesgo_IA']
                df_univ_risk = df_latest_year.groupby('Universidad').agg(Total_Matricula_x_Riesgo=('Matricula_x_Riesgo', 'sum'), Total_Matricula=('Matricula', 'sum')).reset_index()
                df_univ_risk['Riesgo_Cartera'] = df_univ_risk['Total_Matricula_x_Riesgo'] / df_univ_risk['Total_Matricula']
                fig_univ_risk = px.bar(df_univ_risk.sort_values('Riesgo_Cartera', ascending=False), x='Riesgo_Cartera', y='Universidad',
                                     orientation='h', color='Riesgo_Cartera', color_continuous_scale=px.colors.sequential.Reds_r,
                                     title=f"Riesgo IA Ponderado de la Cartera Universitaria ({df_latest_year['Año'].max()})",
                                     labels={'Riesgo_Cartera': 'Riesgo IA Ponderado de Cartera', 'Universidad': ''})
                fig_univ_risk.update_layout(yaxis_title="")
                st.plotly_chart(fig_univ_risk, use_container_width=True)
            else:
                st.info("No hay datos disponibles para el último año con los filtros seleccionados.")

        # --- PESTAÑA 5: PLANIFICADOR DE CARRERA ---
        with tab5:
            st.header("🛠️ Planificador Estratégico de Carrera (Reskilling y Upskilling)")
            st.markdown("""
            **Descripción Académica:** Esta herramienta traslada el análisis del riesgo a la acción individual. Se basa en los conceptos de **Upskilling** (profundizar habilidades existentes) y **Reskilling** (adquirir nuevas habilidades). Al seleccionar una carrera, el sistema identifica sus tareas más vulnerables y permite al usuario simular la adición de competencias resilientes (aquellas con bajo riesgo de automatización, que suelen requerir creatividad, pensamiento crítico o inteligencia social). El objetivo es visualizar cuantitativamente cómo la formación continua puede fortalecer un perfil profesional, promoviendo una mentalidad de aprendizaje permanente como estrategia de carrera.
            """)
            available_carreras = sorted(df_filtered['Carrera_Std'].unique())
            if available_carreras:
                base_career = st.selectbox("1. Selecciona tu carrera base o de interés:", options=available_carreras, key="planner_career")
                if base_career:
                    df_career_tasks = df_tareas[df_tareas['Carrera_Std'] == base_career].copy()
                    current_risk = df_filtered[df_filtered['Carrera_Std'] == base_career]['Riesgo_IA'].iloc[0]
                    st.metric(f"Riesgo IA Actual para {base_career}", f"{current_risk:.2%}", help="Calculado como el promedio ponderado de las tareas actuales de la carrera.")
                    st.markdown("**Tareas más vulnerables de esta carrera (mayor riesgo de automatización):**")
                    st.dataframe(df_career_tasks.sort_values('RIESGO_TAREA', ascending=False).head(5)[['TAREA', 'RIESGO_TAREA']])
                    st.markdown("---")
                    st.subheader("2. Fortalece tu perfil con Habilidades Resilientes")
                    st.info("Selecciona nuevas tareas o habilidades de bajo riesgo que te gustaría adquirir. Observa cómo cambia tu riesgo.")
                    resilient_tasks = df_tareas[(df_tareas['RIESGO_TAREA'] <= 0.3) & (df_tareas['Carrera_Std'] != base_career)].drop_duplicates(subset=['TAREA'])
                    selected_new_tasks = st.multiselect("Selecciona habilidades de otras áreas:", options=sorted(resilient_tasks['TAREA'].unique()))
                    if selected_new_tasks:
                        new_tasks_df = resilient_tasks[resilient_tasks['TAREA'].isin(selected_new_tasks)].copy()
                        if not df_career_tasks.empty:
                            new_tasks_df['PESO'] = df_career_tasks['PESO'].mean() 
                        else:
                            new_tasks_df['PESO'] = 3
                        combined_tasks_df = pd.concat([df_career_tasks, new_tasks_df], ignore_index=True)
                        simulated_risk = np.average(combined_tasks_df['RIESGO_TAREA'], weights=combined_tasks_df['PESO'])
                        delta = simulated_risk - current_risk
                        st.metric(f"Nuevo Riesgo IA Simulado para {base_career}", f"{simulated_risk:.2%}", f"{delta:.2%}")
                        st.success("¡Felicidades! Al diversificar tus habilidades, has construido un perfil profesional más robusto.")
            else:
                st.warning("No hay carreras disponibles en la selección actual para analizar. Ajusta los filtros.")

        # --- PESTAÑA 6: ANÁLISIS ESTADÍSTICO ---
        with tab6:
            st.header("📊 Análisis Estadístico Profundo")
            st.markdown("""
            **Descripción Académica:** Esta sección ofrece herramientas de estadística descriptiva y exploratoria para un análisis técnico. Los histogramas revelan la distribución de frecuencias de las variables clave, mientras que los diagramas de caja (box plots) son excelentes para comparar estas distribuciones entre categorías, mostrando mediana, cuartiles y valores atípicos. Finalmente, el análisis de correlación investiga la relación lineal entre dos variables continuas.
            """)
            st.subheader("Histogramas de Distribución General")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Frecuencia de Niveles de Riesgo**")
                df_unique_risk = df_filtered.drop_duplicates(subset=['Carrera_Std'])
                fig_hist_risk = px.histogram(df_unique_risk, x='Riesgo_IA', nbins=20)
                st.plotly_chart(fig_hist_risk, use_container_width=True)
            with col2:
                st.markdown("**Frecuencia del Tamaño de Matrículas (Último Año)**")
                fig_hist_mat = px.histogram(df_filtered[df_filtered['Año'] == df_filtered['Año'].max()], x='Matricula', nbins=40)
                st.plotly_chart(fig_hist_mat, use_container_width=True)

            st.markdown("---")
            st.subheader("Análisis Comparativo de Distribuciones por Área (Box Plots)")
            st.markdown("""
            **Interpretación de los Diagramas de Caja:** La **línea central** en la caja es la mediana (el valor del medio, percentil 50). La **caja** representa el rango intercuartílico (del percentil 25 al 75), es decir, el 50% central de los datos. Los **"bigotes"** (líneas) se extienden para mostrar el rango de los datos, excluyendo valores atípicos, los cuales se grafican como puntos individuales.
            """)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Distribución del Riesgo IA por Área**")
                df_unique_careers = df_filtered.drop_duplicates(subset=['Carrera_Std'])
                fig_box_risk = px.box(df_unique_careers, x='Area', y='Riesgo_IA', color='Area')
                st.plotly_chart(fig_box_risk, use_container_width=True)
            with col4:
                st.markdown("**Distribución de la Matrícula por Área**")
                df_latest_year_box = df_filtered[df_filtered['Año'] == df_filtered['Año'].max()]
                fig_box_mat = px.box(df_latest_year_box, x='Area', y='Matricula', color='Area', points="outliers")
                st.plotly_chart(fig_box_mat, use_container_width=True)

            st.markdown("---")
            st.subheader("Análisis de Correlación: Riesgo vs. Crecimiento")
            st.markdown("""
            **Interpretación Académica:** Se utiliza una regresión por Mínimos Cuadrados Ordinarios (OLS) para modelar la relación entre el Riesgo IA (variable independiente) y la TCAC de la matrícula (variable dependiente). El valor **R-cuadrado ($R^2$)** indica la proporción de la varianza en el crecimiento de la matrícula que puede ser explicada por el nivel de riesgo. Un $R^2$ cercano a 1 sugiere una fuerte relación lineal, mientras que uno cercano a 0 indica una relación débil o nula. El **coeficiente (coef)** de la variable 'Riesgo_IA' indica la pendiente: si es negativo, sugiere que a mayor riesgo, menor es el crecimiento de la matrícula.
            """)
            if df_filtered['Año'].nunique() > 1:
                df_growth_corr = calculate_growth_data(df_filtered)
                if not df_growth_corr.empty:
                    fig_corr = px.scatter(df_growth_corr, x='Riesgo_IA', y='CAGR', trendline="ols", trendline_color_override="red",
                                        title="Correlación entre Riesgo IA y Crecimiento de Matrícula (TCAC)",
                                        labels={'Riesgo_IA': 'Nivel de Riesgo IA', 'CAGR': 'Tasa de Crecimiento Anual (%)'},
                                        hover_name='Carrera_Std')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    try:
                        results = px.get_trendline_results(fig_corr)
                        st.write("Resultados del modelo de regresión (OLS):")
                        st.dataframe(results.px_fit_results.summary().tables[1])
                    except Exception:
                        st.warning("No se pudieron calcular los resultados de la regresión, posiblemente por datos insuficientes.")
                else:
                    st.info("No hay suficientes datos de crecimiento para este análisis.")
            else:
                st.info("Se necesita un rango de más de un año para el análisis de correlación.")

        # --- PESTAÑA 7: DATOS DETALLADOS ---
        with tab7:
            st.header("📄 Explorador de Datos y Transparencia")
            st.markdown("""
            **Descripción Académica:** La reproducibilidad es un pilar fundamental del análisis cuantitativo. Esta sección proporciona acceso directo y sin procesar a la tabla de datos utilizada para todas las visualizaciones y cálculos, según los filtros aplicados. Permitir la descarga de los datos fomenta la transparencia, la verificación por parte de terceros y la posibilidad de realizar análisis secundarios.
            """)
            st.dataframe(df_filtered[['Año', 'Universidad', 'Carrera', 'Carrera_Std', 'Area', 'Matricula', 'Riesgo_IA']].sort_values(by=['Año', 'Universidad', 'Carrera_Std']))
            
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Descargar Datos Filtrados (CSV)", data=csv,
                               file_name='datos_filtrados_futuro_laboral_puebla.csv',
                               mime='text/csv')
            
    else:
        st.warning("No se encontraron datos para los filtros seleccionados. Por favor, ajusta los filtros en el panel de la izquierda.")
else:
    st.error("Error crítico: No se pudieron cargar o procesar los archivos de datos. Verifica que los archivos 'matricula_puebla_tidy.csv' y 'tareas_riesgo_ia.csv' estén en la misma carpeta que la aplicación.")