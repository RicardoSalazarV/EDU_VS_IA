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

# <<< MEJORA/NUEVO: FUNCIÓN PARA PROYECCIONES >>>
def project_enrollment(df, careers, target_year=2035):
    """
    Proyecta la matrícula para una lista de carreras usando regresión lineal.
    """
    projection_df = pd.DataFrame()
    
    for career in careers:
        career_df = df[df['Carrera_Std'] == career].groupby('Año')['Matricula'].sum().reset_index()
        
        # Necesitamos al menos 2 puntos de datos para una línea
        if len(career_df) < 2:
            continue
            
        # Ajuste de modelo lineal (y = mx + b)
        model = np.polyfit(career_df['Año'], career_df['Matricula'], 1)
        slope, intercept = model[0], model[1]
        
        # Generar años futuros
        last_year = int(career_df['Año'].max())
        future_years = np.arange(last_year + 1, target_year + 1)
        
        # Calcular proyección
        projected_enrollment = slope * future_years + intercept
        
        # Evitar matrículas negativas
        projected_enrollment = np.maximum(0, projected_enrollment).astype(int)
        
        # Crear dataframes para combinar
        hist_part = career_df.copy()
        hist_part['Tipo'] = 'Histórico'
        
        proj_part = pd.DataFrame({
            'Año': future_years,
            'Matricula': projected_enrollment,
            'Carrera_Std': career,
            'Tipo': 'Proyección'
        })
        
        projection_df = pd.concat([projection_df, hist_part, proj_part], ignore_index=True)

    return projection_df


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
        # <<< MEJORA/NUEVO: AÑADIDA PESTAÑA DE PROYECCIONES >>>
        tab_list = ["📈 Panorama General", "🗺️ Riesgo vs. Crecimiento", "🎓 Perfil por Universidad", "🏛️ Cartera Universitaria", "🛠️ Planificador de Carrera", "📊 Análisis Estadístico", "🔮 Proyecciones a 2035", "📄 Datos Detallados"]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)
        
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
            # <<< MEJORA/NUEVO: TEXTO EXPLICATIVO MÁS CLARO >>>
            st.markdown("""
            **¿Qué estamos viendo aquí?**
            Este gráfico es el corazón estratégico de la plataforma. Ubica cada carrera en un mapa basado en dos preguntas clave:
            1.  **¿Qué tan vulnerable es a la IA?** (Eje horizontal, de izquierda a derecha aumenta el riesgo).
            2.  **¿Está creciendo o decreciendo el interés de los estudiantes en ella?** (Eje vertical, hacia arriba crece más rápido).

            **¿Cómo medimos el "crecimiento"?**
            Usamos la **Tasa de Crecimiento Anual Compuesta (TCAC)**. No te asustes por el nombre. Piénsalo como el "interés compuesto" del interés estudiantil. Mide la tasa de crecimiento promedio y constante de la matrícula a lo largo de los años que seleccionaste. Una TCAC positiva significa que la carrera gana popularidad; una negativa, que la pierde. La fórmula es:
            $$ TCAC = \\left( \\frac{\\text{Matrícula Final}}{\\text{Matrícula Inicial}} \\right)^{\\frac{1}{\\text{N° de Años}}} - 1 $$
            Graficamos este "interés" contra el riesgo de IA para obtener una radiografía del futuro laboral.
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
                    # <<< MEJORA/NUEVO: TEXTO EXPLICATIVO MÁS CLARO >>>
                    st.markdown("""
                    **Análisis de Cuadrantes (Tu GPS Estratégico):**
                    Las líneas de promedio dividen el mapa en cuatro zonas:

                    -   **🟢 Zona de Fortaleza (Arriba a la Izquierda):** Bajo riesgo y matrícula en alza. Son las carreras mejor posicionadas para el futuro inmediato. ¡Una apuesta segura!
                    -   **🟡 Zona de Adaptación Urgente (Arriba a la Derecha):** Alto riesgo, pero la gente las sigue estudiando. ¡Son populares pero vulnerables! La demanda existe, pero es crucial que las universidades modernicen sus planes de estudio para enseñar habilidades que la IA no pueda replicar (creatividad, liderazgo, pensamiento crítico).
                    -   **🔵 Zona de Nicho o Reinvención (Abajo a la Izquierda):** Bajo riesgo, pero la matrícula decrece. Pueden ser carreras muy especializadas o que necesitan un "empujón" de marketing para comunicar mejor su valor en el mercado actual.
                    -   **🔴 Zona de Alerta Crítica (Abajo a la Derecha):** Alto riesgo y matrícula en descenso. ¡Doble señal de alerta! Estas carreras necesitan una reinvención radical o corren el riesgo de volverse obsoletas.
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
                                     orientation='h', color='Riesgo_Cartera', color_continuous_scale=px.colors.sequential.Reds,
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
            # <<< MEJORA/NUEVO: TEXTO EXPLICATIVO MÁS CLARO >>>
            st.header("📊 Explorando los Datos a Fondo")
            st.markdown("""
            **¿Qué estamos viendo aquí?**
            Esta sección es para los curiosos que quieren "ver debajo del capó". Usamos herramientas estadísticas sencillas para descubrir patrones ocultos en los datos.
            """)
            st.subheader("1. Histogramas: ¿Qué es lo más común?")
            st.markdown("""
            Un histograma es como una encuesta. Agrupa los datos en "cajones" para ver qué tan frecuentemente aparece cada valor.
            -   **Histograma de Riesgo:** Nos muestra si la mayoría de las carreras en Puebla tienen un riesgo bajo, medio o alto. ¿Hay muchas carreras en la zona de peligro?
            -   **Histograma de Matrícula:** Nos dice si predominan las carreras con pocos estudiantes (de nicho) o las carreras masivas.
            """)
            col1, col2 = st.columns(2)
            with col1:
                df_unique_risk = df_filtered.drop_duplicates(subset=['Carrera_Std'])
                fig_hist_risk = px.histogram(df_unique_risk, x='Riesgo_IA', nbins=20, title="Frecuencia de Niveles de Riesgo")
                st.plotly_chart(fig_hist_risk, use_container_width=True)
            with col2:
                fig_hist_mat = px.histogram(df_filtered[df_filtered['Año'] == df_filtered['Año'].max()], x='Matricula', nbins=40, title="Frecuencia del Tamaño de Matrículas (Últ. Año)")
                st.plotly_chart(fig_hist_mat, use_container_width=True)

            st.markdown("---")
            st.subheader("2. Diagramas de Caja: Comparando Manzanas con Manzanas")
            st.markdown("""
            Un diagrama de caja (box plot) es una de las mejores herramientas para comparar grupos.
            -   **¿Cómo leerlo?** La **caja** de color representa al 50% "promedio" de las carreras de esa área. La **línea dentro de la caja** es la mediana (la carrera que está justo a la mitad). Los **"bigotes"** (las líneas que salen de la caja) muestran el rango de casi todas las demás. Los **puntos sueltos** son los "casos extremos" o valores atípicos.
            -   **¿Para qué sirve?** Podemos responder preguntas como: ¿El área de 'Ciencias de la Salud' tiene, en general, un riesgo más bajo que 'Ingeniería y Tecnología'? ¿O hay más variación en sus matrículas?
            """)
            col3, col4 = st.columns(2)
            with col3:
                df_unique_careers = df_filtered.drop_duplicates(subset=['Carrera_Std'])
                fig_box_risk = px.box(df_unique_careers, x='Area', y='Riesgo_IA', color='Area', title="Distribución del Riesgo IA por Área")
                st.plotly_chart(fig_box_risk, use_container_width=True)
            with col4:
                df_latest_year_box = df_filtered[df_filtered['Año'] == df_filtered['Año'].max()]
                fig_box_mat = px.box(df_latest_year_box, x='Area', y='Matricula', color='Area', points="outliers", title="Distribución de la Matrícula por Área")
                st.plotly_chart(fig_box_mat, use_container_width=True)

            st.markdown("---")
            st.subheader("3. Análisis de Correlación: ¿Una cosa afecta a la otra?")
            # <<< MEJORA/NUEVO: TEXTO EXPLICATIVO MÁS CLARO >>>
            st.markdown("""
            Aquí investigamos si existe una relación entre el riesgo de una carrera y el crecimiento de su matrícula.
            -   **¿Cómo funciona?** Trazamos una **"línea de mejor ajuste"** (la línea roja de regresión) a través de los puntos de datos para ver la tendencia general. ¿A medida que el riesgo sube, el crecimiento tiende a bajar?
            -   **Interpretación:** El valor **R-cuadrado ($R^2$)** nos dice qué tan bien la línea roja explica lo que está pasando. Un $R^2$ de 0.8 significaría que el 80% del crecimiento/decrecimiento de la matrícula puede explicarse por el nivel de riesgo de la carrera. Un $R^2$ bajo (cercano a 0) significa que no hay una relación clara, y que el crecimiento depende de otros factores que no son el riesgo.
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

        # <<< MEJORA/NUEVO: PESTAÑA COMPLETA DE PROYECCIONES >>>
        with tab7:
            st.header(f"🔮 Proyecciones de Matrícula al 2035")
            st.markdown("""
            **Descripción Académica:** Esta sección utiliza un modelo de **regresión lineal** para extrapolar las tendencias históricas de la matrícula hasta el año 2035. En términos simples, dibuja una "línea de tendencia" basada en los datos del pasado y la extiende hacia el futuro.

            **¡Advertencia Importante!**
            -   **Esto NO es una predicción garantizada.** Es una herramienta de visualización de tendencias.
            -   El modelo asume que las condiciones y el comportamiento de los estudiantes **seguirán siendo los mismos** que en el pasado.
            -   Crisis económicas, nuevas tecnologías, pandemias o cambios en políticas públicas pueden alterar drásticamente estas tendencias.

            **¿Cómo usarlo?** Selecciona una o varias carreras para comparar sus trayectorias futuras si todo sigue "como hasta ahora". Es útil para identificar qué carreras muestran un impulso sostenido y cuáles podrían enfrentar desafíos si no se toman acciones.
            """)
            
            if df_filtered['Año'].nunique() > 1:
                carreras_proyeccion = sorted(df_filtered['Carrera_Std'].unique())
                if carreras_proyeccion:
                    # Selecciona por defecto las 2 primeras carreras si hay más de una, sino la única que hay.
                    default_selection = carreras_proyeccion[:2] if len(carreras_proyeccion) > 1 else carreras_proyeccion
                    selected_carreras_proj = st.multiselect(
                        "Selecciona carreras para proyectar su tendencia:", 
                        options=carreras_proyeccion, 
                        default=default_selection
                    )

                    if selected_carreras_proj:
                        df_projections = project_enrollment(df_filtered, selected_carreras_proj, 2035)
                        
                        if not df_projections.empty:
                            fig_proj = go.Figure()

                            # Añadir trazos para cada carrera
                            for career in selected_carreras_proj:
                                career_color = px.colors.qualitative.Plotly[carreras_proyeccion.index(career) % len(px.colors.qualitative.Plotly)]
                                
                                # Datos históricos
                                hist_data = df_projections[(df_projections['Carrera_Std'] == career) & (df_projections['Tipo'] == 'Histórico')]
                                fig_proj.add_trace(go.Scatter(
                                    x=hist_data['Año'], 
                                    y=hist_data['Matricula'], 
                                    mode='lines+markers',
                                    name=f'{career} (Histórico)',
                                    line=dict(width=2.5, color=career_color),
                                    marker=dict(size=8)
                                ))

                                # Datos proyectados
                                proj_data = df_projections[(df_projections['Carrera_Std'] == career) & (df_projections['Tipo'] == 'Proyección')]
                                fig_proj.add_trace(go.Scatter(
                                    x=proj_data['Año'], 
                                    y=proj_data['Matricula'], 
                                    mode='lines',
                                    name=f'{career} (Proyección)',
                                    line=dict(dash='dash', color=career_color, width=2.5)
                                ))
                            
                            fig_proj.update_layout(
                                title="Proyección de Tendencia de Matrícula hasta 2035",
                                xaxis_title="Año",
                                yaxis_title="Número de Estudiantes (Matrícula)",
                                hovermode="x unified",
                                legend_title_text='Carrera'
                            )
                            st.plotly_chart(fig_proj, use_container_width=True)

                        else:
                            st.info("No se pudieron generar proyecciones para las carreras seleccionadas (se requieren al menos 2 años de datos).")
                    else:
                        st.info("Por favor, selecciona al menos una carrera para visualizar su proyección.")
                else:
                    st.warning("No hay carreras disponibles en la selección actual para proyectar. Ajusta los filtros.")
            else:
                st.info("Se necesita un rango de más de un año para poder realizar proyecciones.")


        # --- PESTAÑA 8: DATOS DETALLADOS (ANTES PESTAÑA 7) ---
        with tab8:
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