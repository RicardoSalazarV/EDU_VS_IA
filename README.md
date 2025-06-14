Análisis de Matrícula Educativa vs. Riesgo de IA en Puebla
Este proyecto presenta una aplicación web interactiva desarrollada con Streamlit que visualiza y analiza datos de matrícula educativa en el estado de Puebla, México, y los contrasta con el riesgo laboral asociado a la inteligencia artificial en diversas áreas de conocimiento.

La aplicación permite a los usuarios explorar la distribución de la matrícula por nivel educativo, área y sexo, y comparar estos datos con las tareas que tienen un mayor riesgo de ser automatizadas por la IA.

🚀 Demo
La aplicación permite filtrar los datos de matrícula por:

Nivel Educativo: Licenciatura o Posgrado.
Periodo: Diferentes ciclos escolares.
Área de Conocimiento: Por ejemplo, "Ciencias de la salud", "Tecnologías de la Información y la Comunicación", "Administración y negocios".
Además, presenta visualizaciones como:

Gráficos de barras que muestran la matrícula total por área y sexo.
Una tabla que detalla el riesgo de exposición a la IA para diferentes tareas, permitiendo comparar qué áreas de estudio preparan a los estudiantes para roles con mayor o menor riesgo de automatización.
📂 Estructura del Proyecto
El repositorio contiene los siguientes archivos principales:

app.py: El script principal de la aplicación web de Streamlit. Se encarga de la interfaz de usuario, los filtros y la visualización de datos.
ETL.PY: Un script de Python para el proceso de Extracción, Transformación y Carga (ETL). Este script prepara y limpia los datos originales para su uso en la aplicación.
matricula_puebla_tidy.csv: El conjunto de datos procesado que contiene la matrícula de educación superior en Puebla, desglosada por periodo, nivel, área y sexo.
tareas_riesgo_ia.csv: Un conjunto de datos que contiene una lista de tareas y su nivel de riesgo asociado a la inteligencia artificial.
requeeriments.txt: Un archivo de texto que lista las dependencias de Python necesarias para ejecutar el proyecto.
🛠️ Cómo ejecutar el proyecto localmente
Para ejecutar esta aplicación en tu máquina local, sigue estos pasos:

Clona el repositorio:

Bash

git clone https://github.com/RicardoSalazarV/EDU_VS_IA
Crea un entorno virtual (recomendado):

Bash

python -m venv venv
source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
Instala las dependencias:
Nota: El archivo de requerimientos tiene un error tipográfico (requeeriments.txt). Renómbralo a requirements.txt antes de ejecutar el siguiente comando.

Bash

pip install -r requirements.txt
Las dependencias principales son:

streamlit
pandas
plotly
Ejecuta la aplicación Streamlit:

Bash

streamlit run app.py
Abre tu navegador web y ve a http://localhost:8501.

📊 Fuentes de Datos
Matrícula Educativa: Los datos de matrícula fueron procesados a partir de fuentes de datos abiertos sobre educación en México. El script ETL.PY detalla el proceso de limpieza y transformación.
Riesgo de IA: Los datos sobre el riesgo de las tareas frente a la IA provienen de un análisis externo 
