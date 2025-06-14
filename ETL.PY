import pandas as pd
import os

# Lista para almacenar los DataFrames limpios de cada año
lista_de_dfs = []

# Especifica la carpeta donde descargaste los archivos de ANUIES
ruta_archivos_anuies = './datos_anuies/' 
archivos = os.listdir(ruta_archivos_anuies)

for archivo in archivos:
    if archivo.endswith('.xlsx'):
        # Extrae el año del nombre del archivo (ajusta según sea necesario)
        # Asumimos que el año está en el nombre, ej: ANUARIO_2022_...
        try:
            año = int(archivo.split('_')[1])
        except (IndexError, ValueError):
            print(f"No se pudo extraer el año de '{archivo}'. Saltando archivo.")
            continue

        print(f"Procesando archivo del año: {año}...")

        # 1. Cargar el archivo Excel especificando la hoja y el encabezado
        # DEBES ajustar 'sheet_name' y 'header' según tu inspección inicial
        try:
            df_raw = pd.read_excel(
                os.path.join(ruta_archivos_anuies, archivo),
                sheet_name='2.1.2',  # AJUSTA ESTE NOMBRE
                header=5           # AJUSTA ESTE NÚMERO DE FILA
            )
        except Exception as e:
            print(f"No se pudo leer el archivo {archivo}. Error: {e}")
            continue

        # 2. Filtrar para obtener solo datos de Puebla
        # La columna puede llamarse 'ENTIDAD' o 'NOMBRE DE LA ENTIDAD'
        if 'ENTIDAD' in df_raw.columns:
            df_puebla = df_raw[df_raw['ENTIDAD'] == 'Puebla'].copy()
        else:
            # Puedes necesitar inspeccionar df_raw.columns para encontrar el nombre correcto
            print(f"No se encontró la columna 'ENTIDAD' en {archivo}. Intentando con otros nombres...")
            continue 

        # 3. Seleccionar y renombrar las columnas de interés
        # Estos nombres de columna son ejemplos, AJÚSTALOS a los del archivo real
        columnas_de_interes = {
            'NOMBRE DE LA INSTITUCIÓN DE EDUCACIÓN SUPERIOR': 'Universidad',
            'PROGRAMA EDUCATIVO': 'Carrera',
            'Total': 'Matricula' # Generalmente la matrícula total está en una columna llamada 'Total'
        }
        
        # Verificar que las columnas existan antes de seleccionarlas
        columnas_existentes = [col for col in columnas_de_interes.keys() if col in df_puebla.columns]
        if len(columnas_existentes) != len(columnas_de_interes):
            print(f"Faltan columnas en {archivo}. Columnas esperadas: {list(columnas_de_interes.keys())}")
            continue

        df_puebla = df_puebla[columnas_existentes]
        df_puebla = df_puebla.rename(columns=columnas_de_interes)

        # 4. Limpieza profunda
        df_puebla.dropna(subset=['Universidad', 'Carrera', 'Matricula'], inplace=True)
        
        # Eliminar filas de subtotales que la ANUIES incluye
        df_puebla = df_puebla[~df_puebla['Carrera'].str.contains("Total", na=False)]
        
        # Convertir matrícula a número
        df_puebla['Matricula'] = pd.to_numeric(df_puebla['Matricula'], errors='coerce').fillna(0).astype(int)

        # Eliminar filas con matrícula cero
        df_puebla = df_puebla[df_puebla['Matricula'] > 0]
        
        # 5. Añadir la columna del año
        df_puebla['Año'] = año
        
        # 6. Añadir el DataFrame limpio a nuestra lista
        lista_de_dfs.append(df_puebla)

# --- B. Consolidación Final ---
print("Consolidando todos los años...")
df_final = pd.concat(lista_de_dfs, ignore_index=True)

# Reordenar las columnas para que coincida con tu formato
df_final = df_final[['Año', 'Universidad', 'Carrera', 'Matricula']]