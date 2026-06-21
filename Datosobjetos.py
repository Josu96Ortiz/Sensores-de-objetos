import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import datetime

# Configuración de la interfaz de la página
st.set_page_config(
    page_title="IoT Object Sensor Monitor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Encabezado principal
st.title("🏭 Sistema IoT de Sensores de Objetos - Adquisición y Perfilado")
st.markdown("""
Esta aplicación permite adquirir, visualizar y describir conjuntos de datos provenientes de **Sensores IoT de Objetos**.
Puedes elegir entre activar el **Sensor Óptico (Cámara Web)** para escanear un objeto real que tengas a la mano, o generar datos masivos de una **Banda Transportadora Industrial** automatizada.
""")

# --- 1. SELECCIÓN Y ADQUISICIÓN DE LA FUENTE DE DATOS ---
st.sidebar.header("🔌 1. Adquisición del Hardware / Sensor")
opcion_sensor = st.sidebar.selectbox(
    "Selecciona la fuente del sensor IoT:",
    ["Cámara Web (Objeto Real a la Mano)", "Banda Industrial (Simulador de Proximidad y Peso)"]
)

df = None

if opcion_sensor == "Cámara Web (Objeto Real a la Mano)":
    st.sidebar.info("Coloca un objeto frente a la cámara (ej. un teléfono, una taza, una pluma) y captura la foto para extraer sus variables físicas.")
    foto_sensor = st.sidebar.camera_input("📸 Escanear objeto físico")
    
    if foto_sensor is not None:
        try:
            img_pil = Image.open(foto_sensor)
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Segmentación por umbral dinámico para detectar las dimensiones del objeto
            _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lista_areas, lista_perimetros, lista_rojo, lista_verde, lista_azul = [], [], [], [], []
            
            for c in contornos:
                area = cv2.contourArea(c)
                if area > 150:  # Filtrar ruido de fondo
                    perimetro = cv2.arcLength(c, True)
                    mascara = np.zeros(gris.shape, dtype=np.uint8)
                    cv2.drawContours(mascara, [c], -1, 255, -1)
                    media_colores = cv2.mean(img_bgr, mask=mascara)
                    
                    lista_areas.append(area)
                    lista_perimetros.append(perimetro)
                    lista_azul.append(round(media_colores[0], 2))
                    lista_verde.append(round(media_colores[1], 2))
                    lista_rojo.append(round(media_colores[2], 2))
            
            if len(lista_areas) > 0:
                df = pd.DataFrame({
                    "ID_Componente": [f"Segmento_{i+1}" for i in range(len(lista_areas))],
                    "Sensor_Area_Px": lista_areas,
                    "Sensor_Perimetro_Px": lista_perimetros,
                    "Canal_Rojo_R": lista_rojo,
                    "Canal_Verde_G": lista_verde,
                    "Canal_Azul_B": lista_azul
                })
                st.success("✔️ ¡Datos geométricos del objeto real adquiridos con éxito!")
            else:
                st.warning("Prueba colocando el objeto más cerca o con un fondo de diferente color.")
        except Exception as e:
            st.error(f"Error en el procesamiento del sensor óptico: {e}")

else:
    # Opción de Banda Industrial (Genera un flujo continuo y ordenado de objetos automáticamente)
    st.sidebar.write("Configuración del flujo industrial:")
    num_muestras = st.sidebar.slider("Cantidad de objetos a pasar por la banda:", 50, 500, 200, step=50)
    estado_lote = st.sidebar.selectbox("Estado del lote de producción:", ["Óptimo (Calidad Estable)", "Irregular (Fallas de Proximidad)"])
    
    if st.sidebar.button("🤖 Iniciar Telemetría de la Banda"):
        np.random.seed(42)
        tiempos = pd.date_range(start=datetime.datetime.now(), periods=num_muestras, freq='2S')
        
        if estado_lote == "Óptimo (Calidad Estable)":
            distancia_cm = np.random.normal(12.0, 0.4, num_muestras).round(2)
            volumen_cm3 = np.random.normal(180.0, 3.0, num_muestras).round(1)
            peso_g = (volumen_cm3 * 1.15 + np.random.normal(0, 1, num_muestras)).round(1)
        else:
            distancia_cm = np.random.uniform(5.0, 30.0, num_muestras).round(2)
            volumen_cm3 = np.random.normal(150.0, 20.0, num_muestras).round(1)
            peso_g = (volumen_cm3 * 1.15 + np.random.normal(0, 6, num_muestras)).round(1)
            
        df = pd.DataFrame({
            "Timestamp": tiempos,
            "ID_Objeto_Banda": [f"OBJ-{i:04d}" for i in range(1, num_muestras + 1)],
            "Sensor_Distancia_CM": distancia_cm,
            "Sensor_Volumen_CM3": volumen_cm3,
            "Sensor_Peso_G": peso_g
        })
        st.session_state['datos_banda'] = df
        st.success(f"✔️ ¡Telemetría completada! {num_muestras} registros adquiridos.")

# Recordar datos de la banda si existen en caché para no perder la visualización
if opcion_sensor == "Banda Industrial (Simulador de Proximidad y Peso)" and 'datos_banda' in st.session_state:
    df = st.session_state['datos_banda']


# --- DESARROLLO DE LAS ACTIVIDADES DE VISUALIZACIÓN Y ANÁLISIS ---
if df is not None:
    
    # 2. MOSTRAR EL CONJUNTO DE DATOS EN UNA TABLA INTERACTIVA
    st.header("📋 2. Tabla Interactiva del Conjunto de Datos IoT")
    st.write("Muestra y estructura de las variables adquiridas directamente desde los transductores:")
    st.dataframe(df, use_container_width=True)
    
    # 3. PRESENTAR INFORMACIÓN DESCRIPTIVA DEL CONJUNTO DE DATOS
    st.header("📉 3. Información Descriptiva Básica")
    
    # Tarjetas de resumen ejecutivo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Número de Registros (Filas)", df.shape[0])
    with col2:
        st.metric("Número de Variables (Columnas)", df.shape[1])
    with col3:
        st.metric("Valores Nulos Totales", df.isnull().sum().sum())
    with col4:
        # Calcular una métrica representativa según el tipo de sensor seleccionado
        if "Sensor_Peso_G" in df.columns:
            st.metric("Peso Promedio de Objetos", f"{df['Sensor_Peso_G'].mean().round(2)} g")
        else:
            st.metric("Área Máxima Detectada", f"{int(df['Sensor_Area_Px'].max())} px")
        
    # Tabla requerida: Tipos de datos y nulos
    st.subheader("Tipos de Datos y Valores Nulos por Variable")
    info_df = pd.DataFrame({
        "Tipo de Dato": df.dtypes.astype(str),
        "Valores Nulos": df.isnull().sum(),
        "% de Nulos": (df.isnull().sum() / len(df) * 100).round(2).astype(str) + " %"
    })
    st.table(info_df)
    
    # Estadísticas descriptivas estándar
    st.subheader("Estadísticas Descriptivas Generales")
    st.dataframe(df.describe().T, use_container_width=True)
    
    # 4. REPORTE DE PERFILADO UTILIZANDO YDATA PROFILING
    st.header("🧬 4. Reporte de Perfilado Automatizado")
    st.write("Presiona el botón inferior para procesar las matrices de correlación y distribuciones de frecuencia del sensor.")
    
    if st.button("Generar Reporte de Perfilado Completo"):
        with st.spinner("Compilando el reporte automatizado..."):
            # Generar el reporte optimizado
            pr = ProfileReport(df, explorative=True, minimal=True)
            st_profile_report(pr)
            st.success("¡Reporte de perfilado generado con éxito!")

else:
    st.warning("⚠️ Esperando captura del sensor. Por favor usa la cámara o inicia el botón de la banda en el menú lateral.")