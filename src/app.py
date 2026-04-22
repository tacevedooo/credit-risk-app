import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import keras

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="CreditScore", page_icon="🪙", layout="wide")

# --- ESTILOS PERSONALIZADOS (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4facfe; color: white; border: none; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .banner { 
        padding: 30px; 
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
        color: white; 
        border-radius: 15px; 
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* --- PESTAÑAS MINIMALISTAS --- */
    
    /* 1. Ocultar la línea roja animada por defecto de Streamlit */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    /* 2. Dejar la pestaña sin fondos y preparar el borde invisible */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-bottom: 3px solid transparent !important;
    }
    
    /* 3. Mostrar ÚNICAMENTE la línea azul en la pestaña seleccionada */
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #4facfe !important;
    }
    </style>
    """, unsafe_allow_html=True)

import os

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Obtenemos la ruta de la carpeta 'src' donde vive app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel y entramos a 'assets'
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets')

@st.cache_resource
def load_assets():
    # Construimos las rutas completas para cada archivo
    model_path = os.path.join(ASSETS_DIR, 'modelo_riesgo_crediticio_optimizado.keras')
    scaler_path = os.path.join(ASSETS_DIR, 'scaler.pkl')
    vars_path = os.path.join(ASSETS_DIR, 'variables_modelo.json')
    rangos_path = os.path.join(ASSETS_DIR, 'rangos_validacion.json')
    dist_path = os.path.join(ASSETS_DIR, 'datos_distribucion.json')
    
    # Carga de archivos
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(vars_path, 'r') as f:
        vars_modelo = json.load(f)
    with open(rangos_path, 'r') as f:
        rangos = json.load(f)
    with open(dist_path, 'r') as f:
        dist_data = json.load(f)
        
    return model, scaler, vars_modelo, rangos, dist_data

# --- INTENTAR CARGAR ---
try:
    model, scaler, vars_modelo, rangos, dist_data = load_assets()
except Exception as e:
    st.error("⚠️ Error al cargar archivos de modelo.")
    st.write(f"**Detalle del error:** {e}")
    
    # Debug extra para que veas qué está pasando en la nube
    st.info(f"Buscando en: {os.path.abspath(ASSETS_DIR)}")
    if os.path.exists(ASSETS_DIR):
        st.write("Contenido de assets:", os.listdir(ASSETS_DIR))
    else:
        st.warning("La carpeta 'assets' no fue encontrada en la ruta especificada.")
        
    st.stop()

# --- FUNCIONES AUXILIARES ---
def probabilidad_a_score(probabilidad, pdo=20, score_base=600, odds_base=1):
    factor = pdo / np.log(2)
    offset = score_base - factor * np.log(odds_base)
    probabilidad = np.clip(probabilidad, 1e-6, 1 - 1e-6)
    odds = (1 - probabilidad) / probabilidad
    score = offset + factor * np.log(odds)
    return np.clip(score, 300, 850)

# --- BANNER PRINCIPAL ---
st.markdown('<div class="banner"><h1>🪙 CreditScore App</h1><p>Análisis de riesgo crediticio basado en Deep Learning</p></div>', unsafe_allow_html=True)

# --- NAVEGACIÓN HORIZONTAL (PESTAÑAS NATIVAS) ---
tab_inicio, tab_calc, tab_info = st.tabs(["🏠 Inicio", "🧮 Calculadora de Riesgo", "ℹ️ Información Adicional"])


# --- SECCIÓN 1: INICIO ---
with tab_inicio:
    col1, col2 = st.columns([1.2, 1])  # Ajustamos proporciones para dar espacio al video
    
    with col1:
        st.subheader("¿Qué es esta App?")
        st.write("""
        Esta herramienta utiliza un modelo de **Redes Neuronales** para predecir la probabilidad de que un cliente 
        incumpla con sus obligaciones financieras. A diferencia de los métodos tradicionales, analizamos 
        patrones complejos en 20 variables financieras.
        """)
        
        st.subheader("¿Cómo funciona?")
        st.write("""
        1. **Ingreso de Datos:** Proporcionas información sobre ingresos, deudas y propósito del crédito.
        2. **Procesamiento:** El sistema procesa tus datos y los pasa por el modelo entrenado.
        3. **Resultado:** Obtienes un puntaje crediticio, la probabilidad de incumplimiento y una comparativa frente a la población.
        """)

    with col2:
        st.subheader("🎬 Video de Presentación")
        # Reemplaza el link por tu video real de YouTube
        video_url = "https://www.youtube.com/watch?v=iHfpLri8KlQ" 
        st.video(video_url)
        
        st.caption("Mira este breve video para entender el impacto y la tecnología detrás de CreditScore App.")

        
# --- SECCIÓN 2: CALCULADORA ---
with tab_calc:
    st.subheader("Ingrese los datos del solicitante")
    
    # Formulario de entrada
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        
        # 1. Inputs numéricos principales
        loan_amnt = c1.number_input("Monto del Préstamo ($)", min_value=0, max_value=100000, value=10000)
        annual_inc = c2.number_input("Ingreso Anual ($)", min_value=0, max_value=10000000, value=60000)
        term_val = c3.selectbox("Plazo del crédito", ["36 meses", "60 meses"])
        
        int_rate = c1.slider("Tasa de interés (%)", min_value=0.0, max_value=100.0, value=10.0)
        dti = c2.slider("DTI (Relación Deuda/Ingreso) (%)", 0.0, 100.0, 15.0)
        revol_util = c3.slider("Uso de tarjetas de crédito (%)", 0.0, 100.0, 30.0)
        
        # 2. Variables de comportamiento crediticio (nuevas)
        inq_last_6mths = c1.number_input("Solicitudes de crédito en los últimos 6 meses", 0, 20, 0)
        
        # 3. Diccionarios de traducción para categorías
        nombres_vivienda = {
            "MORTGAGE": "Hipotecada", "RENT": "Alquilada", "OWN": "Propia", "OTHER": "Otro"
        }
        
        nombres_proposito = {
            "debt_consolidation": "Consolidación de deudas", "credit_card": "Tarjeta de crédito", 
            "home_improvement": "Mejoras del hogar", "small_business": "Pequeño negocio", 
            "medical": "Gastos médicos", "major_purchase": "Compra mayor",
            "vacation": "Vacaciones", "wedding": "Boda", "house": "Casa",
            "educational": "Educativo", "moving": "Mudanza", 
            "renewable_energy": "Energía renovable", "other": "Otro"
        }

        home = c2.selectbox(
            "Tipo de Vivienda", 
            options=list(nombres_vivienda.keys()), 
            format_func=lambda x: nombres_vivienda[x]
        )
        
        purpose = c3.selectbox(
            "Propósito del Crédito", 
            options=list(nombres_proposito.keys()), 
            format_func=lambda x: nombres_proposito[x]
        )

        submitted = st.form_submit_button("Calcular Score", type="primary")

    if submitted:
        # --- PROCESAMIENTO DE INPUTS ---
        
        # Inicializamos todas las variables en 0 para las dummies
        inputs = {col: 0 for col in vars_modelo}
        
        # Asignamos valores numéricos directos
        inputs['loan_amnt'] = loan_amnt
        inputs['int_rate'] = int_rate
        inputs['annual_inc'] = annual_inc
        inputs['dti'] = dti
        inputs['inq_last_6mths'] = inq_last_6mths
        inputs['revol_util'] = revol_util
        
        # El modelo espera 'term' como numérico (36 o 60)
        inputs['term'] = 36 if term_val == "36 meses" else 60
        
        # Asignamos las variables Dummie (One-Hot Encoding)
        # Vivienda
        key_home = f"home_ownership_{home}"
        if key_home in inputs:
            inputs[key_home] = 1
            
        # Propósito
        key_purpose = f"purpose_{purpose}"
        if key_purpose in inputs:
            inputs[key_purpose] = 1
        
        # Convertir a DataFrame asegurando el orden exacto de 'vars_modelo'
        data_df = pd.DataFrame([inputs])[vars_modelo]
        
        # Escalado y Predicción
        data_scaled = scaler.transform(data_df)
        prob = model.predict(data_scaled)[0][0]
        score = probabilidad_a_score(prob)
        
        # --- MOSTRAR RESULTADOS ---
        st.divider()
        
        tarjeta_html = """
        <div style="background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white; 
                    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 20px;">
            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">{titulo}</p>
            <h2 style="margin: 0; font-size: 2.5rem; color: white;">{valor}</h2>
        </div>
        """
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(tarjeta_html.format(titulo="Puntaje de Crédito", valor=f"{int(score)} pts"), unsafe_allow_html=True)
        with m2:
            st.markdown(tarjeta_html.format(titulo="Probabilidad de Incumplimiento", valor=f"{prob*100:.2f}%"), unsafe_allow_html=True)
        
        # GRÁFICA DE COMPARACIÓN
        st.subheader("Tu posición frente a la población")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        bins = dist_data['bins'][:-1]
        ax.bar(bins, dist_data['counts_buenos'], width=np.diff(dist_data['bins']), alpha=0.5, color='#2ecc71', label='Buenos Pagadores')
        ax.bar(bins, dist_data['counts_malos'], width=np.diff(dist_data['bins']), alpha=0.5, color='#e74c3c', label='Malos Pagadores')
        
        ax.axvline(score, color='blue', linestyle='--', linewidth=2, label=f'Tu Score: {int(score)}')
        
        ax.set_xlabel('Puntaje de Crédito')
        ax.set_ylabel('Cantidad de Usuarios')
        ax.set_xlim(520, 680) # Ajustado al rango estándar
        
        ax.legend()
        st.pyplot(fig)

# --- SECCIÓN 3: INFO ADICIONAL ---
with tab_info:
    st.subheader("📂 Documentación y Recursos del Proyecto")
    
    
    # Creamos 3 columnas para que los botones queden alineados horizontalmente
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        st.link_button(
            "📄 Reporte Técnico", 
            "https://jihernandezc.github.io/rnaab_riesgo_crediticio/index.html", 
            use_container_width=True,
            help="Accede al reporte detallado de análisis, metodología y resultados del proyecto."
        )
    
    with col_btn2:
        st.link_button(
            "🎬 Video Publicitario", 
            "https://youtu.be/iHfpLri8KlQ", 
            use_container_width=True,
            help="Mira el pitch del proyecto y la demostración de valor."
        )
    
    with col_btn3:
        st.link_button(
            "💻 Repositorio GitHub", 
            "https://github.com/tacevedooo/credit-risk-app", 
            use_container_width=True,
            help="Explora el código fuente y el dataset utilizado."
        )

    st.divider()
    
    # Un extra para que no se vea vacío
    st.markdown("### 🛠️ Tecnologías Utilizadas")
    st.info("""
    - **Backend:** Python con TensorFlow/Keras para la Red Neuronal.
    - **Procesamiento:** Scikit-Learn para escalamiento y transformación de datos.
    - **Interfaz:** Streamlit (UI Nativa).
    - **Visualización:** Matplotlib y Seaborn para el análisis de distribución.
    """)   