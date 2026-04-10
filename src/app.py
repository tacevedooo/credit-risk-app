import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from tensorflow.keras.models import load_model # Descomentar cuando integres el modelo

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="CrediPulse AI", page_icon="⚡", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

/* --- BASE --- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(180deg, #f3f6f4 0%, #eef3f1 100%);
}

/* --- BANNER --- */
.banner {
    background: linear-gradient(135deg, #4f8a8b 0%, #7aa6a1 100%);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

.banner h1 {
    font-size: 2.4rem;
    margin-bottom: 8px;
}

.banner p {
    opacity: 0.9;
}

/* --- TARJETAS --- */
.block-container {
    padding-top: 2rem;
}

div[data-testid="stForm"],
div[data-testid="stMetric"],
div[data-testid="stVerticalBlock"] > div:has(.stPlotlyChart),
div[data-testid="stTabs"] {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e3e8e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

/* --- BOTONES --- */
.stButton>button { 
    width: 100%; 
    border-radius: 12px; 
    background: linear-gradient(135deg, #4f8a8b, #6fa9a6);
    color: white; 
    height: 3.2em; 
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton>button:hover { 
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(79,138,139,0.25);
}

/* --- INPUTS --- */
input, .stSelectbox, .stNumberInput {
    border-radius: 10px !important;
    border: 1px solid #d6dfdc !important;
}

/* --- SLIDERS --- */
.stSlider > div > div > div > div {
    background-color: #4f8a8b !important;
}

/* --- MÉTRICAS --- */
div[data-testid="stMetric"] {
    text-align: center;
    background: linear-gradient(180deg, #ffffff, #f8fbfa);
}

div[data-testid="stMetric"] label {
    font-size: 0.9rem;
    color: #6b7c7c;
}

div[data-testid="stMetric"] div {
    font-size: 2rem;
    font-weight: 700;
    color: #2f3e46;
}

/* --- TABS --- */
button[role="tab"] {
    border-radius: 10px;
    padding: 10px 15px;
    margin-right: 5px;
    font-weight: 500;
    background-color: transparent;
    color: #4b5c5c;
}

button[aria-selected="true"] {
    background-color: #dfeceb !important;
    color: #2f3e46 !important;
}

/* --- SIDEBAR --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f6f9f8);
    border-right: 1px solid #e0e7e5;
}

/* --- TEXTO --- */
h1, h2, h3 {
    color: #2f3e46;
}

p {
    color: #5f6f73;
}

/* --- ALERTAS --- */
.stAlert {
    border-radius: 12px;
}

/* --- SCROLLBAR (detalle pro) --- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #c7d6d3;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)    

# --- CARGA DE RECURSOS (COMENTADO PARA INTEGRACIÓN FUTURA) ---
@st.cache_resource
def load_assets():
    # --- Descomenta esto cuando tengas los archivos ---
    # model = load_model('modelo_final.keras')
    # df_poblacion = pd.read_csv('metadata_poblacion.csv')
    
    # --- Datos temporales para pruebas (Simulación) ---
    model = None 
    df_pob_sim = pd.DataFrame({
        'score': np.random.normal(640, 90, 1000).clip(300, 850),
        'ingresos': np.random.lognormal(10.5, 0.4, 1000),
        'dti': np.random.uniform(5, 35, 1000)
    })
    return model, df_pob_sim

model, data_pob = load_assets()

# --- 0. BANNER SUPERIOR ---
# Aquí puedes agregar st.image("logo.png") cuando lo tengas
st.markdown("""
    <div class="banner">
        <h1>⚡ CrediPulse AI</h1>
        <p>Advanced Credit Risk Assessment | Neural Network Powered</p>
    </div>
    """, unsafe_allow_html=True)

# --- DEFINICIÓN DE SECCIONES ---
tab_eval, tab_descriptivo, tab_links = st.tabs([
    "🎯 Análisis de Perfil", 
    "📊 Diagnóstico de Mercado", 
    "🔗 Recursos & Proyecto"
])

# ==========================================
# SECCIÓN 1: ANÁLISIS DEL PERFIL
# ==========================================
with tab_eval:
    st.subheader("Simulador de Scorecard Individual")
    st.write("Ingrese los datos para calcular la probabilidad de incumplimiento.")
    
    with st.form("score_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 💰 Datos Financieros")
            loan_amnt = st.number_input("Monto del Crédito ($)", min_value=500, value=10000)
            annual_inc = st.number_input("Ingreso Anual ($)", min_value=1000, value=45000)
            int_rate = st.slider("Tasa de Interés (%)", 5.0, 30.0, 12.0)
            dti = st.slider("Relación Deuda/Ingreso (DTI %)", 0.0, 100.0, 15.0)
            revol_util = st.slider("Uso de tarjetas de crédito (%)", 0.0, 120.0, 30.0)
            
        with c2:
            st.markdown("##### 📊 Historial y Contrato")
            term = st.selectbox("Plazo del contrato", ["36 meses", "60 meses"])
            home_ownership = st.selectbox("Situación de Vivienda", ["MORTGAGE", "RENT", "OWN", "OTHER"])
            purpose = st.selectbox("Propósito del Crédito", 
                                   ["Debt Consolidation", "Credit Card", "Small Business", 
                                    "Home Improvement", "Medical", "Other"])
            
            # Variables de historial (Buró)
            delinq_2yrs = st.number_input("Moras en los últimos 2 años", 0, 20, 0)
            pub_rec = st.number_input("Registros públicos negativos", 0, 10, 0)
            open_acc = st.number_input("Cuentas de crédito abiertas", 0, 50, 10)
            total_acc = st.number_input("Total de cuentas históricas", 0, 100, 20)
        
        submit = st.form_submit_button("GENERAR DIAGNÓSTICO")

    if submit:
        # --- PROCESAMIENTO DE VARIABLES (One-Hot Encoding) ---
        # 1. Variable Term
        term_60 = 1 if term == "60 meses" else 0
        
        # 2. Variables Home Ownership
        home_RENT = 1 if home_ownership == "RENT" else 0
        home_MORTGAGE = 1 if home_ownership == "MORTGAGE" else 0
        home_OWN = 1 if home_ownership == "OWN" else 0
        
        # 3. Variables Purpose
        p_business = 1 if purpose == "Small Business" else 0
        p_debt = 1 if purpose == "Debt Consolidation" else 0
        p_card = 1 if purpose == "Credit Card" else 0
        p_home = 1 if purpose == "Home Improvement" else 0
        p_med = 1 if purpose == "Medical" else 0
        p_other = 1 if purpose == "Other" else 0

        # --- CREACIÓN DEL VECTOR PARA EL MODELO ---
        # El orden debe ser EXACTAMENTE el mismo con el que entrenaste
        input_vector = [
            int_rate, dti, annual_inc, loan_amnt, revol_util,
            delinq_2yrs, pub_rec, open_acc, total_acc,
            term_60, home_RENT, home_MORTGAGE, home_OWN,
            p_business, p_debt, p_card, p_home, p_med, p_other
        ]
        
        # --- PREDICCIÓN (Simulada hasta que cargues el modelo) ---
        # prob = model.predict(np.array([input_vector]))[0][0]
        prob_simulada = np.random.uniform(0.02, 0.35) 
        score_usuario = int(850 - (prob_simulada * 550))
        
        # --- MOSTRAR RESULTADOS (Tu diseño original) ---
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("### Resultado del Análisis")
            st.metric(label="Score Final", value=f"{score_usuario} / 850")
            
            if score_usuario >= 680:
                st.success("🏆 PERFIL: EXCELENTE")
            elif score_usuario >= 550:
                st.warning("⚖️ PERFIL: RIESGO MEDIO")
            else:
                st.error("⚠️ PERFIL: RIESGO ALTO")
            
            st.write(f"**Probabilidad de Incumplimiento:** {prob_simulada:.2%}")
        
        with res_col2:
            fig_pob = px.histogram(data_pob, x="score", nbins=50, 
                                   title="Tu posición respecto a la población histórica",
                                   color_discrete_sequence=['#4f8a8b'], opacity=0.6)
            fig_pob.add_vline(x=score_usuario, line_width=5, line_color="#ef4444", 
                              annotation_text="TU SCORE", annotation_position="top left")
            fig_pob.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_pob, use_container_width=True)

# ==========================================
# SECCIÓN 2: ANÁLISIS DESCRIPTIVO (POBLACIÓN)
# ==========================================
with tab_descriptivo:
    st.header("Exploración de Datos y Metodología")
    st.write("Análisis descriptivo de la base de datos utilizada para el entrenamiento del modelo (Dataset LendingClub).")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.subheader("Distribución de Riesgo")
        # Gráfica interactiva de ingresos vs dti
        fig_scatter = px.scatter(data_pob.sample(500), x="ingresos", y="dti", color="score",
                                 title="Relación Ingreso vs Endeudamiento",
                                 color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_stat2:
        st.subheader("Arquitectura de la Red Neuronal")
        st.markdown("""
        El modelo **CrediPulse AI** utiliza una arquitectura de Aprendizaje Profundo (Deep Learning):
        - **Input Layer:** Variables prospectivas (DTI, Ingresos, FICO).
        - **Hidden Layers:** 3 capas densas con regularización Dropout para evitar sobreajuste.
        - **Output Layer:** Activación Sigmoide para estimación de probabilidad binaria.
        - **Optimizador:** Adam (tasa de aprendizaje adaptativa).
        """)
        st.info("💡 El modelo fue optimizado mediante búsqueda de hiperparámetros para maximizar el AUC-ROC.")

# ==========================================
# SECCIÓN 3: LINKS Y REFERENCIAS
# ==========================================
with tab_links:
    st.header("Entregables y Documentación")
    
    c_link1, c_link2, c_link3 = st.columns(3)
    
    with c_link1:
        st.markdown("#### 📑 Reporte Técnico")
        st.write("Consulta la metodología detallada y el análisis estadístico en nuestro blog.")
        st.button("Leer Blog Post", key="btn_blog") # Aquí pondrás el st.link_button más adelante
        
    with c_link2:
        st.markdown("#### 🎬 Video Promocional")
        st.write("Presentación comercial de la aplicación y caso de uso.")
        st.button("Ver en YouTube", key="btn_video")
        
    with c_link3:
        st.markdown("#### 👥 Equipo")
        st.write("**Integrantes:**")
        st.write("- Juan David Ospina Arango")
        st.write("- [Tu Nombre]")
        st.write("- [Nombre de Compañero]")
        st.caption("Curso: Aplicaciones de Redes Neuronales 2026")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Configuración de la App")
st.sidebar.checkbox("Mostrar datos crudos de población", value=False)
st.sidebar.write("Versión 1.0.0-beta")