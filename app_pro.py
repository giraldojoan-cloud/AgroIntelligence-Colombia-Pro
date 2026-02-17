import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de Marca
st.set_page_config(page_title="AgroIntelligence PRO", page_icon="🇨🇴", layout="wide")

# 2. Carga de Activos (Tu Bóveda de Inteligencia)
@st.cache_resource
def load_assets():
    modelo = joblib.load('modelo_colombia_70.pkl')
    le_cultivo = joblib.load('traductor_cultivos_turbo.pkl')
    # Nota: Los encoders de Dept/Muni los usaremos para validar la lógica
    return modelo, le_cultivo

modelo, le_cultivo = load_assets()

# 3. Interfaz de Usuario (El Front-End del Negocio)
st.title("🇨🇴 AgroIntelligence Colombia PRO")
st.subheader("Sistema de Decisión Agrícola basado en XGBoost & Agrosavia Big Data")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📍 Ubicación y Entorno")
    # Para hacerlo rápido, usaremos los códigos internos, pero un Pro después los mapea a nombres
    dept = st.number_input("ID Departamento (Consultar en DB)", 0, 32, 11)
    muni = st.number_input("ID Municipio (Consultar en DB)", 0, 1100, 1)
    ph = st.slider("Nivel de pH", 3.0, 9.0, 5.5)
    mo = st.slider("Materia Orgánica (%)", 0.0, 20.0, 4.0)

with col2:
    st.header("🧪 Química del Suelo (Intercambiables)")
    p = st.number_input("Fósforo (P) - Bray II", 0.0, 200.0, 20.0)
    k = st.number_input("Potasio (K)", 0.0, 10.0, 0.5)
    ca = st.number_input("Calcio (Ca)", 0.0, 50.0, 5.0)
    mg = st.number_input("Magnesio (Mg)", 0.0, 20.0, 2.0)
    al = st.number_input("Aluminio (Al)", 0.0, 10.0, 0.0)

# 4. Ingeniería de Ratios en Tiempo Real (Apalancamiento)
# Tu sistema calcula lo que el agricultor no sabe
ratio_ca_mg = ca / (mg + 0.1)
ratio_p_k = p / (k + 0.1)

# 5. Ejecución del Sistema (La Predicción)
if st.button("📊 ANALIZAR RENTABILIDAD"):
    # Preparamos los datos para el Ferrari (XGBoost)
    input_data = np.array([[dept, muni, ph, mo, p, k, ca, mg, al, ratio_ca_mg, ratio_p_k]])
    
    pred_idx = modelo.predict(input_data)[0]
    cultivo_final = le_cultivo.inverse_transform([pred_idx])[0]
    
    st.markdown("---")
    st.success(f"## 🏆 Cultivo Recomendado: {cultivo_final.upper()}")
    
    # Mostrar Ratios al Inversor
    c1, c2 = st.columns(2)
    c1.metric("Ratio Calcio/Magnesio", f"{ratio_ca_mg:.2f}")
    c2.metric("Ratio Fósforo/Potasio", f"{ratio_p_k:.2f}")
    
    st.balloons()

st.sidebar.info("Este sistema analiza 9 variables críticas y 2 ratios financieros del suelo para maximizar tu ROI agrícola.")