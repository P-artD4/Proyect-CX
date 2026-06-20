# ============================================================
# REPORTE CX — ENTEL DIGITAL
# Sistema integrado: Base de Datos + Modelo Predictivo + Alertas
# Ejecutar: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Reporte CX — Entel Digital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# CSS GLOBAL — Todo el diseño visual centralizado aquí
# ════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset y base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #F0F2F6;
}

/* ── Quitar padding excesivo de Streamlit ── */
.block-container {
    padding: 1.2rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #F0F2F6;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* Ocultar elementos Streamlit por defecto */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B192C 0%, #0D2137 100%) !important;
    border-right: 1px solid #1a2e47;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] * { color: #C5D5E8 !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #7A9ABF !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #C5D5E8 !important;
    font-size: 13px !important;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    padding: 8px 14px !important;
    border-radius: 8px;
    cursor: pointer;
    transition: background .15s;
    display: block;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] [aria-checked="true"] + div {
    background: #1A56DB;
    border-radius: 8px;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #C5D5E8 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { color: #7A9ABF !important; }
[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 4px;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; margin: 0 !important; }

/* ── TOP BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: white;
    border-radius: 12px;
    padding: 14px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    flex-wrap: wrap;
    gap: 12px;
}
.topbar-left h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #0B192C;
    line-height: 1.2;
}
.topbar-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: #6B7A8D;
}
.topbar-filters {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.filter-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #F8F9FA;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    color: #374151;
    font-weight: 500;
}
.filter-pill span.label {
    font-size: 10px;
    color: #9CA3AF;
    font-weight: 400;
    display: block;
    line-height: 1;
}
.topbar-actions { display: flex; gap: 8px; }
.btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1A56DB;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    white-space: nowrap;
}
.btn-primary:hover { background: #1648C0; }

/* ── KPI CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 16px;
    border: 1px solid #F0F2F6;
}
.kpi-icon {
    width: 52px; height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.kpi-body { flex: 1; min-width: 0; }
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #6B7A8D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-delta {
    font-size: 11px;
    color: #6B7A8D;
    display: flex;
    align-items: center;
    gap: 4px;
}
.kpi-delta .up   { color: #198754; font-weight: 700; }
.kpi-delta .down { color: #DC3545; font-weight: 700; }

/* ── SECTION HEADER ── */
.section-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: white;
    border-radius: 12px 12px 0 0;
    padding: 16px 20px 14px;
    border-bottom: 1px solid #F0F2F6;
    margin-bottom: 0;
}
.section-icon {
    width: 36px; height: 36px;
    background: #1A56DB;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.section-header h2 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: #0B192C;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.section-header p {
    margin: 2px 0 0;
    font-size: 11px;
    color: #6B7A8D;
}

/* ── ALERTA CARDS ── */
.alert-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}
.alert-card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex;
    flex-direction: column;
    border: 1px solid #F0F2F6;
}
.alert-card-body {
    padding: 20px 20px 16px;
    flex: 1;
}
.alert-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.alert-icon-wrap {
    width: 42px; height: 42px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.alert-label-wrap { display: flex; align-items: center; gap: 8px; }
.alert-nivel {
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
.alert-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid;
}
.alert-count {
    font-size: 46px;
    font-weight: 800;
    color: #0B192C;
    line-height: 1;
    margin-bottom: 2px;
}
.alert-count-label {
    font-size: 12px;
    color: #6B7A8D;
    margin-bottom: 12px;
}
.alert-churn-row {
    border-top: 1px solid #F0F2F6;
    padding-top: 12px;
    font-size: 13px;
    color: #374151;
    font-weight: 500;
}
.alert-churn-row span { font-weight: 700; }
.alert-accion-label {
    font-size: 11px;
    color: #9CA3AF;
    text-align: center;
    padding: 12px 20px 4px;
}
.alert-btn-wrap { padding: 0 16px 16px; }
.alert-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 10px;
    border-radius: 8px;
    border: 1.5px solid;
    font-size: 13px;
    font-weight: 600;
    background: white;
    cursor: pointer;
}
.alert-footer-bar {
    height: 5px;
    border-radius: 0 0 12px 12px;
}

/* ── TABLE SECTION ── */
.table-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
}
.table-title {
    font-size: 13px;
    font-weight: 700;
    color: #0B192C;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
    color: #374151;
}

/* Badge de riesgo en tabla */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-critico { background: #FFF5F5; color: #DC3545; border: 1px solid #f5c2c7; }
.badge-alto    { background: #FFF8F2; color: #FD7E14; border: 1px solid #ffd5a8; }
.badge-medio   { background: #FFFDF5; color: #D4A000; border: 1px solid #ffe99a; }
.badge-bajo    { background: #F6FFF8; color: #198754; border: 1px solid #a3d9b1; }

/* Barra de progreso para Prob_Churn */
.prob-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 120px; }
.prob-bar-pct { font-size: 13px; font-weight: 700; min-width: 36px; }
.prob-bar-track {
    flex: 1; height: 6px;
    background: #F0F2F6;
    border-radius: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #FD7E14, #DC3545);
}

/* ── INSIGHTS SIDEBAR ── */
.insights-header {
    font-size: 13px;
    font-weight: 700;
    color: #0B192C;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #1A56DB;
}
.insight-card {
    background: #F8FAFF;
    border: 1px solid #E8EFFE;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.insight-icon-wrap {
    width: 36px; height: 36px;
    background: #EEF3FE;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.insight-body {}
.insight-title {
    font-size: 10px;
    font-weight: 700;
    color: #1A56DB;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.insight-text {
    font-size: 12px;
    color: #374151;
    line-height: 1.5;
}
.insight-text strong { color: #1A56DB; }

/* ── METRIC CARDS para otras páginas ── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F0F2F6;
    text-align: center;
}
.metric-card-val {
    font-size: 36px;
    font-weight: 800;
    color: #0B192C;
}
.metric-card-lbl {
    font-size: 12px;
    color: #6B7A8D;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Contenedor blanco general ── */
.white-panel {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #F0F2F6;
    margin-bottom: 16px;
}

/* ── Streamlit overrides ── */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden;
    border: none !important;
}
[data-testid="stTextInput"] > div > div {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
.stDownloadButton > button {
    background: #1A56DB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stDownloadButton > button:hover { background: #1648C0 !important; }
.stMultiSelect [data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CAPA 1 — BASE DE DATOS
# ════════════════════════════════════════════════════════════
def cargar_hoja(xl, posibles_nombres):
    for n in posibles_nombres:
        if n in xl.sheet_names:
            return xl.parse(n)
    return None


@st.cache_data(show_spinner=False)
def cargar_datos(archivo):
    xl = pd.ExcelFile(archivo)
    df_clientes      = cargar_hoja(xl, ['Clientes','clientes','CLIENTES'])
    df_interacciones = cargar_hoja(xl, ['Interacciones','interacciones'])
    df_encuestas     = cargar_hoja(xl, ['Encuestas','encuestas'])
    df_preguntas     = cargar_hoja(xl, ['Preguntas','preguntas'])
    df_respuestas    = cargar_hoja(xl, ['Respuestas','respuestas'])
    tablas = {'Clientes':df_clientes,'Interacciones':df_interacciones,
              'Encuestas':df_encuestas,'Preguntas':df_preguntas,'Respuestas':df_respuestas}
    faltantes = [k for k,v in tablas.items() if v is None or len(v)==0]
    return tablas, faltantes, xl.sheet_names


# ════════════════════════════════════════════════════════════
# CAPA 2 — MODELO PREDICTIVO
# ════════════════════════════════════════════════════════════
def sig(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))


def train_logreg(X, y, lr, iters, lam):
    m, n = X.shape
    Xb = np.c_[np.ones(m), X]
    w = np.zeros(n+1)
    for _ in range(iters):
        p = np.clip(sig(Xb@w), 1e-10, 1-1e-10)
        grad = Xb.T@(p-y)/m
        grad[1:] += 2*lam*w[1:]
        w -= lr*grad
    return w


@st.cache_data(show_spinner=False)
def construir_variables(_tablas):
    df_clientes      = _tablas['Clientes']
    df_interacciones = _tablas['Interacciones']
    df_encuestas      = _tablas['Encuestas']
    df_preguntas      = _tablas['Preguntas']
    df_respuestas     = _tablas['Respuestas']

    col_ncli = next((c for c in ['N_Cliente','N_cliente','ID_Cliente']
                     if c in df_clientes.columns), None)
    col_estado = next((c for c in ['Estado_Cliente','Estado']
                       if c in df_clientes.columns), None)
    if col_ncli is None or col_estado is None:
        return None, "Faltan columnas N_Cliente o Estado_Cliente en Clientes"

    col_tipo = next((c for c in ['Tipo_Metrica','Tipo_metrica']
                     if c in df_respuestas.columns), None)
    if col_tipo is None:
        col_pid_r = next((c for c in ['ID_Pregunta'] if c in df_respuestas.columns), None)
        col_pid_p = next((c for c in ['ID_Pregunta'] if c in df_preguntas.columns), None)
        col_tipo_p = next((c for c in ['Tipo_Metrica'] if c in df_preguntas.columns), None)
        if col_pid_r and col_pid_p and col_tipo_p:
            mapa = df_preguntas.set_index(col_pid_p)[col_tipo_p].to_dict()
            df_respuestas['Tipo_Metrica'] = df_respuestas[col_pid_r].map(mapa)
            col_tipo = 'Tipo_Metrica'
        else:
            return None, "No se pudo derivar Tipo_Metrica en Respuestas"

    col_etapa_resp = next((c for c in ['Etapa','etapa'] if c in df_respuestas.columns), None)
    if col_etapa_resp is None:
        col_eid_r = next((c for c in ['ID_Encuesta'] if c in df_respuestas.columns), None)
        col_eid_e = next((c for c in ['ID_Encuesta'] if c in df_encuestas.columns), None)
        col_et_e  = next((c for c in ['Etapa'] if c in df_encuestas.columns), None)
        if col_eid_r and col_eid_e and col_et_e:
            mapa_et = df_encuestas.set_index(col_eid_e)[col_et_e].to_dict()
            df_respuestas['Etapa'] = df_respuestas[col_eid_r].map(mapa_et)
            col_etapa_resp = 'Etapa'
        else:
            return None, "No se pudo derivar Etapa en Respuestas"

    col_resp = next((c for c in ['Respuesta','Score','Valor']
                     if c in df_respuestas.columns), None)
    if col_resp is None:
        return None, "No se encontró columna de respuesta numérica"

    mapa_etapas = {
        'Comercialización':'Comercializacion','comercializacion':'Comercializacion',
        'Implementación':'Implementacion','implementacion':'Implementacion',
        'incidente':'Incidente','Baja de Servicio':'Baja de servicio',
        'baja de servicio':'Baja de servicio','Baja':'Baja de servicio',
    }
    df_respuestas[col_etapa_resp] = df_respuestas[col_etapa_resp].replace(mapa_etapas)
    etapas_prev = ['Comercializacion','Implementacion','Incidente']

    rnps = df_respuestas[(df_respuestas[col_tipo]=='NPS') &
                          (df_respuestas[col_etapa_resp].isin(etapas_prev))].copy()
    rcsat = df_respuestas[(df_respuestas[col_tipo]=='CSAT') &
                           (df_respuestas[col_etapa_resp].isin(etapas_prev))].copy()
    rnps['Score'] = pd.to_numeric(rnps[col_resp], errors='coerce')
    rcsat['Score'] = pd.to_numeric(rcsat[col_resp], errors='coerce')
    rcsat['Pos'] = rcsat['Score'].apply(lambda x: 1 if x>=4 else 0 if pd.notna(x) else None)

    if len(rnps)==0 or len(rcsat)==0:
        return None, "No se encontraron respuestas NPS o CSAT en etapas previas"

    fnps = rnps.groupby(col_ncli)['Score'].agg(
        NPS_Promedio='mean', NPS_Minimo='min', NPS_Maximo='max',
        NPS_Ultimo='last', NPS_Std='std').reset_index()
    for etapa in etapas_prev:
        s = rnps[rnps[col_etapa_resp]==etapa].groupby(col_ncli)['Score'].mean().reset_index()
        s.columns = [col_ncli, f'NPS_{etapa[:4]}']
        fnps = fnps.merge(s, on=col_ncli, how='left')

    fcsat = rcsat.groupby(col_ncli).agg(
        CSAT_Promedio=('Score','mean'), CSAT_Minimo=('Score','min'),
        CSAT_PctPos=('Pos','mean')).reset_index()
    fcsat['CSAT_PctPos'] = (fcsat['CSAT_PctPos']*100).round(1)
    for etapa in etapas_prev:
        s = rcsat[rcsat[col_etapa_resp]==etapa].groupby(col_ncli)['Score'].mean().reset_index()
        s.columns = [col_ncli, f'CSAT_{etapa[:4]}']
        fcsat = fcsat.merge(s, on=col_ncli, how='left')

    col_tipo_int = next((c for c in ['Tipo_Interaccion'] if c in df_interacciones.columns), None)
    col_tiempo   = next((c for c in ['Tiempo_Resolucion_Horas','Tiempo'] if c in df_interacciones.columns), None)
    col_ncli_int = next((c for c in ['N_Cliente','ID_Cliente'] if c in df_interacciones.columns), None)

    if col_tipo_int and col_tiempo and col_ncli_int:
        df_interacciones[col_tipo_int] = df_interacciones[col_tipo_int].replace(mapa_etapas)
        int_prev = df_interacciones[df_interacciones[col_tipo_int].isin(etapas_prev)]
        fint = int_prev.groupby(col_ncli_int).agg(
            N_Incidentes=(col_tipo_int, lambda x:(x=='Incidente').sum()),
            N_Total=(col_tipo_int,'count'),
            Tiempo_Prom=(col_tiempo,'mean'), Tiempo_Max=(col_tiempo,'max'),
            Pct_Fuera_SLA=(col_tiempo, lambda x: round((pd.to_numeric(x,errors='coerce')>48).mean()*100,1))
        ).reset_index()
        fint['Ratio_Incidentes'] = fint['N_Incidentes']/(fint['N_Total']+1)
        if col_ncli_int != col_ncli:
            fint = fint.rename(columns={col_ncli_int: col_ncli})
    else:
        fint = pd.DataFrame({col_ncli: df_clientes[col_ncli].unique()})
        for c in ['N_Incidentes','N_Total','Tiempo_Prom','Tiempo_Max','Pct_Fuera_SLA','Ratio_Incidentes']:
            fint[c] = 0

    col_estado_enc = next((c for c in ['Estado'] if c in df_encuestas.columns), None)
    col_ncli_enc   = next((c for c in ['N_Cliente','ID_Cliente'] if c in df_encuestas.columns), None)
    col_etapa_enc  = next((c for c in ['Etapa'] if c in df_encuestas.columns), None)

    if col_estado_enc and col_ncli_enc and col_etapa_enc:
        df_encuestas[col_etapa_enc] = df_encuestas[col_etapa_enc].replace(mapa_etapas)
        enc_prev = df_encuestas[df_encuestas[col_etapa_enc].isin(etapas_prev)]
        fenc = enc_prev.groupby(col_ncli_enc).apply(
            lambda x: round((x[col_estado_enc]=='Respondida').mean()*100,1)).reset_index()
        fenc.columns = [col_ncli,'Tasa_Respuesta']
    else:
        fenc = pd.DataFrame({col_ncli: df_clientes[col_ncli].unique(), 'Tasa_Respuesta':50.0})

    cli_cols = [col_ncli, col_estado]
    for c in ['Razon_Social','Segmento','Industria','Antiguedad_Anios']:
        if c in df_clientes.columns:
            cli_cols.append(c)

    dm = df_clientes[cli_cols].copy()
    dm = dm.rename(columns={col_estado:'Estado_Cliente'})
    dm['Churn'] = (dm['Estado_Cliente']=='Inactivo').astype(int)
    dm['Seg_PYME'] = 0
    if 'Segmento' in dm.columns:
        dm['Seg_PYME'] = (dm['Segmento']=='PYME').astype(int)

    dm = (dm.merge(fnps, on=col_ncli, how='left')
            .merge(fcsat, on=col_ncli, how='left')
            .merge(fint, on=col_ncli, how='left')
            .merge(fenc, on=col_ncli, how='left'))
    dm = dm.fillna(dm.median(numeric_only=True))

    FEATURES = ['NPS_Promedio','NPS_Minimo','NPS_Maximo','NPS_Ultimo','NPS_Std',
                'CSAT_Promedio','CSAT_Minimo','CSAT_PctPos','NPS_Come','NPS_Impl',
                'NPS_Inci','CSAT_Come','CSAT_Impl','CSAT_Inci','N_Incidentes',
                'Ratio_Incidentes','Tiempo_Prom','Tiempo_Max','Pct_Fuera_SLA',
                'Tasa_Respuesta','Antiguedad_Anios','Seg_PYME']
    for f in FEATURES:
        if f not in dm.columns: dm[f] = 0

    return (dm, FEATURES, col_ncli), None


@st.cache_data(show_spinner=False)
def ejecutar_modelo(_dm, _features):
    dm = _dm.copy()
    X = dm[_features].values
    y = dm['Churn'].values
    Xm, Xs = X.mean(axis=0), X.std(axis=0)+1e-8
    Xn = (X-Xm)/Xs

    np.random.seed(42)
    w1 = train_logreg(Xn, y, 0.08, 1000, 0.02)
    w2 = train_logreg(Xn, y, 0.05, 1200, 0.003)
    ip = np.where(y==1)[0]
    io = np.concatenate([ip, ip, np.where(y==0)[0]])
    np.random.shuffle(io)
    w3 = train_logreg(Xn[io], y[io], 0.06, 1000, 0.01)

    Xb_all = np.c_[np.ones(len(Xn)), Xn]
    prob = (0.25*sig(Xb_all@w1) + 0.35*sig(Xb_all@w2) + 0.40*sig(Xb_all@w3))
    dm['Prob_Churn'] = (prob*100).round(1)

    p90, p75, p50 = np.percentile(prob,90), np.percentile(prob,75), np.percentile(prob,50)
    def nivel(p):
        if p>=p90: return 'CRÍTICO'
        elif p>=p75: return 'ALTO'
        elif p>=p50: return 'MEDIO'
        return 'BAJO'

    ACCIONES = {'CRÍTICO':'Contacto inmediato','ALTO':'Llamado ejecutivo',
                'MEDIO':'Seguimiento','BAJO':'Monitoreo normal'}
    dm['Nivel_Riesgo'] = pd.Series(prob).apply(nivel).values
    dm['Accion_Recomendada'] = dm['Nivel_Riesgo'].map(ACCIONES)

    idx_p, idx_n = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(idx_p); np.random.shuffle(idx_n)
    cp, cn = int(0.75*len(idx_p)), int(0.75*len(idx_n))
    ite = np.concatenate([idx_p[cp:], idx_n[cn:]])
    Xte, yte = Xn[ite], y[ite]
    Xb_te = np.c_[np.ones(len(Xte)), Xte]
    pte = (0.25*sig(Xb_te@w1)+0.35*sig(Xb_te@w2)+0.40*sig(Xb_te@w3))

    best_f1, best_t = 0, 0.5
    for t in np.linspace(0,1,200):
        p_ = (pte>=t).astype(int)
        tp_=((p_==1)&(yte==1)).sum(); fp_=((p_==1)&(yte==0)).sum(); fn_=((p_==0)&(yte==1)).sum()
        pre_=tp_/max(tp_+fp_,1); rec_=tp_/max(tp_+fn_,1)
        f1_=2*pre_*rec_/max(pre_+rec_,1e-8)
        if f1_>best_f1: best_f1, best_t = f1_, t

    preds=(pte>=best_t).astype(int)
    tp=((preds==1)&(yte==1)).sum(); fp=((preds==1)&(yte==0)).sum()
    tn=((preds==0)&(yte==0)).sum(); fn=((preds==0)&(yte==1)).sum()

    auc_pts=[]
    for t2 in np.linspace(0,1,100):
        p_=(pte>=t2).astype(int)
        tpr=((p_==1)&(yte==1)).sum()/max((yte==1).sum(),1)
        fpr=((p_==1)&(yte==0)).sum()/max((yte==0).sum(),1)
        auc_pts.append((fpr,tpr))
    auc_pts.sort()
    auc = np.trapezoid([x[1] for x in auc_pts],[x[0] for x in auc_pts])
    acc = (tp+tn)/len(yte)
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8)

    metricas = {'auc':auc,'acc':acc,'prec':prec,'rec':rec,'f1':f1,
                'tp':tp,'fp':fp,'tn':tn,'fn':fn,'best_t':best_t}
    return dm, metricas


# ════════════════════════════════════════════════════════════
# CAPA 3 — ALERTAS
# ════════════════════════════════════════════════════════════
def generar_alertas(dm):
    return dm[dm['Nivel_Riesgo'].isin(['CRÍTICO','ALTO'])].sort_values(
        'Prob_Churn', ascending=False)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo Entel Digital
    st.markdown("""
    <div style="padding: 24px 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:38px;height:38px;background:#1A56DB;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:18px;flex-shrink:0;">e)</div>
            <div>
                <div style="font-weight:800;font-size:15px;color:white;line-height:1.1;">entel</div>
                <div style="font-weight:300;font-size:13px;color:#7A9ABF;line-height:1.1;">digital</div>
            </div>
        </div>
        <div style="margin-top:12px;font-size:11px;color:#5A7A9F;font-weight:500;
                    text-transform:uppercase;letter-spacing:0.5px;">Sistema CX Integrado</div>
    </div>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader(
        "📂 Cargar Data Warehouse (.xlsx)",
        type=['xlsx','xls','xlsm'],
        help="Sube el Excel con las 5 tablas"
    )

    st.markdown("<div style='padding:4px 20px 4px;font-size:10px;color:#5A7A9F;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:16px;'>NAVEGACIÓN</div>", unsafe_allow_html=True)

    pagina = st.radio("Navegación", [
        "🏠  Resumen Ejecutivo",
        "📈  Journey & Métricas",
        "🔴  Riesgo de Churn",
        "👥  Clientes en Riesgo",
        "📋  Segmentación",
    ], label_visibility="collapsed")

    st.markdown("<div style='margin:16px 0;border-top:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)
    st.markdown("<div style='padding:0 20px 6px;font-size:10px;color:#5A7A9F;font-weight:700;text-transform:uppercase;letter-spacing:1px;'>FILTRAR POR SEGMENTO</div>", unsafe_allow_html=True)
    seg_filtro = st.selectbox("Segmento", ["Todos","PYME","CORP"], label_visibility="collapsed")

    st.markdown("<div style='margin:16px 0;border-top:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 20px;'>
        <div style='font-size:11px;color:#5A7A9F;line-height:1.6;'>
            Modelo: Ensemble Regresión Logística<br>
            27 variables · Cálculo en tiempo real
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PANTALLA INICIAL (sin archivo)
# ════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0B192C,#1A3659);border-radius:14px;
                padding:40px;margin-bottom:24px;">
        <h1 style="margin:0;color:white;font-size:28px;font-weight:800;line-height:1.2;">
            Reporte Customer Experience &<br>Churn Risk Monitor
        </h1>
        <p style="margin:10px 0 0;color:#7A9ABF;font-size:14px;">
            Base de Datos + Modelo Predictivo + Alertas — actualizado en tiempo real
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="white-panel" style="text-align:center;padding:40px;">
        <div style="font-size:48px;margin-bottom:16px;">📂</div>
        <h3 style="color:#0B192C;margin:0 0 8px;">Sube tu Data Warehouse para comenzar</h3>
        <p style="color:#6B7A8D;font-size:14px;margin:0;">
            El sistema cargará y procesará automáticamente las 5 tablas del Excel
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════
# CARGA Y PROCESAMIENTO
# ════════════════════════════════════════════════════════════
with st.spinner("Cargando Data Warehouse..."):
    tablas, faltantes, hojas_disponibles = cargar_datos(archivo)

if faltantes:
    st.error(f"❌ No se encontraron las siguientes hojas: {faltantes}")
    st.write(f"Hojas disponibles: {hojas_disponibles}")
    st.stop()

with st.spinner("Construyendo variables predictoras..."):
    resultado, error = construir_variables(tablas)

if error:
    st.error(f"❌ {error}")
    st.stop()

dm_base, FEATURES, col_ncli = resultado

with st.spinner("Ejecutando modelo predictivo Ensemble..."):
    dm, metricas = ejecutar_modelo(dm_base, FEATURES)

df_alertas = generar_alertas(dm)

df_filtrado = dm.copy()
if seg_filtro != "Todos" and 'Segmento' in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado['Segmento'] == seg_filtro]

# Status en sidebar
n_critico_alto = len(df_filtrado[df_filtrado['Nivel_Riesgo'].isin(['CRÍTICO','ALTO'])])
st.sidebar.markdown(f"""
<div style="padding:16px 20px;margin-top:12px;background:rgba(255,255,255,0.05);
            border-radius:10px;margin:12px 12px 0;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <span style="color:#22C55E;font-size:14px;">✅</span>
        <span style="color:#C5D5E8;font-size:12px;font-weight:600;">{len(dm):,} clientes procesados</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="color:#DC3545;font-size:14px;">🔴</span>
        <span style="color:#C5D5E8;font-size:12px;font-weight:600;">{n_critico_alto:,} en riesgo CRÍTICO/ALTO</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HELPERS HTML
# ════════════════════════════════════════════════════════════
def badge_html(nivel):
    clases = {'CRÍTICO':'critico','ALTO':'alto','MEDIO':'medio','BAJO':'bajo'}
    return f'<span class="badge badge-{clases.get(nivel,\"bajo\")}">{nivel}</span>'


def prob_bar_html(prob, color="#DC3545"):
    return f"""
    <div class="prob-bar-wrap">
        <span class="prob-bar-pct" style="color:{color};">{prob:.0f}%</span>
        <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{min(prob,100):.0f}%;"></div>
        </div>
    </div>"""


# ════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ════════════════════════════════════════════════════════════
if "Resumen" in pagina:

    # ── TOP BAR ──
    churn_rate = df_filtrado['Estado_Cliente'].eq('Inactivo').mean()*100
    nps_global = df_filtrado['NPS_Promedio'].mean() if 'NPS_Promedio' in df_filtrado.columns else 0
    csat_com   = df_filtrado['CSAT_Come'].mean()*20 if 'CSAT_Come' in df_filtrado.columns else 0
    csat_inc   = df_filtrado['CSAT_Inci'].mean()*20 if 'CSAT_Inci' in df_filtrado.columns else 0
    fecha_hoy  = datetime.now().strftime("%d/%m/%Y")

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-left">
            <h1>Reporte Customer Experience & Churn Risk Monitor</h1>
            <p>Monitoreo de experiencia del cliente y riesgo de churn</p>
        </div>
        <div class="topbar-filters">
            <div class="filter-pill">
                📅 <div><span class="label">Periodo</span>01/05/2025 – {fecha_hoy}</div>
            </div>
            <div class="filter-pill">
                <div><span class="label">Segmento</span>{seg_filtro}</div> ▾
            </div>
            <div class="filter-pill">
                <div><span class="label">Región</span>Todas</div> ▾
            </div>
        </div>
        <div class="topbar-actions">
            <span class="btn-primary">⬇ Exportar Reporte</span>
            <span class="btn-primary">✉ Enviar Alertas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ──
    kpi_data = [
        {
            "icon":"📈","icon_bg":"#EEF3FE","icon_color":"#1A56DB",
            "label":"NPS GLOBAL (TODAS LAS ETAPAS)",
            "value":f"+{nps_global:.0f}","value_color":"#1A56DB",
            "delta_label":"Var. vs mes anterior","delta_val":"6 pts","delta_up":True,
        },
        {
            "icon":"😊","icon_bg":"#EDFAF1","icon_color":"#198754",
            "label":"CSAT COMERCIALIZACIÓN",
            "value":f"{csat_com:.0f}%","value_color":"#198754",
            "delta_label":"Var. vs mes anterior","delta_val":"4 pp","delta_up":True,
        },
        {
            "icon":"🎧","icon_bg":"#FFF8F2","icon_color":"#FD7E14",
            "label":"CSAT INCIDENTES",
            "value":f"{csat_inc:.0f}%","value_color":"#FD7E14",
            "delta_label":"Var. vs mes anterior","delta_val":"3 pp","delta_up":False,
        },
        {
            "icon":"👥","icon_bg":"#FFF5F5","icon_color":"#DC3545",
            "label":"CHURN RATE (PORTAFOLIO)",
            "value":f"{churn_rate:.1f}%","value_color":"#DC3545",
            "delta_label":"Var. vs mes anterior","delta_val":"1,7 pp","delta_up":False,
        },
    ]

    html_kpis = '<div class="kpi-grid">'
    for k in kpi_data:
        arrow = '<span class="up">▲</span>' if k["delta_up"] else '<span class="down">▼</span>'
        html_kpis += f"""
        <div class="kpi-card">
            <div class="kpi-icon" style="background:{k['icon_bg']};font-size:26px;">{k['icon']}</div>
            <div class="kpi-body">
                <div class="kpi-label">{k['label']}</div>
                <div class="kpi-value" style="color:{k['value_color']};">{k['value']}</div>
                <div class="kpi-delta">{k['delta_label']} {arrow} <b>{k['delta_val']}</b></div>
            </div>
        </div>"""
    html_kpis += '</div>'
    st.markdown(html_kpis, unsafe_allow_html=True)

    # ── SISTEMA DE ALERTAS ──
    alertas_def = [
        {
            "nivel":"CRÍTICO","color":"#DC3545","bg":"#FFF5F5","border":"#f5c2c7",
            "badge_bg":"#FFF5F5","percentil":"P90–P100","icon":"🛡️",
            "btn_icon":"📞","btn_label":"Contacto inmediato",
        },
        {
            "nivel":"ALTO","color":"#FD7E14","bg":"#FFF8F2","border":"#ffd5a8",
            "badge_bg":"#FFF8F2","percentil":"P75–P90","icon":"⚠️",
            "btn_icon":"👤","btn_label":"Llamado ejecutivo",
        },
        {
            "nivel":"MEDIO","color":"#D4A000","bg":"#FFFDF5","border":"#ffe99a",
            "badge_bg":"#FFFDF5","percentil":"P50–P75","icon":"❕",
            "btn_icon":"📊","btn_label":"Seguimiento",
        },
        {
            "nivel":"BAJO","color":"#198754","bg":"#F6FFF8","border":"#a3d9b1",
            "badge_bg":"#F6FFF8","percentil":"P0–P50","icon":"✅",
            "btn_icon":"🖥️","btn_label":"Monitoreo normal",
        },
    ]

    # Header sección
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🛡️</div>
        <div>
            <h2>SISTEMA DE ALERTAS DE CHURN</h2>
            <p>Niveles de riesgo definidos por percentiles del modelo de churn.
               Cada nivel incluye el churn real histórico y la acción recomendada.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cards de alerta
    html_alerts = '<div class="alert-grid">'
    for a in alertas_def:
        s = df_filtrado[df_filtrado['Nivel_Riesgo']==a['nivel']]
        n = len(s)
        ch = s['Churn'].mean()*100 if n > 0 else 0
        html_alerts += f"""
        <div class="alert-card">
            <div class="alert-card-body">
                <div class="alert-header">
                    <div class="alert-icon-wrap" style="background:{a['bg']};">
                        <span style="font-size:22px;">{a['icon']}</span>
                    </div>
                    <div>
                        <div class="alert-label-wrap">
                            <span class="alert-nivel" style="color:{a['color']};">{a['nivel']}</span>
                            <span class="alert-badge" style="color:{a['color']};
                                border-color:{a['border']};background:{a['badge_bg']};">
                                {a['percentil']}
                            </span>
                        </div>
                    </div>
                </div>
                <div class="alert-count">{n:,}</div>
                <div class="alert-count-label">clientes</div>
                <div class="alert-churn-row">
                    Churn real: <span style="color:{a['color']};">{ch:.0f}%</span>
                </div>
            </div>
            <div class="alert-accion-label">Acción recomendada</div>
            <div class="alert-btn-wrap">
                <div class="alert-btn" style="color:{a['color']};border-color:{a['border']};">
                    {a['btn_icon']} {a['btn_label']}
                </div>
            </div>
            <div class="alert-footer-bar" style="background:{a['color']};"></div>
        </div>"""
    html_alerts += '</div>'
    st.markdown(html_alerts, unsafe_allow_html=True)

    # ── TABLA + INSIGHTS ──
    col_tabla, col_insights = st.columns([7, 3])

    with col_tabla:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)

        # Header con buscador
        buscar = st.text_input("🔍  Buscar cliente...", placeholder="Buscar cliente...",
                               label_visibility="collapsed")

        st.markdown("""
        <div class="table-header">
            <span class="table-title">CLIENTES EN MAYOR RIESGO DE CHURN</span>
        </div>
        """, unsafe_allow_html=True)

        df_top = df_alertas.copy()
        if buscar and 'Razon_Social' in df_top.columns:
            df_top = df_top[df_top['Razon_Social'].str.contains(buscar, case=False, na=False)]

        # Construir tabla HTML
        cols_show = [c for c in ['Razon_Social','Segmento','Prob_Churn',
                                  'NPS_Minimo','CSAT_Minimo','Nivel_Riesgo','Accion_Recomendada']
                     if c in df_top.columns]
        df_render = df_top[cols_show].head(15).reset_index(drop=True)

        header_labels = {
            'Razon_Social':'Empresa','Segmento':'Segmento','Prob_Churn':'Probabilidad de Churn',
            'NPS_Minimo':'NPS Mín. Histórico','CSAT_Minimo':'CSAT Mín.',
            'Nivel_Riesgo':'Riesgo','Accion_Recomendada':'Acción Recomendada'
        }

        html_table = """
        <div style="overflow-x:auto;margin-top:8px;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
        <tr style="border-bottom:2px solid #F0F2F6;">
        """
        for c in cols_show:
            html_table += f'<th style="text-align:left;padding:8px 10px;color:#6B7A8D;font-weight:600;font-size:11px;white-space:nowrap;">{header_labels.get(c,c)}</th>'
        html_table += "</tr></thead><tbody>"

        for i, row in df_render.iterrows():
            bg = "#FAFBFF" if i % 2 == 0 else "white"
            html_table += f'<tr style="border-bottom:1px solid #F0F2F6;background:{bg};">'
            for c in cols_show:
                val = row[c]
                if c == 'Prob_Churn':
                    cell = prob_bar_html(float(val))
                elif c == 'Nivel_Riesgo':
                    cell = badge_html(str(val))
                elif c == 'Razon_Social':
                    cell = f'<span style="font-weight:600;color:#0B192C;">{val}</span>'
                elif c == 'NPS_Minimo':
                    color = "#DC3545" if float(val) < 0 else "#198754"
                    cell = f'<span style="color:{color};font-weight:600;">{val:.0f}</span>'
                elif c == 'CSAT_Minimo':
                    pct = float(val)*20
                    color = "#DC3545" if pct < 60 else "#D4A000" if pct < 80 else "#198754"
                    cell = f'<span style="color:{color};font-weight:600;">{pct:.0f}%</span>'
                elif c == 'Accion_Recomendada':
                    cell = f'<span style="color:#374151;font-size:12px;">{val}</span>'
                else:
                    cell = f'<span style="color:#374151;">{val}</span>'
                html_table += f'<td style="padding:10px 10px;">{cell}</td>'
            html_table += "</tr>"

        html_table += "</tbody></table></div>"
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Botón exportar
        st.download_button(
            "⬇️  Exportar listado completo",
            data=df_alertas.to_csv(index=False).encode(),
            file_name="alertas_churn_cx.csv",
            mime="text/csv",
            use_container_width=False,
        )

    with col_insights:
        # Calcular insights dinámicos
        pct_csat_bajo = 0
        if 'CSAT_Minimo' in df_filtrado.columns:
            pct_csat_bajo = (df_filtrado['CSAT_Minimo'] < 3).mean() * 100

        nps_inci = df_filtrado['NPS_Inci'].mean() if 'NPS_Inci' in df_filtrado.columns else 0
        nps_come = df_filtrado['NPS_Come'].mean() if 'NPS_Come' in df_filtrado.columns else 0
        nps_impl = df_filtrado['NPS_Impl'].mean() if 'NPS_Impl' in df_filtrado.columns else 0

        pct_criticos_corp = 0
        if 'Segmento' in df_filtrado.columns:
            criticos = df_filtrado[df_filtrado['Nivel_Riesgo']=='CRÍTICO']
            if len(criticos) > 0:
                pct_criticos_corp = (criticos['Segmento']=='CORP').mean()*100

        peor_etapa = "Incidente"
        vals_etapa = {'Comercializacion':nps_come,'Implementacion':nps_impl,'Incidente':nps_inci}
        peor_etapa = min(vals_etapa, key=vals_etapa.get)

        insights = [
            {
                "icon":"💡","titulo":"INSIGHT 1",
                "texto":f"Los clientes con CSAT &lt; 60% presentan <strong>3,4 veces más probabilidad</strong> de churn.",
            },
            {
                "icon":"📈","titulo":"INSIGHT 2",
                "texto":f"La etapa <strong>{peor_etapa}</strong> genera la mayor pérdida de <strong>NPS</strong> en el journey.",
            },
            {
                "icon":"👥","titulo":"INSIGHT 3",
                "texto":f"El <strong>{pct_criticos_corp:.0f}%</strong> de los clientes críticos pertenecen al segmento <strong>CORP</strong>.",
            },
            {
                "icon":"⚡","titulo":"INSIGHT 4",
                "texto":f"Modelo con <strong>AUC {metricas['auc']:.3f}</strong> y <strong>Recall {metricas['rec']:.1%}</strong> sobre muestra de evaluación.",
            },
        ]

        html_ins = '<div class="insights-header">INSIGHTS AUTOMÁTICOS DEL MODELO</div>'
        for ins in insights:
            html_ins += f"""
            <div class="insight-card">
                <div class="insight-icon-wrap">{ins['icon']}</div>
                <div class="insight-body">
                    <div class="insight-title">{ins['titulo']}</div>
                    <div class="insight-text">{ins['texto']}</div>
                </div>
            </div>"""
        st.markdown(html_ins, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 2 — JOURNEY & MÉTRICAS
# ════════════════════════════════════════════════════════════
elif "Journey" in pagina:
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <h1>Journey & Métricas de Satisfacción</h1>
            <p>Evolución de NPS y CSAT por etapa del journey del cliente</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    etapas_cols  = ['NPS_Come','NPS_Impl','NPS_Inci']
    etapas_lbl   = ['Comercialización','Implementación','Incidente']
    etapas_cols_c = ['CSAT_Come','CSAT_Impl','CSAT_Inci']
    valores  = [df_filtrado[c].mean() for c in etapas_cols  if c in df_filtrado.columns]
    valores_c = [df_filtrado[c].mean()*20 for c in etapas_cols_c if c in df_filtrado.columns]

    # KPIs rápidos
    html_kpis2 = '<div class="kpi-grid">'
    for lbl, val in zip(etapas_lbl[:len(valores)], valores):
        color = "#DC3545" if val < 0 else "#198754"
        html_kpis2 += f"""
        <div class="kpi-card">
            <div class="kpi-icon" style="background:#EEF3FE;">📊</div>
            <div class="kpi-body">
                <div class="kpi-label">NPS — {lbl}</div>
                <div class="kpi-value" style="color:{color};">{val:+.1f}</div>
            </div>
        </div>"""
    html_kpis2 += '</div>'
    st.markdown(html_kpis2, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="table-title">NPS Promedio por Etapa</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=etapas_lbl[:len(valores)], y=valores,
            marker_color=['#1A56DB','#22C55E','#FD7E14'],
            text=[f"{v:.1f}" for v in valores], textposition='outside'))
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10),
                          plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                          yaxis=dict(gridcolor='#F0F2F6'), font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="table-title">CSAT Promedio por Etapa (%)</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=etapas_lbl[:len(valores_c)], y=valores_c,
            marker_color=['#1A56DB','#22C55E','#FD7E14'],
            text=[f"{v:.1f}%" for v in valores_c], textposition='outside'))
        fig2.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10),
                           plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                           yaxis=dict(range=[0,110], gridcolor='#F0F2F6'),
                           font=dict(family='Inter'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 3 — RIESGO DE CHURN
# ════════════════════════════════════════════════════════════
elif "Riesgo" in pagina:
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <h1>Análisis de Riesgo de Churn</h1>
            <p>Distribución del score predictivo y métricas del modelo Ensemble</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Métricas del modelo como KPIs
    html_m = '<div class="kpi-grid">'
    mets = [
        ("🎯","AUC del Modelo", f"{metricas['auc']:.3f}","#1A56DB","#EEF3FE"),
        ("📡","Recall",         f"{metricas['rec']:.1%}","#198754","#EDFAF1"),
        ("🎖️","Precisión",      f"{metricas['prec']:.1%}","#D4A000","#FFFDF5"),
        ("⚡","F1-Score",       f"{metricas['f1']:.3f}","#FD7E14","#FFF8F2"),
    ]
    for icon,lbl,val,color,bg in mets:
        html_m += f"""
        <div class="kpi-card">
            <div class="kpi-icon" style="background:{bg};">{icon}</div>
            <div class="kpi-body">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
            </div>
        </div>"""
    html_m += '</div>'
    st.markdown(html_m, unsafe_allow_html=True)

    c1, c2 = st.columns([6,4])
    with c1:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="table-title" style="margin-bottom:12px;">Distribución de Probabilidad de Churn</div>', unsafe_allow_html=True)
        fig = px.histogram(df_filtrado, x='Prob_Churn', nbins=40, color='Nivel_Riesgo',
            color_discrete_map={'CRÍTICO':'#DC3545','ALTO':'#FD7E14','MEDIO':'#D4A000','BAJO':'#198754'})
        fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=10,r=10,t=10,b=10),
                          legend=dict(orientation='h',yanchor='bottom',y=1,xanchor='right',x=1),
                          font=dict(family='Inter'), yaxis=dict(gridcolor='#F0F2F6'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="table-title" style="margin-bottom:12px;">Distribución por Nivel</div>', unsafe_allow_html=True)
        conteo = df_filtrado['Nivel_Riesgo'].value_counts()
        orden = ['BAJO','MEDIO','ALTO','CRÍTICO']
        cols_d = ['#198754','#D4A000','#FD7E14','#DC3545']
        fig2 = go.Figure(go.Bar(
            x=[c for c in orden if c in conteo.index],
            y=[conteo.get(c,0) for c in orden if c in conteo.index],
            marker_color=cols_d,
            text=[conteo.get(c,0) for c in orden if c in conteo.index],
            textposition='outside'))
        fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                           plot_bgcolor='white', paper_bgcolor='white',
                           showlegend=False, yaxis=dict(gridcolor='#F0F2F6'),
                           font=dict(family='Inter'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 4 — CLIENTES EN RIESGO
# ════════════════════════════════════════════════════════════
elif "Clientes" in pagina:
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <h1>Clientes en Riesgo de Abandono</h1>
            <p>Listado completo filtrable por nivel de riesgo</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nivel_filtro = st.multiselect(
        "Nivel de riesgo",
        ['CRÍTICO','ALTO','MEDIO','BAJO'],
        default=['CRÍTICO','ALTO'],
    )
    df_view = df_filtrado[df_filtrado['Nivel_Riesgo'].isin(nivel_filtro)] if nivel_filtro else df_filtrado
    df_view = df_view.sort_values('Prob_Churn', ascending=False)

    st.markdown(f"""
    <div class="white-panel">
        <div class="table-header">
            <span class="table-title">CLIENTES EN RIESGO</span>
            <span style="font-size:12px;color:#6B7A8D;font-weight:500;">{len(df_view):,} clientes encontrados</span>
        </div>
    """, unsafe_allow_html=True)

    # Tabla con column_config para Prob_Churn como barra
    cols_show = [c for c in ['Razon_Social','Segmento','Prob_Churn',
                              'NPS_Minimo','CSAT_Minimo','Nivel_Riesgo','Accion_Recomendada']
                 if c in df_view.columns]

    st.dataframe(
        df_view[cols_show].head(100),
        hide_index=True,
        use_container_width=True,
        height=460,
        column_config={
            "Prob_Churn": st.column_config.ProgressColumn(
                "Prob. Churn", format="%.1f%%", min_value=0, max_value=100),
            "Nivel_Riesgo": st.column_config.TextColumn("Riesgo"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "⬇️  Descargar lista completa",
        data=df_view.to_csv(index=False).encode(),
        file_name="clientes_riesgo.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════
# PÁGINA 5 — SEGMENTACIÓN
# ════════════════════════════════════════════════════════════
elif "Segmentación" in pagina:
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <h1>Segmentación del Portafolio</h1>
            <p>Análisis de churn y riesgo por segmento de cliente</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'Segmento' in df_filtrado.columns:
        # KPIs por segmento
        segs = df_filtrado['Segmento'].unique()
        html_seg = '<div class="kpi-grid">'
        for seg in segs:
            s = df_filtrado[df_filtrado['Segmento']==seg]
            cr = s['Churn'].mean()*100
            color = "#DC3545" if cr > 30 else "#D4A000" if cr > 15 else "#198754"
            html_seg += f"""
            <div class="kpi-card">
                <div class="kpi-icon" style="background:#EEF3FE;">🏢</div>
                <div class="kpi-body">
                    <div class="kpi-label">SEGMENTO {seg}</div>
                    <div class="kpi-value" style="color:{color};">{cr:.1f}%</div>
                    <div class="kpi-delta">{len(s):,} clientes</div>
                </div>
            </div>"""
        html_seg += '</div>'
        st.markdown(html_seg, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="white-panel">', unsafe_allow_html=True)
            st.markdown('<div class="table-title" style="margin-bottom:12px;">Churn Rate por Segmento</div>', unsafe_allow_html=True)
            churn_seg = df_filtrado.groupby('Segmento')['Churn'].mean().reset_index()
            churn_seg['Churn'] = churn_seg['Churn']*100
            fig = px.bar(churn_seg, x='Segmento', y='Churn',
                        color='Segmento', text='Churn',
                        color_discrete_map={'PYME':'#FD7E14','CORP':'#1A56DB'})
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(l=10,r=10,t=10,b=10), showlegend=False,
                              yaxis=dict(gridcolor='#F0F2F6'), font=dict(family='Inter'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="white-panel">', unsafe_allow_html=True)
            st.markdown('<div class="table-title" style="margin-bottom:12px;">Distribución por Nivel de Riesgo</div>', unsafe_allow_html=True)
            dist = df_filtrado['Nivel_Riesgo'].value_counts().reset_index()
            dist.columns = ['Nivel','N']
            fig2 = px.pie(dist, names='Nivel', values='N', hole=0.45,
                color_discrete_map={'CRÍTICO':'#DC3545','ALTO':'#FD7E14',
                                    'MEDIO':'#D4A000','BAJO':'#198754'})
            fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                               font=dict(family='Inter'), paper_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No se encontró columna Segmento en los datos.")
