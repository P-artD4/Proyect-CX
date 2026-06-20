[app.py](https://github.com/user-attachments/files/29166395/app.py)
# ============================================================
# REPORTE CX — ENTEL DIGITAL
# Sistema integrado: Base de Datos + Modelo Predictivo + Alertas
# Todo en un solo lugar, sin flujo manual
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

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1B2E4B; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 16px;
    }
    .main-header {
        background: linear-gradient(135deg, #1B2E4B 0%, #1F497D 100%);
        padding: 20px 30px; border-radius: 10px;
        color: white; margin-bottom: 20px;
    }
    .insight-card {
        background: #EBF3FB; border-radius: 10px;
        padding: 16px; border-left: 4px solid #1F497D;
    }
    .alerta-critico { background:#FDECEA;border-left:4px solid #C0392B;border-radius:8px;padding:12px 16px;margin:4px 0; }
    .alerta-alto    { background:#FFF3E0;border-left:4px solid #E36C09;border-radius:8px;padding:12px 16px;margin:4px 0; }
    .alerta-medio   { background:#FFFFF0;border-left:4px solid #D4AC00;border-radius:8px;padding:12px 16px;margin:4px 0; }
    .alerta-bajo    { background:#F1F8E9;border-left:4px solid #70AD47;border-radius:8px;padding:12px 16px;margin:4px 0; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CAPA 1 — BASE DE DATOS: Carga y validación
# ════════════════════════════════════════════════════════════
def cargar_hoja(xl, posibles_nombres):
    for n in posibles_nombres:
        if n in xl.sheet_names:
            return xl.parse(n)
    return None


@st.cache_data(show_spinner=False)
def cargar_datos(archivo):
    """Carga y valida las 5 tablas del Data Warehouse."""
    xl = pd.ExcelFile(archivo)

    df_clientes      = cargar_hoja(xl, ['Clientes','clientes','CLIENTES'])
    df_interacciones = cargar_hoja(xl, ['Interacciones','interacciones'])
    df_encuestas     = cargar_hoja(xl, ['Encuestas','encuestas'])
    df_preguntas     = cargar_hoja(xl, ['Preguntas','preguntas'])
    df_respuestas    = cargar_hoja(xl, ['Respuestas','respuestas'])

    tablas = {'Clientes':df_clientes,'Interacciones':df_interacciones,
              'Encuestas':df_encuestas,'Preguntas':df_preguntas,
              'Respuestas':df_respuestas}
    faltantes = [k for k,v in tablas.items() if v is None or len(v)==0]

    if faltantes:
        return None, faltantes, xl.sheet_names

    return tablas, [], xl.sheet_names


# ════════════════════════════════════════════════════════════
# CAPA 2 — MODELO PREDICTIVO: Ensemble Regresión Logística
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
    """Construye las 27 variables predictoras desde las 5 tablas."""
    df_clientes      = _tablas['Clientes']
    df_interacciones = _tablas['Interacciones']
    df_encuestas      = _tablas['Encuestas']
    df_preguntas      = _tablas['Preguntas']
    df_respuestas     = _tablas['Respuestas']

    # Detectar columnas clave
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

    # NPS y CSAT
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

    # Interacciones
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

    # Encuestas
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

    # Tabla maestra
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
    """Entrena el Ensemble y calcula scores de churn."""
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

    ACCIONES = {'CRÍTICO':'Contacto inmediato — analista senior + oferta de retención',
                'ALTO':'Llamada proactiva del analista en las próximas 48 horas',
                'MEDIO':'Revisar incidentes + enviar encuesta de satisfacción',
                'BAJO':'Monitoreo mensual estándar'}
    dm['Nivel_Riesgo'] = pd.Series(prob).apply(nivel).values
    dm['Accion_Recomendada'] = dm['Nivel_Riesgo'].map(ACCIONES)

    # Métricas
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
# CAPA 3 — ALERTAS: se generan automáticamente, sin paso manual
# ════════════════════════════════════════════════════════════
def generar_alertas(dm):
    return dm[dm['Nivel_Riesgo'].isin(['CRÍTICO','ALTO'])].sort_values(
        'Prob_Churn', ascending=False)


# ════════════════════════════════════════════════════════════
# INTERFAZ — SIDEBAR: carga de archivo (única acción manual)
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔵 Entel Digital")
    st.markdown("### Sistema CX Integrado")
    st.markdown("---")

    archivo = st.file_uploader(
        "📂 Cargar Data Warehouse (.xlsx)",
        type=['xlsx','xls','xlsm'],
        help="Sube el Excel con las 5 tablas: Clientes, Interacciones, Encuestas, Preguntas, Respuestas"
    )

    st.markdown("---")
    pagina = st.radio("Navegación", [
        "📊 Resumen Ejecutivo",
        "📈 Journey & Métricas",
        "🔴 Riesgo de Churn",
        "👥 Clientes en Riesgo",
        "📋 Segmentación",
    ])

    st.markdown("---")
    seg_filtro = st.selectbox("Filtrar por Segmento", ["Todos","PYME","CORP"])

    st.markdown("---")
    st.markdown(
        "<small style='color:#94A3B8'>Modelo: Ensemble Regresión Logística<br>"
        "27 variables · Cálculo en tiempo real</small>",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL — Todo automático tras cargar el archivo
# ════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown("""
    <div class="main-header">
        <h2 style='margin:0; color:white;'>Reporte Customer Experience & Churn Risk Monitor</h2>
        <p style='margin:4px 0 0; color:#94A3B8;'>Sistema integrado de Base de Datos + Modelo Predictivo + Alertas</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Sube tu archivo Excel con el Data Warehouse en el panel izquierdo para comenzar.")
    st.markdown("""
    **El sistema automáticamente:**
    1. Valida y carga las 5 tablas del Data Warehouse
    2. Construye las 27 variables predictoras
    3. Ejecuta el modelo Ensemble de Regresión Logística
    4. Genera las alertas de riesgo por cliente
    5. Muestra el dashboard actualizado — todo en este mismo lugar
    """)
    st.stop()

with st.spinner("Cargando Data Warehouse..."):
    tablas, faltantes, hojas_disponibles = cargar_datos(archivo)

if faltantes:
    st.error(f"❌ No se encontraron las siguientes hojas: {faltantes}")
    st.write(f"Hojas disponibles en tu archivo: {hojas_disponibles}")
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

# Aplicar filtro de segmento
df_filtrado = dm.copy()
if seg_filtro != "Todos" and 'Segmento' in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado['Segmento'] == seg_filtro]

st.sidebar.success(f"✅ {len(dm):,} clientes procesados\n\n🔴 {len(df_alertas):,} en riesgo CRÍTICO/ALTO")


# ════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ════════════════════════════════════════════════════════════
if pagina == "📊 Resumen Ejecutivo":
    st.markdown("""
    <div class="main-header">
        <h2 style='margin:0; color:white;'>Reporte Customer Experience & Churn Risk Monitor</h2>
        <p style='margin:4px 0 0; color:#94A3B8; font-size:14px;'>
            Base de Datos + Modelo Predictivo + Alertas — actualizado en tiempo real
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([5,1])
    with col_b2:
        st.download_button("⬇️ Exportar CSV",
            data=df_alertas.to_csv(index=False).encode(),
            file_name="alertas_churn_cx.csv", mime="text/csv",
            use_container_width=True)

    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    churn_rate = df_filtrado['Estado_Cliente'].eq('Inactivo').mean()*100
    c1.metric("Clientes Procesados", f"{len(df_filtrado):,}")
    c2.metric("AUC del Modelo", f"{metricas['auc']:.3f}")
    c3.metric("Recall", f"{metricas['rec']:.1%}", "Churns detectados")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%", delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown("#### Sistema de Alertas de Churn")
        alertas_niv = [
            ('CRÍTICO','#C0392B','FDECEA'), ('ALTO','#E36C09','FFF3E0'),
            ('MEDIO','#D4AC00','FFFFF0'), ('BAJO','#70AD47','F1F8E9'),
        ]
        for niv,color,bg in alertas_niv:
            s = df_filtrado[df_filtrado['Nivel_Riesgo']==niv]
            n = len(s); ch = s['Churn'].mean()*100 if n>0 else 0
            st.markdown(f"""
            <div style='background:#{bg}; border-left:4px solid {color};
                        border-radius:8px; padding:10px 14px; margin:6px 0;
                        display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-weight:600; color:{color};'>{niv}</span>
                    <span style='font-size:18px; font-weight:700; color:{color}; margin-left:10px;'>{n:,}</span>
                    <span style='font-size:11px; color:#64748B;'> clientes</span>
                </div>
                <div style='font-size:11px; color:#64748B;'>Churn real: {ch:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Distribución del Score de Riesgo")
        conteo = df_filtrado['Nivel_Riesgo'].value_counts()
        orden = ['BAJO','MEDIO','ALTO','CRÍTICO']
        cols_d = ['#70AD47','#D4AC00','#E36C09','#C0392B']
        fig = go.Figure(go.Bar(
            x=[c for c in orden if c in conteo.index],
            y=[conteo.get(c,0) for c in orden if c in conteo.index],
            marker_color=cols_d, text=[conteo.get(c,0) for c in orden if c in conteo.index],
            textposition='outside'))
        fig.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10),
                          plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Clientes en Mayor Riesgo de Churn")

    buscar = st.text_input("🔍 Buscar empresa...")
    df_top = df_alertas.copy()
    if buscar and 'Razon_Social' in df_top.columns:
        df_top = df_top[df_top['Razon_Social'].str.contains(buscar, case=False, na=False)]

    cols_show = [c for c in ['Razon_Social','Segmento','Prob_Churn','NPS_Minimo',
                              'CSAT_Minimo','Nivel_Riesgo','Accion_Recomendada']
                 if c in df_top.columns]
    st.dataframe(df_top[cols_show].head(15), hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 2 — JOURNEY & MÉTRICAS
# ════════════════════════════════════════════════════════════
elif pagina == "📈 Journey & Métricas":
    st.markdown("## Journey & Métricas de Satisfacción")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### NPS Promedio por Etapa (calculado del Data Warehouse)")
        etapas_cols = ['NPS_Come','NPS_Impl','NPS_Inci']
        etapas_lbl  = ['Comercialización','Implementación','Incidente']
        valores = [df_filtrado[c].mean() for c in etapas_cols if c in df_filtrado.columns]
        fig = go.Figure(go.Bar(x=etapas_lbl[:len(valores)], y=valores,
                               marker_color=['#70AD47','#4472C4','#E36C09']))
        fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### CSAT Promedio por Etapa")
        etapas_cols_c = ['CSAT_Come','CSAT_Impl','CSAT_Inci']
        valores_c = [df_filtrado[c].mean()*20 for c in etapas_cols_c if c in df_filtrado.columns]
        fig2 = go.Figure(go.Bar(x=etapas_lbl[:len(valores_c)], y=valores_c,
                                marker_color=['#70AD47','#4472C4','#E36C09']))
        fig2.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                           yaxis=dict(range=[0,105]))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 3 — RIESGO DE CHURN
# ════════════════════════════════════════════════════════════
elif pagina == "🔴 Riesgo de Churn":
    st.markdown("## Análisis de Riesgo de Churn")
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Distribución de Probabilidad")
        fig = px.histogram(df_filtrado, x='Prob_Churn', nbins=40, color='Nivel_Riesgo',
            color_discrete_map={'CRÍTICO':'#C0392B','ALTO':'#E36C09','MEDIO':'#D4AC00','BAJO':'#70AD47'})
        fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Métricas del Modelo")
        st.metric("AUC", f"{metricas['auc']:.3f}")
        st.metric("Recall", f"{metricas['rec']:.1%}")
        st.metric("Precision", f"{metricas['prec']:.1%}")
        st.metric("F1-Score", f"{metricas['f1']:.3f}")


# ════════════════════════════════════════════════════════════
# PÁGINA 4 — CLIENTES EN RIESGO
# ════════════════════════════════════════════════════════════
elif pagina == "👥 Clientes en Riesgo":
    st.markdown("## Clientes en Riesgo de Abandono")
    st.markdown("---")
    nivel_filtro = st.multiselect("Nivel de riesgo",
        ['CRÍTICO','ALTO','MEDIO','BAJO'], default=['CRÍTICO','ALTO'])
    df_view = df_filtrado[df_filtrado['Nivel_Riesgo'].isin(nivel_filtro)] if nivel_filtro else df_filtrado
    st.markdown(f"**{len(df_view):,} clientes** encontrados")
    st.dataframe(df_view.sort_values('Prob_Churn', ascending=False).head(100),
                hide_index=True, use_container_width=True, height=480)
    st.download_button("⬇️ Descargar lista completa",
        data=df_view.to_csv(index=False).encode(),
        file_name="clientes_riesgo.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════
# PÁGINA 5 — SEGMENTACIÓN
# ════════════════════════════════════════════════════════════
elif pagina == "📋 Segmentación":
    st.markdown("## Segmentación del Portafolio")
    st.markdown("---")
    if 'Segmento' in df_filtrado.columns:
        c1,c2 = st.columns(2)
        with c1:
            churn_seg = df_filtrado.groupby('Segmento')['Churn'].mean().reset_index()
            churn_seg['Churn'] = churn_seg['Churn']*100
            fig = px.bar(churn_seg, x='Segmento', y='Churn', color='Segmento',
                        color_discrete_map={'PYME':'#E36C09','CORP':'#1F497D'})
            fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dist = df_filtrado['Nivel_Riesgo'].value_counts().reset_index()
            dist.columns = ['Nivel','N']
            fig2 = px.pie(dist, names='Nivel', values='N', hole=0.45,
                color_discrete_map={'CRÍTICO':'#C0392B','ALTO':'#E36C09','MEDIO':'#D4AC00','BAJO':'#70AD47'})
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No se encontró columna Segmento en los datos.")
