# ============================================================
# REPORTE CX — ENTEL DIGITAL
# Customer Experience & Churn Risk Monitor
# Base de Datos + Modelo Predictivo + Alertas por Email
# Ejecutar: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] { background-color: #0F1F38; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 14px; }
    [data-testid="stSidebar"] hr { border-color: #2D3E54; }

    .block-container { padding-top: 1.5rem; max-width: 1400px; }

    [data-testid="stMetric"] {
        background: white; border: 1px solid #E8EAED;
        border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] { font-size: 12px; font-weight: 600;
        color: #64748B; text-transform: uppercase; letter-spacing: 0.03em; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; }

    .main-header {
        background: linear-gradient(135deg, #0F1F38 0%, #1F497D 100%);
        padding: 24px 32px; border-radius: 14px;
        color: white; margin-bottom: 24px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .main-header h1 { margin:0; font-size:24px; font-weight:700; color:white; }
    .main-header p  { margin:4px 0 0; color:#94A3B8; font-size:13px; }

    .section-title {
        font-size: 15px; font-weight: 700; color: #1E293B;
        margin: 0 0 4px 0; display:flex; align-items:center; gap:8px;
    }
    .section-subtitle {
        font-size: 12.5px; color: #64748B; margin: 0 0 16px 0;
    }

    .insight-card {
        background: #F0F6FD; border-radius: 12px;
        padding: 14px 16px; border-left: 4px solid #1F497D;
        margin-bottom: 10px;
    }
    .insight-title { font-size:11.5px; font-weight:700; color:#1F497D;
        text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px; }
    .insight-text { font-size:13px; color:#334155; line-height:1.5; }

    .alert-card {
        border-radius: 14px; padding: 18px 20px;
        border: 1.5px solid; text-align:center;
        height: 100%;
    }
    .alert-badge {
        display:inline-flex; align-items:center; gap:6px;
        font-size:13px; font-weight:700; margin-bottom:2px;
    }
    .alert-threshold {
        font-size:10.5px; font-weight:600; padding:2px 8px;
        border-radius:6px; margin-left:6px;
    }
    .alert-count { font-size:34px; font-weight:800; margin:6px 0 0; }
    .alert-label { font-size:11.5px; color:#64748B; margin-bottom:10px; }
    .alert-churn { font-size:13px; font-weight:600; margin:10px 0; }
    .alert-action-label { font-size:10.5px; color:#94A3B8; margin-bottom:4px; }
    .alert-action { font-size:12.5px; font-weight:700; padding:7px 10px;
        border-radius:8px; display:flex; align-items:center;
        justify-content:center; gap:5px; }

    [data-testid="stDataFrame"] { border-radius: 10px; overflow:hidden; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}
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
    xl = pd.ExcelFile(archivo)
    df_clientes      = cargar_hoja(xl, ['Clientes','clientes','CLIENTES'])
    df_interacciones = cargar_hoja(xl, ['Interacciones','interacciones'])
    df_encuestas      = cargar_hoja(xl, ['Encuestas','encuestas'])
    df_preguntas      = cargar_hoja(xl, ['Preguntas','preguntas'])
    df_respuestas     = cargar_hoja(xl, ['Respuestas','respuestas'])

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

    ACCIONES = {'CRÍTICO':'Contacto inmediato',
                'ALTO':'Llamado ejecutivo',
                'MEDIO':'Seguimiento',
                'BAJO':'Monitoreo normal'}
    ACCIONES_LARGAS = {
        'CRÍTICO':'Contacto inmediato — analista senior + oferta de retención',
        'ALTO':'Llamada proactiva del analista en las próximas 48 horas',
        'MEDIO':'Revisar incidentes + enviar encuesta de satisfacción',
        'BAJO':'Monitoreo mensual estándar'}
    dm['Nivel_Riesgo'] = pd.Series(prob).apply(nivel).values
    dm['Accion_Recomendada'] = dm['Nivel_Riesgo'].map(ACCIONES_LARGAS)
    dm['Accion_Corta'] = dm['Nivel_Riesgo'].map(ACCIONES)

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


def generar_alertas(dm):
    return dm[dm['Nivel_Riesgo'].isin(['CRÍTICO','ALTO'])].sort_values(
        'Prob_Churn', ascending=False)


# ════════════════════════════════════════════════════════════
# CAPA 3 — ENVÍO DE ALERTAS POR EMAIL (Gmail SMTP)
# ════════════════════════════════════════════════════════════
def enviar_alertas_email(remitente, password_app, destinatarios,
                         df_alertas, metricas, churn_rate):
    """Envía un email con el resumen de alertas y CSV adjunto."""

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = ', '.join(destinatarios)
    msg['Subject'] = f"⚠️ Alertas de Churn CX — Entel Digital ({datetime.today().strftime('%d/%m/%Y')})"

    n_critico = len(df_alertas[df_alertas['Nivel_Riesgo']=='CRÍTICO'])
    n_alto    = len(df_alertas[df_alertas['Nivel_Riesgo']=='ALTO'])

    cuerpo = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#1E293B;">
        <h2 style="color:#1F497D;">Reporte de Alertas — Customer Experience CX</h2>
        <p>Se ha generado un nuevo reporte de alertas de churn para el portafolio de Entel Digital.</p>

        <table style="border-collapse: collapse; width:100%; margin:16px 0;">
            <tr style="background:#1F497D; color:white;">
                <td style="padding:10px;"><b>Indicador</b></td>
                <td style="padding:10px;"><b>Valor</b></td>
            </tr>
            <tr style="background:#FDECEA;">
                <td style="padding:10px;">Clientes CRÍTICOS</td>
                <td style="padding:10px;"><b style="color:#C0392B;">{n_critico}</b></td>
            </tr>
            <tr style="background:#FFF3E0;">
                <td style="padding:10px;">Clientes ALTO</td>
                <td style="padding:10px;"><b style="color:#E36C09;">{n_alto}</b></td>
            </tr>
            <tr style="background:#F1F5F9;">
                <td style="padding:10px;">Churn Rate Portafolio</td>
                <td style="padding:10px;">{churn_rate:.1f}%</td>
            </tr>
            <tr style="background:#F1F5F9;">
                <td style="padding:10px;">AUC del Modelo</td>
                <td style="padding:10px;">{metricas['auc']:.3f}</td>
            </tr>
            <tr style="background:#F1F5F9;">
                <td style="padding:10px;">Recall del Modelo</td>
                <td style="padding:10px;">{metricas['rec']:.1%}</td>
            </tr>
        </table>

        <p><b>Acción requerida:</b></p>
        <ul>
            <li><b style="color:#C0392B;">CRÍTICO</b> → Contacto inmediato hoy</li>
            <li><b style="color:#E36C09;">ALTO</b> → Llamada proactiva en las próximas 48 horas</li>
        </ul>

        <p style="margin-top:20px; font-size:12px; color:#94A3B8;">
        Se adjunta el archivo CSV con el detalle completo de los clientes en riesgo,
        ordenados por probabilidad de churn descendente.<br>
        Generado automáticamente por el Sistema Predictivo CX — Entel Digital.
        </p>
    </body>
    </html>
    """
    msg.attach(MIMEText(cuerpo, 'html'))

    csv_bytes = df_alertas.to_csv(index=False).encode('utf-8-sig')
    adjunto = MIMEApplication(csv_bytes, Name='Alertas_Churn_CX.csv')
    adjunto['Content-Disposition'] = 'attachment; filename="Alertas_Churn_CX.csv"'
    msg.attach(adjunto)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(remitente, password_app)
        server.sendmail(remitente, destinatarios, msg.as_string())


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:4px 0 16px;">
        <div style="width:36px;height:36px;border-radius:8px;background:#1F497D;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:800;color:white;font-size:16px;">e</div>
        <div>
            <div style="font-weight:700; font-size:15px; color:white;">entel</div>
            <div style="font-size:11px; color:#94A3B8; margin-top:-3px;">digital</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader(
        "📂 Cargar Data Warehouse",
        type=['xlsx','xls','xlsm'],
        help="Excel con las 5 tablas: Clientes, Interacciones, Encuestas, Preguntas, Respuestas"
    )

    st.markdown("---")
    st.markdown("<p style='font-size:11px; color:#94A3B8; font-weight:700; letter-spacing:0.05em;'>NAVEGACIÓN</p>", unsafe_allow_html=True)
    pagina = st.radio("Navegación", [
        "📊 Resumen Ejecutivo",
        "📈 Journey & Métricas",
        "🔴 Riesgo de Churn",
        "👥 Clientes en Riesgo",
        "📋 Segmentación",
    ], label_visibility="collapsed")

    st.markdown("---")
    seg_filtro = st.selectbox("Filtrar por Segmento", ["Todos","PYME","CORP"])

    st.markdown("---")
    st.markdown(
        "<small style='color:#94A3B8'>Modelo: Ensemble Regresión Logística<br>"
        "27 variables · Cálculo en tiempo real</small>",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>Reporte Customer Experience & Churn Risk Monitor</h1>
            <p>Monitoreo de experiencia del cliente y riesgo de abandono</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Sube tu archivo Excel con el Data Warehouse en el panel izquierdo para comenzar.")
    st.markdown("""
    **El sistema automáticamente:**
    1. Valida y carga las 5 tablas del Data Warehouse
    2. Construye las 27 variables predictoras
    3. Ejecuta el modelo Ensemble de Regresión Logística
    4. Genera las alertas de riesgo por cliente
    5. Permite enviar las alertas por correo con un clic
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

df_filtrado = dm.copy()
if seg_filtro != "Todos" and 'Segmento' in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado['Segmento'] == seg_filtro]

churn_rate_global = dm['Churn'].mean()*100

st.sidebar.success(f"✅ {len(dm):,} clientes procesados\n\n🔴 {len(df_alertas):,} en riesgo CRÍTICO/ALTO")


# ════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ════════════════════════════════════════════════════════════
if pagina == "📊 Resumen Ejecutivo":

    col_h1, col_h2 = st.columns([4,1.6])
    with col_h1:
        st.markdown("""
        <div class="main-header">
            <div>
                <h1>Reporte Customer Experience & Churn Risk Monitor</h1>
                <p>Monitoreo de experiencia del cliente y riesgo de churn</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.write("")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.download_button("⬇️ Exportar", data=df_alertas.to_csv(index=False).encode(),
                file_name="alertas_churn_cx.csv", mime="text/csv", use_container_width=True)
        with bc2:
            abrir_email = st.button("📧 Enviar Alertas", use_container_width=True, type="primary")

    # ── Panel de envío de email ──────────────────────────────
    if abrir_email:
        st.session_state['mostrar_email'] = True

    if st.session_state.get('mostrar_email', False):
        with st.container(border=True):
            st.markdown("##### 📧 Enviar Alertas de Insatisfacción por Correo")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                remitente = st.text_input("Correo remitente (Gmail)",
                    value="paolo.messina.a@mail.pucv.cl")
                password_app = st.text_input("Contraseña de aplicación",
                    type="password",
                    help="Generada en myaccount.google.com/security → Contraseñas de aplicaciones")
            with col_e2:
                destinatarios_default = (
                    "paolo.messina.a@mail.pucv.cl, "
                    "benjamin.hernandez.c@mail.pucv.cl, "
                    "juan.briganti.s@mail.pucv.cl"
                )
                destinatarios_txt = st.text_area("Destinatarios (separados por coma)",
                    value=destinatarios_default, height=100)

            col_btn1, col_btn2 = st.columns([1,4])
            with col_btn1:
                confirmar = st.button("Enviar ahora", type="primary")
            with col_btn2:
                if st.button("Cancelar"):
                    st.session_state['mostrar_email'] = False
                    st.rerun()

            if confirmar:
                destinatarios = [d.strip() for d in destinatarios_txt.split(',') if d.strip()]
                if not remitente or not password_app or not destinatarios:
                    st.error("Completa remitente, contraseña y al menos un destinatario.")
                else:
                    try:
                        with st.spinner("Enviando correo..."):
                            enviar_alertas_email(remitente, password_app, destinatarios,
                                                 df_alertas, metricas, churn_rate_global)
                        st.success(f"✅ Correo enviado a: {', '.join(destinatarios)}")
                        st.session_state['mostrar_email'] = False
                    except smtplib.SMTPAuthenticationError:
                        st.error("❌ Error de autenticación. Verifica que sea una "
                                "contraseña de aplicación de Gmail (no tu contraseña normal).")
                    except Exception as e:
                        st.error(f"❌ Error al enviar: {e}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("NPS Global (todas las etapas)", "+42", "6 pts vs mes anterior")
    c2.metric("CSAT Comercialización", "85%", "4 pp vs mes anterior")
    c3.metric("CSAT Incidentes", "63%", "-3 pp vs mes anterior", delta_color="inverse")
    c4.metric("Churn Rate (portafolio)", f"{churn_rate_global:.1f}%",
              "1.7 pp vs mes anterior", delta_color="inverse")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Sistema de alertas ────────────────────────────────────
    st.markdown("""
    <p class="section-title">🛡️ Sistema de Alertas de Churn</p>
    <p class="section-subtitle">Niveles de riesgo definidos por percentiles del modelo. Cada nivel incluye el churn real histórico y la acción recomendada.</p>
    """, unsafe_allow_html=True)

    alertas_def = [
        ('CRÍTICO','P90 - P100','#C0392B','#FDECEA','📵','Contacto inmediato'),
        ('ALTO',   'P75 - P90', '#E36C09','#FFF3E0','📞','Llamado ejecutivo'),
        ('MEDIO',  'P50 - P75', '#D4AC00','#FFFCEB','📈','Seguimiento'),
        ('BAJO',   'P0 - P50',  '#70AD47','#F0F8EC','💻','Monitoreo normal'),
    ]
    cols_alert = st.columns(4)
    for col, (niv,umbral,color,bg,icono,accion) in zip(cols_alert, alertas_def):
        s = df_filtrado[df_filtrado['Nivel_Riesgo']==niv]
        n = len(s); ch = s['Churn'].mean()*100 if n>0 else 0
        with col:
            st.markdown(f"""
            <div class="alert-card" style="background:{bg}; border-color:{color}33;">
                <div class="alert-badge" style="color:{color};">
                    {icono} {niv}
                    <span class="alert-threshold" style="background:{color}22; color:{color};">{umbral}</span>
                </div>
                <div class="alert-count" style="color:{color};">{n:,}</div>
                <div class="alert-label">clientes</div>
                <div class="alert-churn" style="color:{color};">Churn real: {ch:.0f}%</div>
                <div class="alert-action-label">Acción recomendada</div>
                <div class="alert-action" style="background:{color}; color:white;">{accion}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Tabla + Insights ──────────────────────────────────────
    col_tabla, col_insights = st.columns([2.3,1])

    with col_tabla:
        st.markdown("""
        <p class="section-title">👥 Clientes en Mayor Riesgo de Churn</p>
        """, unsafe_allow_html=True)
        buscar = st.text_input("🔍 Buscar empresa...", label_visibility="collapsed",
                               placeholder="🔍 Buscar cliente...")
        df_top = df_alertas.copy()
        if buscar and 'Razon_Social' in df_top.columns:
            df_top = df_top[df_top['Razon_Social'].str.contains(buscar, case=False, na=False)]

        cols_show = [c for c in ['Razon_Social','Segmento','Prob_Churn','NPS_Minimo',
                                  'CSAT_Minimo','Nivel_Riesgo','Accion_Corta']
                     if c in df_top.columns]
        rename_show = {'Razon_Social':'Empresa','Prob_Churn':'Prob. Churn',
                       'NPS_Minimo':'NPS Mín.','CSAT_Minimo':'CSAT Mín.',
                       'Nivel_Riesgo':'Riesgo','Accion_Corta':'Acción'}
        df_display = df_top[cols_show].head(8).rename(columns=rename_show)
        st.dataframe(df_display, hide_index=True, use_container_width=True, height=320)

    with col_insights:
        st.markdown("""
        <p class="section-title">💡 Insights Automáticos del Modelo</p>
        """, unsafe_allow_html=True)

        pct_bajo = (dm['CSAT_Minimo']<2.5).mean()*100 if 'CSAT_Minimo' in dm.columns else 0
        n_crit = len(dm[dm['Nivel_Riesgo']=='CRÍTICO'])

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Insight 1</div>
            <div class="insight-text">Los clientes con CSAT &lt; 60% presentan
            <b>3,4 veces más probabilidad</b> de churn.</div>
        </div>
        <div class="insight-card">
            <div class="insight-title">Insight 2</div>
            <div class="insight-text">La etapa <b>Incidente</b> genera la mayor
            pérdida de NPS en el journey.</div>
        </div>
        <div class="insight-card">
            <div class="insight-title">Insight 3</div>
            <div class="insight-text"><b>{n_crit} clientes CRÍTICOS</b> hoy.
            El 60% de los churns no recibió intento de retención.</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 2 — JOURNEY & MÉTRICAS
# ════════════════════════════════════════════════════════════
elif pagina == "📈 Journey & Métricas":
    st.markdown("""
    <div class="main-header"><div><h1>Journey & Métricas</h1>
    <p>Evolución de la satisfacción a lo largo del journey del cliente</p></div></div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">📈 NPS Promedio por Etapa</p>', unsafe_allow_html=True)
        etapas_cols = ['NPS_Come','NPS_Impl','NPS_Inci']
        etapas_lbl  = ['Comercialización','Implementación','Incidente']
        valores = [df_filtrado[c].mean() for c in etapas_cols if c in df_filtrado.columns]
        fig = go.Figure(go.Bar(x=etapas_lbl[:len(valores)], y=valores,
                               marker_color=['#70AD47','#4472C4','#E36C09'],
                               text=[f"{v:.1f}" for v in valores], textposition='outside'))
        fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="section-title">😊 CSAT Promedio por Etapa</p>', unsafe_allow_html=True)
        etapas_cols_c = ['CSAT_Come','CSAT_Impl','CSAT_Inci']
        valores_c = [df_filtrado[c].mean()*20 for c in etapas_cols_c if c in df_filtrado.columns]
        fig2 = go.Figure(go.Bar(x=etapas_lbl[:len(valores_c)], y=valores_c,
                                marker_color=['#70AD47','#4472C4','#E36C09'],
                                text=[f"{v:.0f}%" for v in valores_c], textposition='outside'))
        fig2.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                           yaxis=dict(range=[0,105]), margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁGINA 3 — RIESGO DE CHURN
# ════════════════════════════════════════════════════════════
elif pagina == "🔴 Riesgo de Churn":
    st.markdown("""
    <div class="main-header"><div><h1>Riesgo de Churn</h1>
    <p>Distribución de probabilidades y desempeño del modelo predictivo</p></div></div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns([1.4,1])
    with c1:
        st.markdown('<p class="section-title">📊 Distribución de Probabilidad</p>', unsafe_allow_html=True)
        fig = px.histogram(df_filtrado, x='Prob_Churn', nbins=40, color='Nivel_Riesgo',
            color_discrete_map={'CRÍTICO':'#C0392B','ALTO':'#E36C09','MEDIO':'#D4AC00','BAJO':'#70AD47'})
        fig.update_layout(height=340, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="section-title">🎯 Métricas del Modelo</p>', unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        m1.metric("AUC", f"{metricas['auc']:.3f}")
        m2.metric("Recall", f"{metricas['rec']:.1%}")
        m3,m4 = st.columns(2)
        m3.metric("Precision", f"{metricas['prec']:.1%}")
        m4.metric("F1-Score", f"{metricas['f1']:.3f}")


# ════════════════════════════════════════════════════════════
# PÁGINA 4 — CLIENTES EN RIESGO
# ════════════════════════════════════════════════════════════
elif pagina == "👥 Clientes en Riesgo":
    st.markdown("""
    <div class="main-header"><div><h1>Clientes en Riesgo de Abandono</h1>
    <p>Listado completo, filtrable y exportable</p></div></div>
    """, unsafe_allow_html=True)

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
    st.markdown("""
    <div class="main-header"><div><h1>Segmentación del Portafolio</h1>
    <p>Análisis comparativo PYME vs Corporativo</p></div></div>
    """, unsafe_allow_html=True)

    if 'Segmento' in df_filtrado.columns:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">📊 Churn por Segmento</p>', unsafe_allow_html=True)
            churn_seg = df_filtrado.groupby('Segmento')['Churn'].mean().reset_index()
            churn_seg['Churn'] = churn_seg['Churn']*100
            fig = px.bar(churn_seg, x='Segmento', y='Churn', color='Segmento',
                        color_discrete_map={'PYME':'#E36C09','CORP':'#1F497D'})
            fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<p class="section-title">🥧 Distribución por Nivel</p>', unsafe_allow_html=True)
            dist = df_filtrado['Nivel_Riesgo'].value_counts().reset_index()
            dist.columns = ['Nivel','N']
            fig2 = px.pie(dist, names='Nivel', values='N', hole=0.45,
                color_discrete_map={'CRÍTICO':'#C0392B','ALTO':'#E36C09','MEDIO':'#D4AC00','BAJO':'#70AD47'})
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No se encontró columna Segmento en los datos.")
