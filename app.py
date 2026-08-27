import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import datetime

# ---------------------------------------------------------
# Page Configurations
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aequitas Terminal // Algorithmic Fairness",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Theme Toggle and Session State Initialization
# ---------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ---------------------------------------------------------
# CSS Design System (Sleek Glassmorphic Stock Terminal Theme)
# ---------------------------------------------------------
# Palette definition
if IS_DARK:
    bg = "#070c16"          # Deep space navy
    bg_subtle = "#0a1120"   # Muted card backdrop
    card = "rgba(13, 22, 41, 0.75)"
    border = "rgba(255, 255, 255, 0.08)"
    border_hover = "rgba(59, 130, 246, 0.45)"
    text = "#f8fafc"        # Slate 50
    text_muted = "#64748b"  # Slate 500
    text_dim = "#334155"    # Slate 700
    accent = "#3b82f6"      # Indigo blue
    green = "#10b981"       # Emerald green
    green_muted = "rgba(16,185,129,0.12)"
    red = "#f43f5e"         # Rose red
    red_muted = "rgba(244,63,94,0.12)"
    amber = "#f59e0b"       # Amber
    amber_muted = "rgba(245,158,11,0.12)"
    glow = "rgba(59, 130, 246, 0.18)"
else:
    bg = "#f8fafc"          # Light slate
    bg_subtle = "#f1f5f9"
    card = "rgba(255, 255, 255, 0.85)"
    border = "rgba(0, 0, 0, 0.08)"
    border_hover = "rgba(37, 99, 235, 0.4)"
    text = "#0f172a"        # Dark slate
    text_muted = "#64748b"
    text_dim = "#cbd5e1"
    accent = "#2563eb"
    green = "#059669"
    green_muted = "rgba(5,150,105,0.08)"
    red = "#e11d48"
    red_muted = "rgba(225,29,72,0.08)"
    amber = "#d97706"
    amber_muted = "rgba(217,119,6,0.08)"
    glow = "rgba(37, 99, 235, 0.08)"

css = f"""
<style>
    /* Hide default Streamlit styles */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    /* Global backgrounds & fonts */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}
    
    section[data-testid="stSidebar"] {{
        background-color: {bg_subtle} !important;
        border-right: 1px solid {border} !important;
    }}
    
    /* Sleek Stock Terminal Card */
    .metric-card {{
        background: {card};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .metric-card:hover {{
        border-color: {border_hover};
        box-shadow: 0 8px 32px 0 {glow};
        transform: translateY(-2px);
    }}
    .metric-label {{
        font-size: 0.72rem;
        color: {text_muted};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {text};
        letter-spacing: -0.04em;
        line-height: 1.1;
    }}
    
    /* Interactive Badges with Pulse Effects */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .badge-green {{ color: {green}; background: {green_muted}; border: 1px solid rgba(16,185,129,0.2); }}
    .badge-red {{ color: {red}; background: {red_muted}; border: 1px solid rgba(244,63,94,0.2); }}
    .badge-amber {{ color: {amber}; background: {amber_muted}; border: 1px solid rgba(245,158,11,0.2); }}
    .badge-blue {{ color: {accent}; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59,130,246,0.2); }}
    
    .pulse-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }}
    .pulse-green {{ background: {green}; box-shadow: 0 0 8px {green}; }}
    .pulse-red {{ background: {red}; box-shadow: 0 0 8px {red}; }}
    
    /* Sleek Chart Wrapper */
    .chart-wrap {{
        background: {card};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.2);
        transition: border-color 0.3s ease;
        margin-bottom: 1.25rem;
    }}
    .chart-wrap:hover {{
        border-color: rgba(59, 130, 246, 0.2);
    }}
    .chart-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.25rem;
    }}
    .chart-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.01em;
    }}
    .chart-subtitle {{
        font-size: 0.72rem;
        color: {text_muted};
        margin-top: 0.15rem;
    }}
    
    /* Scrollable Console Output styling */
    .console-log {{
        background: #030712 !important;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #38bdf8;
        max-height: 180px;
        overflow-y: auto;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.8);
        margin-top: 0.5rem;
    }}
    .console-line {{
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }}
    
    /* Corporate Tabs design */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {text_muted} !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text} !important;
        background: {bg_subtle} !important;
        border-color: {border} !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 6px !important;
        background: rgba(0,0,0,0.15) !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        padding: 4px;
    }}
    
    /* Corporate Data Table */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.78rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 1rem;
        color: {text_muted};
        font-weight: 700;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 1px solid {border};
        background: rgba(0, 0, 0, 0.1);
    }}
    .data-table td {{
        padding: 0.75rem 1rem;
        color: {text};
        border-bottom: 1px solid {border};
    }}
    .data-table tr:hover td {{
        background: rgba(255, 255, 255, 0.015);
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Brand Header styling */
    .brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.5rem;
    }}
    .brand-symbol {{
        font-size: 1.65rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .brand-name {{
        font-size: 1.3rem;
        font-weight: 800;
        color: {text};
        letter-spacing: -0.03em;
    }}
    .brand-tag {{
        font-size: 0.65rem;
        color: {accent};
        border: 1px solid {accent};
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 1.25rem !important;
    }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Generation and Model Training Logic (Cached)
# ---------------------------------------------------------
@st.cache_resource
def generate_and_train_models(bias_strength=0.60):
    np.random.seed(42)
    N = 800
    
    gender = np.random.binomial(1, 0.5, N)
    
    credit_score = np.random.normal(650, 80, N)
    credit_score = np.clip(credit_score, 300, 850)
    
    income = np.random.normal(75000, 25000, N)
    income = np.clip(income, 20000, 250000)
    
    dti = np.random.normal(35, 12, N)
    dti = np.clip(dti, 5, 80)
    
    savings = np.random.normal(25000, 15000, N)
    savings = np.clip(savings, 0, 150000)
    
    df = pd.DataFrame({
        'credit_score': credit_score,
        'income': income,
        'dti': dti,
        'savings': savings,
        'gender': gender
    })
    
    scaler_temp = StandardScaler()
    scaled_feats = scaler_temp.fit_transform(df[['credit_score', 'income', 'dti', 'savings']])
    
    weights_true = np.array([0.45, 0.35, -0.30, 0.20])
    raw_score = np.dot(scaled_feats, weights_true) + np.random.normal(0, 0.1, N)
    
    true_threshold = np.percentile(raw_score, 55)
    y_true = (raw_score >= true_threshold).astype(int)
    df['y_true'] = y_true
    
    y_historic = y_true.copy()
    for i in range(N):
        if gender[i] == 0 and y_true[i] == 1:
            if np.random.rand() < bias_strength:
                y_historic[i] = 0
                
    df['y_historic'] = y_historic
    
    features_biased = ['credit_score', 'income', 'dti', 'savings', 'gender']
    features_fair = ['credit_score', 'income', 'dti', 'savings']
    
    scaler_biased = StandardScaler()
    X_train_biased = scaler_biased.fit_transform(df[features_biased])
    
    scaler_fair = StandardScaler()
    X_train_fair = scaler_fair.fit_transform(df[features_fair])
    
    # Model 1: Biased Model (Direct Fit)
    model_biased = LogisticRegression(C=1.0, random_state=42)
    model_biased.fit(X_train_biased, df['y_historic'])
    
    # Model 2: Fair Model (Feature Masking)
    model_fair = LogisticRegression(C=1.0, random_state=42)
    model_fair.fit(X_train_fair, df['y_historic'])
    
    # Model 3: Mitigated Model (Kamiran-Calders Reweighing)
    n_total = N
    n_male = np.sum(gender == 1)
    n_female = np.sum(gender == 0)
    n_approved = np.sum(y_historic == 1)
    n_denied = np.sum(y_historic == 0)
    
    n_male_approved = np.sum((gender == 1) & (y_historic == 1))
    n_male_denied = np.sum((gender == 1) & (y_historic == 0))
    n_female_approved = np.sum((gender == 0) & (y_historic == 1))
    n_female_denied = np.sum((gender == 0) & (y_historic == 0))
    
    w_m_app = (n_male * n_approved) / (n_total * n_male_approved) if n_male_approved > 0 else 1.0
    w_m_den = (n_male * n_denied) / (n_total * n_male_denied) if n_male_denied > 0 else 1.0
    w_f_app = (n_female * n_approved) / (n_total * n_female_approved) if n_female_approved > 0 else 1.0
    w_f_den = (n_female * n_denied) / (n_total * n_female_denied) if n_female_denied > 0 else 1.0
    
    sample_weights = np.zeros(N)
    for i in range(N):
        if gender[i] == 1:
            sample_weights[i] = w_m_app if y_historic[i] == 1 else w_m_den
        else:
            sample_weights[i] = w_f_app if y_historic[i] == 1 else w_f_den
            
    model_reweighted = LogisticRegression(C=1.0, random_state=42)
    model_reweighted.fit(X_train_fair, df['y_historic'], sample_weight=sample_weights)
    
    return df, {
        'biased': (model_biased, scaler_biased, features_biased),
        'fair_masked': (model_fair, scaler_fair, features_fair),
        'reweighted': (model_reweighted, scaler_fair, features_fair)
    }

# ---------------------------------------------------------
# Sidebar Inputs (Applicant Profiles and Bias Parameter)
# ---------------------------------------------------------
# Header brand
head_left, head_right = st.columns([8, 2])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-symbol">⚡</span>
        <span class="brand-name">AEQUITAS TERMINAL</span>
        <span class="brand-tag">v2.0 XAI</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ LIGHT UI" if IS_DARK else "🌙 DARK UI"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.sidebar.markdown("### 📡 Terminal Configuration")
bias_slider = st.sidebar.slider(
    "Historical Bias Strength",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    format="%d%%",
    help="Historical discrimination probability injected against qualified females."
)
bias_strength = bias_slider / 100.0

# Load dataset and models
df_train, models_dict = generate_and_train_models(bias_strength)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Credit Applicant Profile")
app_name = st.sidebar.text_input("Applicant Identity", value="Elena Rostova")
app_gender = st.sidebar.radio("Demographic Profile", ["Female", "Male"], index=0)
app_gender_val = 0 if app_gender == "Female" else 1

app_credit = st.sidebar.slider("FICO Credit Score", 300, 850, 680, step=5)
app_income = st.sidebar.slider("Annual Income ($)", 20000, 250000, 62000, step=1000)
app_dti = st.sidebar.slider("Debt-to-Income (DTI)", 5, 80, 38, step=1, format="%d%%")
app_savings = st.sidebar.slider("Savings Balance ($)", 0, 150000, 12000, step=1000)

applicant_raw = pd.DataFrame({
    'credit_score': [app_credit],
    'income': [app_income],
    'dti': [app_dti],
    'savings': [app_savings],
    'gender': [app_gender_val]
})

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Active Inference Model")
model_choice = st.sidebar.selectbox(
    "Active Auditing Model",
    [
        "Biased Model (Direct Fit)",
        "Fair Model (Feature Masked)",
        "Mitigated Model (Kamiran-Calders)"
    ]
)

model_key = 'biased' if "Biased" in model_choice else ('fair_masked' if "Masked" in model_choice else 'reweighted')
active_model, active_scaler, active_features = models_dict[model_key]

# ---------------------------------------------------------
# Run Model Inference and Calculate SHAP Explanations
# ---------------------------------------------------------
# Scale applicant data
applicant_scaled = active_scaler.transform(applicant_raw[active_features])
prob_pred = active_model.predict_proba(applicant_scaled)[0, 1]
decision = "Approved" if prob_pred >= 0.50 else "Denied"

# Calculate average feature values in training data
avg_features = df_train[active_features].mean().values

# Calculate SHAP values
weights = active_model.coef_[0]
intercept = active_model.intercept_[0]
means = active_scaler.mean_
scales = active_scaler.scale_

# Effective linear weights: w_eff = coef / scale
w_eff = weights / scales
b_eff = intercept - np.sum(weights * means / scales)

x_app = applicant_raw[active_features].values[0]
x_avg = df_train[active_features].mean().values

# Contribution in log-odds space: phi = w_eff * (x_app - x_avg)
phi_log_odds = w_eff * (x_app - x_avg)
z_base = b_eff + np.sum(w_eff * x_avg)
z_pred = b_eff + np.sum(w_eff * x_app)

p_base = 1 / (1 + np.exp(-z_base))
p_pred = prob_pred

sum_phi = np.sum(phi_log_odds)
if abs(sum_phi) > 1e-9:
    phi_prob = phi_log_odds * ((p_pred - p_base) / sum_phi)
else:
    phi_prob = np.zeros_like(phi_log_odds)

# Get predictions from other models for comparison
compare_preds = {}
for k, (m, sc, feats) in models_dict.items():
    app_sc = sc.transform(applicant_raw[feats])
    compare_preds[k] = m.predict_proba(app_sc)[0, 1]

# Calculate population statistics for audits
df_train_scaled = active_scaler.transform(df_train[active_features])
preds_prob_pop = active_model.predict_proba(df_train_scaled)[:, 1]
preds_pop = (preds_prob_pop >= 0.50).astype(int)

male_indices = df_train['gender'] == 1
female_indices = df_train['gender'] == 0

selection_rate_male = np.mean(preds_pop[male_indices])
selection_rate_female = np.mean(preds_pop[female_indices])

if selection_rate_male > 0:
    air = selection_rate_female / selection_rate_male
else:
    air = 1.0 if selection_rate_female == 0 else np.inf

y_true_pop = df_train['y_true'].values
qualified_male = (y_true_pop == 1) & male_indices
qualified_female = (y_true_pop == 1) & female_indices

tpr_male = np.mean(preds_pop[qualified_male]) if np.sum(qualified_male) > 0 else 0.0
tpr_female = np.mean(preds_pop[qualified_female]) if np.sum(qualified_female) > 0 else 0.0
eog = tpr_male - tpr_female

# ---------------------------------------------------------
# Dynamic Console Logs Builder
# ---------------------------------------------------------
timestamp_now = datetime.datetime.now().strftime("%H:%M:%S")
log_lines = [
    f"<span class='console-info'>[{timestamp_now}] [SYSTEM] Aequitas Core audit engine initialized.</span>",
    f"<span class='console-info'>[{timestamp_now}] [DATA] Loaded synthetic population N=800. Bias strength injected: {bias_strength:.0%}.</span>",
    f"<span class='console-info'>[{timestamp_now}] [MODEL] Loaded model configuration: '{model_choice}' ({len(active_features)} features).</span>"
]

if air < 0.80:
    log_lines.append(f"<span class='console-error'>[{timestamp_now}] [WARN] Disparate Impact violation: AIR = {air:.2f} (EEOC 80% Rule fails).</span>")
else:
    log_lines.append(f"<span class='console-success'>[{timestamp_now}] [AUDIT] Demographic Parity holds: AIR = {air:.2f} (EEOC 80% Rule passes).</span>")
    
if abs(eog) >= 0.05:
    log_lines.append(f"<span class='console-warn'>[{timestamp_now}] [WARN] Equal Opportunity gap is {eog*100:.1f}%. Qualified female penalty detected.</span>")
else:
    log_lines.append(f"<span class='console-success'>[{timestamp_now}] [AUDIT] Equal Opportunity condition met. TPR gap is {eog*100:.1f}%.</span>")
    
log_lines.append(f"<span class='console-info'>[{timestamp_now}] [EVAL] Audited '{app_name}' ({app_gender}): Probability = {prob_pred*100:.1f}%. Decision: {decision.upper()}.</span>")

# ---------------------------------------------------------
# Dashboard KPI Grid
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    status_class = "badge-green" if decision == "Approved" else "badge-red"
    pulse_color = "pulse-green" if decision == "Approved" else "pulse-red"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Inference Decision</div>
        <div class="metric-value">{decision}</div>
        <div style="margin-top:0.5rem;">
            <span class="badge {status_class}"><span class="pulse-dot {pulse_color}"></span>{decision.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Approval Probability</div>
        <div class="metric-value">{prob_pred*100:.1f}%</div>
        <div style="margin-top:0.5rem; font-size:0.75rem; color:{text_muted}">Decision Threshold: 50.0%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    air_status_class = "badge-green" if air >= 0.80 else "badge-red"
    air_text = "EEOC PASS" if air >= 0.80 else "DISPARITY DETECTED"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Adverse Impact Ratio (AIR)</div>
        <div class="metric-value">{air:.2f}</div>
        <div style="margin-top:0.5rem;">
            <span class="badge {air_status_class}">{air_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    eog_status_class = "badge-green" if abs(eog) < 0.05 else "badge-amber"
    eog_text = "EQUAL OPPORTUNITY MET" if abs(eog) < 0.05 else "QUALIFIED GAP DETECTED"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Equal Opportunity Gap</div>
        <div class="metric-value">{eog*100:.1f}%</div>
        <div style="margin-top:0.5rem;">
            <span class="badge {eog_status_class}">{eog_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Tab Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "🔍 DECISION EXPLAINER", 
    "📈 POPULATION AUDITING", 
    "📊 MODEL COMPARISON RADAR", 
    "📜 ACADEMIC REPORT"
])

# ---------------------------------------------------------
# Tab 1: XAI Decision Explainer
# ---------------------------------------------------------
with tabs[0]:
    t1_left, t1_right = st.columns([7, 3])
    
    with t1_left:
        st.markdown(f"""
        <div class="chart-wrap">
            <div class="chart-header">
                <div>
                    <div class="chart-title">Black-Box SHAP Waterfall Plot</div>
                    <div class="chart-subtitle">Deconstructing {app_name}'s credit score contribution ({app_gender})</div>
                </div>
                <span class="badge badge-blue">Linear SHAP Projection</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Prepare Plotly Waterfall
        display_names = []
        for name in active_features:
            if name == 'credit_score': display_names.append("Credit Score")
            elif name == 'income': display_names.append("Annual Income")
            elif name == 'dti': display_names.append("DTI Ratio")
            elif name == 'savings': display_names.append("Savings Balance")
            elif name == 'gender': display_names.append("Gender Indicator")
            else: display_names.append(name.title())
            
        x_labels = ["Base Rate"] + display_names + ["Calculated Probability"]
        measure = ["absolute"] + ["relative"] * len(active_features) + ["total"]
        values = [p_base * 100] + list(phi_prob * 100) + [0]
        
        text_labels = [f"{p_base*100:.1f}%"]
        for val in phi_prob:
            sign = "+" if val >= 0 else ""
            text_labels.append(f"{sign}{val*100:.1f}%")
        text_labels.append(f"{p_pred*100:.1f}%")
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=x_labels,
            textposition="outside",
            text=text_labels,
            y=values,
            connector={"line":{"color": text_muted, "width":1, "dash":"dot"}},
            decreasing={"marker":{"color": red}},
            increasing={"marker":{"color": green}},
            totals={"marker":{"color": "#2563eb"}},
        ))
        
        fig_waterfall.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickangle=-10, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(
                title="Probability (%)",
                range=[0, 115],
                gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
                zerolinecolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)"
            )
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t1_right:
        st.markdown("""
        <div class="chart-wrap" style="height: 100%;">
            <div class="chart-header">
                <div class="chart-title">Plain-English AI Explanation</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Write custom narrative based on SHAP values
        sorted_indices = np.argsort(np.abs(phi_prob))[::-1]
        
        def format_feat_name(f):
            if f == 'credit_score': return f"Credit Score of {app_credit}"
            if f == 'income': return f"Annual Income of ${app_income:,}"
            if f == 'dti': return f"Debt-to-Income ratio of {app_dti}%"
            if f == 'savings': return f"Savings Balance of ${app_savings:,}"
            if f == 'gender': return f"demographic attribute (Gender = {app_gender})"
            return f
            
        narrative = f"The model has **{decision.upper()}** the loan application for **{app_name}**. "
        narrative += f"The applicant's credit score yields an approval probability of **{p_pred*100:.1f}%**, relative to the "
        narrative += f"baseline average of **{p_base*100:.1f}%**.\n\n"
        
        narrative += f"#### Feature Contributions:\n"
        for idx in sorted_indices:
            feat = active_features[idx]
            val = phi_prob[idx]
            effect = "increased" if val >= 0 else "decreased"
            col = "green" if val >= 0 else "red"
            narrative += f"- **{display_names[idx]}**: Pushed probability **{effect}** by **<span style='color:{globals()[col]}; font-weight:700;'>{abs(val)*100:.1f}%</span>**.\n"
            
        narrative += "\n#### Comparative Model Simulations:\n"
        biased_prob = compare_preds['biased']
        fair_prob = compare_preds['fair_masked']
        reweighted_prob = compare_preds['reweighted']
        
        # Table of comparison
        narrative += f"""
        <table class="data-table" style="font-size:0.75rem;">
            <thead>
                <tr>
                    <th>Model Configuration</th>
                    <th>Probability</th>
                    <th>Outcome</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Biased (Direct Fit)</td>
                    <td>{biased_prob*100:.1f}%</td>
                    <td><span class="badge {'badge-green' if biased_prob >= 0.50 else 'badge-red'}">{'APPROVED' if biased_prob >= 0.50 else 'DENIED'}</span></td>
                </tr>
                <tr>
                    <td>Feature Masked</td>
                    <td>{fair_prob*100:.1f}%</td>
                    <td><span class="badge {'badge-green' if fair_prob >= 0.50 else 'badge-red'}">{'APPROVED' if fair_prob >= 0.50 else 'DENIED'}</span></td>
                </tr>
                <tr>
                    <td>Mitigated (Reweighted)</td>
                    <td>{reweighted_prob*100:.1f}%</td>
                    <td><span class="badge {'badge-green' if reweighted_prob >= 0.50 else 'badge-red'}">{'APPROVED' if reweighted_prob >= 0.50 else 'DENIED'}</span></td>
                </tr>
            </tbody>
        </table>
        """
        
        st.markdown(narrative, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Population Auditing & Dial Gauge
# ---------------------------------------------------------
with tabs[1]:
    t2_left, t2_right = st.columns([6, 4])
    
    with t2_left:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-header">
                <div>
                    <div class="chart-title">Adverse Impact Ratio (AIR) Dial Indicator</div>
                    <div class="chart-subtitle">Speedometer audit of Demographic Parity</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Dial Gauge for AIR
        bar_color = red if air < 0.80 else (amber if air < 0.90 else green)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = air,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 1.5], 'tickwidth': 1, 'tickcolor': text_muted},
                'bar': {'color': bar_color},
                'bgcolor': "rgba(0,0,0,0.1)",
                'borderwidth': 1,
                'bordercolor': border,
                'steps': [
                    {'range': [0, 0.8], 'color': 'rgba(244, 63, 94, 0.05)'},
                    {'range': [0.8, 1.0], 'color': 'rgba(16, 185, 129, 0.05)'},
                    {'range': [1.0, 1.5], 'color': 'rgba(59, 130, 246, 0.05)'}
                ],
                'threshold': {
                    'line': {'color': accent, 'width': 2},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text),
            margin=dict(l=20, r=20, t=10, b=10),
            height=220
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        
        # Demographic distribution bar chart
        rates = [selection_rate_female * 100, selection_rate_male * 100]
        fig_dp_bar = go.Figure(data=[
            go.Bar(
                x=['Female Group', 'Male Group'], 
                y=rates,
                marker_color=[red if air < 0.80 else '#3b82f6', '#3b82f6'],
                text=[f"{r:.1f}%" for r in rates],
                textposition='auto'
            )
        ])
        fig_dp_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Selection Rate (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)"),
            height=180
        )
        st.plotly_chart(fig_dp_bar, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t2_right:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-header">
                <div class="chart-title">Population Metrics Matrix</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Audited Metric</th>
                    <th>Female (Minority)</th>
                    <th>Male (Majority)</th>
                    <th>Audit Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Selection Rate</td>
                    <td>{selection_rate_female*100:.1f}%</td>
                    <td>{selection_rate_male*100:.1f}%</td>
                    <td>AIR = {air:.2f} ({"EEOC PASS" if air >= 0.80 else "FAIL"})</td>
                </tr>
                <tr>
                    <td>True Positive Rate (TPR)</td>
                    <td>{tpr_female*100:.1f}%</td>
                    <td>{tpr_male*100:.1f}%</td>
                    <td>TPR Gap = {eog*100:.1f}% ({"PASS" if abs(eog) < 0.05 else "FAIL"})</td>
                </tr>
                <tr>
                    <td>Average FICO Credit Score</td>
                    <td>{df_train[df_train['gender']==0]['credit_score'].mean():.1f}</td>
                    <td>{df_train[df_train['gender']==1]['credit_score'].mean():.1f}</td>
                    <td><span class="badge badge-blue">Identical Input Cap</span></td>
                </tr>
                <tr>
                    <td>Average Annual Income</td>
                    <td>${df_train[df_train['gender']==0]['income'].mean():,.0f}</td>
                    <td>${df_train[df_train['gender']==1]['income'].mean():,.0f}</td>
                    <td><span class="badge badge-blue">Identical Input Cap</span></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        # Calculate confusion tables metrics
        tp_m = np.sum((df_train['y_historic'] == 1) & (preds_pop == 1) & male_indices)
        fp_m = np.sum((df_train['y_historic'] == 0) & (preds_pop == 1) & male_indices)
        fn_m = np.sum((df_train['y_historic'] == 1) & (preds_pop == 0) & male_indices)
        tn_m = np.sum((df_train['y_historic'] == 0) & (preds_pop == 0) & male_indices)
        
        tp_f = np.sum((df_train['y_historic'] == 1) & (preds_pop == 1) & female_indices)
        fp_f = np.sum((df_train['y_historic'] == 0) & (preds_pop == 1) & female_indices)
        fn_f = np.sum((df_train['y_historic'] == 1) & (preds_pop == 0) & female_indices)
        tn_f = np.sum((df_train['y_historic'] == 0) & (preds_pop == 0) & female_indices)
        
        # Model performance
        accuracy = np.mean(preds_pop == df_train['y_historic'])
        
        st.markdown(f"""
        <br>
        
        #### Performance Matrix
        - **Global Model Accuracy:** `{accuracy*100:.1f}%`
        - **False Negative Count (Female):** `{fn_f}` (Qualified applicants denied)
        - **False Negative Count (Male):** `{fn_m}`
        
        #### Audit Log Narrative:
        - **EEOC 4/5ths Rule:** Determines whether the selection rate of the protected class is at least **80%** of the majority group. Under 'Biased Model', it fails significantly due to biased historical training labels.
        - **Equal Opportunity Gap:** Monitors the gap in True Positive Rates. A gap above **5%** indicates a model that is systematically under-approving qualified minority candidates.
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 3: Model Comparison Radar
# ---------------------------------------------------------
with tabs[2]:
    t3_left, t3_right = st.columns(2)
    
    with t3_left:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-header">
                <div>
                    <div class="chart-title">Coefficient Comparison Radar</div>
                    <div class="chart-subtitle">Overlaying model weights (normalized)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Build Radar
        radar_features = ['Credit Score', 'Annual Income', 'DTI Ratio', 'Savings', 'Gender']
        
        def get_norm_weights(key, features_list):
            m, sc, feats = models_dict[key]
            raw_w = m.coef_[0]
            mapped_w = []
            for f in ['credit_score', 'income', 'dti', 'savings', 'gender']:
                if f in feats:
                    mapped_w.append(raw_w[feats.index(f)])
                else:
                    mapped_w.append(0.0)
            mapped_w = np.array(mapped_w)
            abs_sum = np.sum(np.abs(mapped_w))
            return mapped_w / abs_sum if abs_sum > 0 else mapped_w
            
        w_b = get_norm_weights('biased', active_features)
        w_f = get_norm_weights('fair_masked', active_features)
        w_r = get_norm_weights('reweighted', active_features)
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(w_b) + [w_b[0]],
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name='Biased Model',
            line_color=red,
            fillcolor=red_muted
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(w_f) + [w_f[0]],
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name='Feature Masked',
            line_color=amber,
            fillcolor=amber_muted
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(w_r) + [w_r[0]],
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name='Mitigated (Reweighted)',
            line_color=green,
            fillcolor=green_muted
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-0.8, 0.8],
                    showticklabels=False,
                    gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)"
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text_muted, size=10),
            margin=dict(l=40, r=40, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t3_right:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-header">
                <div class="chart-title">Model Weights Audit</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display weights table
        m_b, _, _ = models_dict['biased']
        m_f, _, _ = models_dict['fair_masked']
        m_r, _, _ = models_dict['reweighted']
        
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Biased Model Weight</th>
                    <th>Feature Masked Weight</th>
                    <th>Mitigated Weight</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Credit Score</td>
                    <td>{m_b.coef_[0][0]:.3f}</td>
                    <td>{m_f.coef_[0][0]:.3f}</td>
                    <td>{m_r.coef_[0][0]:.3f}</td>
                </tr>
                <tr>
                    <td>Annual Income</td>
                    <td>{m_b.coef_[0][1]:.3f}</td>
                    <td>{m_f.coef_[0][1]:.3f}</td>
                    <td>{m_r.coef_[0][1]:.3f}</td>
                </tr>
                <tr>
                    <td>DTI Ratio</td>
                    <td>{m_b.coef_[0][2]:.3f}</td>
                    <td>{m_f.coef_[0][2]:.3f}</td>
                    <td>{m_r.coef_[0][2]:.3f}</td>
                </tr>
                <tr>
                    <td>Savings</td>
                    <td>{m_b.coef_[0][3]:.3f}</td>
                    <td>{m_f.coef_[0][3]:.3f}</td>
                    <td>{m_r.coef_[0][3]:.3f}</td>
                </tr>
                <tr>
                    <td>Gender</td>
                    <td>{m_b.coef_[0][4]:.3f}</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
        
        <br>
        
        #### Quantitative Insights:
        1. In the **Biased Model**, the Gender coefficient is highly negative, showing that the model treats Gender as a primary negative indicator.
        2. In the **Feature Masked Model**, we remove Gender, but the model redistributes that weight onto other financial features that correlate with Gender in the biased historical dataset (e.g. Savings or DTI proxies).
        3. In the **Mitigated Model**, we use sample weights. This allows the model to learn a fair decision boundary on financial features alone without reproducing demographic disparities.
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 4: Academic Report
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-header">
            <div>
                <div class="chart-title">Course Project Report</div>
                <div class="chart-subtitle">Academic Paper Presentation</div>
            </div>
            <span class="badge badge-blue">PDF Printable Layout</span>
        </div>
        <hr style="border-color: var(--border); opacity:0.1; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    report_path = "/Users/devanand/.gemini/antigravity/scratch/xai-visualizer/project_report.md"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            st.markdown(f.read())
    else:
        st.markdown("The project report file is currently being created. It will appear here shortly.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Scrollable Live Audit Terminal Console (Fixed bottom/side)
# ---------------------------------------------------------
st.markdown("### 🖥️ Aequitas Real-Time Audit Console")
console_html = "<div class='console-log'>"
for line in log_lines:
    console_html += f"<div class='console-line'>{line}</div>"
console_html += "</div>"
st.markdown(console_html, unsafe_allow_html=True)
