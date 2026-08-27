import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import datetime

# ---------------------------------------------------------
# Page Configurations & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aequitas Terminal // Algorithmic Auditing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Session State for Dynamic Symbol Tickers
# ---------------------------------------------------------
PRESETS = {
    "ROSTOVA/USD (Elena Rostova)": {"gender": "Female", "credit": 680, "income": 62000, "dti": 38, "savings": 12000},
    "VANCE/USD (Marcus Vance)": {"gender": "Male", "credit": 690, "income": 64000, "dti": 35, "savings": 15000},
    "SMITH/USD (Sarah Smith)": {"gender": "Female", "credit": 580, "income": 45000, "dti": 42, "savings": 5000},
    "CHEN/USD (David Chen)": {"gender": "Male", "credit": 780, "income": 120000, "dti": 22, "savings": 65000},
    "PATEL/USD (Aria Patel)": {"gender": "Female", "credit": 640, "income": 58000, "dti": 39, "savings": 8000}
}

if "symbol" not in st.session_state:
    st.session_state.symbol = list(PRESETS.keys())[0]

# Update inputs if symbol selectbox changes
def handle_symbol_change():
    p = PRESETS[st.session_state.symbol]
    st.session_state.app_gender = p["gender"]
    st.session_state.app_credit = p["credit"]
    st.session_state.app_income = p["income"]
    st.session_state.app_dti = p["dti"]
    st.session_state.app_savings = p["savings"]

# Initialize input session states if not present
if "app_gender" not in st.session_state:
    p = PRESETS[st.session_state.symbol]
    st.session_state.app_gender = p["gender"]
    st.session_state.app_credit = p["credit"]
    st.session_state.app_income = p["income"]
    st.session_state.app_dti = p["dti"]
    st.session_state.app_savings = p["savings"]

# ---------------------------------------------------------
# CSS Styling (Crypto Trading Terminal Theme - Charcoal & Zinc)
# ---------------------------------------------------------
bg = "#121315"          # True dark charcoal
bg_subtle = "#15171a"   # Ticker bar and panel header
card = "#1c1d22"        # Panel content backdrop
border = "#282a30"      # Panel border line
text = "#dbdeeb"        # Off-white text
text_muted = "#5d6275"  # Muted slate gray
accent = "#3b82f6"      # Standard blue
green = "#00b074"       # Crypto buy neon green
green_muted = "rgba(0,176,116,0.12)"
red = "#ff3b30"         # Crypto sell neon red
red_muted = "rgba(255,59,48,0.12)"
amber = "#f59e0b"

css = f"""
<style>
    /* Hide default Streamlit layouts */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    /* Apply deep charcoal backgrounds */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 0.5rem 1rem 1rem !important;
        max-width: 1440px !important;
    }}
    
    /* Top Ticker Bar */
    .ticker-bar {{
        background-color: {bg_subtle};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 0.4rem 1rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
    }}
    .ticker-symbol {{
        font-weight: 800;
        font-size: 0.95rem;
        color: {text};
    }}
    .ticker-stat {{
        display: flex;
        flex-direction: column;
    }}
    .ticker-label {{
        font-size: 0.65rem;
        color: {text_muted};
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }}
    .ticker-value {{
        font-weight: 700;
        font-size: 0.85rem;
    }}
    .ticker-green {{ color: {green}; }}
    .ticker-red {{ color: {red}; }}
    
    /* Trading Terminal Layout Panels */
    .terminal-panel {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 6px;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}
    .panel-header {{
        background-color: {bg_subtle};
        border-bottom: 1px solid {border};
        padding: 0.5rem 0.8rem;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {text};
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .panel-content {{
        padding: 0.8rem;
    }}
    
    /* Level 2 Order Book Component */
    .orderbook-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        position: relative;
        overflow: hidden;
    }}
    .orderbook-bg-red {{
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        background: {red_muted};
        z-index: 1;
        transition: width 0.3s ease;
    }}
    .orderbook-bg-green {{
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        background: {green_muted};
        z-index: 1;
        transition: width 0.3s ease;
    }}
    .orderbook-cell {{
        position: relative;
        z-index: 2;
    }}
    .ob-red {{ color: {red}; font-weight: 700; }}
    .ob-green {{ color: {green}; font-weight: 700; }}
    
    /* Big Trade Buy/Sell Buttons */
    .trade-btn {{
        width: 100%;
        padding: 0.6rem;
        border-radius: 4px;
        text-align: center;
        font-weight: 800;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
        color: white;
    }}
    .btn-buy {{ background-color: {green}; }}
    .btn-sell {{ background-color: {red}; }}
    
    /* Terminal Style Table */
    .ledger-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
    }}
    .ledger-table th {{
        text-align: left;
        padding: 0.5rem 0.8rem;
        color: {text_muted};
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 1px solid {border};
        background-color: {bg_subtle};
    }}
    .ledger-table td {{
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid {border};
        color: {text};
    }}
    .ledger-table tr:hover td {{
        background-color: rgba(255,255,255,0.015);
    }}
    
    /* Horizontal flex spacers */
    [data-testid="stHorizontalBlock"] {{
        gap: 0.75rem !important;
    }}
    
    /* Styled widgets to match dark slate inputs */
    div[data-testid="stWidgetLabel"] {{
        font-size: 0.72rem !important;
        color: {text_muted} !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
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

# Load dataset and models (Default 60% bias)
df_train, models_dict = generate_and_train_models(0.60)

# ---------------------------------------------------------
# Top Ticker Header Bar
# ---------------------------------------------------------
# Render selectbox formatted like symbol search in ticker
ticker_cols = st.columns([3.5, 2, 2.2, 2.2, 2.1])

with ticker_cols[0]:
    st.selectbox(
        "Active Asset Ticker",
        list(PRESETS.keys()),
        key="symbol",
        on_change=handle_symbol_change,
        label_visibility="collapsed"
    )

# Pull parameters based on session state
p_gender = st.session_state.app_gender
p_credit = st.session_state.app_credit
p_income = st.session_state.app_income
p_dti = st.session_state.app_dti
p_savings = st.session_state.app_savings

# Active parameters dict
applicant_raw = pd.DataFrame({
    'credit_score': [p_credit],
    'income': [p_income],
    'dti': [p_dti],
    'savings': [p_savings],
    'gender': [0 if p_gender == "Female" else 1]
})

# Choose active model
model_key = 'biased'
active_model, active_scaler, active_features = models_dict[model_key]

# Model Inference
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

# SHAP values in log-odds space
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

# Compute population audit statistics
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
# Dynamic Console Logs Builder (bottom panel)
# ---------------------------------------------------------
timestamp_now = datetime.datetime.now().strftime("%H:%M:%S")
log_lines = [
    f"<span class='console-info'>[{timestamp_now}] [SYSTEM] Aequitas Core audit engine initialized.</span>",
    f"<span class='console-info'>[{timestamp_now}] [DATA] Loaded synthetic population N=800. Bias strength injected: 60%.</span>",
    f"<span class='console-info'>[{timestamp_now}] [MODEL] Loaded model configuration: 'Biased Model (Direct Fit)' ({len(active_features)} features).</span>"
]

if air < 0.80:
    log_lines.append(f"<span class='console-error'>[{timestamp_now}] [WARN] Disparate Impact violation: AIR = {air:.2f} (EEOC 80% Rule fails).</span>")
else:
    log_lines.append(f"<span class='console-success'>[{timestamp_now}] [AUDIT] Demographic Parity holds: AIR = {air:.2f} (EEOC 80% Rule passes).</span>")
    
if abs(eog) >= 0.05:
    log_lines.append(f"<span class='console-warn'>[{timestamp_now}] [WARN] Equal Opportunity gap is {eog*100:.1f}%. Qualified female penalty detected.</span>")
else:
    log_lines.append(f"<span class='console-success'>[{timestamp_now}] [AUDIT] Equal Opportunity condition met. TPR gap is {eog*100:.1f}%.</span>")
    
log_lines.append(f"<span class='console-info'>[{timestamp_now}] [EVAL] Audited applicant '{st.session_state.symbol.split('(')[1].replace(')', '')}' ({p_gender}): Probability = {prob_pred*100:.1f}%. Decision: {decision.upper()}.</span>")

# Fill ticker stats
with ticker_cols[1]:
    # Change % block
    if decision == "Approved":
        color_class = "ticker-green"
        change_text = f"▲ Approved ({prob_pred*100:.1f}%)"
    else:
        color_class = "ticker-red"
        change_text = f"▼ Denied ({prob_pred*100:.1f}%)"
        
    st.markdown(f"""
    <div class="ticker-stat">
        <div class="ticker-label">Approval Status</div>
        <div class="ticker-value {color_class}">{change_text}</div>
    </div>
    """, unsafe_allow_html=True)

with ticker_cols[2]:
    st.markdown(f"""
    <div class="ticker-stat">
        <div class="ticker-label">FICO Credit Score</div>
        <div class="ticker-value">{p_credit}</div>
    </div>
    """, unsafe_allow_html=True)

with ticker_cols[3]:
    air_class = "ticker-green" if air >= 0.80 else "ticker-red"
    st.markdown(f"""
    <div class="ticker-stat">
        <div class="ticker-label">AIR (Adverse Impact)</div>
        <div class="ticker-value {air_class}">{air:.2f} (EEOC {"Pass" if air >= 0.80 else "Fail"})</div>
    </div>
    """, unsafe_allow_html=True)

with ticker_cols[4]:
    eog_class = "ticker-green" if abs(eog) < 0.05 else "ticker-red"
    st.markdown(f"""
    <div class="ticker-stat">
        <div class="ticker-label">TPR Bias Gap</div>
        <div class="ticker-value {eog_class}">{eog*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Middle Dashboard Layout (Graph Column vs Sidebar Order Input)
# ---------------------------------------------------------
mid_left, mid_right = st.columns([7, 3])

with mid_left:
    # 1. Main Candlestick-Style Waterfall Plot
    st.markdown(f"""
    <div class="terminal-panel">
        <div class="panel-header">
            <span>Graph Visualizer // SHAP Waterfall Breakdown</span>
            <span class="badge badge-blue">Interactive view</span>
        </div>
        <div class="panel-content">
    """, unsafe_allow_html=True)
    
    # Custom display names for Plotly Waterfall
    display_names = []
    for name in active_features:
        if name == 'credit_score': display_names.append("Credit Score")
        elif name == 'income': display_names.append("Annual Income")
        elif name == 'dti': display_names.append("DTI Ratio")
        elif name == 'savings': display_names.append("Savings")
        elif name == 'gender': display_names.append("Gender Profile")
        else: display_names.append(name.title())
        
    x_labels = ["Base Rate"] + display_names + ["Total Calculated"]
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
        connector={"line":{"color": "#2a2b30", "width":1.5, "dash":"dot"}},
        decreasing={"marker":{"color": red}},
        increasing={"marker":{"color": green}},
        totals={"marker":{"color": "#2563eb"}},
    ))
    
    fig_waterfall.update_layout(
        paper_bgcolor="#1a1c20",
        plot_bgcolor="#1a1c20",
        font=dict(family="DM Sans, sans-serif", color=text_muted, size=10),
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(tickangle=-10, gridcolor="rgba(255,255,255,0.02)"),
        yaxis=dict(
            title="Approval Probability (%)",
            range=[0, 115],
            gridcolor="#282a30",
            zerolinecolor="#282a30"
        ),
        height=320
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": True})
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # 2. Bottom Applicant Ledger Grid (mimics stock orders grid)
    st.markdown("""
    <div class="terminal-panel">
        <div class="panel-header">
            <span>Applicant Audit Ledger (Historical Population Logs)</span>
        </div>
        <div class="panel-content" style="padding:0;">
            <table class="ledger-table">
                <thead>
                    <tr>
                        <th>Asset ID</th>
                        <th>Demographic Group</th>
                        <th>FICO Score</th>
                        <th>Annual Income</th>
                        <th>Debt Ratio</th>
                        <th>Decision Code</th>
                    </tr>
                </thead>
                <tbody>
    """, unsafe_allow_html=True)
    
    # Display top 5 applicants from training set to look like pending orders
    sample_rows = df_train.head(5)
    for idx, row in sample_rows.iterrows():
        applicant_id = f"AP-{10839 + idx}/USD"
        gender_lbl = "Female (Minority)" if row['gender'] == 0 else "Male (Majority)"
        fico = int(row['credit_score'])
        income_lbl = f"${int(row['income']):,}"
        dti_lbl = f"{int(row['dti'])}%"
        
        # Pull model decision
        scaled_row = active_scaler.transform(pd.DataFrame({
            'credit_score': [row['credit_score']],
            'income': [row['income']],
            'dti': [row['dti']],
            'savings': [row['savings']],
            'gender': [row['gender']]
        })[active_features])
        row_prob = active_model.predict_proba(scaled_row)[0, 1]
        
        if row_prob >= 0.50:
            dec_lbl = "<span class='ob-green'>APPROVED</span>"
        else:
            dec_lbl = "<span class='ob-red'>DENIED</span>"
            
        st.markdown(f"""
                    <tr>
                        <td>{applicant_id}</td>
                        <td>{gender_lbl}</td>
                        <td>{fico}</td>
                        <td>{income_lbl}</td>
                        <td>{dti_lbl}</td>
                        <td>{dec_lbl}</td>
                    </tr>
        """, unsafe_allow_html=True)
        
    st.markdown("""
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

with mid_right:
    # 3. Right-Side Order Entry Panel (styled like limit/market order tab)
    st.markdown(f"""
    <div class="terminal-panel">
        <div class="panel-header">
            <span>Terminal Order Panel</span>
        </div>
        <div class="panel-content">
    """, unsafe_allow_html=True)
    
    # Styled buttons for BUY (Approved) or SELL (Denied)
    if decision == "Approved":
        st.markdown(f"<div class='trade-btn btn-buy'>APPROVED ({prob_pred*100:.1f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='trade-btn btn-sell'>DENIED ({prob_pred*100:.1f}%)</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sliders / interactive sliders inside order entry panel
    g_input = st.selectbox("Demographic", ["Female", "Male"], index=0 if p_gender=="Female" else 1, key="app_gender")
    credit_input = st.slider("FICO Score", 300, 850, p_credit, step=5, key="app_credit")
    income_input = st.slider("Income ($)", 20000, 250000, p_income, step=1000, key="app_income")
    dti_input = st.slider("DTI Ratio (%)", 5, 80, p_dti, step=1, key="app_dti")
    savings_input = st.slider("Savings ($)", 0, 150000, p_savings, step=1000, key="app_savings")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # 4. Right-Side Level 2 Order Book (SHAP contributions)
    st.markdown(f"""
    <div class="terminal-panel">
        <div class="panel-header">
            <span>Order Book // SHAP Contributions</span>
        </div>
        <div class="panel-content" style="padding:0;">
    """, unsafe_allow_html=True)
    
    # Draw Order Book rows (Level 2 bids and asks)
    # Separating positive (bids/green) and negative (asks/red) contributions
    # Normalizing size width
    max_val = max(np.max(np.abs(phi_prob)), 1e-9)
    
    # Create list of tuples of display_name, contribution
    features_contribs = list(zip(display_names, phi_prob))
    
    # Negative contributions (Red/Asks) -> sorted ascending (largest negative first)
    negatives = [c for c in features_contribs if c[1] < 0]
    negatives = sorted(negatives, key=lambda x: x[1]) # largest negative first
    
    # Positive contributions (Green/Bids) -> sorted descending (largest positive first)
    positives = [c for c in features_contribs if c[1] >= 0]
    positives = sorted(positives, key=lambda x: x[1], reverse=True) # largest positive first
    
    # Draw negative (Asks)
    for name, val in negatives:
        pct = abs(val) / max_val * 100
        st.markdown(f"""
        <div class="orderbook-row">
            <div class="orderbook-bg-red" style="width: {pct:.0f}%;"></div>
            <div class="orderbook-cell" style="font-weight:700;">{name}</div>
            <div class="orderbook-cell ob-red">{val*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Draw mid-market price line
    st.markdown(f"""
    <div style="border-top:1px dashed {border}; border-bottom:1px dashed {border}; padding:0.25rem 0.5rem; text-align:center; font-weight:800; font-size:0.75rem; background:#121315;">
        BASE APPR RATE: {p_base*100:.1f}% &nbsp;&bull;&nbsp; SPREAD LOG-ODDS: {sum_phi:.3f}
    </div>
    """, unsafe_allow_html=True)
    
    # Draw positive (Bids)
    for name, val in positives:
        pct = abs(val) / max_val * 100
        st.markdown(f"""
        <div class="orderbook-row">
            <div class="orderbook-bg-green" style="width: {pct:.0f}%;"></div>
            <div class="orderbook-cell" style="font-weight:700;">{name}</div>
            <div class="orderbook-cell ob-green">{val*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Console Logs Builder (bottom panel)
# ---------------------------------------------------------
st.markdown("### 🖥️ Aequitas Real-Time Audit Console")
console_html = f"<div class='console-log'>"
for line in log_lines:
    console_html += f"<div class='console-line'>{line}</div>"
console_html += "</div>"
st.markdown(console_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# Bottom-most Tabs for Auditing details & report
# ---------------------------------------------------------
tabs = st.tabs([
    "📈 COEF COMPARISON", 
    "📈 POPULATION AUDIT", 
    "📜 ACADEMIC PROJECT REPORT"
])

with tabs[0]:
    st.markdown("""
    <div class="terminal-panel">
        <div class="panel-header">Coefficient Normalization Radar Chart</div>
        <div class="panel-content">
    """, unsafe_allow_html=True)
    
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
        fillcolor='rgba(245,158,11,0.05)'
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
                gridcolor="#282a30"
            ),
            angularaxis=dict(
                gridcolor="#282a30"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="#1c1d22",
        plot_bgcolor="#1c1d22",
        font=dict(family="DM Sans, sans-serif", color=text_muted, size=10),
        margin=dict(l=40, r=40, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=320
    )
    
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": True})
    st.markdown("</div></div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("""
    <div class="terminal-panel">
        <div class="panel-header">Demographic parity audit summary</div>
        <div class="panel-content">
    """, unsafe_allow_html=True)
    
    # Visualizing Demographic Parity
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
        paper_bgcolor="#1c1d22",
        plot_bgcolor="#1c1d22",
        font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Selection Rate (%)", range=[0, 100], gridcolor="#282a30"),
        height=240
    )
    st.plotly_chart(fig_dp_bar, use_container_width=True, config={"displayModeBar": True})
    st.markdown("</div></div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("""
    <div class="terminal-panel">
        <div class="panel-header">Course Project Report</div>
        <div class="panel-content">
    """, unsafe_allow_html=True)
    
    report_path = "/Users/devanand/.gemini/antigravity/scratch/xai-visualizer/project_report.md"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            st.markdown(f.read())
    else:
        st.markdown("The project report file is currently being created. It will appear here shortly.")
        
    st.markdown("</div></div>", unsafe_allow_html=True)
