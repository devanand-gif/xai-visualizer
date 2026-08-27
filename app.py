import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------------------------------------
# Page Configurations
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aequitas XAI: Credit Decision Explainer",
    page_icon="◆",
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
# CSS Design System Injection
# ---------------------------------------------------------
# Define colors based on the theme
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

css = f"""
<style>
    /* Hide Streamlit default components */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    /* Apply base styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}
    
    /* Sidebar styling override */
    section[data-testid="stSidebar"] {{
        background-color: {bg_subtle} !important;
        border-right: 1px solid {border} !important;
    }}
    
    /* Tab adjustments */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {text_muted} !important;
        font-size: 0.835rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text} !important;
        background: {card} !important;
        border-color: {border} !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: {bg_subtle} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        padding: 3px;
    }}
    
    /* Custom Card */
    .metric-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        box-shadow: {shadow};
        margin-bottom: 1rem;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {text_muted};
        font-weight: 500;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.03em;
        line-height: 1.2;
    }}
    .metric-delta {{
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.4rem;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }}
    .delta-up {{ color: {green}; background: {green_muted}; }}
    .delta-down {{ color: {red}; background: {red_muted}; }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-green {{ color: {green}; background: {green_muted}; }}
    .badge-red {{ color: {red}; background: {red_muted}; }}
    .badge-blue {{ color: #2563eb; background: rgba(37, 99, 235, 0.1); }}
    
    .chart-wrap {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: {shadow};
        margin-bottom: 1rem;
    }}
    .chart-title {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {text};
    }}
    .chart-subtitle {{
        font-size: 0.75rem;
        color: {text_muted};
        margin-bottom: 1rem;
    }}
    
    /* Custom Data Table */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.6rem 0.8rem;
        color: {text_muted};
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid {border};
        background: {bg_subtle};
    }}
    .data-table td {{
        padding: 0.65rem 0.8rem;
        color: {text};
        border-bottom: 1px solid {border_subtle};
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Header Brand */
    .brand {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.5rem;
    }}
    .brand-symbol {{
        font-size: 1.5rem;
        color: #2563eb;
    }}
    .brand-name {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.02em;
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
    
    # 1. Synthesize demographic group (Gender: 1 = Male, 0 = Female)
    # 50/50 split
    gender = np.random.binomial(1, 0.5, N)
    
    # 2. Synthesize baseline financial features (identical distributions to isolate bias)
    credit_score = np.random.normal(650, 80, N)
    credit_score = np.clip(credit_score, 300, 850)
    
    income = np.random.normal(75000, 25000, N)
    income = np.clip(income, 20000, 250000)
    
    dti = np.random.normal(35, 12, N)
    dti = np.clip(dti, 5, 80)
    
    savings = np.random.normal(25000, 15000, N)
    savings = np.clip(savings, 0, 150000)
    
    # Combine into raw dataframe
    df = pd.DataFrame({
        'credit_score': credit_score,
        'income': income,
        'dti': dti,
        'savings': savings,
        'gender': gender
    })
    
    # 3. True Qualification Model (deserved score based purely on financial capacity)
    # Scale features standardly to calculate true capability
    scaler_temp = StandardScaler()
    scaled_feats = scaler_temp.fit_transform(df[['credit_score', 'income', 'dti', 'savings']])
    
    # Weights for true capability: positive for credit score, income, savings; negative for DTI
    weights_true = np.array([0.45, 0.35, -0.30, 0.20])
    raw_score = np.dot(scaled_feats, weights_true) + np.random.normal(0, 0.1, N) # minor noise
    
    # Decision boundary to get ~45% approval baseline
    true_threshold = np.percentile(raw_score, 55)
    y_true = (raw_score >= true_threshold).astype(int)
    df['y_true'] = y_true
    
    # 4. Inject Historical Bias into Decision Labels
    # If Female (gender=0) and genuinely qualified (y_true=1), reject them with probability of bias_strength
    y_historic = y_true.copy()
    for i in range(N):
        if gender[i] == 0 and y_true[i] == 1:
            if np.random.rand() < bias_strength:
                y_historic[i] = 0 # Discrimination event
                
    df['y_historic'] = y_historic
    
    # Define features
    features_biased = ['credit_score', 'income', 'dti', 'savings', 'gender']
    features_fair = ['credit_score', 'income', 'dti', 'savings']
    
    # Standard Scalers
    scaler_biased = StandardScaler()
    X_train_biased = scaler_biased.fit_transform(df[features_biased])
    
    scaler_fair = StandardScaler()
    X_train_fair = scaler_fair.fit_transform(df[features_fair])
    
    # 5. Train Models
    # Model 1: Biased Model (Direct Fit on biased historical labels including Gender)
    model_biased = LogisticRegression(C=1.0, random_state=42)
    model_biased.fit(X_train_biased, df['y_historic'])
    
    # Model 2: Fair Model (Feature Masking - completely blinds model to gender)
    model_fair = LogisticRegression(C=1.0, random_state=42)
    model_fair.fit(X_train_fair, df['y_historic'])
    
    # Model 3: Mitigated Model (Kamiran-Calders Reweighing pre-processing)
    # Compute weights
    n_total = N
    n_male = np.sum(gender == 1)
    n_female = np.sum(gender == 0)
    n_approved = np.sum(y_historic == 1)
    n_denied = np.sum(y_historic == 0)
    
    # Group counts
    n_male_approved = np.sum((gender == 1) & (y_historic == 1))
    n_male_denied = np.sum((gender == 1) & (y_historic == 0))
    n_female_approved = np.sum((gender == 0) & (y_historic == 1))
    n_female_denied = np.sum((gender == 0) & (y_historic == 0))
    
    # Apply formulas: W = (N_group * N_label) / (N * N_group_label)
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
            
    # Train Logistic Regression with weights (still including gender, or without gender?
    # Usually, we do it without gender to avoid direct disparate treatment, or with it. Let's do it without gender
    # to show a "Mitigated Fair Model" that has corrected weights but is also masked).
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
# Header Layout
head_left, head_right = st.columns([8, 2])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-symbol">◆</span>
        <span class="brand-name">Aequitas XAI // Credit Explainer</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light UI" if IS_DARK else "🌙 Dark UI"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.sidebar.markdown("### 1. Training Parameters")
bias_slider = st.sidebar.slider(
    "Historical Bias Strength (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Discrimination probability applied historically to qualified female applicants."
)
bias_strength = bias_slider / 100.0

# Load dataset and models
df_train, models_dict = generate_and_train_models(bias_strength)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Applicant Profile")
app_name = st.sidebar.text_input("Applicant Name", value="Elena Rostova")
app_gender = st.sidebar.radio("Gender Profile", ["Female", "Male"], index=0)
app_gender_val = 0 if app_gender == "Female" else 1

app_credit = st.sidebar.slider("Credit Score", 300, 850, 680, step=5)
app_income = st.sidebar.slider("Annual Income ($)", 20000, 250000, 62000, step=1000)
app_dti = st.sidebar.slider("Debt-to-Income (DTI) %", 5, 80, 38, step=1)
app_savings = st.sidebar.slider("Savings Balance ($)", 0, 150000, 12000, step=1000)

applicant_raw = pd.DataFrame({
    'credit_score': [app_credit],
    'income': [app_income],
    'dti': [app_dti],
    'savings': [app_savings],
    'gender': [app_gender_val]
})

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Active Decision Model")
model_choice = st.sidebar.selectbox(
    "Model Selection",
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

# ---------------------------------------------------------
# Main UI Dashboard
# ---------------------------------------------------------

# Metric Cards Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    badge_class = "badge-green" if decision == "Approved" else "badge-red"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Decision Outcome</div>
        <div class="metric-value">{decision}</div>
        <div style="margin-top:0.4rem;"><span class="badge {badge_class}">{decision}</span></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Approval Probability</div>
        <div class="metric-value">{prob_pred*100:.1f}%</div>
        <div style="margin-top:0.4rem; font-size:0.75rem; color:{text_muted}">Decision Threshold: 50.0%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    # Compute population statistics for audit
    # Apply prediction to whole training set
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
        
    air_class = "delta-up" if air >= 0.80 else "delta-down"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Adverse Impact Ratio (AIR)</div>
        <div class="metric-value">{air:.2f}</div>
        <div class="metric-delta {air_class}">{"Passes EEOC 4/5ths Rule" if air >= 0.80 else "Disparate Impact Detected"}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    # Equal Opportunity Gap (Difference in TPR)
    y_true_pop = df_train['y_true'].values
    qualified_male = (y_true_pop == 1) & male_indices
    qualified_female = (y_true_pop == 1) & female_indices
    
    tpr_male = np.mean(preds_pop[qualified_male]) if np.sum(qualified_male) > 0 else 0.0
    tpr_female = np.mean(preds_pop[qualified_female]) if np.sum(qualified_female) > 0 else 0.0
    
    eog = tpr_male - tpr_female
    eog_class = "delta-up" if abs(eog) < 0.05 else "delta-down"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Equal Opportunity Gap</div>
        <div class="metric-value">{eog*100:.1f}%</div>
        <div class="metric-delta {eog_class}">{"Equal Opportunity Met" if abs(eog) < 0.05 else "Qualified Females Penalized"}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Tabbed View
# ---------------------------------------------------------
tabs = st.tabs([
    "🔍 XAI Decision Explainer", 
    "📊 Population Audit & Bias", 
    "🛠️ Model Weights & Performance", 
    "📝 Project Report"
])

# ---------------------------------------------------------
# Tab 1: XAI Decision Explainer
# ---------------------------------------------------------
with tabs[0]:
    t1_left, t1_right = st.columns([7, 3])
    
    with t1_left:
        st.markdown(f"""
        <div class="chart-wrap">
            <div class="chart-title">Black-Box SHAP Waterfall Plot</div>
            <div class="chart-subtitle">Deconstructing {app_name}'s Loan Decision ({app_gender})</div>
        """, unsafe_allow_html=True)
        
        # Prepare Plotly Waterfall
        display_names = []
        for name in active_features:
            if name == 'credit_score': display_names.append("Credit Score")
            elif name == 'income': display_names.append("Annual Income")
            elif name == 'dti': display_names.append("Debt-to-Income (DTI)")
            elif name == 'savings': display_names.append("Savings Balance")
            elif name == 'gender': display_names.append("Gender")
            else: display_names.append(name.title())
            
        x_labels = ["Base Rate"] + display_names + ["Final Decision"]
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
            xaxis=dict(tickangle=-15, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(
                title="Approval Probability (%)",
                range=[0, 115],
                gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
                zerolinecolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)"
            )
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t1_right:
        st.markdown("""
        <div class="chart-wrap" style="height: 100%;">
            <div class="chart-title">Plain-English AI Explanation</div>
            <div class="chart-subtitle">Audit Narrative</div>
        """, unsafe_allow_html=True)
        
        # Write custom narrative based on SHAP values
        sorted_indices = np.argsort(np.abs(phi_prob))[::-1]
        dominant_feature = active_features[sorted_indices[0]]
        dominant_val = phi_prob[sorted_indices[0]]
        
        def format_feat_name(f):
            if f == 'credit_score': return f"Credit Score of {app_credit}"
            if f == 'income': return f"Annual Income of ${app_income:,}"
            if f == 'dti': return f"Debt-to-Income ratio of {app_dti}%"
            if f == 'savings': return f"Savings Balance of ${app_savings:,}"
            if f == 'gender': return f"demographic attribute (Gender = {app_gender})"
            return f
            
        narrative = f"### Decision Narrative for **{app_name}**\n\n"
        narrative += f"The active machine learning model has **{decision.upper()}** the credit application. "
        narrative += f"The applicant's calculated approval probability is **{p_pred*100:.1f}%**, relative to the "
        narrative += f"average population base rate of **{p_base*100:.1f}%**.\n\n"
        
        narrative += f"#### Key Decision Drivers:\n"
        
        # List drivers
        for idx in sorted_indices:
            feat = active_features[idx]
            val = phi_prob[idx]
            effect = "increased" if val >= 0 else "decreased"
            color = green if val >= 0 else red
            narrative += f"- **{display_names[idx]}**: The applicant's {format_feat_name(feat)} **{effect}** their approval probability by **{abs(val)*100:.1f}%**.\n"
            
        narrative += "\n#### Model Comparison Audit:\n"
        
        # Comparisons
        biased_prob = compare_preds['biased']
        fair_prob = compare_preds['fair_masked']
        reweighted_prob = compare_preds['reweighted']
        
        narrative += f"- **Biased Model**: {biased_prob*100:.1f}% probability (Decision: {'Approved' if biased_prob >= 0.5 else 'Denied'})\n"
        narrative += f"- **Fair Model (Masked)**: {fair_prob*100:.1f}% probability (Decision: {'Approved' if fair_prob >= 0.5 else 'Denied'})\n"
        narrative += f"- **Mitigated Model (Reweighted)**: {reweighted_prob*100:.1f}% probability (Decision: {'Approved' if reweighted_prob >= 0.5 else 'Denied'})\n\n"
        
        # Callout if bias influenced the decision
        if model_key == 'biased' and app_gender == 'Female' and compare_preds['fair_masked'] >= 0.50 and prob_pred < 0.50:
            narrative += f"> [!WARNING]\n"
            narrative += f"> **Algorithmic Disparity Detected:** Under the Biased Model, {app_name} was **Denied** ({prob_pred*100:.1f}%). "
            narrative += f"However, when the model is blinded to gender, her probability rises to **{fair_prob*100:.1f}%** resulting in **Approval**. "
            narrative += f"This shows that historical gender bias was the sole factor behind her rejection."
        elif model_key == 'biased' and app_gender == 'Female' and phi_prob[active_features.index('gender')] < -0.05:
            narrative += f"> [!IMPORTANT]\n"
            narrative += f"> **Gender Penalty Applied:** The model actively penalized this applicant by **{abs(phi_prob[active_features.index('gender')])*100:.1f}%** "
            narrative += f"solely due to being Female, mirroring historical discrimination patterns."
        else:
            narrative += f"> [!NOTE]\n"
            narrative += f"> The active model's decision is primarily driven by financial metrics. Select 'Biased Model' and 'Female' to inspect gender-related disparities."
            
        st.markdown(narrative)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Population Audit & Bias
# ---------------------------------------------------------
with tabs[1]:
    t2_left, t2_right = st.columns([6, 4])
    
    with t2_left:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Selection Rates by Demographic Group</div>
            <div class="chart-subtitle">Audit of Demographic Parity (EEOC 80% Rule)</div>
        """, unsafe_allow_html=True)
        
        # Visualizing Demographic Parity
        groups = ['Female', 'Male']
        rates = [selection_rate_female * 100, selection_rate_male * 100]
        
        fig_dp = go.Figure(data=[
            go.Bar(
                x=groups, 
                y=rates,
                marker_color=[red if selection_rate_female < selection_rate_male * 0.8 else '#2563eb', '#2563eb'],
                text=[f"{r:.1f}%" for r in rates],
                textposition='auto'
            )
        ])
        
        # Add 80% threshold line
        fig_dp.add_shape(
            type="line",
            x0=-0.5, y0=selection_rate_male * 80, x1=1.5, y1=selection_rate_male * 80,
            line=dict(color=red, width=2, dash="dash"),
            name="EEOC 80% Threshold"
        )
        
        fig_dp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Selection Rate (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)"),
            showlegend=False
        )
        
        st.plotly_chart(fig_dp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t2_right:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Fairness & Bias Audit Summary</div>
            <div class="chart-subtitle">Audit report for 800 applicants</div>
        """, unsafe_allow_html=True)
        
        # Demographic metrics table
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Female Group</th>
                    <th>Male Group</th>
                    <th>Audit Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Selection Rate</td>
                    <td>{selection_rate_female*100:.1f}%</td>
                    <td>{selection_rate_male*100:.1f}%</td>
                    <td>AIR = {air:.2f} ({"Pass" if air >= 0.80 else "Fail"})</td>
                </tr>
                <tr>
                    <td>True Positive Rate (TPR)</td>
                    <td>{tpr_female*100:.1f}%</td>
                    <td>{tpr_male*100:.1f}%</td>
                    <td>Gap = {eog*100:.1f}% ({"Pass" if abs(eog) < 0.05 else "Fail"})</td>
                </tr>
                <tr>
                    <td>Average Credit Score</td>
                    <td>{df_train[df_train['gender']==0]['credit_score'].mean():.1f}</td>
                    <td>{df_train[df_train['gender']==1]['credit_score'].mean():.1f}</td>
                    <td><span class="badge badge-blue">Equal Capacity</span></td>
                </tr>
                <tr>
                    <td>Average Income</td>
                    <td>${df_train[df_train['gender']==0]['income'].mean():,.0f}</td>
                    <td>${df_train[df_train['gender']==1]['income'].mean():,.0f}</td>
                    <td><span class="badge badge-blue">Equal Capacity</span></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <br>
        
        #### Audit Findings:
        1. **Demographic Parity (AIR)** compares the overall selection rates. The EEOC's "4/5ths Rule" requires the minority rate to be at least 80% of the majority.
        2. **Equal Opportunity (EOG)** compares the True Positive Rates. It ensures qualified applicants have the same chance of approval regardless of gender.
        3. Since the underlying financial attributes are synthetically generated from **identical distributions**, any difference in selection rates or TPR is a direct measure of algorithmic bias absorbed from the historical dataset.
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 3: Model Weights & Performance
# ---------------------------------------------------------
with tabs[2]:
    t3_left, t3_right = st.columns(2)
    
    with t3_left:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Model Coefficients (Feature Weights)</div>
            <div class="chart-subtitle">Visualizing what the neural networks learned</div>
        """, unsafe_allow_html=True)
        
        # Plot model weights
        coef_names = []
        for feat in active_features:
            if feat == 'credit_score': coef_names.append("Credit Score")
            elif feat == 'income': coef_names.append("Income")
            elif feat == 'dti': coef_names.append("DTI Ratio")
            elif feat == 'savings': coef_names.append("Savings")
            elif feat == 'gender': coef_names.append("Gender")
            else: coef_names.append(feat)
            
        fig_coef = go.Figure(data=[
            go.Bar(
                x=coef_names,
                y=weights,
                marker_color=['#16a34a' if w >= 0 else '#dc2626' for w in weights],
                text=[f"{w:.3f}" for w in weights],
                textposition='auto'
            )
        ])
        
        fig_coef.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Logistic Regression Coefficient Weight", gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)")
        )
        
        st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown(f"""
        **Active Model Intercept (Base Bias):** `{intercept:.4f}`
        
        *Interpretation:*
        - Positive coefficients increase the approval log-odds (e.g., higher Credit Score, Income, Savings).
        - Negative coefficients decrease the approval log-odds (e.g., higher Debt-to-Income ratio).
        - In the **Biased Model**, a negative Gender coefficient means being Female is treated as an active penalty.
        - In the **Mitigated Model**, we use sample reweighting to remove this dependency.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t3_right:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Confusion Matrices (Demographic Split)</div>
            <div class="chart-subtitle">Comparison of errors across demographics</div>
        """, unsafe_allow_html=True)
        
        # Calculate confusion matrices
        y_hist = df_train['y_historic'].values
        
        def get_cm(indices):
            y_h = y_hist[indices]
            y_p = preds_pop[indices]
            
            tp = np.sum((y_h == 1) & (y_p == 1))
            fp = np.sum((y_h == 0) & (y_p == 1))
            fn = np.sum((y_h == 1) & (y_p == 0))
            tn = np.sum((y_h == 0) & (y_p == 0))
            return tp, fp, fn, tn
            
        tp_m, fp_m, fn_m, tn_m = get_cm(male_indices)
        tp_f, fp_f, fn_f, tn_f = get_cm(female_indices)
        
        # Render Confusion Tables
        st.markdown(f"""
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <h5 style="text-align: center; color: {text_muted};">Male Group (Majority)</h5>
                <table class="data-table" style="text-align: center;">
                    <tr>
                        <td style="border:none;"></td>
                        <td style="font-weight:600; font-size:0.7rem; border:none; background:{bg_subtle};">Predicted Denied</td>
                        <td style="font-weight:600; font-size:0.7rem; border:none; background:{bg_subtle};">Predicted Approved</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; font-size:0.7rem; background:{bg_subtle};">Actual Denied</td>
                        <td style="background:{red_muted}; color:{red}; font-weight:600;">{tn_m} (TN)</td>
                        <td style="background:rgba(255,255,255,0.02);">{fp_m} (FP)</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; font-size:0.7rem; background:{bg_subtle};">Actual Approved</td>
                        <td style="background:rgba(255,255,255,0.02);">{fn_m} (FN)</td>
                        <td style="background:{green_muted}; color:{green}; font-weight:600;">{tp_m} (TP)</td>
                    </tr>
                </table>
            </div>
            
            <div style="flex: 1;">
                <h5 style="text-align: center; color: {text_muted};">Female Group (Minority)</h5>
                <table class="data-table" style="text-align: center;">
                    <tr>
                        <td style="border:none;"></td>
                        <td style="font-weight:600; font-size:0.7rem; border:none; background:{bg_subtle};">Predicted Denied</td>
                        <td style="font-weight:600; font-size:0.7rem; border:none; background:{bg_subtle};">Predicted Approved</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; font-size:0.7rem; background:{bg_subtle};">Actual Denied</td>
                        <td style="background:{red_muted}; color:{red}; font-weight:600;">{tn_f} (TN)</td>
                        <td style="background:rgba(255,255,255,0.02);">{fp_f} (FP)</td>
                    </tr>
                    <tr>
                        <td style="font-weight:600; font-size:0.7rem; background:{bg_subtle};">Actual Approved</td>
                        <td style="background:rgba(255,255,255,0.02);">{fn_f} (FN)</td>
                        <td style="background:{green_muted}; color:{green}; font-weight:600;">{tp_f} (TP)</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compute Accuracy
        accuracy_overall = np.mean(preds_pop == y_hist)
        accuracy_male = np.mean(preds_pop[male_indices] == y_hist[male_indices])
        accuracy_female = np.mean(preds_pop[female_indices] == y_hist[female_indices])
        
        st.markdown(f"""
        <br>
        
        #### Model Accuracy Metrics:
        - **Overall Predictive Accuracy:** `{accuracy_overall*100:.1f}%`
        - **Male Group Accuracy:** `{accuracy_male*100:.1f}%`
        - **Female Group Accuracy:** `{accuracy_female*100:.1f}%`
        
        *Key Observation:* In a biased environment, the model's accuracy metrics can look high (e.g. 80%+), but the confusion matrix reveals a high rate of False Negatives (FN) for the female group, showing they are systematically misclassified as "high risk" relative to their true capabilities.
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 4: Project Report
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Course Project Report</div>
        <div class="chart-subtitle">Academic Paper Presentation</div>
        <hr style="border-color: var(--border);">
    """, unsafe_allow_html=True)
    
    # Read and render project report
    report_path = "/Users/devanand/.gemini/antigravity/scratch/xai-visualizer/project_report.md"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            st.markdown(f.read())
    else:
        st.markdown("The project report file is currently being created. It will appear here shortly.")
        
    st.markdown("</div>", unsafe_allow_html=True)
