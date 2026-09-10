import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & EXECUTIVE STYLING
# ==========================================
st.set_page_config(
    page_title="FinTech Credit Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Glassmorphic FinTech Dashboard Aesthetic)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
        color: #f1f5f9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }
    
    /* Glassmorphism Containers */
    .dashboard-panel {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    /* Typography Overrides */
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #ffffff !important;
    }
    
    /* Sleek Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADER CACHE
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open("logistic_regression_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ CreditMatrix AI")
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>Automated Underwriting Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation Matrix", 
        ["🔍 Single Applicant Assessment", "📂 Portfolio Batch Analysis", "📊 Model Coefficient Deep Dive"]
    )
    
    st.markdown("---")
    st.markdown("### System Telemetry")
    if model is not None:
        st.success("🟢 Model Engine: Active")
    else:
        st.error("🔴 Model Engine: Offline")

# Guard clause if pickle file is missing
if model is None:
    st.error("⚠️ Error: `logistic_regression_model.pkl` could not be located. Please ensure it is saved in your root directory.")
    st.stop()

# ==========================================
# 4. VIEW: SINGLE APPLICANT ASSESSMENT
# ==========================================
if app_mode == "🔍 Single Applicant Assessment":
    st.title("Credit Risk Evaluation Suite")
    st.markdown("Execute real-time logistic regression modeling to determine precise borrower qualification metrics.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
        st.subheader("📥 Borrower Parameters")
        
        with st.form("prediction_form"):
            income = st.number_input(
                "Annual Income ($)", 
                min_value=0, 
                max_value=10000000, 
                value=700000, 
                step=10000,
                help="Total gross annual revenue reported by the candidate."
            )
            
            credit_score = st.slider(
                "Credit Score", 
                min_value=300, 
                max_value=850, 
                value=720,
                help="Applicant Bureau Credit Score evaluation."
            )
            
            employment_years = st.slider(
                "Employment Tenure (Years)", 
                min_value=0, 
                max_value=40, 
                value=7,
                help="Continuously active duration of career/job history."
            )
            
            debt_ratio = st.slider(
                "Debt Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.30, 
                step=0.01,
                help="Ratio of total aggregated liabilities relative to earnings."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Compute Underwriting Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
        st.subheader("📊 Decision Intelligence")
        
        if submitted:
            # Map inputs strictly to your trained feature column names
            input_df = pd.DataFrame([{
                'income': income,
                'credit_score': credit_score,
                'employment_years': employment_years,
                'debt_ratio': debt_ratio
            }])
            
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            approval_prob = proba[1] * 100  # Probability of class '1' (approved)
            
            # Interactive Plotly Vector Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = approval_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of Approval %", 'font': {'color': '#f1f5f9', 'size': 15}},
                number = {'font': {'color': '#f1f5f9', 'size': 26}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#10b981" if prediction == 1 else "#ef4444"},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': approval_prob
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=210,
                margin=dict(t=25, b=10, l=15, r=15)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Outcome Status Banner
            if prediction == 1:
                st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 12px; border-radius: 10px; text-align: center; margin: 15px 0;">
                        <span style="color: #34d399; font-weight: 700; font-size: 1.1rem;">RECOMMENDATION: LOAN APPROVED ✅</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; padding: 12px; border-radius: 10px; text-align: center; margin: 15px 0;">
                        <span style="color: #f87171; font-weight: 700; font-size: 1.1rem;">RECOMMENDATION: LOAN REJECTED ❌</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Explanatory Diagnostics Expandable Panel
            with st.expander("🔍 Risk Attribution Diagnostics"):
                if debt_ratio > 0.45:
                    st.warning("⚠️ Elevated debt ratio is weighing significantly on the scoring margin.")
                if credit_score < 600:
                    st.warning("⚠️ Bureau credit ranking is below conservative risk thresholds.")
                if employment_years < 2:
                    st.info("ℹ️ Minimal continuous tenure introduces variable stability risk.")
                if approval_prob >= 50 and approval_prob < 65:
                    st.info("ℹ️ Borderline evaluation standing. Additional security covenants suggested.")
        else:
            st.info("👈 Enter applicant criteria parameters on the left and select **Compute Underwriting Risk** to initiate calculation.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. VIEW: PORTFOLIO BATCH ANALYSIS
# ==========================================
elif app_mode == "📂 Portfolio Batch Analysis":
    st.title("Bulk Portfolio Scoring Suite")
    st.markdown("Upload extensive customer evaluation matrices to parse bulk batch outcomes automatically.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    
    # Pre-structured template matching your exact historical dataset configuration
    sample_data = pd.DataFrame({
        'income': [754728, 742484, 568300, 239407],
        'credit_score': [424, 780, 745, 483],
        'employment_years': [8, 0, 9, 9],
        'debt_ratio': [0.48, 0.37, 0.60, 0.84]
    })
    
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload multi-client CSV data record", type=["csv"])
    with c2:
        st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>Dataset Structure Schema</p>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_data.to_csv(index=False).encode('utf-8'),
            file_name="credit_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        required_cols = ['income', 'credit_score', 'employment_years', 'debt_ratio']
        
        if all(col in df_batch.columns for col in required_cols):
            st.success("File framework successfully validated!")
            
            if st.button("Execute Bulk Inference Matrix"):
                preds = model.predict(df_batch[required_cols])
                probs = model.predict_proba(df_batch[required_cols])[:, 1] * 100
                
                df_batch['Prediction'] = preds
                df_batch['Approval Probability (%)'] = np.round(probs, 2)
                df_batch['Status'] = df_batch['Prediction'].apply(lambda x: 'Approved' if x == 1 else 'Rejected')
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_batch, use_container_width=True)
                
                st.download_button(
                    label="📥 Export Evaluated Portfolio CSV",
                    data=df_batch.to_csv(index=False).encode('utf-8'),
                    file_name="processed_portfolio_results.csv",
                    mime="text/csv"
                )
        else:
            st.error(f"Missing column fields! Your CSV file must contain columns matching: {required_cols}")
            
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 6. VIEW: MODEL COEFFICIENT DEEP DIVE
# ==========================================
elif app_mode == "📊 Model Coefficient Deep Dive":
    st.title("Logistic Regression Weights & Attribution")
    st.markdown("Analyze the intrinsic mathematical weights attributed by your model to each metric boundary.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        features = ['income', 'credit_score', 'employment_years', 'debt_ratio']
        
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=True)
        
        # Modern Plotly horizontal bar representation
        fig_bar = px.bar(
            coef_df, 
            x='Coefficient', 
            y='Feature', 
            orientation='h',
            title="Feature Weight Impact Vector",
            color='Coefficient',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=340,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.info("""
        * **Positive Weights (Blue spectrum):** Metrics like Credit Score or Income pull the logit boundary upward toward clearance approval (`1`).
        * **Negative Weights:** Metrics like Debt Ratio heavily penalize the candidate evaluation scoring vector, pushing classification toward rejection (`0`).
        """)
    else:
        st.warning("Coefficient inspection is unavailable for this underlying pipeline object configuration.")
    st.markdown('</div>', unsafe_allow_html=True)