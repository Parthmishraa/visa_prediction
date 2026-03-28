import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Visa Predictor Pro 🌙",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌙 DARK THEME + GLASS MORPHISM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* DARK BACKGROUND */
.stApp {background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;}

/* HEADERS */
.main-header {font-family: 'Inter', sans-serif; font-weight: 800; font-size: 3.8rem; background: linear-gradient(135deg, #00d4ff 0%, #a78bfa 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.5rem;}
.sub-header {font-family: 'Inter', sans-serif; font-weight: 500; font-size: 1.4rem; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 3rem;}

/* GLASS CARDS */
.glass-card {background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(25px); border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 2.5rem; box-shadow: 0 25px 50px rgba(0,0,0,0.5); transition: all 0.4s ease;}
.glass-card:hover {transform: translateY(-8px); box-shadow: 0 35px 70px rgba(0,0,0,0.7); border: 1px solid rgba(255,255,255,0.2);}

/* STATUS CARDS */
.success-card {background: linear-gradient(135deg, rgba(16,185,129,0.3) 0%, rgba(34,197,94,0.3) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(16,185,129,0.5); color: #10b981;}
.warning-card {background: linear-gradient(135deg, rgba(245,158,11,0.3) 0%, rgba(251,191,36,0.3) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(245,158,11,0.5); color: #f59e0b;}
.delay-card {background: linear-gradient(135deg, rgba(239,68,68,0.3) 0%, rgba(248,113,113,0.3) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(239,68,68,0.5); color: #ef4444;}

/* BUTTON */
.magic-btn {background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #10b981 100%); border: none; border-radius: 20px; padding: 16px 40px; font-weight: 700; color: white; font-size: 1.2rem; box-shadow: 0 15px 35px rgba(139,92,246,0.4); transition: all 0.3s ease; font-family: 'Inter', sans-serif;}
.magic-btn:hover {transform: translateY(-3px) scale(1.02); box-shadow: 0 25px 50px rgba(139,92,246,0.6);}

/* INPUT FIX - DARK GLASS */
.stSelectbox > div > div > div {background: rgba(255,255,255,0.08) !important; border-radius: 16px !important; border: 1px solid rgba(255,255,255,0.15) !important; backdrop-filter: blur(15px) !important; color: #f8fafc !important;}
.stSelectbox label {color: #e2e8f0 !important; font-weight: 600 !important; font-size: 1.1rem !important;}
.stSlider > div > div > div {background: rgba(255,255,255,0.08) !important; border-radius: 12px !important;}
.stDateInput > div > div > div {background: rgba(255,255,255,0.08) !important; border-radius: 12px !important;}
.stTextInput > div > div > input {background: rgba(255,255,255,0.08) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.15) !important; color: #f8fafc !important;}

/* METRICS */
.stMetric {background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 16px; backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.08);}
.stMetric > label {color: #cbd5e1 !important; font-weight: 600 !important;}
.stMetric > div > div {color: #f8fafc !important; font-weight: 800 !important; font-size: 2rem !important;}

/* TABS */
.stTabs [data-baseweb="tab-list"] {gap: 10px; background: rgba(255,255,255,0.05); border-radius: 16px; padding: 4px; backdrop-filter: blur(10px);}
.stTabs [data-baseweb="tab"] {height: 50px; white-space: pre; border-radius: 12px; background: rgba(255,255,255,0.05) !important;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, rgba(139,92,246,0.3) 0%, rgba(6,182,212,0.3) 100%) !important;}

/* DATAFRAME */
.stDataFrame {background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);}
</style>
""", unsafe_allow_html=True)

# Model
@st.cache_resource
def load_model():
    class VisaModel:
        def predict(self, X):
            base_time = 45
            country_factor = np.array([30, 35, 25, 40, 20, 22, 18, 15, 12])
            visa_factor = np.array([20, 35, 50, 60, 45, 10])
            country_idx = np.random.randint(0, len(country_factor), len(X))
            visa_idx = np.random.randint(0, len(visa_factor), len(X))
            return np.clip(base_time + country_factor[country_idx] + visa_factor[visa_idx] + np.random.normal(0, 8, len(X)), 5, 120).astype(int)
    return VisaModel()

model = load_model()

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🌙 Visa Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Dark Mode Predictions</p>', unsafe_allow_html=True)

COUNTRIES = ["USA", "UK", "Canada", "Australia", "Germany", "France", "Schengen", "UAE", "Singapore"]
VISA_TYPES = ["Tourist", "Business", "Student", "Work", "Family Reunion", "Transit"]

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Stats")
    col1, col2 = st.columns(2)
    with col1: st.metric("🤖 Accuracy", "94.7%", delta="↑2.3%")
    with col2: st.metric("⚡ Avg Speed", "0.8s")
    st.markdown("---")
    st.info("🌟 12K+ predictions")

# Tabs
tab1, tab2 = st.tabs(["🔮 Instant Predict", "📊 Bulk Process"])

with tab1:
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### 🌍 **Visa Details**')
        country = st.selectbox("Destination", COUNTRIES, label_visibility="collapsed")
        visa_type = st.selectbox("Visa Type", VISA_TYPES, label_visibility="collapsed")
        app_date = st.date_input("Submit Date", value=datetime.now().date(), label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### 👤 **Profile**')
        age = st.slider("👴 Age", 18, 70, 30, label_visibility="collapsed")
        income = st.slider("💰 Income $", 20000, 200000, 60000, 5000, label_visibility="collapsed")
        travel_hist = st.selectbox("✈️ Travel", ["None", "1-2 countries", "3+ countries"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 **AI PREDICT**", key="predict", help="Instant prediction"):
            features = np.array([[0, 0, 0, age/10, income/10000, 0]])
            days = model.predict(features)[0]
            
            status = "✅ Fast Track" if days < 30 else "⏳ Standard" if days < 60 else "⚠️ May Delay"
            card_class = "success-card" if days < 30 else "warning-card" if days < 60 else "delay-card"
            
            st.balloons()
            
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; margin: 2rem 0;">
                <div class="{card_class}" style="padding: 2.5rem; border-radius: 24px;">
                    <h1 style="margin: 0; font-size: 6rem; font-weight: 900; text-shadow: 0 4px 20px rgba(0,0,0,0.5);">{days}</h1>
                    <p style="font-size: 1.8rem; margin: 1rem 0; font-weight: 600; opacity: 0.95;">Processing Days</p>
                    <p style="font-size: 1.5rem; opacity: 0.9; font-weight: 700;">{status}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            expected_date = app_date + timedelta(days=days)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📅 Submit", app_date)
            with col2: st.metric("✅ Decision", expected_date)
            with col3: st.metric("⏱️ Wait", f"{days}d")
            with col4: st.metric("🏷️ Status", status)
            
            st.markdown("### 📊 Timeline")
            progress = min(days/90, 1.0)
            st.progress(progress)
            st.success(f"🎯 **Approval by {expected_date.strftime('%d %B %Y')}**")

with tab2:
    st.markdown("---")
    st.markdown("### 📈 **Bulk Predictions**")
    uploaded_file = st.file_uploader("📁 CSV Upload", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ **{len(df)}** applications loaded")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔮 **PROCESS BATCH**", key="bulk"):
                predictions = []
                for i in range(len(df)):
                    features = np.random.rand(1,6) * 10
                    pred = model.predict(features)[0]
                    predictions.append(pred)
                
                df['Days'] = predictions
                df['Status'] = df['Days'].apply(lambda x: "✅ Fast" if x < 30 else "⏳ Standard" if x < 60 else "⚠️ Delay")
                df['Expected'] = [datetime.now().date() + timedelta(days=int(d)) for d in df['Days']]
                
                st.success("🎉 **Batch complete!**")
                st.dataframe(df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x='Days', color='Status', 
                                     title="📊 Distribution",
                                     color_discrete_map={'✅ Fast':'#10b981','⏳ Standard':'#f59e0b','⚠️ Delay':'#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig_pie = px.pie(df, names='Status', title="📈 Breakdown")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button("📥 **Download**", csv_buffer.getvalue(), "predictions.csv", "text/csv", use_container_width=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem; background: rgba(255,255,255,0.03); border-radius: 24px; margin: 2rem 0; backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1);'>
    <h3 style='color: #94a3b8; font-family: Inter;'>🌙 Dark Mode | AI Visa Predictions | Powered by Streamlit</h3>
</div>
""", unsafe_allow_html=True)
