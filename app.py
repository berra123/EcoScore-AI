import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Harici modül importları (Hata vermemesi için try-except bloğunda)
try:
    from src.calculator import calculate_carbon_footprint, get_ecoscore_grade
except ImportError:
    def calculate_carbon_footprint(gas, elec): return (gas * 2.1) + (elec * 0.45)
    def get_ecoscore_grade(score): return "A" if score < 100 else "B"

try:
    from src.llm_agent import generate_green_recommendation
except ImportError:
   def generate_green_recommendation(
    siparis_detaylari,
    mevcut_co2_kg,
    ecoscore_harfi,
):
    return """
Gemini API bulunamadığı için demo modu çalışıyor.
"""
# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="EcoScore - Karbon Ayak İzi Tahminleme Motoru",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DENTAŞ KURUMSAL RENK PALETİ VE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    section[data-testid="stSidebar"] { background-color: #0F2C59 !important; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #FFFFFF !important; }
    .main-title { color: #0F2C59; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 800; margin-bottom: 5px; }
    .subtitle { color: #6C757D; font-size: 1.1rem; margin-bottom: 30px; }
    .ecoscore-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 20px; border-radius: 12px; text-align: center;
        font-size: 3rem; font-weight: bold; box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    div[data-testid="stMetricValue"] { color: #0F2C59 !important; font-weight: 700; }
    .stButton>button {
        background-color: #38ef7d !important; color: #0F2C59 !important; font-weight: bold !important;
        border: none !important; border-radius: 8px !important; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #11998e !important; color: white !important; transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME FONKSİYONU (SADELEŞTİRİLDİ) ---
@st.cache_resource
def load_models():
    models = {}
    paths = {
        'gas': 'models/xgboost_gas_model.pkl',
        'elec': 'models/xgboost_elec_model.pkl'
    }
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key] = joblib.load(f)
        else:
            models[key] = None
    return models

models = load_models()

# --- ANA EKRAN BAŞLIK ALANI ---
st.markdown("<h1 class='main-title'>🌱 EcoScore</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Dentaş Kağıt — Sipariş Bazlı Ürün Karbon Ayak İzi ve Enerji Optimizasyonu Tahmin Motoru</p>", unsafe_allow_html=True)
st.markdown("---")

# --- YAN MENÜ (SIDEBAR) - GİRDİ FORMU ---
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-ecology-environmentalism-flatart-icons-flat-flatarticons.png", width=80)
    st.markdown("### 📋 Sipariş Parametreleri")
    
    product_type = st.selectbox(
        "Ürün Tipi",
        ["Oluklu Mukavva", "Viyol (Yumurta Kasası)", "Endüstriyel Viyol", "Kutu / Ambalaj"]
    )
    
    order_quantity = st.number_input("Sipariş Miktarı (Adet veya Toplam m²)*", min_value=1, value=10000, step=500)
    target_weight = st.number_input("Hedef Gramaj (gr/m²)", min_value=10, value=120, step=5)
    
    st.markdown("### ⚙️ Proses Değişkenleri")
    recycled_ratio = st.slider("Geri Dönüştürülmüş Kağıt Oranı", 0.70, 1.00, 0.85, step=0.01)
    moisture_content = st.slider("Hammadde Nem Oranı", 0.05, 0.20, 0.08, step=0.01)
    oven_temp = st.slider("Hedef Fırın Sıcaklığı (°C)", 180, 240, 210, step=5)
    oven_speed = st.slider("Fırın Hızı (m/dk)", 15, 45, 30, step=1)
    
    st.caption("\*Not: Mevcut sürümde sipariş miktarının m² cinsinden olduğu varsayılmıştır.")
    st.markdown(" ")
    calculate_btn = st.button("🚀 HESAPLA VE OPTİMİZE ET")

# --- HESAPLAMA VE SONUÇ EKRANI ---
if calculate_btn:
    
    with st.spinner("ML Pipeline çalıştırılıyor, enerji tüketimi tahmin ediliyor..."):
        try:
            # Sadece eğitilmiş gas ve elec pipeline modelleri kontrol ediliyor
            if models['gas'] is not None and models['elec'] is not None:
                
                # Sütun isimleri model_train.py ile tam uyumlu DataFrame yapısı
                input_df = pd.DataFrame([{
                    "Urun_Tipi": product_type,
                    "Siparis_Adedi": order_quantity,
                    "Hedef_Gramaj_g": target_weight,
                    "Geri_Donusumlu_Kagit_Orani": recycled_ratio,
                    "Hammadde_Nem_Orani": moisture_content,
                    "Firin_Hizi_m_dk": oven_speed,
                    "Firin_Sicakligi_C": oven_temp,
                }])
                
                # Pipeline kendi içinde dönüşümleri yapıp tahmini üretiyor
                predicted_gas = max(0.0, float(models["gas"].predict(input_df)[0]))
                predicted_elec = max(0.0, float(models["elec"].predict(input_df)[0]))
                
            else:
                raise FileNotFoundError("Model dosyaları yüklenemedi. Klasör yollarını kontrol edin.")
                
        except Exception as e:
            # Gerçek hatayı takip edebilmek için st.error(e) eklendi
               st.error(f"🚨 Pipeline Hatası Yakalandı: {e}")
               st.warning("⚠️ Uygulama simülasyon (demo) modunda çalıştırılıyor.")
            
            # Fallback / Mockup Hesaplama
            # Pipeline tahminleri
               gas_raw = float(models["gas"].predict(input_df)[0])
               elec_raw = float(models["elec"].predict(input_df)[0])

    # Geçici kontrol
               st.write("Ham Gaz Tahmini:", gas_raw)
               st.write("Ham Elektrik Tahmini:", elec_raw)

               predicted_gas = max(0.0, gas_raw)
               predicted_elec = max(0.0, elec_raw)
    
    # Karbon Ayak İzi ve Tonaj Tabanlı Hesaplama
    total_co2 = calculate_carbon_footprint(predicted_gas, predicted_elec)
    
    product_tonnage = (order_quantity * target_weight) / 1_000_000
    if product_tonnage <= 0:
        product_tonnage = 0.001
        
    co2_per_ton = total_co2 / product_tonnage
    ecoscore_letter = get_ecoscore_grade(co2_per_ton)
    
    # Yeşil Senaryo
    green_gas = predicted_gas * 0.95
    green_elec = predicted_elec * 0.95
    green_co2 = calculate_carbon_footprint(green_gas, green_elec)

    # --- METRİKLER ---
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
    with col1:
        st.metric(label="🔥 Tahmini Doğalgaz Tüketimi", value=f"{predicted_gas:,.2f} m³")
    with col2:
        st.metric(label="⚡ Tahmini Elektrik Tüketimi", value=f"{predicted_elec:,.2f} kWh")
    with col3:
        st.metric(label="🌍 Toplam Karbon Ayak İzi", value=f"{total_co2:,.2f} kg CO2e")
    with col4:
        st.markdown("<p style='font-size:0.9rem; font-weight:bold; color:#6C757D; margin-bottom:5px;'>EcoScore Grade</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='ecoscore-badge'>{ecoscore_letter}</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- GRAFİK VE AI BÖLÜMÜ ---
    graph_col, ai_col = st.columns([1, 1])
    
    with graph_col:
        st.markdown("### 📊 Emisyon Karşılaştırma Analizi")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Mevcut Sipariş Reçetesi', x=['Karbon Emisyonu'], y=[total_co2], marker_color='#0F2C59'))
        fig.add_trace(go.Bar(name='AI Optimizasyonlu Yeşil Senaryo', x=['Karbon Emisyonu'], y=[green_co2], marker_color='#38ef7d'))
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350,
            margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with ai_col:
        st.markdown("### 🤖 Gemini AI Yeşil Reçete Önerileri")
        
        agent_inputs = {
            "product_type": product_type, "quantity": order_quantity, "recycled_ratio": recycled_ratio,
            "moisture": moisture_content, "temp": oven_temp, "speed": oven_speed,
            "calculated_co2": total_co2, "co2_per_ton": co2_per_ton
        }
        
        with st.spinner("AI Ajanı reçeteyi analiz ediyor ve yeşil reçete önerileri hazırlıyor..."):
          ai_recommendations = generate_green_recommendation(
          siparis_detaylari=agent_inputs,
          mevcut_co2_kg=total_co2,
          ecoscore_harfi=ecoscore_letter,
)

            
        st.success(ai_recommendations)

else:
    st.info("💡 Projenin emisyon tahminlerini ve AI optimizasyon önerilerini görmek için sol taraftaki panelden girdileri ayarlayıp **'Hesapla ve Optimize Et'** butonuna basın.")