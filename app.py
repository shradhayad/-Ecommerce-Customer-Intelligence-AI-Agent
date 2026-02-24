import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Agent", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
    }
    .success-box {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .warning-box {
        background: linear-gradient(90deg, #f2994a 0%, #f2c94c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .error-box {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .info-box {
        background: linear-gradient(90deg, #2193b0 0%, #6dd5ed 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .metric-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .welcome-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🤖 Ecommerce Customer Intelligence AI Agent</h1>
    <p style="color: white; margin: 0;">Interactive ML Dashboard for Customer Predictions</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    df = pd.read_csv("ecommerce_customer_data_custom_ratios.csv.zip")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    snapshot_date = df['purchase_date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (snapshot_date - x.max()).days,
        'customer_id': 'count',
        'total_purchase_amount': 'sum'
    }).rename(columns={
        'purchase_date': 'recency',
        'customer_id': 'frequency',
        'total_purchase_amount': 'monetary'
    }).reset_index()

    rfm['next_purchase_days'] = rfm['recency']
    X = rfm[['recency', 'frequency', 'monetary']]

    model_time = RandomForestRegressor(n_estimators=50, random_state=42)
    model_time.fit(X, rfm['next_purchase_days'])

    rfm['churn'] = np.where(rfm['recency'] > 30, 1, 0)
    model_churn = RandomForestClassifier(n_estimators=50, random_state=42)
    model_churn.fit(X, rfm['churn'])

    top_product = df.groupby('customer_id')['product_category'] \
                    .agg(lambda x: x.value_counts().index[0]).reset_index()

    rfm = rfm.merge(top_product, on='customer_id')

    le = LabelEncoder()
    rfm['product_category'] = le.fit_transform(rfm['product_category'])

    model_product = RandomForestClassifier(n_estimators=50, random_state=42)
    model_product.fit(X, rfm['product_category'])

    return df, rfm, model_time, model_churn, model_product, le

df, rfm, model_time, model_churn, model_product, le = load_models()

st.markdown("## 🔮 AI-Powered Customer Prediction")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📝 Customer Input")
    recency = st.slider("🔄 Recency (days)", 0, 100, 10)
    frequency = st.number_input("📦 Frequency", 0, 50, 5)
    monetary = st.number_input("💵 Total Spending ($)", 0.0, 5000.0, 500.0, 50.0)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Presets")
    preset = st.selectbox("Customer Type", ["Custom", "🆕 New Customer", "⭐ Loyal Customer", "⚠️ At-Risk Customer", "💎 High-Value Customer"])
    
    if preset == "🆕 New Customer":
        recency, frequency, monetary = 5, 1, 100.0
    elif preset == "⭐ Loyal Customer":
        recency, frequency, monetary = 10, 25, 2000.0
    elif preset == "⚠️ At-Risk Customer":
        recency, frequency, monetary = 60, 10, 800.0
    elif preset == "💎 High-Value Customer":
        recency, frequency, monetary = 15, 30, 3500.0
    
    if preset != "Custom":
        recency = st.slider("🔄 Recency", 0, 100, recency)
        frequency = st.number_input("📦 Frequency", 0, 50, frequency)
        monetary = st.number_input("💵 Total Spending", 0.0, 5000.0, float(monetary), 50.0)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Run AI Analysis", type="primary")

with col2:
    if analyze_btn:
        input_data = pd.DataFrame([{"recency": recency, "frequency": frequency, "monetary": monetary}])
        
        next_purchase = int(model_time.predict(input_data)[0])
        churn_risk = float(model_churn.predict_proba(input_data)[0][1])
        product = le.inverse_transform(model_product.predict(input_data))[0]
        
        st.markdown('<div class="success-box">✅ <b>Analysis Complete!</b></div>', unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        
        m1.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #667eea, #764ba2);">
            <h3 style="margin: 0;">⏳ Next Purchase</h3>
            <h2 style="margin: 0;">{next_purchase} days</h2>
        </div>
        """, unsafe_allow_html=True)
        
        m2.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #f093fb, #f5576c);">
            <h3 style="margin: 0;">⚠️ Churn Risk</h3>
            <h2 style="margin: 0;">{churn_risk * 100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        m3.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #4facfe, #00f2fe);">
            <h3 style="margin: 0;">🛒 Likely Product</h3>
            <h2 style="margin: 0; font-size: 18px;">{product}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🤖 AI Recommendations")
        
        if churn_risk > 0.75:
            st.markdown("""
            <div class="error-box">
                <h4>🔴 HIGH RISK - Immediate Action Required</h4>
                <p>• Send 25% discount code via email</p>
                <p>• Trigger retention campaign</p>
                <p>• Personal outreach from customer success team</p>
                <p>• Offer free shipping on next order</p>
            </div>
            """, unsafe_allow_html=True)
        elif churn_risk > 0.5:
            st.markdown("""
            <div class="warning-box">
                <h4>🟠 MEDIUM RISK - Proactive Engagement</h4>
                <p>• Send personalized product offer</p>
                <p>• Share trending products in category</p>
                <p>• Invite to loyalty program</p>
                <p>• Send cart abandonment reminder</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>🟢 LOW RISK - Nurture & Upsell</h4>
                <p>• Recommend trending products</p>
                <p>• Highlight new arrivals</p>
                <p>• Suggest complementary products</p>
                <p>• Encourage reviews/referrals</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📉 Churn Risk Gauge")
        risk_value = int(churn_risk * 100)
        if risk_value < 50:
            color = "#38ef7d"
        elif risk_value < 75:
            color = "#f2c94c"
        else:
            color = "#ef473a"
        st.markdown(f"""
        <div style="background: #f0f0f0; border-radius: 10px; padding: 10px;">
            <div style="background: {color}; width: {risk_value}%; height: 30px; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
                {risk_value}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="welcome-box">
            👈 Adjust customer details and click <b>'Run AI Analysis'</b> to get predictions!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Sample Customer Profiles")
        st.table(pd.DataFrame({
            'Type': ['🆕 New', '⭐ Loyal', '⚠️ At-Risk', '💎 High-Value'],
            'Recency': [5, 10, 60, 15],
            'Frequency': [1, 25, 10, 30],
            'Spend ($)': [100, 2000, 800, 3500],
            'Risk': ['15%', '20%', '85%', '10%']
        }))