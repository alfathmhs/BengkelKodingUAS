import streamlit as st
import pandas as pd
import joblib

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

CHURN_THRESHOLD = 0.70
   
def interpret_risk(prob):
    if prob >= 0.70:
        return "RISIKO TINGGI"
    elif prob >= 0.40:
        return "RISIKO MENENGAH"
    else:
        return "RISIKO RENDAH"

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except Exception:
        st.error("❌ Model tidak dapat dimuat. Silakan hubungi administrator sistem.")
        st.stop()

model = load_model()

# ===============================
# UI HEADER
# ===============================
st.title("📉 Telco Customer Churn Prediction")
st.markdown(
    """
    Aplikasi ini memprediksi **potensi churn pelanggan** menggunakan  **Random Forest Classifier**  
    yang telah melalui *feature engineering* dan *hyperparameter tuning*.
    """
)

st.divider()

# ===============================
# Dictionary
# ===============================
feature_desc = {
    "Gender": "Jenis kelamin pelanggan: Male / Female",
    "SeniorCitizen": "Apakah pelanggan adalah warga senior (1 = Ya, 0 = Tidak)",
    "Partner": "Apakah pelanggan memiliki pasangan",
    "Dependents": "Apakah pelanggan memiliki tanggungan",
    "Tenure": "Lama pelanggan menggunakan layanan (bulan)",
    "PhoneService": "Apakah pelanggan menggunakan layanan telepon",
    "MultipleLines": "Apakah pelanggan memiliki beberapa jalur telepon",
    "InternetService": "Jenis layanan internet (DSL, Fiber optic, No)",
    "OnlineSecurity": "Apakah pelanggan memiliki keamanan online",
    "OnlineBackup": "Apakah pelanggan menggunakan backup online",
    "DeviceProtection": "Apakah perangkat pelanggan dilindungi",
    "TechSupport": "Layanan dukungan teknis",
    "StreamingTV": "Apakah pelanggan menggunakan layanan Streaming TV",
    "StreamingMovies": "Apakah pelanggan menggunakan layanan Streaming Movies",
    "Contract": "Jenis kontrak pelanggan (Month-to-month, One year, Two year)",
    "PaperlessBilling": "Apakah pelanggan menggunakan tagihan tanpa kertas",
    "PaymentMethod": "Metode pembayaran",
    "MonthlyCharges": "Biaya bulanan yang dibayarkan pelanggan",
}

selected_feature = st.selectbox("Pilih fitur untuk melihat deskripsinya", list(feature_desc.keys()))
st.info(feature_desc[selected_feature])

st.divider()

# ===============================
# INPUT FORM
# ===============================
st.subheader("🔧 Data Pelanggan")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
        
    st.divider()
    
    submit = st.form_submit_button(
        label="🔍 Prediksi Churn",
        help="Klik untuk prediksi",
    )

# ===============================
# PREDICTION
# ===============================
if submit:
    # Feature Engineering
    total_charges = tenure * monthly_charges
    risk_score = int(contract == "Month-to-month") + \
                 int(tech_support == "No") + \
                 int(online_security == "No")

    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "RiskScore" : risk_score
    }])

    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= CHURN_THRESHOLD)
    risk_level = interpret_risk(proba)

    st.divider()

    st.subheader("📊 Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ Pelanggan **BERPOTENSI CHURN**")
    else:
        st.success(f"✅ Pelanggan **TIDAK CHURN**")

    st.metric(
    label="Probabilitas Churn",
    value=f"{proba:.2%}"
    )
    st.progress(min(proba, 1.0))

    st.info(
    f"""
    **Interpretasi Risiko:** {risk_level}  
    - ≥ 70% → Risiko churn tinggi  
    - 40% – 69% → Risiko churn menengah  
    - < 40% → Risiko churn rendah  
    """
    )

    st.caption(
    "📌 Catatan: Pelanggan dikategorikan churn jika probabilitas ≥ 70%."
    )