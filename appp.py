# ==========================================================
# HYBRID SOFT COMPUTING SMART AGRICULTURE MANAGEMENT SYSTEM
# Version 3.0
# Developed by Monisha Mariappan
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Smart Agriculture Management System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# THEME STATE (must be set before CSS is injected)
# -------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -------------------------------------------------
# CUSTOM CSS (Light / Dark)
# -------------------------------------------------

if st.session_state.dark_mode:

    st.markdown("""
    <style>

    body, .main, .stApp{
        background:#121212;
        color:#e6e6e6;
    }

    h1,h2,h3{
        color:#4CD787;
    }

    .block-container{
        padding-top:1rem;
    }

    .stButton>button{
        width:100%;
        height:45px;
        border-radius:10px;
        background:#2E8B57;
        color:white;
        font-size:16px;
        border:none;
    }

    .stButton>button:hover{
        background:#3fae74;
    }

    .metric-card{
        background:#1e1e1e;
        color:#e6e6e6;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,.5);
    }

    [data-testid="stSidebar"]{
        background-color:#1a1a1a;
    }

    [data-testid="stMetricValue"]{
        color:#4CD787;
    }

    [data-testid="stMetricLabel"]{
        color:#cccccc;
    }

    .stDataFrame, .stTable{
        background:#1e1e1e;
    }

    .footer{
        text-align:center;
        color:#999999;
        font-size:14px;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    body{
        background:#f5f7fa;
    }

    .main{
        background:#f5f7fa;
    }

    h1,h2,h3{
        color:#2E8B57;
    }

    .block-container{
        padding-top:1rem;
    }

    .stButton>button{
        width:100%;
        height:45px;
        border-radius:10px;
        background:#2E8B57;
        color:white;
        font-size:16px;
        border:none;
    }

    .stButton>button:hover{
        background:#1f6b44;
    }

    .metric-card{
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,.15);
    }

    .footer{
        text-align:center;
        color:gray;
        font-size:14px;
    }

    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# ENSURE REQUIRED FOLDERS EXIST
# -------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("history", exist_ok=True)
os.makedirs("auth", exist_ok=True)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv("data/Crop_recommendation.csv")

df = load_dataset()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_models():
    crop_model = joblib.load("models/crop_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    q_table = joblib.load("models/q_table.pkl")

    # -------------------------------------------------
    # ARTIFICIAL NEURAL NETWORK (ANN)
    # -------------------------------------------------
    ann_path = "models/ann_crop_model.pkl"
    X_ann = df.drop("label", axis=1)
    y_ann = encoder.transform(df["label"])
    X_ann_scaled = scaler.transform(X_ann)

    if os.path.exists(ann_path):
        ann_model = joblib.load(ann_path)
        ann_accuracy = accuracy_score(y_ann, ann_model.predict(X_ann_scaled))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_ann_scaled, y_ann, test_size=0.20, random_state=42, stratify=y_ann
        )
        ann_model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.10
        )
        ann_model.fit(X_train, y_train)
        ann_accuracy = accuracy_score(y_test, ann_model.predict(X_test))
        joblib.dump(ann_model, ann_path)

    return crop_model, encoder, scaler, kmeans, q_table, ann_model, ann_accuracy

crop_model, label_encoder, scaler, kmeans_model, q_table, ann_model, ann_accuracy = load_models()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

HISTORY_FILE = "history/predictions.csv"
USERS_FILE = "auth/users.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "Date", "N", "P", "K", "Temperature", "Humidity",
        "pH", "Rainfall", "Crop", "Cluster", "Irrigation"
    ]).to_csv(HISTORY_FILE, index=False)

# -------------------------------------------------
# AUTH HELPER FUNCTIONS
# -------------------------------------------------

def load_users():
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)
    return pd.read_csv(USERS_FILE, dtype=str)

def save_user(username, password):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    new_user = pd.DataFrame({"username": [username], "password": [password]})
    new_user.to_csv(USERS_FILE, mode="a", header=False, index=False)

def username_exists(username):
    users = load_users()
    if len(users) == 0:
        return False
    return username.strip().lower() in users["username"].astype(str).str.strip().str.lower().values

def validate_login(username, password):
    users = load_users()
    if len(users) == 0:
        return False
    users["username"] = users["username"].astype(str).str.strip()
    users["password"] = users["password"].astype(str).str.strip()
    match = users[
        (users["username"].str.lower() == username.strip().lower()) &
        (users["password"] == password.strip())
    ]
    return len(match) > 0

# -------------------------------------------------
# LOGIN / REGISTER PAGE
# -------------------------------------------------

if not st.session_state.logged_in:

    st.title("🌱 Smart Agriculture Management System")

    auth_mode = st.radio(
        "Choose an option",
        ["Login", "Register"],
        horizontal=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        if auth_mode == "Login":

            st.markdown("### Login")

            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login"):

                if username.strip() == "" or password.strip() == "":
                    st.error("Please enter both username and password.")
                elif validate_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.success("Login Successful")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password. Don't have an account? Register above.")

        else:

            st.markdown("### Register")

            new_username = st.text_input("Choose a Username", key="reg_user")
            new_password = st.text_input("Choose a Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")

            if st.button("Register"):

                if new_username.strip() == "" or new_password.strip() == "":
                    st.error("Username and password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif username_exists(new_username):
                    st.error("Username already exists. Please choose another.")
                else:
                    save_user(new_username.strip(), new_password.strip())
                    st.success("Registration successful! Please switch to Login.")

    st.stop()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("🌱 Smart Agriculture")

st.sidebar.write(f"👤 Logged in as: **{st.session_state.username}**")

st.sidebar.markdown("---")

dark_toggle = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)

if dark_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_toggle
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🌾 Crop Prediction",
        "🧠 Neural Network",
        "📊 Farm Segmentation",
        "🔷 Fuzzy Irrigation",
        "💧 Smart Irrigation",
        "🔀 Hybrid Soft Computing",
        "📈 Analytics",
        "📜 Prediction History",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("AI Powered Agriculture")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.kpi_animated = False
    st.rerun()

# ==========================================================
# DASHBOARD HELPER FUNCTIONS
# ==========================================================

def make_gauge(value, title, max_value=100, color="#2E8B57"):
    """Build a Plotly gauge indicator chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%"},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "#fde2e2"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "#fff3cd"},
                {'range': [max_value * 0.8, max_value], 'color': "#d4edda"}
            ]
        }
    ))

    if st.session_state.dark_mode:
        fig.update_layout(
            height=270,
            margin=dict(t=50, b=10, l=20, r=20),
            paper_bgcolor="#1e1e1e",
            font={'color': "#e6e6e6"}
        )
    else:
        fig.update_layout(
            height=270,
            margin=dict(t=50, b=10, l=20, r=20)
        )

    return fig


def animated_metric(placeholder, label, target_value, suffix="", decimals=0, steps=20, delay=0.02):
    """Animate an st.metric counting up from 0 to target_value."""
    for i in range(steps + 1):
        progress = i / steps
        if decimals == 0:
            current = int(round(target_value * progress))
            placeholder.metric(label, f"{current}{suffix}")
        else:
            current = round(target_value * progress, decimals)
            placeholder.metric(label, f"{current}{suffix}")
        time.sleep(delay)

    final = round(target_value, decimals) if decimals else int(target_value)
    placeholder.metric(label, f"{final}{suffix}")


# ==========================================================
# 🔷 FUZZY LOGIC IRRIGATION
# ==========================================================

def left_shoulder(x, a, b):
    if x <= a: return 1.0
    if x >= b: return 0.0
    return (b-x)/(b-a)

def right_shoulder(x, a, b):
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x-a)/(b-a)

def triangle(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x-a)/(b-a) if x < b else (c-x)/(c-b)

def fuzzy_irrigation(moisture, temperature, humidity):
    ml = left_shoulder(moisture, 25, 45)
    mm = triangle(moisture, 30, 50, 70)
    mh = right_shoulder(moisture, 55, 75)
    tl = left_shoulder(temperature, 15, 25)
    tm = triangle(temperature, 20, 30, 40)
    th = right_shoulder(temperature, 35, 45)
    hl = left_shoulder(humidity, 30, 50)
    hm = triangle(humidity, 40, 60, 80)
    hh = right_shoulder(humidity, 70, 90)

    high1 = min(ml, th, hl)
    high2 = min(ml, max(tm, th))
    med1 = min(mm, tm)
    med2 = min(ml, hh)
    med3 = min(mm, hl)
    low1 = mh
    low2 = min(hh, max(tl, tm))

    low = max(low1, low2)
    medium = max(med1, med2, med3)
    high = max(high1, high2)
    total = low + medium + high
    score = 50 if total == 0 else (low*20 + medium*55 + high*90)/total

    if score < 35:
        level, action = "Low", "Decrease / maintain irrigation"
    elif score < 70:
        level, action = "Medium", "Maintain moderate irrigation"
    else:
        level, action = "High", "Increase irrigation"

    memberships = {"Moisture Low":ml,"Moisture Medium":mm,"Moisture High":mh,"Temperature Low":tl,"Temperature Medium":tm,"Temperature High":th,"Humidity Low":hl,"Humidity Medium":hm,"Humidity High":hh}
    rules = {"High: Low moisture + High temperature + Low humidity":high1,"High: Low moisture + Medium/High temperature":high2,"Medium: Medium moisture + Medium temperature":med1,"Medium: Low moisture + High humidity":med2,"Medium: Medium moisture + Low humidity":med3,"Low: High moisture":low1,"Low: High humidity + Low/Medium temperature":low2}
    return score, level, action, memberships, rules


# ==========================================================
# 🏠 DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.title("🌱 Smart Agriculture Management System")

    st.markdown("""
### Welcome to the AI Powered Smart Agriculture Dashboard

This application combines **Machine Learning**, **Neural Networks**, **Fuzzy Logic** and **Reinforcement Learning**
to provide intelligent agricultural decisions.

The extended system is presented as a **Hybrid Soft Computing Smart Agriculture System**.
""")

    st.markdown("---")

    total_records = len(df)
    total_features = len(df.columns) - 1
    total_crops = df["label"].nunique()
    accuracy = 99.32

    col1, col2, col3, col4 = st.columns(4)

    p1 = col1.empty()
    p2 = col2.empty()
    p3 = col3.empty()
    p4 = col4.empty()

    if not st.session_state.get("kpi_animated", False):
        animated_metric(p1, "📄 Total Records", total_records, steps=25, delay=0.015)
        animated_metric(p2, "🌾 Crop Types", total_crops, steps=15, delay=0.02)
        animated_metric(p3, "📊 Features", total_features, steps=10, delay=0.02)
        animated_metric(p4, "🎯 Model Accuracy", accuracy, suffix="%", decimals=2, steps=25, delay=0.015)
        st.session_state.kpi_animated = True
    else:
        p1.metric("📄 Total Records", total_records)
        p2.metric("🌾 Crop Types", total_crops)
        p3.metric("📊 Features", total_features)
        p4.metric("🎯 Model Accuracy", f"{accuracy}%")

    st.markdown("---")

    st.subheader("📟 Performance Gauges")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.plotly_chart(
            make_gauge(accuracy, "Random Forest Accuracy"),
            use_container_width=True
        )

    with g2:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = round((1 - missing_cells / total_cells) * 100, 2) if total_cells else 0
        st.plotly_chart(
            make_gauge(completeness, "Data Completeness", color="#1f77b4"),
            use_container_width=True
        )

    with g3:
        crop_counts_all = df["label"].value_counts()
        balance = round((crop_counts_all.min() / crop_counts_all.max()) * 100, 2)
        st.plotly_chart(
            make_gauge(balance, "Crop Class Balance", color="#e67e22"),
            use_container_width=True
        )

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📌 Project Overview")

        st.info("""
This project integrates Machine Learning and Soft Computing techniques.

🌳 Supervised Learning
- Random Forest → Crop Prediction

📊 Unsupervised Learning
- K-Means → Farm Segmentation

🧠 Neural Network
- MLP / ANN → Crop Prediction

🔷 Fuzzy Logic
- Fuzzy Irrigation → Irrigation Level

🤖 Reinforcement Learning
- Q-Learning → Smart Irrigation
""")

    with right:

        st.subheader("🤖 Models Used")

        models = pd.DataFrame({
            "Algorithm": ["Random Forest", "K-Means", "MLP Neural Network", "Fuzzy Logic", "Q-Learning"],
            "Type": ["Supervised", "Unsupervised", "Neural Network", "Fuzzy Logic", "Reinforcement"]
        })

        st.dataframe(models, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("🌾 Crop Distribution")

    crop_count = df["label"].value_counts().reset_index()
    crop_count.columns = ["Crop", "Count"]

    fig = px.bar(crop_count, x="Crop", y="Count", color="Count", text="Count")
    fig.update_layout(height=500, xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Crop Percentage")
        fig = px.pie(crop_count, names="Crop", values="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌡 Temperature vs Humidity")
        fig = px.scatter(df, x="temperature", y="humidity", color="label", hover_name="label")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")

    statistics = pd.DataFrame({
        "Statistic": [
            "Maximum Temperature", "Minimum Temperature", "Average Temperature",
            "Maximum Rainfall", "Average Rainfall", "Average Humidity", "Average pH"
        ],
        "Value": [
            round(df["temperature"].max(), 2),
            round(df["temperature"].min(), 2),
            round(df["temperature"].mean(), 2),
            round(df["rainfall"].max(), 2),
            round(df["rainfall"].mean(), 2),
            round(df["humidity"].mean(), 2),
            round(df["ph"].mean(), 2)
        ]
    })

    st.dataframe(statistics, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Feature Distribution")

    feature = st.selectbox(
        "Choose Feature",
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    fig = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=["green"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("⚡ Quick Navigation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🌾 Crop Prediction")
        st.write("Predict the best crop based on soil conditions.")

    with col2:
        st.info("📊 Farm Segmentation")
        st.write("Cluster farms into similar productivity groups.")

    with col3:
        st.warning("💧 Smart Irrigation")
        st.write("Recommend the best irrigation action using Q-Learning.")

    st.markdown("---")

    st.success("✅ Dashboard Loaded Successfully")

# ==========================================================
# 🌾 CROP PREDICTION
# ==========================================================

elif page == "🌾 Crop Prediction":

    st.title("🌾 AI Crop Recommendation System")

    st.write("Enter the soil nutrients and environmental conditions to predict the most suitable crop.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 150.0, 90.0)
        P = st.number_input("Phosphorus (P)", 0.0, 150.0, 42.0)
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)

    with col2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

    st.markdown("")

    if st.button("🌱 Predict Best Crop"):

        sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = crop_model.predict(sample)
        crop = label_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Recommended Crop: {crop.upper()}")

        probability = None

        try:
            probability = crop_model.predict_proba(sample)
            confidence = np.max(probability) * 100
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
        except Exception:
            st.info("Confidence score unavailable.")

        st.markdown("---")

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Feature": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [N, P, K, temperature, humidity, ph, rainfall]
        })

        st.dataframe(summary, use_container_width=True)

        history_row = pd.DataFrame({
            "Date": [datetime.now()],
            "N": [N], "P": [P], "K": [K],
            "Temperature": [temperature], "Humidity": [humidity],
            "pH": [ph], "Rainfall": [rainfall],
            "Crop": [crop], "Cluster": ["-"], "Irrigation": ["-"]
        })

        history_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

        st.success("Prediction saved successfully.")

        st.markdown("---")

        st.subheader("Feature Importance")

        try:
            importance = crop_model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": df.drop("label", axis=1).columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", text="Importance")

            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Feature importance unavailable.")

        st.markdown("---")

        if probability is not None:
            try:
                prob_df = pd.DataFrame({
                    "Crop": label_encoder.classes_,
                    "Probability": probability[0]
                }).sort_values("Probability", ascending=False).head(10)

                fig = px.bar(prob_df, x="Crop", y="Probability", color="Probability",
                             text="Probability", title="Top Prediction Probabilities")

                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.markdown("---")

        result = pd.DataFrame({
            "Feature": ["Nitrogen", "Phosphorus", "Potassium", "Temperature",
                        "Humidity", "pH", "Rainfall", "Predicted Crop"],
            "Value": [N, P, K, temperature, humidity, ph, rainfall, crop]
        })

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button("📥 Download Prediction", csv, file_name="crop_prediction.csv", mime="text/csv")

        st.markdown("---")

        st.info(f"""
### Recommendation

The Random Forest model recommends **{crop.upper()}** for the given
soil and weather conditions.

✔ High prediction accuracy

✔ Suitable nutrient combination

✔ Environmental conditions match this crop
""")

# ==========================================================
# 🧠 NEURAL NETWORK CROP PREDICTION
# ==========================================================

elif page == "🧠 Neural Network":

    st.title("🧠 Artificial Neural Network Crop Prediction")
    st.write("""
    This module uses an **Artificial Neural Network (ANN)** based on an
    MLP classifier to predict the most suitable crop from soil and
    environmental conditions.
    """)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        ann_N = st.number_input("Nitrogen (N)", 0.0, 150.0, 90.0, key="ann_n")
        ann_P = st.number_input("Phosphorus (P)", 0.0, 150.0, 42.0, key="ann_p")
        ann_K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0, key="ann_k")
        ann_temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="ann_temp")
    with col2:
        ann_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0, key="ann_humidity")
        ann_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="ann_ph")
        ann_rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0, key="ann_rainfall")

    if st.button("🧠 Predict Crop using Neural Network", key="ann_predict"):
        sample = np.array([[ann_N, ann_P, ann_K, ann_temperature, ann_humidity, ann_ph, ann_rainfall]])
        sample_scaled = scaler.transform(sample)
        prediction = ann_model.predict(sample_scaled)
        crop = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🌱 ANN Recommended Crop: **{crop.upper()}**")
        try:
            confidence = np.max(ann_model.predict_proba(sample_scaled)) * 100
            st.metric("ANN Prediction Confidence", f"{confidence:.2f}%")
        except Exception:
            st.info("ANN confidence score unavailable.")

        info = pd.DataFrame({
            "Parameter": ["Model", "Architecture", "Activation", "Optimizer", "Input Features", "Output", "Accuracy"],
            "Value": ["MLPClassifier", "64 → 32 hidden neurons", "ReLU", "Adam", "7", "Crop Class", f"{ann_accuracy*100:.2f}%"]
        })
        st.dataframe(info, use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Neural Network Workflow")
        st.code("""Agriculture Inputs
      ↓
Feature Scaling
      ↓
Input Layer
      ↓
Hidden Layer 1: 64 Neurons
      ↓
Hidden Layer 2: 32 Neurons
      ↓
Output Layer
      ↓
Recommended Crop
""")

# ==========================================================
# 📜 PREDICTION HISTORY
# ==========================================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    st.write("View, search, filter and download all previous crop predictions.")

    st.markdown("---")

    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        st.warning("No prediction history found.")
    else:

        history = pd.read_csv(HISTORY_FILE)

        if len(history) == 0:
            st.warning("No prediction history found.")
        else:

            history["Date"] = pd.to_datetime(history["Date"])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Predictions", len(history))

            with col2:
                st.metric("Unique Crops", history["Crop"].nunique())

            with col3:
                latest = history.iloc[-1]["Crop"]
                st.metric("Latest Prediction", latest)

            st.markdown("---")

            crop_search = st.text_input("🔍 Search Crop")

            if crop_search != "":
                history = history[history["Crop"].astype(str).str.contains(crop_search, case=False)]

            st.subheader("Filter by Date")

            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input("From", history["Date"].min())

            with col2:
                end_date = st.date_input("To", history["Date"].max())

            history = history[
                (history["Date"] >= pd.to_datetime(start_date)) &
                (history["Date"] <= pd.to_datetime(end_date))
            ]

            st.markdown("---")

            st.subheader("Prediction Records")

            st.dataframe(history, use_container_width=True)

            st.markdown("---")

            st.subheader("Most Predicted Crops")

            crop_count = history["Crop"].value_counts().reset_index()
            crop_count.columns = ["Crop", "Count"]

            fig = px.bar(crop_count, x="Crop", y="Count", color="Count", text="Count")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.subheader("Prediction Percentage")

            fig = px.pie(crop_count, names="Crop", values="Count")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.subheader("Latest Prediction Details")

            st.dataframe(history.tail(1), use_container_width=True)

            st.markdown("---")

            csv = history.to_csv(index=False).encode("utf-8")

            st.download_button("📥 Download Prediction History", csv,
                               file_name="Prediction_History.csv", mime="text/csv")

            st.markdown("---")

            if st.button("🗑 Delete Entire History"):
                header = pd.DataFrame(columns=[
                    "Date", "N", "P", "K", "Temperature", "Humidity",
                    "pH", "Rainfall", "Crop", "Cluster", "Irrigation"
                ])
                header.to_csv(HISTORY_FILE, index=False)
                st.success("Prediction history cleared successfully.")
                st.rerun()

            st.markdown("---")

            st.success("Prediction history loaded successfully.")

# ==========================================================
# 📊 FARM SEGMENTATION
# ==========================================================

elif page == "📊 Farm Segmentation":

    st.title("📊 AI Farm Segmentation")

    st.write("""
This module groups farms into different clusters based on
soil nutrients and environmental conditions using
**K-Means Clustering**.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 150.0, 90.0, key="cluster_n")
        P = st.number_input("Phosphorus (P)", 0.0, 150.0, 42.0, key="cluster_p")
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0, key="cluster_k")
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="cluster_temp")

    with col2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0, key="cluster_humidity")
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="cluster_ph")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0, key="cluster_rain")

    st.markdown("")

    if st.button("📊 Predict Farm Cluster"):

        sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        sample_scaled = scaler.transform(sample)
        cluster = int(kmeans_model.predict(sample_scaled)[0])

        cluster_name = {
            0: "🌾 Medium Productivity Farm",
            1: "🌱 High Productivity Farm",
            2: "🌵 Low Productivity Farm"
        }

        st.success(f"Predicted Cluster : {cluster}")
        st.info(cluster_name.get(cluster, "Unknown Cluster"))

        st.markdown("---")

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Feature": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value": [N, P, K, temperature, humidity, ph, rainfall]
        })

        st.dataframe(summary, use_container_width=True)

        st.markdown("---")

        st.subheader("Recommendation")

        if cluster == 0:
            st.warning("""
Moderate soil fertility.

✔ Apply balanced fertilizer.

✔ Monitor irrigation.

✔ Improve organic matter.
""")
        elif cluster == 1:
            st.success("""
Excellent farm condition.

✔ Continue current practices.

✔ Suitable for high-yield crops.

✔ Maintain irrigation schedule.
""")
        else:
            st.error("""
Poor soil condition.

✔ Improve soil nutrients.

✔ Add compost.

✔ Optimize irrigation.
""")

        st.markdown("---")

        history_row = pd.DataFrame({
            "Date": [datetime.now()],
            "N": [N], "P": [P], "K": [K],
            "Temperature": [temperature], "Humidity": [humidity],
            "pH": [ph], "Rainfall": [rainfall],
            "Crop": ["-"], "Cluster": [cluster], "Irrigation": ["-"]
        })

        history_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

        st.success("Cluster prediction saved.")

    st.markdown("---")

    st.subheader("Cluster Distribution")

    X = df.drop("label", axis=1)
    X_scaled = scaler.transform(X)
    clusters = kmeans_model.predict(X_scaled)

    cluster_count = pd.Series(clusters).value_counts().sort_index().reset_index()
    cluster_count.columns = ["Cluster", "Count"]

    fig = px.bar(cluster_count, x="Cluster", y="Count", color="Cluster", text="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("PCA Cluster Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters.astype(str),
        "Crop": df["label"]
    })

    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", hover_data=["Crop"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Cluster Statistics")

    stats = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Meaning": ["Medium Productivity", "High Productivity", "Low Productivity"]
    })

    st.dataframe(stats, use_container_width=True)

    st.success("Farm segmentation completed successfully.")

elif page == "🔷 Fuzzy Irrigation":
    st.title("🔷 Fuzzy Logic Irrigation Controller")
    st.write("This module uses fuzzy membership functions and rules to determine irrigation level from soil moisture, temperature and humidity.")
    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1: moisture=st.slider("Soil Moisture (%)",0,100,35,key="fuzzy_moisture")
    with c2: temperature=st.slider("Temperature (°C)",0,60,30,key="fuzzy_temperature")
    with c3: humidity=st.slider("Humidity (%)",0,100,60,key="fuzzy_humidity")
    if st.button("🔷 Calculate Fuzzy Irrigation",key="fuzzy_calculate"):
        score,level,action,memberships,rules=fuzzy_irrigation(moisture,temperature,humidity)
        st.success(f"💧 Irrigation Level: **{level.upper()}**")
        st.metric("Fuzzy Irrigation Score",f"{score:.2f}%")
        st.info(f"Recommended Action: **{action}**")
        st.markdown("---")
        st.subheader("🔷 Membership Values")
        st.dataframe(pd.DataFrame({"Membership":list(memberships.keys()),"Value":[round(v,3) for v in memberships.values()]}),use_container_width=True)
        st.subheader("📜 Rule Strengths")
        st.dataframe(pd.DataFrame({"Rule":list(rules.keys()),"Strength":[round(v,3) for v in rules.values()]}),use_container_width=True)
        st.subheader("🧠 Fuzzy Logic Workflow")
        st.code("""Crisp Inputs
    ↓
Fuzzification
    ↓
Fuzzy Rules
    ↓
Rule Evaluation
    ↓
Aggregation
    ↓
Defuzzification
    ↓
Irrigation Level
""")

# ==========================================================
# 💧 SMART IRRIGATION (Q-LEARNING)
# ==========================================================

elif page == "💧 Smart Irrigation":

    st.title("💧 Smart Irrigation Recommendation")

    st.write("""
This module uses **Reinforcement Learning (Q-Learning)** to recommend
the best irrigation action based on the current soil moisture condition.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        soil_state = st.selectbox("Current Soil Condition", ["Dry", "Optimal", "Wet"])

    with col2:
        moisture = st.slider("Estimated Soil Moisture (%)", 0, 100, 50)

    st.markdown("---")

    state_map = {"Dry": 2, "Optimal": 1, "Wet": 0}
    actions = ["Increase Water 💦", "Maintain Current Level ✅", "Decrease Water 🚫"]

    if st.button("🚰 Get Irrigation Recommendation"):

        state = state_map[soil_state]
        best_action = np.argmax(q_table[state])
        recommendation = actions[best_action]

        st.success(f"Recommended Action : {recommendation}")

        st.markdown("---")

        history_row = pd.DataFrame({
            "Date": [datetime.now()],
            "N": ["-"], "P": ["-"], "K": ["-"],
            "Temperature": ["-"], "Humidity": [moisture],
            "pH": ["-"], "Rainfall": ["-"],
            "Crop": ["-"], "Cluster": ["-"], "Irrigation": [recommendation]
        })

        history_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

        st.success("Recommendation saved successfully.")

        st.markdown("---")

        st.subheader("Q-Table")

        q_df = pd.DataFrame(
            q_table,
            columns=["Increase Water", "Maintain", "Decrease Water"],
            index=["Dry", "Optimal", "Wet"]
        )

        st.dataframe(q_df, use_container_width=True)

        st.markdown("---")

        st.subheader("Action Scores")

        chart = pd.DataFrame({
            "Action": ["Increase Water", "Maintain", "Decrease Water"],
            "Q Value": q_table[state]
        })

        fig = px.bar(chart, x="Action", y="Q Value", color="Q Value", text="Q Value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Recommendation Details")

        if soil_state == "Dry":
            st.error("""
### Dry Soil

• Increase irrigation immediately.

• Water availability is low.

• Frequent irrigation is recommended.
""")
        elif soil_state == "Optimal":
            st.success("""
### Optimal Soil

• Maintain current irrigation.

• Soil moisture is balanced.

• Continue monitoring.
""")
        else:
            st.warning("""
### Wet Soil

• Reduce irrigation.

• Allow proper drainage.

• Avoid root damage.
""")

    st.markdown("---")

    st.subheader("Reinforcement Learning Workflow")

    workflow = pd.DataFrame({
        "Step": ["State", "Action", "Reward", "Policy Update"],
        "Description": ["Observe soil condition", "Choose irrigation", "Receive reward", "Update Q-Table"]
    })

    st.table(workflow)

    st.markdown("---")

    st.subheader("Q-Learning Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("States", "3")

    with col2:
        st.metric("Actions", "3")

    with col3:
        st.metric("Algorithm", "Q-Learning")

    st.markdown("---")

    st.subheader("Current Q-Values Heatmap")

    fig = go.Figure(
        data=go.Heatmap(
            z=q_table,
            x=["Increase", "Maintain", "Decrease"],
            y=["Dry", "Optimal", "Wet"],
            colorscale="Viridis"
        )
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.success("Smart Irrigation module completed successfully.")

# ==========================================================
# 🔀 HYBRID SOFT COMPUTING
# ==========================================================

elif page == "🔀 Hybrid Soft Computing":
    st.title("🔀 Hybrid Soft Computing Agriculture System")
    st.write("This module combines ANN, Fuzzy Logic, Random Forest, K-Means and Q-Learning in one intelligent agriculture application.")
    st.markdown("---")
    st.subheader("🌱 Hybrid Architecture")
    st.code("""Soil & Environmental Data
          |
   ┌──────┼──────────┐
   ↓      ↓          ↓
  ANN   Random     K-Means
   ↓    Forest        ↓
Crop Prediction   Farm Segmentation
   |
   └──────────┐
              ↓
         Fuzzy Logic
              ↓
       Irrigation Level
              ↓
         Q-Learning
              ↓
        Final Action
""")
    st.markdown("---")
    hybrid_df=pd.DataFrame({"Technique":["Artificial Neural Network","Fuzzy Logic","Random Forest","K-Means","Q-Learning"],"Purpose":["Crop prediction","Irrigation decision","Crop prediction","Farm segmentation","Irrigation action optimization"],"Domain":["Neural Network","Fuzzy Logic","Supervised ML","Unsupervised ML","Reinforcement Learning"]})
    st.dataframe(hybrid_df,use_container_width=True)
    st.markdown("---")
    st.metric("ANN Accuracy",f"{ann_accuracy*100:.2f}%")
    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1: hm=st.slider("Soil Moisture (%)",0,100,35,key="hybrid_moisture")
    with c2: ht=st.slider("Temperature (°C)",0,60,30,key="hybrid_temperature")
    with c3: hh=st.slider("Humidity (%)",0,100,60,key="hybrid_humidity")
    if st.button("🔀 Run Hybrid Irrigation Decision",key="hybrid_run"):
        score,level,action,_,_=fuzzy_irrigation(hm,ht,hh)
        state_map={"Low":2,"Medium":1,"High":0}
        state=state_map[level]
        q_actions=["Increase Water 💦","Maintain Current Level ✅","Decrease Water 🚫"]
        final_action=q_actions[int(np.argmax(q_table[state]))]
        st.success(f"Fuzzy Logic Result: **{level.upper()} irrigation**")
        st.metric("Fuzzy Score",f"{score:.2f}%")
        st.info(f"Fuzzy recommendation: **{action}**")
        st.success(f"Q-Learning optimized action: **{final_action}**")
        st.dataframe(pd.DataFrame({"Stage":["Fuzzy Logic","Fuzzy Score","Q-Learning","Final Action"],"Result":[level,f"{score:.2f}%","Best Q-value action",final_action]}),use_container_width=True)

# ==========================================================
# 📈 ANALYTICS DASHBOARD
# ==========================================================

elif page == "📈 Analytics":

    st.title("📈 Agriculture Analytics Dashboard")

    st.write("""
Analyze the agricultural dataset using interactive charts,
statistics, and visualizations.
""")

    st.markdown("---")

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Features", len(df.columns) - 1)

    with col3:
        st.metric("Crop Types", df["label"].nunique())

    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    st.subheader("📑 Descriptive Statistics")

    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    feature = st.selectbox(
        "Select Feature",
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    st.subheader("📊 Histogram")

    fig = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=["green"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📦 Box Plot")

    fig = px.box(df, y=feature, color="label")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🎻 Violin Plot")

    fig = px.violin(df, y=feature, color="label", box=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Scatter Plot")

    fig = px.scatter(df, x="temperature", y="humidity", color="label", hover_name="label")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🌾 Crop Distribution")

    crop_counts = df["label"].value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]

    fig = px.bar(crop_counts, x="Crop", y="Count", color="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🥧 Crop Percentage")

    fig = px.pie(crop_counts, names="Crop", values="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Average Feature Values")

    avg_df = pd.DataFrame({
        "Feature": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
        "Average": [
            df["N"].mean(), df["P"].mean(), df["K"].mean(),
            df["temperature"].mean(), df["humidity"].mean(),
            df["ph"].mean(), df["rainfall"].mean()
        ]
    })

    fig = px.bar(avg_df, x="Feature", y="Average", color="Average", text="Average")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Feature Summary")

    summary = pd.DataFrame({
        "Feature": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
        "Minimum": [
            df["N"].min(), df["P"].min(), df["K"].min(),
            df["temperature"].min(), df["humidity"].min(),
            df["ph"].min(), df["rainfall"].min()
        ],
        "Maximum": [
            df["N"].max(), df["P"].max(), df["K"].max(),
            df["temperature"].max(), df["humidity"].max(),
            df["ph"].max(), df["rainfall"].max()
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    st.subheader("🔥 Feature Correlation Heatmap")

    corr = df.drop("label", axis=1).corr()

    heat = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Viridis")
    )

    heat.update_layout(height=600)

    st.plotly_chart(heat, use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 PCA Visualization")

    X = df.drop("label", axis=1)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Crop": df["label"]})

    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Crop", hover_data=["Crop"],
                     title="PCA Projection of Crop Dataset")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 K-Means Cluster Visualization")

    clusters = kmeans_model.predict(X_scaled)

    cluster_pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": clusters.astype(str), "Crop": df["label"]
    })

    fig = px.scatter(cluster_pca_df, x="PC1", y="PC2", color="Cluster", hover_data=["Crop"],
                     title="Farm Clusters (K-Means)")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Cluster Size Distribution")

    cluster_size = pd.Series(clusters).value_counts().sort_index().reset_index()
    cluster_size.columns = ["Cluster", "Count"]

    fig = px.bar(cluster_size, x="Cluster", y="Count", color="Cluster", text="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Random Forest Feature Importance")

    try:
        importance = crop_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": X.columns, "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", text="Importance",
                     title="Feature Importance for Crop Prediction")

        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Feature importance unavailable for this model.")

    st.markdown("---")

    st.subheader("🏆 Model Performance Dashboard")

    performance = pd.DataFrame({
        "Algorithm": ["Random Forest", "K-Means", "MLP Neural Network", "Fuzzy Logic", "Q-Learning"],
        "Type": ["Supervised Learning", "Unsupervised Learning", "Neural Network", "Fuzzy Logic", "Reinforcement Learning"],
        "Purpose": ["Crop Prediction", "Farm Segmentation", "Crop Prediction", "Irrigation Decision", "Smart Irrigation"],
        "Performance": ["99.32% Accuracy", f"{kmeans_model.n_clusters} Clusters", f"{ann_accuracy*100:.2f}% Accuracy", "Rule-Based Decision", "Policy Learned"]
    })

    st.dataframe(performance, use_container_width=True)

    st.markdown("---")

    st.subheader("📥 Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("Download Full Dataset", csv,
                       file_name="Crop_recommendation.csv", mime="text/csv")

    st.markdown("---")

    st.subheader("📄 Analytics Report Summary")

    st.success(f"""
### Analytics Summary

✔ Dataset contains **{len(df)}** records across **{df['label'].nunique()}** crop types.

✔ Correlation heatmap shows relationships between soil and weather features.

✔ PCA reduces the 7 features into 2 principal components for visualization.

✔ K-Means groups farms into **{kmeans_model.n_clusters}** productivity clusters.

✔ Random Forest achieves **99.32%** accuracy in crop prediction.

✔ All charts above are interactive — hover, zoom, and pan for deeper insights.
""")

    st.markdown("---")

    st.success("✅ Analytics Dashboard Loaded Successfully")

# ==========================================================
# ℹ ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.title("🌱 Smart Agriculture Management System")

    st.markdown("### AI-Powered Agriculture using Hybrid Soft Computing")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628324.png", width=180)

    with col2:
        st.subheader("Project Overview")

        st.write("""
The Smart Agriculture Management System is an AI-powered decision
support application built using three major Machine Learning paradigms.

It helps farmers and researchers to:

✔ Predict the best crop for given soil and weather conditions

✔ Segment farms into productivity groups

✔ Receive smart irrigation recommendations

✔ Explore agricultural data through interactive analytics

✔ Track and review past predictions
""")

    st.markdown("---")

    st.subheader("Machine Learning Algorithms")

    algo = pd.DataFrame({
        "Learning Type": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
        "Algorithm": ["Random Forest", "K-Means Clustering", "Q-Learning"],
        "Application": ["Crop Prediction", "Farm Segmentation", "Smart Irrigation"]
    })

    st.dataframe(algo, use_container_width=True)

    st.markdown("---")

    st.subheader("Technologies Used")

    tech = pd.DataFrame({
        "Technology": ["Python", "Streamlit", "Pandas", "NumPy", "Scikit-Learn", "Plotly", "Joblib"],
        "Purpose": [
            "Core Programming", "Web Dashboard", "Data Handling", "Numerical Computing",
            "Machine Learning", "Interactive Visualization", "Model Persistence"
        ]
    })

    st.dataframe(tech, use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Information")

    st.write(f"""
**Dataset Name:** Crop Recommendation Dataset

**Number of Records:** {len(df)}

**Number of Features:** {len(df.columns) - 1}

**Target Variable:** Crop Label

**Total Crop Types:** {df['label'].nunique()}
""")

    st.markdown("---")

    st.subheader("Project Workflow")

    workflow = """
Dataset
  ↓
Data Preprocessing
  ↓
Random Forest → Crop Prediction
  ↓
ANN / Neural Network → Crop Prediction
  ↓
K-Means → Farm Segmentation
  ↓
Fuzzy Logic → Irrigation Level
  ↓
Q-Learning → Irrigation Action Optimization
  ↓
Hybrid Soft Computing Decision
  ↓
Interactive Analytics Dashboard
  ↓
Prediction History & Reports
"""

    st.code(workflow)

    st.markdown("---")

    st.subheader("Project Features")

    st.write("""
✅ Login & Register System

✅ Crop Prediction with Confidence Score

✅ Farm Segmentation with PCA Visualization

✅ Smart Irrigation using Q-Learning

✅ Prediction History with Search & Filter

✅ Correlation Heatmap

✅ Feature Importance Charts

✅ Interactive Plotly Visualizations

✅ Dataset & History Download

✅ Professional, Modern UI
""")

    st.markdown("---")

    st.subheader("Developer")

    st.success("""
**Name:** Monisha Mariappan

**Course:** Artificial Intelligence & Data Science

**Project:** Smart Agriculture Management System (Version 2.0)

**Algorithms Used:**
• Random Forest
• K-Means Clustering
• Q-Learning
""")

    st.markdown("---")

    st.info("""
This project demonstrates the integration of Supervised Learning,
Unsupervised Learning, and Reinforcement Learning into a single
real-world AI application.
""")

    st.markdown("---")

    st.caption("© 2026 Smart Agriculture Management System | Developed by Monisha Mariappan")

# ==========================================================
# 🦶 GLOBAL FOOTER (shown on every page after login)
# ==========================================================

st.markdown("---")

st.markdown(
    "<p class='footer'>🌱 Hybrid Soft Computing Smart Agriculture System | "
    "Powered by Random Forest, K-Means, ANN, Fuzzy Logic & Q-Learning | "
    "© 2026 Monisha Mariappan</p>",
    unsafe_allow_html=True
)