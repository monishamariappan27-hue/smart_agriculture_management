import streamlit as st
import pandas as pd
import numpy as np
import os
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hybrid Soft Computing - Smart Agriculture",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Hybrid Soft Computing Based Smart Agriculture Management System")
st.caption("Neural Network + Fuzzy Logic + Neuro-Fuzzy Hybrid Integration")

# ============================================================
# DATASET
# ============================================================

@st.cache_data
def load_dataset():
    paths = [
        "data/Crop_recommendation.csv",
        "Crop_recommendation.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            data.columns = [str(c).strip().lower() for c in data.columns]
            return data
    return None

df = load_dataset()

if df is None:
    st.error(
        "Crop_recommendation.csv not found. Put it in either "
        "'data/Crop_recommendation.csv' or the same folder as app.py."
    )
    st.stop()

FEATURES = [
    "n", "p", "k", "temperature",
    "humidity", "ph", "rainfall"
]
TARGET = "label"

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error("Missing dataset columns: " + ", ".join(missing))
    st.stop()

# ============================================================
# NEURAL NETWORK - CORE TRAINING (default architecture)
# ============================================================

@st.cache_resource
def train_ann(data, hidden_layers=(64, 32), activation="relu", max_iter=700):
    X = data[FEATURES]
    y = data[TARGET].astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, encoder, accuracy, cm


ann_model, scaler, label_encoder, ann_accuracy, cm = train_ann(df)

# ============================================================
# FUZZY LOGIC - MEMBERSHIP FUNCTION BUILDING BLOCKS
# ============================================================

def left_shoulder(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def right_shoulder(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


# ----- Fuzzy System 1: Irrigation -----

def fuzzy_irrigation(moisture, temperature, humidity):
    moisture_low = left_shoulder(moisture, 25, 45)
    moisture_medium = triangular(moisture, 30, 50, 70)
    moisture_high = right_shoulder(moisture, 55, 75)

    temp_low = left_shoulder(temperature, 15, 25)
    temp_medium = triangular(temperature, 20, 30, 40)
    temp_high = right_shoulder(temperature, 35, 45)

    humidity_low = left_shoulder(humidity, 30, 50)
    humidity_medium = triangular(humidity, 40, 60, 80)
    humidity_high = right_shoulder(humidity, 70, 90)

    high_1 = min(moisture_low, temp_high, humidity_low)
    high_2 = min(moisture_low, max(temp_medium, temp_high))

    medium_1 = min(moisture_medium, temp_medium)
    medium_2 = min(moisture_low, humidity_high)
    medium_3 = min(moisture_medium, humidity_low)

    low_1 = moisture_high
    low_2 = min(humidity_high, max(temp_low, temp_medium))

    low_strength = max(low_1, low_2)
    medium_strength = max(medium_1, medium_2, medium_3)
    high_strength = max(high_1, high_2)

    total = low_strength + medium_strength + high_strength

    if total == 0:
        score = 50
    else:
        score = (
            low_strength * 20 +
            medium_strength * 55 +
            high_strength * 90
        ) / total

    if score < 35:
        level = "Low"
        action = "Irrigation is not highly required."
    elif score < 70:
        level = "Medium"
        action = "Provide moderate irrigation."
    else:
        level = "High"
        action = "Provide more irrigation."

    memberships = {
        "Moisture Low": moisture_low,
        "Moisture Medium": moisture_medium,
        "Moisture High": moisture_high,
        "Temperature Low": temp_low,
        "Temperature Medium": temp_medium,
        "Temperature High": temp_high,
        "Humidity Low": humidity_low,
        "Humidity Medium": humidity_medium,
        "Humidity High": humidity_high
    }

    return score, level, action, memberships


# ----- Fuzzy System 2: Fertilizer Advisor (NEW) -----

def fuzzy_fertilizer(n, p, k):
    n_low = left_shoulder(n, 40, 80)
    n_med = triangular(n, 50, 90, 130)
    n_high = right_shoulder(n, 100, 140)

    p_low = left_shoulder(p, 30, 70)
    p_med = triangular(p, 40, 80, 120)
    p_high = right_shoulder(p, 90, 130)

    k_low = left_shoulder(k, 40, 90)
    k_med = triangular(k, 60, 110, 160)
    k_high = right_shoulder(k, 130, 180)

    deficiency_high = max(n_low, p_low, k_low)
    deficiency_medium = max(n_med, p_med, k_med)
    deficiency_low = max(n_high, p_high, k_high)

    total = deficiency_high + deficiency_medium + deficiency_low

    if total == 0:
        score = 50
    else:
        score = (
            deficiency_high * 90 +
            deficiency_medium * 55 +
            deficiency_low * 20
        ) / total

    if score < 35:
        level = "Low"
        action = "Soil nutrients are sufficient. Minimal fertilizer needed."
    elif score < 70:
        level = "Medium"
        action = "Apply a balanced NPK fertilizer in moderate quantity."
    else:
        level = "High"
        action = "Soil is nutrient deficient. Apply fertilizer at a higher dose."

    memberships = {
        "N Low": n_low, "N Medium": n_med, "N High": n_high,
        "P Low": p_low, "P Medium": p_med, "P High": p_high,
        "K Low": k_low, "K Medium": k_med, "K High": k_high
    }

    return score, level, action, memberships


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧠 Neural Network",
        "🧪 NN Model Lab",
        "🔷 Fuzzy Logic - Irrigation",
        "🌿 Fuzzy Logic - Fertilizer",
        "📈 Fuzzy Membership Explorer",
        "🔀 Neuro-Fuzzy Hybrid",
        "📁 Batch Prediction",
        "📊 Results",
        "ℹ️ About"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.header("🏠 Project Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", len(df))
    c2.metric("Crop Classes", df[TARGET].nunique())
    c3.metric("ANN Accuracy", f"{ann_accuracy * 100:.2f}%")
    c4.metric("Fuzzy Systems", "2 (Irrigation, Fertilizer)")

    st.subheader("🎯 Domain")
    st.success("Hybrid Soft Computing")

    st.subheader("🔄 Project Workflow")
    st.code("""
Agricultural Data
       ↓
Data Preprocessing
       ↓
 ┌────────────────────┬────────────────────┐
 ↓                     ↓
🧠 Neural Network    🔷 Fuzzy Logic
 - Crop Prediction     - Irrigation Decision
 - Configurable arch   - Fertilizer Advisor
 ↓                     ↓
 └────────────────────┬────────────────────┘
                       ↓
             🔀 Neuro-Fuzzy Hybrid
                       ↓
           Smart Agriculture Decision
""")

    st.info(
        "This Soft Computing version focuses on Neural Network, "
        "Fuzzy Logic, and their Neuro-Fuzzy Hybrid integration, with "
        "extra tools for architecture experimentation, membership "
        "visualization, and batch prediction."
    )

# ============================================================
# NEURAL NETWORK PAGE
# ============================================================

elif page == "🧠 Neural Network":
    st.header("🧠 Artificial Neural Network - Crop Prediction")

    st.write(
        "The ANN (Multi-Layer Perceptron) predicts a suitable crop using "
        "soil nutrients and environmental parameters."
    )

    c1, c2 = st.columns(2)

    with c1:
        n = st.number_input("Nitrogen (N)", 0.0, 150.0, 90.0)
        p = st.number_input("Phosphorus (P)", 0.0, 150.0, 42.0)
        k = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)

    with c2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

    if st.button("🧠 Predict Crop", type="primary"):
        values = np.array([[
            n, p, k, temperature,
            humidity, ph, rainfall
        ]])

        values_scaled = scaler.transform(values)
        prediction = ann_model.predict(values_scaled)
        crop = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🌾 Recommended Crop: **{crop.upper()}**")

        if hasattr(ann_model, "predict_proba"):
            probs = ann_model.predict_proba(values_scaled)[0]
            confidence = np.max(probs) * 100
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            top_idx = np.argsort(probs)[::-1][:5]
            top_df = pd.DataFrame({
                "Crop": label_encoder.inverse_transform(top_idx),
                "Probability (%)": [round(probs[i] * 100, 2) for i in top_idx]
            })
            st.subheader("🔝 Top-5 Crop Probabilities")
            fig_top = px.bar(top_df, x="Crop", y="Probability (%)",
                              text="Probability (%)", color="Probability (%)")
            st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("🧠 ANN Architecture (Default Model)")
    st.dataframe(
        pd.DataFrame({
            "Layer": [
                "Input Layer",
                "Hidden Layer 1",
                "Hidden Layer 2",
                "Output Layer"
            ],
            "Details": [
                "7 agricultural features",
                "64 neurons - ReLU",
                "32 neurons - ReLU",
                "Crop classes"
            ]
        }),
        use_container_width=True
    )

    if hasattr(ann_model, "loss_curve_"):
        st.subheader("📉 Training Loss Curve")
        loss_df = pd.DataFrame({
            "Iteration": list(range(1, len(ann_model.loss_curve_) + 1)),
            "Loss": ann_model.loss_curve_
        })
        fig_loss = px.line(loss_df, x="Iteration", y="Loss",
                            title="ANN Training Loss vs Iteration")
        st.plotly_chart(fig_loss, use_container_width=True)

# ============================================================
# NN MODEL LAB - CUSTOM ARCHITECTURE + COMPARISON (NEW)
# ============================================================

elif page == "🧪 NN Model Lab":
    st.header("🧪 Neural Network Model Lab")
    st.write(
        "Experiment with different ANN architectures and activation "
        "functions, and compare their accuracy."
    )

    st.subheader("🔧 Train a Custom Architecture")

    c1, c2, c3 = st.columns(3)
    with c1:
        layer1 = st.slider("Neurons - Hidden Layer 1", 4, 128, 64, step=4)
    with c2:
        layer2 = st.slider("Neurons - Hidden Layer 2 (0 = disable)", 0, 128, 32, step=4)
    with c3:
        activation_choice = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])

    max_iter_choice = st.slider("Max Training Iterations", 100, 1500, 700, step=100)

    if st.button("🧪 Train Custom Model", type="primary"):
        hidden_layers = (layer1,) if layer2 == 0 else (layer1, layer2)
        with st.spinner("Training custom ANN..."):
            custom_model, custom_scaler, custom_encoder, custom_acc, custom_cm = train_ann(
                df, hidden_layers=hidden_layers,
                activation=activation_choice,
                max_iter=max_iter_choice
            )
        st.success(f"Custom model trained! Accuracy: {custom_acc * 100:.2f}%")

        if hasattr(custom_model, "loss_curve_"):
            loss_df = pd.DataFrame({
                "Iteration": list(range(1, len(custom_model.loss_curve_) + 1)),
                "Loss": custom_model.loss_curve_
            })
            fig_loss = px.line(loss_df, x="Iteration", y="Loss",
                                title=f"Loss Curve - {hidden_layers} - {activation_choice}")
            st.plotly_chart(fig_loss, use_container_width=True)

    st.subheader("📊 Compare Standard Architectures")
    if st.button("Run Architecture Comparison"):
        configs = [
            ("(32,)", (32,)),
            ("(64,32)", (64, 32)),
            ("(64,64,32)", (64, 64, 32)),
            ("(128,64,32)", (128, 64, 32)),
        ]
        results = []
        progress = st.progress(0)
        for i, (label, layers) in enumerate(configs):
            _, _, _, acc, _ = train_ann(df, hidden_layers=layers)
            results.append({"Architecture": label, "Accuracy (%)": round(acc * 100, 2)})
            progress.progress((i + 1) / len(configs))

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        fig_cmp = px.bar(results_df, x="Architecture", y="Accuracy (%)",
                          text="Accuracy (%)", title="Architecture Accuracy Comparison")
        st.plotly_chart(fig_cmp, use_container_width=True)

# ============================================================
# FUZZY LOGIC - IRRIGATION PAGE
# ============================================================

elif page == "🔷 Fuzzy Logic - Irrigation":
    st.header("🔷 Fuzzy Logic - Irrigation Decision")

    st.write(
        "Fuzzy Logic converts uncertain agricultural conditions into "
        "Low, Medium or High irrigation decisions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        moisture = st.slider("💧 Soil Moisture (%)", 0, 100, 35)

    with c2:
        fuzzy_temp = st.slider("🌡️ Temperature (°C)", 0, 60, 30)

    with c3:
        fuzzy_humidity = st.slider("💦 Humidity (%)", 0, 100, 60)

    if st.button("🔷 Calculate Irrigation", type="primary"):
        score, level, action, memberships = fuzzy_irrigation(
            moisture, fuzzy_temp, fuzzy_humidity
        )

        st.success(f"💧 Irrigation Level: **{level.upper()}**")
        st.metric("Fuzzy Irrigation Score", f"{score:.2f}")
        st.info(action)

        st.subheader("Fuzzy Membership Values")
        membership_df = pd.DataFrame({
            "Fuzzy Set": list(memberships.keys()),
            "Membership Value": [round(v, 3) for v in memberships.values()]
        })
        fig_mem = px.bar(membership_df, x="Fuzzy Set", y="Membership Value",
                          color="Membership Value", title="Rule Firing Strengths")
        st.plotly_chart(fig_mem, use_container_width=True)
        st.dataframe(membership_df, use_container_width=True)

    st.subheader("📜 Example Fuzzy Rules")
    st.markdown("""
    - IF soil moisture is **LOW** AND temperature is **HIGH** → irrigation **HIGH**
    - IF soil moisture is **MEDIUM** AND temperature is **MEDIUM** → irrigation **MEDIUM**
    - IF soil moisture is **HIGH** → irrigation **LOW**
    - IF humidity is **HIGH** → irrigation can be reduced
    """)

# ============================================================
# FUZZY LOGIC - FERTILIZER PAGE (NEW)
# ============================================================

elif page == "🌿 Fuzzy Logic - Fertilizer":
    st.header("🌿 Fuzzy Logic - Fertilizer Advisor")

    st.write(
        "This fuzzy system evaluates Nitrogen, Phosphorus and Potassium "
        "levels to recommend a fertilizer dose."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fn = st.slider("Nitrogen (N)", 0, 150, 40)
    with c2:
        fp = st.slider("Phosphorus (P)", 0, 150, 40)
    with c3:
        fk = st.slider("Potassium (K)", 0, 200, 40)

    if st.button("🌿 Calculate Fertilizer Need", type="primary"):
        score, level, action, memberships = fuzzy_fertilizer(fn, fp, fk)

        st.success(f"🌿 Fertilizer Requirement: **{level.upper()}**")
        st.metric("Fuzzy Fertilizer Score", f"{score:.2f}")
        st.info(action)

        st.subheader("Fuzzy Membership Values")
        membership_df = pd.DataFrame({
            "Fuzzy Set": list(memberships.keys()),
            "Membership Value": [round(v, 3) for v in memberships.values()]
        })
        fig_mem = px.bar(membership_df, x="Fuzzy Set", y="Membership Value",
                          color="Membership Value", title="Nutrient Rule Firing Strengths")
        st.plotly_chart(fig_mem, use_container_width=True)
        st.dataframe(membership_df, use_container_width=True)

    st.subheader("📜 Example Fuzzy Rules")
    st.markdown("""
    - IF N, P, or K is **LOW** → fertilizer requirement **HIGH**
    - IF N, P, K are around **MEDIUM** → fertilizer requirement **MEDIUM**
    - IF N, P, K are **HIGH** → fertilizer requirement **LOW**
    """)

# ============================================================
# FUZZY MEMBERSHIP EXPLORER (NEW)
# ============================================================

elif page == "📈 Fuzzy Membership Explorer":
    st.header("📈 Fuzzy Membership Function Explorer")
    st.write(
        "Visualize the shape of the fuzzy membership functions used by "
        "the Irrigation and Fertilizer systems."
    )

    variable = st.selectbox(
        "Choose a variable to visualize",
        ["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)",
         "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
    )

    if variable == "Soil Moisture (%)":
        x_vals = np.linspace(0, 100, 200)
        low = [left_shoulder(x, 25, 45) for x in x_vals]
        med = [triangular(x, 30, 50, 70) for x in x_vals]
        high = [right_shoulder(x, 55, 75) for x in x_vals]
    elif variable == "Temperature (°C)":
        x_vals = np.linspace(0, 60, 200)
        low = [left_shoulder(x, 15, 25) for x in x_vals]
        med = [triangular(x, 20, 30, 40) for x in x_vals]
        high = [right_shoulder(x, 35, 45) for x in x_vals]
    elif variable == "Humidity (%)":
        x_vals = np.linspace(0, 100, 200)
        low = [left_shoulder(x, 30, 50) for x in x_vals]
        med = [triangular(x, 40, 60, 80) for x in x_vals]
        high = [right_shoulder(x, 70, 90) for x in x_vals]
    elif variable == "Nitrogen (N)":
        x_vals = np.linspace(0, 150, 200)
        low = [left_shoulder(x, 40, 80) for x in x_vals]
        med = [triangular(x, 50, 90, 130) for x in x_vals]
        high = [right_shoulder(x, 100, 140) for x in x_vals]
    elif variable == "Phosphorus (P)":
        x_vals = np.linspace(0, 150, 200)
        low = [left_shoulder(x, 30, 70) for x in x_vals]
        med = [triangular(x, 40, 80, 120) for x in x_vals]
        high = [right_shoulder(x, 90, 130) for x in x_vals]
    else:
        x_vals = np.linspace(0, 200, 200)
        low = [left_shoulder(x, 40, 90) for x in x_vals]
        med = [triangular(x, 60, 110, 160) for x in x_vals]
        high = [right_shoulder(x, 130, 180) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=low, name="Low", mode="lines"))
    fig.add_trace(go.Scatter(x=x_vals, y=med, name="Medium", mode="lines"))
    fig.add_trace(go.Scatter(x=x_vals, y=high, name="High", mode="lines"))
    fig.update_layout(
        title=f"Membership Functions - {variable}",
        xaxis_title=variable,
        yaxis_title="Membership Degree",
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig, use_container_width=True)

    pick = st.slider(f"Pick a value of {variable} to inspect", float(x_vals.min()), float(x_vals.max()), float(x_vals.mean()))
    idx = (np.abs(x_vals - pick)).argmin()
    st.write(
        f"At value **{pick:.2f}** → Low = **{low[idx]:.3f}**, "
        f"Medium = **{med[idx]:.3f}**, High = **{high[idx]:.3f}**"
    )

# ============================================================
# NEURO-FUZZY HYBRID PAGE
# ============================================================

elif page == "🔀 Neuro-Fuzzy Hybrid":
    st.header("🔀 Neuro-Fuzzy Hybrid System")

    st.write(
        "The Hybrid module combines ANN crop prediction confidence with "
        "Fuzzy Logic irrigation and fertilizer decisions into a single "
        "Smart Agriculture recommendation."
    )

    st.subheader("🌾 ANN Inputs")
    c1, c2 = st.columns(2)
    with c1:
        hn = st.number_input("Nitrogen", 0.0, 150.0, 90.0, key="hn")
        hp = st.number_input("Phosphorus", 0.0, 150.0, 42.0, key="hp")
        hk = st.number_input("Potassium", 0.0, 250.0, 43.0, key="hk")
        ht = st.number_input("Temperature", 0.0, 60.0, 25.0, key="ht")
    with c2:
        hh = st.number_input("Humidity", 0.0, 100.0, 80.0, key="hh")
        hph = st.number_input("pH", 0.0, 14.0, 6.5, key="hph")
        hr = st.number_input("Rainfall", 0.0, 500.0, 200.0, key="hr")

    st.subheader("💧 Fuzzy Irrigation Inputs")
    f1, f2, f3 = st.columns(3)
    with f1:
        hm = st.slider("Soil Moisture", 0, 100, 35, key="hm")
    with f2:
        hft = st.slider("Temperature (Fuzzy)", 0, 60, 30, key="hft")
    with f3:
        hfh = st.slider("Humidity (Fuzzy)", 0, 100, 60, key="hfh")

    st.subheader("⚖️ Hybrid Weighting")
    alpha = st.slider(
        "Weight given to ANN confidence vs Fuzzy urgency (α)",
        0.0, 1.0, 0.5, step=0.05,
        help="α close to 1 favors the ANN's crop confidence; α close to 0 favors fuzzy irrigation/fertilizer urgency."
    )

    if st.button("🔀 Generate Hybrid Decision", type="primary"):
        ann_input = np.array([[hn, hp, hk, ht, hh, hph, hr]])
        ann_input_scaled = scaler.transform(ann_input)
        pred = ann_model.predict(ann_input_scaled)
        crop = label_encoder.inverse_transform(pred)[0]

        confidence = 50.0
        if hasattr(ann_model, "predict_proba"):
            confidence = float(np.max(ann_model.predict_proba(ann_input_scaled)) * 100)

        irr_score, irrigation, irr_action, _ = fuzzy_irrigation(hm, hft, hfh)
        fert_score, fertilizer, fert_action, _ = fuzzy_fertilizer(hn, hp, hk)

        a, b, c = st.columns(3)
        with a:
            st.success(f"🌾 ANN Crop\n\n**{crop.upper()}**\n\nConfidence: {confidence:.1f}%")
        with b:
            st.info(f"💧 Fuzzy Irrigation\n\n**{irrigation.upper()}**\n\nScore: {irr_score:.1f}")
        with c:
            st.warning(f"🌿 Fuzzy Fertilizer\n\n**{fertilizer.upper()}**\n\nScore: {fert_score:.1f}")

        fuzzy_urgency = (irr_score + fert_score) / 2
        smart_score = alpha * confidence + (1 - alpha) * fuzzy_urgency

        st.subheader("🌱 Final Neuro-Fuzzy Decision")
        st.success(
            f"Recommended Crop: **{crop.upper()}**  |  "
            f"Irrigation: **{irrigation.upper()}**  |  "
            f"Fertilizer: **{fertilizer.upper()}**"
        )
        st.metric("🔀 Smart Agriculture Score", f"{smart_score:.2f} / 100")
        st.write(irr_action)
        st.write(fert_action)

        st.subheader("🔄 Hybrid Flow")
        st.code("""
ANN → Crop Recommendation + Confidence
             +
Fuzzy Logic → Irrigation Decision + Score
             +
Fuzzy Logic → Fertilizer Decision + Score
             ↓
   Weighted Neuro-Fuzzy Integration (α)
             ↓
     Smart Agriculture Score & Decision
""")

# ============================================================
# BATCH PREDICTION PAGE (NEW)
# ============================================================

elif page == "📁 Batch Prediction":
    st.header("📁 Batch Prediction from CSV")
    st.write(
        "Upload a CSV with columns: n, p, k, temperature, humidity, ph, rainfall "
        "to get crop predictions for many rows at once."
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df.columns = [str(c).strip().lower() for c in batch_df.columns]

        missing_cols = [c for c in FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error("Missing columns: " + ", ".join(missing_cols))
        else:
            X_batch = batch_df[FEATURES]
            X_batch_scaled = scaler.transform(X_batch)
            preds = ann_model.predict(X_batch_scaled)
            batch_df["predicted_crop"] = label_encoder.inverse_transform(preds)

            if hasattr(ann_model, "predict_proba"):
                probs = ann_model.predict_proba(X_batch_scaled)
                batch_df["confidence_%"] = np.round(np.max(probs, axis=1) * 100, 2)

            st.success(f"Predicted crops for {len(batch_df)} rows.")
            st.dataframe(batch_df, use_container_width=True)

            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="crop_predictions.csv",
                mime="text/csv"
            )

            st.subheader("📊 Predicted Crop Distribution")
            dist_df = batch_df["predicted_crop"].value_counts().reset_index()
            dist_df.columns = ["Crop", "Count"]
            fig_dist = px.bar(dist_df, x="Crop", y="Count", color="Count")
            st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# RESULTS PAGE
# ============================================================

elif page == "📊 Results":
    st.header("📊 Results")

    st.metric("Neural Network Accuracy", f"{ann_accuracy * 100:.2f}%")

    accuracy_df = pd.DataFrame({
        "Model": ["Artificial Neural Network"],
        "Accuracy": [ann_accuracy * 100]
    })

    fig = px.bar(
        accuracy_df, x="Model", y="Accuracy",
        text="Accuracy", title="ANN Accuracy"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔢 Confusion Matrix")
    labels = label_encoder.classes_
    fig_cm = px.imshow(
        cm, x=labels, y=labels,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="ANN Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📋 Dataset Information")
    st.dataframe(
        pd.DataFrame({
            "Item": ["Records", "Features", "Crop Classes", "Domain"],
            "Value": [len(df), len(FEATURES), df[TARGET].nunique(), "Hybrid Soft Computing"]
        }),
        use_container_width=True
    )

    st.subheader("📈 Feature Distribution")
    feature_pick = st.selectbox("Select a feature to view its distribution", FEATURES)
    fig_hist = px.histogram(df, x=feature_pick, color=TARGET, title=f"Distribution of {feature_pick}")
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":
    st.header("ℹ️ About the Project")

    st.success("Hybrid Soft Computing Based Smart Agriculture Management System")

    st.subheader("Domain")
    st.write("**Hybrid Soft Computing**")

    st.subheader("Techniques Used")
    st.dataframe(
        pd.DataFrame({
            "Technique": [
                "Artificial Neural Network (MLP)",
                "Fuzzy Logic - Irrigation",
                "Fuzzy Logic - Fertilizer",
                "Neuro-Fuzzy Hybrid Integration"
            ],
            "Application": [
                "Crop Recommendation",
                "Irrigation Decision",
                "Fertilizer Dose Recommendation",
                "Combined Smart Agriculture Score & Decision"
            ]
        }),
        use_container_width=True
    )

    st.subheader("Extra Features Added")
    st.markdown("""
    - Configurable ANN architecture and activation function (Model Lab)
    - Architecture accuracy comparison across multiple ANN configurations
    - Training loss curve visualization
    - Top-5 crop probability ranking
    - Second fuzzy system for fertilizer recommendation (N, P, K)
    - Interactive fuzzy membership function explorer
    - Weighted Neuro-Fuzzy hybrid score (adjustable α)
    - Batch prediction from an uploaded CSV file with downloadable results
    - Feature distribution explorer on the Results page
    """)

    st.subheader("What We Learned")
    st.markdown("""
    - ANN architecture design, training, and hyperparameter tuning
    - Feature scaling and multi-class classification
    - Fuzzy membership functions (triangular, shoulder)
    - Fuzzy IF-THEN rule design for multiple systems
    - Fuzzification and defuzzification (weighted average)
    - Neuro-Fuzzy hybrid integration with adjustable weighting
    - Streamlit application development and interactive visualization
    """)

st.markdown("---")
st.caption(
    "🌱 Hybrid Soft Computing Smart Agriculture Management System | 2026"
)