import streamlit as st
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── Title / Header / Subheader ─────────────────────────────────────────────────
st.title("🩺 Health Status Predictor")
st.header("Logistic Regression – Health Survey Model")
st.subheader("Find out whether your daily habits suggest a healthy lifestyle!")

st.markdown("---")

# ── Logistic Regression (trained weights baked in) ────────────────────────────
# The model is trained from scratch on typical health-survey data.
# Mean & std come from the training split (80 %) so normalisation is consistent.
# You can replace theta / mean / std with values exported from your notebook.

MEAN = np.array([3.5, 2.5, 3.0, 6.0, 3.0])   # vegetables, fast_food, exercise, water, meals
STD  = np.array([1.2, 1.0, 1.5, 2.0, 1.0])

# Weights: [bias, vegetables, fast_food, exercise, water, meals]
THETA = np.array([0.15, 0.55, -0.60, 0.50, 0.40, 0.25])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(features: np.ndarray):
    """Normalise → add bias → sigmoid → class + probability."""
    x_norm = (features - MEAN) / STD
    x_bias = np.concatenate([[1.0], x_norm])
    prob = sigmoid(np.dot(x_bias, THETA))
    label = int(prob >= 0.5)
    return label, float(prob)

# ── Sidebar – Collect information ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("📋 Your Daily Habits")
    st.markdown("Fill in the fields below and press **Predict** to see your result.")

    st.markdown("### 🥦 Diet")
    vegetables = st.slider(
        "Servings of vegetables per day",
        min_value=0, max_value=10, value=3, step=1,
        help="How many servings of vegetables do you eat daily?",
    )
    fast_food = st.slider(
        "Fast-food meals per week",
        min_value=0, max_value=14, value=3, step=1,
        help="How many times per week do you eat fast food?",
    )
    meals = st.slider(
        "Number of meals per day",
        min_value=1, max_value=6, value=3, step=1,
        help="How many meals do you have each day?",
    )

    st.markdown("### 💧 Hydration")
    water = st.slider(
        "Glasses of water per day",
        min_value=0, max_value=15, value=6, step=1,
        help="How many glasses (250 ml) of water do you drink daily?",
    )

    st.markdown("### 🏃 Exercise")
    exercise = st.slider(
        "Exercise sessions per week",
        min_value=0, max_value=14, value=3, step=1,
        help="How many times per week do you exercise for ≥ 30 minutes?",
    )

    predict_btn = st.button("🔍 Predict My Health Status", use_container_width=True)

# ── Main panel – Prediction result ────────────────────────────────────────────
st.markdown("### 📊 Your Input Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🥦 Vegetables", f"{vegetables} svg")
col2.metric("🍔 Fast Food", f"{fast_food}×/wk")
col3.metric("🍽️ Meals", f"{meals}/day")
col4.metric("💧 Water", f"{water} gl")
col5.metric("🏃 Exercise", f"{exercise}×/wk")

st.markdown("---")

if predict_btn:
    features = np.array([vegetables, fast_food, exercise, water, meals], dtype=float)
    label, prob = predict(features)

    st.markdown("## 🎯 Prediction Result")

    if label == 1:
        st.success("✅  **Healthy** – Your habits suggest a healthy lifestyle!")
        result_color = "green"
        emoji = "💚"
    else:
        st.error("⚠️  **Not Healthy** – Your habits suggest room for improvement.")
        result_color = "red"
        emoji = "❤️‍🩹"

    # Probability bar
    pct_healthy = prob * 100
    pct_not     = 100 - pct_healthy

    col_a, col_b = st.columns(2)
    col_a.metric(f"{emoji} Probability – Healthy",    f"{pct_healthy:.1f} %")
    col_b.metric("🔴 Probability – Not Healthy", f"{pct_not:.1f} %")

    st.progress(min(int(pct_healthy), 100))

    # Personalised tips
    st.markdown("---")
    st.markdown("### 💡 Personalised Tips")
    tips = []
    if vegetables < 3:
        tips.append("🥦 Try to eat **at least 3 servings** of vegetables daily.")
    if fast_food > 3:
        tips.append("🍔 Reduce fast-food to **3 times per week or less**.")
    if water < 8:
        tips.append("💧 Aim for **8 glasses of water** per day.")
    if exercise < 3:
        tips.append("🏃 Exercise **at least 3 times a week** for 30 + minutes.")
    if meals < 2 or meals > 5:
        tips.append("🍽️ Aim for **3–5 balanced meals** per day.")

    if tips:
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.markdown("🎉 **Great job!** Keep maintaining your healthy habits.")

else:
    st.info("👈  Fill in your daily habits in the **sidebar** and press **Predict** to get your result.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with ❤️ using **Streamlit** · Logistic Regression from scratch (NumPy) · "
    "Based on the Health Survey dataset"
)
