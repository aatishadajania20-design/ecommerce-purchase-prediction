import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_model.pkl"

st.set_page_config(
    page_title="E-Commerce Customer Purchase Prediction System",
    page_icon="🛒",
    layout="centered",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 900px;}
    .main-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
    .subtitle {color: #6b7280; margin-bottom: 1.2rem;}
    .result-box {padding: 1.6rem 1rem; border-radius: 14px; text-align: center;
                 font-size: 1.7rem; font-weight: 800; margin: 1rem 0 0.5rem 0;}
    .result-yes {background: #dcfce7; color: #166534; border: 2px solid #22c55e;}
    .result-no {background: #fee2e2; color: #991b1b; border: 2px solid #ef4444;}
    .prob-text {text-align: center; font-size: 1.15rem; margin-bottom: 0.5rem;}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; height: 3.2rem; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ load the model
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("`best_model.pkl` was not found. Run `python train_model.py` first, "
             "then restart the app.")
    st.stop()

try:
    artifact = load_artifact()
except Exception as e:  # e.g. scikit-learn version mismatch
    st.error(f"Could not load the model: {e}\n\nInstall the package versions in "
             "`requirements.txt`, or re-run `python train_model.py` in your environment.")
    st.stop()

pipeline = artifact["pipeline"]
cats = artifact["categories"]
FEATURES = artifact["features"]

# ------------------------------------------------------------------- header
st.markdown('<p class="main-title">🛒 E-Commerce Customer Purchase Prediction System</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter a customer\'s shopping-session details below and the machine '
    'learning model will predict whether the customer is likely to purchase the product.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- input form
with st.form("prediction_form"):
    st.subheader("Customer Profile")
    c1, c2, c3 = st.columns(3)
    customer_age = c1.slider("Customer age", 18, 75, 35)
    gender = c2.selectbox("Gender", cats["gender"])
    city = c3.selectbox("City", cats["city"])

    is_returning = st.radio("Returning customer?", ["Yes", "No"], horizontal=True)

    st.subheader("Purchase History")
    h1, h2 = st.columns(2)
    previous_orders = h1.number_input("Previous orders", 0, 40, 3, step=1)
    previous_total_spend = h2.number_input("Previous total spend ($)", 0.0, 6000.0, 150.0, step=10.0)
    h3, h4 = st.columns(2)
    average_order_value = h3.number_input("Average order value ($)", 0.0, 600.0, 40.0, step=5.0)
    days_since_last = h4.number_input(
        "Days since last purchase", -1, 365, 20, step=1,
        help="Use -1 if the customer has never purchased before (this is how the training data records it).",
    )

    st.subheader("Session Details")
    s1, s2, s3 = st.columns(3)
    device_type = s1.selectbox("Device type", cats["device_type"])
    traffic_source = s2.selectbox("Traffic source", cats["traffic_source"])
    product_category = s3.selectbox("Product category", cats["product_category"])
    s4, s5 = st.columns(2)
    cart_items = s4.number_input("Cart items", 0, 12, 2, step=1)
    page_views = s5.number_input("Page views", 1, 45, 10, step=1)
    session_duration = st.slider("Session duration (minutes)", 0.5, 60.0, 11.0, step=0.5)

    st.subheader("Product Details")
    p1, p2 = st.columns(2)
    product_price = p1.number_input("Product price ($)", 5.0, 1500.0, 60.0, step=5.0)
    discount_percent = p2.slider("Discount percentage (%)", 0, 50, 10)
    product_rating = st.slider("Product rating", 1.0, 5.0, 4.2, step=0.1)

    submitted = st.form_submit_button("PREDICT PURCHASE", type="primary")

# --------------------------------------------------------------- prediction
if submitted:
    # Keep the inputs consistent with how the training data was recorded:
    # a customer with no previous orders has zero spend / AOV and days = -1.
    if previous_orders == 0:
        previous_total_spend, average_order_value, days_since_last = 0.0, 0.0, -1
        is_returning = "No"
        st.caption("ℹ️ Previous orders is 0, so spend and average order value were set to 0, "
                   "days since last purchase to -1, and the customer was treated as new.")
    elif is_returning == "No":
        st.caption("ℹ️ Customer has previous orders, but was marked as not returning. "
                   "In the training data, customers with previous orders are always returning.")

    row = pd.DataFrame([{
        "customer_age": customer_age,
        "product_price": product_price,
        "discount_percent": discount_percent,
        "previous_orders": previous_orders,
        "previous_total_spend": previous_total_spend,
        "average_order_value": average_order_value,
        "days_since_last_purchase": days_since_last,
        "cart_items": cart_items,
        "page_views": page_views,
        "session_duration_minutes": session_duration,
        "product_rating": product_rating,
        "is_returning_customer": 1 if is_returning == "Yes" else 0,
        "gender": gender,
        "city": city,
        "device_type": device_type,
        "traffic_source": traffic_source,
        "product_category": product_category,
    }])[FEATURES]

    # The saved pipeline applies the same scaling + one-hot encoding used in training.
    prediction = int(pipeline.predict(row)[0])
    prob = float(pipeline.predict_proba(row)[0][1])

    if prediction == 1:
        st.markdown('<div class="result-box result-yes">YES! THE CUSTOMER IS LIKELY TO PURCHASE.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box result-no">NO! THE CUSTOMER IS NOT LIKELY TO PURCHASE.</div>',
                    unsafe_allow_html=True)

    st.markdown(f'<p class="prob-text">Estimated purchase probability: <b>{prob * 100:.1f}%</b></p>',
                unsafe_allow_html=True)
    st.progress(min(max(prob, 0.0), 1.0))

# ------------------------------------------------------------ model & info
st.divider()
m1, m2 = st.columns(2)
m1.metric("Selected model", artifact["model_name"])
m2.metric("Test accuracy", f"{artifact['test_accuracy'] * 100:.2f}%")
st.caption(
    f"Evaluated on {artifact['n_test']:,} held-out sessions the model never saw during training. "
    f"Cross-validated accuracy: {artifact['cv_accuracy'] * 100:.2f}%."
)

with st.expander("How does the prediction work?"):
    st.markdown(
        f"""
1. **Your inputs** are collected into a single row, in the same format as the original dataset.
2. **Preprocessing** (saved inside the model file) scales the numeric values and one-hot
   encodes categories such as city, device, and product category, exactly as during training.
3. **The {artifact['model_name']} model** — trained on {artifact['n_train']:,} past shopping
   sessions — calculates the probability that the customer purchases.
4. If that probability is **50% or higher**, the result is **YES**; otherwise **NO**.

*Note: this is a statistical estimate based on patterns in past data, so it will sometimes be
wrong. The model is best used as a guide, not a guarantee.*
        """
    )
