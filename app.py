from pathlib import Path

import altair as alt
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
    .block-container {padding-top: 2rem; max-width: 1000px;}
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
report = artifact.get("report")  # evaluation + analytics data saved by train_model.py

FRIENDLY = {
    "customer_age": "Customer age", "gender": "Gender", "city": "City",
    "device_type": "Device type", "traffic_source": "Traffic source",
    "product_category": "Product category", "product_price": "Product price",
    "discount_percent": "Discount %", "previous_orders": "Previous orders",
    "previous_total_spend": "Previous total spend", "average_order_value": "Average order value",
    "days_since_last_purchase": "Days since last purchase", "cart_items": "Cart items",
    "page_views": "Page views", "session_duration_minutes": "Session duration (min)",
    "product_rating": "Product rating", "is_returning_customer": "Returning customer",
}
BLUE, GREEN, RED, GREY = "#2563eb", "#16a34a", "#dc2626", "#9ca3af"


def pretty_coef(name):
    for c in cats:  # one-hot columns look like "device_type_Mobile"
        if name.startswith(c + "_"):
            return f"{FRIENDLY[c]}: {name[len(c) + 1:]}"
    return FRIENDLY.get(name, name)


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------- header
st.markdown('<p class="main-title">🛒 E-Commerce Customer Purchase Prediction System</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter a customer\'s shopping-session details below and the machine '
    'learning model will predict whether the customer is likely to purchase the product.</p>',
    unsafe_allow_html=True,
)

tab_predict, tab_analytics = st.tabs(["🔮 Predict Purchase", "📊 Model Analytics"])

# =========================================================== TAB 1: PREDICT
with tab_predict:
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

    # ---------------------------------------------------------- model & info
    st.divider()
    st.markdown("##### Model performance (on unseen test data)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Selected model", artifact["model_name"])
    a2.metric("Accuracy", pct(artifact["test_accuracy"]))
    a3.metric("ROC-AUC", f"{artifact['roc_auc']:.3f}")
    if report:
        bm = report["best_metrics"]
        b1, b2, b3 = st.columns(3)
        b1.metric("Precision", pct(bm["precision"]))
        b2.metric("Recall", pct(bm["recall"]))
        b3.metric("F1 score", pct(bm["f1"]))
    st.caption(
        f"Evaluated on {artifact['n_test']:,} held-out sessions the model never saw during training. "
        f"Cross-validated accuracy: {artifact['cv_accuracy'] * 100:.2f}%. "
        "See the **Model Analytics** tab for charts, the tests used, and the most influential variables."
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

# ========================================================= TAB 2: ANALYTICS
with tab_analytics:
    if not report:
        st.warning("This model file has no analytics data. Run `python train_model.py "
                   "ecommerce_purchase_prediction.csv` to regenerate `best_model.pkl`, then restart the app.")
        st.stop()

    bm, ds, res = report["best_metrics"], report["dataset"], report["results"]
    sel = report["selected"]
    cm = report["confusion_matrix"]

    # ---------------------------------------------------------- 1. overview
    st.subheader("1. Model performance")
    st.caption(f"Selected model: **{sel}** · scores below are from the {ds['n_test']:,} test sessions "
               "that were hidden from the model during training.")
    r1 = st.columns(3)
    r1[0].metric("Accuracy", pct(bm["test_accuracy"]))
    r1[1].metric("Precision", pct(bm["precision"]))
    r1[2].metric("Recall (sensitivity)", pct(bm["recall"]))
    r2 = st.columns(3)
    r2[0].metric("F1 score", pct(bm["f1"]))
    r2[1].metric("ROC-AUC", f"{bm['roc_auc']:.3f}")
    r2[2].metric("Specificity", pct(bm["specificity"]))

    gain = (bm["test_accuracy"] - report["baseline_accuracy"]) * 100
    st.info(
        f"**Baseline check:** a lazy model that always says “will not purchase” would be right "
        f"{pct(report['baseline_accuracy'])} of the time. Our model reaches {pct(bm['test_accuracy'])}, "
        f"an improvement of **{gain:.1f} percentage points**."
    )

    with st.expander("What do these metrics mean?"):
        st.markdown(
            f"""
- **Accuracy ({pct(bm['test_accuracy'])})** – out of every 100 customers, about {bm['test_accuracy'] * 100:.0f} are classified correctly.
- **Precision ({pct(bm['precision'])})** – when the model says *“will purchase”*, it is right about {bm['precision'] * 100:.0f}% of the time.
- **Recall ({pct(bm['recall'])})** – of the customers who *really did* purchase, the model finds about {bm['recall'] * 100:.0f}%.
- **F1 score ({pct(bm['f1'])})** – a single number that balances precision and recall.
- **Specificity ({pct(bm['specificity'])})** – of the customers who did *not* purchase, the model correctly identifies about {bm['specificity'] * 100:.0f}%.
- **ROC-AUC ({bm['roc_auc']:.3f})** – how well the model ranks real buyers above non-buyers. 0.5 is a coin flip and 1.0 is perfect.

**Reading the results:** the model is more cautious than generous. It is good at spotting
customers who will *not* buy ({pct(bm['specificity'], 0)}) but misses roughly half of the real buyers
({pct(1 - bm['recall'], 0)} are missed). This is typical when the data has no strong single signal.
            """
        )

    # ------------------------------------------------- 2. model comparison
    st.subheader("2. Comparison of all models tried")
    comp = pd.DataFrame([{
        "Model": n, "Accuracy": r["test_accuracy"], "Precision": r["precision"],
        "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"],
        "CV Accuracy": r["cv_accuracy_mean"],
    } for n, r in res.items()])
    show = comp.copy()
    show["Selected"] = show["Model"].map(lambda m: "✅" if m == sel else "")
    st.dataframe(show.set_index("Model").style.format(
        {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Accuracy"]}),
        width="stretch")

    long = comp.drop(columns="CV Accuracy").melt("Model", var_name="Metric", value_name="Score")
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("Metric:N", title=None, sort=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                axis=alt.Axis(labelAngle=0)),
        xOffset="Model:N",
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
    ).properties(height=320)
    st.altair_chart(bars, width="stretch")
    st.caption(f"**{sel}** was selected because it had the highest cross-validated accuracy "
               "(chosen on training data only, so the test set stays an honest final check). "
               "All four models score within a few points of each other.")

    # ------------------------------------- 3. confusion matrix + ROC curve
    st.subheader("3. Where the model is right and wrong")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (test set)")
        cdf = pd.DataFrame([
            {"Actual": "Did not purchase", "Predicted": "Predicted: No", "Count": cm["tn"], "Kind": "Correct"},
            {"Actual": "Did not purchase", "Predicted": "Predicted: Yes", "Count": cm["fp"], "Kind": "Wrong"},
            {"Actual": "Purchased", "Predicted": "Predicted: No", "Count": cm["fn"], "Kind": "Wrong"},
            {"Actual": "Purchased", "Predicted": "Predicted: Yes", "Count": cm["tp"], "Kind": "Correct"},
        ])
        base = alt.Chart(cdf).encode(
            x=alt.X("Predicted:N", title=None, axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("Actual:N", title=None, sort=["Did not purchase", "Purchased"]),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Actual", "Predicted", "Count"])
        txt = base.mark_text(fontSize=22, fontWeight="bold").encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > cdf["Count"].max() / 2,
                                alt.value("white"), alt.value("#111827")))
        st.altair_chart((heat + txt).properties(height=260), width="stretch")
        st.caption(f"✅ Correct: {cm['tn'] + cm['tp']:,} · ❌ Wrong: {cm['fp'] + cm['fn']:,}. "
                   f"{cm['fn']:,} real buyers were missed; {cm['fp']:,} non-buyers were wrongly flagged.")
    with right:
        st.markdown("**ROC curves** (higher & further left is better)")
        rows = []
        for name, d in report["roc_curves"].items():
            lbl = f"{name} (AUC {res[name]['roc_auc']:.3f})"
            rows += [{"Model": lbl, "FPR": f, "TPR": t, "Selected": name == sel}
                     for f, t in zip(d["fpr"], d["tpr"])]
        roc = pd.DataFrame(rows)
        lines = alt.Chart(roc).mark_line().encode(
            x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None, columns=1)),
            strokeWidth=alt.condition(alt.datum.Selected, alt.value(3.5), alt.value(1.5)),
            tooltip=["Model", alt.Tooltip("FPR:Q", format=".2f"), alt.Tooltip("TPR:Q", format=".2f")])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[5, 5], color=GREY).encode(x="x:Q", y="y:Q")
        st.altair_chart((lines + diag).properties(height=260), width="stretch")
        st.caption("The dashed line is random guessing (AUC 0.5).")

    # ------------------------------------------------ 4. tests performed
    st.subheader("4. How was the model tested?")
    folds = res[sel]["cv_accuracy_folds"]
    tests = pd.DataFrame([
        {"Test": "Train / test split",
         "What it does": f"{ds['n_train']:,} sessions (80%) train the model; {ds['n_test']:,} (20%) are locked away "
                         f"and used only for the final exam. “Stratified” keeps the same {pct(ds['purchase_rate'], 1)} buyer ratio in both.",
         "Result": f"Test accuracy {pct(bm['test_accuracy'])}"},
        {"Test": "5-fold cross-validation",
         "What it does": "Training data is cut into 5 parts. The model is trained 5 times, each time tested on a "
                         "different part. This shows the score isn't just luck from one lucky split.",
         "Result": f"{pct(res[sel]['cv_accuracy_mean'])} ± {res[sel]['cv_accuracy_std'] * 100:.2f}%"},
        {"Test": "Baseline comparison",
         "What it does": "Compares the model with a dummy that always predicts the most common outcome (“no purchase”).",
         "Result": f"Baseline {pct(report['baseline_accuracy'])} → model +{gain:.1f} pts"},
        {"Test": "Accuracy, precision, recall, F1",
         "What it does": "Measure overall correctness, how trustworthy a “YES” is, how many real buyers are found, "
                         "and the balance between the last two.",
         "Result": f"{pct(bm['test_accuracy'], 1)} / {pct(bm['precision'], 1)} / {pct(bm['recall'], 1)} / {pct(bm['f1'], 1)}"},
        {"Test": "Confusion matrix",
         "What it does": "Counts exactly how many predictions were right and wrong, split by type of mistake.",
         "Result": f"TP {cm['tp']:,} · TN {cm['tn']:,} · FP {cm['fp']:,} · FN {cm['fn']:,}"},
        {"Test": "ROC-AUC",
         "What it does": "Checks how well the model ranks buyers above non-buyers at every possible cut-off, "
                         "not just the 50% cut-off.",
         "Result": f"Test {bm['roc_auc']:.3f} · CV {res[sel]['cv_roc_auc_mean']:.3f}"},
        {"Test": "Multi-model comparison",
         "What it does": "Logistic Regression, Decision Tree, Random Forest and Gradient Boosting were all trained "
                         "on the same data and scored the same way. The best cross-validated accuracy was chosen.",
         "Result": f"Selected: {sel}"},
        {"Test": "Permutation importance",
         "What it does": "Shuffles one variable at a time on the test data and measures how much the model gets worse. "
                         "Used to find the variables that matter most (section 5).",
         "Result": "17 variables tested"},
    ]).set_index("Test")
    st.table(tests)

    fdf = pd.DataFrame({"Fold": [f"Fold {i + 1}" for i in range(len(folds))], "Accuracy": folds})
    fbars = alt.Chart(fdf).mark_bar(color=BLUE).encode(
        x=alt.X("Fold:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=["Fold", alt.Tooltip("Accuracy:Q", format=".2%")])
    ftxt = fbars.mark_text(dy=-8, fontWeight="bold").encode(text=alt.Text("Accuracy:Q", format=".1%"))
    rule = alt.Chart(pd.DataFrame({"y": [report["baseline_accuracy"]]})).mark_rule(
        color=RED, strokeDash=[6, 4], size=2).encode(y="y:Q")
    st.markdown(f"**Cross-validation accuracy per fold** — red dashed line = baseline ({pct(report['baseline_accuracy'], 1)})")
    st.altair_chart((fbars + ftxt + rule).properties(height=260), width="stretch")
    st.caption("Scores are very similar across all 5 folds, which means the model is stable and not overfitting to one slice of the data.")

    # ----------------------------------------------- 5. variable importance
    st.subheader("5. Which variables affect purchases the most?")
    imp = pd.DataFrame(report["permutation_importance"])
    imp["Variable"] = imp["feature"].map(FRIENDLY)
    top3 = imp.head(3)["Variable"].tolist()
    useless = imp[imp["importance"] < 0.001]["Variable"].tolist()
    msg = f"**Top 3 drivers:** {top3[0]}, {top3[1]}, and {top3[2]}."
    if useless:
        msg += f" **Almost no effect:** {', '.join(useless)}."
    st.success(msg)

    ibars = alt.Chart(imp).mark_bar().encode(
        x=alt.X("importance:Q", title="Drop in ROC-AUC when this variable is shuffled (bigger = more important)"),
        y=alt.Y("Variable:N", sort=None, title=None),
        color=alt.condition(alt.datum.importance >= 0.005, alt.value(BLUE), alt.value(GREY)),
        tooltip=["Variable", alt.Tooltip("importance:Q", title="Importance", format=".4f")],
    ).properties(height=440)
    st.altair_chart(ibars, width="stretch")
    st.caption("Blue = meaningful influence · Grey = little or none. Values at or near zero mean the model "
               "doesn't use that variable in any useful way.")

    if report["coefficients"]:
        st.markdown("**Direction of influence** — does it push towards buying or away from it?")
        coef = pd.DataFrame(report["coefficients"]).head(12)
        coef["Variable"] = coef["feature"].map(pretty_coef)
        cbars = alt.Chart(coef).mark_bar().encode(
            x=alt.X("coefficient:Q", title="← less likely to purchase      more likely to purchase →"),
            y=alt.Y("Variable:N", sort=None, title=None),
            color=alt.condition(alt.datum.coefficient > 0, alt.value(GREEN), alt.value(RED)),
            tooltip=["Variable", alt.Tooltip("coefficient:Q", format=".3f")],
        ).properties(height=380)
        st.altair_chart(cbars, width="stretch")
        st.caption("Green bars increase the chance of purchase, red bars reduce it. "
                   "For numeric variables, the bar shows the effect of a 1-standard-deviation increase.")

    with st.expander("Simple correlation with purchase (numeric variables)"):
        corr = pd.DataFrame(report["correlations"])
        corr["Variable"] = corr["feature"].map(FRIENDLY)
        corr_chart = alt.Chart(corr).mark_bar().encode(
            x=alt.X("correlation:Q", title="Correlation with purchased"),
            y=alt.Y("Variable:N", sort=None, title=None),
            color=alt.condition(alt.datum.correlation > 0, alt.value(GREEN), alt.value(RED)),
            tooltip=["Variable", alt.Tooltip("correlation:Q", format=".3f")],
        ).properties(height=380)
        st.altair_chart(corr_chart, width="stretch")
        st.caption("Correlation looks at each variable on its own; permutation importance (above) "
                   "shows what the model actually relies on. Page views and session duration correlate with "
                   "purchase but overlap with cart items, so the model leans less on them.")

    # ---------------------------------------------------- 6. explore data
    st.subheader("6. Explore the data")
    s = st.columns(4)
    s[0].metric("Total sessions", f"{ds['rows']:,}")
    s[1].metric("Overall purchase rate", pct(ds["purchase_rate"], 1))
    s[2].metric("Training / test", f"{ds['n_train']:,} / {ds['n_test']:,}")
    s[3].metric("Input variables", ds["n_features"])

    ordered = imp["feature"].tolist()  # most important first
    var = st.selectbox("Show purchase rate by:", ordered, format_func=lambda f: FRIENDLY[f])
    rdf = pd.DataFrame(report["purchase_rate_by"][var]).rename(
        columns={"group": "Group", "purchase_rate": "Purchase rate", "sessions": "Sessions"})
    rbars = alt.Chart(rdf).mark_bar(color=BLUE).encode(
        x=alt.X("Group:N", sort=None, title=FRIENDLY[var], axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("Purchase rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Group", alt.Tooltip("Purchase rate:Q", format=".1%"), "Sessions"])
    avg = alt.Chart(pd.DataFrame({"y": [ds["purchase_rate"]]})).mark_rule(
        color=RED, strokeDash=[6, 4], size=2).encode(y="y:Q")
    st.altair_chart((rbars + avg).properties(height=320), width="stretch")
    st.caption(f"Red dashed line = overall purchase rate ({pct(ds['purchase_rate'], 1)}). "
               "Numeric variables are grouped into ranges of roughly equal size.")

    st.markdown("**How well does the model separate buyers from non-buyers?**")
    hist = pd.DataFrame(report["probability_histogram"]).melt(
        id_vars=["bin", "start"], var_name="Actual outcome", value_name="Customers")
    hchart = alt.Chart(hist).mark_bar().encode(
        x=alt.X("bin:N", sort=alt.EncodingSortField("start"), title="Predicted purchase probability",
                axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("Customers:Q"),
        color=alt.Color("Actual outcome:N", scale=alt.Scale(
            domain=["Did not purchase", "Purchased"], range=[GREY, GREEN]),
            legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["bin", "Actual outcome", "Customers"]).properties(height=300)
    st.altair_chart(hchart, width="stretch")
    st.caption("Ideally the grey bars sit on the left (low probability) and the green bars on the right. "
               "The overlap in the middle is where the model is unsure and mistakes happen.")
