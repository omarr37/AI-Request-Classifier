import streamlit as st
from model import load_model
from preprocess import preprocess_message as preprocess_text
from real_helpers import classify_request, generate_agent_response
import pandas as pd

# Load model once
model, _ = load_model()
# Track classified results
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI setup
st.set_page_config(page_title="AI Request Classifier", layout="centered")
st.title("🤖 Smart Request Classifier + Agent Response")

st.markdown("Enter a service request below and get an instant classification with a smart response.")

# Input field
user_input = st.text_area("📝 Your Request", placeholder="e.g., My printer is not working...")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess input
        cleaned = preprocess_text(user_input)
        category = classify_request(cleaned, model)
        response = generate_agent_response(category)

        # Show results
        st.success(f"🔍 Classification: **{category}**")
        st.info(f"🤖 Agent Response: {response}")

        # Save to history
        st.session_state.history.append({"Request": user_input, "Category": category})
    else:
        st.warning("Please enter a request.")

# Divider
st.markdown("---")

# Dashboard Preview
st.subheader("📊 Dashboard: Request Distribution")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    chart_data = df["Category"].value_counts().reset_index()
    chart_data.columns = ["Category", "Count"]
    st.bar_chart(chart_data.set_index("Category"))
else:
    st.info("No data yet. Classify some requests to see the dashboard.")

# Evaluation Report section (always shown)
st.subheader("📋 Model Evaluation Report")

try:
    report_df = pd.read_csv("classification_report.csv")
    st.dataframe(
        report_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        })
    )
except FileNotFoundError:
    st.warning("No evaluation report found. Please retrain the model.")


# Footer
st.markdown("---")
st.caption("2025 Omar Alreshidi. All rights reserved.")
