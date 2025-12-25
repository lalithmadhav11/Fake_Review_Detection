import streamlit as st
import joblib
from scipy.sparse import hstack

from src.preprocess import clean_text
from src.rules import rule_based_score

# ===============================
# Load model and vectorizers
# ===============================
model = joblib.load("models/fake_review_model.pkl")
word_vectorizer = joblib.load("models/word_vectorizer.pkl")
char_vectorizer = joblib.load("models/char_vectorizer.pkl")

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Fake Review Risk Scoring",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Review Risk Scoring System")
st.write(
    "This system assigns a **Suspicion Score (0–100)** to online reviews "
    "using Machine Learning and explainable rule-based heuristics."
)

# ===============================
# User input
# ===============================
review = st.text_area(
    "Enter a review:",
    height=180,
    placeholder="Paste a product or restaurant review here..."
)

# ===============================
# Prediction
# ===============================
if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Clean
        cleaned = clean_text(review)

        # Vectorize (WORD + CHAR)
        word_vec = word_vectorizer.transform([cleaned])
        char_vec = char_vectorizer.transform([cleaned])
        vector = hstack([word_vec, char_vec])

        # ML decision score
        decision_score = model.decision_function(vector)[0]

        # Normalize ML score → 0–100
        ml_risk = int((decision_score + 4) / 8 * 100)
        ml_risk = max(0, min(100, ml_risk))

        # Rule-based score → 0–100
        rule_score = rule_based_score(review)
        rule_risk = int((rule_score / 5) * 100)

        # Aggregation
        final_risk = int((ml_risk * 0.7) + (rule_risk * 0.3))

        # Output
        if final_risk >= 70:
            st.error("🔴 High Risk Review")
        elif final_risk >= 40:
            st.warning("🟠 Medium Risk Review")
        else:
            st.success("🟢 Low Risk Review")

        st.progress(final_risk)
        st.write(f"**Final Suspicion Score:** {final_risk} / 100")

        st.markdown("### 🔍 Risk Breakdown")
        st.write(f"- ML Risk Score: {ml_risk}/100")
        st.write(f"- Rule-Based Risk Score: {rule_risk}/100")

st.markdown("---")
st.caption("Explainable Fake Review Detection using Word + Character TF-IDF, ML, and Rules")
