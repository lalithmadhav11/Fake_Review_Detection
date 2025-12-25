
##  Overview

Fake or paid reviews can mislead users and damage trust on online platforms.  
Instead of treating fake review detection as a strict binary classification problem, this project implements an **explainable risk-based system** that estimates how suspicious a review is.

The system outputs a **Suspicion Score (0–100)** and categorizes reviews into:
- 🟢 Low Risk  
- 🟠 Medium Risk  
- 🔴 High Risk  

This approach aligns with how real-world platforms handle deceptive content.



##  Key Features

- Risk-based prediction instead of fake/genuine labeling  
- Hybrid approach: Machine Learning + Rule-Based heuristics  
- Word-level and Character-level TF-IDF feature extraction  
- Explainable output with score breakdown  
- Interactive Streamlit web application  


##  Dataset

- **Dataset:** Yelp Fake Review Dataset  
- **Labels:**
  - `0` → Genuine  
  - `1` → Fake  

### Why this dataset?
- Realistic and noisy
- Fake reviews closely resemble genuine ones
- Highlights limitations of text-only ML
- Commonly used in research and benchmarking



##  System Architecture
User Review
↓
Text Preprocessing
↓
Word + Character TF-IDF
↓
Logistic Regression (ML Risk Score)
↓
Rule-Based Heuristics
↓
Weighted Aggregation
↓
Final Suspicion Score (0–100)


---

## 🔧 Machine Learning Pipeline

### Text Preprocessing
- Lowercasing
- Stopword removal
- Lemmatization

### Feature Engineering
- **Word TF-IDF (1–2 grams):** captures semantic meaning  
- **Character TF-IDF (3–5 grams):** captures stylistic patterns such as repetition and emphasis  

### Model
- **Logistic Regression**
  - Suitable for sparse, high-dimensional text data
  - Interpretable and stable
  - Industry-standard baseline for NLP tasks



##  Rule-Based Heuristics

To improve explainability and robustness, rule-based signals were added:

- Very short reviews  
- Excessive exclamation marks  
- Uppercase emphasis  
- Marketing phrases (e.g., *best ever*, *must buy*)  
- Low vocabulary diversity  

Each rule contributes to a rule-based risk score.


##  Aggregation Strategy

Final suspicion score is calculated as:
Final Risk Score = (0.7 × ML Risk Score) + (0.3 × Rule-Based Risk Score)


This balances learned linguistic patterns with deterministic signals.



## Web Application

The Streamlit application allows users to:
1. Enter a review
2. Analyze its suspiciousness
3. View:
   - Final risk score
   - Risk category
   - ML vs rule-based score breakdown



##  Project Structure

fake-review-detection/
│
├── app.py
├── requirements.txt
├── models/
│ ├── fake_review_model.pkl
│ ├── word_vectorizer.pkl
│ └── char_vectorizer.pkl
│
├── src/
│ ├── preprocess.py
│ └── rules.py
│
├── data/
│ └── reviews.csv
│
└── README.md

## Sample Inputs
🔴 High Risk
BEST PRODUCT EVER!!! LIFE CHANGING!!! MUST BUY!!!
Highly recommended to everyone!!!

🟢 Low Risk
I visited the restaurant with my family on Sunday evening.
We ordered dosa and coffee. Food arrived in about 15 minutes.
Parking was limited but overall experience was good.

## Note on Accuracy

Fake review detection is a fraud-style problem.
On realistic datasets, classical ML models typically plateau around 65–75% accuracy.

This project prioritizes:

Correct problem framing

Explainability

Practical usability over raw accuracy 
