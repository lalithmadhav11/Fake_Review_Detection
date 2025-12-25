import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

from preprocess import clean_text


# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/reviews.csv", sep="\t", engine="python")

# ===============================
# 2. Preprocess Text
# ===============================
df["clean_text"] = df["reviewContent"].astype(str).apply(clean_text)

X = df["clean_text"]
y = df["flagged"]   # 0 = genuine, 1 = fake

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Word-level TF-IDF
# ===============================
word_vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

# ===============================
# 5. Character-level TF-IDF (KEY IMPROVEMENT)
# ===============================
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

# ===============================
# 6. Combine Features
# ===============================
X_train_vec = hstack([X_train_word, X_train_char])
X_test_vec = hstack([X_test_word, X_test_char])

# ===============================
# 7. Train Model
# ===============================
model = LogisticRegression(
    max_iter=2000,
    C=1.5,
    solver="liblinear"
)

model.fit(X_train_vec, y_train)

# ===============================
# 8. Evaluation
# ===============================
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9. Save Artifacts
# ===============================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_review_model.pkl")
joblib.dump(word_vectorizer, "models/word_vectorizer.pkl")
joblib.dump(char_vectorizer, "models/char_vectorizer.pkl")

print("\nModel and vectorizers saved successfully.")
