import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean
df = df.dropna()
df["text"] = df["text"].str.lower().str.strip()

X = df["text"]
y = df["label"]

# Train-test split (IMPORTANT FOR REAL ACCURACY)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# BEST MODEL PIPELINE
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,3),   # ⭐ IMPROVED (better context)
        max_features=15000   # ⭐ INCREASED FEATURES
    )),
    ("clf", LinearSVC(C=1.5))  # ⭐ BETTER TUNING
])

# Train
model.fit(X_train, y_train)

# Accuracy check
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "news_model.pkl")

print("Model trained successfully!")