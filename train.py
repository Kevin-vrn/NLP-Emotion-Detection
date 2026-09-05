import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load processed dataset
df = pd.read_csv("dataset/train_processed.csv")

# Input and target
X = df["clean_text"]
y = df["label"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit TF-IDF only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform validation data
X_val_tfidf = tfidf.transform(X_val)

print("=" * 70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 70)

print("Training features:", X_train_tfidf.shape)
print("Validation features:", X_val_tfidf.shape)

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")

# Training accuracy
train_accuracy = model.score(X_train_tfidf, y_train)

# Validation accuracy
val_accuracy = model.score(X_val_tfidf, y_val)

print("\n" + "=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

print("Training Accuracy:", round(train_accuracy, 4))
print("Validation Accuracy:", round(val_accuracy, 4))