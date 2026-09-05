import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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

# TF-IDF
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_val_tfidf)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("\nValidation Accuracy:")
print(round(accuracy, 4))

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_names
    )
)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

disp.plot(xticks_rotation=45)
plt.title("Emotion Classification - Confusion Matrix")
plt.tight_layout()
plt.show()