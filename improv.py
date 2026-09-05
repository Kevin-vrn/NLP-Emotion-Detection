import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("dataset/train_processed.csv")

X = df["clean_text"]
y = df["label"]

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# TF-IDF WITH UNIGRAMS + BIGRAMS
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

print("TF-IDF feature matrix:")
print(X_train_tfidf.shape)

# ============================================================
# TRAIN SVM
# ============================================================

model = LinearSVC(
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_val_tfidf)

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_val, y_pred)

print("\n" + "=" * 70)
print("IMPROVED SVM MODEL EVALUATION")
print("=" * 70)

print("\nValidation Accuracy:")
print(round(accuracy, 4))

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_names
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, y_pred)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

os.makedirs("results", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

disp.plot(xticks_rotation=45)

plt.title("Improved SVM - Emotion Classification")
plt.tight_layout()

plt.savefig(
    "results/improved_svm_confusion_matrix.png",
    dpi=300
)

plt.show()