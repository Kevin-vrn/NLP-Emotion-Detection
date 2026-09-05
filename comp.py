import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

print("=" * 70)
print("TF-IDF FEATURE MATRIX")
print("=" * 70)

print("Training shape:", X_train_tfidf.shape)
print("Validation shape:", X_val_tfidf.shape)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

best_svm_model = None
best_svm_predictions = None


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_val_tfidf)

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # --------------------------------------------------------
    # SAVE SVM RESULTS FOR FINAL EVALUATION
    # --------------------------------------------------------

    if model_name == "Linear SVM":

        best_svm_model = model
        best_svm_predictions = y_pred


# ============================================================
# FINAL LINEAR SVM EVALUATION
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
print("FINAL LINEAR SVM EVALUATION")
print("=" * 70)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val,
        best_svm_predictions,
        target_names=label_names,
        zero_division=0
    )
)


# ============================================================
# FINAL SVM CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    best_svm_predictions
)

print("\n" + "=" * 70)
print("FINAL LINEAR SVM CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE FINAL CONFUSION MATRIX
# ============================================================

os.makedirs("results", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

disp.plot(xticks_rotation=45)

plt.title("Linear SVM - Emotion Classification")

plt.tight_layout()

plt.savefig(
    "results/final_svm_confusion_matrix.png",
    dpi=300
)

plt.show()