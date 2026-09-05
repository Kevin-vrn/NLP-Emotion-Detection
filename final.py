import os
import re
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

import matplotlib.pyplot as plt


# ==================================================
# 1. SETUP
# ==================================================

nltk.download("punkt")
nltk.download("stopwords")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==================================================
# 2. TEXT PREPROCESSING
# ==================================================

stop_words = set(stopwords.words("english"))


def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words back together
    return " ".join(words)


# ==================================================
# 3. LOAD DATASETS
# ==================================================

train_df = pd.read_csv("dataset/train.csv")
validation_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Training samples   :", len(train_df))
print("Validation samples :", len(validation_df))
print("Test samples       :", len(test_df))


# ==================================================
# 4. PREPROCESS DATA
# ==================================================

print("\nPreprocessing datasets...")

train_df["clean_text"] = train_df["text"].apply(preprocess_text)
validation_df["clean_text"] = validation_df["text"].apply(preprocess_text)
test_df["clean_text"] = test_df["text"].apply(preprocess_text)

print("Preprocessing completed.")


# ==================================================
# 5. SEPARATE FEATURES AND LABELS
# ==================================================

X_train = train_df["clean_text"]
y_train = train_df["label"]

X_validation = validation_df["clean_text"]
y_validation = validation_df["label"]

X_test = test_df["clean_text"]
y_test = test_df["label"]


# ==================================================
# 6. TF-IDF FEATURE EXTRACTION
# ==================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_validation_tfidf = tfidf.transform(X_validation)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed.")

print("Training matrix   :", X_train_tfidf.shape)
print("Validation matrix :", X_validation_tfidf.shape)
print("Test matrix       :", X_test_tfidf.shape)


# ==================================================
# 7. DEFINE MODELS
# ==================================================

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


# ==================================================
# 8. MODEL COMPARISON
# ==================================================

results = {}

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    validation_pred = model.predict(X_validation_tfidf)

    accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    precision = precision_score(
        y_validation,
        validation_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_pred,
        average="macro",
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ==================================================
# 9. SAVE MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results).T

results_df.to_csv(
    "results/model_comparison.csv"
)

print("\nModel comparison saved to:")
print("results/model_comparison.csv")


# ==================================================
# 10. SELECT BEST MODEL
# ==================================================

best_model_name = results_df["F1 Score"].idxmax()

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected model:", best_model_name)


# ==================================================
# 11. FINAL MODEL
# ==================================================

final_model = models[best_model_name]

final_model.fit(
    X_train_tfidf,
    y_train
)


# ==================================================
# 12. SAVE MODEL AND VECTORIZER
# ==================================================

joblib.dump(
    final_model,
    "models/svm_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved to:")
print("models/svm_model.pkl")

print("TF-IDF vectorizer saved to:")
print("models/tfidf_vectorizer.pkl")


# ==================================================
# 13. FINAL TEST PREDICTIONS
# ==================================================

test_pred = final_model.predict(
    X_test_tfidf
)


# ==================================================
# 14. FINAL TEST ACCURACY
# ==================================================

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ==================================================
# 15. CLASSIFICATION REPORT
# ==================================================

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

report = classification_report(
    y_test,
    test_pred,
    target_names=label_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# Save classification report
with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==================================================
# 16. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    test_pred
)

print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

display.plot(
    xticks_rotation=45
)

plt.title("Emotion Detection - Linear SVM")

plt.tight_layout()

plt.savefig(
    "results/final_confusion_matrix.png",
    dpi=300
)

plt.show()


print("\nConfusion matrix saved to:")
print("results/final_confusion_matrix.png")


# ==================================================
# 17. SAVE TEST PREDICTIONS
# ==================================================

prediction_df = test_df[
    ["text", "label"]
].copy()

prediction_df["predicted_label"] = test_pred

prediction_df.to_csv(
    "results/test_predictions.csv",
    index=False
)

print("\nTest predictions saved to:")
print("results/test_predictions.csv")


# ==================================================
# 18. FINISHED
# ==================================================

print("\n" + "=" * 70)
print("PROJECT TRAINING AND EVALUATION COMPLETED")
print("=" * 70)