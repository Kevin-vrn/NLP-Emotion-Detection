import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("dataset/train_processed.csv")

X = df["clean_text"]
y = df["label"]


# ============================================================
# LABEL NAMES
# ============================================================

label_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


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


# ============================================================
# TRAIN FINAL LINEAR SVM
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
# CREATE ERROR ANALYSIS DATAFRAME
# ============================================================

error_df = pd.DataFrame({
    "text": X_val.values,
    "actual_label": y_val.values,
    "predicted_label": y_pred
})


# Convert numeric labels to emotion names

error_df["actual_emotion"] = error_df["actual_label"].map(
    dict(enumerate(label_names))
)

error_df["predicted_emotion"] = error_df["predicted_label"].map(
    dict(enumerate(label_names))
)


# Keep only incorrect predictions

errors = error_df[
    error_df["actual_label"] != error_df["predicted_label"]
].copy()


# ============================================================
# ERROR COUNT
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print("\nTotal validation samples:", len(error_df))

print("Incorrect predictions:", len(errors))

print(
    "Error rate:",
    round((len(errors) / len(error_df)) * 100, 2),
    "%"
)


# ============================================================
# MOST COMMON CONFUSION PAIRS
# ============================================================

confusion_pairs = (
    errors
    .groupby(
        ["actual_emotion", "predicted_emotion"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)


print("\n" + "=" * 70)
print("MOST COMMON CONFUSION PAIRS")
print("=" * 70)

print(confusion_pairs.to_string(index=False))


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

os.makedirs("results", exist_ok=True)

errors.to_csv(
    "results/incorrect_predictions.csv",
    index=False
)

confusion_pairs.to_csv(
    "results/confusion_pairs.csv",
    index=False
)


# ============================================================
# DISPLAY TOP 20 ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 INCORRECT PREDICTIONS")
print("=" * 70)

top_errors = errors.head(20)

for index, row in top_errors.iterrows():

    print("\nText:", row["text"])
    print("Actual:", row["actual_emotion"])
    print("Predicted:", row["predicted_emotion"])
    print("-" * 70)


# ============================================================
# CONFUSION PAIR GRAPH
# ============================================================

top_pairs = confusion_pairs.head(10).copy()

top_pairs["pair"] = (
    top_pairs["actual_emotion"]
    + " → "
    + top_pairs["predicted_emotion"]
)

plt.figure(figsize=(10, 6))

plt.bar(
    top_pairs["pair"],
    top_pairs["count"]
)

plt.xlabel("Actual → Predicted")

plt.ylabel("Number of Errors")

plt.title("Most Common Emotion Classification Errors")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/top_confusion_pairs.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved files:")
print("1. results/incorrect_predictions.csv")
print("2. results/confusion_pairs.csv")
print("3. results/top_confusion_pairs.png")