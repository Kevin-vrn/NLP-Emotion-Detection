import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD FINAL TEST PREDICTIONS
# ============================================================

df = pd.read_csv(
    "results/test_predictions.csv"
)


# ============================================================
# LABEL NAMES
# ============================================================

label_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# ============================================================
# CONVERT LABELS TO EMOTION NAMES
# ============================================================

df["actual_emotion"] = df["label"].map(label_names)

df["predicted_emotion"] = df["predicted_label"].map(label_names)


# ============================================================
# FIND INCORRECT PREDICTIONS
# ============================================================

errors = df[
    df["label"] != df["predicted_label"]
].copy()


# ============================================================
# ERROR COUNT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST ERROR ANALYSIS")
print("=" * 70)

print(
    "\nTotal test samples:",
    len(df)
)

print(
    "Incorrect predictions:",
    len(errors)
)

print(
    "Error rate:",
    round(
        (len(errors) / len(df)) * 100,
        2
    ),
    "%"
)


# ============================================================
# MOST COMMON CONFUSION PAIRS
# ============================================================

confusion_pairs = (
    errors
    .groupby(
        [
            "actual_emotion",
            "predicted_emotion"
        ]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("MOST COMMON CONFUSION PAIRS")
print("=" * 70)

print(
    confusion_pairs.to_string(
        index=False
    )
)


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


errors.to_csv(
    "results/incorrect_test_predictions.csv",
    index=False
)


confusion_pairs.to_csv(
    "results/test_confusion_pairs.csv",
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

    print("\nActual:",
          row["actual_emotion"])

    print("Predicted:",
          row["predicted_emotion"])

    print("Text:",
          row["text"])

    print("-" * 70)


# ============================================================
# CREATE CONFUSION PAIR GRAPH
# ============================================================

top_pairs = confusion_pairs.head(10).copy()


top_pairs["pair"] = (
    top_pairs["actual_emotion"]
    + " → "
    + top_pairs["predicted_emotion"]
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    top_pairs["pair"],
    top_pairs["count"]
)


plt.xlabel(
    "Actual → Predicted"
)

plt.ylabel(
    "Number of Errors"
)

plt.title(
    "Most Common Emotion Classification Errors"
)


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


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(
    "1. results/incorrect_test_predictions.csv"
)

print(
    "2. results/test_confusion_pairs.csv"
)

print(
    "3. results/top_confusion_pairs.png"
)