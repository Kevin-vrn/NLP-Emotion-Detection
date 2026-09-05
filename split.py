import pandas as pd
from sklearn.model_selection import train_test_split

# Load processed training dataset
df = pd.read_csv("dataset/train_processed.csv")

# Input and target
X = df["clean_text"]
y = df["label"]

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=" * 70)
print("TRAINING AND VALIDATION SPLIT")
print("=" * 70)

print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation label distribution:")
print(y_val.value_counts().sort_index())