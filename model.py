import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed dataset
df = pd.read_csv("dataset/train_processed.csv")

# Input and target
X = df["clean_text"]
y = df["label"]

# Split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform validation data using the same vocabulary
X_val_tfidf = tfidf.transform(X_val)

print("=" * 70)
print("TF-IDF TRAINING / VALIDATION TRANSFORMATION")
print("=" * 70)

print("Training text samples:", len(X_train))
print("Validation text samples:", len(X_val))

print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("\nValidation TF-IDF shape:")
print(X_val_tfidf.shape)

print("\nNumber of TF-IDF features:")
print(len(tfidf.get_feature_names_out()))

print("\nFirst 20 TF-IDF features:")
print(tfidf.get_feature_names_out()[:20])