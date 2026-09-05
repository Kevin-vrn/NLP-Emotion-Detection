import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed dataset
df = pd.read_csv("dataset/train_processed.csv")

# Separate input and target
X_text = df["clean_text"]
y = df["label"]

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Transform text into numerical features
X_tfidf = tfidf.fit_transform(X_text)

print("=" * 70)
print("FEATURE AND TARGET SEPARATION")
print("=" * 70)

print("X (TF-IDF features):", X_tfidf.shape)
print("y (Target labels):", y.shape)

print("\nFirst 10 target labels:")
print(y.head(10).to_list())

print("\nTarget label distribution:")
print(y.value_counts().sort_index())