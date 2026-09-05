import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed training dataset
df = pd.read_csv("dataset/train_processed.csv")

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Transform cleaned text into TF-IDF features
X_tfidf = tfidf.fit_transform(df["clean_text"])

print("=" * 70)
print("TF-IDF TRANSFORMATION")
print("=" * 70)

print("Number of documents:", X_tfidf.shape[0])
print("Number of TF-IDF features:", X_tfidf.shape[1])

print("\nTF-IDF Matrix Shape:")
print(X_tfidf.shape)

# Get feature/word names
feature_names = tfidf.get_feature_names_out()

print("\nFirst 20 TF-IDF Features:")
print(feature_names[:20])

# Display TF-IDF values for the first document
tfidf_first = pd.DataFrame(
    X_tfidf[0].toarray(),
    columns=feature_names
)

print("\nTF-IDF values for the first document:")
print(
    tfidf_first.T[
        tfidf_first.T[0] > 0
    ].sort_values(by=0, ascending=False).head(10)
)

# Save the TF-IDF vectorizer vocabulary for reference
vocabulary = pd.DataFrame({
    "feature": feature_names
})

vocabulary.to_csv("tfidf_features.csv", index=False)

print("\nTF-IDF feature list saved successfully!")
