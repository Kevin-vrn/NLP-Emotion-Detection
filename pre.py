import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("dataset/train.csv")

# English stopwords
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


# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Display examples
print("=" * 70)
print("RAW TEXT VS PREPROCESSED TEXT")
print("=" * 70)

for i in range(10):
    print("\nOriginal:")
    print(df.loc[i, "text"])

    print("\nCleaned:")
    print(df.loc[i, "clean_text"])

    print("-" * 70)

# Save processed dataset
df.to_csv("dataset/train_processed.csv", index=False)

print("\nProcessed dataset saved successfully!")