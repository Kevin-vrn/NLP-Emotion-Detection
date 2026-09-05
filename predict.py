import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ==================================================
# 1. DOWNLOAD NLTK RESOURCES
# ==================================================

nltk.download("punkt")
nltk.download("stopwords")


# ==================================================
# 2. LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ==================================================

model = joblib.load(
    "models/svm_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==================================================
# 3. PREPROCESSING
# ==================================================

stop_words = set(
    stopwords.words("english")
)


def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)


# ==================================================
# 4. EMOTION LABELS
# ==================================================

label_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# ==================================================
# 5. GET USER INPUT
# ==================================================

print("=" * 60)
print("      EMOTION DETECTION FROM TEXT")
print("=" * 60)

text = input("\nEnter a sentence: ")


# ==================================================
# 6. PREPROCESS INPUT
# ==================================================

clean_text = preprocess_text(text)


# ==================================================
# 7. CONVERT TEXT TO TF-IDF
# ==================================================

text_tfidf = tfidf.transform(
    [clean_text]
)


# ==================================================
# 8. PREDICT EMOTION
# ==================================================

prediction = model.predict(
    text_tfidf
)[0]


emotion = label_names[prediction]


# ==================================================
# 9. DISPLAY RESULT
# ==================================================

print("\nOriginal Text:")
print(text)

print("\nProcessed Text:")
print(clean_text)

print("\nPredicted Emotion:")
print(emotion)

print("\n" + "=" * 60)