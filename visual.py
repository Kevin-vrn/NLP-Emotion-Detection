import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Emotion mapping
emotion_map = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Convert labels to emotion names
df["emotion"] = df["label"].map(emotion_map)

# Count emotions
emotion_counts = df["emotion"].value_counts()

print("\nEmotion Distribution:")
print(emotion_counts)

# Create bar chart
plt.figure(figsize=(10, 6))

emotion_counts.plot(kind="bar")

plt.title("Emotion Distribution in Training Dataset")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("emotion_distribution.png", dpi=300)

plt.show()