import os

# Set Hugging Face cache inside the project folder
os.environ["HF_HOME"] = r"C:\emotion_nlp\huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = r"C:\emotion_nlp\huggingface_cache\datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"C:\emotion_nlp\huggingface_cache\hub"

from datasets import load_dataset
import pandas as pd

print("Downloading Emotion Dataset...")

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Convert datasets to pandas
train_df = pd.DataFrame(dataset["train"])
validation_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Create dataset folder
os.makedirs(r"C:\emotion_nlp\dataset", exist_ok=True)

# Save CSV files
train_df.to_csv(r"C:\emotion_nlp\dataset\train.csv", index=False)
validation_df.to_csv(r"C:\emotion_nlp\dataset\validation.csv", index=False)
test_df.to_csv(r"C:\emotion_nlp\dataset\test.csv", index=False)

print("\n===== DATASET INFORMATION =====")

print("Training data:", train_df.shape)
print("Validation data:", validation_df.shape)
print("Test data:", test_df.shape)

print("\n===== SAMPLE DATA =====")
print(train_df.head())

print("\n===== LABEL DISTRIBUTION =====")
print(train_df["label"].value_counts())

print("\nDataset downloaded successfully!")