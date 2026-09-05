import pandas as pd
import matplotlib.pyplot as plt


# Load model comparison results
results = pd.read_csv(
    "results/model_comparison.csv",
    index_col=0
)


# Create accuracy graph
plt.figure(figsize=(8, 5))

plt.bar(
    results.index,
    results["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Comparison of Emotion Classification Models")

plt.ylim(0, 100)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "results/model_accuracy_comparison.png",
    dpi=300
)

plt.show()


print("Graph saved successfully!")
print(
    "results/model_accuracy_comparison.png"
)