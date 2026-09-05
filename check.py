import pandas as pd

files = [
    "dataset/train.csv",
    "dataset/train_processed.csv",
    "dataset/validation.csv",
    "dataset/test.csv"
]

for file in files:
    df = pd.read_csv(file)

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("=" * 60)

    print(df.head(3))