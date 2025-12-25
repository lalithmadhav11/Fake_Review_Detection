import pandas as pd

# load dataset
df = pd.read_csv(
    "data/reviews.csv",
    sep="\t",
    engine="python"
)


# show first 5 rows
print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nLabel distribution:")
print(df['flagged'].value_counts())
