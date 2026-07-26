


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("farm_data.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # reproducible split
    stratify=df["yield_category"]  # keep class balance consistent
)

# Sanity check: confirm zero plot_id overlap between the two sets
overlap = set(train_df["plot_id"]) & set(test_df["plot_id"])
assert len(overlap) == 0, f"Overlap found: {overlap}"

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train set: {len(train_df)} rows -> saved to train.csv")
print(f"Test set:  {len(test_df)} rows -> saved to test.csv")
print(f"Overlapping plot_ids between train and test: {len(overlap)}")

print("\nTrain set class balance:")
print(train_df["yield_category"].value_counts())

print("\nTest set class balance:")
print(test_df["yield_category"].value_counts())
