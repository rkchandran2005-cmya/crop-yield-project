

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

train_df = pd.read_csv("train.csv")

# --- Encode categorical columns into numbers the model can use ---
# We keep one LabelEncoder per column, and save them all in model.pkl
# so the exact same encoding can be applied later to test data / new input.
categorical_cols = ["crop", "irrigation_type", "fertiliser_used", "previous_yield"]
encoders = {}

df_encoded = train_df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

# Target column also needs encoding
target_encoder = LabelEncoder()
df_encoded["yield_category"] = target_encoder.fit_transform(df_encoded["yield_category"])
encoders["yield_category"] = target_encoder

feature_cols = ["crop", "area_acres", "irrigation_type", "fertiliser_used",
                 "rainfall_mm", "previous_yield"]

X_train = df_encoded[feature_cols]
y_train = df_encoded["yield_category"]

# --- Train the model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Save model + encoders + feature order together ---
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": encoders,
        "feature_cols": feature_cols
    }, f)

print("Model trained and saved to model.pkl")
print(f"Trained on {len(X_train)} rows")
print(f"Classes learned: {list(target_encoder.classes_)}")
