

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# --- Load model + encoders saved in Task 3 ---
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]

test_df = pd.read_csv("test.csv")

# --- Apply the SAME encoders used during training ---
categorical_cols = ["crop", "irrigation_type", "fertiliser_used", "previous_yield"]
df_encoded = test_df.copy()
for col in categorical_cols:
    le = encoders[col]
    df_encoded[col] = le.transform(df_encoded[col])

target_encoder = encoders["yield_category"]
y_true = target_encoder.transform(df_encoded["yield_category"])

X_test = df_encoded[feature_cols]

# --- Predict ---
y_pred = model.predict(X_test)

class_names = list(target_encoder.classes_)

print("=" * 55)
print("PER-CATEGORY PRECISION / RECALL / F1")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in class_names],
                        columns=[f"Pred_{c}" for c in class_names])
print(cm_df)


