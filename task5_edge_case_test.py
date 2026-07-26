

import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
target_encoder = encoders["yield_category"]

# Confidence threshold: below this, we refuse to commit to a category.
# 0.5 is chosen because with 3 classes, "random guessing" would sit
# around 0.33 -- so 0.5 requires the model to be meaningfully more
# confident than chance before we trust its answer.
CONFIDENCE_THRESHOLD = 0.5


def predict_with_threshold(record: dict):
    """record must contain all feature_cols as raw (unencoded) values."""
    row = pd.DataFrame([record])

    for col in ["crop", "irrigation_type", "fertiliser_used", "previous_yield"]:
        le = encoders[col]
        # Handle a category the encoder has never seen (truly unseen input)
        if record[col] not in le.classes_:
            print(f"  Note: '{record[col]}' was never seen for '{col}' during training.")
            row[col] = -1  # flag as unknown; RandomForest can still take a raw split on this
        else:
            row[col] = le.transform([record[col]])

    X = row[feature_cols]
    probs = model.predict_proba(X)[0]
    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    predicted_label = target_encoder.classes_[pred_idx]

    print(f"  Raw probabilities: "
          f"{dict(zip(target_encoder.classes_, [round(p, 3) for p in probs]))}")
    print(f"  Predicted category: {predicted_label} (confidence: {confidence:.2f})")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  -> CONFIDENCE BELOW THRESHOLD ({CONFIDENCE_THRESHOLD}): "
              f"System says UNSURE -- refer this plot to an extension officer.")
        return "Unsure - refer to officer", confidence
    else:
        print(f"  -> Confidence acceptable, prediction stands.")
        return predicted_label, confidence


# --- Case 1: A deliberately extreme / unusual record ---
# Very high rainfall AND rainfed irrigation AND no fertiliser AND a
# previous Good yield -- a contradictory combination unlike any
# clean pattern in the training data.
print("Case 1: Extreme / contradictory record")
weird_case = {
    "crop": "Sugarcane",
    "area_acres": 0.6,          # unusually tiny plot
    "irrigation_type": "Rainfed",
    "fertiliser_used": "No",
    "rainfall_mm": 1800,        # far beyond the training range (~350-1150mm)
    "previous_yield": "Good",
}
predict_with_threshold(weird_case)

print()

# --- Case 2: A record using a category never seen in training at all ---
print("Case 2: Never-before-seen crop type")
unseen_crop_case = {
    "crop": "Barley",           # not in ['Rice','Wheat','Cotton','Maize','Sugarcane']
    "area_acres": 4.0,
    "irrigation_type": "Canal",
    "fertiliser_used": "Yes",
    "rainfall_mm": 700,
    "previous_yield": "Average",
}
predict_with_threshold(unseen_crop_case)
