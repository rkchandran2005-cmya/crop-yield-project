

import pandas as pd
import numpy as np

# Fixed seed so this dataset is reproducible every time it's regenerated
np.random.seed(42)

N = 100

crops = ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane"]
irrigation_types = ["Rainfed", "Canal", "Borewell", "Drip"]
fert_options = ["Yes", "No"]
prev_yield_options = ["Poor", "Average", "Good"]

# Reliability weight per irrigation type (used only to build realistic labels)
irrigation_reliability = {"Rainfed": 0.3, "Canal": 0.6, "Borewell": 0.7, "Drip": 0.9}
prev_yield_score = {"Poor": 0.2, "Average": 0.55, "Good": 0.85}

rows = []
for i in range(1, N + 1):
    plot_id = f"P{i:03d}"
    crop = np.random.choice(crops)
    area_acres = round(np.random.uniform(0.5, 10.0), 2)
    irrigation_type = np.random.choice(irrigation_types, p=[0.35, 0.25, 0.25, 0.15])
    fertiliser_used = np.random.choice(fert_options, p=[0.65, 0.35])
    rainfall_mm = round(np.random.normal(750, 200), 1)
    rainfall_mm = max(50, rainfall_mm)  # no negative rainfall
    previous_yield = np.random.choice(prev_yield_options, p=[0.25, 0.45, 0.30])

    # --- Build a composite score to decide this season's yield category ---
    rainfall_score = np.clip(rainfall_mm / 1000, 0, 1.2)  # too little or too much both hurt slightly
    irrigation_score = irrigation_reliability[irrigation_type]
    fert_score = 0.15 if fertiliser_used == "Yes" else 0.0
    prev_score = prev_yield_score[previous_yield]

    composite = (
        0.35 * rainfall_score
        + 0.30 * irrigation_score
        + 0.15 * fert_score / 0.15  # normalize fert contribution
        + 0.20 * prev_score
    )
    composite += np.random.normal(0, 0.08)  # real-world noise
    if composite < 0.40:
            yield_category = "Poor"
    if composite < 0.45:
        yield_category = "Poor"
    elif composite < 0.70:
        yield_category = "Average"
    else:
        yield_category = "Good"

    rows.append([
        plot_id, crop, area_acres, irrigation_type, fertiliser_used,
        round(rainfall_mm, 1), previous_yield, yield_category
    ])

df = pd.DataFrame(rows, columns=[
    "plot_id", "crop", "area_acres", "irrigation_type", "fertiliser_used",
    "rainfall_mm", "previous_yield", "yield_category"
])

df.to_csv("farm_data.csv", index=False)

print("Dataset saved to farm_data.csv")
print("\nFirst 5 rows:")
print(df.head())

print("\nClass balance (yield_category counts):")
print(df["yield_category"].value_counts())
print("\nClass balance (proportion):")
print(df["yield_category"].value_counts(normalize=True).round(3))
