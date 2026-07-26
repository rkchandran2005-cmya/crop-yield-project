## Demo Video
[Watch the demo](https://1drv.ms/v/c/fdbe9eea4c82f97f/IQBZ-icdn5RGSonejA0_IIZyAZDfxzOaW6xj-03Qu_mMDMo?e=rcryk2)

# Crop Yield Category Estimator

Estimates whether a farm plot is heading for a **Poor**, **Average**, or **Good**
yield this season, based on farm records (area, irrigation, fertiliser use,
rainfall, and previous yield) — so extension officers can direct visits to the
plots most at risk, instead of giving uniform advice.

## How to Run (one command, after setup)

```bash
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the full pipeline from scratch (dataset → split → train → evaluate → edge-case test):

```bash
python task1_generate_dataset.py
python task2_split.py
python task3_train_model.py
python task4_evaluate.py
python task5_edge_case_test.py
streamlit run app.py
```

## Field Documentation

| Field              | Description                                                              |
|---------------------|---------------------------------------------------------------------------|
| `plot_id`           | Unique identifier for each farm plot                                     |
| `crop`              | Crop grown (Rice, Wheat, Cotton, Maize, Sugarcane)                        |
| `area_acres`        | Plot size in acres                                                        |
| `irrigation_type`   | Rainfed, Canal, Borewell, or Drip                                         |
| `fertiliser_used`   | Whether fertiliser was applied this season (Yes/No)                      |
| `rainfall_mm`       | Rainfall received during the growing season, in mm                       |
| `previous_yield`    | Yield category recorded last season (Poor/Average/Good)                  |
| `yield_category`    | **Target** — predicted yield category for this season                    |

The dataset (100 records) is synthetic, generated with a fixed random seed
(`random_state=42`) so it is fully reproducible. Labels are derived from a
composite score combining rainfall adequacy, irrigation reliability,
fertiliser use, and previous yield, plus random noise (since real farm
outcomes are never perfectly predictable).

**Class balance:** Average 60, Good 27, Poor 13 — deliberately imbalanced,
like real agricultural data, which is why per-category metrics (not just
accuracy) are reported below.

## Train/Test Split

80/20 split using `train_test_split` with `stratify=yield_category`, so the
Poor/Average/Good ratio is preserved in both sets. Each row is one
independently generated plot, so a plot can never appear in both the
training and test sets — verified with an explicit assertion in
`task2_split.py`.

## Model

A single `RandomForestClassifier` (scikit-learn), `n_estimators=100`,
`random_state=42`. No custom or from-scratch model — a standard library
classifier, as required for this level.

## Evaluation (honest, per-category)

*(Run `task4_evaluate.py` to reproduce these numbers exactly.)*

| Category | Precision | Recall | F1  |
|----------|-----------|--------|-----|
| Average  | 0.62      | 0.83   | 0.71 |
| Good     | 0.33      | 0.20   | 0.25 |
| Poor     | 1.00      | 0.33   | 0.50 |

**Confusion Matrix**

|              | Pred: Average | Pred: Good | Pred: Poor |
|--------------|:--:|:--:|:--:|
| **Actual: Average** | 10 | 2 | 0 |
| **Actual: Good**    | 4  | 1 | 0 |
| **Actual: Poor**    | 2  | 0 | 1 |

Overall accuracy is 0.60 — reported here only for context, since it is not
used as the headline metric.

## Error Analysis

*(Fill in your own read of the confusion matrix — a starting draft below.)*

The category predicted worst is **"Good"** (recall 0.20, the lowest of the
three categories). This happens mainly because both "Good" and "Poor"
are minority classes (27 and 13 examples respectively) relative to
"Average" (60), so the model has fewer examples to learn their patterns
from and defaults toward predicting "Average" when uncertain.

In practice, the costliest error is a genuinely **Poor** plot predicted as
**Average** or **Good**: the extension officer would not visit that farm,
so the farmer receives no help and a poor season goes unaddressed. A
**Good** plot wrongly predicted as **Poor** only costs the officer an
unnecessary visit — wasted time, but no harm to the farmer. This asymmetry
is why recall on the "Poor" class matters more than overall accuracy for
this use case.

## Testing on Unseen/Difficult Input

`task5_edge_case_test.py` feeds the model two unusual cases:

1. An extreme/contradictory record (very high rainfall, rainfed irrigation,
   no fertiliser, but a "Good" previous yield) — predicted **Average** at
   61% confidence.
2. A crop type ("Barley") never seen during training — predicted
   **Average** at **93% confidence**, despite the model having no real
   basis for that crop. This demonstrates exactly the risk described in
   the task: a model will confidently return an answer even for input it
   has never truly learned from.

## Confidence Threshold

A threshold of **0.5** is used: below this, the system reports "Unsure —
refer to officer" instead of a forced prediction. 0.5 is chosen because
with 3 classes, random guessing sits around 0.33, so 0.5 requires the
model to be meaningfully more confident than chance before its prediction
is trusted. As shown above, this threshold does not catch every
confidently-wrong case (e.g. the unseen-crop example) — a limitation
worth noting rather than hiding.

## Streamlit Interface

Run `streamlit run app.py` for an interactive form: enter plot details,
get the predicted category, confidence percentage, and full probability
breakdown, with a warning banner shown whenever confidence falls below
the threshold.

## Project Files

```
task1_generate_dataset.py   # creates farm_data.csv
task2_split.py               # creates train.csv / test.csv
task3_train_model.py         # trains model.pkl
task4_evaluate.py            # per-category metrics + confusion matrix
task5_edge_case_test.py      # unseen-input test + confidence threshold
app.py                       # Streamlit interface
requirements.txt
README.md
```
