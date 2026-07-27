import pandas as pd
import pickletoolswith open("model.pkl","rb") as f:
bundle = pickle.load(f)
model = bundel["model"]
encoders = bundel["encoders"]
feature_cols = bundel["feature_cols"]
target_encoder = encoders["yield_category"]
CONFIDENCE_THRESHOLD = 0.5
test_df = pd.read_csv("test.csv")
categorical_cols = ["crop", "irrigation_type", "fertiliser_used", "previous_yield"]
df_encoded = test_df.copy()
for col in categorical_cols:
    df_encoded[col]  = encoders[col].transform(df_encoded[col])
    x_test = df_encoded[feature_cols]
    probs_all = model.predict_proba(x_test)
    results = []
    for i, row in test_df.iterrows():
        probs = probs_all[test_df.index.get_loc(i)]
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]
        predicted_label = target_encoder.classes_[pred_idx]
        decision = "ABSTAIN - refer to officer" 
        if confidence < CONFIDENCE_THRESHOLD:
        else:
            predicted_label
            result.append({
                "plot_id":row["plot_id"],
                "actual":row["yield_category"],
                "predicted":predicted_label,
                "confidence":
                round(confidence,3),
                "decision":decision
            })
            result_df = pd.DataFrame(result).sort_values("confidence")
            print("="*70)
            print(results_df.to_string(index=False))
            abstrain = result_df[results_df["confidence"]<CONFIDENCE_THRESHOLD]
            print("\n" + "="*70)
            print(f"CASES SET ASIDE FOR HUMAN REVIEW(confidence<{CONFIDENCE_THRESHOLD})")
            print("="*70)
            if len(abstrained.to_string(index=False))
            print(f"\n{len(abstrained)}of {len(results_df)} test cases were set aside"
                  f"insted of being foece-guessed.")
    else:
        print("no test case fell below the threshold this run--showing the "
              "lowest-confidence case insted,as the closest borderline example:")
        print(result_df.iloc[[0]].to_string(index=False)) 

