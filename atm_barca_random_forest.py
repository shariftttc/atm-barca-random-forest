"""
ATM vs Barcelona — Random Forest model trained ONLY on atm_barca_training_only.csv.

Install:
    python -m pip install pandas scikit-learn

Run:
    python atm_barca_random_forest.py

IMPORTANT:
The CSV is synthetic educational data, not real La Liga match history.
The target-match feature values below are illustrative placeholders, not verified
statistics for the actual Atletico Madrid vs Barcelona fixture.
Do not use the resulting probabilities as real-world forecasts or betting advice.
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "atm_barca_training_only.csv"
FEATURES = [
    "home_points_last5", "away_points_last5",
    "home_gf_last5", "home_ga_last5",
    "away_gf_last5", "away_ga_last5",
    "home_h2h_points", "away_h2h_points",
    "home_advantage",
]
LABEL = "result"
LABEL_NAMES = {"H": "Atletico Madrid win", "D": "Draw", "A": "Barcelona win"}

# Illustrative values only — replace with verified pre-match statistics.
TARGET_MATCH = {
    "home_points_last5": 2.05,
    "away_points_last5": 2.35,
    "home_gf_last5": 1.80,
    "home_ga_last5": 0.80,
    "away_gf_last5": 2.20,
    "away_ga_last5": 0.90,
    "home_h2h_points": 1.55,
    "away_h2h_points": 1.20,
    "home_advantage": 1,
}

def main():
    path = Path(DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Put this Python file and the CSV in the same folder."
        )

    df = pd.read_csv(path)
    required = FEATURES + [LABEL, "match_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=FEATURES + [LABEL, "match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # Chronological split: first 80% train, latest 20% test.
    split = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    X_train, y_train = train_df[FEATURES], train_df[LABEL]
    X_test, y_test = test_df[FEATURES], test_df[LABEL]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("random_forest", RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("=== DATASET ===")
    print("Rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("\nClass distribution:")
    print(df[LABEL].value_counts().rename(index=LABEL_NAMES))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== TEST PERFORMANCE ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=["H", "D", "A"],
        target_names=["Atletico win", "Draw", "Barcelona win"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted; H,D,A):")
    print(confusion_matrix(y_test, y_pred, labels=["H", "D", "A"]))

    # Refit on all available training-only records.
    model.fit(df[FEATURES], df[LABEL])
    target_X = pd.DataFrame([TARGET_MATCH], columns=FEATURES)
    probabilities = model.predict_proba(target_X)[0]
    classes = model.named_steps["random_forest"].classes_

    result = pd.DataFrame([
        {"outcome": LABEL_NAMES.get(label, label), "probability": float(prob)}
        for label, prob in zip(classes, probabilities)
    ]).sort_values("probability", ascending=False)
    result["probability_percent"] = (100 * result["probability"]).round(2)
    result.to_csv("atm_barca_model_prediction.csv", index=False)

    print("\n=== MODEL OUTPUT FOR ILLUSTRATIVE ATM vs BARCA INPUT ===")
    print(result[["outcome", "probability_percent"]].to_string(index=False))
    print("\nSaved: atm_barca_model_prediction.csv")
    print("\nCAUTION: synthetic training data + placeholder target features.")
    print("This is a coding exercise, not a valid real-match forecast.")

if __name__ == "__main__":
    main()
