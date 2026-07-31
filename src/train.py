import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from data_prep import load_data, preprocess


def train_model(data_path: str, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)
    df = preprocess(df)

    target = "demand_units"
    feature_cols = [c for c in df.columns if c not in ["date", target]]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TODO [UNCOMMENT TO TRY A STRONGER BASELINE]:
    model = RandomForestRegressor(
         n_estimators=400, max_depth=12, min_samples_split=5, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "baseline_demand_model.joblib")
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "X_test": X_test,
            "y_test": y_test,
        },
        model_path,
    )

    print(f"Model saved to {model_path}")
    # [Customize before commit] Consider adding experiment tracking (e.g., SageMaker Experiments/MLflow).


if __name__ == "__main__":
    train_model("data/raw/customer_demand_sample.csv")
