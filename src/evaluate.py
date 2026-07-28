import os
import json
import joblib
# import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model_path: str = "models/baseline_demand_model.joblib", output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        # TODO [UNCOMMENT AFTER ENABLED NUMPY IMPORT]:
        "mape": float(np.mean(np.abs((y_test - preds) / np.clip(np.abs(y_test), 1e-8, None))) * 100),
        "under_forecast_rate": float(np.mean(preds < y_test)),
    }

    report_path = os.path.join(output_dir, "metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete:", metrics)
    print(f"Saved metrics to {report_path}")
    # [Customize before commit] Add business-facing KPIs and error breakdown by segment.


if __name__ == "__main__":
    evaluate()
