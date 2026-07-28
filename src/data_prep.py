import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load customer demand data from CSV."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing and feature engineering for baseline training."""
    # [Customize before commit] Add richer feature engineering for your use case.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    # TODO [UNCOMMENT TO ADD TEMPORAL DEMAND FEATURES]:
    df = df.sort_values("date")
    df["lag_1_demand"] = df["demand_units"].shift(1)
    df["rolling_4w_demand_mean"] = df["demand_units"].rolling(window=4, min_periods=1).mean()
    df["rolling_4w_demand_std"] = df["demand_units"].rolling(window=4, min_periods=1).std().fillna(0)

    categorical_cols = ["region", "product_category", "customer_segment"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # TODO [UNCOMMENT IF lag_1_demand IS ENABLED]:
    # df = df.dropna(subset=["lag_1_demand"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # [Customize before commit] Point to your preferred data path.
    data = load_data("data/raw/customer_demand_sample.csv")
    processed = preprocess(data)
    processed.to_csv("data/processed/customer_demand_processed.csv", index=False)
    print("Preprocessing complete. Saved data/processed/customer_demand_processed.csv")
