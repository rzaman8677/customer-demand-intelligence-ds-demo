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

    categorical_cols = ["region", "product_category", "customer_segment"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


if __name__ == "__main__":
    # [Customize before commit] Point to your preferred data path.
    data = load_data("data/raw/customer_demand_sample.csv")
    processed = preprocess(data)
    processed.to_csv("data/processed/customer_demand_processed.csv", index=False)
    print("Preprocessing complete. Saved data/processed/customer_demand_processed.csv")
