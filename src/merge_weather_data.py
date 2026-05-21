from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FILES = {
    "min_temp": RAW_DIR / "min_temp" / "IDCJAC0011_009225_1800_Data.csv",
    "max_temp": RAW_DIR / "max_temp" / "IDCJAC0010_009225_1800_Data.csv",
    "rainfall": RAW_DIR / "rainfall" / "IDCJAC0009_009225_1800_Data.csv",
    "sunshine": RAW_DIR / "sunshine" / "IDCJAC0016_009225_1800_Data.csv",
}

VALUE_COLUMNS = {
    "min_temp": "Minimum temperature (Degree C)",
    "max_temp": "Maximum temperature (Degree C)",
    "rainfall": "Rainfall amount (millimetres)",
    "sunshine": "Daily global solar exposure (MJ/m*m)",
}


def load_feature(feature_name: str) -> pd.DataFrame:
    path = FILES[feature_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    df = pd.read_csv(path)
    value_column = VALUE_COLUMNS[feature_name]
    df = df[["Year", "Month", "Day", value_column]].copy()
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df[feature_name] = pd.to_numeric(df[value_column], errors="coerce")
    return df[["date", feature_name]]


def main() -> None:
    merged = None

    for feature_name in ["min_temp", "max_temp", "rainfall", "sunshine"]:
        feature_df = load_feature(feature_name)
        if merged is None:
            merged = feature_df
        else:
            merged = merged.merge(feature_df, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Turn next day's rainfall into a binary label for tomorrow's rain.
    merged["rain_tomorrow"] = (merged["rainfall"].shift(-1) > 0).astype("Int64")
    merged["rain_today"] = (merged["rainfall"] > 0).astype("Int64")

    cleaned = merged.dropna(
        subset=["min_temp", "max_temp", "rainfall", "sunshine", "rain_tomorrow"]
    ).copy()

    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")
    cleaned["rain_tomorrow"] = cleaned["rain_tomorrow"].astype(int)
    cleaned["rain_today"] = cleaned["rain_today"].astype(int)

    output_columns = [
        "date",
        "min_temp",
        "max_temp",
        "rainfall",
        "sunshine",
        "rain_today",
        "rain_tomorrow",
    ]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "weather_project_dataset.csv"
    cleaned.to_csv(output_path, index=False, columns=output_columns)

    print(f"Saved merged dataset to: {output_path}")
    print(f"Rows: {len(cleaned)}")
    print(f"Date range: {cleaned['date'].iloc[0]} to {cleaned['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
