from pathlib import Path

import pandas as pd


ESSENTIAL_COLUMNS = ["CustomerID", "Date", "Quantity", "Price"]


def preprocess(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if "Product" in df.columns:
        df["Product"] = df["Product"].fillna("Unknown Product").astype(str).str.strip()

    for column in ["ProductID", "Country"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.strip(), errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    before_rows = len(df)
    df = df.dropna(subset=ESSENTIAL_COLUMNS).copy()
    df["LineAmount"] = df["Quantity"] * df["Price"]
    df["IsReturnOrAdjustment"] = (
        (df["Quantity"] <= 0) | (df["Price"] <= 0) | (df["LineAmount"] <= 0)
    ).astype(int)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    removed_rows = before_rows - len(df)
    flagged_rows = int(df["IsReturnOrAdjustment"].sum())
    print(f"Saved cleaned data to {output}")
    print(f"Rows kept: {len(df):,}")
    print(f"Rows removed for missing essential fields: {removed_rows:,}")
    print(f"Rows flagged as returns/adjustments: {flagged_rows:,}")


if __name__ == "__main__":
    preprocess("data/live_streaming_sales_data.csv", "data/cleaned_data.csv")
