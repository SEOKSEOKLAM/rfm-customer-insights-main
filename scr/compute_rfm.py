from pathlib import Path

import numpy as np
import pandas as pd


def _quantile_score(series: pd.Series, reverse: bool = False, bins: int = 5) -> pd.Series:
    labels = list(range(1, bins + 1))
    if reverse:
        labels = labels[::-1]

    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=labels).astype(int)


def _segment_customer(row: pd.Series) -> str:
    r_score = row["R_Score"]
    f_score = row["F_Score"]
    m_score = row["M_Score"]

    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    if f_score >= 4 and m_score >= 3:
        return "Loyal Customers"
    if r_score >= 4 and f_score >= 2:
        return "Potential Loyalists"
    if r_score == 5 and f_score == 1:
        return "New Customers"
    if r_score <= 2 and f_score >= 3:
        return "At Risk"
    if r_score <= 2 and f_score <= 2:
        return "Hibernating"
    if m_score >= 4:
        return "Big Spenders"
    return "Need Attention"


def compute_rfm(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, parse_dates=["Date"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if "LineAmount" not in df.columns:
        df["LineAmount"] = df["Quantity"] * df["Price"]
    if "IsReturnOrAdjustment" not in df.columns:
        df["IsReturnOrAdjustment"] = (
            (df["Quantity"] <= 0) | (df["Price"] <= 0) | (df["LineAmount"] <= 0)
        ).astype(int)

    analysis_df = df[df["IsReturnOrAdjustment"] == 0].copy()
    analysis_df["PurchaseDay"] = analysis_df["Date"].dt.normalize()
    analysis_df = analysis_df.sort_values(["CustomerID", "PurchaseDay"])

    order_proxy = (
        analysis_df.groupby(["CustomerID", "PurchaseDay"], as_index=False)
        .agg(
            DailyAmount=("LineAmount", "sum"),
            DailyUnits=("Quantity", "sum"),
            ItemLines=("EventID", "count"),
        )
    )

    anchor_date = analysis_df["PurchaseDay"].max()
    rfm = (
        order_proxy.groupby("CustomerID")
        .agg(
            Recency=("PurchaseDay", lambda x: (anchor_date - x.max()).days),
            Frequency=("PurchaseDay", "nunique"),
            Monetary=("DailyAmount", "sum"),
            TotalUnits=("DailyUnits", "sum"),
            AvgOrderValue=("DailyAmount", "mean"),
            ActiveSpanDays=("PurchaseDay", lambda x: (x.max() - x.min()).days),
        )
        .sort_index()
    )

    rfm["PurchaseIntervalDays"] = np.where(
        rfm["Frequency"] > 1,
        rfm["ActiveSpanDays"] / (rfm["Frequency"] - 1),
        np.nan,
    )

    return_counts = (
        df[df["IsReturnOrAdjustment"] == 1]
        .groupby("CustomerID")
        .size()
        .rename("ReturnRowCount")
    )
    positive_counts = (
        analysis_df.groupby("CustomerID").size().rename("PositiveRowCount")
    )
    rfm = rfm.join(return_counts, how="left").join(positive_counts, how="left")
    rfm["ReturnRowCount"] = rfm["ReturnRowCount"].fillna(0).astype(int)
    rfm["PositiveRowCount"] = rfm["PositiveRowCount"].fillna(0).astype(int)
    rfm["ReturnRowRate"] = (
        rfm["ReturnRowCount"] / (rfm["ReturnRowCount"] + rfm["PositiveRowCount"])
    ).fillna(0)

    rfm["R_Score"] = _quantile_score(rfm["Recency"], reverse=True)
    rfm["F_Score"] = _quantile_score(rfm["Frequency"])
    rfm["M_Score"] = _quantile_score(rfm["Monetary"])
    rfm["RFM_Total"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["Segment"] = rfm.apply(_segment_customer, axis=1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(output)
    print(f"RFM saved to {output}")
    print(f"Customers scored: {len(rfm):,}")
    print("RFM features use positive spend only, with quantity-adjusted monetary values.")


if __name__ == "__main__":
    compute_rfm("data/cleaned_data.csv", "data/customer_rfm.csv")
