from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _prepare_transaction_frame(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if "LineAmount" not in df.columns:
        df["LineAmount"] = df["Quantity"] * df["Price"]
    if "IsReturnOrAdjustment" not in df.columns:
        df["IsReturnOrAdjustment"] = (
            (df["Quantity"] <= 0) | (df["Price"] <= 0) | (df["LineAmount"] <= 0)
        ).astype(int)

    df["Product"] = df["Product"].fillna("Unknown Product")
    return df.dropna(subset=["CustomerID", "Date"]).copy()


def evaluate(raw_path: str, rec_path: str) -> None:
    transaction_df = _prepare_transaction_frame(raw_path)
    customer_df = pd.read_csv(rec_path, index_col=0)

    enriched = transaction_df.merge(
        customer_df[
            [
                "Segment",
                "Coupon",
                "Recency",
                "Frequency",
                "Monetary",
                "AvgOrderValue",
                "ReturnRowRate",
            ]
        ],
        left_on="CustomerID",
        right_index=True,
        how="inner",
    )

    positive_df = enriched[enriched["LineAmount"] > 0].copy()
    total_revenue = positive_df["LineAmount"].sum()

    segment_summary = (
        customer_df.groupby("Segment")
        .agg(
            customers=("Recency", "size"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            avg_order_value=("AvgOrderValue", "mean"),
            repeat_customer_rate=("Frequency", lambda x: (x > 1).mean()),
            active_90d_share=("Recency", lambda x: (x <= 90).mean()),
            avg_return_row_rate=("ReturnRowRate", "mean"),
        )
    )

    revenue_by_segment = positive_df.groupby("Segment")["LineAmount"].sum().rename("revenue")
    segment_summary = segment_summary.join(revenue_by_segment, how="left").fillna({"revenue": 0})
    segment_summary["customer_share"] = (
        segment_summary["customers"] / segment_summary["customers"].sum()
    )
    segment_summary["revenue_share"] = (
        segment_summary["revenue"] / total_revenue if total_revenue else 0
    )
    segment_summary["Coupon"] = (
        customer_df.groupby("Segment")["Coupon"].first()
    )
    segment_summary = segment_summary.sort_values("revenue", ascending=False)

    top_products = (
        positive_df.groupby(["Segment", "Product"])["LineAmount"]
        .sum()
        .rename("Revenue")
        .reset_index()
        .sort_values(["Segment", "Revenue"], ascending=[True, False])
        .groupby("Segment")
        .head(3)
    )

    data_dir = Path("data")
    image_dir = Path("images")
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "evaluation_summary.csv"
    top_products_path = data_dir / "segment_top_products.csv"
    image_path = image_dir / "performance_metrics.png"

    segment_summary.round(4).to_csv(summary_path)
    top_products.round(2).to_csv(top_products_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    segment_summary["customers"].plot.bar(ax=axes[0, 0], color="#2f6b7d")
    axes[0, 0].set_title("Customer Count by Segment")
    axes[0, 0].set_ylabel("Customers")

    (segment_summary["revenue_share"] * 100).plot.bar(ax=axes[0, 1], color="#d97b2d")
    axes[0, 1].set_title("Revenue Share by Segment")
    axes[0, 1].set_ylabel("Revenue Share (%)")

    segment_summary["avg_order_value"].plot.bar(ax=axes[1, 0], color="#5c8a3d")
    axes[1, 0].set_title("Average Order Value")
    axes[1, 0].set_ylabel("Amount")

    (segment_summary["repeat_customer_rate"] * 100).plot.bar(
        ax=axes[1, 1], color="#8e5ea2"
    )
    axes[1, 1].set_title("Repeat Customer Rate")
    axes[1, 1].set_ylabel("Repeat Rate (%)")

    for axis in axes.flatten():
        axis.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)

    print("Evaluation complete")
    print(f"Summary saved to {summary_path}")
    print(f"Top products saved to {top_products_path}")
    print(f"Chart saved to {image_path}")


if __name__ == "__main__":
    evaluate("data/cleaned_data.csv", "data/customer_rfm_recommendations.csv")
