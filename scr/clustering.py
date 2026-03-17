from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None
    silhouette_score = None
    SKLEARN_AVAILABLE = False


FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalUnits"]
PLOT_CLIP_QUANTILES = {
    "Recency": 1.0,
    "Frequency": 0.99,
    "Monetary": 0.99,
    "AvgOrderValue": 0.99,
}


def _scale_features(frame: pd.DataFrame) -> pd.DataFrame:
    scaled = np.log1p(frame.clip(lower=0))
    denominator = scaled.max() - scaled.min()
    denominator = denominator.replace(0, 1)
    return (scaled - scaled.min()) / denominator


def _fit_kmeans(features: pd.DataFrame) -> tuple[pd.Series, dict[int, float]]:
    best_k = 2
    best_score = -1.0
    scores: dict[int, float] = {}

    upper_bound = min(6, len(features) - 1)
    for k in range(2, upper_bound + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(features)
        score = silhouette_score(features, labels)
        scores[k] = score
        if score > best_score:
            best_k = k
            best_score = score

    final_labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(features)
    return pd.Series(final_labels, index=features.index), scores


def _fallback_clusters(rfm: pd.DataFrame) -> pd.Series:
    codes, _ = pd.factorize(rfm["Segment"], sort=True)
    return pd.Series(codes, index=rfm.index)


def _save_cluster_plot(rfm: pd.DataFrame, image_dir: Path) -> None:
    counts = rfm["Segment"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(11, 5))
    counts.plot.bar(color="#2f6b7d")
    plt.title("Customer Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(image_dir / "cluster_distribution.png")
    plt.close()


def _format_compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _plot_metric_distribution(axis: plt.Axes, series: pd.Series, metric: str) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    clip_quantile = PLOT_CLIP_QUANTILES.get(metric, 1.0)
    upper_bound = float(values.quantile(clip_quantile))
    plotted = values.clip(upper=upper_bound) if clip_quantile < 1.0 else values

    if metric == "Frequency":
        max_bin = max(int(np.ceil(upper_bound)), 1)
        bins = np.arange(0.5, max_bin + 1.5, 1)
    else:
        bins = 30

    axis.hist(plotted, bins=bins, color="#d97b2d", edgecolor="white")
    axis.set_title(metric if clip_quantile == 1.0 else f"{metric} (capped at P99)")
    axis.set_ylabel("Customers")

    median = float(values.median())
    p90 = float(values.quantile(0.90))
    clipped_count = int((values > upper_bound).sum()) if clip_quantile < 1.0 else 0

    for line_value, color, label in (
        (median, "#2f6b7d", "Median"),
        (p90, "#5c8a3d", "P90"),
    ):
        if line_value <= upper_bound:
            axis.axvline(line_value, color=color, linestyle="--", linewidth=1.5, label=label)

    note_lines = [
        f"Median: {_format_compact_number(median)}",
        f"P90: {_format_compact_number(p90)}",
    ]
    if clipped_count:
        note_lines.append(
            f"Clipped > {_format_compact_number(upper_bound)}: {clipped_count}"
        )

    axis.text(
        0.98,
        0.95,
        "\n".join(note_lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="upper left", fontsize=8)


def _save_rfm_plot(rfm: pd.DataFrame, image_dir: Path) -> None:
    metrics = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, metric in zip(axes.flatten(), metrics):
        _plot_metric_distribution(axis, rfm[metric], metric)

    fig.tight_layout()
    fig.savefig(image_dir / "rfm_distribution.png")
    plt.close(fig)


def cluster(input_path: str, output_path: str) -> None:
    rfm = pd.read_csv(input_path, index_col=0)
    features = _scale_features(rfm[FEATURE_COLUMNS])

    if SKLEARN_AVAILABLE and len(rfm) >= 3:
        cluster_labels, scores = _fit_kmeans(features)
        rfm["ClusterMethod"] = "kmeans"
        rfm["Cluster"] = cluster_labels
        for k, score in scores.items():
            print(f"K={k}, Silhouette={score:.4f}")
    else:
        rfm["ClusterMethod"] = "segment_fallback"
        rfm["Cluster"] = _fallback_clusters(rfm)
        print("scikit-learn not available. Using segment-based fallback clusters.")

    cluster_summary = (
        rfm.groupby(["Cluster", "Segment"])
        .agg(
            Customers=("Recency", "size"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgOrderValue=("AvgOrderValue", "mean"),
        )
        .sort_values(["AvgMonetary", "AvgFrequency"], ascending=[False, False])
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary.to_csv(output.parent / "cluster_summary.csv")
    rfm.to_csv(output)

    image_dir = Path("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    _save_cluster_plot(rfm, image_dir)
    _save_rfm_plot(rfm, image_dir)

    print(f"Clusters saved to {output}")
    print(f"Cluster summary saved to {output.parent / 'cluster_summary.csv'}")


if __name__ == "__main__":
    cluster("data/customer_rfm.csv", "data/customer_rfm_clusters.csv")
