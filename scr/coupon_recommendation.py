import pandas as pd


SEGMENT_STRATEGY = {
    "Champions": {
        "Coupon": "VIP bundle coupon",
        "CampaignObjective": "Protect high-value customers and lift repeat spend",
        "SuggestedAction": "Offer premium bundles or member-only perks instead of deep discounts",
    },
    "Loyal Customers": {
        "Coupon": "50-off loyalty coupon",
        "CampaignObjective": "Encourage steady repeat purchases",
        "SuggestedAction": "Use replenishment reminders and cross-sell recommendations",
    },
    "Potential Loyalists": {
        "Coupon": "30-off next-order coupon",
        "CampaignObjective": "Convert recent buyers into repeat customers",
        "SuggestedAction": "Follow up quickly with limited-time repeat purchase offers",
    },
    "New Customers": {
        "Coupon": "Welcome coupon",
        "CampaignObjective": "Drive the second purchase early",
        "SuggestedAction": "Send onboarding messages with best-selling products",
    },
    "Big Spenders": {
        "Coupon": "High-cart threshold coupon",
        "CampaignObjective": "Maintain high basket size without over-discounting",
        "SuggestedAction": "Push add-on products and premium collections",
    },
    "At Risk": {
        "Coupon": "Reactivation coupon",
        "CampaignObjective": "Win back valuable but cooling customers",
        "SuggestedAction": "Use urgency messaging and highlight new arrivals",
    },
    "Hibernating": {
        "Coupon": "Low-barrier comeback coupon",
        "CampaignObjective": "Recover inactive customers at controlled cost",
        "SuggestedAction": "Test low-cost reactivation campaigns before larger offers",
    },
    "Need Attention": {
        "Coupon": "General-purpose coupon",
        "CampaignObjective": "Increase engagement and identify purchase triggers",
        "SuggestedAction": "A/B test timing and product recommendations",
    },
}


def apply(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, index_col=0)
    strategy = pd.DataFrame.from_dict(SEGMENT_STRATEGY, orient="index")
    df = df.join(strategy, on="Segment")
    df.to_csv(output_path)
    print(f"Coupon recommendations saved to {output_path}")


if __name__ == "__main__":
    apply("data/customer_rfm_clusters.csv", "data/customer_rfm_recommendations.csv")
