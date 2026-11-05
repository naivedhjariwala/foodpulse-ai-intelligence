"""
Customer Behavior Segmentation for FoodPulse AI Delivery Intelligence System.
Applies K-Means clustering on RFM and behavioral metrics, analyzes Silhouette & Elbow metrics,
and projects customer personas via PCA.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


CLUSTER_FEATURE_COLS = [
    "total_orders",
    "avg_order_value",
    "avg_items_per_order",
    "weekend_order_ratio",
    "dinner_peak_ratio",
    "lunch_rush_ratio",
    "delayed_order_ratio",
    "avg_rating_given"
]


class CustomerSegmentation:
    """
    K-Means clustering pipeline for customer behavioral profiling.
    """
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.elbow_scores = {}
        self.silhouette_scores = {}
        self.cluster_summaries = {}
        self.persona_names = {}

    def extract_customer_profiles(self, df_orders: pd.DataFrame, df_reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates raw orders and reviews into behavioral RFM features per customer.
        """
        # Review stats per customer
        rev_stats = df_reviews.groupby("customer_id").agg(
            review_count=("review_id", "count"),
            avg_rating_given=("rating", "mean")
        ).reset_index()

        # Orders stats per customer
        orders_agg = []
        for cust_id, group in df_orders.groupby("customer_id"):
            tot_orders = len(group)
            avg_val = group["order_value"].mean()
            avg_items = group["item_count"].mean()

            # Ratios
            weekend_cnt = group["day_of_week"].isin(["Saturday", "Sunday"]).sum()
            dinner_cnt = ((group["hour_of_day"] >= 18) & (group["hour_of_day"] <= 21)).sum()
            lunch_cnt = ((group["hour_of_day"] >= 11) & (group["hour_of_day"] <= 14)).sum()
            delayed_cnt = (group["is_delayed"] == 1).sum()

            orders_agg.append({
                "customer_id": cust_id,
                "total_orders": tot_orders,
                "total_spend": round(group["order_value"].sum(), 2),
                "avg_order_value": round(avg_val, 2),
                "avg_items_per_order": round(avg_items, 2),
                "weekend_order_ratio": round(weekend_cnt / tot_orders, 3),
                "dinner_peak_ratio": round(dinner_cnt / tot_orders, 3),
                "lunch_rush_ratio": round(lunch_cnt / tot_orders, 3),
                "delayed_order_ratio": round(delayed_cnt / tot_orders, 3)
            })

        df_cust = pd.DataFrame(orders_agg)
        df_cust = df_cust.merge(rev_stats, on="customer_id", how="left")
        df_cust["review_count"] = df_cust["review_count"].fillna(0).astype(int)
        df_cust["avg_rating_given"] = df_cust["avg_rating_given"].fillna(df_cust["avg_rating_given"].mean())

        return df_cust

    def evaluate_clusters(self, df_features: pd.DataFrame, k_range: range = range(2, 8)) -> Dict[str, Any]:
        """
        Computes inertia (elbow) and silhouette scores across candidate K values.
        """
        X_scaled = self.scaler.fit_transform(df_features[CLUSTER_FEATURE_COLS])
        elbow_data = {}
        silhouette_data = {}

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            elbow_data[k] = round(float(km.inertia_), 2)
            sil = silhouette_score(X_scaled, labels)
            silhouette_data[k] = round(float(sil), 4)

        self.elbow_scores = elbow_data
        self.silhouette_scores = silhouette_data

        return {
            "elbow_inertia": elbow_data,
            "silhouette_scores": silhouette_data
        }

    def fit_predict(self, df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Fits optimal K-Means model, assigns cluster IDs, and creates PCA coordinates.
        """
        X_scaled = self.scaler.fit_transform(df_cust[CLUSTER_FEATURE_COLS])
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        pca_coords = self.pca_2d.fit_transform(X_scaled)

        df_result = df_cust.copy()
        df_result["cluster_id"] = clusters
        df_result["pca_x"] = np.round(pca_coords[:, 0], 3)
        df_result["pca_y"] = np.round(pca_coords[:, 1], 3)

        # Label personas based on centroid features
        self._generate_cluster_summaries(df_result)
        df_result["persona"] = df_result["cluster_id"].map(self.persona_names)

        return df_result

    def _generate_cluster_summaries(self, df_result: pd.DataFrame) -> None:
        """Computes summary statistics and descriptive names for each cluster."""
        summaries = {}
        for c_id in range(self.n_clusters):
            c_df = df_result[df_result["cluster_id"] == c_id]
            stats = {
                "size": len(c_df),
                "pct_of_base": round(100.0 * len(c_df) / len(df_result), 1),
                "avg_orders": round(c_df["total_orders"].mean(), 1),
                "avg_order_value": round(c_df["avg_order_value"].mean(), 2),
                "avg_items": round(c_df["avg_items_per_order"].mean(), 2),
                "weekend_pct": round(c_df["weekend_order_ratio"].mean() * 100, 1),
                "dinner_pct": round(c_df["dinner_peak_ratio"].mean() * 100, 1),
                "lunch_pct": round(c_df["lunch_rush_ratio"].mean() * 100, 1),
                "avg_rating": round(c_df["avg_rating_given"].mean(), 2)
            }
            summaries[c_id] = stats

            # Derive Archetype Persona name
            if stats["avg_order_value"] > 55.0 and stats["dinner_pct"] > 35.0:
                name = "Gourmet Dinner Connoisseurs"
                strategy = "VIP courier routing, priority packaging, high-margin sommelier & dessert upsells."
            elif stats["lunch_pct"] > 35.0:
                name = "Fast-Paced Office Lunchers"
                strategy = "Guaranteed 25-minute SLA, express combo lunch deals, corporate multi-order discounts."
            elif stats["weekend_pct"] > 35.0 or stats["avg_items"] >= 4.0:
                name = "Weekend Family & Group Feasters"
                strategy = "Catering bundle promotions, pre-scheduled delivery slots, family beverage pairings."
            else:
                name = "Budget Everyday Frequent Orderers"
                strategy = "Loyalty rewards pass, free delivery tiers, off-peak discount coupons."

            self.persona_names[c_id] = name
            summaries[c_id]["persona_name"] = name
            summaries[c_id]["recommended_strategy"] = strategy

        self.cluster_summaries = summaries

    def save(self, filepath: str) -> None:
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "CustomerSegmentation":
        """Loads serialized model."""
        return joblib.load(filepath)

