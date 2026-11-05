"""
End-to-End Predictive Pipeline for FoodPulse AI Delivery Intelligence System.
Orchestrates:
1. Data loading from SQLite database
2. Feature engineering & transformation
3. Delivery duration regression (ETA in minutes)
4. SLA delay risk classification (Severity & risk tiers)
5. Customer segmentation (K-Means & PCA profiling)
6. NLP sentiment analysis & complaint extraction
7. Model serialization & benchmark evaluation reporting
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.db import query_df, DEFAULT_DB_PATH
from src.features import FoodPulseFeatureEngineer, save_feature_engineer
from src.models.eta_regressor import ETARegressor
from src.models.delay_classifier import DelayClassifier
from src.models.customer_segmentation import CustomerSegmentation
from src.models.nlp_sentiment import NLPSentimentAnalyzer

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def run_pipeline(db_path: str = DEFAULT_DB_PATH, model_dir: str = MODEL_DIR) -> Dict[str, Any]:
    """Runs the full training, feature extraction, and evaluation pipeline."""
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 70)
    print("FOODPULSE AI: EXECUTING END-TO-END PREDICTIVE TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data from SQLite
    print("[1/6] Loading operational datasets from SQLite...")
    df_hubs = query_df("SELECT * FROM urban_hubs;", db_path=db_path)
    df_restaurants = query_df("SELECT * FROM restaurants;", db_path=db_path)
    df_couriers = query_df("SELECT * FROM delivery_partners;", db_path=db_path)
    df_customers = query_df("SELECT * FROM customers;", db_path=db_path)
    df_orders = query_df("SELECT * FROM orders;", db_path=db_path)
    df_reviews = query_df("SELECT * FROM reviews;", db_path=db_path)

    print(f"Loaded {len(df_orders)} orders, {len(df_reviews)} reviews across {len(df_hubs)} urban hubs.")

    # 2. Feature Engineering
    print("[2/6] Engineering timestamp, geospatial, turnaround, and courier features...")
    fe = FoodPulseFeatureEngineer()
    fe.fit(df_orders, df_hubs, df_restaurants, df_couriers)
    X = fe.transform(df_orders)
    save_feature_engineer(fe, os.path.join(model_dir, "feature_engineer.joblib"))

    # Regression Target: actual_delivery_time_min
    y_reg = df_orders["actual_delivery_time_min"].astype(float)

    # Classification Target: delay_severity ('on_time', 'mild_delay', 'severe_delay')
    y_clf = df_orders["delay_severity"]

    # 3. Train ETA Regressor
    print("[3/6] Training ETA Regressor (Gradient Boosting, Random Forest, Ridge)...")
    eta_model = ETARegressor(model_type="gradient_boosting")
    reg_metrics = eta_model.train(X, y_reg)
    eta_model.save(os.path.join(model_dir, "eta_regressor.joblib"))
    print(f"  -> Regressor Best Model: {reg_metrics['primary_model']}")
    print(f"  -> MAE: {reg_metrics['mae']} min | RMSE: {reg_metrics['rmse']} min | R²: {reg_metrics['r2']}")

    # 4. Train SLA Delay Risk Classifier
    print("[4/6] Training SLA Delay Risk Classifier (Gradient Boosting, Random Forest, Logistic)...")
    delay_clf = DelayClassifier(model_type="gradient_boosting")
    clf_metrics = delay_clf.train(X, y_clf)
    delay_clf.save(os.path.join(model_dir, "delay_classifier.joblib"))
    print(f"  -> Classifier Best Model: {clf_metrics['primary_model']}")
    print(f"  -> Precision: {clf_metrics['precision']} | Recall: {clf_metrics['recall']} | F1: {clf_metrics['f1_score']}")

    # 5. Customer Segmentation (K-Means & PCA)
    print("[5/6] Executing Customer Behavioral Segmentation (K-Means & PCA)...")
    seg_model = CustomerSegmentation(n_clusters=4)
    df_cust_profiles = seg_model.extract_customer_profiles(df_orders, df_reviews)
    eval_clusters = seg_model.evaluate_clusters(df_cust_profiles)
    df_clustered_cust = seg_model.fit_predict(df_cust_profiles)
    seg_model.save(os.path.join(model_dir, "customer_segmentation.joblib"))

    # Save segmented customers to CSV for Streamlit consumption
    data_dir = os.path.dirname(db_path)
    df_clustered_cust.to_csv(os.path.join(data_dir, "customer_segments.csv"), index=False)
    print(f"  -> Identified 4 behavioral personas across {len(df_clustered_cust)} customers.")
    for c_id, stats in seg_model.cluster_summaries.items():
        print(f"     Cluster {c_id}: {stats['persona_name']} ({stats['size']} users, {stats['pct_of_base']}%)")

    # 6. NLP Review Sentiment & Complaint Analysis
    print("[6/6] Fitting NLP Sentiment Analyzer on Customer Reviews...")
    nlp_analyzer = NLPSentimentAnalyzer(max_features=600)
    nlp_analyzer.fit(df_reviews)
    nlp_analyzer.save(os.path.join(model_dir, "nlp_sentiment.joblib"))
    print("  -> Extracted top complaint categories and sentiment n-grams.")

    # 7. Compile Evaluation Report
    report = {
        "dataset_summary": {
            "total_orders": len(df_orders),
            "total_reviews": len(df_reviews),
            "total_customers": len(df_customers),
            "total_restaurants": len(df_restaurants),
            "total_couriers": len(df_couriers),
            "urban_hubs": df_hubs["hub_name"].tolist()
        },
        "eta_regressor_metrics": reg_metrics,
        "delay_classifier_metrics": clf_metrics,
        "customer_segmentation": {
            "n_clusters": seg_model.n_clusters,
            "silhouette_scores": seg_model.silhouette_scores,
            "elbow_inertia": seg_model.elbow_scores,
            "cluster_summaries": seg_model.cluster_summaries
        },
        "nlp_sentiment": {
            "total_reviews_analyzed": len(df_reviews),
            "complaint_distribution": nlp_analyzer.complaint_distribution,
            "top_positive_ngrams": nlp_analyzer.top_positive_ngrams[:8],
            "top_negative_ngrams": nlp_analyzer.top_negative_ngrams[:8]
        }
    }

    report_path = os.path.join(model_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"SUCCESS! All trained models and evaluation reports saved to: {model_dir}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_pipeline()
