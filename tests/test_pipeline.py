"""
Unit and Integration Tests for FoodPulse AI Delivery Intelligence System.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.db import query_df, ANALYTICAL_QUERIES, DEFAULT_DB_PATH
from src.features import FoodPulseFeatureEngineer, FEATURE_COLUMNS, load_feature_engineer
from src.models.eta_regressor import ETARegressor
from src.models.delay_classifier import DelayClassifier
from src.models.customer_segmentation import CustomerSegmentation
from src.models.nlp_sentiment import NLPSentimentAnalyzer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def test_database_and_tables():
    """Validates that SQLite tables exist and contain records."""
    assert os.path.exists(DEFAULT_DB_PATH), "Database file does not exist"
    
    df_hubs = query_df("SELECT * FROM urban_hubs;")
    assert len(df_hubs) >= 5, "Urban hubs table should have at least 5 hubs"

    df_orders = query_df("SELECT COUNT(*) AS cnt FROM orders;")
    assert df_orders["cnt"].iloc[0] >= 1000, "Orders table should contain records"

    df_reviews = query_df("SELECT COUNT(*) AS cnt FROM reviews;")
    assert df_reviews["cnt"].iloc[0] >= 500, "Reviews table should contain records"


def test_analytical_queries():
    """Ensures analytical SQL queries execute without syntax errors."""
    for query_name, sql in ANALYTICAL_QUERIES.items():
        df = query_df(sql)
        assert isinstance(df, pd.DataFrame), f"Query {query_name} should return a DataFrame"
        assert not df.empty, f"Query {query_name} returned empty results"


def test_feature_engineering_single():
    """Validates single real-time inference feature engineering."""
    fe_path = os.path.join(MODEL_DIR, "feature_engineer.joblib")
    assert os.path.exists(fe_path), "Fitted feature engineer artifact missing"
    fe = load_feature_engineer(fe_path)

    sample_input = {
        "hour_of_day": 19.5,
        "day_of_week": 4,
        "distance_km": 4.5,
        "hub_density_score": 9.2,
        "hub_traffic_base": 1.8,
        "prep_time_base": 18.0,
        "kitchen_capacity": 40,
        "rest_rating": 4.6,
        "item_count": 3,
        "order_value": 45.0,
        "courier_speed_factor": 1.25,
        "courier_rating": 4.8,
        "vehicle_type": "E-Scooter",
        "weather": "Rain",
        "temperature_c": 14.0
    }

    df_feat = fe.transform_single(sample_input)
    assert list(df_feat.columns) == FEATURE_COLUMNS, "Columns do not match FEATURE_COLUMNS"
    assert not df_feat.isna().any().any(), "Features contain unexpected NaN values"
    assert df_feat["is_adverse_weather"].iloc[0] == 1, "Rain should trigger is_adverse_weather=1"


def test_eta_regressor_inference():
    """Validates ETA Regressor loading and inference."""
    reg_path = os.path.join(MODEL_DIR, "eta_regressor.joblib")
    assert os.path.exists(reg_path), "Regressor model artifact missing"
    reg = ETARegressor.load(reg_path)

    fe = load_feature_engineer(os.path.join(MODEL_DIR, "feature_engineer.joblib"))
    df_feat = fe.transform_single({"distance_km": 5.0, "item_count": 4, "weather": "Clear"})

    res = reg.predict_single_with_breakdown(df_feat)
    assert "predicted_eta_minutes" in res
    assert 10.0 <= res["predicted_eta_minutes"] <= 120.0, "Predicted ETA outside realistic bounds"
    assert res["estimated_kitchen_prep_min"] > 0
    assert res["estimated_transit_min"] > 0
    assert len(res["primary_delay_drivers"]) >= 1


def test_delay_classifier_inference():
    """Validates Delay Risk Classifier predictions and risk tiers."""
    clf_path = os.path.join(MODEL_DIR, "delay_classifier.joblib")
    assert os.path.exists(clf_path), "Classifier model artifact missing"
    clf = DelayClassifier.load(clf_path)

    fe = load_feature_engineer(os.path.join(MODEL_DIR, "feature_engineer.joblib"))
    df_feat = fe.transform_single({"distance_km": 11.0, "weather": "Snow / Ice", "hour_of_day": 19})

    risk = clf.predict_risk_single(df_feat)
    assert "delay_risk_percentage" in risk
    assert 0.0 <= risk["delay_risk_percentage"] <= 100.0
    assert "recommended_mitigation" in risk
    assert risk["risk_tier"] in ["LOW RISK (Green)", "MODERATE RISK (Yellow)", "CRITICAL / HIGH RISK (Red)"]


def test_nlp_sentiment_analyzer():
    """Validates NLP polarity classification and complaint category tagging."""
    nlp_path = os.path.join(MODEL_DIR, "nlp_sentiment.joblib")
    assert os.path.exists(nlp_path), "NLP model artifact missing"
    nlp = NLPSentimentAnalyzer.load(nlp_path)

    # Positive review
    pos_res = nlp.analyze_single_review("Arrived blazing fast, piping hot, and driver was polite!")
    assert pos_res["sentiment_label"] == "Positive"
    assert pos_res["polarity_score"] > 0

    # Negative review with complaint
    neg_res = nlp.analyze_single_review("Delayed by 50 minutes, arrived completely ice cold and half the items were missing!")
    assert neg_res["sentiment_label"] == "Negative"
    assert neg_res["polarity_score"] < 0
    assert any("Late" in c or "Cold" in c or "Missing" in c for c in neg_res["detected_complaints"])
