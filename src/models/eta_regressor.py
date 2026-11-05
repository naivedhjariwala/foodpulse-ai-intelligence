"""
ETA Regressor for FoodPulse AI Delivery Intelligence System.
Predicts end-to-end delivery duration (minutes) from engineered operational features.
Evaluates Random Forest, Gradient Boosting, and Ridge Regressors.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class ETARegressor:
    """
    Regression model for predicting actual delivery duration in minutes.
    """
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.feature_importances_ = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Splits data, trains models, evaluates metrics, and selects the primary regressor.
        """
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        candidates = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.08,
                max_depth=5,
                random_state=random_state
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=140,
                max_depth=12,
                n_jobs=-1,
                random_state=random_state
            ),
            "ridge": Ridge(alpha=1.0)
        }

        eval_results = {}
        for name, candidate in candidates.items():
            candidate.fit(X_train, y_train)
            preds = candidate.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            eval_results[name] = {
                "mae": round(float(mae), 3),
                "rmse": round(float(rmse), 3),
                "r2": round(float(r2), 4),
                "model_instance": candidate
            }

        # Select primary model (defaults to gradient_boosting)
        best_name = self.model_type if self.model_type in eval_results else "gradient_boosting"
        self.model = eval_results[best_name]["model_instance"]
        self.metrics = {
            "primary_model": best_name,
            "mae": eval_results[best_name]["mae"],
            "rmse": eval_results[best_name]["rmse"],
            "r2": eval_results[best_name]["r2"],
            "benchmark_comparison": {k: {m: v[m] for m in ["mae", "rmse", "r2"]} for k, v in eval_results.items()}
        }

        # Extract feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importances_ = dict(
                sorted(
                    zip(self.feature_names, [round(float(x), 4) for x in importances]),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_)
            self.feature_importances_ = dict(
                sorted(
                    zip(self.feature_names, [round(float(x), 4) for x in coefs]),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates continuous ETA predictions in minutes."""
        if self.model is None:
            raise ValueError("ETARegressor is not fitted yet.")
        preds = self.model.predict(X)
        return np.round(np.clip(preds, a_min=8.0, a_max=120.0), 1)

    def predict_single_with_breakdown(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates prediction for a single order along with estimated kitchen/transit breakdown.
        """
        eta = float(self.predict(X_single)[0])
        # Approximate component contribution from features
        prep_base = float(X_single["prep_time_base"].iloc[0])
        items = float(X_single["item_count"].iloc[0])
        est_prep = round(min(eta * 0.65, prep_base + items * 1.5), 1)
        est_transit = round(max(3.0, eta - est_prep - 3.0), 1)
        est_buffer = round(max(1.5, eta - est_prep - est_transit), 1)

        # Top 3 driving factors for this specific order
        top_factors = []
        if float(X_single["weather_severity_idx"].iloc[0]) >= 2:
            top_factors.append("Adverse Weather Multiplier")
        if float(X_single["is_dinner_rush"].iloc[0]) == 1 or float(X_single["is_lunch_rush"].iloc[0]) == 1:
            top_factors.append("Peak Urban Rush Hour Surge")
        if float(X_single["distance_km"].iloc[0]) >= 7.0:
            top_factors.append("High Hub Transit Distance")
        if float(X_single["items_per_capacity"].iloc[0]) >= 0.15:
            top_factors.append("Kitchen Capacity Saturation")
        if not top_factors:
            top_factors.append("Standard Urban Route Transit")

        return {
            "predicted_eta_minutes": eta,
            "estimated_kitchen_prep_min": est_prep,
            "estimated_transit_min": est_transit,
            "estimated_handover_buffer_min": est_buffer,
            "primary_delay_drivers": top_factors[:3]
        }

    def save(self, filepath: str) -> None:
        """Saves model and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "ETARegressor":
        """Loads serialized model."""
        return joblib.load(filepath)
