"""
Delay Risk Classifier for FoodPulse AI Delivery Intelligence System.
Predicts delivery SLA breach risk (binary delayed/on-time) and multiclass delay severity
(On-Time, Mild Delay, Severe Delay) with calibrated probabilities and mitigation recommendations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

SEVERITY_CLASSES = ["on_time", "mild_delay", "severe_delay"]


class DelayClassifier:
    """
    Classification model forecasting delivery SLA breaches and severity tiers.
    """
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.feature_importances_ = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains multi-class classification models and evaluates precision/recall/F1.
        y can be categorical string labels ('on_time', 'mild_delay', 'severe_delay')
        or integer encoded.
        """
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        candidates = {
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                random_state=random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=140,
                max_depth=12,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            )
        }

        eval_results = {}
        for name, candidate in candidates.items():
            candidate.fit(X_train, y_train)
            preds = candidate.predict(X_test)
            probs = candidate.predict_proba(X_test) if hasattr(candidate, "predict_proba") else None

            prec_weighted = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec_weighted = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
            
            # ROC-AUC if multi-class probabilities are available
            auc_val = None
            if probs is not None and len(np.unique(y_test)) > 1:
                try:
                    auc_val = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
                except Exception:
                    auc_val = None

            eval_results[name] = {
                "precision": round(float(prec_weighted), 4),
                "recall": round(float(rec_weighted), 4),
                "f1_score": round(float(f1_weighted), 4),
                "roc_auc": round(float(auc_val), 4) if auc_val is not None else "N/A",
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                "model_instance": candidate
            }

        # Select primary model
        best_name = self.model_type if self.model_type in eval_results else "gradient_boosting"
        self.model = eval_results[best_name]["model_instance"]
        
        # Test predictions for detailed classification report
        test_preds = self.model.predict(X_test)
        report_dict = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

        self.metrics = {
            "primary_model": best_name,
            "precision": eval_results[best_name]["precision"],
            "recall": eval_results[best_name]["recall"],
            "f1_score": eval_results[best_name]["f1_score"],
            "roc_auc": eval_results[best_name]["roc_auc"],
            "confusion_matrix": eval_results[best_name]["confusion_matrix"],
            "classification_report": report_dict,
            "benchmark_comparison": {
                k: {m: v[m] for m in ["precision", "recall", "f1_score", "roc_auc"]}
                for k, v in eval_results.items()
            }
        }

        # Feature importances
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
            coefs = np.mean(np.abs(self.model.coef_), axis=0)
            self.feature_importances_ = dict(
                sorted(
                    zip(self.feature_names, [round(float(x), 4) for x in coefs]),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts delay severity class."""
        if self.model is None:
            raise ValueError("DelayClassifier is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities."""
        if self.model is None:
            raise ValueError("DelayClassifier is not fitted yet.")
        return self.model.predict_proba(X)

    def predict_risk_single(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes detailed risk analysis, probability breakdown, and mitigation actions for a single order.
        """
        probs = self.predict_proba(X_single)[0]
        classes = list(self.model.classes_)
        class_prob_map = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}

        # Combined delay risk = sum of mild_delay + severe_delay probabilities
        delay_risk_pct = round(
            float(class_prob_map.get("mild_delay", 0.0) + class_prob_map.get("severe_delay", 0.0)) * 100, 1
        )
        severe_risk_pct = round(float(class_prob_map.get("severe_delay", 0.0)) * 100, 1)

        predicted_severity = self.predict(X_single)[0]

        if delay_risk_pct < 25.0:
            risk_tier = "LOW RISK (Green)"
            color_hex = "#22c55e"
            mitigation = "Standard workflow. High confidence in on-time customer delivery."
        elif delay_risk_pct < 60.0:
            risk_tier = "MODERATE RISK (Yellow)"
            color_hex = "#eab308"
            mitigation = "Alert courier to expedite pickup; prioritize restaurant packing station."
        else:
            risk_tier = "CRITICAL / HIGH RISK (Red)"
            color_hex = "#ef4444"
            mitigation = "Immediate SLA risk: dispatch priority motorized courier, throttle new orders for restaurant, and send proactive app notification to customer."

        return {
            "predicted_severity": str(predicted_severity),
            "delay_risk_percentage": delay_risk_pct,
            "severe_delay_percentage": severe_risk_pct,
            "class_probabilities": class_prob_map,
            "risk_tier": risk_tier,
            "color_hex": color_hex,
            "recommended_mitigation": mitigation
        }

    def save(self, filepath: str) -> None:
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "DelayClassifier":
        """Loads serialized model."""
        return joblib.load(filepath)

