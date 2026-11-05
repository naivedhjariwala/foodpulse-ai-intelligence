"""
Feature Engineering Pipeline for FoodPulse AI Delivery Intelligence System.
Transforms raw timestamp, geospatial, restaurant turnaround, courier dynamics,
and weather data into rich feature matrices for regression and classification.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import joblib

WEATHER_SEVERITY_MAP = {
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Heavy Rain / Storm": 3,
    "Snow / Ice": 4
}

SEVERITY_LABEL_TO_INT = {
    "on_time": 0,
    "mild_delay": 1,
    "severe_delay": 2
}

SEVERITY_INT_TO_LABEL = {v: k for k, v in SEVERITY_LABEL_TO_INT.items()}

FEATURE_COLUMNS = [
    # Timestamp cyclical & behavioral
    "sin_hour", "cos_hour", "sin_day", "cos_day",
    "is_weekend", "is_lunch_rush", "is_dinner_rush", "is_late_night",
    # Geospatial & Hub
    "distance_km", "hub_density_score", "hub_traffic_base",
    "distance_x_density", "distance_x_traffic",
    # Restaurant turnaround & capacity
    "prep_time_base", "kitchen_capacity", "rest_rating",
    "item_count", "order_value", "items_per_capacity", "val_per_item",
    # Courier dynamics
    "courier_speed_factor", "courier_rating",
    # Vehicle types (one-hot)
    "vehicle_Bicycle", "vehicle_E-Scooter", "vehicle_Electric Car", "vehicle_Motorcycle",
    # Weather & environmental
    "weather_severity_idx", "temperature_c", "is_adverse_weather", "dist_x_weather_severity"
]


class FoodPulseFeatureEngineer:
    """
    Transforms raw operational data into numerical ML feature matrices.
    Maintains metadata for restaurant averages and hub properties.
    """
    def __init__(self):
        self.hub_metadata = {}
        self.restaurant_metadata = {}
        self.courier_metadata = {}
        self.is_fitted = False

    def fit(self, df_orders: pd.DataFrame, df_hubs: pd.DataFrame, 
            df_restaurants: pd.DataFrame, df_couriers: pd.DataFrame) -> "FoodPulseFeatureEngineer":
        """Fits metadata maps from reference data."""
        # Hub metadata
        for _, r in df_hubs.iterrows():
            self.hub_metadata[r["hub_id"]] = {
                "avg_density_score": float(r["avg_density_score"]),
                "traffic_congestion_base": float(r["traffic_congestion_base"])
            }

        # Restaurant metadata
        for _, r in df_restaurants.iterrows():
            self.restaurant_metadata[r["restaurant_id"]] = {
                "prep_time_base": float(r["prep_time_base"]),
                "kitchen_capacity": int(r["kitchen_capacity"]),
                "rating": float(r["rating"]),
                "cuisine": str(r["cuisine"])
            }

        # Courier metadata
        for _, r in df_couriers.iterrows():
            self.courier_metadata[r["partner_id"]] = {
                "speed_factor": float(r["speed_factor"]),
                "rating": float(r["rating"]),
                "vehicle_type": str(r["vehicle_type"])
            }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms an orders DataFrame into engineered features."""
        df_out = pd.DataFrame(index=df.index)

        # 1. Timestamp Features
        timestamps = pd.to_datetime(df["order_timestamp"])
        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        days = timestamps.dt.dayofweek

        df_out["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        df_out["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
        df_out["sin_day"] = np.sin(2 * np.pi * days / 7.0)
        df_out["cos_day"] = np.cos(2 * np.pi * days / 7.0)

        df_out["is_weekend"] = (days >= 5).astype(int)
        df_out["is_lunch_rush"] = ((hours >= 11.5) & (hours <= 14.0)).astype(int)
        df_out["is_dinner_rush"] = ((hours >= 18.0) & (hours <= 21.5)).astype(int)
        df_out["is_late_night"] = ((hours >= 22.0) | (hours <= 4.5)).astype(int)

        # 2. Geospatial & Hub Properties
        hub_densities = []
        hub_traffics = []
        for h_id in df["hub_id"]:
            meta = self.hub_metadata.get(h_id, {"avg_density_score": 8.0, "traffic_congestion_base": 1.5})
            hub_densities.append(meta["avg_density_score"])
            hub_traffics.append(meta["traffic_congestion_base"])

        df_out["distance_km"] = df["distance_km"].astype(float)
        df_out["hub_density_score"] = hub_densities
        df_out["hub_traffic_base"] = hub_traffics
        df_out["distance_x_density"] = df_out["distance_km"] * df_out["hub_density_score"]
        df_out["distance_x_traffic"] = df_out["distance_km"] * df_out["hub_traffic_base"]

        # 3. Restaurant turnaround & capacity
        prep_bases = []
        capacities = []
        rest_ratings = []
        for r_id in df["restaurant_id"]:
            meta = self.restaurant_metadata.get(r_id, {
                "prep_time_base": 18.0, "kitchen_capacity": 40, "rating": 4.5
            })
            prep_bases.append(meta["prep_time_base"])
            capacities.append(meta["kitchen_capacity"])
            rest_ratings.append(meta["rating"])

        df_out["prep_time_base"] = prep_bases
        df_out["kitchen_capacity"] = capacities
        df_out["rest_rating"] = rest_ratings

        df_out["item_count"] = df["item_count"].astype(float)
        df_out["order_value"] = df["order_value"].astype(float)
        df_out["items_per_capacity"] = df_out["item_count"] / df_out["kitchen_capacity"].replace(0, 1)
        df_out["val_per_item"] = df_out["order_value"] / df_out["item_count"].replace(0, 1)

        # 4. Courier dynamics
        speeds = []
        c_ratings = []
        vehicles = []
        for p_id in df["partner_id"]:
            meta = self.courier_metadata.get(p_id, {
                "speed_factor": 1.2, "rating": 4.6, "vehicle_type": "E-Scooter"
            })
            speeds.append(meta["speed_factor"])
            c_ratings.append(meta["rating"])
            vehicles.append(meta["vehicle_type"])

        df_out["courier_speed_factor"] = speeds
        df_out["courier_rating"] = c_ratings

        # One-hot vehicle types
        for v in ["Bicycle", "E-Scooter", "Electric Car", "Motorcycle"]:
            df_out[f"vehicle_{v}"] = [1 if veh == v else 0 for veh in vehicles]

        # 5. Weather & Environmental
        weather_sev = [WEATHER_SEVERITY_MAP.get(w, 0) for w in df["weather"]]
        df_out["weather_severity_idx"] = weather_sev
        df_out["temperature_c"] = df["temperature_c"].astype(float)
        df_out["is_adverse_weather"] = [1 if s >= 2 else 0 for s in weather_sev]
        df_out["dist_x_weather_severity"] = df_out["distance_km"] * df_out["weather_severity_idx"]

        # Ensure exact column order
        return df_out[FEATURE_COLUMNS]

    def transform_single(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms a single real-time inference request dict into a 1-row feature DataFrame.
        """
        hour = params.get("hour_of_day", 19.0)
        day = params.get("day_of_week", 4)  # 0=Monday, 6=Sunday

        sin_hour = math.sin(2 * math.pi * hour / 24.0)
        cos_hour = math.cos(2 * math.pi * hour / 24.0)
        sin_day = math.sin(2 * math.pi * day / 7.0)
        cos_day = math.cos(2 * math.pi * day / 7.0)

        is_weekend = 1 if day >= 5 else 0
        is_lunch_rush = 1 if (11.5 <= hour <= 14.0) else 0
        is_dinner_rush = 1 if (18.0 <= hour <= 21.5) else 0
        is_late_night = 1 if (hour >= 22.0 or hour <= 4.5) else 0

        distance_km = float(params.get("distance_km", 4.2))
        hub_density_score = float(params.get("hub_density_score", 9.0))
        hub_traffic_base = float(params.get("hub_traffic_base", 1.8))

        prep_time_base = float(params.get("prep_time_base", 18.0))
        kitchen_capacity = int(params.get("kitchen_capacity", 40))
        rest_rating = float(params.get("rest_rating", 4.6))

        item_count = float(params.get("item_count", 3))
        order_value = float(params.get("order_value", 45.0))

        courier_speed_factor = float(params.get("courier_speed_factor", 1.25))
        courier_rating = float(params.get("courier_rating", 4.7))
        vehicle_type = params.get("vehicle_type", "E-Scooter")

        weather = params.get("weather", "Clear")
        weather_sev = WEATHER_SEVERITY_MAP.get(weather, 0)
        temp_c = float(params.get("temperature_c", 20.0))

        row = {
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "sin_day": sin_day,
            "cos_day": cos_day,
            "is_weekend": is_weekend,
            "is_lunch_rush": is_lunch_rush,
            "is_dinner_rush": is_dinner_rush,
            "is_late_night": is_late_night,
            "distance_km": distance_km,
            "hub_density_score": hub_density_score,
            "hub_traffic_base": hub_traffic_base,
            "distance_x_density": distance_km * hub_density_score,
            "distance_x_traffic": distance_km * hub_traffic_base,
            "prep_time_base": prep_time_base,
            "kitchen_capacity": kitchen_capacity,
            "rest_rating": rest_rating,
            "item_count": item_count,
            "order_value": order_value,
            "items_per_capacity": item_count / max(1, kitchen_capacity),
            "val_per_item": order_value / max(1, item_count),
            "courier_speed_factor": courier_speed_factor,
            "courier_rating": courier_rating,
            "vehicle_Bicycle": 1 if vehicle_type == "Bicycle" else 0,
            "vehicle_E-Scooter": 1 if vehicle_type == "E-Scooter" else 0,
            "vehicle_Electric Car": 1 if vehicle_type == "Electric Car" else 0,
            "vehicle_Motorcycle": 1 if vehicle_type == "Motorcycle" else 0,
            "weather_severity_idx": weather_sev,
            "temperature_c": temp_c,
            "is_adverse_weather": 1 if weather_sev >= 2 else 0,
            "dist_x_weather_severity": distance_km * weather_sev
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]


def save_feature_engineer(fe: FoodPulseFeatureEngineer, filepath: str) -> None:
    """Serializes the feature engineer instance."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(fe, filepath)


def load_feature_engineer(filepath: str) -> FoodPulseFeatureEngineer:
    """Loads serialized feature engineer instance."""
    return joblib.load(filepath)

