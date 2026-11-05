"""
Database Layer for FoodPulse AI Delivery Intelligence System.
Handles SQLite connection, schema migrations, and enterprise SQL analytics queries.
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "foodpulse.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the relational schema with indexes."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS urban_hubs (
        hub_id TEXT PRIMARY KEY,
        hub_name TEXT NOT NULL,
        city TEXT NOT NULL,
        center_lat REAL NOT NULL,
        center_lon REAL NOT NULL,
        traffic_congestion_base REAL NOT NULL,
        avg_density_score REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS restaurants (
        restaurant_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        hub_id TEXT NOT NULL,
        cuisine TEXT NOT NULL,
        prep_time_base REAL NOT NULL,
        kitchen_capacity INTEGER NOT NULL,
        rating REAL NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        FOREIGN KEY (hub_id) REFERENCES urban_hubs(hub_id)
    );

    CREATE TABLE IF NOT EXISTS delivery_partners (
        partner_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        speed_factor REAL NOT NULL,
        rating REAL NOT NULL,
        active_hub_id TEXT NOT NULL,
        FOREIGN KEY (active_hub_id) REFERENCES urban_hubs(hub_id)
    );

    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        hub_id TEXT NOT NULL,
        registered_date TEXT NOT NULL,
        loyalty_tier TEXT NOT NULL,
        FOREIGN KEY (hub_id) REFERENCES urban_hubs(hub_id)
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        restaurant_id TEXT NOT NULL,
        partner_id TEXT NOT NULL,
        hub_id TEXT NOT NULL,
        order_timestamp TEXT NOT NULL,
        day_of_week TEXT NOT NULL,
        hour_of_day INTEGER NOT NULL,
        order_value REAL NOT NULL,
        item_count INTEGER NOT NULL,
        weather TEXT NOT NULL,
        temperature_c REAL NOT NULL,
        distance_km REAL NOT NULL,
        kitchen_prep_time_min REAL NOT NULL,
        transit_time_min REAL NOT NULL,
        actual_delivery_time_min REAL NOT NULL,
        promised_delivery_time_min REAL NOT NULL,
        is_delayed INTEGER NOT NULL,
        delay_minutes REAL NOT NULL,
        delay_severity TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id),
        FOREIGN KEY (partner_id) REFERENCES delivery_partners(partner_id),
        FOREIGN KEY (hub_id) REFERENCES urban_hubs(hub_id)
    );

    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        restaurant_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        review_text TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        sentiment_label TEXT NOT NULL,
        complaint_category TEXT NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
    );

    -- Performance Indexes
    CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(order_timestamp);
    CREATE INDEX IF NOT EXISTS idx_orders_hub ON orders(hub_id);
    CREATE INDEX IF NOT EXISTS idx_orders_restaurant ON orders(restaurant_id);
    CREATE INDEX IF NOT EXISTS idx_orders_is_delayed ON orders(is_delayed);
    CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
    CREATE INDEX IF NOT EXISTS idx_reviews_complaint ON reviews(complaint_category);
    """)

    conn.commit()
    conn.close()


def query_df(query: str, params: tuple = (), db_path: Optional[str] = None) -> pd.DataFrame:
    """Executes a SQL query and returns results as a pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


def execute_sql(query: str, params: tuple = (), db_path: Optional[str] = None) -> None:
    """Executes an INSERT, UPDATE, or DELETE SQL statement."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(query, params)
    finally:
        conn.close()


# Pre-defined Analytical SQL Queries for Operational Insights

ANALYTICAL_QUERIES = {
    "hub_sla_performance": """
        SELECT 
            h.hub_name,
            h.city,
            COUNT(o.order_id) AS total_orders,
            ROUND(AVG(o.kitchen_prep_time_min), 2) AS avg_kitchen_turnaround_min,
            ROUND(AVG(o.transit_time_min), 2) AS avg_transit_min,
            ROUND(AVG(o.actual_delivery_time_min), 2) AS avg_total_delivery_min,
            ROUND(AVG(o.distance_km), 2) AS avg_distance_km,
            ROUND(100.0 * SUM(o.is_delayed) / COUNT(o.order_id), 2) AS delay_rate_pct,
            ROUND(100.0 * SUM(CASE WHEN o.delay_severity = 'severe_delay' THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS severe_delay_pct
        FROM orders o
        JOIN urban_hubs h ON o.hub_id = h.hub_id
        GROUP BY h.hub_id, h.hub_name, h.city
        ORDER BY delay_rate_pct DESC;
    """,

    "hourly_bottleneck_patterns": """
        SELECT 
            hour_of_day,
            COUNT(order_id) AS order_volume,
            ROUND(AVG(kitchen_prep_time_min), 2) AS avg_prep_time,
            ROUND(AVG(transit_time_min), 2) AS avg_transit_time,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
            ROUND(100.0 * SUM(is_delayed) / COUNT(order_id), 2) AS delay_rate_pct
        FROM orders
        GROUP BY hour_of_day
        ORDER BY hour_of_day ASC;
    """,

    "restaurant_turnaround_diagnostics": """
        SELECT 
            r.name AS restaurant_name,
            r.cuisine,
            h.hub_name,
            r.rating AS restaurant_rating,
            COUNT(o.order_id) AS total_orders,
            ROUND(AVG(o.kitchen_prep_time_min), 2) AS avg_prep_time,
            ROUND(MAX(o.kitchen_prep_time_min), 2) AS max_prep_time,
            ROUND(100.0 * SUM(CASE WHEN o.kitchen_prep_time_min > 25 THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS slow_kitchen_rate_pct
        FROM restaurants r
        JOIN orders o ON r.restaurant_id = o.restaurant_id
        JOIN urban_hubs h ON r.hub_id = h.hub_id
        GROUP BY r.restaurant_id, r.name, r.cuisine, h.hub_name
        HAVING COUNT(o.order_id) >= 15
        ORDER BY avg_prep_time DESC;
    """,

    "weather_impact_analysis": """
        SELECT 
            weather,
            COUNT(order_id) AS order_count,
            ROUND(AVG(transit_time_min), 2) AS avg_transit_min,
            ROUND(AVG(actual_delivery_time_min), 2) AS avg_total_delivery_min,
            ROUND(100.0 * SUM(is_delayed) / COUNT(order_id), 2) AS delay_rate_pct,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_min
        FROM orders
        GROUP BY weather
        ORDER BY delay_rate_pct DESC;
    """,

    "customer_review_complaints_breakdown": """
        SELECT 
            complaint_category,
            sentiment_label,
            COUNT(review_id) AS review_count,
            ROUND(AVG(rating), 2) AS avg_review_rating,
            ROUND(AVG(sentiment_score), 3) AS avg_sentiment_score
        FROM reviews
        GROUP BY complaint_category, sentiment_label
        ORDER BY review_count DESC;
    """
}
