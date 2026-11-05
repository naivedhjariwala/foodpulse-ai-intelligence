"""
Synthetic Data Generator for FoodPulse AI Delivery Intelligence System.
Generates multi-hub urban operational data spanning 2025 - 2026.
Simulates realistic bottlenecks, kitchen turnarounds, courier dynamics, weather anomalies, and customer sentiment reviews.
"""

import os
import math
import random
import datetime
import sqlite3
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from src.db import init_db, get_connection, DEFAULT_DB_PATH

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

URBAN_HUBS = [
    {
        "hub_id": "HUB_NYC_01",
        "hub_name": "Manhattan Midtown & Downtown",
        "city": "New York",
        "center_lat": 40.7580,
        "center_lon": -73.9855,
        "traffic_congestion_base": 1.85,
        "avg_density_score": 9.4
    },
    {
        "hub_id": "HUB_SF_02",
        "hub_name": "Financial District & SoMa",
        "city": "San Francisco",
        "center_lat": 37.7891,
        "center_lon": -122.4014,
        "traffic_congestion_base": 1.65,
        "avg_density_score": 8.7
    },
    {
        "hub_id": "HUB_CHI_03",
        "hub_name": "The Loop & River North",
        "city": "Chicago",
        "center_lat": 41.8818,
        "center_lon": -87.6231,
        "traffic_congestion_base": 1.55,
        "avg_density_score": 8.1
    },
    {
        "hub_id": "HUB_ATX_04",
        "hub_name": "Downtown & East Austin",
        "city": "Austin",
        "center_lat": 30.2672,
        "center_lon": -97.7431,
        "traffic_congestion_base": 1.35,
        "avg_density_score": 6.8
    },
    {
        "hub_id": "HUB_LDN_05",
        "hub_name": "City & West End",
        "city": "London",
        "center_lat": 51.5136,
        "center_lon": -0.1165,
        "traffic_congestion_base": 1.75,
        "avg_density_score": 9.1
    }
]

RESTAURANT_TEMPLATES = [
    ("Tokyo Drift Sushi", "Sushi", 16.0, 35, 4.7),
    ("Sakura Ramen House", "Japanese Ramen", 18.0, 40, 4.6),
    ("Napoletana Craft Pizza", "Artisan Pizza", 20.0, 45, 4.5),
    ("Trattoria Bella Roma", "Italian Pasta", 22.0, 30, 4.4),
    ("Prime Smash Burgers", "American Burgers", 14.0, 55, 4.3),
    ("Green Oasis Salad Bar", "Salads & Bowls", 9.0, 60, 4.6),
    ("Spice Symphony Indian", "Indian Curry", 26.0, 28, 4.2),
    ("Bangkok Street Wok", "Thai & Street Food", 17.0, 42, 4.5),
    ("El Molino Cantina", "Mexican Tacos", 15.0, 50, 4.4),
    ("Le Petit Bistro", "French Gourmet", 28.0, 22, 4.7),
    ("Habibi Shawarma & Falafel", "Mediterranean", 12.0, 65, 4.6),
    ("Golden Dragon Dim Sum", "Cantonese Dim Sum", 24.0, 32, 4.3),
]

VEHICLE_PROFILES = [
    ("E-Scooter", 1.25, 0.35),
    ("Bicycle", 1.00, 0.25),
    ("Motorcycle", 1.45, 0.25),
    ("Electric Car", 1.30, 0.15)
]

WEATHER_CONDITIONS = [
    ("Clear", 0.50, 1.00, 22.0),
    ("Cloudy", 0.25, 1.05, 18.0),
    ("Rain", 0.14, 1.32, 14.0),
    ("Heavy Rain / Storm", 0.06, 1.65, 12.0),
    ("Snow / Ice", 0.05, 1.85, -2.0)
]

CUSTOMER_PERSONAS = [
    {"name": "High-Value Gourmet Regulars", "weight": 0.20, "order_val_mean": 68.0, "order_val_std": 20.0, "freq": 14, "peak": "dinner"},
    {"name": "Office Lunch Rushers", "weight": 0.35, "order_val_mean": 24.0, "order_val_std": 7.0, "freq": 22, "peak": "lunch"},
    {"name": "Late-Night Cravers", "weight": 0.20, "order_val_mean": 32.0, "order_val_std": 10.0, "freq": 12, "peak": "late_night"},
    {"name": "Weekend Family Feasters", "weight": 0.15, "order_val_mean": 95.0, "order_val_std": 28.0, "freq": 6, "peak": "weekend"},
    {"name": "Budget Students / Promos", "weight": 0.10, "order_val_mean": 18.0, "order_val_std": 4.0, "freq": 8, "peak": "any"}
]

REVIEW_TEMPLATES = {
    "positive": [
        "Delivered blazing fast and piping hot! Will definitely order again.",
        "Crispy fries, perfectly packed, and the courier arrived right to my door.",
        "Outstanding meal! Arrived 10 minutes ahead of scheduled ETA.",
        "Generous portions, fresh ingredients, seamless tracking.",
        "Super polite delivery driver and everything was securely sealed.",
        "Best Thai in the hub! Arrived hot and fresh on a Friday night.",
        "Five stars! The packaging kept the soup completely spill-free.",
        "Top tier turnaround time, kitchen got the special instructions 100% right."
    ],
    "neutral": [
        "Food was delicious but courier took a bit longer than expected to navigate the apartment lobby.",
        "Decent meal, arrived lukewarm. Microwave solved it.",
        "Slightly longer wait time than promised, but the food quality compensated.",
        "Portion size was average for the price, delivery was right on the dot.",
        "Okay experience overall. Burger was warm but soda ice had melted.",
        "Driver was friendly but ETA kept fluctuating in the tracker."
    ],
    "negative_late": [
        "Unacceptable delivery delay! Order took over 75 minutes, completely ice cold when it arrived.",
        "Driver sat around at the hub for 20 minutes. My dinner was ruined by the massive delay.",
        "Food was supposed to take 30 mins, arrived in 85 mins. Unacceptable for a busy weekday.",
        "Tracked the courier taking three bizarre detours. Delayed by 40 minutes, cold food.",
        "Terrible service today. 50 minutes late and no update from operations."
    ],
    "negative_cold_food": [
        "Pizza cheese had completely solidified into rubber. Cold food upon arrival.",
        "Lukewarm soup and soggy fries. Why wasn't an insulated bag used?",
        "Everything was freezing cold. What happened to thermal transport?",
        "Cold food and congealed sauce. Very disappointed."
    ],
    "negative_missing_items": [
        "Driver gave me the bag with missing drinks and no garlic bread. Kitchen forgot half the order.",
        "Missing items in my order! Paid $65 and only got two out of four entrees.",
        "No dipping sauces and dessert was completely left out of the bag.",
        "Half of our office lunch was missing from the sealed bag."
    ],
    "negative_rude_courier": [
        "Rude courier refused to come up to the lobby and threw the bag on the floor.",
        "Courier was extremely impatient and argumentative on the intercom.",
        "Driver was rude and shoved the bag into my hands without saying a word."
    ],
    "negative_packaging": [
        "Curry sauce leaked completely through the paper bag. Mess all over my table.",
        "Damaged packaging, sushi was completely crushed upside down.",
        "The drink lid was loose and spilled soda all over the burger box."
    ]
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates haversine distance in km between two geo-coordinates."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def generate_synthetic_data(num_orders: int = 12000, db_path: str = DEFAULT_DB_PATH) -> Dict[str, pd.DataFrame]:
    """Generates synthetic dataset and writes to SQLite database and CSV files."""
    print(f"Initializing database at: {db_path}")
    init_db(db_path)
    conn = get_connection(db_path)

    # 1. Generate Hubs
    df_hubs = pd.DataFrame(URBAN_HUBS)
    df_hubs.to_sql("urban_hubs", conn, if_exists="replace", index=False)

    # 2. Generate Restaurants (35 across hubs)
    restaurants_data = []
    rest_id_counter = 1
    for hub in URBAN_HUBS:
        # Create 7 restaurants per hub
        for i in range(7):
            tpl = RESTAURANT_TEMPLATES[(rest_id_counter + i) % len(RESTAURANT_TEMPLATES)]
            lat_jitter = np.random.normal(0, 0.015)
            lon_jitter = np.random.normal(0, 0.018)
            rest_id = f"REST_{hub['hub_id'][-2:]}_{rest_id_counter:03d}"
            restaurants_data.append({
                "restaurant_id": rest_id,
                "name": f"{tpl[0]} ({hub['city']})",
                "hub_id": hub["hub_id"],
                "cuisine": tpl[1],
                "prep_time_base": tpl[2] + np.random.uniform(-2, 3),
                "kitchen_capacity": int(tpl[3] + np.random.randint(-5, 10)),
                "rating": round(tpl[4] + np.random.uniform(-0.3, 0.2), 2),
                "lat": hub["center_lat"] + lat_jitter,
                "lon": hub["center_lon"] + lon_jitter
            })
            rest_id_counter += 1
    df_restaurants = pd.DataFrame(restaurants_data)
    df_restaurants.to_sql("restaurants", conn, if_exists="replace", index=False)

    # 3. Generate Delivery Partners (Couriers)
    couriers_data = []
    courier_id = 1
    first_names = ["Alex", "Jordan", "Marcus", "Elena", "Carlos", "Priya", "Liam", "Chen", "Aisha", "Mateo", "Sophie", "Zack"]
    last_names = ["Silva", "Vance", "Rossi", "Patel", "Smith", "Kim", "O'Connor", "Mendoza", "Novak", "Dubois"]

    for hub in URBAN_HUBS:
        for _ in range(12):  # 12 couriers per hub
            v_type, speed_mult, _ = random.choices(
                VEHICLE_PROFILES, 
                weights=[v[2] for v in VEHICLE_PROFILES]
            )[0]
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            couriers_data.append({
                "partner_id": f"DP_{courier_id:04d}",
                "name": name,
                "vehicle_type": v_type,
                "speed_factor": round(speed_mult * np.random.uniform(0.92, 1.10), 2),
                "rating": round(np.random.beta(8, 1.2) * 2.0 + 3.0, 2), # skew 4.2 - 4.9
                "active_hub_id": hub["hub_id"]
            })
            courier_id += 1
    df_couriers = pd.DataFrame(couriers_data)
    df_couriers.to_sql("delivery_partners", conn, if_exists="replace", index=False)

    # 4. Generate Customers (1,200 customers across hubs)
    customers_data = []
    for c_idx in range(1, 1201):
        hub = random.choice(URBAN_HUBS)
        persona = random.choices(
            CUSTOMER_PERSONAS,
            weights=[p["weight"] for p in CUSTOMER_PERSONAS]
        )[0]
        customers_data.append({
            "customer_id": f"CUST_{c_idx:05d}",
            "name": f"User {c_idx}",
            "email": f"user{c_idx}@pulsemail.io",
            "hub_id": hub["hub_id"],
            "registered_date": (datetime.date(2025, 1, 1) + datetime.timedelta(days=random.randint(0, 500))).isoformat(),
            "loyalty_tier": random.choices(["Bronze", "Silver", "Gold", "Platinum"], weights=[0.45, 0.30, 0.18, 0.07])[0],
            "persona_archetype": persona["name"]
        })
    df_customers = pd.DataFrame(customers_data)
    # Save base schema without internal persona_archetype in SQL
    df_customers[["customer_id", "name", "email", "hub_id", "registered_date", "loyalty_tier"]].to_sql(
        "customers", conn, if_exists="replace", index=False
    )

    # 5. Generate Orders spanning Jan 2025 – Oct 2026
    start_date = datetime.datetime(2025, 1, 1, 10, 0, 0)
    end_date = datetime.datetime(2026, 10, 31, 23, 0, 0)
    total_seconds = int((end_date - start_date).total_seconds())

    # Pre-index objects for fast lookups
    hub_dict = {h["hub_id"]: h for h in URBAN_HUBS}
    rest_dict = df_restaurants.set_index("restaurant_id").to_dict("index")
    rests_by_hub = df_restaurants.groupby("hub_id")["restaurant_id"].apply(list).to_dict()
    couriers_by_hub = df_couriers.groupby("active_hub_id")["partner_id"].apply(list).to_dict()
    courier_dict = df_couriers.set_index("partner_id").to_dict("index")
    cust_list = df_customers.to_dict("records")

    orders_data = []
    reviews_data = []

    # Weather probability distribution
    weather_labels = [w[0] for w in WEATHER_CONDITIONS]
    weather_weights = [w[1] for w in WEATHER_CONDITIONS]
    weather_factors = {w[0]: (w[2], w[3]) for w in WEATHER_CONDITIONS}

    raw_hour_probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 00-05
        0.02, 0.03, 0.04, 0.05, 0.07, 0.11,     # 06-11
        0.14, 0.08, 0.05, 0.04, 0.05, 0.07,     # 12-17
        0.11, 0.12, 0.08, 0.05, 0.03, 0.015     # 18-23
    ], dtype=float)
    hour_probs = raw_hour_probs / raw_hour_probs.sum()

    print(f"Generating {num_orders} orders with feature dynamics...")
    for o_idx in range(1, num_orders + 1):
        hour = int(np.random.choice(range(24), p=hour_probs))
        day_offset = random.randint(0, (end_date - start_date).days)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        order_time = (start_date + datetime.timedelta(days=day_offset)).replace(hour=hour, minute=minute, second=second)

        day_of_week = order_time.strftime("%A")
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        hour_val = order_time.hour

        # Customer & Hub
        cust = random.choice(cust_list)
        hub_id = cust["hub_id"]
        hub_info = hub_dict[hub_id]

        # Restaurant in same hub
        rest_id = random.choice(rests_by_hub[hub_id])
        rest_info = rest_dict[rest_id]

        # Delivery Partner in same hub
        partner_id = random.choice(couriers_by_hub[hub_id])
        partner_info = courier_dict[partner_id]

        # Weather condition
        weather = random.choices(weather_labels, weights=weather_weights)[0]
        weather_mult, base_temp = weather_factors[weather]
        temperature_c = round(base_temp + np.random.normal(0, 4.0), 1)

        # Distance: Delivery location within hub radius (1.0 km to 11.0 km)
        cust_lat = hub_info["center_lat"] + np.random.normal(0, 0.025)
        cust_lon = hub_info["center_lon"] + np.random.normal(0, 0.028)
        distance_km = haversine_distance(rest_info["lat"], rest_info["lon"], cust_lat, cust_lon)
        distance_km = max(0.8, min(distance_km, 14.5))

        # Basket items & order value
        item_count = max(1, int(np.random.poisson(3.2)))
        order_value = round(max(12.0, (item_count * 14.5) + np.random.normal(10, 15)), 2)

        # Kitchen Turnaround Dynamics:
        # 1. Base prep time
        # 2. Kitchen volume backlog penalty during lunch/dinner peaks
        peak_factor = 1.0
        if 11 <= hour_val <= 13 or 18 <= hour_val <= 21:
            peak_factor = 1.35
            if is_weekend:
                peak_factor = 1.50

        # Kitchen capacity stress: if restaurant rating is lower or capacity is small
        capacity_stress = max(0.9, 45.0 / rest_info["kitchen_capacity"])
        prep_noise = np.random.normal(0, 2.5)

        kitchen_prep_time_min = (
            rest_info["prep_time_base"] 
            + (item_count * 1.4) 
            + (peak_factor * 3.5 * capacity_stress)
            + prep_noise
        )
        kitchen_prep_time_min = round(max(8.0, kitchen_prep_time_min), 2)

        # Transit Time Dynamics:
        # Base speed in city ~ 20 km/h (0.33 km/min -> ~3 min per km)
        # Hub congestion multiplier
        hub_congestion = hub_info["traffic_congestion_base"]
        if (8 <= hour_val <= 10) or (17 <= hour_val <= 19):
            hub_congestion *= 1.30  # Commute traffic rush

        vehicle_speed_factor = partner_info["speed_factor"]  # Higher is faster
        base_transit_speed_min_per_km = 3.2 / vehicle_speed_factor

        transit_time_min = (
            (distance_km * base_transit_speed_min_per_km * hub_congestion * weather_mult)
            + np.random.normal(0, 2.0)
        )
        transit_time_min = round(max(4.0, transit_time_min), 2)

        courier_handover_min = round(random.uniform(2.0, 4.5), 1)

        # Total Actual Delivery Time
        actual_delivery_time_min = round(kitchen_prep_time_min + transit_time_min + courier_handover_min, 2)

        # Promised ETA (Standard algorithmic baseline without real-time AI adjustment)
        # Baseline typically estimates: standard prep + flat transit estimate + small buffer
        standard_prep = rest_info["prep_time_base"] + (item_count * 1.0)
        standard_transit = distance_km * 3.8
        promised_delivery_time_min = round(standard_prep + standard_transit + 7.0, 2)

        # Delay computation
        delay_delta = actual_delivery_time_min - promised_delivery_time_min
        if delay_delta <= 1.0:
            is_delayed = 0
            delay_minutes = 0.0
            delay_severity = "on_time"
        elif 1.0 < delay_delta <= 14.0:
            is_delayed = 1
            delay_minutes = round(delay_delta, 2)
            delay_severity = "mild_delay"
        else:
            is_delayed = 1
            delay_minutes = round(delay_delta, 2)
            delay_severity = "severe_delay"

        order_id = f"ORD_{order_time.strftime('%y%m')}_{o_idx:06d}"

        orders_data.append({
            "order_id": order_id,
            "customer_id": cust["customer_id"],
            "restaurant_id": rest_id,
            "partner_id": partner_id,
            "hub_id": hub_id,
            "order_timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": day_of_week,
            "hour_of_day": hour_val,
            "order_value": order_value,
            "item_count": item_count,
            "weather": weather,
            "temperature_c": temperature_c,
            "distance_km": distance_km,
            "kitchen_prep_time_min": kitchen_prep_time_min,
            "transit_time_min": transit_time_min,
            "actual_delivery_time_min": actual_delivery_time_min,
            "promised_delivery_time_min": promised_delivery_time_min,
            "is_delayed": is_delayed,
            "delay_minutes": delay_minutes,
            "delay_severity": delay_severity
        })

        # Generate Reviews for ~35% of orders
        if random.random() < 0.35:
            # Rating & Sentiment determined by delay and performance
            if delay_severity == "on_time":
                rating = random.choices([4, 5, 3], weights=[0.40, 0.55, 0.05])[0]
                if rating >= 4:
                    rev_text = random.choice(REVIEW_TEMPLATES["positive"])
                    sent_label = "Positive"
                    sent_score = round(random.uniform(0.65, 0.95), 3)
                    complaint = "None (Praise)"
                else:
                    rev_text = random.choice(REVIEW_TEMPLATES["neutral"])
                    sent_label = "Neutral"
                    sent_score = round(random.uniform(-0.10, 0.25), 3)
                    complaint = "Minor Inconvenience"

            elif delay_severity == "mild_delay":
                rating = random.choices([2, 3, 4], weights=[0.25, 0.55, 0.20])[0]
                if rating == 4:
                    rev_text = random.choice(REVIEW_TEMPLATES["neutral"])
                    sent_label = "Neutral"
                    sent_score = round(random.uniform(0.10, 0.40), 3)
                    complaint = "Minor Inconvenience"
                elif rating == 3:
                    rev_text = random.choice(REVIEW_TEMPLATES["neutral"])
                    sent_label = "Neutral"
                    sent_score = round(random.uniform(-0.25, 0.15), 3)
                    complaint = "Mild Turnaround Delay"
                else:
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_late"])
                    sent_label = "Negative"
                    sent_score = round(random.uniform(-0.75, -0.30), 3)
                    complaint = "Late Delivery"

            else:  # severe_delay
                rating = random.choices([1, 2], weights=[0.70, 0.30])[0]
                sent_label = "Negative"
                sent_score = round(random.uniform(-0.95, -0.45), 3)

                complaint_pick = random.choices(
                    ["Late Delivery", "Cold Food", "Missing Items", "Rude Courier", "Damaged Packaging"],
                    weights=[0.45, 0.25, 0.12, 0.08, 0.10]
                )[0]

                if complaint_pick == "Late Delivery":
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_late"])
                elif complaint_pick == "Cold Food":
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_cold_food"])
                elif complaint_pick == "Missing Items":
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_missing_items"])
                elif complaint_pick == "Rude Courier":
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_rude_courier"])
                else:
                    rev_text = random.choice(REVIEW_TEMPLATES["negative_packaging"])

                complaint = complaint_pick

            reviews_data.append({
                "review_id": f"REV_{len(reviews_data)+1:06d}",
                "order_id": order_id,
                "customer_id": cust["customer_id"],
                "restaurant_id": rest_id,
                "rating": rating,
                "review_text": rev_text,
                "sentiment_score": sent_score,
                "sentiment_label": sent_label,
                "complaint_category": complaint
            })

    print(f"Writing {len(orders_data)} orders to database...")
    df_orders = pd.DataFrame(orders_data)
    df_orders.to_sql("orders", conn, if_exists="replace", index=False)

    print(f"Writing {len(reviews_data)} reviews to database...")
    df_reviews = pd.DataFrame(reviews_data)
    df_reviews.to_sql("reviews", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    # Also save as CSVs in data/ for direct portability and inspection
    data_dir = os.path.dirname(db_path)
    df_orders.to_csv(os.path.join(data_dir, "orders.csv"), index=False)
    df_reviews.to_csv(os.path.join(data_dir, "reviews.csv"), index=False)
    df_restaurants.to_csv(os.path.join(data_dir, "restaurants.csv"), index=False)
    df_hubs.to_csv(os.path.join(data_dir, "urban_hubs.csv"), index=False)
    df_couriers.to_csv(os.path.join(data_dir, "delivery_partners.csv"), index=False)
    df_customers.to_csv(os.path.join(data_dir, "customers.csv"), index=False)

    print("Data generation complete! All tables and CSV artifacts saved.")

    return {
        "hubs": df_hubs,
        "restaurants": df_restaurants,
        "couriers": df_couriers,
        "customers": df_customers,
        "orders": df_orders,
        "reviews": df_reviews
    }


if __name__ == "__main__":
    generate_synthetic_data(num_orders=12500)
