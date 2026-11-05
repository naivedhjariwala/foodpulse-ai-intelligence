"""
FoodPulse: AI Delivery Intelligence & Prediction System (2025 – 2026)
Interactive Streamlit Application for Operations Teams, Analytics, and Real-Time Risk Simulation.
"""

import os
import sys
import json
import tempfile

# Configure Matplotlib temp directory before importing
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_foodpulse")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from src.db import query_df, ANALYTICAL_QUERIES, DEFAULT_DB_PATH
from src.features import load_feature_engineer, FoodPulseFeatureEngineer, WEATHER_SEVERITY_MAP
from src.models.eta_regressor import ETARegressor
from src.models.delay_classifier import DelayClassifier
from src.models.customer_segmentation import CustomerSegmentation
from src.models.nlp_sentiment import NLPSentimentAnalyzer

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Streamlit Page Setup
st.set_page_config(
    page_title="FoodPulse: AI Delivery Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2e3440;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-on-time {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-mild {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-severe {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .prediction-box {
        background: #111827;
        border-left: 5px solid #ff4b4b;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Caching Data and Models
@st.cache_data(ttl=3600)
def load_app_data():
    """Loads operational tables from SQLite or CSV cache."""
    hubs = query_df("SELECT * FROM urban_hubs ORDER BY hub_name ASC;")
    restaurants = query_df("SELECT * FROM restaurants ORDER BY name ASC;")
    couriers = query_df("SELECT * FROM delivery_partners;")
    orders = query_df("SELECT * FROM orders ORDER BY order_timestamp DESC;")
    reviews = query_df("SELECT * FROM reviews ORDER BY rating ASC;")

    cust_seg_path = os.path.join(DATA_DIR, "customer_segments.csv")
    if os.path.exists(cust_seg_path):
        customers = pd.read_csv(cust_seg_path)
    else:
        customers = query_df("SELECT * FROM customers;")

    return hubs, restaurants, couriers, orders, reviews, customers


@st.cache_resource
def load_ml_models():
    """Loads trained artifacts."""
    fe_path = os.path.join(MODEL_DIR, "feature_engineer.joblib")
    reg_path = os.path.join(MODEL_DIR, "eta_regressor.joblib")
    clf_path = os.path.join(MODEL_DIR, "delay_classifier.joblib")
    seg_path = os.path.join(MODEL_DIR, "customer_segmentation.joblib")
    nlp_path = os.path.join(MODEL_DIR, "nlp_sentiment.joblib")
    rep_path = os.path.join(MODEL_DIR, "evaluation_report.json")

    fe = load_feature_engineer(fe_path) if os.path.exists(fe_path) else None
    reg = ETARegressor.load(reg_path) if os.path.exists(reg_path) else None
    clf = DelayClassifier.load(clf_path) if os.path.exists(clf_path) else None
    seg = CustomerSegmentation.load(seg_path) if os.path.exists(seg_path) else None
    nlp = NLPSentimentAnalyzer.load(nlp_path) if os.path.exists(nlp_path) else None

    report = {}
    if os.path.exists(rep_path):
        with open(rep_path, "r") as f:
            report = json.load(f)

    return fe, reg, clf, seg, nlp, report


try:
    df_hubs, df_restaurants, df_couriers, df_orders, df_reviews, df_customers = load_app_data()
    fe, reg_model, clf_model, seg_model, nlp_model, eval_report = load_ml_models()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


# Sidebar Navigation & Filters
st.sidebar.markdown("### ⚡ FoodPulse Control Panel")
st.sidebar.caption("AI Delivery Intelligence & Operational Forecasting (2025 – 2026)")

# Global Hub Filter
all_hubs = ["All Urban Hubs"] + list(df_hubs["hub_name"].unique())
selected_hub = st.sidebar.selectbox("Filter by Urban Hub", all_hubs)

# Date & Weather filter
weather_options = ["All Conditions"] + sorted(list(df_orders["weather"].unique()))
selected_weather = st.sidebar.selectbox("Filter by Weather", weather_options)

# Filter dataframe
filtered_orders = df_orders.copy()
if selected_hub != "All Urban Hubs":
    hub_id = df_hubs[df_hubs["hub_name"] == selected_hub]["hub_id"].iloc[0]
    filtered_orders = filtered_orders[filtered_orders["hub_id"] == hub_id]

if selected_weather != "All Conditions":
    filtered_orders = filtered_orders[filtered_orders["weather"] == selected_weather]


# Header Banner
st.markdown('<div class="main-header">⚡ FoodPulse: AI Delivery Intelligence & Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Hub Turnaround Diagnostics • Machine Learning ETA Forecasting • Customer Archetype Segmentation • NLP Complaint Intelligence</div>', unsafe_allow_html=True)

# Top KPI Metric Strip
col1, col2, col3, col4, col5 = st.columns(5)
total_orders_count = len(filtered_orders)
on_time_pct = round(100.0 * (filtered_orders["is_delayed"] == 0).sum() / max(1, total_orders_count), 1)
avg_delivery_time = round(filtered_orders["actual_delivery_time_min"].mean(), 1)
avg_kitchen_prep = round(filtered_orders["kitchen_prep_time_min"].mean(), 1)
severe_breach_pct = round(100.0 * (filtered_orders["delay_severity"] == "severe_delay").sum() / max(1, total_orders_count), 1)

with col1:
    st.metric("Total Monitored Orders", f"{total_orders_count:,}", "2025–2026")
with col2:
    st.metric("On-Time SLA Rate", f"{on_time_pct}%", f"{round(on_time_pct - 80.0, 1)}% vs Target")
with col3:
    st.metric("Avg End-to-End Delivery", f"{avg_delivery_time} min", f"{round(avg_delivery_time - 35.0, 1)} min delta")
with col4:
    st.metric("Avg Kitchen Turnaround", f"{avg_kitchen_prep} min", "Kitchen Prep")
with col5:
    st.metric("Severe SLA Breach Rate", f"{severe_breach_pct}%", f"{round(15.0 - severe_breach_pct, 1)}% headroom", delta_color="inverse")

st.markdown("---")


# Tab Navigation
tabs = st.tabs([
    "🌐 Hub Bottlenecks & KPIs",
    "⏱️ Real-Time ETA & Risk Simulator",
    "🍳 Restaurant Turnaround Diagnostics",
    "👥 Customer Segmentation (K-Means)",
    "💬 NLP Review & Complaint Intelligence",
    "🧪 Model Benchmarks & Governance"
])


# ==============================================================================
# TAB 1: Hub Bottlenecks & Executive KPIs
# ==============================================================================
with tabs[0]:
    st.subheader("Urban Hub Operations & Delivery Bottlenecks")
    
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("**Hourly Order Volume & Delay Propensity**")
        hourly_summary = filtered_orders.groupby("hour_of_day").agg(
            order_volume=("order_id", "count"),
            delay_rate=("is_delayed", "mean")
        ).reset_index()
        hourly_summary["delay_rate_pct"] = (hourly_summary["delay_rate"] * 100).round(1)

        # Altair Dual Axis Chart
        base = alt.Chart(hourly_summary).encode(x=alt.X("hour_of_day:O", title="Hour of Day (00:00 - 23:00)"))
        bars = base.mark_bar(color="#3b82f6", opacity=0.8).encode(
            y=alt.Y("order_volume:Q", title="Order Volume"),
            tooltip=["hour_of_day", "order_volume", "delay_rate_pct"]
        )
        line = base.mark_line(color="#ef4444", strokeWidth=3).encode(
            y=alt.Y("delay_rate_pct:Q", title="Delay Rate (%)"),
            tooltip=["hour_of_day", "delay_rate_pct"]
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)

    with col_chart2:
        st.markdown("**Turnaround Decomposition (Kitchen Prep vs Courier Transit)**")
        turnaround_by_weather = filtered_orders.groupby("weather").agg(
            avg_prep=("kitchen_prep_time_min", "mean"),
            avg_transit=("transit_time_min", "mean")
        ).reset_index()
        
        melted_turnaround = pd.melt(
            turnaround_by_weather,
            id_vars=["weather"],
            value_vars=["avg_prep", "avg_transit"],
            var_name="stage",
            value_name="minutes"
        )
        melted_turnaround["stage"] = melted_turnaround["stage"].map({
            "avg_prep": "Kitchen Prep (min)",
            "avg_transit": "Courier Transit (min)"
        })

        chart_turnaround = alt.Chart(melted_turnaround).mark_bar().encode(
            x=alt.X("weather:N", title="Weather Condition"),
            y=alt.Y("minutes:Q", title="Duration (Minutes)"),
            color=alt.Color("stage:N", scale=alt.Scale(range=["#f59e0b", "#06b6d4"])),
            tooltip=["weather", "stage", alt.Tooltip("minutes:Q", format=".1f")]
        )
        st.altair_chart(chart_turnaround, use_container_width=True)

    st.markdown("#### 🏢 Urban Hub Comparative SLA Scorecard")
    hub_card_df = query_df(ANALYTICAL_QUERIES["hub_sla_performance"])
    st.dataframe(
        hub_card_df,
        column_config={
            "hub_name": "Urban Hub",
            "city": "City",
            "total_orders": "Total Orders",
            "avg_kitchen_turnaround_min": st.column_config.NumberColumn("Avg Kitchen (min)", format="%.1f min"),
            "avg_transit_min": st.column_config.NumberColumn("Avg Transit (min)", format="%.1f min"),
            "avg_total_delivery_min": st.column_config.NumberColumn("Total Delivery (min)", format="%.1f min"),
            "delay_rate_pct": st.column_config.ProgressColumn("Delay Rate", format="%.1f%%", min_value=0, max_value=100),
            "severe_delay_pct": st.column_config.ProgressColumn("Severe SLA Breaches", format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True,
        use_container_width=True
    )

    # SQL Analytics Workbench
    with st.expander("🔍 Interactive SQL Operations Workbench (Live SQLite Explorer)", expanded=False):
        st.caption("Execute custom SQL or run pre-configured data engineering analytical queries on the relational database.")
        selected_preset = st.selectbox(
            "Select Pre-configured SQL Query",
            ["Custom Query"] + list(ANALYTICAL_QUERIES.keys())
        )
        
        if selected_preset != "Custom Query":
            default_sql = ANALYTICAL_QUERIES[selected_preset]
        else:
            default_sql = "SELECT order_id, hub_id, restaurant_id, distance_km, actual_delivery_time_min, delay_severity FROM orders LIMIT 15;"

        sql_input = st.text_area("SQL Statement", value=default_sql, height=140)
        if st.button("Execute SQL Query", type="primary"):
            try:
                res_df = query_df(sql_input)
                st.success(f"Query returned {len(res_df)} rows.")
                st.dataframe(res_df, use_container_width=True)
            except Exception as sql_err:
                st.error(f"SQL Execution Error: {sql_err}")


# ==============================================================================
# TAB 2: Real-Time ETA & Delay Risk Simulator
# ==============================================================================
with tabs[1]:
    st.subheader("Real-Time Delivery ETA & SLA Risk Simulator")
    st.caption("Simulate incoming orders to forecast duration and identify operational delay risks before dispatch.")

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("##### 📦 Order & Restaurant Parameters")
        sim_hub = st.selectbox("Urban Hub", df_hubs["hub_name"].tolist(), index=0)
        selected_hub_row = df_hubs[df_hubs["hub_name"] == sim_hub].iloc[0]

        hub_restaurants = df_restaurants[df_restaurants["hub_id"] == selected_hub_row["hub_id"]]
        sim_rest_name = st.selectbox("Restaurant", hub_restaurants["name"].tolist(), index=0)
        selected_rest_row = hub_restaurants[hub_restaurants["name"] == sim_rest_name].iloc[0]

        st.caption(f"Cuisine: **{selected_rest_row['cuisine']}** | Base Prep Time: **{selected_rest_row['prep_time_base']:.1f} min** | Kitchen Rating: **⭐ {selected_rest_row['rating']}**")

        sim_items = st.slider("Item Count (Basket Size)", min_value=1, max_value=12, value=3)
        sim_val = st.number_input("Order Value ($ USD)", min_value=8.0, max_value=350.0, value=float(sim_items * 16.5))

        st.markdown("##### 🚴 Logistics & Environmental Conditions")
        sim_distance = st.slider("Delivery Route Distance (km)", min_value=0.5, max_value=15.0, value=3.8, step=0.1)
        sim_vehicle = st.selectbox("Courier Transport Mode", ["E-Scooter", "Bicycle", "Motorcycle", "Electric Car"], index=0)
        
        sim_weather = st.selectbox("Weather Condition", list(WEATHER_SEVERITY_MAP.keys()), index=0)
        sim_hour = st.slider("Order Placement Time (Hour of Day)", min_value=0, max_value=23, value=19)
        sim_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
        day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(sim_day)

    with sim_col2:
        st.markdown("##### 🤖 Dual-Model Predictive AI Inference")

        # Map vehicle speed factor
        vehicle_speed_factors = {"E-Scooter": 1.25, "Bicycle": 1.0, "Motorcycle": 1.45, "Electric Car": 1.30}
        speed_f = vehicle_speed_factors.get(sim_vehicle, 1.25)

        # Build inference input
        input_dict = {
            "hour_of_day": sim_hour,
            "day_of_week": day_idx,
            "distance_km": sim_distance,
            "hub_density_score": selected_hub_row["avg_density_score"],
            "hub_traffic_base": selected_hub_row["traffic_congestion_base"],
            "prep_time_base": selected_rest_row["prep_time_base"],
            "kitchen_capacity": selected_rest_row["kitchen_capacity"],
            "rest_rating": selected_rest_row["rating"],
            "item_count": sim_items,
            "order_value": sim_val,
            "courier_speed_factor": speed_f,
            "courier_rating": 4.7,
            "vehicle_type": sim_vehicle,
            "weather": sim_weather,
            "temperature_c": 16.0
        }

        # Transform using fitted feature engineer
        X_infer = fe.transform_single(input_dict)

        # Model 1: Regression ETA
        eta_breakdown = reg_model.predict_single_with_breakdown(X_infer)
        predicted_eta = eta_breakdown["predicted_eta_minutes"]

        # Model 2: Classification Risk
        risk_analysis = clf_model.predict_risk_single(X_infer)

        # Render Live Prediction Box
        st.markdown(f"""
        <div style="background-color: #111827; border: 2px solid {risk_analysis['color_hex']}; border-radius: 12px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1rem; color: #94a3b8; font-weight: 600;">PREDICTED DELIVERY DURATION</span>
                <span style="background-color: {risk_analysis['color_hex']}; color: #ffffff; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                    {risk_analysis['risk_tier']}
                </span>
            </div>
            <div style="font-size: 3rem; font-weight: 800; color: #ffffff; margin: 10px 0;">
                {predicted_eta:.1f} <span style="font-size: 1.5rem; color: #9ca3af; font-weight: 500;">minutes</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.95rem;">
                Estimated SLA Breach Risk: <strong style="color: {risk_analysis['color_hex']};">{risk_analysis['delay_risk_percentage']}%</strong> 
                (Severe Breach Risk: <strong>{risk_analysis['severe_delay_percentage']}%</strong>)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###### ⏱️ End-to-End Turnaround Stage Decomposition")
        stage_df = pd.DataFrame({
            "Turnaround Stage": ["1. Kitchen Prep", "2. Courier Transit", "3. Handover Buffer"],
            "Estimated Minutes": [
                eta_breakdown["estimated_kitchen_prep_min"],
                eta_breakdown["estimated_transit_min"],
                eta_breakdown["estimated_handover_buffer_min"]
            ]
        })
        chart_stages = alt.Chart(stage_df).mark_bar(cornerRadius=4).encode(
            x=alt.X("Estimated Minutes:Q", title="Minutes"),
            y=alt.Y("Turnaround Stage:N", sort=None),
            color=alt.Color("Turnaround Stage:N", scale=alt.Scale(range=["#f59e0b", "#3b82f6", "#10b981"])),
            tooltip=["Turnaround Stage", "Estimated Minutes"]
        )
        st.altair_chart(chart_stages, use_container_width=True)

        # Explainability & Mitigations
        st.markdown("###### 🔍 Primary Delay Drivers Detected")
        for factor in eta_breakdown["primary_delay_drivers"]:
            st.markdown(f"- ⚠️ **{factor}**")

        st.markdown(f"""
        <div style="background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 12px; border-radius: 6px; margin-top: 15px;">
            <strong style="color: #38bdf8;">🛡️ Recommended Operations Action:</strong><br>
            <span style="color: #e2e8f0; font-size: 0.95rem;">{risk_analysis['recommended_mitigation']}</span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: Restaurant Turnaround & Kitchen Diagnostics
# ==============================================================================
with tabs[2]:
    st.subheader("Restaurant Turnaround & Kitchen Prep Diagnostics")
    st.caption("Identify chronic preparation bottlenecks across restaurant partners, cuisines, and capacity constraints.")

    # Cuisine Level Breakdown
    cuisine_turnaround = df_orders.groupby("restaurant_id").agg(
        prep_mean=("kitchen_prep_time_min", "mean"),
        prep_max=("kitchen_prep_time_min", "max"),
        orders_count=("order_id", "count")
    ).reset_index()

    cuisine_merged = cuisine_turnaround.merge(df_restaurants, on="restaurant_id")
    cuisine_summary = cuisine_merged.groupby("cuisine").agg(
        avg_prep_min=("prep_mean", "mean"),
        max_prep_min=("prep_max", "max"),
        total_orders=("orders_count", "sum"),
        restaurant_count=("restaurant_id", "count")
    ).reset_index().sort_values("avg_prep_min", ascending=False)

    col_diag1, col_diag2 = st.columns([1.2, 1])

    with col_diag1:
        st.markdown("**Average Kitchen Prep Time by Cuisine**")
        chart_cuisine = alt.Chart(cuisine_summary).mark_bar().encode(
            x=alt.X("avg_prep_min:Q", title="Average Kitchen Prep Time (min)"),
            y=alt.Y("cuisine:N", sort="-x", title="Cuisine"),
            color=alt.Color("avg_prep_min:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
            tooltip=["cuisine", alt.Tooltip("avg_prep_min:Q", format=".1f"), "total_orders"]
        )
        st.altair_chart(chart_cuisine, use_container_width=True)

    with col_diag2:
        st.markdown("**Kitchen Capacity vs Actual Prep Time**")
        scatter_chart = alt.Chart(cuisine_merged).mark_circle(size=80, opacity=0.8).encode(
            x=alt.X("kitchen_capacity:Q", title="Kitchen Hourly Capacity"),
            y=alt.Y("prep_mean:Q", title="Mean Kitchen Prep (min)"),
            color=alt.Color("cuisine:N"),
            tooltip=["name", "cuisine", "kitchen_capacity", alt.Tooltip("prep_mean:Q", format=".1f")]
        )
        st.altair_chart(scatter_chart, use_container_width=True)

    st.markdown("#### 🚨 Top Chronic Kitchen Bottleneck Restaurants (>20 min prep)")
    bottlenecks_df = query_df(ANALYTICAL_QUERIES["restaurant_turnaround_diagnostics"])
    st.dataframe(
        bottlenecks_df,
        column_config={
            "restaurant_name": "Restaurant",
            "cuisine": "Cuisine",
            "hub_name": "Hub",
            "restaurant_rating": "Rating",
            "total_orders": "Orders",
            "avg_prep_time": st.column_config.NumberColumn("Avg Prep (min)", format="%.1f min"),
            "max_prep_time": st.column_config.NumberColumn("Max Prep (min)", format="%.1f min"),
            "slow_kitchen_rate_pct": st.column_config.ProgressColumn("Slow Kitchen Rate (>25m)", format="%.1f%%", min_value=0, max_value=100)
        },
        hide_index=True,
        use_container_width=True
    )


# ==============================================================================
# TAB 4: Customer Segmentation (K-Means Clustering)
# ==============================================================================
with tabs[3]:
    st.subheader("Customer Behavior Segmentation (K-Means Clustering & PCA)")
    st.caption("AI-driven clustering on order frequency, spend elasticity, time-of-day preferences, and SLA delay tolerance.")

    if "pca_x" in df_customers.columns:
        col_seg_plot, col_seg_cards = st.columns([1.1, 1])

        with col_seg_plot:
            st.markdown("**Customer Behavioral Landscape (2D PCA Projection)**")
            pca_chart = alt.Chart(df_customers).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("pca_x:Q", title="Principal Component 1 (Spend & Frequency)"),
                y=alt.Y("pca_y:Q", title="Principal Component 2 (Temporal & Sensitivity)"),
                color=alt.Color("persona:N", title="Customer Persona", scale=alt.Scale(scheme="category10")),
                tooltip=["customer_id", "persona", "avg_order_value", "total_orders", "avg_rating_given"]
            ).interactive()
            st.altair_chart(pca_chart, use_container_width=True)

            # Elbow and Silhouette evaluation metrics
            if eval_report and "customer_segmentation" in eval_report:
                seg_eval = eval_report["customer_segmentation"]
                sil_df = pd.DataFrame([
                    {"k": int(k), "Silhouette Score": v} for k, v in seg_eval.get("silhouette_scores", {}).items()
                ])
                if not sil_df.empty:
                    sil_chart = alt.Chart(sil_df).mark_line(point=True, color="#10b981").encode(
                        x=alt.X("k:O", title="Number of Clusters (K)"),
                        y=alt.Y("Silhouette Score:Q", title="Silhouette Score"),
                        tooltip=["k", "Silhouette Score"]
                    ).properties(height=180, title="Optimal Cluster Validation (Silhouette Analysis)")
                    st.altair_chart(sil_chart, use_container_width=True)

        with col_seg_cards:
            st.markdown("**Behavioral Persona Archetypes & Operational Playbooks**")
            if seg_model and hasattr(seg_model, "cluster_summaries"):
                for c_id, summ in seg_model.cluster_summaries.items():
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong style="color: #fbbf24; font-size: 1.05rem;">{summ['persona_name']}</strong>
                                <span style="color: #94a3b8; font-size: 0.85rem;">{summ['size']} users ({summ['pct_of_base']}%)</span>
                            </div>
                            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                                • Avg Basket Spend: <strong>${summ['avg_order_value']:.2f}</strong><br>
                                • Order Frequency: <strong>{summ['avg_orders']} orders</strong> | Avg Rating: <strong>⭐ {summ['avg_rating']}</strong><br>
                                • Peak Habits: <strong>{summ['dinner_pct']}% Dinner</strong>, <strong>{summ['lunch_pct']}% Lunch</strong>, <strong>{summ['weekend_pct']}% Weekend</strong>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.82rem; color: #38bdf8;">
                                🎯 <strong>Strategy:</strong> {summ['recommended_strategy']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("Customer segmentation data is compiling. Please refresh in a moment.")


# ==============================================================================
# TAB 5: NLP Customer Review & Complaint Intelligence
# ==============================================================================
with tabs[4]:
    st.subheader("NLP Customer Review & Service Complaint Intelligence")
    st.caption("Sentiment polarity analysis and automated service failure extraction from customer feedback.")

    nlp_col1, nlp_col2 = st.columns([1, 1])

    with nlp_col1:
        st.markdown("**Recurring Service Failure Breakdown**")
        complaints_df = query_df(ANALYTICAL_QUERIES["customer_review_complaints_breakdown"])
        
        # Aggregate by complaint category
        cat_agg = complaints_df.groupby("complaint_category")["review_count"].sum().reset_index()
        cat_agg = cat_agg.sort_values("review_count", ascending=False)

        chart_complaints = alt.Chart(cat_agg).mark_bar(cornerRadius=4).encode(
            x=alt.X("review_count:Q", title="Number of Reviews"),
            y=alt.Y("complaint_category:N", sort="-x", title="Complaint Category"),
            color=alt.Color("complaint_category:N", legend=None),
            tooltip=["complaint_category", "review_count"]
        )
        st.altair_chart(chart_complaints, use_container_width=True)

    with nlp_col2:
        st.markdown("**Top Negative Complaint Phrases (N-Grams)**")
        if nlp_model and hasattr(nlp_model, "top_negative_ngrams"):
            neg_ngrams = pd.DataFrame(nlp_model.top_negative_ngrams, columns=["Keyword / N-Gram", "Frequency"])
            st.dataframe(
                neg_ngrams,
                column_config={
                    "Frequency": st.column_config.ProgressColumn("Frequency Count", format="%d", min_value=0, max_value=int(neg_ngrams["Frequency"].max() * 1.1))
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.dataframe(complaints_df, use_container_width=True)

    st.markdown("#### 🧪 Real-Time NLP Review Tester")
    st.caption("Paste any customer review below to test NLP sentiment polarity extraction and complaint tagger:")
    
    sample_text = st.text_area(
        "Customer Review Text",
        value="The courier took 75 minutes to arrive and the pizza was freezing cold upon arrival. Cheese was completely solidified."
    )
    if st.button("Analyze Review Sentiment", type="primary"):
        if nlp_model:
            nlp_result = nlp_model.analyze_single_review(sample_text)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Detected Sentiment", nlp_result["sentiment_label"])
            with res_col2:
                st.metric("Polarity Score", f"{nlp_result['polarity_score']}")
            with res_col3:
                st.metric("Extracted Complaint", ", ".join(nlp_result["detected_complaints"]))

            if nlp_result["key_negative_signals"]:
                st.markdown(f"**Negative Triggers Detected:** `{', '.join(nlp_result['key_negative_signals'])}`")
            if nlp_result["key_positive_signals"]:
                st.markdown(f"**Positive Signals Detected:** `{', '.join(nlp_result['key_positive_signals'])}`")


# ==============================================================================
# TAB 6: Model Governance & Benchmarks
# ==============================================================================
with tabs[5]:
    st.subheader("Model Governance, Benchmarks & Explainability")
    st.caption("Comprehensive evaluation of predictive pipelines against production SLAs and holdout test sets.")

    if eval_report:
        col_bench1, col_bench2 = st.columns(2)

        with col_bench1:
            st.markdown("##### 📈 Delivery Duration Regressors (ETA Benchmark)")
            reg_bench = eval_report.get("eta_regressor_metrics", {}).get("benchmark_comparison", {})
            if reg_bench:
                bench_reg_df = pd.DataFrame.from_dict(reg_bench, orient="index")
                bench_reg_df.index.name = "Candidate Algorithm"
                st.dataframe(
                    bench_reg_df.style.highlight_min(subset=["mae", "rmse"], color="#064e3b")
                                      .highlight_max(subset=["r2"], color="#064e3b"),
                    use_container_width=True
                )

        with col_bench2:
            st.markdown("##### 🚨 SLA Delay Risk Classifiers Benchmark")
            clf_bench = eval_report.get("delay_classifier_metrics", {}).get("benchmark_comparison", {})
            if clf_bench:
                bench_clf_df = pd.DataFrame.from_dict(clf_bench, orient="index")
                bench_clf_df.index.name = "Candidate Algorithm"
                st.dataframe(
                    bench_clf_df.style.highlight_max(subset=["precision", "recall", "f1_score"], color="#064e3b"),
                    use_container_width=True
                )

        st.markdown("##### 🔑 Global Feature Importance (ETA Prediction)")
        if reg_model and hasattr(reg_model, "feature_importances_"):
            top_feats = list(reg_model.feature_importances_.items())[:12]
            feat_df = pd.DataFrame(top_feats, columns=["Feature", "Relative Importance"])

            chart_feat = alt.Chart(feat_df).mark_bar(color="#ff8c00").encode(
                x=alt.X("Relative Importance:Q", title="Importance Weight"),
                y=alt.Y("Feature:N", sort="-x", title="Feature"),
                tooltip=["Feature", alt.Tooltip("Relative Importance:Q", format=".4f")]
            )
            st.altair_chart(chart_feat, use_container_width=True)

        # Classification Report Table
        st.markdown("##### 📋 Detailed Multiclass Classification Breakdown")
        clf_rep = eval_report.get("delay_classifier_metrics", {}).get("classification_report", {})
        if clf_rep:
            clf_rep_df = pd.DataFrame(clf_rep).transpose()
            st.dataframe(clf_rep_df.round(3), use_container_width=True)

    else:
        st.info("Evaluation report is being compiled. Please check back shortly.")


# Footer
st.markdown("---")
st.caption("FoodPulse: AI Delivery Intelligence & Prediction System (2025–2026) • Engineered with Python, Scikit-Learn, Pandas, Streamlit, SQL & Altair.")
