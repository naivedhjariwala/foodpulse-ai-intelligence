# ⚡ FoodPulse: AI Delivery Intelligence & Prediction System (2025 – 2026)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/SQL-SQLite%203-003B57.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end predictive intelligence pipeline and operational decision-support system architected to identify urban delivery bottlenecks, analyze restaurant turnaround times (Kitchen Prep Time - KPT), forecast SLA delivery delays with high precision, segment customer behavioral archetypes using K-Means clustering, and extract recurring service failure complaints through Natural Language Processing (NLP).

---

## 🌟 Key System Capabilities

1. **End-to-End Predictive Pipeline**:
   - Automated ingestion and synchronization across relational databases (`urban_hubs`, `restaurants`, `delivery_partners`, `orders`, `reviews`).
   - Feature engineering engine extracting cyclical temporal signals, geospatial distance/density matrices, kitchen capacity stress factors, and courier transit dynamics.

2. **Delivery Duration (ETA) Regression**:
   - Trained and benchmarked **Gradient Boosting**, **Random Forest**, and **Ridge** Regressors.
   - Achieved **$R^2 = 0.9418$** and **MAE = 2.88 minutes**, decomposing delivery into kitchen prep, road transit, and courier handover buffers.

3. **SLA Delay Risk Classification**:
   - Multi-class classification forecasting delay severity: `On-Time`, `Mild Delay (1-14 min)`, and `Severe SLA Breach (>15 min)`.
   - Generates calibrated probability scores and triggers prescriptive operational mitigations (e.g., dispatch motorized couriers, throttle restaurant order intake).

4. **Customer Behavioral Profiling (K-Means Clustering & PCA)**:
   - RFM & behavioral segmentation evaluated using Silhouette scores ($k=2 \dots 7$) and the Elbow method.
   - Identified 4 distinct customer archetypes:
     - 🍱 **Gourmet Dinner Connoisseurs** (High basket value, evening peaks, low delay tolerance)
     - ⚡ **Fast-Paced Office Lunchers** (Strict 25-min SLA sensitivity, weekday lunch peaks)
     - 🍕 **Weekend Family & Group Feasters** (High item counts, weekend group orders)
     - 🏷️ **Budget Everyday Frequent Orderers** (High cadence, promotion-sensitive)

5. **NLP Review Sentiment & Complaint Intelligence**:
   - Unigram/Bigram TF-IDF vectorization and rule-based service failure categorizer.
   - Automatically segments feedback into 5 recurring complaint domains:
     - ⏳ *Late Delivery / Transit Delays*
     - ❄️ *Cold Food / Thermal Transport Failure*
     - 📦 *Missing Items / Order Incomplete*
     - 🚴 *Courier Conduct / Navigation Friction*
     - 🥤 *Damaged Packaging & Liquid Spills*

6. **Interactive Streamlit Operations Control Center**:
   - 6 production modules: Executive KPIs, Real-Time ETA Sandbox, Turnaround Diagnostics, Customer Clusters, NLP Review Tester, and Model Governance Benchmarks.
   - Embedded SQL Workbench for ad-hoc operational data warehouse querying.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data_Layer ["Data & Storage Layer (SQLite / CSV)"]
        DB[(foodpulse.db)]
        T1[urban_hubs]
        T2[restaurants]
        T3[delivery_partners]
        T4[orders: 12,500 records]
        T5[reviews: 4,350+ records]
        DB --> T1 & T2 & T3 & T4 & T5
    end

    subgraph Feature_Pipeline ["Feature Engineering Engine"]
        F1[Timestamp Cyclical Sin/Cos, Rush Hour Flags]
        F2[Geospatial Haversine, Hub Density & Traffic Multiplier]
        F3[Kitchen Capacity Stress, Items Per Capacity]
        F4[Courier Vehicle Profiles & Weather Penalty Index]
        T4 --> F1 & F2 & F3 & F4
    end

    subgraph Models ["Machine Learning & NLP Suite"]
        M1[ETA Regressor<br/>Gradient Boosting<br/>MAE: 2.88m, R2: 0.9418]
        M2[Delay Risk Classifier<br/>Gradient Boosting<br/>Precision: 78.5%, F1: 78.2%]
        M3[Customer Segmentation<br/>K-Means + PCA 2D/3D<br/>Silhouette Validation]
        M4[NLP Sentiment Engine<br/>TF-IDF + Complaint Categorizer<br/>Polarity & Failure Extraction]
        F1 & F2 & F3 & F4 --> M1 & M2
        T4 & T5 --> M3
        T5 --> M4
    end

    subgraph UI ["Streamlit Ops Control Application"]
        P1[🌐 Executive Ops Center & Bottlenecks]
        P2[⏱️ Real-Time Dual-Model Simulator]
        P3[🍳 Kitchen Turnaround Diagnostics]
        P4[👥 Customer Segmentation & PCA]
        P5[💬 NLP Review & Complaint Intelligence]
        P6[🧪 Model Governance & Benchmarks]
        M1 & M2 & M3 & M4 --> UI
        UI --> P1 & P2 & P3 & P4 & P5 & P6
    end
```

---

## 📊 Benchmark Results

### 1. Delivery Duration Regressor (ETA Forecast)

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Regressor (Primary)** | **2.884 min** | **3.665 min** | **0.9418** |
| Random Forest Regressor | 3.120 min | 3.980 min | 0.9312 |
| Ridge Regression (Baseline) | 4.450 min | 5.620 min | 0.8640 |

### 2. SLA Delay Risk Classifier

| Algorithm | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Multi-Class ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Primary)** | **0.7852** | **0.7952** | **0.7821** | **0.9120** |
| Random Forest Classifier | 0.7710 | 0.7820 | 0.7705 | 0.8990 |
| Logistic Regression (Baseline) | 0.6940 | 0.6880 | 0.6890 | 0.8120 |

---

## 📂 Repository Structure

```text
FoodPulse AI Delivery Intelligence & Prediction System/
│
├── app/
│   └── main.py                     # Interactive Streamlit application (6 operational tabs)
│
├── data/
│   ├── foodpulse.db                # SQLite operational database with relational tables & indexes
│   ├── orders.csv                  # Exported 12,500 order records with engineered targets
│   ├── reviews.csv                 # 4,359 customer reviews with sentiment tags
│   ├── restaurants.csv             # 35 restaurants across 5 urban hubs
│   ├── urban_hubs.csv              # Multi-hub geographic & density reference
│   ├── delivery_partners.csv       # Courier logistics profiles & vehicle types
│   ├── customers.csv               # Customer records
│   └── customer_segments.csv       # Segmented customer profiles with PCA coordinates
│
├── models/
│   ├── eta_regressor.joblib        # Serialized Gradient Boosting delivery duration model
│   ├── delay_classifier.joblib     # Serialized multi-class SLA delay risk model
│   ├── customer_segmentation.joblib# Serialized K-Means clustering & PCA transformers
│   ├── nlp_sentiment.joblib        # Serialized TF-IDF vectorizer & complaint categorizer
│   ├── feature_engineer.joblib     # Serialized feature engineering metadata pipeline
│   └── evaluation_report.json      # Complete benchmark evaluation report
│
├── src/
│   ├── __init__.py
│   ├── db.py                       # SQLite connection, schema migrations & enterprise queries
│   ├── data_generator.py           # Multi-hub synthetic data engine with realistic dynamics
│   ├── features.py                 # Feature transformation pipeline (cyclical, geo, turnaround)
│   ├── pipeline.py                 # End-to-end training, evaluation & serialization orchestration
│   └── models/
│       ├── __init__.py
│       ├── eta_regressor.py        # Continuous duration regression & breakdown
│       ├── delay_classifier.py     # SLA risk classifier & prescriptive mitigations
│       ├── customer_segmentation.py# K-Means clustering, Elbow/Silhouette & PCA
│       └── nlp_sentiment.py        # TF-IDF N-Gram sentiment & complaint extractor
│
├── tests/
│   └── test_pipeline.py            # Automated Pytest suite covering all modules
│
├── requirements.txt                # Python package dependencies
└── README.md                       # System documentation & technical overview
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/foodpulse-ai-intelligence.git
cd "FoodPulse AI Delivery Intelligence & Prediction System"

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Predictive Models

To re-generate the multi-hub operational dataset and train all machine learning and NLP pipelines:

```bash
# Generate synthetic dataset into SQLite database and CSV files
python -m src.data_generator

# Execute the complete end-to-end training and evaluation pipeline
python -m src.pipeline
```

### 3. Run Automated Tests

Execute the unit and integration test suite with `pytest`:

```bash
python -m pytest -v
```

### 4. Launch the Streamlit Interactive Application

Start the web application:

```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501` to explore the live dashboard.

---

## 🖥️ Streamlit Application Modules

1. **🌐 Executive Ops Center & Hub Bottlenecks**:
   - Live KPI counter (Total Orders, On-Time SLA %, Avg Turnaround, Severe Delays).
   - Dual-axis hourly volume vs delay rate charts.
   - Interactive SQL Workbench to execute custom queries against `foodpulse.db`.

2. **⏱️ Real-Time Delivery ETA & SLA Risk Simulator**:
   - Customize restaurant, cuisine, basket items, order value, courier mode, distance, weather, and time of day.
   - View predicted ETA, kitchen/transit component breakdown, delay risk gauge, primary delay drivers, and prescriptive operational actions.

3. **🍳 Restaurant Turnaround Diagnostics**:
   - Kitchen preparation time breakdown across cuisines.
   - Capacity stress vs turnaround correlation.
   - Chronic slow kitchen partner identification table.

4. **👥 Customer Behavior Segmentation (K-Means)**:
   - Interactive 2D PCA customer scatter projection.
   - Behavioral persona cards with targeted marketing and retention strategies.
   - Silhouette analysis validation curves.

5. **💬 NLP Review & Complaint Intelligence**:
   - Recurring service failure diagnostics (Late Delivery, Cold Food, Missing Items, Rude Courier, Packaging Spills).
   - Top negative N-gram frequency charts.
   - Live interactive NLP review tester.

6. **🧪 Model Governance & Benchmarks**:
   - Cross-model benchmark comparison tables (MAE, RMSE, $R^2$, Precision, Recall, F1).
   - Global feature importance charts for operational explainability.
   - Detailed multiclass classification reports.

---

## 🛠️ Technologies Used

- **Core Programming**: Python 3.13
- **Machine Learning**: Scikit-Learn (Gradient Boosting, Random Forest, Ridge, Logistic Regression, K-Means, PCA)
- **Data Engineering & Manipulation**: Pandas, NumPy
- **NLP**: Scikit-Learn (TF-IDF Vectorizer, CountVectorizer, N-Gram parsing), Regex
- **Database**: SQLite 3 with optimized B-Tree indexing and analytical SQL queries
- **Data Visualization & Dashboard**: Streamlit, Altair, Matplotlib
- **Model Serialization & Testing**: Joblib, Pytest

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

