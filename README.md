# Phense — The Sense Before the Storm

> **AI-Powered Composite Flood Risk Classifier for Nigerian Communities**

[![3MTT Cohort](https://img.shields.io/badge/3MTT-NextGen%20Cohort-blue.svg)](https://3mtt.nitda.gov.ng/)
[![Track](https://img.shields.io/badge/Track-AI%20%26%20Machine%20Learning-orange.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-81.25%25-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

**Fellow:** Adeniji Yusuf | `FE/24/5369659950`  
**Track:** 3MTT NextGen Cohort — AI & Machine Learning  
**Brief:** AI-19 — Flood Risk Classifier | Ogun / Grazac Technologies Limited  

---

## Executive Summary

**Phense** is an end-to-end, AI-powered composite flood risk classification system tailored for Nigerian riverine and urban communities. By combining 5 core environmental hazard factors—**Elevation, Distance to Rivers, Average Monthly Rainfall, Soil Type, and Drainage Infrastructure**—Phense calculates localized risk levels (**Low, Medium, High, Critical**) accompanied by dynamic seasonal adjustments, primary risk driver attribution, and plain-language emergency advisories.

Designed for non-technical users across all 36 Nigerian states and the FCT, Phense offers **zero-friction 1-Tap GPS Geolocation**: clicking a single button auto-fills satellite elevation (SRTM 30m DEM), nearest river proximity, LGA boundary, and NiMet seasonal rainfall baselines, requiring only two simple visual taps for local soil and drainage conditions.

---

## Key Achievements & Technical Milestones

Phense delivers a complete production-grade machine learning application:

- **4-Pillar Real-World Data Pipeline**: Anchored strictly to real spatial coordinates, NASA SRTM 30m Digital Elevation Models, a vector database of 12 major Nigerian rivers (`data/reference/nigeria_rivers.json`), 774 LGA centroids (`data/reference/nigeria_lgas.json`), and NiMet climatological station records.
- **Optimized Machine Learning Engine**: Built with `scikit-learn` using a tuned `RandomForestClassifier` with 5-fold Stratified K-Fold CV & GridSearchCV, achieving **81.25% Test Accuracy** and **81.20% Weighted F1-Score**.
- **Model Explainability & Risk Attribution**: Computes normalized feature contribution scores for every prediction, identifying the single primary risk driver (e.g., low elevation vs. river proximity) to provide targeted advisories.
- **Production FastAPI Service**: High-performance asynchronous REST backend serving prediction models, live SRTM elevation lookups, spatial river proximity calculations, reverse-geocoding, and seasonal rainfall baselines.
- **Modern Multi-Page Web Interface**: Responsive, glassmorphic UI across 5 dedicated views (`/`, `/assess`, `/results`, `/methodology`, `/about`) featuring 1-tap GPS auto-fill and real-time risk gauges.
- **Cloud & Container Ready**: Dockerized (`Dockerfile`) and configured for zero-downtime deployment on Render.com or cloud platforms.

---

## Model Performance & Evaluation Metrics

Evaluated on a held-out 20% test dataset (272 real-grounded samples), Phense achieved balanced performance across all flood risk severity tiers:

### Global Metrics

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **81.25%** |
| **Weighted F1-Score** | **81.20%** |
| **Best CV Accuracy (GridSearch)** | **81.23%** |
| **Training Samples** | 1,088 |
| **Testing Samples** | 272 |

### Class-Wise Evaluation Report

| Risk Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 75.51% | 80.43% | **77.89%** | 46 |
| **Medium** | 81.52% | 75.76% | **78.53%** | 99 |
| **High** | 84.26% | 87.50% | **85.85%** | 104 |
| **Critical** | 78.26% | 78.26% | **78.26%** | 23 |

### Quantified Feature Importance

The model's decisions correlate directly with established physical hydrological principles in Nigeria:

| Rank | Feature | Importance | Physical Role |
| :---: | :--- | :---: | :--- |
| **1** | **Elevation (`elevation_m`)** | **33.90%** | **Top Driver**: Low-lying floodplains retain water longest |
| **2** | **River Proximity (`river_distance_km`)** | **23.64%** | Direct exposure to riverbank overflow and backflow |
| **3** | **Monthly Rainfall (`rainfall_mm`)** | **22.06%** | The primary precipitation trigger |
| **4** | **Soil Type (`soil_type_enc`)** | **12.77%** | Infiltration capacity (Clay retains vs. Sandy drains) |
| **5** | **Drainage Infrastructure (`drainage_enc`)** | **7.64%** | Egress capacity of culverts and gutters |

---

## The 5 Composite Features & Seasonal Modifier

| # | Feature | Unit / Categories | Data Source |
|---|---|---|---|
| 1 | **Average Monthly Rainfall** | `0 – 800 mm` | NiMet station records & latitudinal baselines |
| 2 | **Elevation** | `0 – 1500 m` | NASA SRTM 30m DEM (Open Topo Data API) |
| 3 | **Distance to Nearest River** | `0 – 100 km` | `data/reference/nigeria_rivers.json` (12 Major Rivers) |
| 4 | **Soil Type** | `Clay`, `Loamy`, `Sandy`, `Silty` | FAO Soil Composition Map / User Selector |
| 5 | **Drainage Infrastructure** | `None`, `Basic`, `Moderate`, `Good` | Field Rating / User Selector |

### Seasonal Context Modifier Logic

The Random Forest model outputs a physical base risk class. A post-prediction engine applies seasonal precipitation intensity modifiers during the peak Nigerian wet season (June – October):

| Base Model Output | Dry Season Output | Wet Season Output |
| :---: | :---: | :---: |
| **Low** | Low | Low |
| **Medium** | Medium | **Medium-High** |
| **High** | High | High |
| **Critical** | Critical | Critical |

---

## Repository Structure

```
phense/
├── data/
│   ├── processed/
│   │   └── nigeria_flood_training.csv  ← Cleaned, real-grounded dataset (1,360 samples)
│   └── reference/
│       ├── nigeria_rivers.json         ← 12 major Nigerian rivers with vector paths
│       └── nigeria_lgas.json           ← 774 LGA centroids with coordinates
├── models/
│   ├── phense_model.pkl                ← Trained Random Forest classifier artifact
│   ├── encoders.pkl                    ← Ordinal & LabelEncoders
│   └── metrics.json                    ← Evaluation metrics, confusion matrix, feature importances
├── notebooks/
│   ├── 01_data_exploration.ipynb       ← EDA & feature distribution analysis
│   └── ML_WALKTHROUGH.md               ← Defense preparation & pipeline walkthrough
├── src/
│   ├── __init__.py
│   ├── build_real_dataset.py           ← Enriched spatial dataset builder script
│   ├── data_utils.py                   ← Data loading, validation, and ordinal encoding helpers
│   ├── fetch_lgas.py                   ← Nigerian LGA spatial reference setup
│   ├── geo.py                          ← Geo-enrichment engine (SRTM DEM, River distance, LGA lookup)
│   └── predict.py                      ← Inference pipeline with explainability & advisories
├── static/                             ← Modern Glassmorphic Frontend
│   ├── index.html                      ← Landing page & overview
│   ├── assess.html                     ← 1-Tap GPS assessment portal
│   ├── results.html                    ← Risk dashboard & advisory view
│   ├── methodology.html                ← Model architecture & metrics breakdown
│   └── about.html                      ← Capstone context & 3MTT identity
├── app.py                              ← FastAPI web server & static routes
├── train.py                            ← ML model training & evaluation pipeline
├── Dockerfile                          ← Docker container specification
├── Procfile                            ← Cloud deployment process manager
├── render.yaml                         ← Render deployment configuration
├── requirements.txt                    ← Python dependencies
└── README.md                           ← Project documentation
```

---

## Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/Usefulmech/phense.git
cd phense
pip install -r requirements.txt
```

### 2. Environment Setup

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

### 3. Model Training & Pipeline Execution

To train the Random Forest classifier and generate evaluation metrics:

```bash
python train.py
```

*Outputs created:* `models/phense_model.pkl`, `models/encoders.pkl`, `models/metrics.json`.

### 4. Run the Web Server & Application

Launch the FastAPI web server locally:

```bash
python app.py
```

Open your browser at `http://localhost:8000` to interact with the full web UI!

---

## API Reference

Phense provides RESTful JSON endpoints under `/api/`:

| Endpoint | Method | Payload / Parameters | Description |
| :--- | :---: | :--- | :--- |
| `/api/health` | `GET` | None | Returns API operational status |
| `/api/predict` | `POST` | `RiskRequest` JSON | Predicts composite risk class, driver, and advisory |
| `/api/elevation` | `GET` | `?lat=&lon=` | Fetches NASA SRTM 30m elevation in metres |
| `/api/river-proximity` | `GET` | `?lat=&lon=` | Finds nearest major Nigerian river and distance (km) |
| `/api/lga` | `GET` | `?lat=&lon=` | Reverse-geocodes lat/lon to Nigerian State and LGA |
| `/api/rainfall` | `GET` | `?lat=&lon=&season=` | Returns NiMet seasonal rainfall baseline (mm) |

### Sample Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 215.0,
    "elevation_m": 12.0,
    "river_distance_km": 0.8,
    "soil_type": "clay",
    "drainage": "none",
    "season": "wet"
  }'
```

### Sample Prediction Response

```json
{
  "risk_class": "Critical",
  "seasonal_label": "Critical",
  "confidence": 0.89,
  "primary_driver": "river_distance_km",
  "advisory": "IMMEDIATE ACTION REQUIRED: High vulnerability due to river proximity and poor drainage during peak rainfall. Prepare emergency evacuation protocols.",
  "feature_scores": {
    "elevation_m": 0.92,
    "river_distance_km": 0.96,
    "rainfall_mm": 0.74,
    "soil_type": 1.0,
    "drainage": 1.0
  }
}
```

---

## Cloud Deployment (Render / Docker)

### Deploying via Docker

```bash
docker build -t phense-app .
docker run -p 8000:8000 phense-app
```

### Deploying on Render.com

1. Connect your repository to Render.
2. Select **Web Service**.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## Capstone Information

* **Project:** Phense — Flood Risk Classifier (`AI-19`)
* **Program:** 3MTT NextGen Cohort — AI & Machine Learning
* **Fellow:** Adeniji Yusuf (`FE/24/5369659950`)
* **Training Provider:** Grazac Technologies Limited, Ogun State
* **License:** MIT License

---
