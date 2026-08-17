"""
Phense — FastAPI backend
Serves the prediction API and the static frontend.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Phense API",
    description="AI-powered composite flood risk assessment for Nigerian communities.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class RiskRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0, le=800,  description="Average monthly rainfall in mm")
    elevation_m: float = Field(..., ge=0, le=1500, description="Elevation above sea level in metres")
    river_distance_km: float = Field(..., ge=0, le=100, description="Distance to nearest river in km")
    soil_type: str     = Field(..., pattern="^(clay|sandy|loamy|silty)$")
    drainage: str      = Field(..., pattern="^(none|basic|moderate|good)$")
    season: str        = Field(..., pattern="^(dry|wet)$")
    state: str | None  = Field(None, description="Nigerian state (optional, for logging)")
    lga: str | None    = Field(None, description="LGA name (optional, for logging)")


class RiskResponse(BaseModel):
    risk_class: str        # Low | Medium | High | Critical
    seasonal_label: str    # May differ from risk_class during wet season
    confidence: float      # 0.0 – 1.0
    primary_driver: str    # The single most influential feature
    advisory: str          # Plain-language recommended action
    feature_scores: dict   # All 5 normalised feature contributions (0–1)


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "phense-api"}


@app.post("/api/predict", response_model=RiskResponse)
def predict(req: RiskRequest):
    """
    Run the Phense Random Forest classifier.
    Raises 503 if the model has not been trained yet.
    """
    try:
        from src.predict import predict_risk
        return predict_risk(req.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not yet trained. Run `python train.py` first.",
        )


@app.get("/api/elevation")
async def elevation(lat: float, lon: float):
    """
    Fetch elevation for a lat/lon pair from Open Topo Data (SRTM 30m).
    Returns elevation in metres above sea level.
    """
    from src.geo import get_elevation
    elev = await get_elevation(lat, lon)
    return {"elevation_m": elev}


@app.get("/api/river-proximity")
def river_proximity(lat: float, lon: float):
    """
    Return the name and distance (km) of the nearest major Nigerian river.
    Uses the bundled data/reference/nigeria_rivers.json database.
    """
    from src.geo import get_nearest_river
    return get_nearest_river(lat, lon)


@app.get("/api/lga")
def lga_lookup(lat: float, lon: float):
    """
    Reverse-geocode a lat/lon to a Nigerian State and LGA.
    Uses the bundled data/reference/nigeria_lgas.json database.
    """
    from src.geo import reverse_geocode_lga
    return reverse_geocode_lga(lat, lon)


@app.get("/api/rainfall")
def rainfall(lat: float, lon: float, season: str = "wet"):
    """
    Return the average monthly rainfall baseline for a location.
    Derived from NiMet station records mapped to geofenced regions.
    """
    from src.geo import get_baseline_rainfall
    return {"rainfall_mm": get_baseline_rainfall(lat, lon, season)}


# ── Static frontend ───────────────────────────────────────────────────────────
# Clean multi-page routes for progressive navigation
@app.get("/")
def home_page():
    return FileResponse("static/index.html")

@app.get("/assess")
def assess_page():
    return FileResponse("static/assess.html")

@app.get("/results")
def results_page():
    return FileResponse("static/results.html")

@app.get("/methodology")
def methodology_page():
    return FileResponse("static/methodology.html")

@app.get("/about")
def about_page():
    return FileResponse("static/about.html")


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("APP_PORT", 8000))
    print("\n" + "="*60)
    print(f"  Phense AI Server Started!")
    print(f"  Frontend Local URL: http://localhost:{port}")
    print("="*60 + "\n")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("APP_ENV", "development") == "development",
    )
