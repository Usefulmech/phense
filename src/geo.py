"""
Phense Geo Utilities
Handles all location-based auto-detection:
  - Elevation  (Open Topo Data API)
  - River proximity (local JSON database)
  - LGA lookup  (local JSON database)
  - Baseline rainfall (region-mapped NiMet averages)
"""

import os
import json
import math
import httpx
import requests


_ROOT = os.path.join(os.path.dirname(__file__), "..")
_REF  = os.path.join(_ROOT, "data", "reference")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_json(filename: str) -> list | dict:
    path = os.path.join(_REF, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Elevation — Open Topo Data API ────────────────────────────────────────────

TOPO_BASE = os.getenv("OPEN_TOPO_BASE_URL", "https://api.opentopodata.org/v1/srtm30m")

async def get_elevation(lat: float, lon: float) -> float:
    """
    Fetch elevation in metres from Open Topo Data (SRTM 30m dataset).
    Returns -1.0 on failure so the caller can show a manual-entry fallback.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                TOPO_BASE,
                params={"locations": f"{lat},{lon}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data["results"][0]["elevation"])
    except Exception:
        return -1.0


# ── River proximity ───────────────────────────────────────────────────────────

def get_nearest_river(lat: float, lon: float) -> dict:
    """
    Find the nearest major Nigerian river from the reference database.

    Returns
    -------
    dict: { river_name, state, distance_km, lat, lon }
    """
    rivers = _load_json("nigeria_rivers.json")
    if not rivers:
        # Fallback: no database yet
        return {"river_name": "Unknown", "distance_km": None, "error": "River database not yet populated."}

    best = None
    best_dist = float("inf")
    for r in rivers:
        for point in r.get("points", []):
            d = _haversine(lat, lon, point["lat"], point["lon"])
            if d < best_dist:
                best_dist = d
                best = {
                    "river_name":   r["name"],
                    "state":        r.get("states", []),
                    "distance_km":  round(best_dist, 2),
                    "nearest_lat":  point["lat"],
                    "nearest_lon":  point["lon"],
                }

    return best or {"river_name": "Unknown", "distance_km": None}


# ── LGA reverse-geocode ───────────────────────────────────────────────────────

def reverse_geocode_lga(lat: float, lon: float) -> dict:
    """
    Find the detailed capture address and canonical Nigerian LGA.
    First tries Nominatim OSM API for the real street/village address,
    and falls back to the local centroids database.
    """
    address_str = None
    state_name = None
    lga_name = None

    # 1. Try Nominatim API for detailed street-level/village-level address
    headers = {"User-Agent": "Phense-App/1.0 (contact: yusuf.adeniji@student.3mtt.gov.ng)"}
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1&accept-language=en"
    
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4.0) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode("utf-8"))
                address_str = res_data.get("display_name")
                addr = res_data.get("address", {})
                state_name = addr.get("state")
                lga_name = addr.get("county") or addr.get("city_district") or addr.get("suburb") or addr.get("city")
                
                if state_name:
                    state_name = state_name.replace(" State", "").strip()
                if lga_name:
                    lga_name = lga_name.replace(" Local Government Area", "").strip()
    except Exception:
        pass

    # 2. Local database fallback for LGA mapping (to ensure robust, canonical names)
    lgas = _load_json("nigeria_lgas.json")
    best = None
    if lgas:
        best_dist = float("inf")
        for entry in lgas:
            d = _haversine(lat, lon, entry["lat"], entry["lon"])
            if d < best_dist:
                best_dist = d
                best = {
                    "state":       entry["state"],
                    "lga":         entry["lga"],
                    "lat":         entry["lat"],
                    "lon":         entry["lon"],
                    "distance_km": round(best_dist, 2),
                }

    # 3. Compile output structure
    output_state = state_name or (best["state"] if best else "Unknown")
    output_lga = lga_name or (best["lga"] if best else "Unknown")
    output_address = address_str or (f"{output_lga}, {output_state} Region" if best else "Unknown Location")

    return {
        "state":       output_state,
        "lga":         output_lga,
        "address":     output_address,
        "lat":         lat,
        "lon":         lon,
        "distance_km": best["distance_km"] if best else 0.0
    }



# ── Baseline rainfall ─────────────────────────────────────────────────────────

# Geofenced rainfall bands derived from NiMet station records.
# Keyed by approximate latitude bands across Nigeria.
# Values are average monthly mm during wet season (April–October).
# Dry season values are ~30% of wet season.
# TODO: Replace with per-LGA table once NiMet data is fully processed.
_RAINFALL_BANDS = [
    # (max_lat, wet_avg_mm, label)
    (4.5,  320, "Coastal South"),      # Lagos, Rivers — high rainfall coastal belt
    (5.5,  270, "South South"),        # Edo, Delta, Cross River
    (6.5,  230, "South West"),         # Ogun, Ondo, Oyo
    (7.5,  210, "South East / Niger"), # Anambra, Enugu, Kogi, Niger
    (8.5,  180, "Middle Belt"),        # Benue, Plateau, Kwara, FCT
    (9.5,  145, "Upper Middle Belt"),  # Kaduna, Niger north, Nasarawa
    (10.5, 110, "North Central"),      # Kano, Jigawa, Bauchi
    (11.5,  75, "North West"),         # Sokoto, Zamfara, Kebbi
    (14.0,  45, "North East / Far North"), # Borno, Yobe, Adamawa north
]

def get_baseline_rainfall(lat: float, lon: float, season: str = "wet") -> float:
    """
    Return average monthly rainfall (mm) for a given location and season.
    """
    for (max_lat, wet_mm, _) in _RAINFALL_BANDS:
        if lat <= max_lat:
            return wet_mm if season == "wet" else round(wet_mm * 0.28, 1)
    # Default: far north
    return 45.0 if season == "wet" else 12.0
