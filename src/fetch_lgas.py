"""
fetch_lgas.py
Downloads ward-level coordinates for all Nigerian LGAs from the open-source
temikeezy/nigeria-geojson-data repository, computes the centroid coordinate
for each LGA by averaging ward coordinates, and saves the database to
data/reference/nigeria_lgas.json.
"""

import os
import json
import urllib.request

URL = "https://raw.githubusercontent.com/temikeezy/nigeria-geojson-data/master/data/full.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reference")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nigeria_lgas.json")

def main():
    print("=" * 60)
    print("  PHENSE — Fetching Nigeria LGA Centroids")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading ward dataset from:\n  {URL} ...")
    try:
        req = urllib.request.Request(
            URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error downloading LGA data: {e}")
        return

    print("Processing ward coordinates into LGA centroids...")
    lgas_output = []
    
    for state_entry in data:
        state_name = state_entry["state"]
        for lga_entry in state_entry["lgas"]:
            lga_name = lga_entry["name"]
            wards = lga_entry.get("wards", [])
            
            latitudes = [w["latitude"] for w in wards if w.get("latitude") is not None]
            longitudes = [w["longitude"] for w in wards if w.get("longitude") is not None]
            
            if latitudes and longitudes:
                avg_lat = sum(latitudes) / len(latitudes)
                avg_lon = sum(longitudes) / len(longitudes)
                lgas_output.append({
                    "state": state_name,
                    "lga": lga_name,
                    "lat": round(avg_lat, 6),
                    "lon": round(avg_lon, 6)
                })
            else:
                # Fallback if no wards have coordinates (unlikely in this dataset)
                print(f"Warning: No coordinates for LGA {lga_name}, {state_name}")

    print(f"Successfully processed {len(lgas_output)} LGAs.")
    
    # Save the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lgas_output, f, indent=2, ensure_ascii=False)
        
    print(f"Saved LGA centroids to:\n  {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
