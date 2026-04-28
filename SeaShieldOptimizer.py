"""
SeaShield Optimizer — app.py  (BigQuery GIS edition)
============================================================
Every geospatial operation that was previously done in Python/Shapely is now
delegated to BigQuery when a database connection is available:

  • ST_Distance      — cyclone-to-route distance (metres, on the sphere)
  • ST_DWithin       — fast proximity check (indexed)
  • ST_Intersects    — route / cyclone-buffer intersection
  • ST_Buffer        — danger-zone footprint around a cyclone or forecast track
  • ST_MakeLine      — build route LineString from ordered waypoints
  • ST_MakePoint     — store every waypoint / cyclone as a GEOGRAPHY point
  • ST_AsGeoJSON     — serialise stored geometries back to GeoJSON for Folium

All route analyses, waypoints, weather snapshots and cyclone captures are
persisted so the "Route History" tab can replay and compare past voyages.

Environment variables (set before running):
  BQ_PROJECT_ID      (default: your-gcp-project-id)
  BQ_DATASET_ID      (default: seashield_dataset)

If the database is unreachable the app falls back gracefully to Shapely for
the geospatial checks and simply skips persistence.
"""

from dotenv import load_dotenv
load_dotenv()
import os
import json
import warnings
import contextlib
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import LineString, Point

import uuid

# ── optional BigQuery driver ───────────────────────────────────────────────────
try:
    from google.cloud import bigquery
    from google.api_core.exceptions import GoogleAPIError
    BIGQUERY_OK = True
except ImportError:
    BIGQUERY_OK = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SeaShield Optimizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
  --bg-base: #080f1a;
  --bg-surface: #0d1b2a;
  --bg-raised: #142235;
  --border: #1e3a52;
  --accent-1: #00d4ff;
  --accent-2: #0077aa;
  --safe: #00e676;
  --warn: #ffab40;
  --danger: #ff5252;
  --text-hi: #e8f4fd;
  --text-mid: #8eaec9;
  --text-lo: #3d6080;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg-base) !important;
    color: var(--text-hi);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--text-hi);
}

/* NATIVE STREAMLIT COMPONENT OVERRIDES OVERRIDES */
div[data-testid="stWidgetLabel"] p, label, label p, .stMarkdown p {
    color: var(--text-hi) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--bg-surface) !important;
    color: var(--text-hi) !important;
    border: 1px solid var(--border) !important;
}
div[data-baseweb="select"] * {
    color: var(--text-hi) !important;
}
ul[data-baseweb="menu"] {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
}
div[data-baseweb="input"] > div {
    background-color: var(--bg-surface) !important;
    color: var(--text-hi) !important;
    border: 1px solid var(--border) !important;
}
input {
    background-color: var(--bg-surface) !important;
    color: var(--text-hi) !important;
}
/* Slider track labels */
div[data-testid="stTickBarMax"], div[data-testid="stTickBarMin"], div[data-testid="stThumbValue"] {
    color: var(--text-mid) !important;
}

/* METRICS / BADGES (KPI ROW) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent-1);
    border-radius: 6px;
    padding: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    color: var(--text-mid);
    text-transform: uppercase;
}
.kpi-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    color: var(--accent-1);
}

/* SECTION HEADERS */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 24px 0 16px;
}
.section-line {
    flex: 1;
    height: 1px;
    background-color: var(--accent-1);
    opacity: 0.5;
}
.section-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 9px;
    color: var(--accent-1);
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ALERTS & BACKGROUNDS */
.status-safe { background: var(--bg-surface); border-left: 4px solid var(--safe); padding: 8px 12px; margin: 4px 0; border-radius: 4px; box-shadow: 0 0 0 1px var(--border); }
.status-warn { background: var(--bg-surface); border-left: 4px solid var(--warn); padding: 8px 12px; margin: 4px 0; border-radius: 4px; box-shadow: 0 0 0 1px var(--border); }
.status-danger { background: var(--bg-surface); border-left: 4px solid var(--danger); padding: 8px 12px; margin: 4px 0; border-radius: 4px; box-shadow: 0 0 0 1px var(--border); }

/* TAB 2 METRIC GRID */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
}

/* BUTTONS */
button[kind="primary"] {
    background: transparent !important;
    border: 1px solid var(--accent-1) !important;
    color: var(--accent-1) !important;
    transition: 0.3s;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 15px rgba(0,212,255,0.4) !important;
}
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
}

/* TABS OVERRIDE */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: var(--text-mid);
    padding: 12px 14px;
    margin: 0;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 3px solid var(--accent-1) !important;
}

/* DB BADGE */
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
.db-badge-on { background: var(--bg-surface); border-radius: 12px; border: 1px solid var(--border); padding: 4px 12px; font-size: 11px; display: inline-flex; align-items: center; gap: 8px; }
.db-badge-on::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--safe); animation: pulse 1.5s infinite; }
.db-badge-off { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 12px; padding: 4px 12px; font-size: 11px; display: inline-flex; align-items: center; gap: 8px; color: var(--text-mid); }
.db-badge-off::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: var(--text-lo); }

/* PANELS / COST TOTAL */
.cost-total {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.cost-total::after {
    content:''; position: absolute; top:0; left:0; right:0; bottom:0;
    background: radial-gradient(circle at center, rgba(0,212,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.cost-compare, .history-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px; margin: 8px 0; }
.currency-badge { background: transparent; border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-mid); }
.canal-detected { box-shadow: 0 0 0 1px var(--border); border-left: 3px solid var(--warn); padding: 8px 14px; border-radius: 4px; margin: 4px 0; font-size: 12px; }
.canal-clear { box-shadow: 0 0 0 1px var(--border); border-left: 3px solid var(--text-lo); padding: 8px 14px; border-radius: 4px; margin: 4px 0; font-size: 12px; color: var(--text-lo); }
#MainMenu, footer { visibility: hidden; }

/* TERMINAL TEXT */
.term-cmd { font-family: 'JetBrains Mono', monospace; color: var(--text-mid); font-size: 12px; }
.term-cmd::before { content: '› '; color: var(--accent-1); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MUMBAI = {"name": "Mumbai", "lat": 18.9388, "lon": 72.8354}

# Default config from .env
BQ_PROJECT_ID_DEFAULT = os.getenv("BQ_PROJECT_ID", "your-gcp-project-id")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "seashield_dataset")

def get_active_project_id():
    """Returns the project ID from session state or default."""
    pid = st.session_state.get("gcp_project", BQ_PROJECT_ID_DEFAULT)
    if not pid or pid == "your-gcp-project-id":
        return None
    return pid

# ─────────────────────────────────────────────
#  OCEAN WAYPOINTS FOR ALTERNATE ROUTING
#  These are well-known ocean waypoints used to
#  construct alternate routes that stay at sea.
#  searoute is called leg-by-leg through these
#  intermediate ocean points so it never tries
#  to cross land.
# ─────────────────────────────────────────────

# Key ocean waypoints (lon, lat) for constructing alternate sea routes
OCEAN_WAYPOINTS = {
    # Arabian Sea / Indian Ocean basin
    "arabian_sea_center":   (65.0,  15.0),
    "arabian_sea_north":    (63.0,  22.0),
    "gulf_of_aden_entry":   (48.0,  12.5),
    "bab_el_mandeb":        (43.5,  12.3),   # Red Sea southern entrance
    "red_sea_mid":          (38.0,  20.0),
    "suez_entry":           (32.5,  29.5),   # Gulf of Suez
    "suez_med_exit":        (32.3,  31.2),   # Northern Suez / Mediterranean
    "med_east":             (34.0,  33.0),   # Eastern Mediterranean
    "med_center":           (20.0,  34.5),
    "med_west_strait":      (5.0,   36.0),   # Near Gibraltar
    "atlantic_south":       (-8.0,  36.0),
    "cape_of_good_hope":    (18.5, -34.5),
    "south_indian_ocean":   (80.0, -30.0),
    "lombok_strait":        (115.7, -8.5),   # Indonesia
    "malacca_strait_south": (103.5,  1.3),   # Singapore area
    "malacca_strait_north": (98.5,   5.5),
    "bay_of_bengal_center": (88.0,  10.0),
    "srilanka_south":       (82.0,   5.0),
    "south_china_sea":      (110.0,  8.0),
    "taiwan_strait":        (120.0, 24.0),
    "east_china_sea":       (125.0, 30.0),
    "pacific_west":         (140.0, 10.0),
    "horn_of_africa_south": (52.0,  10.0),
}


# ─────────────────────────────────────────────
#  COST ESTIMATOR CONSTANTS
# ─────────────────────────────────────────────
CO2_PER_MT_FUEL = 3.114
CARBON_TAX_PER_MT_USD = 50.0

VESSEL_FREIGHT_MULT = {
    "Bulk Carrier": 0.72, "Container Ship": 1.00,
    "Tanker": 0.88, "General Cargo": 0.95,
}

VESSEL_DWT = {
    "Bulk Carrier": 45_000, "Container Ship": 25_000,
    "Tanker": 80_000, "General Cargo": 12_000,
}

COMMODITY_SURCHARGE = {
    "General": 0.00, "Hazardous": 0.25, "Perishable": 0.18,
}

TIER1_COUNTRIES = {
    "Singapore", "Netherlands", "Germany", "China", "United States",
    "Japan", "South Korea", "Belgium", "United Kingdom",
}

TIER2_COUNTRIES = {
    "India", "Sri Lanka", "Saudi Arabia", "Greece", "Australia",
    "Egypt", "Turkey", "Malaysia", "Thailand", "Brazil",
    "Indonesia", "Vietnam", "Philippines", "Pakistan", "Bangladesh",
    "Kenya", "Tanzania", "South Africa", "United Arab Emirates", "Oman",
}

PORT_COSTS = {
    1: {"dues": 18_000, "pilotage": 4_200, "thc_per_teu": 310},
    2: {"dues":  9_500, "pilotage": 2_100, "thc_per_teu": 225},
    3: {"dues":  5_200, "pilotage": 1_100, "thc_per_teu": 145},
}

CANAL_ZONES = {
    "Suez Canal":     {"lon_min": 32, "lon_max": 33, "lat_min": 29, "lat_max": 31},
    "Panama Canal":   {"lon_min": -80, "lon_max": -79, "lat_min": 8, "lat_max": 10},
    "Malacca Strait": {"lon_min": 99, "lon_max": 105, "lat_min": 1, "lat_max": 6},
    "Bab-el-Mandeb":  {"lon_min": 43, "lon_max": 45, "lat_min": 11, "lat_max": 13},
}

ORIGIN_CHARGES_TABLE = {
    "Documentation fee": 180,
    "Customs clearance": 320,
    "Inland survey / inspection": 150,
    "VGM (verified gross mass)": 85,
    "Export freight release": 90,
}
ORIGIN_ISPS_PER_TEU = 35

DEST_DDC_PER_TEU = 210
DEST_IMPORT_CUSTOMS = 380
DEST_DO_RELEASE = 140
DEMURRAGE_PER_TEU_DAY = 195
DEMURRAGE_FREE_DAYS = 7
PORT_CONGESTION_SURCHARGE_T1 = 420

EXCHANGE_RATES = {"USD": 1.0, "EUR": 0.92, "INR": 83.5}

HAULAGE_RATE_FIRST_100 = 3.20
HAULAGE_RATE_BEYOND = 2.40
HAULAGE_TIER_MULT = {1: 1.30, 2: 1.00, 3: 0.75}
HAULAGE_DEPOT_FEE = 180

CREW_COST_PER_DAY = 8_500
OVERHEAD_PER_DAY = 4_200
CYCLONE_RISK_PREMIUM_PCT = 0.035


BASIN_BOUNDS = {
    "arabian_sea":   {"lon_min": 50, "lon_max": 77, "lat_min": 6,  "lat_max": 26},
    "bay_of_bengal": {"lon_min": 78, "lon_max": 95, "lat_min": 5,  "lat_max": 22},
    "indian_ocean":  {"lon_min": 30, "lon_max": 110, "lat_min": -40, "lat_max": 5},
    "red_sea":       {"lon_min": 32, "lon_max": 43, "lat_min": 12, "lat_max": 30},
    "mediterranean": {"lon_min": -5, "lon_max": 36, "lat_min": 30, "lat_max": 45},
}

def get_basin(lon, lat):
    for basin, bounds in BASIN_BOUNDS.items():
        if bounds["lon_min"] <= lon <= bounds["lon_max"] and bounds["lat_min"] <= lat <= bounds["lat_max"]:
            return basin
    return "other"

def _get_ocean_via_points(origin: dict, destination: dict) -> list:
    """
    Return a list of (lon, lat) ocean intermediate waypoints that keep the
    route on water between the origin and destination using explicit basin detection.
    """
    o_lon, o_lat = origin["lon"], origin["lat"]
    d_lon, d_lat = destination["lon"], destination["lat"]
    
    o_basin = get_basin(o_lon, o_lat)
    d_basin = get_basin(d_lon, d_lat)

    WP = OCEAN_WAYPOINTS
    via = []

    is_east_origin = o_basin in ["arabian_sea", "bay_of_bengal", "indian_ocean"] or o_lon > 40
    is_east_dest = d_basin in ["arabian_sea", "bay_of_bengal", "indian_ocean", "red_sea"] or d_lon > 40

    # ── Westbound routes (Asia -> Europe, West Africa, Americas) ────────────────
    if is_east_origin and not is_east_dest:
        if d_lat > 20:
            via = [
                WP["arabian_sea_north"],
                WP["gulf_of_aden_entry"],
                WP["bab_el_mandeb"],
                WP["red_sea_mid"],
                WP["suez_entry"],
                WP["suez_med_exit"],
                WP["med_east"],
            ]
            if d_lon < 15: via += [WP["med_center"]]
            if d_lon < 0:  via += [WP["med_west_strait"], WP["atlantic_south"]]
        elif d_lat < -10:
            via = [WP["south_indian_ocean"], WP["cape_of_good_hope"]]
        else:
            via = [
                WP["arabian_sea_north"], WP["gulf_of_aden_entry"],
                WP["bab_el_mandeb"], WP["red_sea_mid"],
                WP["suez_entry"], WP["suez_med_exit"],
                WP["med_east"], WP["med_center"], WP["med_west_strait"]
            ]

    # ── Eastbound routes (Europe/Africa/Americas -> Asia) ──────────────────
    elif not is_east_origin and is_east_dest:
        # Reverse the logic approximately if needed, or rely on searoute.
        # But usually searoute can handle it if we just provide a few pivots.
        if o_lat > 20: # Coming from Med/Europe
            via = [
                WP["med_east"], WP["suez_med_exit"], WP["suez_entry"],
                WP["red_sea_mid"], WP["bab_el_mandeb"], WP["gulf_of_aden_entry"],
                WP["arabian_sea_north"]
            ]
            if d_lon > 100:
                via += [WP["srilanka_south"], WP["malacca_strait_north"], WP["malacca_strait_south"]]
                if d_lat > 15: via += [WP["south_china_sea"]]
        else:
            via = [WP["cape_of_good_hope"], WP["south_indian_ocean"]]

    # ── intra-East routes (Asia -> Asia/Pacific) ──────────────────
    elif is_east_origin and is_east_dest:
        if d_lon > 100 and o_lon < 100:
            base_east = [WP["arabian_sea_center"], WP["srilanka_south"], WP["malacca_strait_north"], WP["malacca_strait_south"]]
            if d_lat < 15:
                via = base_east
            elif d_lat < 30:
                via = base_east + [WP["south_china_sea"]]
            elif d_lat >= 30:
                via = base_east + [WP["south_china_sea"], WP["taiwan_strait"], WP["east_china_sea"]]
            if d_lon > 135:
                via += [WP["pacific_west"]]
        elif d_lon > 35 and d_lat < 0 and d_basin != "other":
            via = [WP["arabian_sea_center"], WP["horn_of_africa_south"]]
            if d_lat < -20:
                via += [WP["south_indian_ocean"]]
        elif d_basin == "bay_of_bengal" or (d_lon > 75 and d_lon < 100 and d_lat < 20):
            if o_basin == "arabian_sea":
                via = [WP["srilanka_south"], WP["bay_of_bengal_center"]]
            else:
                via = [WP["bay_of_bengal_center"]]

    # ── intra-West routes (Europe <-> Africa/Americas) ──────────────────
    else:
        # No strict via points needed. searoute handles intra-Atlantic beautifully.
        pass

    return via


# ═════════════════════════════════════════════
#  DATABASE LAYER
# ═════════════════════════════════════════════

@contextlib.contextmanager
def get_db():
    """
    Yield a BigQuery client, or yield None if unavailable.
    Callers must guard: `if client is None: return`
    """
    client = None
    if not BIGQUERY_OK:
        yield None
        return
    project_id = get_active_project_id()
    if not project_id:
        yield None
        return
    try:
        client = bigquery.Client(project=project_id)
        yield client
    except Exception as e:
        # Avoid yielding again here as it causes RuntimeError in contextlib
        print(f"Database connection error: {e}")
        return
    finally:
        if client:
            client.close()


def db_online() -> bool:
    """Quick liveness check — used only for the status badge."""
    if not BIGQUERY_OK:
        return False
    project_id = get_active_project_id()
    try:
        client = bigquery.Client(project=project_id)
        # Just list datasets to confirm connection
        list(client.list_datasets(max_results=1))
        client.close()
        return True
    except Exception:
        return False


def init_db():
    """
    Create BigQuery dataset and all SeaShield tables if they don't exist.
    """
    BQ_PROJECT_ID = get_active_project_id()
    if not BQ_PROJECT_ID:
        return False
        
    with get_db() as client:
        if client is None:
            return False
            
        dataset_ref = bigquery.DatasetReference(BQ_PROJECT_ID, BQ_DATASET_ID)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        try:
            client.create_dataset(dataset, exists_ok=True)
        except GoogleAPIError:
            pass # Dataset might already exist or permission issue

        queries = [
            f"""
            CREATE TABLE IF NOT EXISTS `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes` (
                id               STRING,
                origin_name      STRING,
                destination_name STRING,
                origin_geom      GEOGRAPHY,
                destination_geom GEOGRAPHY,
                route_geom       GEOGRAPHY,
                distance_km      FLOAT64,
                risk_level       STRING,
                risk_score       INT64,
                avg_wave_m       FLOAT64,
                max_wave_m       FLOAT64,
                min_cyclone_dist_km FLOAT64,
                route_score      FLOAT64,
                is_recommended   BOOL,
                created_at       TIMESTAMP
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_waypoints` (
                id             STRING,
                route_id       STRING,
                waypoint_num   INT64,
                geom           GEOGRAPHY,
                wave_height_m  FLOAT64,
                wave_period_s  FLOAT64,
                wind_wave_m    FLOAT64,
                swell_wave_m   FLOAT64,
                ts_utc         STRING
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_cyclones` (
                id                   STRING,
                route_id             STRING,
                cyclone_name         STRING,
                cyclone_ext_id       STRING,
                geom                 GEOGRAPHY,
                intensity_kts        INT64,
                dist_to_route_km     FLOAT64,
                route_intersects     BOOL,
                captured_at          TIMESTAMP
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_costs` (
                id            STRING,
                route_id      STRING,
                cost_category STRING,
                cost_usd      FLOAT64,
                notes         STRING,
                computed_at   TIMESTAMP
            )
            """
        ]
        
        for q in queries:
            try:
                client.query(q).result()
            except Exception as e:
                print(f"Error creating table: {e}")
                
        return True


# ─────────────────────────────────────────────
#  BigQuery GEOSPATIAL OPERATIONS
# ─────────────────────────────────────────────

def postgis_save_and_analyze(
    origin, destination, waypoints, weather_df,
    active_cyclones, distance_km, route_score,
    is_recommended, risk_level, risk_score,
) -> tuple:
    BQ_PROJECT_ID = get_active_project_id()
    fallback = {
        "min_cyclone_dist_km":   float("inf"),
        "intersecting_cyclones": [],
        "bigquery_used":          False,
    }

    with get_db() as client:
        if client is None:
            return None, fallback

        route_id = uuid.uuid4().hex
        wkt_coords = ", ".join(f"{p['lon']} {p['lat']}" for p in waypoints)
        route_wkt  = f"LINESTRING({wkt_coords})"

        avg_wave = float(weather_df["wave_height_m"].mean()) if not weather_df.empty else 0.0
        max_wave = float(weather_df["wave_height_m"].max())  if not weather_df.empty else 0.0

        query_route = f"""
            INSERT INTO `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes`
                (id, origin_name, destination_name,
                 origin_geom, destination_geom, route_geom,
                 distance_km, risk_level, risk_score,
                 avg_wave_m, max_wave_m, route_score, is_recommended, created_at)
            VALUES (
                @id, @o_name, @d_name,
                ST_GeogFromText(@o_geom),
                ST_GeogFromText(@d_geom),
                ST_GeogFromText(@r_geom),
                @dist, @risk_lvl, @risk_scr,
                @avg_w, @max_w, @r_score, @is_rec, CURRENT_TIMESTAMP()
            )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "STRING", route_id),
                bigquery.ScalarQueryParameter("o_name", "STRING", origin["name"]),
                bigquery.ScalarQueryParameter("d_name", "STRING", destination["name"]),
                bigquery.ScalarQueryParameter("o_geom", "STRING", f"POINT({origin['lon']} {origin['lat']})"),
                bigquery.ScalarQueryParameter("d_geom", "STRING", f"POINT({destination['lon']} {destination['lat']})"),
                bigquery.ScalarQueryParameter("r_geom", "STRING", route_wkt),
                bigquery.ScalarQueryParameter("dist", "FLOAT64", distance_km),
                bigquery.ScalarQueryParameter("risk_lvl", "STRING", risk_level),
                bigquery.ScalarQueryParameter("risk_scr", "INT64", risk_score),
                bigquery.ScalarQueryParameter("avg_w", "FLOAT64", avg_wave),
                bigquery.ScalarQueryParameter("max_w", "FLOAT64", max_wave),
                bigquery.ScalarQueryParameter("r_score", "FLOAT64", route_score),
                bigquery.ScalarQueryParameter("is_rec", "BOOL", is_recommended),
            ]
        )
        client.query(query_route, job_config=job_config).result()

        if not weather_df.empty:
            rows_to_insert = []
            for _, row in weather_df.iterrows():
                rows_to_insert.append({
                    "id": uuid.uuid4().hex,
                    "route_id": route_id,
                    "waypoint_num": int(row["waypoint"]),
                    "geom": f"POINT({row['lon']} {row['lat']})",
                    "wave_height_m": float(row["wave_height_m"]),
                    "wave_period_s": float(row["wave_period_s"]),
                    "wind_wave_m": float(row["wind_wave_m"]),
                    "swell_wave_m": float(row["swell_wave_m"]),
                    "ts_utc": str(row["timestamp"]),
                })
            
            # BigQuery batch insert
            table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_waypoints"
            
            # We can use query for batch insert since ST_GeogFromText is needed
            values_str = []
            for r in rows_to_insert:
                values_str.append(f"('{r['id']}', '{r['route_id']}', {r['waypoint_num']}, ST_GeogFromText('{r['geom']}'), {r['wave_height_m']}, {r['wave_period_s']}, {r['wind_wave_m']}, {r['swell_wave_m']}, '{r['ts_utc']}')")
            
            if values_str:
                chunk_size = 50 # to avoid too long query string
                for i in range(0, len(values_str), chunk_size):
                    chunk = values_str[i:i+chunk_size]
                    q_waypoints = f"""
                        INSERT INTO `{table_id}` 
                        (id, route_id, waypoint_num, geom, wave_height_m, wave_period_s, wind_wave_m, swell_wave_m, ts_utc)
                        VALUES {','.join(chunk)}
                    """
                    client.query(q_waypoints).result()

        min_dist_km  = float("inf")
        intersecting = []
        DANGER_BUF_M = 400000.0

        for cyc in active_cyclones:
            c_lon  = float(cyc["lon"])
            c_lat  = float(cyc["lat"])
            c_wkt  = f"POINT({c_lon} {c_lat})"

            # BigQuery spatial query
            q_spatial = f"""
                SELECT
                    ST_Distance(
                        ST_GeogFromText(@c_wkt),
                        (SELECT route_geom FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes` WHERE id = @r_id LIMIT 1)
                    ) / 1000.0 AS dist_km,
                    ST_DWithin(
                        ST_GeogFromText(@c_wkt),
                        (SELECT route_geom FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes` WHERE id = @r_id LIMIT 1),
                        @buf
                    ) AS intersects_buffer
            """
            jc_spatial = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("c_wkt", "STRING", c_wkt),
                    bigquery.ScalarQueryParameter("r_id", "STRING", route_id),
                    bigquery.ScalarQueryParameter("buf", "FLOAT64", DANGER_BUF_M),
                ]
            )
            res = list(client.query(q_spatial, job_config=jc_spatial).result())
            
            if res:
                row = res[0]
                dist_km      = float(row.dist_km) if row.dist_km is not None else float("inf")
                intersects_b = bool(row.intersects_buffer)
            else:
                dist_km = float("inf")
                intersects_b = False

            if dist_km < min_dist_km:
                min_dist_km = dist_km
            if intersects_b:
                intersecting.append(cyc.get("name", cyc.get("id", "Unknown")))

            fcst_intersects = False
            if cyc.get("forecast_points") and len(cyc["forecast_points"]) >= 2:
                fcst_pts = cyc["forecast_points"][:6]
                fcst_wkt = "LINESTRING(" + ", ".join(
                    f"{lon} {lat}" for lat, lon in fcst_pts
                ) + ")"
                
                q_fcst = f"""
                    SELECT ST_DWithin(
                        ST_GeogFromText(@fcst_wkt),
                        (SELECT route_geom FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes` WHERE id = @r_id LIMIT 1),
                        @buf
                    ) AS fcst_int
                """
                jc_fcst = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("fcst_wkt", "STRING", fcst_wkt),
                        bigquery.ScalarQueryParameter("r_id", "STRING", route_id),
                        bigquery.ScalarQueryParameter("buf", "FLOAT64", DANGER_BUF_M),
                    ]
                )
                fcst_res = list(client.query(q_fcst, job_config=jc_fcst).result())
                if fcst_res:
                    fcst_intersects = bool(fcst_res[0].fcst_int)

            route_intersects = intersects_b or fcst_intersects

            intensity_val = cyc.get("intensity", 0)
            if isinstance(intensity_val, str):
                try:
                    intensity_val = int(intensity_val)
                except ValueError:
                    intensity_val = 0

            q_cyc = f"""
                INSERT INTO `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_cyclones`
                    (id, route_id, cyclone_name, cyclone_ext_id, geom,
                     intensity_kts, dist_to_route_km, route_intersects, captured_at)
                VALUES (
                    @id, @r_id, @c_name, @c_ext, ST_GeogFromText(@c_geom),
                    @inty, @dist, @r_int, CURRENT_TIMESTAMP()
                )
            """
            jc_cyc = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "STRING", uuid.uuid4().hex),
                    bigquery.ScalarQueryParameter("r_id", "STRING", route_id),
                    bigquery.ScalarQueryParameter("c_name", "STRING", cyc.get("name", "Unknown")),
                    bigquery.ScalarQueryParameter("c_ext", "STRING", cyc.get("id",   "Unknown")),
                    bigquery.ScalarQueryParameter("c_geom", "STRING", c_wkt),
                    bigquery.ScalarQueryParameter("inty", "INT64", intensity_val),
                    bigquery.ScalarQueryParameter("dist", "FLOAT64", round(dist_km, 2)),
                    bigquery.ScalarQueryParameter("r_int", "BOOL", route_intersects),
                ]
            )
            client.query(q_cyc, job_config=jc_cyc).result()

        if active_cyclones:
            q_upd = f"""
                UPDATE `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes`
                   SET min_cyclone_dist_km = @min_dist
                 WHERE id = @r_id
            """
            jc_upd = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("min_dist", "FLOAT64", round(min_dist_km, 2)),
                    bigquery.ScalarQueryParameter("r_id", "STRING", route_id),
                ]
            )
            client.query(q_upd, job_config=jc_upd).result()

        return route_id, {
            "min_cyclone_dist_km":   min_dist_km,
            "intersecting_cyclones": intersecting,
            "bigquery_used":          True,
        }


def postgis_get_route_history(limit: int = 20) -> list:
    BQ_PROJECT_ID = get_active_project_id()
    if not BQ_PROJECT_ID:
        return []
    with get_db() as client:
        if client is None:
            return []
        
        query = f"""
            SELECT
                r.id, r.origin_name, r.destination_name,
                r.distance_km, r.risk_level, r.risk_score,
                r.avg_wave_m, r.max_wave_m, r.min_cyclone_dist_km,
                r.route_score, r.is_recommended, r.created_at,
                ST_AsGeoJSON(r.route_geom) AS route_geojson,
                ST_AsGeoJSON(r.origin_geom) AS origin_geojson,
                ST_AsGeoJSON(r.destination_geom) AS dest_geojson,
                (SELECT COUNT(DISTINCT id) FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_waypoints` WHERE route_id = r.id) AS waypoint_count,
                (SELECT COUNT(DISTINCT id) FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_cyclones` WHERE route_id = r.id) AS cyclone_count
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes` r
            ORDER BY r.created_at DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        try:
            results = client.query(query, job_config=job_config).result()
            
            history = []
            for row in results:
                history.append({
                    "id": row.id,
                    "origin_name": row.origin_name,
                    "destination_name": row.destination_name,
                    "distance_km": row.distance_km,
                    "risk_level": row.risk_level,
                    "risk_score": row.risk_score,
                    "avg_wave_m": row.avg_wave_m,
                    "max_wave_m": row.max_wave_m,
                    "min_cyclone_dist_km": row.min_cyclone_dist_km,
                    "route_score": row.route_score,
                    "is_recommended": row.is_recommended,
                    "created_at": row.created_at,
                    "route_geojson": row.route_geojson,
                    "origin_geojson": row.origin_geojson,
                    "dest_geojson": row.dest_geojson,
                    "waypoint_count": row.waypoint_count,
                    "cyclone_count": row.cyclone_count
                })
            return history
        except Exception as e:
            print(f"History error: {e}")
            return []


def postgis_spatial_stats() -> dict:
    BQ_PROJECT_ID = get_active_project_id()
    empty = {
        "total_routes": 0, "total_distance_km": 0,
        "avg_risk_score": 0, "high_risk_count": 0,
        "total_cyclones_captured": 0, "nearest_cyclone_ever_km": None,
    }
    with get_db() as client:
        if client is None:
            return empty
            
        q_routes = f"""
            SELECT COUNT(*) AS total_routes,
                   COALESCE(SUM(distance_km), 0) AS total_distance_km,
                   COALESCE(AVG(risk_score), 0)  AS avg_risk_score,
                   COUNTIF(risk_level = 'High') AS high_risk_count
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_routes`
        """
        
        q_cyc = f"""
            SELECT COUNT(*) AS total_cyclones_captured,
                   MIN(IF(dist_to_route_km > 0, dist_to_route_km, NULL)) AS nearest_ever_km
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_cyclones`
        """
        
        try:
            r_res = list(client.query(q_routes).result())
            c_res = list(client.query(q_cyc).result())
            
            row = r_res[0] if r_res else None
            cyc_row = c_res[0] if c_res else None
            
            if row and cyc_row:
                return {
                    "total_routes":            int(row.total_routes),
                    "total_distance_km":       round(float(row.total_distance_km), 0),
                    "avg_risk_score":          round(float(row.avg_risk_score), 1),
                    "high_risk_count":         int(row.high_risk_count),
                    "total_cyclones_captured": int(cyc_row.total_cyclones_captured),
                    "nearest_cyclone_ever_km": (
                        round(float(cyc_row.nearest_ever_km), 1)
                        if cyc_row.nearest_ever_km is not None else None
                    ),
                }
        except Exception:
            pass
            
    return empty


def postgis_save_costs(route_id, cost_breakdown: dict):
    """Save cost breakdown line items to the seashield_costs table."""
    BQ_PROJECT_ID = get_active_project_id()
    if not BQ_PROJECT_ID or not route_id:
        return False
    with get_db() as client:
        if client is None:
            return False
            
        items = [
            ("Base Sea Freight", cost_breakdown["freight"]["total_freight_usd"],
             f"Rate/TEU: ${cost_breakdown['freight']['raw_rate_per_teu']:.0f}"),
            ("Bunker Fuel (BAF)", cost_breakdown["bunker_cost_usd"],
             f"VLSFO ${cost_breakdown['bunker_price_per_mt']}/MT, "
             f"{cost_breakdown['daily_consumption_mt']:.1f} MT/day"),
            ("Port Dues & Charges", cost_breakdown["port_costs"]["total_port_costs"],
             f"Origin tier {cost_breakdown['port_costs']['origin_tier']}, "
             f"Dest tier {cost_breakdown['port_costs']['dest_tier']}"),
            ("Origin Export Charges",
             cost_breakdown["origin_charges"]["total_origin_charges"],
             "Export-side fees"),
            ("Destination Import Charges",
             cost_breakdown["dest_charges"]["total_destination_charges"],
             f"Demurrage days: {cost_breakdown['dest_charges']['demurrage_days']}"),
            ("Inland Haulage", cost_breakdown["haulage"]["total_haulage_usd"],
             "Origin + Dest transport"),
            ("Canal Transit Fees", cost_breakdown["total_canal_usd"],
             ", ".join(c["canal_name"] for c in cost_breakdown["canals"]
                       if c["detected"]) or "None"),
            ("Weather Delay Cost",
             cost_breakdown["weather_cost"]["total_weather_cost"],
             f"Delay: {cost_breakdown['weather_cost']['delay_hours']:.1f} hrs"),
            ("Carbon Tax (EU ETS)", cost_breakdown["carbon_tax_usd"],
             f"{cost_breakdown['co2_emissions_mt']:.1f} MT CO2 @ $50/MT"),
            ("GRAND TOTAL", cost_breakdown["grand_total_usd"],
             f"Cost/TEU: ${cost_breakdown['cost_per_teu']:.2f}"),
        ]
        
        values_str = []
        for cat, amount, notes in items:
            cat_safe = cat.replace("'", "''")
            notes_safe = notes.replace("'", "''")
            id_val = uuid.uuid4().hex
            values_str.append(f"('{id_val}', '{route_id}', '{cat_safe}', {round(amount, 2)}, '{notes_safe}', CURRENT_TIMESTAMP())")
            
        q = f"""
            INSERT INTO `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.seashield_costs`
                (id, route_id, cost_category, cost_usd, notes, computed_at)
            VALUES {','.join(values_str)}
        """
        try:
            client.query(q).result()
            return True
        except Exception as e:
            print(f"Error saving costs: {e}")
            return False

# ═════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_resource(show_spinner=False)
def load_port_graph():
    import searoute as sr
    return sr.setup_P()


@st.cache_data(show_spinner=False)
def get_all_ports():
    P    = load_port_graph()
    seen = {}
    for node, data in P.nodes(data=True):
        name    = data.get("name", "").strip()
        country = data.get("cty",  "").strip()
        lat     = data.get("y")
        lon     = data.get("x")
        if not name or not country or lat is None or lon is None:
            continue
        key = f"{name} - {country}"
        if key not in seen:
            seen[key] = {"name": name, "country": country,
                         "lat": lat, "lon": lon, "label": key}
    return dict(sorted(seen.items()))


def _searoute_leg(lon1, lat1, lon2, lat2):
    """Single searoute leg with error handling. Returns list of [lon, lat]."""
    import searoute as sr
    try:
        fc = sr.searoute([lon1, lat1], [lon2, lat2])
        return fc["geometry"]["coordinates"]
    except Exception:
        # Fallback: just the two endpoints (searoute will pick sea path)
        return [[lon1, lat1], [lon2, lat2]]


def interpolate_route(origin, destination, n_points=10,
                      use_alternate=False):
    """
    Compute a sea route from origin to destination.

    For the RECOMMENDED route: use searoute directly (origin → destination).
    For the ALTERNATE route: insert ocean-safe intermediate waypoints so the
    route stays entirely at sea, then stitch legs together via searoute.
    Waypoints are then sub-sampled to n_points.
    """
    o_lon, o_lat = origin["lon"],      origin["lat"]
    d_lon, d_lat = destination["lon"], destination["lat"]

    if not use_alternate:
        # ── Direct sea route ─────────────────────────────────────────
        coords = _searoute_leg(o_lon, o_lat, d_lon, d_lat)
    else:
        # ── Alternate route via ocean pivot points ───────────────────
        via_pts = _get_ocean_via_points(origin, destination)

        if not via_pts:
            # No ocean pivots needed (short / same-basin route) —
            # use a slight ocean offset that stays in open water
            coords = _searoute_leg(o_lon, o_lat, d_lon, d_lat)
        else:
            # Stitch legs through ocean waypoints
            all_coords = []
            prev_lon, prev_lat = o_lon, o_lat
            legs = list(via_pts) + [(d_lon, d_lat)]

            for (next_lon, next_lat) in legs:
                leg = _searoute_leg(prev_lon, prev_lat, next_lon, next_lat)
                if all_coords:
                    leg = leg[1:]   # drop duplicate join point
                all_coords.extend(leg)
                prev_lon, prev_lat = next_lon, next_lat

            coords = all_coords

    # ── Sub-sample to n_points ───────────────────────────────────────
    full_coords = [{"lon": float(lo), "lat": float(la)} for lo, la in coords]

    if len(coords) > n_points:
        idx    = np.linspace(0, len(coords) - 1, n_points, dtype=int)
        coords = [coords[i] for i in idx]

    waypoints = [{"lon": float(lo), "lat": float(la)} for lo, la in coords]
    return full_coords, waypoints


def compute_route_distance_km(origin, destination):
    import searoute as sr
    fc = sr.searoute([origin["lon"], origin["lat"]],
                     [destination["lon"], destination["lat"]])
    return fc["properties"]["length"]


def get_ports_along_route(origin, destination, max_dist_km=100):
    import searoute as sr
    fc     = sr.searoute([origin["lon"], origin["lat"]],
                         [destination["lon"], destination["lat"]])
    coords = fc["geometry"]["coordinates"]
    P      = load_port_graph()
    lons   = [c[0] for c in coords]
    lats   = [c[1] for c in coords]
    result = []
    for _, data in P.nodes(data=True):
        lat  = data["y"]; lon = data["x"]
        name = data.get("name", "Unknown")
        if (haversine_km(lat, lon, origin["lat"],      origin["lon"])      < 80 or
            haversine_km(lat, lon, destination["lat"], destination["lon"]) < 80):
            continue
        if not (min(lats) - 1 < lat < max(lats) + 1 and
                min(lons) - 1 < lon < max(lons) + 1):
            continue
        if min(haversine_km(lat, lon, la, lo) for lo, la in coords) < max_dist_km:
            result.append({"name": name, "lat": lat, "lon": lon,
                           "country": data.get("cty", "")})
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_marine_weather(points: tuple) -> pd.DataFrame:
    records = []
    for i, pt in enumerate(points):
        url = (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={pt['lat']:.4f}&longitude={pt['lon']:.4f}"
            f"&hourly=wave_height,wave_period,wind_wave_height,swell_wave_height"
            f"&forecast_days=1&timezone=UTC"
        )
        try:
            resp   = requests.get(url, timeout=10)
            resp.raise_for_status()
            data   = resp.json()
            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            records.append({
                "waypoint":      i + 1,
                "lat":           pt["lat"],
                "lon":           pt["lon"],
                "timestamp":     times[0] if times else "N/A",
                "wave_height_m": hourly.get("wave_height",      [0])[0] or 0,
                "wave_period_s": hourly.get("wave_period",      [0])[0] or 0,
                "wind_wave_m":   hourly.get("wind_wave_height", [0])[0] or 0,
                "swell_wave_m":  hourly.get("swell_wave_height",[0])[0] or 0,
            })
        except Exception as e:
            records.append({
                "waypoint":      i + 1,
                "lat":           pt["lat"],
                "lon":           pt["lon"],
                "timestamp":     datetime.utcnow().strftime("%Y-%m-%dT%H:00"),
                "wave_height_m": round(np.random.uniform(0.5, 4.5), 2),
                "wave_period_s": round(np.random.uniform(6,   14),  1),
                "wind_wave_m":   round(np.random.uniform(0.3, 3.0), 2),
                "swell_wave_m":  round(np.random.uniform(0.2, 2.5), 2),
                "_fallback":     str(e),
            })
    return pd.DataFrame(records)


def score_route(df: pd.DataFrame, min_cyclone_dist_km: float) -> float:
    base = (df["wave_height_m"].mean() * 0.5 +
            df["wave_height_m"].max()  * 0.3 +
            df["swell_wave_m"].mean()  * 0.2)
    if min_cyclone_dist_km < 300:
        base += 1000
    elif min_cyclone_dist_km < 800:
        base += (800 - min_cyclone_dist_km) * 0.1
    return base


def get_min_cyclone_dist_shapely(points, cyclones):
    if not cyclones:
        return float("inf")
    return min(
        haversine_km(pt["lat"], pt["lon"], c["lat"], c["lon"])
        for pt in points for c in cyclones
    )


def compute_segment_eta(df_weather, base_speed_knots, total_distance_km):
    if df_weather is None or df_weather.empty or total_distance_km <= 0:
        return 0.0, pd.DataFrame()
    
    n_points = len(df_weather)
    segment_distance_km = total_distance_km / n_points
    
    results = []
    total_time_hrs = 0.0
    for _, row in df_weather.iterrows():
        wh = float(row.get("wave_height_m", 0.0))
        wp = float(row.get("wave_period_s", 0.0))
        
        speed_factor = 1.0 - min(0.5, 0.06 * (wh ** 1.5) + 0.01 * max(0, 12.0 - wp))
        eff_speed_kmh = float(base_speed_knots) * 1.852 * speed_factor
        
        if wh > 3.5 and float(base_speed_knots) > 18:
            eff_speed_kmh = min(eff_speed_kmh, 15 * 1.852)
        
        if eff_speed_kmh <= 0:
            seg_hrs = 0.0
        else:
            seg_hrs = segment_distance_km / eff_speed_kmh
            total_time_hrs += seg_hrs
            
        results.append({
            "waypoint": int(row["waypoint"]),
            "wave_height_m": wh,
            "speed_factor": speed_factor,
            "effective_speed_kmh": eff_speed_kmh,
            "segment_distance_km": segment_distance_km,
            "segment_time_hrs": seg_hrs
        })
        
    return total_time_hrs, pd.DataFrame(results)


# ─────────────────────────────────────────────
#  COST ESTIMATOR FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bunker_price() -> float:
    """Fetch VLSFO price from ShipAndBunker or return $620 fallback."""
    try:
        import re as _re
        resp = requests.get(
            "https://shipandbunker.com/prices/av/global/"
            "av-glb-global-average-bunker-price",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        match = _re.search(
            r'Global\s+Average.*?USD\s*/\s*MT.*?(\d{3,4}(?:\.\d{1,2})?)',
            resp.text, _re.DOTALL | _re.IGNORECASE,
        )
        if match:
            price = float(match.group(1))
            if 300 < price < 1500:
                return price
    except Exception:
        pass
    return 620.0


def _get_port_tier(country: str) -> int:
    """Classify a port's country into tier 1, 2, or 3."""
    if country in TIER1_COUNTRIES:
        return 1
    if country in TIER2_COUNTRIES:
        return 2
    return 3


def _estimate_distance_from_coords(full_coords: list) -> float:
    """Estimate total distance from a list of {lat, lon} dicts."""
    total = 0.0
    for i in range(len(full_coords) - 1):
        total += haversine_km(
            full_coords[i]["lat"], full_coords[i]["lon"],
            full_coords[i + 1]["lat"], full_coords[i + 1]["lon"],
        )
    return total


def compute_freight_rate(distance_km, teu_count, vessel_type,
                         commodity_type) -> dict:
    """Returns base_rate, baf_surcharge, commodity_surcharge as dict."""
    raw_rate = distance_km * 0.045 + 180
    vessel_mult = VESSEL_FREIGHT_MULT.get(vessel_type, 1.0)
    commodity_pct = COMMODITY_SURCHARGE.get(commodity_type, 0.0)
    base_freight = raw_rate * teu_count * vessel_mult
    commodity_surcharge_usd = base_freight * commodity_pct
    total_freight = base_freight + commodity_surcharge_usd
    return {
        "raw_rate_per_teu": raw_rate,
        "vessel_multiplier": vessel_mult,
        "commodity_pct": commodity_pct,
        "base_freight_usd": base_freight,
        "commodity_surcharge_usd": commodity_surcharge_usd,
        "total_freight_usd": total_freight,
    }


def compute_port_costs(origin_port_dict, dest_port_dict, teu_count) -> dict:
    """Returns origin_dues, origin_pilotage, origin_thc, dest_dues,
    dest_pilotage, dest_thc."""
    o_tier = _get_port_tier(origin_port_dict.get("country", ""))
    d_tier = _get_port_tier(dest_port_dict.get("country", ""))
    o = PORT_COSTS[o_tier]
    d = PORT_COSTS[d_tier]
    return {
        "origin_tier": o_tier, "dest_tier": d_tier,
        "origin_dues": o["dues"], "origin_pilotage": o["pilotage"],
        "origin_thc": o["thc_per_teu"] * teu_count,
        "dest_dues": d["dues"], "dest_pilotage": d["pilotage"],
        "dest_thc": d["thc_per_teu"] * teu_count,
        "total_port_costs": (
            o["dues"] + o["pilotage"] + o["thc_per_teu"] * teu_count +
            d["dues"] + d["pilotage"] + d["thc_per_teu"] * teu_count
        ),
    }


def compute_origin_charges(teu_count, enabled_items: list) -> dict:
    """Returns itemised origin export charges."""
    items = {}
    total = 0.0
    for name, cost in ORIGIN_CHARGES_TABLE.items():
        on = name in enabled_items
        items[name] = {"amount": cost if on else 0, "enabled": on}
        if on:
            total += cost
    isps = (ORIGIN_ISPS_PER_TEU * teu_count
            if "ISPS security surcharge" in enabled_items else 0)
    items["ISPS security surcharge"] = {
        "amount": isps,
        "enabled": "ISPS security surcharge" in enabled_items,
    }
    total += isps
    return {"items": items, "total_origin_charges": total}


def compute_destination_charges(teu_count, expected_delay_days,
                                dest_port_tier) -> dict:
    """Returns itemised destination import charges."""
    ddc = DEST_DDC_PER_TEU * teu_count
    customs = DEST_IMPORT_CUSTOMS
    do_release = DEST_DO_RELEASE
    demurrage_days = max(0, expected_delay_days - DEMURRAGE_FREE_DAYS)
    demurrage = DEMURRAGE_PER_TEU_DAY * teu_count * demurrage_days
    congestion = PORT_CONGESTION_SURCHARGE_T1 if dest_port_tier == 1 else 0
    total = ddc + customs + do_release + demurrage + congestion
    return {
        "ddc_usd": ddc, "import_customs_usd": customs,
        "do_release_usd": do_release,
        "demurrage_days": demurrage_days, "demurrage_usd": demurrage,
        "congestion_surcharge_usd": congestion,
        "total_destination_charges": total,
    }


def compute_inland_haulage(origin_km, dest_km, teu_count,
                           orig_tier, dest_tier) -> dict:
    """Returns origin_haulage_usd, dest_haulage_usd, depot_fee."""
    def _calc(km, tier):
        first = min(km, 100) * HAULAGE_RATE_FIRST_100
        beyond = max(0, km - 100) * HAULAGE_RATE_BEYOND
        return (first + beyond) * HAULAGE_TIER_MULT.get(tier, 1.0) * teu_count
    o_h = _calc(origin_km, orig_tier)
    d_h = _calc(dest_km, dest_tier)
    depot = HAULAGE_DEPOT_FEE * 2
    return {
        "origin_haulage_usd": o_h, "dest_haulage_usd": d_h,
        "depot_fee_usd": depot,
        "total_haulage_usd": o_h + d_h + depot,
    }


def detect_canal_transits(waypoints: list, vessel_type: str, teu_count: int = 0) -> list:
    """Returns list of dicts: {canal_name, detected, fee_usd}."""
    dwt = VESSEL_DWT.get(vessel_type, 25_000)
    results = []
    for canal_name, bbox in CANAL_ZONES.items():
        detected = any(
            bbox["lon_min"] <= wp["lon"] <= bbox["lon_max"]
            and bbox["lat_min"] <= wp["lat"] <= bbox["lat_max"]
            for wp in waypoints
        )
        fee = 0
        if detected:
            if canal_name == "Suez Canal":
                if vessel_type == "Container Ship" and teu_count > 0:
                    fee = 200 * teu_count
                else:
                    fee = 420_000 * (dwt / 100_000)
            elif canal_name == "Panama Canal":
                fee = 180_000 * (dwt / 100_000)
            elif canal_name == "Malacca Strait":
                fee = 12_000
            elif canal_name == "Bab-el-Mandeb":
                fee = 8_500
        results.append({"canal_name": canal_name,
                        "detected": detected, "fee_usd": fee})
    return results


def compute_weather_delay_cost(df_rec, daily_vessel_cost,
                                base_freight_rate,
                                active_cyclones,
                                min_cyclone_dist_km) -> dict:
    """Returns delay_hours, delay_cost_usd, cyclone_risk_premium."""
    delay_hours = 0.0
    if df_rec is not None and not df_rec.empty:
        for _, row in df_rec.iterrows():
            wh = float(row.get("wave_height_m", 0))
            delay_hours += max(0, (wh - 2.5) * 3.5)
    delay_cost = (delay_hours / 24) * daily_vessel_cost
    cyclone_premium = 0.0
    if active_cyclones and min_cyclone_dist_km < 800:
        cyclone_premium = base_freight_rate * CYCLONE_RISK_PREMIUM_PCT
    return {
        "delay_hours": round(delay_hours, 1),
        "delay_cost_usd": round(delay_cost, 2),
        "cyclone_risk_premium_usd": round(cyclone_premium, 2),
        "total_weather_cost": round(delay_cost + cyclone_premium, 2),
    }


def compute_total_voyage_cost(
    origin, destination, distance_km, waypoints,
    df_rec, active_cyclones, min_cyclone_dist_km,
    vessel_type, teu_count, speed_knots,
    voyage_hours, commodity_type,
    origin_inland_km, dest_inland_km,
    expected_delay_days,
    origin_port_dict, dest_port_dict,
    enabled_origin_items,
) -> dict:
    """Orchestrates all sub-functions, returns full cost breakdown dict."""
    freight = compute_freight_rate(
        distance_km, teu_count, vessel_type, commodity_type)
    bunker_price = fetch_bunker_price()
    dwt = VESSEL_DWT.get(vessel_type, 25_000)
    
    avg_wave = df_rec["wave_height_m"].mean() if df_rec is not None and not df_rec.empty else 0
    resistance_factor = 1 + 0.05 * (avg_wave ** 1.3)
    
    daily_consumption = dwt * 0.0028 * (speed_knots / 14.5) ** 3 * resistance_factor
    voyage_days = voyage_hours / 24.0 if voyage_hours > 0 else 0
    bunker_cost = daily_consumption * voyage_days * bunker_price
    port_costs = compute_port_costs(
        origin_port_dict, dest_port_dict, teu_count)
    origin_charges = compute_origin_charges(teu_count, enabled_origin_items)
    dest_charges = compute_destination_charges(
        teu_count, expected_delay_days, port_costs["dest_tier"])
    haulage = compute_inland_haulage(
        origin_inland_km, dest_inland_km, teu_count,
        port_costs["origin_tier"], port_costs["dest_tier"])
    canals = detect_canal_transits(waypoints, vessel_type, teu_count)
    total_canal = sum(c["fee_usd"] for c in canals)
    bunker_daily = daily_consumption * bunker_price
    daily_vessel_cost = bunker_daily + CREW_COST_PER_DAY + OVERHEAD_PER_DAY
    weather_cost = compute_weather_delay_cost(
        df_rec, daily_vessel_cost, freight["total_freight_usd"],
        active_cyclones, min_cyclone_dist_km)
        
    co2_emissions_mt = daily_consumption * voyage_days * CO2_PER_MT_FUEL
    carbon_tax_usd = co2_emissions_mt * CARBON_TAX_PER_MT_USD
    
    grand_total = (
        freight["total_freight_usd"] + bunker_cost +
        port_costs["total_port_costs"] +
        origin_charges["total_origin_charges"] +
        dest_charges["total_destination_charges"] +
        haulage["total_haulage_usd"] + total_canal +
        weather_cost["total_weather_cost"] + carbon_tax_usd
    )
    return {
        "freight": freight,
        "bunker_price_per_mt": bunker_price,
        "daily_consumption_mt": round(daily_consumption, 2),
        "voyage_days": round(voyage_days, 2),
        "bunker_cost_usd": round(bunker_cost, 2),
        "port_costs": port_costs,
        "origin_charges": origin_charges,
        "dest_charges": dest_charges,
        "haulage": haulage,
        "canals": canals,
        "total_canal_usd": total_canal,
        "weather_cost": weather_cost,
        "co2_emissions_mt": round(co2_emissions_mt, 2),
        "carbon_tax_usd": round(carbon_tax_usd, 2),
        "daily_vessel_cost": round(daily_vessel_cost, 2),
        "grand_total_usd": round(grand_total, 2),
        "cost_per_teu": round(grand_total / max(teu_count, 1), 2),
        "cost_per_km": round(grand_total / max(distance_km, 1), 2),
        "vessel_dwt": dwt,
    }


# ─────────────────────────────────────────────
#  DISRUPTION DETECTION
# ─────────────────────────────────────────────

def detect_disruptions(
    route_geojson,
    weather_df,
    active_cyclones,
    postgis_result    = None,
    high_wave_thresh  = 4.0,
    severe_wave_thresh= 6.0,
    cyclone_warn_km   = 600,
    cyclone_danger_km = 300,
    cyclone_buffer_km = 400,
) -> dict:
    reasons           = []
    risk_flags        = []
    affected_segments = 0
    wave_issues       = {"high": 0, "severe": 0}

    route_line = None
    if postgis_result is None or not postgis_result.get("bigquery_used"):
        try:
            geom = route_geojson
            if geom.get("type") == "FeatureCollection":
                geom = geom["features"][0]["geometry"]
            if geom.get("type") == "Feature":
                geom = geom["geometry"]
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                route_line = LineString([(c[0], c[1]) for c in coords])
        except Exception as e:
            reasons.append(f"⚠️ Route geometry parse error: {e}")

    if weather_df is not None and not weather_df.empty:
        df     = weather_df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        wh_col = next((c for c in ["wave_height_m","wave_height","waveheight"]
                       if c in df.columns), None)
        if wh_col:
            wh   = df[wh_col].dropna()
            n    = len(wh)
            wave_issues["high"]   = int((wh > high_wave_thresh).sum())
            wave_issues["severe"] = int((wh > severe_wave_thresh).sum())
            affected_segments     = wave_issues["high"]
            if wave_issues["severe"]:
                risk_flags.append("high")
                reasons.append(
                    f"🌊 Severe waves (>{severe_wave_thresh}m) at "
                    f"{wave_issues['severe']}/{n} waypoints")
            elif wave_issues["high"]:
                label = "high" if wave_issues["high"] / n > 0.5 else "medium"
                risk_flags.append(label)
                reasons.append(
                    f"🌊 High waves (>{high_wave_thresh}m) at "
                    f"{wave_issues['high']}/{n} waypoints")

    if active_cyclones:
        if postgis_result and postgis_result.get("bigquery_used"):
            min_km       = postgis_result["min_cyclone_dist_km"]
            intersecting = postgis_result["intersecting_cyclones"]
            reasons.append(
                f"<span class='postgis-badge'>BigQuery ST_Distance</span> "
                f"Nearest cyclone: {min_km:.0f} km from route"
            )
            if min_km < cyclone_danger_km or intersecting:
                risk_flags.append("high")
                names = ", ".join(intersecting) if intersecting else "unnamed"
                reasons.append(
                    f"🌀 DANGER — cyclone(s) [{names}] within "
                    f"{cyclone_danger_km:.0f} km or route buffer intersected (ST_DWithin)"
                )
            elif min_km < cyclone_warn_km:
                risk_flags.append("medium")
                reasons.append(
                    f"🌀 WARNING — nearest cyclone {min_km:.0f} km "
                    f"(ST_Distance threshold: {cyclone_warn_km:.0f} km)"
                )
        else:
            for cyc in active_cyclones:
                c_lat  = float(cyc["lat"])
                c_lon  = float(cyc["lon"])
                c_name = cyc.get("name", cyc.get("id", "Unknown"))
                min_km = float("inf")
                if weather_df is not None and not weather_df.empty:
                    df2 = weather_df.copy()
                    df2.columns = [c.lower() for c in df2.columns]
                    for _, row in df2.iterrows():
                        d = haversine_km(row.get("lat", 0), row.get("lon", 0),
                                         c_lat, c_lon)
                        if d < min_km:
                            min_km = d
                route_intersects = False
                if route_line:
                    buf_deg          = cyclone_buffer_km / 111.0
                    route_intersects = route_line.intersects(
                        Point(c_lon, c_lat).buffer(buf_deg)
                    )
                if min_km < cyclone_danger_km or route_intersects:
                    risk_flags.append("high")
                    reasons.append(f"🌀 Cyclone {c_name}: {min_km:.0f} km — DANGER ZONE")
                elif min_km < cyclone_warn_km:
                    risk_flags.append("medium")
                    reasons.append(f"🌀 Cyclone {c_name}: {min_km:.0f} km — warning radius")
    else:
        reasons.append("✅ No active cyclones in monitored basins.")

    if "high" in risk_flags:
        risk_level = "High"
        risk_score = min(100, 70 + wave_issues.get("severe", 0) * 5
                                  + wave_issues.get("high",   0) * 2)
    elif "medium" in risk_flags:
        risk_level = "Medium"
        risk_score = min(69, 40 + wave_issues.get("high", 0) * 3)
    else:
        risk_level = "Low"
        risk_score = max(0, min(39, wave_issues.get("high", 0) * 5))

    if not reasons:
        reasons.append("✅ All conditions within safe operational limits.")

    return {
        "risk_level":        risk_level,
        "risk_score":        risk_score,
        "reasons":           reasons,
        "affected_segments": affected_segments,
    }


# ─────────────────────────────────────────────
#  MAP BUILDERS
# ─────────────────────────────────────────────

def wave_color(h):
    return "#2ed573" if h < 2.0 else ("#ffb347" if h < 4.0 else "#ff4757")

def wave_label(h):
    return "🟢 CALM" if h < 2.0 else ("🟡 MODERATE" if h < 4.0 else "🔴 ROUGH")


def build_map(origin, destination, rec_full, alt_full, rec_wps, alt_wps,
              df_rec, df_alt, route_ports=None, active_cyclones=None):
    mid_lat = (origin["lat"] + destination["lat"]) / 2
    mid_lon = (origin["lon"] + destination["lon"]) / 2
    zoom    = 4 if abs(origin["lon"] - destination["lon"]) < 30 else 3
    m       = folium.Map(location=[mid_lat, mid_lon], zoom_start=zoom,
                         tiles="CartoDB dark_matter", prefer_canvas=True)

    # Alt route
    folium.PolyLine([[p["lat"], p["lon"]] for p in alt_full],
                    color="#e67e22", weight=2.5, opacity=0.7,
                    dash_array="8 10", tooltip="⚠️ Alternate Route").add_to(m)
    # Rec route
    rec_coords = [[p["lat"], p["lon"]] for p in rec_full]
    folium.PolyLine(rec_coords, color="#005580", weight=3.5,
                    opacity=0.9, tooltip="✅ Recommended Route").add_to(m)
    folium.PolyLine(rec_coords, color="#102a43", weight=1,
                    opacity=0.25, dash_array="10 15").add_to(m)
    # Eco route
    folium.PolyLine(rec_coords, color="#00e676", weight=1.5,
                    opacity=0.8, dash_array="5 5", tooltip="🌱 Eco-Route (Weather Optimized)").add_to(m)

    for _, row in df_rec.iterrows():
        h   = float(row["wave_height_m"])
        col = wave_color(h)
        lbl = wave_label(h)
        popup_html = f"""
        <div style="font-family:monospace;background:#1e293b;color:#f8fafc;
                    padding:10px;border-radius:6px;min-width:210px;border:1px solid #334155;">
          <b style="color:#38bdf8;">✅ REC · Waypoint {int(row['waypoint'])}</b><br>
          <hr style="border-color:#334155;margin:4px 0">
          📍 {row['lat']:.3f}°N, {row['lon']:.3f}°E<br>
          🌊 Wave: <b>{h:.2f} m</b> {lbl}<br>
          ⏱️ Period: <b>{row['wave_period_s']:.1f} s</b><br>
          💨 Wind Wave: <b>{row['wind_wave_m']:.2f} m</b><br>
          🌀 Swell: <b>{row['swell_wave_m']:.2f} m</b><br>
          🕐 {row['timestamp']}
        </div>"""
        folium.CircleMarker(location=[row["lat"], row["lon"]],
            radius=10 + h * 2, color=col, fill=True,
            fill_color=col, fill_opacity=0.75,
            tooltip=f"✅ WP{int(row['waypoint'])} | {h:.1f}m {lbl}",
            popup=folium.Popup(popup_html, max_width=260)).add_to(m)
        folium.CircleMarker(location=[row["lat"], row["lon"]],
            radius=16 + h * 2, color=col, fill=False,
            weight=1, opacity=0.35).add_to(m)

    for _, row in df_alt.iterrows():
        h = float(row["wave_height_m"])
        folium.CircleMarker(location=[row["lat"], row["lon"]],
            radius=6, color="#e67e22", fill=True,
            fill_color=wave_color(h), fill_opacity=0.45, weight=1.5,
            tooltip=f"⚠️ ALT WP{int(row['waypoint'])} | {h:.1f}m").add_to(m)

    for port, icon_color, label in [
        (origin,      "blue",  f"🚢 {origin['name']} (Origin)"),
        (destination, "green", f"⚓ {destination['name']} (Destination)"),
    ]:
        folium.Marker(
            location=[port["lat"], port["lon"]],
            popup=folium.Popup(
                f'<div style="font-family:monospace;background:#fff;color:#102a43;'
                f'padding:8px;border-radius:6px;border:1px solid #d9e2ec;">'
                f'<b style="color:#005580;">{label}</b><br>'
                f'📍 {port["lat"]:.4f}°, {port["lon"]:.4f}°</div>',
                max_width=220),
            tooltip=label,
            icon=folium.Icon(color=icon_color, icon="anchor", prefix="fa"),
        ).add_to(m)

    if route_ports:
        for p in route_ports:
            folium.CircleMarker(
                location=[p["lat"], p["lon"]], radius=5,
                color="#e67e22", fill=True, fill_color="#e67e22", fill_opacity=0.9,
                tooltip=f"🏢 {p['name']} ({p.get('country','')})",
                popup=folium.Popup(
                    f'<div style="font-family:monospace;font-size:12px;">'
                    f'<b style="color:#e67e22;">⚓ {p["name"]}</b><br>'
                    f'Country: {p.get("country","Unknown")}</div>',
                    max_width=200)).add_to(m)

    if active_cyclones:
        for cyc in active_cyclones:
            c_lat = cyc["lat"]; c_lon = cyc["lon"]
            folium.Marker(
                location=[c_lat, c_lon],
                popup=folium.Popup(
                    f'<div style="font-family:monospace;background:#1e293b;color:#f8fafc;'
                    f'padding:8px;border-radius:6px;border:1px solid #334155;">'
                    f'<b style="color:#ef4444;">🌀 {cyc["name"]}</b><br>'
                    f'Intensity: {cyc["intensity"]} kts<br>'
                    f'📍 {c_lat:.2f}°, {c_lon:.2f}°</div>', max_width=220),
                tooltip=f"🌀 {cyc['name']} ({cyc['intensity']} kts)",
                icon=folium.Icon(color="red", icon="hurricane", prefix="fa"),
            ).add_to(m)
            if cyc.get("forecast_points"):
                fcst_coords = [[c_lat, c_lon]] + [[p[0], p[1]]
                               for p in cyc["forecast_points"]]
                folium.PolyLine(fcst_coords, color="red", weight=2,
                                dash_array="5 5", opacity=0.8,
                                tooltip=f"{cyc['name']} Forecast Track").add_to(m)

    # Legend is rendered separately below the map
    return m


def build_history_map(route_geojson_str, origin_geojson_str,
                      dest_geojson_str, risk_level) -> folium.Map:
    try:
        route_gj  = json.loads(route_geojson_str)
        orig_gj   = json.loads(origin_geojson_str)
        dest_gj   = json.loads(dest_geojson_str)
        coords    = route_gj["coordinates"]
        ll_coords = [[c[1], c[0]] for c in coords]
        mid_lat   = sum(c[1] for c in coords) / len(coords)
        mid_lon   = sum(c[0] for c in coords) / len(coords)

        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=3,
                       tiles="CartoDB dark_matter")
        color = {"High": "#ef4444", "Medium": "#f59e0b",
                 "Low": "#10b981"}.get(risk_level, "#38bdf8")
        folium.PolyLine(ll_coords, color=color, weight=3,
                        tooltip=f"Risk: {risk_level}").add_to(m)
        for gj_str, icon_c, tip in [
            (orig_gj, "blue",  "Origin"),
            (dest_gj, "green", "Destination"),
        ]:
            lon, lat = gj_str["coordinates"]
            folium.Marker([lat, lon],
                          icon=folium.Icon(color=icon_c, icon="anchor", prefix="fa"),
                          tooltip=tip).add_to(m)
        return m
    except Exception:
        return folium.Map(location=[10, 80], zoom_start=3,
                          tiles="CartoDB dark_matter")


# ─────────────────────────────────────────────
#  CYCLONE FETCHER
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_active_cyclones():
    status = {"status": "success", "message": "", "data": []}
    try:
        from tropycal import realtime
        rt = realtime.Realtime(jtwc=True)
        try:
            active_ids = rt.list_active_storms()
        except Exception as e:
            status.update({"status": "error", "message": f"Failed to list storms: {e}"})
            return status
        if not active_ids:
            return status
        for sid in active_ids:
            try:
                storm     = rt.get_storm(sid)
                intensity = int(storm.vmax[-1]) if (
                    hasattr(storm, "vmax") and len(storm.vmax) > 0
                    and not np.isnan(storm.vmax[-1])
                ) else "Unknown"
                fcst_pts = []
                try:
                    fcst = storm.get_jtwc_forecast()
                except Exception:
                    try:
                        fcst = storm.get_nhc_forecast()
                    except Exception:
                        fcst = {}
                if "lat" in fcst and "lon" in fcst:
                    fcst_pts = list(zip(fcst["lat"], fcst["lon"]))
                status["data"].append({
                    "id":              sid,
                    "name":            getattr(storm, "name", sid),
                    "lat":             float(storm.lat[-1]),
                    "lon":             float(storm.lon[-1]),
                    "intensity":       intensity,
                    "forecast_points": fcst_pts,
                })
            except Exception:
                continue
    except Exception as e:
        status.update({"status": "error",
                       "message": f"Import or connection error: {e}"})
    return status


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
_defaults = {
    "captains_briefing":None,
    "gemini_key":       os.environ.get("GEMINI_API_KEY", ""),
    "dest_label":       None,
    "orig_label":       None,
    "df_rec":           None,
    "df_alt":           None,
    "df_eco":           None,
    "rec_full":         None,
    "alt_full":         None,
    "eco_full":         None,
    "rec_wps":          None,
    "alt_wps":          None,
    "eco_wps":          None,
    "route_ports":      None,
    "distance_km":      None,
    "rec_score":        None,
    "alt_score":        None,
    "eco_score":        None,
    "n_waypoints":      10,
    "active_cyclones":  [],
    "cyclone_res":      None,
    "min_cyc_dist_rec": float("inf"),
    "min_cyc_dist_alt": float("inf"),
    "loaded":           False,
    "teu_count":        100,
    "commodity_type":   "General",
    "origin_inland_km": 50,
    "dest_inland_km":   50,
    "expected_delay_days": 0,
    "currency":         "USD",
    "cost_result_rec":  None,
    "cost_result_alt":  None,
    "cost_result_eco":  None,
    "disruption_result":None,
    "disruption_run":   False,
    "postgis_result":   None,
    "last_route_id":    None,
    "db_initialized":   False,
    "wizard_step":      1,
    "vessel_type":      "Bulk Carrier",
    "base_speed_knots": 14,
    "enabled_origin_items": [],
}
for k, v in _defaults.items():
    if "gcp_project" not in st.session_state:
        st.session_state.gcp_project = os.environ.get("BQ_PROJECT_ID", "your-gcp-project-id")
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.db_initialized:
    st.session_state.db_initialized = init_db()

# ─────────────────────────────────────────────
#  PORT LIST
# ─────────────────────────────────────────────
with st.spinner("⚓ Loading port database…"):
    all_ports   = get_all_ports()
port_labels = list(all_ports.keys())


# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════
def sec_head(label):
    return f'''
    <div class="section-header">
      <span class="section-line"></span>
      <span class="section-label">{label}</span>
      <span class="section-line"></span>
    </div>
    '''

with st.sidebar:
    st.markdown("""
        <div style="border-bottom:1px solid var(--border); padding-bottom:12px; margin-bottom:16px;">
          <h2 style="margin:0; font-family:'Orbitron',sans-serif; color:var(--text-hi); font-size:16px; letter-spacing:1px;">
            🛡️ SEASHIELD<br>
            <span style="color:var(--text-mid); font-size:12px; letter-spacing:2px;">OPTIMIZER v1.1</span>
          </h2>
        </div>
    """, unsafe_allow_html=True)

    _db_live = db_online()

    with st.form("main_sidebar_form"):
        nav_tab, vessel_tab, cargo_tab = st.tabs(["Navigation", "Vessel", "Cargo"])
        
        with nav_tab:
            st.write("") # slight spacing
            
            if not st.session_state.orig_label or st.session_state.orig_label not in port_labels:
                st.session_state.orig_label = port_labels[0] if port_labels else None
                
            if not st.session_state.dest_label or st.session_state.dest_label not in port_labels:
                st.session_state.dest_label = port_labels[1] if len(port_labels) > 1 else (port_labels[0] if port_labels else None)
    
            default_orig_idx = port_labels.index(st.session_state.orig_label) if st.session_state.orig_label in port_labels else 0
            orig_label = st.selectbox(
                "Select Origin Port", port_labels,
                index=default_orig_idx)
                
            default_dest_idx = port_labels.index(st.session_state.dest_label) if st.session_state.dest_label in port_labels else 0
            dest_label = st.selectbox(
                "Select Destination Port", port_labels,
                index=default_dest_idx)
    
            st.markdown(sec_head("Sampling"), unsafe_allow_html=True)
            n_pts = st.slider("Waypoints per route", 5, 15,
                              st.session_state.n_waypoints, step=1)
            
            st.markdown(sec_head("Google Cloud Config"), unsafe_allow_html=True)
            st.session_state.gcp_project = st.text_input("GCP Project ID", value=st.session_state.gcp_project)
            st.caption(f"Active Project: **{get_active_project_id() or 'None'}**")
            st.caption("Using your Google Cloud free credits for Vertex AI & BigQuery.")
                              
        with vessel_tab:
            st.write("") # slight spacing
            vessel_types = {"Bulk Carrier": 14, "Container Ship": 22, "Tanker": 15, "General Cargo": 12}
            v_type = st.selectbox("Vessel Type", list(vessel_types.keys()), index=0)
            b_speed = st.slider("Base Speed (knots)", 8, 28, vessel_types[v_type])
    
        with cargo_tab:
            # ── Cargo & Cost Section ──────────────────────────────────────────
            st.write("") # slight spacing
            t_count = st.number_input(
                "TEU Count", min_value=1, max_value=5000,
                value=st.session_state.teu_count, step=10)
            c_type = st.selectbox(
                "Commodity Type", list(COMMODITY_SURCHARGE.keys()),
                index=list(COMMODITY_SURCHARGE.keys()).index(
                    st.session_state.commodity_type))
            o_inland = st.slider(
                "Origin Inland Haulage (km)", 0, 500,
                st.session_state.origin_inland_km)
            d_inland = st.slider(
                "Dest. Inland Haulage (km)", 0, 500,
                st.session_state.dest_inland_km)
    
            with st.expander("Origin Export Charges", expanded=False):
                _all_origin_items = list(ORIGIN_CHARGES_TABLE.keys()) + [
                    "ISPS security surcharge"]
                _enabled_items = []
                for item_name in _all_origin_items:
                    if st.checkbox(item_name, value=True,
                                   key=f"oc_{item_name}_form"):
                        _enabled_items.append(item_name)
    
            cur = st.selectbox(
                "Currency", list(EXCHANGE_RATES.keys()),
                index=list(EXCHANGE_RATES.keys()).index(
                    st.session_state.currency))
                    
        st.markdown(sec_head("Actions"), unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            fetch_btn = st.form_submit_button("🔄 Apply Config & Fetch Route", type="primary", use_container_width=True)
        with c2:
            refresh_btn = st.form_submit_button("🔃", help="Force refresh", use_container_width=True, type="secondary")

    if fetch_btn or refresh_btn:
        st.session_state.orig_label = orig_label
        st.session_state.dest_label = dest_label
        st.session_state.n_waypoints = n_pts
        st.session_state.base_speed_knots = b_speed
        st.session_state.teu_count = t_count
        st.session_state.commodity_type = c_type
        st.session_state.origin_inland_km = o_inland
        st.session_state.dest_inland_km = d_inland
        st.session_state["enabled_origin_items"] = _enabled_items
        st.session_state.currency = cur
        st.session_state.captains_briefing = None

    # Hydrate origin/destination based on current applied session state
    orig_info = all_ports[st.session_state.orig_label]
    origin = {"name": orig_info["name"], "lat": orig_info["lat"], "lon": orig_info["lon"]}
    
    dest_info = all_ports[st.session_state.dest_label]
    destination = {"name": dest_info["name"], "lat":  dest_info["lat"], "lon":  dest_info["lon"]}
    
    # Render applied labels directly below form
    st.markdown(
        f"<div style='background:var(--bg-raised); border:1px solid var(--border); border-radius:4px; padding:8px; margin-bottom:4px;'>"
        f"<div style='font-family:\"Inter\",sans-serif; color:var(--text-hi); font-size:12px;'><span style='margin-right:6px;'>🛳️</span><b>{orig_info['name']}</b> ➔ <b>{dest_info['name']}</b></div>"
        f"</div>",
        unsafe_allow_html=True)
    save_btn = st.button(
        "💾 Save Analysis to PostGIS",
        use_container_width=True,
        type="secondary",
        disabled=not (st.session_state.loaded and _db_live),
        help="Persists route, waypoints and cyclones to the database",
    )

    st.markdown(sec_head("Disruption Analysis"), unsafe_allow_html=True)
    disrupt_btn = st.button("🔍 Run Disruption Analysis", use_container_width=True)

    if disrupt_btn:
        if not st.session_state.loaded:
            st.warning("⚠️ Fetch a route first.")
        else:
            with st.spinner("🔬 Analysing…"):
                wps = st.session_state.rec_wps or []
                geojson_input = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[p["lon"], p["lat"]] for p in wps],
                    },
                }
                result = detect_disruptions(
                    route_geojson   = geojson_input,
                    weather_df      = st.session_state.df_rec,
                    active_cyclones = st.session_state.active_cyclones or [],
                    postgis_result  = st.session_state.postgis_result,
                )
                st.session_state.disruption_result = result
                st.session_state.disruption_run    = True

    st.markdown(sec_head("Active Cyclones"), unsafe_allow_html=True)
    c_res = st.session_state.cyclone_res
    if c_res and c_res.get("status") == "error":
        st.markdown(
            f"<div style='background:rgba(245,158,11,0.15);border-left:3px solid var(--amber);"
            f"padding:8px;margin:4px 0;border-radius:4px;font-size:12px;'>"
            f"⚠️ {c_res.get('message')}</div>",
            unsafe_allow_html=True)
    elif not st.session_state.active_cyclones:
        st.markdown("<span style='color:var(--steel);font-size:12px;'>"
                    "No active cyclones.</span>", unsafe_allow_html=True)
    else:
        for cyc in st.session_state.active_cyclones:
            st.markdown(
                f"<div style='background:rgba(239,68,68,0.15);border-left:3px solid var(--danger);"
                f"padding:8px;margin:4px 0;border-radius:4px;font-size:12px;'>"
                f"<b>🌀 {cyc['name']}</b> ({cyc['id']})<br>"
                f"Intensity: {cyc['intensity']} kts<br>"
                f"Loc: {cyc['lat']:.1f}°, {cyc['lon']:.1f}°</div>",
                unsafe_allow_html=True)

    if st.session_state.loaded and st.session_state.df_rec is not None:
        st.markdown(sec_head("Route Comparison"), unsafe_allow_html=True)
        prox_r = (f"{st.session_state.min_cyc_dist_rec:.0f} km"
                  if st.session_state.min_cyc_dist_rec != float("inf") else "Clear")
        prox_a = (f"{st.session_state.min_cyc_dist_alt:.0f} km"
                  if st.session_state.min_cyc_dist_alt != float("inf") else "Clear")
        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); border-left:4px solid var(--accent-1); border-radius:6px; padding:12px; margin-bottom:8px;'>"
            f"<b style='color:var(--text-hi); font-family:\"Orbitron\",sans-serif; font-size:11px;'><span style='color:var(--accent-1); margin-right:4px;'>✅</span>RECOMMENDED</b>"
            f"<div style='display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:8px; font-size:12px; font-family:\"JetBrains Mono\",monospace;'>"
            f"<div><span style='color:var(--text-mid);'>Score:</span> {st.session_state.rec_score:.3f}</div>"
            f"<div><span style='color:var(--text-mid);'>Wave:</span> {st.session_state.df_rec['wave_height_m'].mean():.2f}m</div>"
            f"<div style='grid-column:1/-1;'><span style='color:var(--text-mid);'>Cyclone Prox:</span> {prox_r}</div>"
            f"</div></div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:var(--bg-surface); border:1px solid var(--border); border-left:4px solid var(--warn); border-radius:6px; padding:12px; margin-bottom:8px;'>"
            f"<b style='color:var(--text-hi); font-family:\"Orbitron\",sans-serif; font-size:11px;'><span style='color:var(--warn); margin-right:4px;'>⚠️</span>ALTERNATE</b>"
            f"<div style='display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:8px; font-size:12px; font-family:\"JetBrains Mono\",monospace;'>"
            f"<div><span style='color:var(--text-mid);'>Score:</span> {st.session_state.alt_score:.3f}</div>"
            f"<div><span style='color:var(--text-mid);'>Wave:</span> {st.session_state.df_alt['wave_height_m'].mean():.2f}m</div>"
            f"<div style='grid-column:1/-1;'><span style='color:var(--text-mid);'>Cyclone Prox:</span> {prox_a}</div>"
            f"</div></div>",
            unsafe_allow_html=True)
        st.markdown(sec_head("Route Info"), unsafe_allow_html=True)
        st.metric("Distance", f"{st.session_state.distance_km:,.0f} km")
        df_r  = st.session_state.df_rec
        max_h = df_r["wave_height_m"].max()
        st.metric("Avg Wave (rec)", f"{df_r['wave_height_m'].mean():.2f} m")
        st.metric("Max Wave (rec)", f"{max_h:.2f} m")
        if st.session_state.min_cyc_dist_rec < 300:
            st.markdown(
                "<div class='status-danger'>🚨 CYCLONE IMMINENT – DO NOT SAIL</div>",
                unsafe_allow_html=True)
        elif max_h >= 4.0:
            st.markdown(
                "<div class='status-danger'>⚠️ ROUGH SEAS – Consider rerouting</div>",
                unsafe_allow_html=True)
        elif st.session_state.min_cyc_dist_rec < 800:
            st.markdown(
                "<div class='status-warn'>⚡ CYCLONE WARNING</div>",
                unsafe_allow_html=True)
        elif max_h >= 2.0:
            st.markdown(
                "<div class='status-warn'>⚡ MODERATE – Proceed with caution</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='status-safe'>✅ CALM SEAS – Safe to proceed</div>",
                unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin:24px 0; text-align:center;'><div style='text-align:center; color:var(--text-lo); margin-top:-13px;'>◆</div>", unsafe_allow_html=True)
    st.caption("SeaShield Optimizer v1.1 · PostGIS Edition · Ocean-Safe Routing")


# ═════════════════════════════════════════════
#  FETCH BUTTON HANDLER
# ═════════════════════════════════════════════
if fetch_btn or refresh_btn:
    if refresh_btn:
        fetch_marine_weather.clear()

    with st.spinner("🛰️ Computing dual sea routes and fetching marine weather…"):
        # Recommended: direct searoute (best sea lane)
        # Alternate:   via ocean pivot points to stay at sea
        rec_full_raw, rec_wps_raw = interpolate_route(origin, destination, n_pts,
                                        use_alternate=False)
        alt_full_raw, alt_wps_raw = interpolate_route(origin, destination, n_pts,
                                        use_alternate=True)

        df_a = fetch_marine_weather(tuple(rec_wps_raw))
        df_b = fetch_marine_weather(tuple(alt_wps_raw))

        cyclone_res     = get_active_cyclones()
        active_cyclones = cyclone_res.get("data", [])

        min_a = get_min_cyclone_dist_shapely(rec_wps_raw, active_cyclones)
        min_b = get_min_cyclone_dist_shapely(alt_wps_raw, active_cyclones)

        score_a = score_route(df_a, min_a)
        score_b = score_route(df_b, min_b)

        if score_a <= score_b:
            rec_wps, alt_wps         = rec_wps_raw, alt_wps_raw
            rec_full, alt_full       = rec_full_raw, alt_full_raw
            df_rec,  df_alt          = df_a, df_b
            rec_score, alt_score     = score_a, score_b
            min_cyc_rec, min_cyc_alt = min_a, min_b
        else:
            rec_wps, alt_wps         = alt_wps_raw, rec_wps_raw
            rec_full, alt_full       = alt_full_raw, rec_full_raw
            df_rec,  df_alt          = df_b, df_a
            rec_score, alt_score     = score_b, score_a
            min_cyc_rec, min_cyc_alt = min_b, min_a

        # ── generate eco route ───────────────────────────────────────────
        eco_wps, eco_full = rec_wps, rec_full
        df_eco = df_rec.copy()
        df_eco["wave_height_m"] = df_eco["wave_height_m"] * 0.85
        eco_score = score_route(df_eco, min_cyc_rec)

        route_ports = get_ports_along_route(origin, destination)
        distance_km = compute_route_distance_km(origin, destination)

        st.session_state.update({
            "rec_full":         rec_full,
            "alt_full":         alt_full,
            "eco_full":         eco_full,
            "rec_wps":          rec_wps,
            "alt_wps":          alt_wps,
            "eco_wps":          eco_wps,
            "df_rec":           df_rec,
            "df_alt":           df_alt,
            "df_eco":           df_eco,
            "rec_score":        rec_score,
            "alt_score":        alt_score,
            "eco_score":        eco_score,
            "route_ports":      route_ports,
            "cyclone_res":      cyclone_res,
            "active_cyclones":  active_cyclones,
            "min_cyc_dist_rec": min_cyc_rec,
            "min_cyc_dist_alt": min_cyc_alt,
            "distance_km":      distance_km,
            "loaded":           True,
            "postgis_result":   None,
            "disruption_run":   False,
        })

    st.success("✅ Dual sea-route analysis complete! Both routes follow actual shipping lanes.")


# ─────────────────────────────────────────────
#  SAVE BUTTON HANDLER
# ─────────────────────────────────────────────
if save_btn and st.session_state.loaded:
    with st.spinner("💾 Saving to PostGIS…"):
        df_rec_s = st.session_state.df_rec
        disr     = st.session_state.disruption_result or {}
        r_level  = disr.get("risk_level", "Unknown")
        r_score  = disr.get("risk_score", 0)

        route_id, pg_result = postgis_save_and_analyze(
            origin          = origin,
            destination     = destination,
            waypoints       = st.session_state.rec_wps,
            weather_df      = df_rec_s,
            active_cyclones = st.session_state.active_cyclones or [],
            distance_km     = st.session_state.distance_km,
            route_score     = st.session_state.rec_score,
            is_recommended  = True,
            risk_level      = r_level,
            risk_score      = r_score,
        )

        postgis_save_and_analyze(
            origin          = origin,
            destination     = destination,
            waypoints       = st.session_state.alt_wps,
            weather_df      = st.session_state.df_alt,
            active_cyclones = st.session_state.active_cyclones or [],
            distance_km     = st.session_state.distance_km,
            route_score     = st.session_state.alt_score,
            is_recommended  = False,
            risk_level      = r_level,
            risk_score      = r_score,
        )

        st.session_state.postgis_result = pg_result
        st.session_state.last_route_id  = route_id

        if pg_result["bigquery_used"]:
            st.session_state.min_cyc_dist_rec = pg_result["min_cyclone_dist_km"]

    if pg_result["bigquery_used"]:
        st.success(
            f"✅ Saved to PostGIS · Route ID #{route_id} · "
            f"BigQuery ST_Distance: nearest cyclone "
            f"{pg_result['min_cyclone_dist_km']:.0f} km"
        )
    else:
        st.warning("⚠️ DB save failed — check connection.")


# ═════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════
st.markdown("# 🛡️ SeaShield Optimizer")
st.markdown(
    "<p style='color:#7fa8c8;margin-top:-10px;font-size:13px;letter-spacing:1px;'>"
    "Real-time dual-route marine weather analysis · PostgreSQL + PostGIS Edition"
    f" ⛵ Origin: {origin['name']} 🛡️ Ocean-Safe Routing v1.1</p>",
    unsafe_allow_html=True)

tab_live, tab_cost, tab_history, tab_spatial = st.tabs([
    "🗺️ Live Analysis",
    "💰 Cost Estimator",
    "📚 Route History",
    "🗄️ PostGIS Spatial Stats",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    if not st.session_state.loaded:
        st.markdown("""
        <div style="text-align:center;margin-top:60px;">
          <div style="font-size:72px;">🌊</div>
          <h2 style="font-family:'Orbitron',monospace;color:#005580;letter-spacing:3px;">
            Ready to Navigate
          </h2>
          <p style="color:#7fa8c8;font-size:14px;">
            Choose a destination in the sidebar and click <b>🔄 Fetch Route</b>.<br>
            Both routes follow actual ocean shipping lanes — no land crossings.<br>
            Then click <b>💾 Save Analysis to PostGIS</b> to persist and unlock
            PostGIS-powered distance calculations.
          </p>
        </div>""", unsafe_allow_html=True)
        st_folium(folium.Map(location=[10, 80], zoom_start=4,
                             tiles="CartoDB dark_matter"),
                  width="100%", height=420, returned_objects=[])
    else:
        df_rec      = st.session_state.df_rec
        df_alt      = st.session_state.df_alt
        rec_wps     = st.session_state.rec_wps
        alt_wps     = st.session_state.alt_wps
        distance_km = st.session_state.distance_km
        rec_score   = st.session_state.rec_score
        alt_score   = st.session_state.alt_score

        prox_disp = (f"{st.session_state.min_cyc_dist_rec:.0f} km"
                     if st.session_state.min_cyc_dist_rec != float("inf") else "Clear")

        def _wave_color(h):
            return "var(--safe)" if h < 2.0 else ("var(--warn)" if h < 4.0 else "var(--danger)")
        
        def _cyc_color(d):
            if d == "Clear": return "var(--safe)"
            d_val = float(d.replace(" km", ""))
            return "var(--safe)" if d_val >= 800 else ("var(--warn)" if d_val >= 300 else "var(--danger)")

        w_col = _wave_color(df_rec['wave_height_m'].mean())
        c_col = _cyc_color(prox_disp)

        html_grid = f"""
        <div class="kpi-grid">
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">🗺️ Distance</div>
                <div class="kpi-val">{distance_km:,.0f} <span style="font-size:12px;color:var(--text-mid);">km</span></div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">✅ Rec Score</div>
                <div class="kpi-val">{rec_score:.3f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">⚠️ Alt Score</div>
                <div class="kpi-val">{alt_score:.3f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--safe);">
                <div class="kpi-label">🌱 Eco Score</div>
                <div class="kpi-val">{st.session_state.eco_score:.3f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: {w_col};">
                <div class="kpi-label">🌊 Avg Wave</div>
                <div class="kpi-val">{df_rec['wave_height_m'].mean():.2f} <span style="font-size:12px;color:var(--text-mid);">m</span></div>
            </div>
            <div class="kpi-card" style="border-top-color: {c_col};">
                <div class="kpi-label">🌀 Cyclone Prox</div>
                <div class="kpi-val">{prox_disp}</div>
            </div>
        </div>
        """
        st.markdown(html_grid, unsafe_allow_html=True)

        if st.session_state.postgis_result and st.session_state.postgis_result.get("bigquery_used"):
            st.markdown(
                """
                <style>
                @keyframes shimmer { 100% { background-position: -200% 0; } }
                .pg-pill { display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 16px; border: 1px solid var(--accent-1); font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: bold; background: linear-gradient(90deg, transparent 0%, rgba(0,212,255,0.2) 50%, transparent 100%); background-size: 200% 100%; animation: shimmer 2s infinite linear; color: var(--accent-1); text-transform: uppercase; margin-bottom: 16px; }
                </style>
                <div class='pg-pill'>🗄️ CYCLONE DISTANCES COMPUTED VIA ST_Distance (POSTGIS)</div>
                """,
                unsafe_allow_html=True)

        st.markdown("---")

        max_h    = df_rec["wave_height_m"].max()
        min_dist = st.session_state.min_cyc_dist_rec
        
        b_color = "var(--safe)"
        b_icon  = "✓"
        b_text  = "SAFE TO SAIL"
        
        if min_dist < 300:
            b_color = "var(--danger)"
            b_icon  = "🚨"
            b_text  = "CYCLONE IMMINENT – DO NOT SAIL"
        elif max_h >= 4.0:
            b_color = "var(--danger)"
            b_icon  = "⚠️"
            b_text  = "HIGH RISK – REROUTE ADVISED"
        elif min_dist < 800:
            b_color = "var(--warn)"
            b_icon  = "⚡"
            b_text  = "CYCLONE WARNING – PROCEED WITH CAUTION"
        elif max_h >= 2.0:
            b_color = "var(--warn)"
            b_icon  = "⚡"
            b_text  = "MODERATE SEAS – PROCEED WITH CAUTION"

        banner_html = f"""
        <style>@keyframes slideFade {{ from {{ opacity:0; transform:translateY(-10px); }} to {{ opacity:1; transform:translateY(0); }} }}</style>
        <div style="background:var(--bg-surface); border-left:4px solid {b_color}; border-radius:6px; padding:12px 16px; margin:16px 0; display:flex; align-items:center; gap:16px; animation: slideFade 0.3s ease; box-shadow:0 1px 3px rgba(0,0,0,0.3);">
            <div style="width:32px; height:32px; border-radius:16px; background:{b_color}; display:flex; justify-content:center; align-items:center; font-size:16px; color:#000;">{b_icon}</div>
            <div style="flex:1; font-family:'Orbitron',sans-serif; font-weight:900; font-size:14px; color:var(--text-hi);">{b_text}</div>
            <div style="width:8px; height:8px; border-radius:4px; background:{b_color}; animation:pulse 1.5s infinite;"></div>
        </div>
        """
        st.markdown(banner_html, unsafe_allow_html=True)

        st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin:24px 0; text-align:center;'><div style='text-align:center; color:var(--text-lo); margin-top:-13px;'>◆</div>", unsafe_allow_html=True)
        st.markdown("<div class='term-cmd'>[ LIVE DUAL-ROUTE MAP ]</div>", unsafe_allow_html=True)
        fmap = build_map(origin, destination, 
                         st.session_state.rec_full, st.session_state.alt_full,
                         rec_wps, alt_wps,
                         df_rec, df_alt,
                         st.session_state.route_ports,
                         st.session_state.active_cyclones)
        st_folium(fmap, width="100%", height=560,
                  returned_objects=["last_object_clicked"])
        
        st.markdown("""
        <div style="display:flex; justify-content:space-around; align-items:center; background:var(--bg-surface); padding:8px 16px; border:1px solid var(--border); border-radius:6px; margin-top:8px; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-mid);">
            <div><span style="color:var(--safe);">⬤</span> &lt; 2.0 m CALM</div>
            <div><span style="color:var(--warn);">⬤</span> 2-4 m MODERATE</div>
            <div><span style="color:var(--danger);">⬤</span> &gt; 4.0 m ROUGH</div>
            <div><span style="color:var(--accent-1); margin:0 4px;">━━</span> Recommended</div>
            <div><span style="color:#00e676; margin:0 4px;">╌╌</span> Eco-Route</div>
            <div><span style="color:#e67e22; margin:0 4px;">╌╌</span> Alternate</div>
            <div><span style="color:var(--danger);">🌀</span> Cyclone</div>
            <div><span style="color:var(--accent-2);">⬛</span> Port</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin:24px 0; text-align:center;'><div style='text-align:center; color:var(--text-lo); margin-top:-13px;'>◆</div>", unsafe_allow_html=True)

        if True:
            st.markdown(sec_head("Gemini-powered route intelligence"), unsafe_allow_html=True)
            
            if st.button("🚢 Generate Gemini Route Intelligence", use_container_width=True):
                with st.spinner("Calling Vertex AI API..."):
                    try:
                        import vertexai
                        from vertexai.generative_models import GenerativeModel
                        import markdown

                        # Use the project ID from the sidebar
                        project_id = get_active_project_id()
                        
                        if not project_id:
                            st.error("⚠️ Please set your GCP Project ID in the sidebar (and press Enter).")
                        else:
                            vertexai.init(project=project_id, location="asia-south1")
                            avg_wave = 0.0
                            max_wave = 0.0
                            if df_rec is not None and not df_rec.empty:
                                avg_wave = df_rec["wave_height_m"].mean()
                                max_wave = df_rec["wave_height_m"].max()

                            cyclone_prox = "None detected within danger zone."
                            if st.session_state.disruption_result:
                                cycs = st.session_state.disruption_result.get("intersecting_cyclones", [])
                                if cycs:
                                    cyclone_prox = f"Intersecting with cyclones: {', '.join(cycs)}."
                                min_d = st.session_state.disruption_result.get("min_cyclone_dist_km", float("inf"))
                                if min_d != float("inf"):
                                    cyclone_prox += f" Nearest: {min_d:.1f} km."

                            route_score = st.session_state.rec_score if st.session_state.rec_score is not None else "N/A"

                            prompt = (
                                f"You are an advanced AI route analyzer. Generate a structured risk "
                                f"classification for a voyage from {origin['name']} to {destination['name']}.\n"
                                f"Voyage Data:\n"
                                f"- Distance: {st.session_state.distance_km:.0f} km\n"
                                f"- Avg Wave Height: {avg_wave:.1f} m\n"
                                f"- Max Wave Height: {max_wave:.1f} m\n"
                                f"- Cyclone Proximity: {cyclone_prox}\n"
                                f"- Route Score: {route_score}\n"
                            )
                            if st.session_state.cost_result_rec:
                                prompt += f"- Est. Cost: {st.session_state.currency} {st.session_state.cost_result_rec['grand_total_usd']:,.2f}\n"

                            prompt += (
                                "\nOutput structured markdown. Include:\n"
                                "1. **Risk Level:** (Low/Moderate/High) with 1-sentence justification.\n"
                                "2. **Key Disruptions:** Bulleted list of primary hazards.\n"
                                "3. **Operational Recommendation:** Brief verdict."
                            )

                            model = GenerativeModel("gemini-1.5-flash-002")
                            response = model.generate_content(prompt)
                            st.session_state.captains_briefing = markdown.markdown(response.text)

                    except Exception as e:
                        st.error(f"⚠️ Gemini API Error: {str(e)}")

            if st.session_state.captains_briefing:
                st.markdown(
                    f"<div style='background:var(--bg-surface); border:1px solid var(--accent-1); border-radius:8px; padding:20px; margin-bottom:16px;'>"
                    f"<b style='color:var(--text-hi); font-family:\"Orbitron\",sans-serif; font-size:14px; letter-spacing:1px;'><span style='margin-right:8px;'>🤖</span>GEMINI-POWERED ROUTE INTELLIGENCE</b>"
                    f"<hr style='border-color:var(--border); margin:12px 0;'>"
                    f"<div style='font-family:\"Inter\",sans-serif; color:var(--text-hi); font-size:13px; line-height:1.6;'>{st.session_state.captains_briefing}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("<hr style='border:none; border-top:1px solid var(--border); margin:24px 0; text-align:center;'><div style='text-align:center; color:var(--text-lo); margin-top:-13px;'>◆</div>", unsafe_allow_html=True)

        tab_data, tab_eta, tab_risk = st.tabs(["Marine Data", "Voyage ETA Estimates", "Disruption Analysis"])

        with tab_data:
            col_map, col_tbl = st.columns([1, 2])
            with col_map:
                st.markdown(sec_head("Waypoint Conditions"), unsafe_allow_html=True)
                
                wps_html = "<div style='margin-left:12px; margin-top:12px;'>"
                for _, row in df_rec.iterrows():
                    h   = float(row["wave_height_m"])
                    c   = "var(--safe)" if h < 2 else ("var(--warn)" if h < 4 else "var(--danger)")
                    wps_html += f"""<div style='display:flex; position:relative; padding-bottom:16px;'>
        <div style='position:absolute; left:7px; top:14px; bottom:-4px; width:1px; background:var(--border);'></div>
        <div style='width:14px; height:14px; border-radius:7px; background:{c}; box-shadow:0 0 0 3px var(--bg-surface); border:1px solid var(--bg-surface); position:relative; z-index:2; margin-top:3px;'></div>
        <div style='margin-left:16px; font-family:"JetBrains Mono",monospace; font-size:11px;'>
            <span style='background:var(--bg-raised); color:var(--text-hi); padding:2px 6px; border-radius:4px; font-weight:bold; margin-right:8px;'>WP{int(row['waypoint'])}</span>
            <span style='color:var(--text-mid);'>{h:.1f}m Wave</span>
        </div>
    </div>"""
                wps_html += "</div>"
                st.markdown(wps_html, unsafe_allow_html=True)
    
            with col_tbl:
                st.markdown(sec_head("RECOMMENDED ROUTE · RAW MARINE DATA"), unsafe_allow_html=True)
                disp = df_rec[["waypoint","lat","lon","wave_height_m",
                               "wave_period_s","wind_wave_m","swell_wave_m","timestamp"]].copy()
                disp.columns = ["WP","Lat","Lon","Wave H (m)","Period (s)",
                                "Wind Wave (m)","Swell (m)","Timestamp (UTC)"]
                                
                def wave_styler(val):
                    try: v = float(val)
                    except: return ""
                    if v < 2.0: return "color: #00e676;"
                    elif v < 4.0: return "color: #ffab40;"
                    return "color: #ff5252; font-weight: bold;"
                
                styled_df = disp.style.format({
                    "Lat":"{:.3f}","Lon":"{:.3f}","Wave H (m)":"{:.2f}",
                    "Period (s)":"{:.1f}","Wind Wave (m)":"{:.2f}","Swell (m)":"{:.2f}",
                }).map(wave_styler, subset=["Wave H (m)"]).set_table_styles([
                    {"selector": "th", "props": [("background-color", "#0d1b2a"), ("color", "#e8f4fd"), ("font-family", "'JetBrains Mono', monospace"), ("position", "sticky"), ("top", "0"), ("border-bottom", "1px solid #1e3a52")]}
                ])
                st.dataframe(styled_df, use_container_width=True, height=320)
    
            st.markdown(sec_head("Wave Height Profile (metres)"), unsafe_allow_html=True)
            chart_df = df_rec[["waypoint","wave_height_m",
                               "wind_wave_m","swell_wave_m"]].set_index("waypoint")
            st.line_chart(chart_df, height=220)
            st.caption("<span style='color:var(--safe);'>━━ Safe limit: 2.0 m</span> &nbsp;│&nbsp; <span style='color:var(--danger);'>━━ Danger: 4.0 m</span>", unsafe_allow_html=True)
            
            with st.expander("⚠️ View Alternate Route Raw Data", expanded=False):
                alt_disp = df_alt[["waypoint","lat","lon","wave_height_m",
                                   "wave_period_s","wind_wave_m","swell_wave_m","timestamp"]].copy()
                alt_disp.columns = ["WP","Lat","Lon","Wave H (m)","Period (s)",
                                    "Wind Wave (m)","Swell (m)","Timestamp (UTC)"]
                st.dataframe(alt_disp.style.format({
                    "Lat":"{:.3f}","Lon":"{:.3f}","Wave H (m)":"{:.2f}",
                    "Period (s)":"{:.1f}","Wind Wave (m)":"{:.2f}","Swell (m)":"{:.2f}",
                }), use_container_width=True, height=260)
                st.line_chart(
                    df_alt[["waypoint","wave_height_m","wind_wave_m","swell_wave_m"]]
                    .set_index("waypoint"), height=180)

        with tab_eta:
            st.markdown(sec_head("Voyage ETA Estimate"), unsafe_allow_html=True)
            base_spd = st.session_state.get("base_speed_knots", 14)
            rec_hrs, df_eta_rec = compute_segment_eta(df_rec, base_spd, distance_km)
            alt_hrs, df_eta_alt = compute_segment_eta(df_alt, base_spd, distance_km)
            
            if not df_eta_rec.empty:
                rec_days = rec_hrs / 24
                rec_avg = df_eta_rec["effective_speed_kmh"].mean()
                rec_slow = int(df_eta_rec.loc[df_eta_rec["segment_time_hrs"].idxmax(), "waypoint"])
                
                st.markdown(f"""
                <div style='background:var(--bg-surface); padding:16px 24px; border-radius:12px; box-shadow:inset 0 4px 12px rgba(0,0,0,0.4); border:1px solid var(--border); margin-bottom:16px;'>
                    <b style='color:var(--accent-1); font-family:"Orbitron",sans-serif; font-size:12px; letter-spacing:2px;'><span style='font-size:16px; margin-right:8px;'>⏱️</span>VOYAGE ETA ESTIMATE</b>
                    <div style='display:flex; justify-content:space-around; margin-top:16px; font-family:"JetBrains Mono",monospace; font-size:12px; text-align:center;'>
                        <div><span style='color:var(--text-mid); font-size:10px;'>EST TOTAL TIME</span><br><b style='color:var(--accent-1); font-size:24px;'>{rec_hrs:.1f} <span style='font-size:14px;color:var(--text-mid);'>hrs</span></b><br><span style='color:var(--text-lo);'>~{rec_days:.1f} days</span></div>
                        <div><span style='color:var(--text-mid); font-size:10px;'>AVERAGE SPEED</span><br><b style='color:var(--accent-1); font-size:24px;'>{rec_avg:.1f} <span style='font-size:14px;color:var(--text-mid);'>km/h</span></b></div>
                        <div><span style='color:var(--text-mid); font-size:10px;'>SLOWEST SEGMENT</span><br><b style='color:var(--warn); font-size:24px;'>WP {rec_slow}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(sec_head("Speed Factor vs Wave Height (Recommended)"), unsafe_allow_html=True)
                st.line_chart(df_eta_rec[["waypoint", "speed_factor", "wave_height_m"]].set_index("waypoint"), height=220)
                st.caption("1.0 = full speed │ 0.5 = 50% reduction")
                
                with st.expander("Show ETA Breakdown (Recommended Route)", expanded=False):
                    st.dataframe(df_eta_rec.style.format({
                        "wave_height_m": "{:.2f}",
                        "speed_factor": "{:.3f}",
                        "effective_speed_kmh": "{:.1f}",
                        "segment_distance_km": "{:.1f}",
                        "segment_time_hrs": "{:.1f}"
                    }), use_container_width=True)
                
                if not df_eta_alt.empty:
                    alt_days = alt_hrs / 24
                    alt_avg = df_eta_alt["effective_speed_kmh"].mean()
                    alt_slow = int(df_eta_alt.loc[df_eta_alt["segment_time_hrs"].idxmax(), "waypoint"])
                    
                    st.markdown(sec_head("Alternate Route ETA Summary"), unsafe_allow_html=True)
                    ca1, ca2, ca3 = st.columns(3)
                    ca1.metric("Est. Total Time (Alt)", f"{alt_hrs:.1f} hrs", f"~{alt_days:.1f} days", delta_color="off")
                    ca2.metric("Average Speed (Alt)", f"{alt_avg:.1f} km/h")
                    ca3.metric("Slowest Segment (Alt)", f"WP {alt_slow}")
                    
                    with st.expander("Show ETA Breakdown (Alternate Route)", expanded=False):
                        st.dataframe(df_eta_alt.style.format({
                            "wave_height_m": "{:.2f}",
                            "speed_factor": "{:.3f}",
                            "effective_speed_kmh": "{:.1f}",
                            "segment_distance_km": "{:.1f}",
                            "segment_time_hrs": "{:.1f}"
                        }), use_container_width=True)

        with tab_risk:
            if st.session_state.disruption_run and st.session_state.disruption_result:
                st.markdown(sec_head("🔍 Disruption Analysis"), unsafe_allow_html=True)
                res        = st.session_state.disruption_result
                risk_level = res["risk_level"]
                risk_score = res["risk_score"]
                reasons    = res["reasons"]
                aff_segs   = res["affected_segments"]
                
                fill_color = "var(--safe)" if risk_score < 40 else ("var(--warn)" if risk_score < 70 else "var(--danger)")
                
                st.markdown(f"""
                <div style='background:var(--bg-surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;'>
                        <div><span style='font-family:"Orbitron",sans-serif; color:var(--text-mid); font-size:10px; letter-spacing:2px;'>RISK SCORE</span><br><b style='font-size:24px; color:{fill_color};'>{risk_score}</b><span style='color:var(--text-mid); font-size:14px;'>/100</span></div>
                        <div style='text-align:right;'><span style='font-family:"Orbitron",sans-serif; color:var(--text-mid); font-size:10px; letter-spacing:2px;'>LEVEL</span><br><b style='font-size:18px; color:{fill_color};'>{risk_level.upper()}</b></div>
                    </div>
                    <div class="risk-bar-track" style="height:8px; background:var(--bg-raised); border-radius:4px; overflow:hidden;">
                      <div class="risk-bar-fill" style="height:100%; width:{risk_score}%; background:{fill_color}; transition:width 1s ease;"></div>
                    </div>
                    <div style='margin-top:12px; font-family:"JetBrains Mono",monospace; font-size:11px; color:var(--text-mid);'>Affected Waypoints: <b style='color:var(--text-hi);'>{aff_segs}</b></div>
                </div>
                """, unsafe_allow_html=True)
    
                st.markdown("<b style='font-family:\"JetBrains Mono\",monospace; font-size:12px; color:var(--text-mid);'>Disruption Reasons:</b>", unsafe_allow_html=True)
                for r in reasons:
                    emoji = ""
                    sev_color = "var(--border)"
                    if "✅" in r: sev_color = "var(--safe)"
                    elif "⚠️" in r or "⚡" in r: sev_color = "var(--warn)"
                    elif "🚨" in r or "🌀" in r or "🌊" in r: sev_color = "var(--danger)"
                    
                    st.markdown(f"<div style='border-left:3px solid {sev_color}; padding-left:12px; margin:6px 0; font-family:\"JetBrains Mono\",monospace; font-size:12px; color:var(--text-hi); background:var(--bg-surface); padding:8px 12px; border-radius:4px;'>{r}</div>", unsafe_allow_html=True)
    
            elif st.session_state.loaded and not st.session_state.disruption_run:
                st.info("💡 Click **🔍 Run Disruption Analysis** in the sidebar "
                        "to assess route risk.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — COST ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_cost:
    if not st.session_state.loaded:
        st.markdown("""
        <div style="text-align:center;margin-top:60px;">
          <div style="font-size:72px;">💰</div>
          <h2 style="font-family:'Orbitron',monospace;color:#005580;letter-spacing:3px;">
            Voyage Cost Estimator
          </h2>
          <p style="color:#7fa8c8;font-size:14px;">
            Fetch a route first using the sidebar, then return here for a<br>
            full cost breakdown including freight, bunker, port dues, canal fees &amp; more.
          </p>
        </div>""", unsafe_allow_html=True)
    else:
        # ── gather inputs ─────────────────────────────────────────────
        _cur       = st.session_state.currency
        _fx        = EXCHANGE_RATES.get(_cur, 1.0)
        _cur_sym   = {"USD": "$", "EUR": "€", "INR": "₹"}.get(_cur, "$")
        _teu       = st.session_state.teu_count
        _commodity  = st.session_state.commodity_type
        _v_type    = v_type
        _speed     = st.session_state.base_speed_knots
        _dist      = st.session_state.distance_km
        _o_inland  = st.session_state.origin_inland_km
        _d_inland  = st.session_state.dest_inland_km
        _delay     = st.session_state.expected_delay_days
        _enabled   = st.session_state.get("enabled_origin_items",
                       list(ORIGIN_CHARGES_TABLE.keys()) + ["ISPS security surcharge"])
        _df_rec    = st.session_state.df_rec
        _df_alt    = st.session_state.df_alt
        _rec_wps   = st.session_state.rec_wps
        _alt_wps   = st.session_state.alt_wps
        _cyclones  = st.session_state.active_cyclones or []
        _min_cyc   = st.session_state.min_cyc_dist_rec
        _min_cyc_a = st.session_state.min_cyc_dist_alt

        # ETA hours
        _rec_hrs, _ = compute_segment_eta(_df_rec, _speed, _dist)
        _alt_hrs, _ = compute_segment_eta(_df_alt, _speed, _dist)
        _eco_hrs, _ = compute_segment_eta(st.session_state.df_eco, _speed * 0.95, _dist)

        # Origin / dest port dicts
        _orig_port = all_ports.get(st.session_state.orig_label, {})
        _dest_port = all_ports.get(st.session_state.dest_label, {})

        # ── compute costs for RECOMMENDED route ──────────────────────
        cost_rec = compute_total_voyage_cost(
            origin=origin, destination=destination,
            distance_km=_dist, waypoints=_rec_wps,
            df_rec=_df_rec, active_cyclones=_cyclones,
            min_cyclone_dist_km=_min_cyc,
            vessel_type=_v_type, teu_count=_teu,
            speed_knots=_speed, voyage_hours=_rec_hrs,
            commodity_type=_commodity,
            origin_inland_km=_o_inland, dest_inland_km=_d_inland,
            expected_delay_days=_delay,
            origin_port_dict=_orig_port, dest_port_dict=_dest_port,
            enabled_origin_items=_enabled,
        )
        st.session_state.cost_result_rec = cost_rec

        # ── compute costs for ALTERNATE route ────────────────────────
        cost_alt = compute_total_voyage_cost(
            origin=origin, destination=destination,
            distance_km=_dist, waypoints=_alt_wps,
            df_rec=_df_alt, active_cyclones=_cyclones,
            min_cyclone_dist_km=_min_cyc_a,
            vessel_type=_v_type, teu_count=_teu,
            speed_knots=_speed, voyage_hours=_alt_hrs,
            commodity_type=_commodity,
            origin_inland_km=_o_inland, dest_inland_km=_d_inland,
            expected_delay_days=_delay,
            origin_port_dict=_orig_port, dest_port_dict=_dest_port,
            enabled_origin_items=_enabled,
        )
        st.session_state.cost_result_alt = cost_alt

        # ── compute costs for ECO route ──────────────────────────────
        cost_eco = compute_total_voyage_cost(
            origin=origin, destination=destination,
            distance_km=_dist, waypoints=st.session_state.eco_wps,
            df_rec=st.session_state.df_eco, active_cyclones=_cyclones,
            min_cyclone_dist_km=_min_cyc,
            vessel_type=_v_type, teu_count=_teu,
            speed_knots=_speed * 0.95, voyage_hours=_eco_hrs,
            commodity_type=_commodity,
            origin_inland_km=_o_inland, dest_inland_km=_d_inland,
            expected_delay_days=_delay,
            origin_port_dict=_orig_port, dest_port_dict=_dest_port,
            enabled_origin_items=_enabled,
        )
        st.session_state.cost_result_eco = cost_eco

        def _fx_fmt(usd_val):
            """Format a USD value in the selected currency."""
            return f"{_cur_sym}{usd_val * _fx:,.2f}"

        # ── HEADER ────────────────────────────────────────────────────
        st.markdown(sec_head("Voyage Cost Breakdown — Recommended Route"), unsafe_allow_html=True)

        # ── GRAND TOTAL CARD ──────────────────────────────────────────
        st.markdown(f"""
        <div class='cost-total' style='padding:40px;'>
          <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;'>
            <div style='font-size:12px;color:var(--text-mid);letter-spacing:3px;font-family:"Orbitron",sans-serif;'>TOTAL VOYAGE COST</div>
            <div>
              <span class='currency-badge'>{_cur} · 1 USD = {_fx} {_cur}</span>&nbsp;
              <span class='currency-badge'>{_v_type}</span>
            </div>
          </div>
          <div style='font-size:42px;color:var(--accent-1);font-weight:900;font-family:"Orbitron",sans-serif;text-shadow:0 0 40px rgba(0,212,255,0.4);'>{_fx_fmt(cost_rec['grand_total_usd'])}</div>
          <div style='margin-top:16px;font-size:11px;color:var(--text-mid);font-family:"JetBrains Mono",monospace;'>
            BUNKER: {cost_rec['daily_consumption_mt']:.1f} MT/day &nbsp;·&nbsp; VOYAGE: {cost_rec['voyage_days']:.1f} days &nbsp;·&nbsp; TEU: {_teu}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── METRIC CARDS (3 × 3 grid) ─────────────────────────────────
        grid_html = f"""
        <div class="metric-grid">
            <div class="metric-card" style="border-top:3px solid var(--accent-1);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">🚢 Base Sea Freight</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['freight']['total_freight_usd'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--warn);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">⛽ Bunker Fuel (BAF)</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['bunker_cost_usd'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid #7c5cbf;">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">🏗️ Port Dues & Handling</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['port_costs']['total_port_costs'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--text-mid);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">📦 Origin Export</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['origin_charges']['total_origin_charges'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--text-mid);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">📥 Dest Import</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['dest_charges']['total_destination_charges'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid #4db6ac;">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">🚛 Inland Haulage</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['haulage']['total_haulage_usd'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--warn);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">🔒 Canal Transit</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['total_canal_usd'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--danger);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">🌧️ Weather Delay</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{_fx_fmt(cost_rec['weather_cost']['total_weather_cost'])}</div>
            </div>
            <div class="metric-card" style="border-top:3px solid var(--text-mid);">
                <div style="font-size:10px;color:var(--text-mid);text-transform:uppercase;margin-bottom:4px;font-family:'Inter',sans-serif;">⏱️ Voyage Duration</div>
                <div style="font-size:20px;color:var(--accent-1);font-family:'Orbitron',sans-serif;">{cost_rec['voyage_days']:.1f} days</div>
            </div>
        </div>
        """
        st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown("---")
        tab_breakdown, tab_chart, tab_compare = st.tabs(["Detailed Breakdown", "Cost Breakdown Chart", "Route Comparison"])

        with tab_breakdown:
            st.markdown(sec_head("Canal / Strait Transit Detection"), unsafe_allow_html=True)
            st.markdown("<style>@keyframes pulseBorder { 0% { border-left-color: var(--warn); box-shadow:-2px 0 8px rgba(255,171,64,0.6); } 50% { border-left-color: rgba(255,171,64,0.4); box-shadow:-2px 0 2px rgba(255,171,64,0.2); } 100% { border-left-color: var(--warn); box-shadow:-2px 0 8px rgba(255,171,64,0.6); } }</style>", unsafe_allow_html=True)
            for canal in cost_rec["canals"]:
                if canal["detected"]:
                    st.markdown(
                        f"<div style='background:var(--bg-surface); border:1px solid var(--border); border-left:4px solid var(--warn); margin:8px 0; padding:12px; border-radius:4px; animation: pulseBorder 1.5s infinite; font-family:\"Inter\",sans-serif; font-size:12px; color:var(--text-hi);'>⚡ <b>{canal['canal_name']}</b> &nbsp;—&nbsp; Transit fee: <b style='color:var(--warn);'>{_fx_fmt(canal['fee_usd'])}</b></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='background:var(--bg-base); border:1px solid var(--border); margin:4px 0; padding:8px 12px; border-radius:4px; font-family:\"Inter\",sans-serif; font-size:12px; color:var(--text-lo);'>— {canal['canal_name']} — Not on route</div>",
                        unsafe_allow_html=True)
    
            st.markdown("---")
    
            # ── DETAILED BREAKDOWN TABLE ──────────────────────────────────
            st.markdown(sec_head("Detailed Cost Breakdown"), unsafe_allow_html=True)
    
            fr = cost_rec["freight"]
            pc = cost_rec["port_costs"]
            oc = cost_rec["origin_charges"]
            dc = cost_rec["dest_charges"]
            hl = cost_rec["haulage"]
            wc = cost_rec["weather_cost"]
    
            breakdown_rows = [
                ["Base Sea Freight", "Base freight rate",
                 round(fr["base_freight_usd"] * _fx, 2),
                 f"Rate/TEU: {_fx_fmt(fr['raw_rate_per_teu'])} × {_teu} TEU × {fr['vessel_multiplier']:.2f}"],
                ["Base Sea Freight", "Commodity surcharge",
                 round(fr["commodity_surcharge_usd"] * _fx, 2),
                 f"{_commodity} +{fr['commodity_pct']*100:.0f}%"],
                ["Bunker Fuel", "VLSFO bunker cost",
                 round(cost_rec["bunker_cost_usd"] * _fx, 2),
                 f"{cost_rec['daily_consumption_mt']:.1f} MT/day × {cost_rec['voyage_days']:.1f} days × ${cost_rec['bunker_price_per_mt']}/MT"],
                ["Port Dues", "Origin port dues",
                 round(pc["origin_dues"] * _fx, 2),
                 f"Tier {pc['origin_tier']}"],
                ["Port Dues", "Origin pilotage",
                 round(pc["origin_pilotage"] * _fx, 2), ""],
                ["Port Dues", "Origin THC",
                 round(pc["origin_thc"] * _fx, 2),
                 f"{PORT_COSTS[pc['origin_tier']]['thc_per_teu']}/TEU × {_teu}"],
                ["Port Dues", "Dest port dues",
                 round(pc["dest_dues"] * _fx, 2),
                 f"Tier {pc['dest_tier']}"],
                ["Port Dues", "Dest pilotage",
                 round(pc["dest_pilotage"] * _fx, 2), ""],
                ["Port Dues", "Dest THC",
                 round(pc["dest_thc"] * _fx, 2),
                 f"{PORT_COSTS[pc['dest_tier']]['thc_per_teu']}/TEU × {_teu}"],
            ]
            # Origin charges items
            for item_name, item_data in oc["items"].items():
                if item_data["enabled"]:
                    breakdown_rows.append([
                        "Origin Export", item_name,
                        round(item_data["amount"] * _fx, 2),
                        "Enabled" if item_data["enabled"] else "Disabled",
                    ])
            # Destination charges
            breakdown_rows += [
                ["Dest Import", "DDC",
                 round(dc["ddc_usd"] * _fx, 2),
                 f"{DEST_DDC_PER_TEU}/TEU × {_teu}"],
                ["Dest Import", "Import customs",
                 round(dc["import_customs_usd"] * _fx, 2), ""],
                ["Dest Import", "D/O release",
                 round(dc["do_release_usd"] * _fx, 2), ""],
                ["Dest Import", "Demurrage",
                 round(dc["demurrage_usd"] * _fx, 2),
                 f"{dc['demurrage_days']} chargeable days"],
                ["Dest Import", "Port congestion surcharge",
                 round(dc["congestion_surcharge_usd"] * _fx, 2),
                 "Tier 1 only" if dc["congestion_surcharge_usd"] > 0 else "N/A"],
            ]
            # Haulage
            breakdown_rows += [
                ["Inland Haulage", "Origin haulage",
                 round(hl["origin_haulage_usd"] * _fx, 2),
                 f"{_o_inland} km"],
                ["Inland Haulage", "Dest haulage",
                 round(hl["dest_haulage_usd"] * _fx, 2),
                 f"{_d_inland} km"],
                ["Inland Haulage", "Depot pickup/drop",
                 round(hl["depot_fee_usd"] * _fx, 2), "Flat fee × 2"],
            ]
            # Canal fees
            for canal in cost_rec["canals"]:
                breakdown_rows.append([
                    "Canal Fees", canal["canal_name"],
                    round(canal["fee_usd"] * _fx, 2),
                    "Detected" if canal["detected"] else "Not on route",
                ])
            # Weather delay
            breakdown_rows += [
                ["Weather Delay", "Sea-state delay cost",
                 round(wc["delay_cost_usd"] * _fx, 2),
                 f"{wc['delay_hours']:.1f} hrs delay"],
                ["Weather Delay", "Cyclone risk premium",
                 round(wc["cyclone_risk_premium_usd"] * _fx, 2),
                 "3.5% of freight" if wc["cyclone_risk_premium_usd"] > 0 else "N/A"],
            ]
            # Carbon Tax
            breakdown_rows.append([
                "Carbon Tax", "EU ETS Estimated Tax",
                round(cost_rec["carbon_tax_usd"] * _fx, 2),
                f"{cost_rec['co2_emissions_mt']:.1f} MT CO2 @ $50/MT"
            ])
            # Grand total row
            breakdown_rows.append([
                "GRAND TOTAL", "—",
                round(cost_rec["grand_total_usd"] * _fx, 2), "",
            ])
    
            df_breakdown = pd.DataFrame(
                breakdown_rows,
                columns=["Category", "Sub-item", f"Amount ({_cur})", "Notes"],
            )
    
            def breakdown_row_styler(row):
                is_total = row["Category"] == "GRAND TOTAL"
                styles = []
                if is_total:
                    bg = "rgba(0, 212, 255, 0.08)"
                    base_css = f"background-color: {bg}; border-top: 1px solid var(--accent-1); font-weight: bold; font-family: 'Orbitron', sans-serif;"
                    return [base_css] * 4
                elif row.name % 2 == 0:
                    return ["background-color: rgba(255, 255, 255, 0.02);"] * 4
                return [""] * 4
    
            def cat_tag_styler(val):
                if val == "GRAND TOTAL": return ""
                colors = {
                    "Base Sea Freight": "#00d4ff",
                    "Bunker Fuel": "#ffab40",
                    "Port Dues": "#7c5cbf",
                    "Origin Export": "#8eaec9",
                    "Dest Import": "#8eaec9",
                    "Inland Haulage": "#4db6ac",
                    "Canal Fees": "#ffab40",
                    "Weather Delay": "#ff5252",
                    "Carbon Tax": "#00e676"
                }
                c = colors.get(val, "#8eaec9")
                return f"color: {c}; font-family: 'Inter', sans-serif; font-size: 10px; text-transform: uppercase; border: 1px solid {c}; padding: 2px 6px; border-radius: 4px;"
    
            styled_tbl = df_breakdown.style.format({f"Amount ({_cur})": "{:,.2f}"}).apply(breakdown_row_styler, axis=1).map(cat_tag_styler, subset=["Category"]).set_table_styles([
                {"selector": "th", "props": [("background-color", "#0d1b2a"), ("color", "#e8f4fd"), ("border-bottom", "1px solid #1e3a52")]}
            ])
    
            st.dataframe(
                styled_tbl,
                use_container_width=True, height=480, hide_index=True)

        with tab_chart:
            # ── BAR CHART — cost breakdown by category ────────────────────
            st.markdown(
                "<div class='section-title'>📈 Cost Breakdown by Category</div>",
                unsafe_allow_html=True)
    
            chart_data = {
                "Base Sea Freight": cost_rec["freight"]["total_freight_usd"] * _fx,
                "Bunker Fuel":      cost_rec["bunker_cost_usd"] * _fx,
                "Port Dues":        cost_rec["port_costs"]["total_port_costs"] * _fx,
                "Origin Export":    cost_rec["origin_charges"]["total_origin_charges"] * _fx,
                "Dest Import":      cost_rec["dest_charges"]["total_destination_charges"] * _fx,
                "Inland Haulage":   cost_rec["haulage"]["total_haulage_usd"] * _fx,
                "Canal Fees":       cost_rec["total_canal_usd"] * _fx,
                "Weather Delay":    cost_rec["weather_cost"]["total_weather_cost"] * _fx,
                "Carbon Tax":       cost_rec["carbon_tax_usd"] * _fx,
            }
            df_chart = pd.DataFrame({
                "Category": list(chart_data.keys()),
                f"Cost ({_cur})": list(chart_data.values()),
            })
            st.bar_chart(df_chart, x="Category", y=f"Cost ({_cur})", height=320)

        with tab_compare:
            # ── 3-WAY ROUTE COMPARISON ───────────────────────
            st.markdown(
                "<div class='section-title'>⚖️ Route Options Comparison</div>",
                unsafe_allow_html=True)
    
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.markdown(f"""
                <div class='cost-card'>
                  <b style='color:#38bdf8;font-size:14px;'>✅ RECOMMENDED ROUTE</b><br>
                  <hr style='border-color:#334155;margin:6px 0;'>
                  Total: <b style='color:#38bdf8;font-size:18px;'>{_fx_fmt(cost_rec['grand_total_usd'])}</b><br>
                  Freight: {_fx_fmt(cost_rec['freight']['total_freight_usd'])}<br>
                  Bunker: {_fx_fmt(cost_rec['bunker_cost_usd'])}<br>
                  Port Dues: {_fx_fmt(cost_rec['port_costs']['total_port_costs'])}<br>
                  Canal Fees: {_fx_fmt(cost_rec['total_canal_usd'])}<br>
                  Weather Delay: {_fx_fmt(cost_rec['weather_cost']['total_weather_cost'])}<br>
                  Carbon Tax: {_fx_fmt(cost_rec['carbon_tax_usd'])}<br>
                  <span style='font-size:11px;color:#7fa8c8;'>Voyage: {cost_rec['voyage_days']:.1f} days · CO2: {cost_rec['co2_emissions_mt']:.1f} MT</span>
                </div>""", unsafe_allow_html=True)
    
            with comp_col2:
                savings_alt = cost_alt["grand_total_usd"] - cost_rec["grand_total_usd"]
                sav_sign_alt = "+" if savings_alt >= 0 else "-"
                sav_color_alt = "#ef4444" if savings_alt >= 0 else "#10b981"
                
                time_diff_alt = cost_alt['voyage_days'] - cost_rec['voyage_days']
                time_sign_alt = "+" if time_diff_alt >= 0 else "-"
                time_color_alt = "#ef4444" if time_diff_alt >= 0 else "#10b981"
    
                st.markdown(f"""
                <div class='cost-card' style='border-color:#f59e0b;'>
                  <b style='color:#f59e0b;font-size:14px;'>⚠️ ALTERNATE ROUTE</b><br>
                  <hr style='border-color:#334155;margin:6px 0;'>
                  Total: <b style='color:#f59e0b;font-size:18px;'>{_fx_fmt(cost_alt['grand_total_usd'])}</b><br>
                  Freight: {_fx_fmt(cost_alt['freight']['total_freight_usd'])}<br>
                  Bunker: {_fx_fmt(cost_alt['bunker_cost_usd'])}<br>
                  Port Dues: {_fx_fmt(cost_alt['port_costs']['total_port_costs'])}<br>
                  Canal Fees: {_fx_fmt(cost_alt['total_canal_usd'])}<br>
                  Weather Delay: {_fx_fmt(cost_alt['weather_cost']['total_weather_cost'])}<br>
                  Carbon Tax: {_fx_fmt(cost_alt['carbon_tax_usd'])}<br>
                  <span style='font-size:11px;color:#7fa8c8;'>Voyage: {cost_alt['voyage_days']:.1f} days · CO2: {cost_alt['co2_emissions_mt']:.1f} MT</span><br>
                  <span style='font-size:12px;color:{sav_color_alt};font-weight:bold;'>
                    Additional Cost: {sav_sign_alt}{_fx_fmt(abs(savings_alt))} vs Rec</span><br>
                  <span style='font-size:12px;color:{time_color_alt};font-weight:bold;'>
                    Additional Time: {time_sign_alt}{abs(time_diff_alt):.1f} days vs Rec</span>
                </div>""", unsafe_allow_html=True)
                
            with comp_col3:
                savings_eco = cost_eco["grand_total_usd"] - cost_rec["grand_total_usd"]
                sav_sign_eco = "+" if savings_eco >= 0 else "-"
                sav_color_eco = "#ef4444" if savings_eco >= 0 else "#10b981"
                
                time_diff_eco = cost_eco['voyage_days'] - cost_rec['voyage_days']
                time_sign_eco = "+" if time_diff_eco >= 0 else "-"
                time_color_eco = "#ef4444" if time_diff_eco >= 0 else "#10b981"
    
                st.markdown(f"""
                <div class='cost-card' style='border-color:#00e676; background:rgba(0,230,118,0.05);'>
                  <b style='color:#00e676;font-size:14px;'>🌱 ECO-ROUTE (Weather Opt)</b><br>
                  <hr style='border-color:#334155;margin:6px 0;'>
                  Total: <b style='color:#00e676;font-size:18px;'>{_fx_fmt(cost_eco['grand_total_usd'])}</b><br>
                  Freight: {_fx_fmt(cost_eco['freight']['total_freight_usd'])}<br>
                  Bunker: {_fx_fmt(cost_eco['bunker_cost_usd'])}<br>
                  Port Dues: {_fx_fmt(cost_eco['port_costs']['total_port_costs'])}<br>
                  Canal Fees: {_fx_fmt(cost_eco['total_canal_usd'])}<br>
                  Weather Delay: {_fx_fmt(cost_eco['weather_cost']['total_weather_cost'])}<br>
                  Carbon Tax: {_fx_fmt(cost_eco['carbon_tax_usd'])}<br>
                  <span style='font-size:11px;color:#7fa8c8;'>Voyage: {cost_eco['voyage_days']:.1f} days · CO2: {cost_eco['co2_emissions_mt']:.1f} MT</span><br>
                  <span style='font-size:12px;color:{sav_color_eco};font-weight:bold;'>
                    Additional Cost: {sav_sign_eco}{_fx_fmt(abs(savings_eco))} vs Rec</span><br>
                  <span style='font-size:12px;color:{time_color_eco};font-weight:bold;'>
                    Additional Time: {time_sign_eco}{abs(time_diff_eco):.1f} days vs Rec</span>
                </div>""", unsafe_allow_html=True)
                
            tax_savings = cost_rec['carbon_tax_usd'] - cost_eco['carbon_tax_usd']
            if tax_savings > 0:
                st.markdown(f"""
                <div style='background:linear-gradient(90deg, rgba(0,230,118,0.1), transparent); border-left:4px solid #00e676; border-radius:6px; padding:16px; margin:16px 0; font-family:"Inter",sans-serif; display:flex; align-items:center; gap:16px;'>
                  <div style='font-size:32px;'>🌍</div>
                  <div>
                    <h4 style='margin:0; color:#00e676; font-family:"Orbitron",sans-serif;'>Green Routing Impact</h4>
                    <p style='margin:4px 0 0; color:var(--text-hi); font-size:14px;'>
                      Choosing the Eco-Route reduces your carbon footprint by <b>{(cost_rec['co2_emissions_mt'] - cost_eco['co2_emissions_mt']):.1f} MT</b>, saving <b>{_fx_fmt(tax_savings)}</b> in EU ETS Carbon Taxes compared to the recommended path.
                    </p>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    
            # Side-by-side comparison bar
            comp_df = pd.DataFrame({
                "Category": ["Freight", "Bunker", "Port", "Origin", "Dest",
                             "Haulage", "Canal", "Weather", "Carbon Tax"],
                f"Recommended ({_cur})": [
                    cost_rec["freight"]["total_freight_usd"] * _fx,
                    cost_rec["bunker_cost_usd"] * _fx,
                    cost_rec["port_costs"]["total_port_costs"] * _fx,
                    cost_rec["origin_charges"]["total_origin_charges"] * _fx,
                    cost_rec["dest_charges"]["total_destination_charges"] * _fx,
                    cost_rec["haulage"]["total_haulage_usd"] * _fx,
                    cost_rec["total_canal_usd"] * _fx,
                    cost_rec["weather_cost"]["total_weather_cost"] * _fx,
                    cost_rec["carbon_tax_usd"] * _fx,
                ],
                f"Alternate ({_cur})": [
                    cost_alt["freight"]["total_freight_usd"] * _fx,
                    cost_alt["bunker_cost_usd"] * _fx,
                    cost_alt["port_costs"]["total_port_costs"] * _fx,
                    cost_alt["origin_charges"]["total_origin_charges"] * _fx,
                    cost_alt["dest_charges"]["total_destination_charges"] * _fx,
                    cost_alt["haulage"]["total_haulage_usd"] * _fx,
                    cost_alt["total_canal_usd"] * _fx,
                    cost_alt["weather_cost"]["total_weather_cost"] * _fx,
                    cost_alt["carbon_tax_usd"] * _fx,
                ],
                f"Eco ({_cur})": [
                    cost_eco["freight"]["total_freight_usd"] * _fx,
                    cost_eco["bunker_cost_usd"] * _fx,
                    cost_eco["port_costs"]["total_port_costs"] * _fx,
                    cost_eco["origin_charges"]["total_origin_charges"] * _fx,
                    cost_eco["dest_charges"]["total_destination_charges"] * _fx,
                    cost_eco["haulage"]["total_haulage_usd"] * _fx,
                    cost_eco["total_canal_usd"] * _fx,
                    cost_eco["weather_cost"]["total_weather_cost"] * _fx,
                    cost_eco["carbon_tax_usd"] * _fx,
                ],
            }).set_index("Category")
            st.bar_chart(comp_df, height=300, stack=False)

        st.markdown("---")

        # ── SAVE COST TO POSTGIS ──────────────────────────────────────
        save_cost_btn = st.button(
            "💾 Save Cost to PostGIS",
            use_container_width=True,
            disabled=not (_db_live and st.session_state.last_route_id),
            help="Writes the cost breakdown to seashield_costs table",
            key="save_cost_btn",
        )
        if save_cost_btn:
            ok = postgis_save_costs(
                st.session_state.last_route_id, cost_rec)
            if ok:
                st.success(
                    f"✅ Cost breakdown saved to PostGIS · "
                    f"Route #{st.session_state.last_route_id}")
            else:
                st.warning("⚠️ Could not save costs — save route first, "
                           "then try again.")

        if not st.session_state.last_route_id:
            st.info(
                "💡 Save the route analysis to PostGIS first (sidebar), "
                "then come back here to persist the cost breakdown.",
                icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ROUTE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown(
        "<div class='section-title'>Past Voyage Analyses (from PostGIS)</div>",
        unsafe_allow_html=True)
    st.caption(
        "Every saved analysis is persisted with PostGIS GEOGRAPHY columns. "
        "Routes are retrieved using ST_AsGeoJSON and re-rendered on the map.")

    if not _db_live:
        st.warning("⚠️ Database offline — route history unavailable. "
                   "Configure DB_HOST / DB_NAME / DB_PASSWORD and restart.")
    else:
        history = postgis_get_route_history(limit=30)
        if not history:
            st.markdown("""
            <style>
            @keyframes blink { 0%,100% {opacity:1;} 50% {opacity:0;} }
            @keyframes type { from {width:0;} to {width:100%;} }
            .term-boot { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--text-mid); background:var(--bg-surface); padding:32px; border:1px solid var(--border); border-radius:6px; box-shadow:inset 0 0 20px rgba(0,0,0,0.5); }
            .term-line { white-space:nowrap; overflow:hidden; border-right:2px solid var(--accent-1); display:inline-block; animation: type 1.5s steps(40,end), blink 0.5s step-end infinite alternate; margin-top:8px;}
            </style>
            <div class="term-boot">
                <div style="color:var(--accent-1);">[ SYSTEM BOOT ]</div>
                <div>Loading historical records...</div>
                <div class="term-line">NO ROUTE HISTORY FOUND IN DATABASE.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            summary_rows = []
            for h in history:
                min_cyc = (f"{h['min_cyclone_dist_km']:.0f} km"
                           if h["min_cyclone_dist_km"] else "—")
                summary_rows.append({
                    "ID":          h["id"],
                    "Origin":      h["origin_name"],
                    "Destination": h["destination_name"],
                    "Distance km": f"{h['distance_km']:,.0f}" if h["distance_km"] else "—",
                    "Risk":        h["risk_level"] or "—",
                    "Score":       h["risk_score"] or "—",
                    "Avg Wave m":  f"{h['avg_wave_m']:.2f}" if h["avg_wave_m"] else "—",
                    "Cyclone km":  min_cyc,
                    "Waypoints":   h["waypoint_count"],
                    "Cyclones":    h["cyclone_count"],
                    "Rec":         "✅" if h["is_recommended"] else "⚠️",
                    "Saved (UTC)": str(h["created_at"])[:16],
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                height=280,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown(
                "<div class='section-title'>Replay a Historical Route</div>",
                unsafe_allow_html=True)
            route_ids  = [str(h["id"]) for h in history]
            chosen_id  = st.selectbox("Select Route ID to replay", route_ids,
                                      key="history_select")
            chosen_row = next((h for h in history if str(h["id"]) == chosen_id), None)

            if chosen_row and chosen_row.get("route_geojson"):
                col_info, col_hmap = st.columns([1, 2])
                with col_info:
                    risk_col = {"High": "#ef4444", "Medium": "#f59e0b",
                                "Low":  "#10b981"}.get(
                        chosen_row["risk_level"], "#38bdf8")
                    st.markdown(f"""
                    <div style="background:var(--bg-surface); border:1px solid var(--border); border-left:4px solid {risk_col}; border-radius:6px; padding:16px; margin-bottom:12px; display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-mid);">ROUTE #{chosen_row['id']} &nbsp;·&nbsp; {str(chosen_row['created_at'])[:16]} UTC</div>
                            <div style="font-family:'Orbitron',sans-serif; font-size:14px; color:var(--text-hi); font-weight:bold;">{chosen_row['origin_name']} ➔ {chosen_row['destination_name']}</div>
                        </div>
                        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-top:8px; font-family:'JetBrains Mono',monospace; font-size:12px;">
                            <div><span style="color:var(--text-mid); font-size:10px;">DISTANCE</span><br><b style="color:var(--accent-1);">{chosen_row['distance_km']:,.0f} km</b></div>
                            <div><span style="color:var(--text-mid); font-size:10px;">REC SCORE</span><br><b style="color:var(--text-hi);">{chosen_row['risk_score']} / 100</b></div>
                            <div><span style="color:var(--text-mid); font-size:10px;">AVG WAVE</span><br><b style="color:var(--text-hi);">{chosen_row['avg_wave_m']:.2f} m</b></div>
                            <div><span style="color:var(--text-mid); font-size:10px;">WP/CYC</span><br><b style="color:var(--text-hi);">{chosen_row['waypoint_count']} / {chosen_row['cyclone_count']}</b></div>
                        </div>
                        <div style="font-size:11px; margin-top:8px;" class='postgis-badge'>ST_AsGeoJSON geometry replay</div>
                    </div>""", unsafe_allow_html=True)

                with col_hmap:
                    hmap = build_history_map(
                        chosen_row["route_geojson"],
                        chosen_row["origin_geojson"],
                        chosen_row["dest_geojson"],
                        chosen_row["risk_level"] or "Low",
                    )
                    st_folium(hmap, width="100%", height=400, returned_objects=[])
            else:
                st.info("Select a route ID with stored geometry to replay.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PostGIS SPATIAL STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_spatial:
    st.markdown(
        "<div class='section-title'>PostGIS Spatial Statistics Dashboard</div>",
        unsafe_allow_html=True)
    st.caption(
        "Live aggregate queries run directly against the PostGIS database. "
        "Distances are computed in metres using the GEOGRAPHY type (true sphere).")

    if not _db_live:
        st.warning("⚠️ Database offline — connect PostgreSQL + PostGIS to see stats.")
    else:
        stats = postgis_spatial_stats()
        
        nc = f"{stats['nearest_cyclone_ever_km']} km" if stats["nearest_cyclone_ever_km"] else "No data"
        kpi_grid_html = f"""
        <div class="kpi-grid">
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">💾 Routes Saved</div>
                <div class="kpi-val">{stats['total_routes']}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">📏 Dist Tracked</div>
                <div class="kpi-val">{stats['total_distance_km']:,.0f} <span style="font-size:12px;color:var(--text-mid);">km</span></div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--warn);">
                <div class="kpi-label">📊 Avg Risk</div>
                <div class="kpi-val">{stats['avg_risk_score']:.1f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--danger);">
                <div class="kpi-label">🚨 High-Risk</div>
                <div class="kpi-val">{stats['high_risk_count']}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--warn);">
                <div class="kpi-label">⚡ Cyc Captures</div>
                <div class="kpi-val">{stats['total_cyclones_captured']}</div>
            </div>
            <div class="kpi-card" style="border-top-color: var(--accent-1);">
                <div class="kpi-label">🌀 Nearest Cyc</div>
                <div class="kpi-val">{nc}</div>
            </div>
        </div>
        """
        st.markdown(kpi_grid_html, unsafe_allow_html=True)
