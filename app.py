"""
HyPIS Ug — PRODUCTION v4.0
═══════════════════════════════════════════════════════════════════════════════
Multi-source weather: ERA5 Archive (Open-Meteo) + NWP Forecast (Open-Meteo)
  + OpenWeather current conditions — all three averaged where available.
Data attribution: Open-Meteo (ERA5 + ICON + GFS; nearest stations via data
  assimilation) · OpenWeather current snapshot.
FAO-56 Penman-Monteith ET₀ — parameters as per FAO-56 / NASA-Power training.
XGBoost ML model v4 trained on NASA-Power Uganda dataset (552,258 rows, 1990-2025).
  — Reads model BUNDLE (feature_cols + target + metrics) saved by Colab trainer.
  — Adapts prediction flow based on MODEL_TARGET (ETO_PM or IWR or others).
HWSD v2 soil database for Uganda districts.
Hourly cache-bust · Real-time RH min/mean/max · MAD water balance.
7-day rolling rain / ET₀ / drystreak computation fed to XGBoost features.
Author: Prosper BYARUHANGA · HyPIS App v4.0
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time as _time
import subprocess
import sys

# ── Auto-install optional packages ───────────────────────────────────────────
for _pkg in ("joblib", "xgboost"):
    try:
        __import__(_pkg)
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", _pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HyPIS Ug",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — v4 BUNDLE AWARE
# Searches for the .pkl saved by the Colab trainer (bundle dict).
# Also handles legacy plain-model pkl for backward compatibility.
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))

# Search order for the model file
_MODEL_CANDIDATES = [
    # Colab trainer saves: xgboost_{target.lower()}_uganda.pkl
    os.path.join(_HERE, "xgboost_eto_pm_uganda.pkl"),
    os.path.join(_HERE, "xgboost_iwr_uganda.pkl"),
    os.path.join(_HERE, "xgboost_prectotcorr_uganda.pkl"),
    # Legacy name (old app)
    os.path.join(_HERE, "irrigation_scheduling_model.pkl"),
    # Generic fallback
    os.path.join(_HERE, "model.pkl"),
]

MODEL_PATH   = None
MODEL        = None           # the XGBoost estimator object
MODEL_FEATURES = []           # list of feature column names (in training order)
MODEL_TARGET = "IWR"          # what the model predicts (ETO_PM / IWR / …)
MODEL_METRICS = {}            # {"mae":…, "rmse":…, "r2":…, "mape":…}
MODEL_ERROR  = ""

for _c in _MODEL_CANDIDATES:
    if os.path.exists(_c):
        MODEL_PATH = _c
        break

try:
    import joblib
    import xgboost  # noqa: F401

    if MODEL_PATH:
        _raw = joblib.load(MODEL_PATH)

        # ── v4 Bundle (dict) ──────────────────────────────────────────────────
        if isinstance(_raw, dict) and "model" in _raw:
            MODEL          = _raw["model"]
            MODEL_FEATURES = _raw.get("feature_cols", [])
            MODEL_TARGET   = _raw.get("target", "ETO_PM")
            MODEL_METRICS  = _raw.get("metrics", {})
        # ── Legacy plain XGBoost model ────────────────────────────────────────
        else:
            MODEL          = _raw
            MODEL_TARGET   = "IWR"
            MODEL_FEATURES = []   # legacy: will use full row build
    else:
        MODEL_ERROR = "No .pkl model found — using FAO-56 fallback only."
except ModuleNotFoundError as e:
    MODEL_ERROR = f"xgboost/joblib not installed ({e}) — FAO-56 fallback"
except Exception as e:
    MODEL_ERROR = f"Model load error: {e} — FAO-56 fallback"

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (identical to v3 — no visual regression)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """<style>
:root{
  --hb:#1a5fc8;--hg:#0b6b1b;--hr:#b81c1c;
  --bg:#f4f8f2;--sf:#fff;--bd:#dbe9db;
  --tx:#17301b;--gn:#0b6b1b;--gd:#075214;--gs:#e7f3e6;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]
  {background:var(--bg)!important;color:var(--tx)!important;}
[data-testid="stHeader"],[data-testid="stToolbar"]
  {background:transparent!important;}
[data-testid="stMetric"]
  {background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:.5rem .7rem;}
[data-testid="stMetricLabel"] p{font-size:.76rem!important;margin:0!important;}
[data-testid="stMetricValue"] div{font-size:1.05rem!important;font-weight:700!important;}
[data-testid="stMetricDelta"] div{font-size:.72rem!important;}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"]
  {color:var(--tx)!important;}
div[data-baseweb="tab-list"]{gap:.3rem;background:transparent!important;}
button[data-baseweb="tab"]{
  background:var(--gs)!important;border:1px solid #b8d1b8!important;
  border-radius:999px!important;color:var(--gd)!important;
  padding:.35rem .75rem!important;font-size:.83rem!important;
}
button[data-baseweb="tab"]>div{color:var(--gd)!important;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"]
  {background:var(--gn)!important;border-color:var(--gn)!important;}
button[data-baseweb="tab"][aria-selected="true"]>div{color:#fff!important;}
[data-baseweb="select"]>div,div[data-baseweb="input"]>div,
.stNumberInput>div>div,.stTextInput>div>div
  {background:var(--sf)!important;color:var(--tx)!important;border-color:#c9d9c9!important;}
.stButton>button,.stDownloadButton>button
  {background:var(--gn)!important;color:#fff!important;
   border:1px solid var(--gn)!important;border-radius:10px!important;}
.stButton>button:hover{background:var(--gd)!important;}
.stAlert{border-radius:12px!important;}
.stSuccess{background:#e8f6ea!important;color:#12361a!important;}
.stWarning{background:#fff5df!important;color:#5a4308!important;}
.stError{background:#fdeaea!important;color:#5a1616!important;}
.stInfo{background:#eaf3ff!important;color:#14324d!important;}
section[data-testid="stSidebar"]{background:#eef5ec!important;}
button[title="Fork this app"],[data-testid="stToolbarActionButtonIcon"],
[data-testid="stBottomBlockContainer"],.stDeployButton,footer
  {display:none!important;}
.block-container{padding-top:.8rem!important;}
/* ── Brand header ── */
.hx-outer{
  border-radius:20px;overflow:hidden;margin:0 0 10px 0;
  background:linear-gradient(90deg,
    var(--hb) 0%,var(--hb) 33.3%,
    var(--hg) 33.3%,var(--hg) 66.6%,
    var(--hr) 66.6%,var(--hr) 100%);
  padding:9px 9px 7px 9px;
}
.hx-panel{
  background:#fff;border:2px solid #d0ddd0;
  border-radius:14px;padding:9px 16px 7px 16px;
}
.hx-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.hx-wm{
  font-family:Georgia,serif;font-size:2.4rem;font-weight:700;
  line-height:1;letter-spacing:-1px;flex-shrink:0;
}
.hx-wm .H{color:#1a5fc8;}
.hx-wm .y{color:#0b6b1b;}
.hx-wm .P{color:#b81c1c;}
.hx-wm .I{color:#1a5fc8;}
.hx-wm .S{color:#0b6b1b;}
.hx-wm .Ug{color:#b81c1c;font-size:1.4rem;vertical-align:middle;margin-left:4px;}
.hx-sub{font-family:Georgia,serif;font-size:.95rem;flex:1 1 160px;color:#444;}
.hx-sub .sh{color:#1a5fc8;}
.hx-sub .sd{color:#aaa;margin:0 4px;}
.hx-sub .si{color:#0b6b1b;}
.hx-auth{margin:4px 0 0 4px;font-family:Georgia,serif;font-size:.78rem;color:#ddd;}
.hx-auth strong{color:#fff;}
/* ── Live dot ── */
.live-dot{
  width:7px;height:7px;background:#22c55e;border-radius:50%;
  display:inline-block;margin-right:4px;
  animation:blink 1.4s infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
/* ── Soil-moisture bar ── */
.sm-bar-wrap{background:#ddd;border-radius:6px;height:18px;width:100%;}
.sm-bar-fill{height:18px;border-radius:6px;transition:width .4s;}
/* ── RH pills ── */
.rh-row{display:flex;gap:6px;margin:4px 0;}
.rh-box{
  flex:1;text-align:center;border-radius:8px;
  padding:5px 2px;font-size:.82rem;font-weight:600;
}
.rh-min {background:#dbeafe;color:#1e40af;}
.rh-mean{background:#d1fae5;color:#065f46;}
.rh-max {background:#fef9c3;color:#92400e;}
/* ── Model info badge ── */
.ml-badge{
  background:#e8f6ea;border:1px solid #a8d8a8;border-radius:10px;
  padding:6px 14px;font-size:.82rem;color:#073f12;margin:4px 0;
}
@media(max-width:640px){
  .hx-wm{font-size:1.7rem!important;}
  .hx-wm .Ug{font-size:1rem!important;}
  .hx-sub{font-size:.78rem!important;}
}
</style>""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH every 1 hour
# ══════════════════════════════════════════════════════════════════════════════
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = _time.time()

_el = _time.time() - st.session_state["last_refresh"]
if _el >= 3600:
    st.cache_data.clear()
    st.session_state["last_refresh"] = _time.time()
    st.rerun()

_rem = max(0, 3600 - int(_el))

# ── Brand header ──────────────────────────────────────────────────────────────
st.markdown(
    """<div class="hx-outer"><div class="hx-panel"><div class="hx-row">
<span style="font-size:1.5rem;">&#127807;</span>
<span class="hx-wm">
  <span class="H">H</span><span class="y">y</span><span class="P">P</span>
  <span class="I">I</span><span class="S">S</span><span class="Ug"> Ug</span>
</span>
<span class="hx-sub">
  <span class="sh">HydroPredict</span>
  <span class="sd">&#xB7;</span>
  <span class="si">IrrigSched</span>
</span>
</div></div>
<div class="hx-auth">by: Prosper <strong>BYARUHANGA</strong> &nbsp;&#xB7;&nbsp; HyPIS App v4.0</div>
</div>""",
    unsafe_allow_html=True,
)

_now = datetime.now().strftime("%d %b %Y %H:%M")
st.caption(
    f'<span class="live-dot"></span> Live &middot; <b>{_now}</b>'
    f" &nbsp;&middot;&nbsp; Refresh in <b>{_rem // 3600}h {(_rem % 3600) // 60}m</b>",
    unsafe_allow_html=True,
)

# ── Model status banner ───────────────────────────────────────────────────────
if MODEL is None:
    st.warning(f"⚠️ **ML Model:** {MODEL_ERROR}")
else:
    _r2   = MODEL_METRICS.get("r2", "–")
    _rmse = MODEL_METRICS.get("rmse", "–")
    _mae  = MODEL_METRICS.get("mae", "–")
    _feat_n = len(MODEL_FEATURES) if MODEL_FEATURES else "legacy"
    st.markdown(
        f'<div class="ml-badge">✅ <b>XGBoost v4 loaded</b> · '
        f'<code>{os.path.basename(MODEL_PATH)}</code> · '
        f'Target: <b>{MODEL_TARGET}</b> · '
        f'Features: <b>{_feat_n}</b> · '
        f'R²: <b>{_r2}</b> · RMSE: <b>{_rmse}</b> · MAE: <b>{_mae}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# HWSD v2 SOIL DATABASE — Uganda districts
# ══════════════════════════════════════════════════════════════════════════════
HWSD_SOILS = {
    "Kabuyanda, Isingiro": {"fc": 0.28, "pwp": 0.14, "source": "HWSD v2 – Loam"},
    "Makerere Uni MAIN":   {"fc": 0.32, "pwp": 0.18, "source": "HWSD v2 – Clay Loam"},
    "MUARiK":              {"fc": 0.26, "pwp": 0.12, "source": "HWSD v2 – Sandy Clay Loam"},
    "Mbarara":             {"fc": 0.30, "pwp": 0.15, "source": "HWSD v2 – Loam"},
    "Gulu":                {"fc": 0.24, "pwp": 0.11, "source": "HWSD v2 – Sandy Loam"},
    "Jinja":               {"fc": 0.31, "pwp": 0.16, "source": "HWSD v2 – Clay Loam"},
    "Mbale":               {"fc": 0.27, "pwp": 0.13, "source": "HWSD v2 – Loam"},
    "Kabale":              {"fc": 0.33, "pwp": 0.19, "source": "HWSD v2 – Clay"},
    "Fort Portal":         {"fc": 0.29, "pwp": 0.14, "source": "HWSD v2 – Loam"},
    "Masaka":              {"fc": 0.25, "pwp": 0.12, "source": "HWSD v2 – Sandy Loam"},
    "Lira":                {"fc": 0.23, "pwp": 0.10, "source": "HWSD v2 – Sandy Loam"},
    "Soroti":              {"fc": 0.22, "pwp": 0.09, "source": "HWSD v2 – Sandy Loam"},
    "Arua":                {"fc": 0.21, "pwp": 0.08, "source": "HWSD v2 – Sand"},
    "Hoima":               {"fc": 0.28, "pwp": 0.13, "source": "HWSD v2 – Sandy Loam"},
    "Kasese":              {"fc": 0.35, "pwp": 0.20, "source": "HWSD v2 – Clay"},
    "Tororo":              {"fc": 0.26, "pwp": 0.12, "source": "HWSD v2 – Sandy Clay Loam"},
}


def get_soil_for_location(loc: str) -> dict:
    return HWSD_SOILS.get(loc, {"fc": 0.28, "pwp": 0.14, "source": "HWSD v2 – Default Loam"})


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER — UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _safe_f(v, fallback=None):
    try:
        r = float(v)
        return r if not np.isnan(r) else fallback
    except Exception:
        return fallback


def _avg_sources(*vals, fallback=None):
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(sum(clean) / len(clean), 2) if clean else fallback


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — ERA5 ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_archive_today(lat: float, lon: float, cache_hour: str):
    today_str = datetime.today().strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={today_str}&end_date={today_str}"
            f"&hourly=temperature_2m,precipitation,relative_humidity_2m,"
            f"windspeed_10m,shortwave_radiation"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"shortwave_radiation_sum,relative_humidity_2m_max,"
            f"relative_humidity_2m_min,windspeed_10m_max"
            f"&timezone=Africa%2FNairobi",
            timeout=15,
        ).json()

        if "daily" not in resp:
            return None

        dd = resp["daily"]
        tmax    = _safe_f(dd["temperature_2m_max"][0], 28.0)
        tmin    = _safe_f(dd["temperature_2m_min"][0], 16.0)
        rh_max  = _safe_f(dd["relative_humidity_2m_max"][0], 70.0)
        rh_min  = _safe_f(dd["relative_humidity_2m_min"][0], 50.0)
        wind_kh = _safe_f(dd["windspeed_10m_max"][0], 7.2)
        rs      = _safe_f(dd["shortwave_radiation_sum"][0], 18.0)

        if "hourly" in resp and "precipitation" in resp["hourly"]:
            cur_hr = datetime.now().hour
            precip = sum(_safe_f(p, 0.0) for p in resp["hourly"]["precipitation"][: cur_hr + 1])
        else:
            precip = _safe_f(dd["precipitation_sum"][0], 0.0)

        cloud_cover = None
        if "hourly" in resp and "cloudcover" in resp["hourly"]:
            cur_hr = datetime.now().hour
            cc_vals = [_safe_f(v) for v in resp["hourly"]["cloudcover"][: cur_hr + 1]
                       if _safe_f(v) is not None]
            cloud_cover = round(sum(cc_vals) / len(cc_vals), 0) if cc_vals else None

        return {
            "source":      "ERA5 Archive (Observed)",
            "tmax":        tmax,
            "tmin":        tmin,
            "rh_max":      rh_max,
            "rh_min":      rh_min,
            "wind":        round(wind_kh / 3.6, 3),
            "rs":          rs,
            "precip":      round(precip, 2),
            "cloud_cover": cloud_cover,
            "is_actual":   True,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — NWP FORECAST (Open-Meteo ICON + GFS)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast_today(lat: float, lon: float, cache_hour: str):
    try:
        resp = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "shortwave_radiation_sum,relative_humidity_2m_max,"
            "relative_humidity_2m_min,windspeed_10m_max,cloudcover_mean"
            "&timezone=Africa%2FNairobi&forecast_days=1",
            timeout=10,
        ).json()

        if "daily" not in resp:
            return None

        dd = resp["daily"]
        cloud_cover = None
        if "cloudcover_mean" in dd and dd["cloudcover_mean"]:
            cloud_cover = _safe_f(dd["cloudcover_mean"][0])

        return {
            "source":      "Open-Meteo NWP Forecast (ICON+GFS)",
            "tmax":        _safe_f(dd["temperature_2m_max"][0], 28.0),
            "tmin":        _safe_f(dd["temperature_2m_min"][0], 16.0),
            "rh_max":      _safe_f(dd["relative_humidity_2m_max"][0], 70.0),
            "rh_min":      _safe_f(dd["relative_humidity_2m_min"][0], 50.0),
            "wind":        round(_safe_f(dd["windspeed_10m_max"][0], 7.2) / 3.6, 3),
            "rs":          _safe_f(dd["shortwave_radiation_sum"][0], 18.0),
            "precip":      _safe_f(dd["precipitation_sum"][0], 0.0),
            "cloud_cover": cloud_cover,
            "is_actual":   False,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — OPENWEATHER current snapshot
# ══════════════════════════════════════════════════════════════════════════════
OW_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openweather(lat: float, lon: float, cache_hour: str):
    if OW_API_KEY in ("", "YOUR_OPENWEATHER_API_KEY"):
        return None
    try:
        resp = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric",
            timeout=10,
        ).json()
        if "main" not in resp:
            return None

        temp     = _safe_f(resp["main"].get("temp"), 25.0)
        temp_max = _safe_f(resp["main"].get("temp_max"), temp)
        temp_min = _safe_f(resp["main"].get("temp_min"), temp)
        rh       = _safe_f(resp["main"].get("humidity"), 60.0)
        wind_ms  = _safe_f(resp["wind"].get("speed"), 2.0)
        precip   = _safe_f((resp.get("rain") or {}).get("1h"), 0.0)
        clouds   = _safe_f((resp.get("clouds") or {}).get("all"), None)

        return {
            "source":      "OpenWeather Current",
            "tmax":        temp_max,
            "tmin":        temp_min,
            "temp_now":    temp,
            "rh_max":      rh,
            "rh_min":      max(0.0, rh - 15.0),
            "wind":        round(wind_ms, 3),
            "rs":          18.0,
            "precip":      precip,
            "cloud_cover": clouds,
            "is_actual":   True,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Open-Meteo current conditions (live snapshot)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def get_current_conditions(lat: float, lon: float, cache_hour: str):
    try:
        resp = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,weather_code,"
            "wind_speed_10m,precipitation,cloudcover",
            timeout=10,
        ).json()
        if "current" not in resp:
            return None
        c = resp["current"]
        return {
            "temp":         round(_safe_f(c.get("temperature_2m"), 25.0), 1),
            "humidity":     _safe_f(c.get("relative_humidity_2m"), 60.0),
            "wind":         round(_safe_f(c.get("wind_speed_10m"), 2.0), 1),
            "precip_now":   round(_safe_f(c.get("precipitation"), 0.0), 1),
            "weather_code": c.get("weather_code", 0),
            "cloud_cover":  _safe_f(c.get("cloudcover"), None),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL SOURCES
# ══════════════════════════════════════════════════════════════════════════════
def build_merged_weather(arc, fc_d, ow):
    sources = [s for s in [arc, fc_d, ow] if s is not None]
    n = len(sources)

    if n == 0:
        return None, "No weather data available"

    keys = ["tmax", "tmin", "rh_max", "rh_min", "wind"]
    merged = {}
    for k in keys:
        vals = [s[k] for s in sources if s.get(k) is not None]
        merged[k] = _avg_sources(*vals)

    rs_sources = [s["rs"] for s in [arc, fc_d] if s is not None and s.get("rs") is not None]
    merged["rs"] = _avg_sources(*rs_sources) if rs_sources else 18.0

    if arc and arc.get("precip") is not None:
        merged["precip"] = arc["precip"]
    else:
        precip_vals = [s["precip"] for s in sources if s.get("precip") is not None]
        merged["precip"] = _avg_sources(*precip_vals, fallback=0.0)

    cc_vals = [s["cloud_cover"] for s in sources if s.get("cloud_cover") is not None]
    merged["cloud_cover"] = round(sum(cc_vals) / len(cc_vals), 0) if cc_vals else None
    merged["rh_mean"]  = round((merged["rh_max"] + merged["rh_min"]) / 2, 1)
    merged["is_actual"] = arc is not None

    src_names = " ⊕ ".join(s["source"] for s in sources)
    if n == 3:
        quality = f"✅ 3-source averaged (ERA5 + NWP + OW) · {src_names}"
    elif arc is not None and fc_d is not None:
        quality = f"✅ OBSERVED + FORECAST averaged · {src_names}"
    elif arc is not None:
        quality = f"✅ ERA5 Observed only · {src_names}"
    else:
        quality = f"🔮 Forecast only · {src_names}"

    return merged, quality


# ══════════════════════════════════════════════════════════════════════════════
# 5-DAY FORECAST — Open-Meteo
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def get_om_fc(lat: float, lon: float, cache_hour: str):
    try:
        resp = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "shortwave_radiation_sum,relative_humidity_2m_max,"
            "relative_humidity_2m_min,windspeed_10m_max,weathercode"
            "&timezone=Africa%2FNairobi&forecast_days=5",
            timeout=12,
        ).json()

        if "daily" not in resp:
            return None

        dd = resp["daily"]
        n  = len(dd["time"])
        rows = []
        for i in range(n):
            tmax   = _safe_f(dd["temperature_2m_max"][i], 28.0)
            tmin   = _safe_f(dd["temperature_2m_min"][i], 16.0)
            rh_max = _safe_f(dd["relative_humidity_2m_max"][i], 70.0)
            rh_min = _safe_f(dd["relative_humidity_2m_min"][i], 50.0)
            wind   = _safe_f(dd["windspeed_10m_max"][i], 7.2)
            rs     = _safe_f(dd["shortwave_radiation_sum"][i], 18.0)
            precip = _safe_f(dd["precipitation_sum"][i], 0.0)
            rows.append({
                "date":          dd["time"][i],
                "tmax":          tmax,
                "tmin":          tmin,
                "rh_max":        rh_max,
                "rh_min":        rh_min,
                "rh_mean":       round((rh_max + rh_min) / 2, 1),
                "wind":          round(wind / 3.6, 3),
                "rs":            rs,
                "precipitation": precip,
                "wcode":         (dd.get("weathercode") or [0] * n)[i],
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.dropna(subset=["tmax", "tmin", "rh_max", "rh_min", "wind", "rs"], inplace=True)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"❌ Forecast API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL ARCHIVE — ERA5 daily
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def get_arc(lat: float, lon: float, s: str, e: str):
    try:
        resp = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={s}&end_date={e}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"precipitation_sum,shortwave_radiation_sum,"
            f"relative_humidity_2m_max,relative_humidity_2m_min,"
            f"windspeed_10m_max"
            f"&timezone=Africa%2FNairobi",
            timeout=20,
        ).json()

        if "daily" not in resp:
            return None

        dd = resp["daily"]
        df = pd.DataFrame({
            "date":          pd.to_datetime(dd["time"]),
            "tmax":          [_safe_f(x) for x in dd["temperature_2m_max"]],
            "tmin":          [_safe_f(x) for x in dd["temperature_2m_min"]],
            "rh_max":        [_safe_f(x) for x in dd["relative_humidity_2m_max"]],
            "rh_min":        [_safe_f(x) for x in dd["relative_humidity_2m_min"]],
            "wind":          [
                _safe_f(x, 0) / 3.6 if _safe_f(x) is not None else np.nan
                for x in dd["windspeed_10m_max"]
            ],
            "rs":            [_safe_f(x) for x in dd["shortwave_radiation_sum"]],
            "precipitation": [_safe_f(x, 0.0) for x in dd["precipitation_sum"]],
        })
        df["rh_mean"] = (df["rh_max"] + df["rh_min"]) / 2
        df.dropna(subset=["tmax", "tmin", "rh_max", "rh_min", "wind", "rs"], inplace=True)
        if df.empty:
            st.error("❌ No valid historical data for this period.")
            return None
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Archive API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FAO-56 PENMAN-MONTEITH ET₀ (exact FAO-56 §3.3.2)
# ══════════════════════════════════════════════════════════════════════════════
def et0_fao56_pm(
    tmax: float, tmin: float,
    rh_max: float, rh_min: float,
    u2: float, rs: float,
    elev: float, doy: int, lat_deg: float,
) -> float:
    for x in [tmax, tmin, rh_max, rh_min, u2, rs]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0

    tmax   = float(tmax)
    tmin   = float(tmin)
    rh_max = max(0.0, min(100.0, float(rh_max)))
    rh_min = max(0.0, min(100.0, float(rh_min)))
    u2     = max(0.0, float(u2))
    rs     = max(0.0, float(rs))
    doy    = int(doy) if doy else 1

    Gsc   = 0.0820
    sigma = 2.042e-10
    cn, cd = 900, 0.34
    tmean = (tmax + tmin) / 2.0

    P     = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P

    es_max = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    es_min = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    es     = (es_max + es_min) / 2.0
    ea     = max(0.0, min(((rh_max / 100.0) * es_min + (rh_min / 100.0) * es_max) / 2.0, es))
    Delta  = (4098.0 * es) / (tmean + 237.3) ** 2.0

    b     = 2.0 * np.pi * doy / 365.0
    dr    = 1.0 + 0.033 * np.cos(b)
    phi   = np.radians(abs(lat_deg))
    delta = 0.409 * np.sin(b - 1.39)
    ws    = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    Ra    = max(0.0,
        (24.0 * 60.0 / np.pi) * Gsc * dr * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(ws)
        )
    )

    Rso = max(0.0, (0.75 + 2e-5 * elev) * Ra)
    Rns = 0.77 * rs
    fcd = max(0.0, 1.35 * min(1.0, rs / max(Rso, 0.1)) - 0.35)
    Rnl = max(0.0,
        sigma
        * ((tmax + 273.16) ** 4.0 + (tmin + 273.16) ** 4.0) / 2.0
        * (0.34 - 0.14 * np.sqrt(max(0.0, ea)))
        * fcd
    )
    Rn  = max(0.0, Rns - Rnl)

    num = 0.408 * Delta * Rn + gamma * (cn / (tmean + 273.0)) * u2 * (es - ea)
    den = Delta + gamma * (1.0 + cd * u2)
    if den <= 0:
        return 0.0

    et0 = max(0.0, num / den)
    return 0.0 if np.isnan(et0) else round(et0, 3)


# ══════════════════════════════════════════════════════════════════════════════
# NEW v4 — HARGREAVES-SAMANI ET₀  (used as ETO_H feature)
# ET₀_H = 0.0023 × Ra × (Tmean + 17.8) × √(Tmax - Tmin)
# ══════════════════════════════════════════════════════════════════════════════
def et0_hargreaves(tmax: float, tmin: float, doy: int, lat_deg: float) -> float:
    """
    Hargreaves-Samani reference ET₀ (mm/day).
    Requires only temperature + day of year + latitude.
    Matches the ETO_H column from the NASA-POWER Uganda training dataset.
    """
    Gsc   = 0.0820
    b     = 2.0 * np.pi * doy / 365.0
    dr    = 1.0 + 0.033 * np.cos(b)
    phi   = np.radians(abs(lat_deg))
    delta = 0.409 * np.sin(b - 1.39)
    ws    = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    Ra    = max(0.0,
        (24.0 * 60.0 / np.pi) * Gsc * dr * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(ws)
        )
    )
    tmean = (tmax + tmin) / 2.0
    td    = max(0.0, tmax - tmin)
    return round(max(0.0, 0.0023 * Ra * (tmean + 17.8) * (td ** 0.5)), 3)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def eff_rain(p: float) -> float:
    """USDA SCS effective rainfall (FAO-56 §6.1)."""
    p = float(p) if p else 0.0
    if p <= 0:
        return 0.0
    if p <= 25.4:
        return p * (125.0 - 0.6 * p) / 125.0
    return p - 12.7 - 0.1 * p


def mad_sched(dr: float, taw: float, raw: float, pe: float, etc: float) -> dict:
    """FAO-56 MAD water balance step."""
    dr1 = min(taw, max(0.0, dr + etc))
    dr2 = max(0.0, dr1 - pe)
    dp  = max(0.0, pe - dr1)

    if dr2 <= 0.5:
        return dict(iwr=0.0, dp=dp, dr_new=0.0, status="full",
                    msg=f"Soil recharged by rain (Dr={dr2:.1f} mm ≈ FC). No irrigation.")
    if dr2 < raw:
        return dict(iwr=0.0, dp=dp, dr_new=dr2, status="ok",
                    msg=f"Dr={dr2:.1f} mm < RAW={raw:.1f} mm. Safe.")
    return dict(iwr=round(dr2, 2), dp=dp, dr_new=0.0, status="irrigate",
                msg=f"Dr={dr2:.1f} mm ≥ RAW={raw:.1f} mm. Irrigate {dr2:.2f} mm to FC.")


# ══════════════════════════════════════════════════════════════════════════════
# NEW v4 — BUILD NASA-POWER ALIGNED FEATURE DICT
# Maps live weather inputs → exact NASA-POWER column names used in training
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_dict(
    tmax: float, tmin: float,
    rh_max: float, rh_min: float,
    u2: float, rs: float,
    prec: float,
    gwt: float, gwr: float,
    dr: float,
    dt: datetime,
    lat: float, lon: float,
    rain_7d: float = 0.0,
    eto_7d:  float = 0.0,
    drystreak: int = 0,
    iwr_estimate: float = 0.0,
    eto_pm_override: float = None,
) -> dict:
    """
    Computes all features that the v4 NASA-POWER XGBoost model may require.
    Column names EXACTLY match the training dataset column headers.

    Rolling fields (rain_7d, eto_7d, drystreak) are maintained by the
    caller across the forecast/history loop.
    """
    doy   = int(dt.strftime("%j"))
    rh2m  = (rh_max + rh_min) / 2.0
    t2m   = (tmax + tmin) / 2.0

    # ET₀ PM — use override if available (avoids recomputing inside loop)
    eto_pm = (
        eto_pm_override
        if eto_pm_override is not None
        else et0_fao56_pm(tmax, tmin, rh_max, rh_min, u2, rs, 0.0, doy, lat)
    )

    # ET₀ Hargreaves (= ETO_H in dataset)
    eto_h  = et0_hargreaves(tmax, tmin, doy, lat)

    # Effective rain
    eff_r  = eff_rain(prec)

    # GWETPROF — profile wetness (weighted average, NASA-POWER depth >root)
    # Approximated as slightly lower than root-zone wetness
    gwetprof = max(0.0, min(1.0, gwr * 0.85 + 0.05))

    # PRECTOTCORR_SUM — cumulative rainfall proxy (7-day window)
    prec_sum = rain_7d

    return {
        # Date identifiers
        "YEAR":              float(dt.year),
        "MONTH":             float(dt.month),
        "DAY":               float(dt.day),
        "DOY":               float(doy),
        # Location
        "LAT":               float(lat),
        "LON":               float(lon),
        # Solar / radiation
        "ALLSKY_SFC_SW_DWN": float(rs),
        # Soil moisture (NASA-POWER GWET variables, 0–1 fractions)
        "GWETPROF":          float(gwetprof),
        "GWETROOT":          float(gwr),
        "GWETTOP":           float(gwt),
        # Precipitation
        "PRECTOTCORR":       float(prec),
        "PRECTOTCORR_SUM":   float(prec_sum),
        # Temperature
        "T2M":               float(t2m),
        "T2M_MAX":           float(tmax),
        "T2M_MIN":           float(tmin),
        # Humidity
        "RH2M":              float(rh2m),
        # Wind
        "WS2M":              float(u2),
        "WS2M_MAX":          float(min(u2 * 1.30, u2 + 3.0)),
        "WS2M_MIN":          float(max(0.0, u2 * 0.35)),
        # ET₀
        "ETO_H":             float(eto_h),
        "ETO_PM":            float(eto_pm),
        # Derived water balance
        "EFF_RAIN":          float(eff_r),
        "IWR":               float(max(0.0, iwr_estimate)),
        # 7-day rolling
        "RAIN_7D":           float(rain_7d),
        "ETO_7D":            float(eto_7d),
        "DRYSTREAK":         float(drystreak),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEW v4 — RUN ML PREDICTION (bundle-aware)
# ══════════════════════════════════════════════════════════════════════════════
def run_ml(feature_dict: dict) -> float:
    """
    Predict using the v4 XGBoost bundle.
    Selects ONLY the features in MODEL_FEATURES (correct training order).
    Returns raw prediction (ETO_PM in mm/d, IWR in mm, or whatever target).
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    if MODEL_FEATURES:
        row = {k: feature_dict.get(k, 0.0) for k in MODEL_FEATURES}
        X   = pd.DataFrame([row])[MODEL_FEATURES]
    else:
        # Legacy plain model — build the old-style row
        X = pd.DataFrame([{
            "PRECTOTCORR":       feature_dict.get("PRECTOTCORR", 0),
            "T2M":               feature_dict.get("T2M", 22),
            "RH2M":              feature_dict.get("RH2M", 65),
            "WS2M":              feature_dict.get("WS2M", 2),
            "ALLSKY_SFC_SW_DWN": feature_dict.get("ALLSKY_SFC_SW_DWN", 18),
            "GWETTOP":           feature_dict.get("GWETTOP", 0.5),
            "GWETROOT":          feature_dict.get("GWETROOT", 0.5),
            "T2M_MAX":           feature_dict.get("T2M_MAX", 28),
            "T2M_MIN":           feature_dict.get("T2M_MIN", 16),
            "WS2M_MAX":          feature_dict.get("WS2M_MAX", 3),
            "WS2M_MIN":          feature_dict.get("WS2M_MIN", 0.7),
            "DOY":               feature_dict.get("DOY", 180),
            "MONTH":             feature_dict.get("MONTH", 6),
            "ETO_PM":            feature_dict.get("ETO_PM", 4),
            "IWR":               feature_dict.get("IWR", 0),
        }])

    pred = float(MODEL.predict(X)[0])
    return round(max(0.0, pred), 4)


# ══════════════════════════════════════════════════════════════════════════════
# NEW v4 — INTERPRET ML OUTPUT BASED ON MODEL TARGET
# ══════════════════════════════════════════════════════════════════════════════
def ml_to_iwr(
    pred: float,
    kc: float,
    dr: float,
    taw: float,
    raw: float,
    pe: float,
    tmax: float, tmin: float,
    rh_max: float, rh_min: float,
    u2: float, rs: float,
    elev: float, doy: int, lat: float,
) -> tuple:
    """
    Converts ML raw prediction → IWR (mm/day) based on MODEL_TARGET.

    Returns (iwr_mm, label_string)

    If target = ETO_PM: ML gives ET₀ → ETc = Kc × ET₀_ml → MAD water balance
    If target = IWR:    ML gives IWR directly — clamp only to [0, taw].
                        Do NOT clamp by (dr - pe): that zeroes out Days 2–5
                        after soil is recharged on Day 1, causing the
                        "first-day-only IWR" bug.
    Other targets:      treat as IWR with same logic.
    """
    if MODEL_TARGET == "ETO_PM":
        # ML predicted ET₀; propagate through ETc → MAD
        etc_ml = kc * pred
        sch_ml = mad_sched(dr, taw, raw, pe, etc_ml)
        iwr    = sch_ml["iwr"]
        label  = f"XGBoost ML ET₀={pred:.3f} mm/d → ETc={etc_ml:.3f} mm/d (FAO-56 MAD)"
    elif MODEL_TARGET in ("IWR", "iwr"):
        # Physical bounds only: non-negative, cannot exceed total available water
        iwr   = min(max(0.0, pred), taw)
        label = "XGBoost ML IWR (NASA-Power Uganda 552k rows)"
    else:
        iwr   = min(max(0.0, pred), taw)
        label = f"XGBoost ML {MODEL_TARGET} (NASA-Power Uganda)"

    return round(iwr, 3), label


# ══════════════════════════════════════════════════════════════════════════════
# CROP PARAMETERS (FAO-56 Table 12 — Uganda calibrated)
# ══════════════════════════════════════════════════════════════════════════════
CP = {
    "Tomatoes":       {"ini": 0.60, "mid": 1.15, "end": 0.80, "zr": 0.70, "mad": 0.40},
    "Cabbages":       {"ini": 0.70, "mid": 1.05, "end": 0.95, "zr": 0.50, "mad": 0.45},
    "Maize":          {"ini": 0.30, "mid": 1.20, "end": 0.60, "zr": 1.00, "mad": 0.55},
    "Beans":          {"ini": 0.40, "mid": 1.15, "end": 0.75, "zr": 0.60, "mad": 0.45},
    "Rice":           {"ini": 1.05, "mid": 1.30, "end": 0.95, "zr": 0.50, "mad": 0.20},
    "Potatoes":       {"ini": 0.50, "mid": 1.15, "end": 0.75, "zr": 0.60, "mad": 0.35},
    "Onions":         {"ini": 0.70, "mid": 1.05, "end": 0.95, "zr": 0.30, "mad": 0.30},
    "Peppers":        {"ini": 0.60, "mid": 1.10, "end": 0.80, "zr": 0.50, "mad": 0.30},
    "Cassava":        {"ini": 0.40, "mid": 0.85, "end": 0.70, "zr": 1.00, "mad": 0.60},
    "Bananas":        {"ini": 0.50, "mid": 1.00, "end": 0.80, "zr": 0.90, "mad": 0.35},
    "Wheat":          {"ini": 0.70, "mid": 1.15, "end": 0.40, "zr": 1.00, "mad": 0.55},
    "Sorghum":        {"ini": 0.30, "mid": 1.00, "end": 0.55, "zr": 1.00, "mad": 0.55},
    "Groundnuts":     {"ini": 0.40, "mid": 1.15, "end": 0.75, "zr": 0.50, "mad": 0.50},
    "Sweet Potatoes": {"ini": 0.50, "mid": 1.15, "end": 0.75, "zr": 1.00, "mad": 0.65},
}


def get_kc(days_after_planting: int, crop: str) -> float:
    p = CP[crop]
    return p["ini"] if days_after_planting < 30 else (p["mid"] if days_after_planting < 90 else p["end"])


def taw_f(fc: float, pwp: float, zr: float) -> float:
    return max(0.1, (fc - pwp) * zr * 1000)


def raw_f(taw: float, mad: float) -> float:
    return mad * taw


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rh_display(rh_min: float, rh_mean: float, rh_max: float) -> None:
    st.markdown(
        f'<div class="rh-row">'
        f'<div class="rh-box rh-min">Min RH<br><b>{rh_min:.0f}%</b></div>'
        f'<div class="rh-box rh-mean">Mean RH<br><b>{rh_mean:.0f}%</b></div>'
        f'<div class="rh-box rh-max">Max RH<br><b>{rh_max:.0f}%</b></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def sm_bar(pct: float, mad_pct: float) -> None:
    t = 1.0 - mad_pct / 100.0
    if pct / 100.0 < t * 0.9:
        color = "#d73027"
    elif pct / 100.0 < t:
        color = "#fc8d59"
    else:
        color = "#1a9850"
    st.markdown(
        f'<div class="sm-bar-wrap">'
        f'<div class="sm-bar-fill" style="background:{color};width:{pct:.1f}%"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"🔴 Stress < {100 - mad_pct:.0f}%FC (MAD={mad_pct:.0f}%) | "
        f"🟢 Safe above | Now: **{pct:.0f}%FC**"
    )


def wmo_icon(code) -> str:
    if not code:
        return "🌤️"
    code = int(code)
    if code == 0:
        return "☀️"
    if code in (1, 2, 3):
        return "🌤️"
    if 51 <= code <= 67:
        return "🌧️"
    if 80 <= code <= 82:
        return "🌦️"
    if 95 <= code <= 99:
        return "⛈️"
    return "🌥️"


def cloud_label(cc) -> str:
    if cc is None:
        return "N/A"
    cc = int(cc)
    if cc <= 10:
        return f"{cc}% ☀️ Clear"
    if cc <= 30:
        return f"{cc}% 🌤️ Few clouds"
    if cc <= 60:
        return f"{cc}% ⛅ Partly cloudy"
    if cc <= 85:
        return f"{cc}% 🌥️ Mostly cloudy"
    return f"{cc}% ☁️ Overcast"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — LOCATION SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("📍 Location")

LOCS = {
    "Kabuyanda, Isingiro": (-0.9579,  30.609,    1400),
    "Makerere Uni MAIN":   ( 0.339956, 32.567683, 1181),
    "MUARiK":              ( 0.46758,  32.6092,   1181),
    "Mbarara":             (-0.6064,   30.6582,   1430),
    "Gulu":                ( 2.7724,   32.2903,   1108),
    "Jinja":               ( 0.4478,   33.2029,   1136),
    "Mbale":               ( 1.0785,   34.1778,   1155),
    "Kabale":              (-1.2514,   29.99,     1869),
    "Fort Portal":         ( 0.671,    30.275,    1540),
    "Masaka":              (-0.3136,   31.7357,   1194),
    "Lira":                ( 2.2499,   32.9002,   1074),
    "Soroti":              ( 1.7153,   33.6107,   1110),
    "Arua":                ( 3.02,     30.91,     1045),
    "Hoima":               ( 1.435,    31.3524,   1550),
    "Kasese":              ( 0.1833,   30.0833,    920),
    "Tororo":              ( 0.6969,   34.1818,   1181),
    "Custom Location":     (None,      None,      None),
}

sel = st.sidebar.selectbox("Select District / Area", list(LOCS.keys()), index=0)
loc = LOCS[sel]

if sel == "Custom Location":
    LAT  = st.sidebar.number_input("Latitude",      value=-0.9579, format="%.6f")
    LON  = st.sidebar.number_input("Longitude",     value=30.609,  format="%.6f")
    ELEV = st.sidebar.number_input("Elevation (m)", value=1200,    step=10)
    soil = {"fc": 0.28, "pwp": 0.14, "source": "HWSD v2 – Default Loam"}
else:
    LAT, LON, ELEV = loc
    soil = get_soil_for_location(sel)
    st.sidebar.info(
        f"📌 **{sel}**\n\n"
        f"Lat `{LAT:.4f}` · Lon `{LON:.4f}`\n"
        f"Elev `{ELEV} m asl`\n\n"
        f"🌍 **Soil (HWSD v2):** {soil['source']}\n\n"
        f"FC: {soil['fc']*100:.0f}% | PWP: {soil['pwp']*100:.0f}%"
    )

st.sidebar.markdown(
    f"[🗺️ View on Google Maps](https://maps.google.com/?q={LAT},{LON})"
)

# ── Sidebar — Model info (v4) ─────────────────────────────────────────────────
if MODEL is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 ML Model v4")
    st.sidebar.markdown(
        f"**File:** `{os.path.basename(MODEL_PATH)}`\n\n"
        f"**Target:** `{MODEL_TARGET}`\n\n"
        f"**Features:** {len(MODEL_FEATURES) if MODEL_FEATURES else 'legacy'}\n\n"
        f"**R²:** {MODEL_METRICS.get('r2','–')} · **RMSE:** {MODEL_METRICS.get('rmse','–')}\n\n"
        f"**MAE:** {MODEL_METRICS.get('mae','–')} · **MAPE:** {MODEL_METRICS.get('mape','–')}%\n\n"
        f"*Trained on NASA-POWER Uganda 1990–2025*\n"
        f"*(552,258 rows · 90/10 split)*"
    )

# ══════════════════════════════════════════════════════════════════════════════
# FETCH ALL WEATHER SOURCES — hourly cache key
# ══════════════════════════════════════════════════════════════════════════════
_cache_hour = datetime.now().strftime("%Y%m%d%H")

with st.spinner("📡 Fetching real-time weather from ERA5, NWP and OpenWeather…"):
    arc  = fetch_archive_today(LAT, LON, _cache_hour)
    fc_d = fetch_forecast_today(LAT, LON, _cache_hour)
    ow   = fetch_openweather(LAT, LON, _cache_hour)
    curr = get_current_conditions(LAT, LON, _cache_hour)

today_wx, data_quality = build_merged_weather(arc, fc_d, ow)

if today_wx:
    lt      = today_wx["tmax"]
    ln      = today_wx["tmin"]
    lr_max  = today_wx["rh_max"]
    lr_min  = today_wx["rh_min"]
    lr_mea  = today_wx["rh_mean"]
    lw      = today_wx["wind"]
    ls      = today_wx["rs"]
    lp      = today_wx["precip"]
    lm      = (lt + ln) / 2.0
    lcc     = today_wx.get("cloud_cover")
else:
    lt, ln, lr_max, lr_min, lr_mea = 28.0, 16.0, 70.0, 50.0, 60.0
    lw, ls, lp, lm, lcc = 1.5, 18.0, 0.0, 22.0, None

lt_now  = curr["temp"]       if curr else lm
lcl     = curr["humidity"]   if curr else lr_mea
lr1h    = curr["precip_now"] if curr else 0.0
lcc_now = curr["cloud_cover"] if curr else lcc

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
# ── Pre-compute today's FAO-56 ET₀ from MERGED weather (shared by all tabs) ──
# This single number anchors "today" consistently across Tab 1 and Tab 2.
_doy_today = int(datetime.today().strftime("%j"))
et0_today  = (
    et0_fao56_pm(lt, ln, lr_max, lr_min, lw, ls, ELEV, _doy_today, LAT)
    if today_wx else 3.5
)

tab1, tab2, tab3 = st.tabs(["📊 Today's IWR", "☁️ 5-Day Forecast", "📅 Historical"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TODAY'S IWR
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header(f"📊 Today's IWR — {sel}")
    st.caption(
        f"{data_quality} · FAO-56 Penman-Monteith"
        + (f" + XGBoost {MODEL_TARGET}" if MODEL is not None else "")
    )

    if today_wx and today_wx.get("is_actual"):
        st.success(
            f"✅ **ERA5 Observed + NWP Forecast averaged** · "
            f"Rain today: **{lp} mm** · Temp now: **{lt_now}°C** · "
            f"Cloud cover: **{cloud_label(lcc_now)}** · "
            f"Source: Open-Meteo (ERA5 + ICON + GFS; nearest weather stations via data assimilation)"
        )
    elif today_wx:
        st.warning(
            "🔮 Only forecast data available (ERA5 archive not yet ready for today). "
            "Source: Open-Meteo (ERA5 + ICON + GFS)."
        )
    else:
        st.error("❌ Weather APIs unavailable.")

    if today_wx:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🌡️ Temp Now",    f"{lt_now}°C",    f"Max {lt}° / Min {ln}°")
        c2.metric("💧 Mean RH",     f"{lr_mea:.0f}%", f"Max {lr_max:.0f}% / Min {lr_min:.0f}%")
        c3.metric("🌬️ Wind (2 m)",  f"{lw:.2f} m/s",  "at 2 m height")
        c4.metric("🌧️ Rain Today",  f"{lp:.1f} mm",
                  "✅ Observed" if (today_wx and today_wx.get("is_actual")) else "🔮 Forecast")
        c5.metric("☀️ Solar Rad",   f"{ls:.1f} MJ",   "per m²/day")
        c6.metric("☁️ Cloud Cover", cloud_label(lcc_now) if lcc_now is not None else "N/A", "% cover")

        st.markdown("#### 💧 Relative Humidity — Min / Mean / Max")
        rh_display(lr_min, lr_mea, lr_max)

        st.info(
            f"📡 **Data sources:** Open-Meteo (ERA5 + ICON + GFS) ⊕ OpenWeather\n\n"
            f"📌 **{sel}** · {LAT:.4f}°N, {LON:.4f}°E · {ELEV} m asl\n\n"
            f"🌍 **Soil (HWSD v2):** {soil['source']} "
            f"(FC={soil['fc']*100:.0f}%, PWP={soil['pwp']*100:.0f}%)"
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Climate Inputs (averaged, live)")
        rows_display = [
            ("Max Temp",         f"{lt} °C"),
            ("Min Temp",         f"{ln} °C"),
            ("Mean Temp",        f"{lm:.1f} °C"),
            ("Min RH",           f"{lr_min:.0f} %"),
            ("Mean RH",          f"{lr_mea:.0f} %"),
            ("Max RH",           f"{lr_max:.0f} %"),
            ("Wind Speed (2 m)", f"{lw:.3f} m/s"),
            ("Solar Radiation",  f"{ls:.1f} MJ/m²/day"),
            ("Cloud Cover",      cloud_label(lcc_now) if lcc_now is not None else "N/A"),
            ("Rain (observed)",  f"{lp:.1f} mm "
             + ("✅" if today_wx and today_wx.get("is_actual") else "🔮")),
        ]
        for label, val in rows_display:
            st.write(f"**{label}:** {val}")

        # v4: show Hargreaves ET₀ preview
        _doy_now = int(datetime.today().strftime("%j"))
        _etH_now = et0_hargreaves(lt, ln, _doy_now, LAT)
        st.write(f"**ET₀ Hargreaves (ETO_H):** {_etH_now:.3f} mm/d")

    with col2:
        st.subheader("🌱 Crop, Soil & Moisture Setup")

        cr1 = st.selectbox("Crop Type", list(CP.keys()), key="crop_t1")
        st.write(
            f"Soil: **{sel if sel != 'Custom Location' else 'Custom'}** (HWSD v2) · "
            f"FC: {soil['fc']*100:.0f}% · PWP: {soil['pwp']*100:.0f}%"
        )

        ar1 = st.number_input("Field Area (hectares)",    value=1.0, min_value=0.1, step=0.1)
        fl1 = st.number_input("Pump Flow Rate (m³/hour)", value=5.0, min_value=0.5, step=0.5)
        dy1 = st.number_input("Days After Planting",      value=45,  min_value=0,   step=1)

        cp1  = CP[cr1]
        kc1  = get_kc(int(dy1), cr1)
        taw1 = taw_f(soil["fc"], soil["pwp"], cp1["zr"])
        raw1 = raw_f(taw1, cp1["mad"])

        st.markdown("---")
        st.markdown("**🪱 Today's Soil Moisture**")
        gwt1 = st.slider("Surface Wetness (GWETTOP)",    0.0, 1.0, 0.55, 0.01)
        gwr1 = st.slider("Root-zone Wetness (GWETROOT)", 0.0, 1.0, 0.50, 0.01)

        smp  = gwr1 * 100.0
        dr1  = taw1 * (1.0 - gwr1)
        madp = cp1["mad"] * 100.0

        st.markdown(f"""
| Parameter | Value |
|---|---|
| Crop Coefficient Kc | **{kc1:.3f}** |
| Root Depth Zr (FAO-56) | **{cp1['zr']:.2f} m** |
| MAD | **{int(madp)}%** |
| Field Capacity (FC) | **{soil['fc']*100:.0f}%** |
| Permanent Wilting Point (PWP) | **{soil['pwp']*100:.0f}%** |
| TAW (Total Available Water) | **{taw1:.1f} mm** |
| RAW — irrigation trigger | **{raw1:.1f} mm** |
| Current Depletion Dr | **{dr1:.1f} mm** |
| Root-zone SM | **{smp:.0f}% of FC** |
| Soil Source | ✔️ {soil['source']} |
""")

    # ── CALCULATE BUTTON ──────────────────────────────────────────────────────
    if st.button("🧮 Calculate Today's IWR", type="primary", use_container_width=True):
        tdt  = datetime.today()
        doy  = int(tdt.strftime("%j"))

        # ── FAO-56 ET₀ & water balance ────────────────────────────────────────
        et0  = et0_fao56_pm(lt, ln, lr_max, lr_min, lw, ls, ELEV, doy, LAT)
        etc  = kc1 * et0
        pe   = eff_rain(lp)
        sch  = mad_sched(dr1, taw1, raw1, pe, etc)

        iwr      = 0.0
        ml_label = "FAO-56 PM + MAD"

        # ── XGBoost v4 — run for every day, not only when MAD says irrigate ──
        # Pass the FAO-56 MAD IWR estimate as the hint feature (consistent
        # with how IWR was computed in the NASA-POWER training dataset).
        if MODEL is not None:
            try:
                feat = build_feature_dict(
                    lt, ln, lr_max, lr_min, lw, ls, lp,
                    gwt1, gwr1, dr1, tdt, LAT, LON,
                    rain_7d         = lp * 7,      # 7-day cumulative estimate; matches Tab 2 seed default
                    eto_7d          = et0 * 7.0,
                    drystreak       = 0,
                    iwr_estimate    = sch["iwr"],    # FAO-56 MAD amount as hint
                    eto_pm_override = et0,
                )
                pred_raw = run_ml(feat)
                iwr, ml_label = ml_to_iwr(
                    pred_raw, kc1, dr1, taw1, raw1, pe,
                    lt, ln, lr_max, lr_min, lw, ls, ELEV, doy, LAT
                )
            except Exception as _e:
                iwr      = sch["iwr"]
                ml_label = f"FAO-56 PM + MAD (ML error: {_e})"
        else:
            iwr = sch["iwr"]

        dp   = sch["dp"]
        dr_n = max(0.0, dr1 - pe - iwr)
        sma  = round((1.0 - dr_n / taw1) * 100.0, 1)
        vm   = iwr * ar1 * 10.0
        vl   = vm * 1000.0
        mins = round((vm / fl1) * 60.0, 1) if fl1 > 0 and vm > 0 else 0

        # ── ET₀ comparison: FAO-56 PM vs Hargreaves ───────────────────────────
        et0_h = et0_hargreaves(lt, ln, doy, LAT)

        st.markdown(
            f"### 📋 Results — "
            f"<small style='font-size:.75rem;color:#666'>{ml_label}</small>",
            unsafe_allow_html=True,
        )
        st.info(
            f"📍 **Location:** {sel} · 🌍 **Soil:** {soil['source']}\n\n"
            f"📡 **Weather:** {data_quality}"
        )

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("ET₀ (PM)",      f"{et0:.3f} mm/d",  "FAO-56 PM")
        r2.metric("ET₀ (Hargrv.)", f"{et0_h:.3f} mm/d","Hargreaves")
        r3.metric("ETc",           f"{etc:.3f} mm/d",  f"Kc={kc1:.3f}")
        r4.metric("Eff. Rain Pe",  f"{pe:.2f} mm",     "USDA SCS")
        r5.metric("Depletion Dr",  f"{dr1:.1f} mm",    f"RAW={raw1:.1f} mm")
        r6.metric("IWR Today",     f"{iwr:.2f} mm",    "Irrigation needed")

        if iwr == 0:
            if sch["status"] == "full" or lp >= raw1:
                st.success(f"🌧️ **Rain ({lp:.1f} mm) recharged soil.** No irrigation needed.")
            else:
                st.success(f"✅ **No irrigation needed.** {sch['msg']}")
        else:
            st.warning(f"💧 **Irrigate today: {iwr:.2f} mm** — Method: {ml_label}")
            st.info(
                f"💦 Volume: **{vm:.1f} m³** ({vl:,.0f} L) for {ar1} ha\n\n"
                f"⏱️ Pump runtime: **{mins} min** at {fl1} m³/hr\n\n"
                f"📈 SM after irrigation: **{sma:.0f}%FC** (was {smp:.0f}%FC)\n\n"
                f"⚠️ MAD threshold: **{100 - madp:.0f}%FC** ({raw1:.1f} mm)"
            )

        st.markdown(f"**Root-zone SM: {smp:.0f}% of FC**")
        sm_bar(smp, madp)

        st.markdown("#### 💧 RH used in FAO-56 PM calculation")
        rh_display(lr_min, lr_mea, lr_max)
        st.caption(
            "FAO-56 §3.3.2: ea = (rh_max/100 × es_min + rh_min/100 × es_max) / 2. "
            "rh_max and rh_min are used separately. Mean RH shown for reference."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — 5-DAY FORECAST  (with 7-day rolling buffers for XGBoost features)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header(f"☁️ 5-Day IWR Forecast — {sel}")
    st.caption(
        "FAO-56 Penman-Monteith · Open-Meteo NWP (ICON + GFS) · "
        f"XGBoost v4 target={MODEL_TARGET if MODEL else 'N/A'} · "
        "7-day rolling rain/ET₀/drystreak fed to model"
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        pd_ = st.date_input("Planting Date", datetime.today().date())
    with fc2:
        cr2 = st.selectbox("Crop", list(CP.keys()), key="crop_t2")
    with fc3:
        st.write(f"Soil: **{sel}** (HWSD v2)\n\n{soil['source']}")

    fa1, fa2, fa3 = st.columns(3)
    with fa1:
        ar2   = st.number_input("Field Area (ha)",   value=1.0, min_value=0.1, step=0.1, key="area_fc")
    with fa2:
        fl2   = st.number_input("Pump Flow (m³/hr)", value=5.0, min_value=0.5, step=0.5, key="flow_fc")
    with fa3:
        gwr_f = st.slider("Root-zone Wetness", 0.0, 1.0, 0.50, 0.01, key="gwetroot_fc")

    # 7-day seed sliders (today's conditions to initialise rolling buffers)
    with st.expander("⚙️ 7-day Rolling Seed (for ML features)"):
        st.caption("Initial estimates used to seed the 7-day rain/ET₀/drystreak buffers.")
        s1, s2, s3 = st.columns(3)
        seed_rain7   = s1.number_input("Last 7-day Rain (mm)",  value=float(round(lp * 7, 1)),         step=1.0)
        seed_eto7    = s2.number_input("Last 7-day ET₀ (mm)",   value=float(round(et0_today * 7, 1)),  step=1.0)
        seed_dry     = s3.number_input("Current Dry Streak (d)", value=0, min_value=0, step=1)

    if st.button("📥 Get 5-Day Forecast", type="primary", use_container_width=True):
        with st.spinner("Fetching forecast…"):
            daily = get_om_fc(LAT, LON, _cache_hour)

        if daily is None or daily.empty:
            st.error("❌ Forecast unavailable.")
            st.stop()

        cp2  = CP[cr2]
        taw2 = taw_f(soil["fc"], soil["pwp"], cp2["zr"])
        raw2 = raw_f(taw2, cp2["mad"])
        pln  = pd.Timestamp(pd_)

        # ══════════════════════════════════════════════════════════════════════
        # ROOT-CAUSE FIX: Override today's NWP-only row with MERGED weather
        # ══════════════════════════════════════════════════════════════════════
        # Tab 1 ("Today's IWR") uses build_merged_weather():
        #   ERA5 Archive (observed) + Open-Meteo NWP Forecast + OpenWeather,
        #   all three averaged.
        # Tab 2 ("5-Day Forecast") calls get_om_fc() which uses NWP ONLY —
        #   no ERA5 archive, no OpenWeather.  So today's tmax/tmin/RH/wind/Rs
        #   and therefore ET₀ and IWR are computed from a DIFFERENT data set.
        # Fix: patch daily.iloc[0] (= today) with the same merged values used
        # in Tab 1 BEFORE ET₀/ETc/Pe are computed from the row.
        _today_ts = pd.Timestamp(datetime.today().date())
        if today_wx is not None and _today_ts in daily.index:
            for _col, _val in [
                ("tmax",          lt),
                ("tmin",          ln),
                ("rh_max",        lr_max),
                ("rh_min",        lr_min),
                ("rh_mean",       lr_mea),
                ("wind",          lw),
                ("rs",            ls),
                ("precipitation", lp),
            ]:
                daily.at[_today_ts, _col] = _val

        daily["kc"]  = daily.index.map(lambda d: get_kc((d - pln).days, cr2))
        daily["doy"] = daily.index.map(lambda d: int(d.strftime("%j")))
        daily["ET0"] = daily.apply(
            lambda r: et0_fao56_pm(
                r["tmax"], r["tmin"], r["rh_max"], r["rh_min"],
                r["wind"], r["rs"], ELEV, r["doy"], LAT,
            ), axis=1,
        )
        daily["ETc"] = daily["kc"] * daily["ET0"]
        daily["Pe"]  = daily["precipitation"].apply(eff_rain)

        # ── Initialise rolling buffers & dual soil trackers ────────────────────
        # Seed buffers with per-day averages from the user-supplied 7-day totals
        # (avoids the bug of repeating today's single-day rain 7 times)
        rain_buf  = [float(seed_rain7) / 7.0] * 7   # historical daily avg
        eto_buf   = [float(seed_eto7)  / 7.0] * 7
        drystreak = int(seed_dry)

        # TWO soil depletion trackers — this is the key fix for consistent
        # multi-day predictions:
        #
        #  dr_no_irr  — natural depletion assuming NO irrigation is applied
        #               between days. Used as ML input features so each day's
        #               model call sees realistically drying soil, not a soil
        #               that was magically recharged by yesterday's IWR output.
        #               Without this, Day 1 irrigates → soil → 100 % FC →
        #               Days 2–5 see "wet soil" → model correctly predicts 0 →
        #               but runs as "0 today, big number tomorrow" every day.
        #
        #  dr_display — tracks depletion assuming IWR IS applied (for SM_pct
        #               and Depletion columns so the user sees what happens if
        #               they follow the schedule).
        dr_no_irr = taw2 * (1.0 - gwr_f)   # starting depletion (live today)
        dr_display = dr_no_irr

        iw_l, dep_l, sm_l, dp_l, ml_l = [], [], [], [], []

        for i, (dt_idx, row) in enumerate(daily.iterrows()):
            # 7-day rolling sums from the ring buffer
            rain_7d = sum(rain_buf[-7:])
            eto_7d  = sum(eto_buf[-7:])

            # Update drystreak
            if row["precipitation"] < 1.0:
                drystreak += 1
            else:
                drystreak = 0

            # ── Soil state for ML features = natural depletion (no irrigation) ─
            # This ensures each day the model sees soil that has dried by the
            # cumulative ETc – Pe of preceding days, not a magically wet soil.
            # When run again tomorrow, tomorrow's dr_no_irr will be today's + 1
            # day of depletion → predictions are consistent across daily runs.
            gwt_est = max(0.0, min(1.0, 1.0 - dr_no_irr / taw2 * 0.8))
            gwr_est = max(0.0, min(1.0, 1.0 - dr_no_irr / taw2))

            # FAO-56 MAD water balance (uses no-irr soil state as reference)
            sch = mad_sched(dr_no_irr, taw2, raw2, row["Pe"], row["ETc"])

            id_     = 0.0
            ml_used = "FAO-56 MAD"

            # ── ML runs for EVERY day with the natural-depletion soil state ────
            if MODEL is not None:
                try:
                    feat = build_feature_dict(
                        row["tmax"], row["tmin"],
                        row["rh_max"], row["rh_min"],
                        row["wind"], row["rs"],
                        row["precipitation"],
                        gwt_est, gwr_est, dr_no_irr,
                        dt_idx.to_pydatetime(),
                        LAT, LON,
                        rain_7d         = rain_7d,
                        eto_7d          = eto_7d,
                        drystreak       = drystreak,
                        iwr_estimate    = sch["iwr"],    # FAO-56 MAD estimate as hint
                        eto_pm_override = row["ET0"],
                    )
                    pred_raw = run_ml(feat)
                    id_, ml_used = ml_to_iwr(
                        pred_raw, row["kc"], dr_no_irr, taw2, raw2, row["Pe"],
                        row["tmax"], row["tmin"],
                        row["rh_max"], row["rh_min"],
                        row["wind"], row["rs"],
                        ELEV, row["doy"], LAT,
                    )
                except Exception as _ml_err:
                    id_     = sch["iwr"]
                    ml_used = f"FAO-56 MAD (ML err: {_ml_err})"
            else:
                id_     = sch["iwr"]
                ml_used = "FAO-56 MAD"

            # Update rolling ring buffers with this day's forecast values
            rain_buf.append(row["precipitation"])
            eto_buf.append(row["ET0"])
            rain_buf = rain_buf[-7:]
            eto_buf  = eto_buf[-7:]

            # ── Update BOTH soil trackers ──────────────────────────────────────
            # 1. Natural depletion (NO irrigation subtracted) — drives ML features
            dr_no_irr = max(0.0, min(taw2, dr_no_irr + row["ETc"] - row["Pe"]))

            # 2. Display balance (IWR subtracted) — shows SM if schedule followed
            dr_display = max(0.0, min(taw2, dr_display + row["ETc"] - row["Pe"] - id_))
            gwr_disp   = max(0.0, min(1.0,  1.0 - dr_display / taw2))

            iw_l.append(id_)
            dep_l.append(round(dr_display, 2))
            sm_l.append(round(gwr_disp * 100.0, 1))
            dp_l.append(round(sch["dp"], 2))
            ml_l.append(ml_used)

        daily["IWR"]       = iw_l
        daily["Depletion"] = dep_l
        daily["SM_pct"]    = sm_l
        daily["DP"]        = dp_l
        daily["ML_Method"] = ml_l
        daily["Volume_m3"] = daily["IWR"] * ar2 * 10.0
        daily["Minutes"]   = daily["Volume_m3"].apply(
            lambda v: round((v / fl2) * 60.0, 1) if fl2 > 0 and v > 0 else 0
        )

        tot = daily["IWR"].sum()
        nd  = (daily["IWR"] > 0).sum()

        if nd > 0:
            st.warning(f"🗓️ **{nd} irrigation event(s)** · Total IWR = {tot:.1f} mm")
        else:
            st.success("✅ No irrigation needed over next 5 days.")

        cols = st.columns(len(daily))
        for i, (dt_idx, row) in enumerate(daily.iterrows()):
            icon = wmo_icon(row.get("wcode"))
            lbl  = f"💧 {row['IWR']:.1f}mm" if row["IWR"] > 0 else f"{icon} OK"
            dlt  = f"⏱ {row['Minutes']}min" if row["Minutes"] > 0 else f"🌧 {row['precipitation']:.1f}mm"
            with cols[i]:
                st.metric(dt_idx.strftime("%a %d"), lbl, dlt)

        st.subheader("📋 Forecast Table")
        tc = [
            "tmax", "tmin", "rh_min", "rh_mean", "rh_max",
            "precipitation", "ET0", "ETc", "Pe", "IWR",
            "SM_pct", "Depletion", "ML_Method",
        ]
        tb = daily[[c for c in tc if c in daily.columns]].round(3).copy()
        tb.index = tb.index.strftime("%Y-%m-%d")
        st.dataframe(tb, use_container_width=True)

        # ── Plotly IWR chart ──────────────────────────────────────────────────
        fig = go.Figure()
        fig.add_bar(
            x=daily.index.strftime("%a %d"),
            y=daily["IWR"],
            name="IWR (mm)",
            marker_color="#1a5fc8",
        )
        fig.add_scatter(
            x=daily.index.strftime("%a %d"),
            y=daily["ET0"],
            name="ET₀ (mm/d)",
            mode="lines+markers",
            marker_color="#0b6b1b",
            yaxis="y2",
        )
        fig.update_layout(
            title=f"5-Day IWR Forecast — {sel}",
            yaxis=dict(title="IWR (mm)"),
            yaxis2=dict(title="ET₀ (mm/d)", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            height=350,
            plot_bgcolor="#f4f8f2",
            paper_bgcolor="#f4f8f2",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header(f"📅 Historical Analysis — {sel}")
    st.caption(
        "FAO-56 PM + Open-Meteo Archive (ERA5) · MAD water balance · "
        "Source: Open-Meteo (ERA5 + ICON + GFS; nearest weather stations via data assimilation)"
    )

    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("Last 30 Days",  use_container_width=True):
        st.session_state["hdays"] = 30
    if pc2.button("Last 90 Days",  use_container_width=True):
        st.session_state["hdays"] = 90
    if pc3.button("Last 365 Days", use_container_width=True):
        st.session_state["hdays"] = 365

    prs = st.session_state.get("hdays", 30)

    hc1, hc2 = st.columns(2)
    with hc1:
        h_e = st.date_input(
            "End Date",
            value=datetime.today().date() - timedelta(days=1),
            max_value=datetime.today().date() - timedelta(days=1),
            key="h_end",
        )
    with hc2:
        h_s = st.date_input(
            "Start Date",
            value=datetime.today().date() - timedelta(days=prs),
            max_value=h_e - timedelta(days=1),
            key="h_start",
        )

    hd1, hd2 = st.columns(2)
    with hd1:
        cr3 = st.selectbox("Crop", list(CP.keys()), key="crop_t3")
    with hd2:
        ph = st.date_input(
            "Planting Date",
            value=datetime.today().date() - timedelta(days=45),
            key="plant_h",
        )

    gwr_h = st.slider("Starting Root-zone Wetness", 0.0, 1.0, 0.60, 0.01, key="gwetroot_h")

    if (h_e - h_s).days > 730:
        st.warning("⚠️ Capped at 2 years.")
        h_s = h_e - timedelta(days=730)

    if st.button("📥 Retrieve Historical Data", type="primary", use_container_width=True):
        with st.spinner("Fetching archive…"):
            hist = get_arc(LAT, LON, str(h_s), str(h_e))

        if hist is None or hist.empty:
            st.error("❌ No data available.")
            st.stop()

        cp3  = CP[cr3]
        taw3 = taw_f(soil["fc"], soil["pwp"], cp3["zr"])
        raw3 = raw_f(taw3, cp3["mad"])
        pl3  = pd.Timestamp(ph)

        hist["kc"]  = hist.index.map(lambda d: get_kc((d - pl3).days, cr3))
        hist["doy"] = hist.index.map(lambda d: int(d.strftime("%j")))
        hist["ET0"] = hist.apply(
            lambda r: et0_fao56_pm(
                r["tmax"], r["tmin"], r["rh_max"], r["rh_min"],
                r["wind"], r["rs"], ELEV, r["doy"], LAT,
            ), axis=1,
        )
        hist["ETc"] = hist["kc"] * hist["ET0"]
        hist["Pe"]  = hist["precipitation"].apply(eff_rain)
        # Hargreaves ET₀ for reference
        hist["ET0_H"] = hist.apply(
            lambda r: et0_hargreaves(r["tmax"], r["tmin"], r["doy"], LAT), axis=1
        )

        dr3 = taw3 * (1.0 - gwr_h)
        iw3, dep3, dp3_l = [], [], []

        # 7-day rolling for historical (pure FAO-56 on historical — no ML on archive)
        for _, row in hist.iterrows():
            sc   = mad_sched(dr3, taw3, raw3, row["Pe"], row["ETc"])
            iwd  = sc["iwr"]
            dr3  = max(0.0, min(taw3, dr3 + row["ETc"] - row["Pe"] - iwd))
            iw3.append(iwd)
            dep3.append(round(dr3, 2))
            dp3_l.append(round(sc["dp"], 2))

        hist["IWR"]       = iw3
        hist["Depletion"] = dep3
        hist["DP"]        = dp3_l

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("📆 Days",        len(hist))
        m2.metric("🌧️ Total Rain",  f"{hist['precipitation'].sum():.1f} mm")
        m3.metric("💧 Total IWR",   f"{hist['IWR'].sum():.1f} mm")
        m4.metric("🚿 Irrig Days",  str((hist["IWR"] > 0).sum()))
        m5.metric("☀️ Avg ET₀ PM",  f"{hist['ET0'].mean():.3f} mm/d")
        m6.metric("☀️ Avg ET₀ H",   f"{hist['ET0_H'].mean():.3f} mm/d")

        st.subheader("📋 Historical Table")
        ht = hist[[
            "tmax", "tmin", "rh_min", "rh_mean", "rh_max",
            "precipitation", "ET0", "ET0_H", "ETc", "Pe", "IWR", "Depletion",
        ]].round(3).copy()
        ht.columns = [
            "Tmax °C", "Tmin °C", "RHmin %", "RHmean %", "RHmax %",
            "Rain mm", "ET₀ PM mm", "ET₀ H mm", "ETc mm", "Pe mm", "IWR mm", "Dr mm",
        ]
        ht.index = ht.index.strftime("%Y-%m-%d")
        st.dataframe(ht, use_container_width=True)

        # ── Plotly time-series chart ──────────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_scatter(
            x=hist.index, y=hist["ET0"],
            name="ET₀ PM (mm/d)", mode="lines",
            line=dict(color="#1a5fc8", width=1.5),
        )
        fig2.add_scatter(
            x=hist.index, y=hist["ET0_H"],
            name="ET₀ Hargreaves (mm/d)", mode="lines",
            line=dict(color="#b81c1c", width=1, dash="dot"),
        )
        fig2.add_bar(
            x=hist.index, y=hist["IWR"],
            name="IWR (mm)", marker_color="#0b6b1b", opacity=0.6, yaxis="y2",
        )
        fig2.update_layout(
            title=f"Historical ET₀ & IWR — {sel}",
            yaxis=dict(title="ET₀ (mm/d)"),
            yaxis2=dict(title="IWR (mm)", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            height=380,
            plot_bgcolor="#f4f8f2",
            paper_bgcolor="#f4f8f2",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            ht.to_csv().encode("utf-8"),
            f"HyPIS_hist_{h_s}_{h_e}.csv",
            "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    f"HyPIS Ug v4.0 · ERA5 Observed + NWP Forecast + OpenWeather Averaged · "
    f"Source: Open-Meteo (ERA5 + ICON + GFS; nearest weather stations via data assimilation) · "
    f"FAO-56 Penman-Monteith · HWSD v2 Soil · "
    f"XGBoost v4 NASA-Power Uganda 1990–2025 (552,258 rows) · "
    f"Target: {MODEL_TARGET} · Features: {len(MODEL_FEATURES) if MODEL_FEATURES else 'legacy'} · "
    f"Byaruhanga Prosper · {sel}"
)