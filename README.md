# Oahu-Clarity
Interconnectvity and translation of organic data for Weather Safety and Awareness and Streamlining Disaster Response
# FILE: pyproject.toml
[tool.poetry]
name = "oahu-clarity"
version = "0.1.0"
description = "Oahu Clarity Suite v0.1 — plain-language daily cues for residents"
authors = ["Carl McKay & Collaborators"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.115.0"
uvicorn = "^0.30.6"
pydantic = "^2.9.2"
httpx = "^0.27.2"
PyYAML = "^6.0.2"
apscheduler = "^3.10.4"
python-dotenv = "^1.0.1"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"
ruff = "^0.6.9"
black = "^24.8.0"

[tool.ruff]
line-length = 100

[tool.black]
line-length = 100

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# FILE: requirements.txt
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
httpx==0.27.2
PyYAML==6.0.2
apscheduler==3.10.4
python-dotenv==1.0.1

# FILE: config.yml
area_defaults:
  timezone: "Pacific/Honolulu"
  valid_minutes: 90

floodsense:
  level_alert_ft: 3.0
  rise_alert_ft_per_30min: 0.5
  king_tide_ft: 2.7
  onshore_wind_kts: 18
  subsidence_mm_per_yr_watch: 10
  subsidence_mm_per_yr_avoid: 20

debris_mapper:
  sustained_wind_kts_watch: 20
  gust_kts_watch: 30
  gust_kts_avoid: 40
  rain_24h_in_watch: 1.5
  rain_24h_in_avoid: 3.0
  report_cooldown_min: 30

microgrid_pulse:
  outage_clusters_watch: 3
  outage_customers_watch: 1500
  outage_clusters_avoid: 8
  outage_customers_avoid: 5000

# FILE: .env.example
# Runtime switches
LIVE_MODE=false
LOG_LEVEL=INFO
UPDATE_INTERVAL_MIN=15

# Data endpoints / IDs (optional; used in live mode)
USGS_GAUGES=16247100
NOAA_STATION=1612340
NWS_GRIDPOINT=HFO/154,42
USER_AGENT=OahuClarityPrototype/0.1 (contact: community@example.org)

# FILE: README.md
# Oahu Clarity Suite v0.1 (Prototype)

Plain-language daily cues for residents. Runs offline with fixtures or live against public APIs.

## Quickstart (fixtures / offline)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LIVE_MODE=false
uvicorn app.api.routes:app --reload

Open: http://127.0.0.1:8000/api/today

## Quickstart (live)
cp .env.example .env
edit .env with station ids and USER_AGENT
export LIVE_MODE=true
uvicorn app.api.routes:app --reload

## Endpoints
- /health
- /api/today
- /api/modules/{name}   (name: floodsense|debris|microgrid)
- /

Thresholds live in config.yml. See STYLE.md for copy rules.

## Boundaries and Assumptions (v0.1)
- Data freshness depends on source agencies; outages may delay updates.
- Thresholds are provisional and should be tuned by local stewards.
- Offline mode uses synthetic fixtures for demo only.
- No PII collected; read-only public data.
- Complements, not replaces, official alerts; cues link to sources.
- Open-source, easy to decommission if not useful.

# FILE: ASSUMPTIONS.md
Assumptions (v0.1)
1. Station IDs are representative: Manoa stream (USGS 16247100), Kailua tide (NOAA 1612340), NWS HFO gridpoint.
2. Thresholds chosen as conservative starters; hosts will tune.
3. Outage data is a fixture/stub until a sanctioned API is provided.
4. Messages suggest small, useful actions; never alarm.
5. Suite refreshes ~15 minutes; cues valid for 90 minutes.

Non-goals (v0.1)
- No maps, accounts, or push notifications.
- No scraping of utility pages.
- No permanent storage.

# FILE: STYLE.md
Copy & Localization
- One sentence, <= 18 words.
- Calm verbs: "avoid", "pick", "check", "delay".
- Place-first when relevant.
- No "every X a Y" phrasing, no techno-jargon.
- English only in v0.1; bilingual planned.

# FILE: public/index.html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Oahu Clarity — Today</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto; margin:2rem; color:#123;}
    .wrap{max-width:860px;margin:auto}
    .chip{display:inline-block;padding:.25rem .6rem;border-radius:999px;font-weight:600}
    .CLEAR{background:#e7f7ee;color:#146c43}
    .WATCH{background:#fff4cc;color:#8a6d00}
    .AVOID{background:#fdecea;color:#b42318}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin:.75rem 0}
    small{color:#64748b}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Oahu Clarity</h1>
  <p><small id="meta"></small></p>
  <div id="cues"></div>
</div>
<script>
async function load() {
  const res = await fetch('/api/today');
  const j = await res.json();
  document.getElementById('meta').textContent =
    `${j.area} • generated ${new Date(j.generated_at).toLocaleString()} • valid ${j.valid_minutes} min`;
  const root = document.getElementById('cues'); root.innerHTML = '';
  j.cues.forEach(c => {
    const card = document.createElement('div'); card.className='card';
    card.innerHTML = `<div class="chip ${c.status}">${c.module}: ${c.status}</div>
      <p style="margin:.5rem 0 0 0">${c.message}</p>
      <small>${c.rationale}</small>`;
    root.appendChild(card);
  });
}
load().catch(console.error);
</script>
</body>
</html>

# FILE: fixtures/usgs_stream_manoa.json
{"gauge_id":"16247100","ts":"2025-11-10T20:00:00Z","level_ft":[2.3,2.6,3.1],"rain_in_1h":0.4}

# FILE: fixtures/noaa_tide_kailua.json
{"station":"1612340","latest_ts":"2025-11-10T20:00:00Z","tide_ft":2.8,"datum":"MLLW"}

# FILE: fixtures/nws_hfo_grid.json
{"place":"Windward Oahu","sustained_kts":16,"gust_kts":32,"rain_24h_in":2.1,"onshore":true}

# FILE: fixtures/outages_stub.json
{"clusters":5,"customers_affected":2100,"areas":["Kailua","Kaneohe"]}

# FILE: fixtures/subsidence_lookup.json
{"Mapunapuna":22,"Manoa":8,"Kakaako":15}

# FILE: app/__init__.py
# package marker

# FILE: app/core/utils.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os

HONOLULU_OFFSET = -10  # HST (no DST)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def valid_until(minutes: int) -> str:
    return (now_utc() + timedelta(minutes=minutes)).isoformat()

def get_live_mode() -> bool:
    return os.getenv("LIVE_MODE", "false").lower() == "true"

def to_iso(dt: datetime | None) -> str:
    return (dt or now_utc()).isoformat()

# FILE: app/core/config.py
from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict
import yaml

class Settings(BaseModel):
    area_defaults: Dict[str, Any]
    floodsense: Dict[str, Any]
    debris_mapper: Dict[str, Any]
    microgrid_pulse: Dict[str, Any]

def load_config(path: str = "config.yml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings(**data)

CFG = load_config()

# FILE: app/core/cache.py
from __future__ import annotations
from time import time
from typing import Any, Dict, Tuple

class TTLCache:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time(), value)

CACHE = TTLCache()

# FILE: app/core/scheduler.py
from __future__ import annotations
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Callable

def start_scheduler(job: Callable, minutes: int = 15) -> BackgroundScheduler:
    sched = BackgroundScheduler()
    sched.add_job(job, "interval", minutes=minutes)
    sched.start()
    return sched

# FILE: app/logic/status.py
from __future__ import annotations
from enum import Enum
from typing import Literal, TypedDict, Dict, Any
from ..core.utils import valid_until
from ..core.config import CFG

class Status(str, Enum):
    CLEAR = "CLEAR"
    WATCH = "WATCH"
    AVOID = "AVOID"

class Cue(TypedDict, total=False):
    module: Literal["FloodSense","Debris Mapper","Microgrid Pulse"]
    status: Literal["CLEAR","WATCH","AVOID"]
    message: str
    rationale: str
    inputs: Dict[str, Any]
    valid_until: str

def cue(module: str, status: Status, message: str, rationale: str, inputs: Dict[str, Any]) -> Cue:
    return {
        "module": module,
        "status": status.value,
        "message": message,
        "rationale": rationale,
        "inputs": inputs,
        "valid_until": valid_until(CFG.area_defaults["valid_minutes"]),
    }

# FILE: app/logic/floodsense.py
from __future__ import annotations
from .status import Status, cue, Cue
from ..core.config import CFG

TEMPLATES = {
    "CLEAR": "Creeks steady around {place}—no issues expected this afternoon.",
    "WATCH": "Stream rising near {place}—avoid low crossings this afternoon.",
    "AVOID": "Street ponding likely around {place}—pick higher routes today.",
}

def assess_floodsense(usgs, tide, wind, subsidence_mm: int = 0, place: str = "Manoa") -> Cue:
    cfg = CFG.floodsense
    levels = usgs["data"].get("level_ft", [])
    level_now = levels[-1] if levels else None
    rise = 0.0
    if len(levels) >= 2:
        rise = float(levels[-1] - levels[0])
    tide_ft = float(tide["data"].get("tide_ft", 0))
    onshore_kts = int(wind["data"].get("gust_kts", 0)) if wind else 0

    status = Status.CLEAR
    rationale = "Levels steady."
    if (rise >= cfg["rise_alert_ft_per_30min"] and tide_ft >= cfg["king_tide_ft"]) or onshore_kts >= cfg["onshore_wind_kts"]:
        status = Status.WATCH
        rationale = "Rapid rise or high tide pushing inland."
    if (status == Status.WATCH and subsidence_mm >= cfg["subsidence_mm_per_yr_avoid"]) or (
        level_now is not None and level_now >= cfg["level_alert_ft"] and rise > 0
    ):
        status = Status.AVOID
        rationale = "High water with risk factors present."
    if status == Status.CLEAR and level_now is not None and rise > 0:
        rationale = "Minor rise; monitoring."

    msg = TEMPLATES[status.value].format(place=place)
    return cue(
        "FloodSense",
        status,
        _trim(msg),
        rationale,
        {
            "level_ft": level_now,
            "rise_ft_30min": round(rise, 2),
            "tide_ft": tide_ft,
            "onshore_wind_kts": onshore_kts,
            "subsidence_mm_per_yr": subsidence_mm,
        },
    )

def _trim(text: str) -> str:
    return text if len(text) <= 120 else text[:117] + "..."

# FILE: app/logic/debris_mapper.py
from __future__ import annotations
from .status import Status, cue, Cue
from ..core.config import CFG

TEMPLATES = {
    "CLEAR": "Winds light and rain pockets small—roads look fine today.",
    "WATCH": "Gusty pockets near {place}; secure loose items and watch for downed branches.",
    "AVOID": "Strong gusts and saturated ground near {place}—skip tree-lined shortcuts today.",
}

def assess_debris(nws, place: str = "Windward Oahu") -> Cue:
    cfg = CFG.debris_mapper
    sustained = float(nws["data"].get("sustained_kts", 0))
    gust = float(nws["data"].get("gust_kts", 0))
    rain24 = float(nws["data"].get("rain_24h_in", 0))

    status = Status.CLEAR
    rationale = "Winds modest; rain limited."
    if gust >= cfg["gust_kts_watch"] or rain24 >= cfg["rain_24h_in_watch"]:
        status = Status.WATCH
        rationale = "Gusty pockets or wet ground."
    if gust >= cfg["gust_kts_avoid"] or rain24 >= cfg["rain_24h_in_avoid"]:
        status = Status.AVOID
        rationale = "High gusts or heavy rain accumulation."

    msg = TEMPLATES[status.value].format(place=place)
    return cue(
        "Debris Mapper",
        status,
        msg,
        rationale,
        {"sustained_kts": sustained, "gust_kts": gust, "rain_24h_in": rain24},
    )

# FILE: app/logic/microgrid_pulse.py
from __future__ import annotations
from .status import Status, cue, Cue
from ..core.config import CFG

TEMPLATES = {
    "CLEAR": "Power stable—charge phones during the day as a good habit.",
    "WATCH": "Scattered outages in {place}; have a flashlight handy and check neighbors.",
    "AVOID": "Multiple outages across {place}—delay non-essential trips and conserve battery.",
}

def assess_microgrid(outages, place: str = "Oahu") -> Cue:
    cfg = CFG.microgrid_pulse
    clusters = int(outages["data"].get("clusters", 0))
    customers = int(outages["data"].get("customers_affected", 0))

    status = Status.CLEAR
    rationale = "No significant outage clusters."
    if clusters >= cfg["outage_clusters_watch"] or customers >= cfg["outage_customers_watch"]:
        status = Status.WATCH
        rationale = "Some outage activity reported."
    if clusters >= cfg["outage_clusters_avoid"] or customers >= cfg["outage_customers_avoid"]:
        status = Status.AVOID
        rationale = "Widespread outages reported."

    msg = TEMPLATES[status.value].format(place=place)
    return cue(
        "Microgrid Pulse",
        status,
        msg,
        rationale,
        {"clusters": clusters, "customers_affected": customers},
    )

# FILE: app/adapters/_common.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict
from ..core.utils import now_utc

FIXTURES = Path("fixtures")

def read_fixture(name: str) -> Dict[str, Any]:
    with open(FIXTURES / name, "r", encoding="utf-8") as f:
        return json.load(f)

def standard(source: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"source": source, "ts": now_utc().isoformat(), "data": data}

def live_mode() -> bool:
    return os.getenv("LIVE_MODE", "false").lower() == "true"

# FILE: app/adapters/usgs.py
from __future__ import annotations
from typing import Dict, Any, List
import os
import httpx
from ._common import read_fixture, standard, live_mode

USGS_URL = "https://waterservices.usgs.gov/nwis/iv/"

async def fetch() -> Dict[str, Any]:
    if not live_mode():
        j = read_fixture("usgs_stream_manoa.json")
        return standard("usgs", {"level_ft": j["level_ft"], "rain_in_1h": j["rain_in_1h"], "gauge_id": j["gauge_id"]})
    params = {
        "format": "json",
        "sites": os.getenv("USGS_GAUGES", "16247100"),
        "parameterCd": "00065,00045",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(USGS_URL, params=params)
        r.raise_for_status()
        data = r.json()
    level_series: List[float] = []
    try:
        series = data["value"]["timeSeries"]
        for s in series:
            code = s["variable"]["variableCode"][0]["value"]
            if code == "00065":
                vals = [float(v["value"]) for v in s["values"][0]["value"][-3:]]
                level_series = vals
    except Exception:
        level_series = []
    return standard("usgs", {"level_ft": level_series, "rain_in_1h": 0.0})

# FILE: app/adapters/noaa_coops.py
from __future__ import annotations
import os
import httpx
from ._common import read_fixture, standard, live_mode

COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

async def fetch():
    if not live_mode():
        j = read_fixture("noaa_tide_kailua.json")
        return standard("noaa", {"tide_ft": j["tide_ft"], "station": j["station"]})
    station = os.getenv("NOAA_STATION", "1612340")
    params = {
        "station": station,
        "product": "water_level",
        "datum": "MLLW",
        "units": "english",
        "time_zone": "gmt",
        "format": "json",
        "range": "180",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(COOPS_URL, params=params, headers={"User-Agent": os.getenv("USER_AGENT", "OahuClarityPrototype/0.1")})
        r.raise_for_status()
        data = r.json()
    try:
        last = data["data"][-1]
        tide_ft = float(last["v"])
    except Exception:
        tide_ft = 0.0
    return standard("noaa", {"tide_ft": tide_ft, "station": station})

# FILE: app/adapters/nws.py
from __future__ import annotations
import os
import httpx
from ._common import read_fixture, standard, live_mode

NWS_BASE = "https://api.weather.gov/gridpoints"

async def fetch():
    if not live_mode():
        j = read_fixture("nws_hfo_grid.json")
        return standard("nws", j)
    grid = os.getenv("NWS_GRIDPOINT", "HFO/154,42")
    headers = {"User-Agent": os.getenv("USER_AGENT", "OahuClarityPrototype/0.1"), "Accept": "application/geo+json"}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{NWS_BASE}/{grid}", headers=headers)
        r.raise_for_status()
        data = r.json()
    try:
        props = data["properties"]
        sustained = float(props["windSpeed"]["values"][0]["value"]) * 1.94384
        gust = float(props["windGust"]["values"][0]["value"]) * 1.94384 if props.get("windGust") else sustained
        rain = float(props["quantitativePrecipitation"]["values"][0]["value"]) / 25.4
    except Exception:
        sustained, gust, rain = 0.0, 0.0, 0.0
    return standard("nws", {"place":"Oahu","sustained_kts":round(sustained,1),"gust_kts":round(gust,1),"rain_24h_in":round(rain,2),"onshore":True})

# FILE: app/adapters/outages.py
from __future__ import annotations
from ._common import read_fixture, standard

async def fetch():
    j = read_fixture("outages_stub.json")
    return standard("outages", j)

# FILE: app/api/routes.py
from __future__ import annotations
import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pathlib import Path
from ..core.config import CFG
from ..core.cache import CACHE
from ..core.scheduler import start_scheduler
from ..core.utils import now_utc
from ..logic.floodsense import assess_floodsense
from ..logic.debris_mapper import assess_debris
from ..logic.microgrid_pulse import assess_microgrid
from ..logic.status import Status
from ..adapters import usgs, noaa_coops, nws, outages

app = FastAPI(title="Oahu Clarity Suite", version="0.1.0")
PUBLIC = Path("public")

async def compute_today():
    usgs_j = await usgs.fetch()
    tide_j = await noaa_coops.fetch()
    nws_j = await nws.fetch()
    out_j = await outages.fetch()

    subsidence_lookup = {"Mapunapuna":22, "Manoa":8, "Kakaako":15}
    subs_mm = subsidence_lookup.get("Mapunapuna", 0)

    cues = [
        assess_floodsense(usgs_j, tide_j, nws_j, subsidence_mm=subs_mm, place="Manoa"),
        assess_debris(nws_j, place="Windward Oahu"),
        assess_microgrid(out_j, place="Oahu"),
    ]
    order = {Status.AVOID.value: 0, Status.WATCH.value: 1, Status.CLEAR.value: 2}
    cues.sort(key=lambda c: order[c["status"]])
    payload = {
        "area": "Honolulu/Oahu",
        "generated_at": now_utc().isoformat(),
        "valid_minutes": CFG.area_defaults["valid_minutes"],
        "cues": cues,
    }
    CACHE.set("today", payload)

@app.on_event("startup")
async def startup():
    await compute_today()
    interval = int(os.getenv("UPDATE_INTERVAL_MIN", "15"))
    # schedule background refresh
    def job():
        asyncio.run(compute_today())
    start_scheduler(job, minutes=interval)

@app.get("/health")
async def health():
    return {"ok": True, "ts": now_utc().isoformat()}

@app.get("/api/today")
async def api_today():
    data = CACHE.get("today")
    if not data:
        await compute_today()
        data = CACHE.get("today")
    return JSONResponse(data)

@app.get("/api/modules/{name}")
async def api_module(name: str):
    data = CACHE.get("today")
    if not data:
        await compute_today()
        data = CACHE.get("today")
    mapping = {
        "floodsense": "FloodSense",
        "debris": "Debris Mapper",
        "microgrid": "Microgrid Pulse",
    }
    target = mapping.get(name.lower())
    if not target:
        return JSONResponse({"error": "unknown module"}, status_code=404)
    for c in data["cues"]:
        if c["module"] == target:
            return JSONResponse(c)
    return JSONResponse({"error": "not found"}, status_code=404)

@app.get("/")
async def index():
    path = PUBLIC / "index.html"
    if not path.exists():
        return HTMLResponse("Oahu Clarity API is running.")
    return FileResponse(path)

# FILE: tests/test_adapters_contract.py
import asyncio
from app.adapters import usgs, noaa_coops, nws, outages

def test_usgs_fixture_contract():
    j = asyncio.run(usgs.fetch())
    assert j["source"] == "usgs"
    assert "level_ft" in j["data"]

def test_noaa_fixture_contract():
    j = asyncio.run(noaa_coops.fetch())
    assert j["source"] == "noaa"
    assert "tide_ft" in j["data"]

def test_nws_fixture_contract():
    j = asyncio.run(nws.fetch())
    assert j["source"] == "nws"
    assert "gust_kts" in j["data"]

def test_outages_fixture_contract():
    j = asyncio.run(outages.fetch())
    assert j["source"] == "outages"
    assert "clusters" in j["data"]

# FILE: tests/test_logic_floodsense.py
from app.logic.floodsense import assess_floodsense
from app.logic.status import Status

def mk_usgs(series): return {"data":{"level_ft":series}}
def mk_tide(v): return {"data":{"tide_ft":v}}
def mk_wind(g): return {"data":{"gust_kts":g}}

def test_watch_on_rise_and_tide():
    c = assess_floodsense(mk_usgs([2.0,2.6]), mk_tide(2.8), mk_wind(10), subsidence_mm=8)
    assert c["status"] in (Status.WATCH.value, Status.AVOID.value)

def test_avoid_on_subsidence():
    c = assess_floodsense(mk_usgs([2.4,3.0]), mk_tide(2.9), mk_wind(20), subsidence_mm=22)
    assert c["status"] == Status.AVOID.value

def test_clear_when_steady():
    c = assess_floodsense(mk_usgs([2.5,2.5]), mk_tide(1.0), mk_wind(5), subsidence_mm=0)
    assert c["status"] == Status.CLEAR.value

# FILE: tests/test_logic_debris.py
from app.logic.debris_mapper import assess_debris
from app.logic.status import Status

def test_watch_on_gust_or_rain():
    c = assess_debris({"data":{"sustained_kts":16,"gust_kts":35,"rain_24h_in":1.2}})
    assert c["status"] == Status.WATCH.value

def test_avoid_on_high_gusts():
    c = assess_debris({"data":{"sustained_kts":18,"gust_kts":42,"rain_24h_in":0.5}})
    assert c["status"] == Status.AVOID.value

# FILE: tests/test_logic_microgrid.py
from app.logic.microgrid_pulse import assess_microgrid
from app.logic.status import Status

def test_watch_on_clusters():
    c = assess_microgrid({"data":{"clusters":4,"customers_affected":1200}})
    assert c["status"] == Status.WATCH.value

def test_avoid_on_many_clusters():
    c = assess_microgrid({"data":{"clusters":9,"customers_affected":5200}})
    assert c["status"] == Status.AVOID.value

# FILE: tests/test_api_contract.py
from fastapi.testclient import TestClient
from app.api.routes import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_today_schema():
    r = client.get("/api/today")
    assert r.status_code == 200
    j = r.json()
    assert "area" in j and "cues" in j
    assert isinstance(j["cues"], list) and len(j["cues"]) >= 3
    for c in j["cues"]:
        for k in ["module","status","message","rationale","inputs","valid_until"]:
            assert k in c

# FILE: app/adapters/__init__.py
from . import usgs, noaa_coops, nws, outages  # noqa: F401

# FILE: Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV LIVE_MODE=false
ENV LOG_LEVEL=INFO
ENV UPDATE_INTERVAL_MIN=15
ENV USER_AGENT="OahuClarityPrototype/0.1 (contact: community@example.org)"

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app app
COPY public public
COPY fixtures fixtures
COPY config.yml config.yml
COPY STYLE.md STYLE.md
COPY README.md README.md

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn","app.api.routes:app","--host","0.0.0.0","--port","8000"]

# FILE: .dockerignore
__pycache__/
*.pyc
*.pyo
*.pytest_cache/
.venv/
.env
.git
.gitignore
*.log
dist/
build/

# FILE: Makefile
APP?=oahu-clarity
IMAGE?=oahu-clarity:0.1
PORT?=8000

.PHONY: help
help:
	@echo "Targets:"
	@echo "  run           - run locally (fixtures/offline)"
	@echo "  live          - run locally (live mode; set env vars)"
	@echo "  test          - run pytest suite"
	@echo "  docker-build  - build Docker image"
	@echo "  docker-run    - run container (fixtures)"
	@echo "  docker-live   - run container (live; pass env file)"

.PHONY: run
run:
	LIVE_MODE=false uvicorn app.api.routes:app --reload --port $(PORT)

.PHONY: live
live:
	LIVE_MODE=true uvicorn app.api.routes:app --reload --port $(PORT)

.PHONY: test
test:
	pytest -q

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm -p $(PORT):8000 -e LIVE_MODE=false --name $(APP) $(IMAGE)

.PHONY: docker-live
docker-live:
	@if [ ! -f .env ]; then echo "Create .env first (see .env.example)"; exit 1; fi
	docker run --rm -p $(PORT):8000 --env-file .env --name $(APP) $(IMAGE)