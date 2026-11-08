"""Fetches Loop data from Nightscout and parses it into plain tables.

Loop uploads boluses, temp basals (a suspend is a 0 U/h temp basal) and carb entries, but not
scheduled basal. Scheduled basal is reconstructed from the profile wherever no temp basal runs.
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import MGDL_PER_MMOL

RAW_FILES = ("entries.json", "treatments.json", "profile.json")


def _iso(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch(url, out_dir, days, token=None, api_secret=None, chunk_days=5, start=None, end=None):
    import requests

    url = url.rstrip("/")
    session = requests.Session()
    base_params = {}
    if token:
        base_params["token"] = token
    if api_secret:
        session.headers["api-secret"] = hashlib.sha1(api_secret.encode()).hexdigest()

    def get(path, params, count):
        r = session.get(url + path, params={**base_params, **params, "count": count}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if len(data) >= count:
            print(f"varning: {path} returnerade {len(data)} poster (taket), data kan saknas", file=sys.stderr)
        return data

    end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").floor("min")
    start = pd.Timestamp(start, tz="UTC") if start else end - pd.Timedelta(days=days)
    entries, treatments = [], []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(days=chunk_days), end)
        print(f"hämtar {t.date()} .. {t2.date()}", file=sys.stderr)
        entries += get(
            "/api/v1/entries/sgv.json",
            {"find[date][$gte]": int(t.timestamp() * 1000), "find[date][$lt]": int(t2.timestamp() * 1000)},
            50000,
        )
        treatments += get(
            "/api/v1/treatments.json",
            {"find[created_at][$gte]": _iso(t), "find[created_at][$lt]": _iso(t2)},
            50000,
        )
        t = t2
    profiles = get("/api/v1/profile.json", {}, 1000)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in zip(RAW_FILES, (entries, treatments, profiles)):
        (out / name).write_text(json.dumps(data))
    print(f"sparade {len(entries)} glukosvärden, {len(treatments)} behandlingar, {len(profiles)} profiler i {out}",
          file=sys.stderr)


@dataclass
class Schedule:
    offsets: np.ndarray  # seconds since local midnight
    values: np.ndarray

    def at(self, sec_of_day):
        idx = np.searchsorted(self.offsets, np.asarray(sec_of_day), side="right") - 1
        return self.values[np.clip(idx, 0, len(self.values) - 1)]


@dataclass
class Profile:
    start: float  # epoch seconds
    timezone: str
    units: str  # "mmol" or "mg/dL"
    basal: Schedule  # U/h
    isf: Schedule  # mg/dL per U
    cr: Schedule  # g per U


@dataclass
class Dataset:
    glucose: pd.DataFrame  # time, mgdl
    boluses: pd.DataFrame  # time, units, duration_s
    temp_basals: pd.DataFrame  # time, rate, duration_s
    carbs: pd.DataFrame  # time, grams, absorption_min
    overrides: pd.DataFrame  # time, end, scale
    events: pd.DataFrame  # time, kind
    profiles: list = field(default_factory=list)


def _time_of(t):
    for key in ("created_at", "timestamp", "date", "mills"):
        v = t.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v) / 1000.0
        try:
            ts = pd.Timestamp(v)
        except (ValueError, TypeError):
            continue
        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts).timestamp()
    return None


def _schedule(items, to_mgdl=1.0):
    offs, vals = [], []
    for it in items:
        if "timeAsSeconds" in it:
            sec = float(it["timeAsSeconds"])
        else:
            hh, mm = str(it["time"]).split(":")[:2]
            sec = int(hh) * 3600 + int(mm) * 60
        offs.append(sec)
        vals.append(float(it["value"]) * to_mgdl)
    order = np.argsort(offs)
    return Schedule(np.array(offs)[order], np.array(vals)[order])


def _timezone(name, default_tz):
    """Loop writes fixed offset zones as e.g. "ETC/GMT-1"; the tz database spells them "Etc/GMT-1"."""
    from zoneinfo import ZoneInfo

    if not name:
        return default_tz
    if name.upper().startswith("ETC/"):
        name = "Etc/" + name[4:].upper()
    try:
        ZoneInfo(name)
        return name
    except (KeyError, ValueError):
        print(f"varning: okänd tidszon {name!r}, använder {default_tz}", file=sys.stderr)
        return default_tz


def parse_profiles(docs, default_tz):
    profiles = []
    for doc in docs:
        store = doc.get("store") or {}
        if not store:
            continue
        p = store.get(doc.get("defaultProfile", "Default")) or next(iter(store.values()))
        units = str(p.get("units") or doc.get("units") or "mg/dL").lower()
        mmol = "mmol" in units
        sens = _schedule(p["sens"], MGDL_PER_MMOL if mmol else 1.0)
        if mmol and np.median(sens.values) > 30 * MGDL_PER_MMOL:
            print("varning: ISF ser ut att vara i mg/dL trots mmol-profil", file=sys.stderr)
            sens = _schedule(p["sens"])
        start = None
        for key in ("startDate", "created_at"):
            if doc.get(key):
                start = pd.Timestamp(doc[key])
                start = (start.tz_localize("UTC") if start.tzinfo is None else start).timestamp()
                break
        if start is None and doc.get("mills"):
            start = float(doc["mills"]) / 1000
        profiles.append(Profile(
            start=start or 0.0,
            timezone=_timezone(p.get("timezone"), default_tz),
            units="mmol" if mmol else "mg/dL",
            basal=_schedule(p["basal"]),
            isf=sens,
            cr=_schedule(p["carbratio"]),
        ))
    profiles.sort(key=lambda pr: pr.start)
    # Loop re-uploads the profile on every settings change; keep only changes.
    deduped = []
    for pr in profiles:
        if deduped and all(
            pr.timezone == deduped[-1].timezone
            and np.array_equal(getattr(pr, s).offsets, getattr(deduped[-1], s).offsets)
            and np.allclose(getattr(pr, s).values, getattr(deduped[-1], s).values)
            for s in ("basal", "isf", "cr")
        ):
            continue
        deduped.append(pr)
    return deduped


def load(data_dir, default_tz="Europe/Stockholm"):
    d = Path(data_dir)
    entries = json.loads((d / "entries.json").read_text())
    treatments = json.loads((d / "treatments.json").read_text())
    profile_docs = json.loads((d / "profile.json").read_text())

    g = pd.DataFrame(
        [(float(e["date"]) / 1000.0, float(e["sgv"])) for e in entries
         if e.get("sgv") is not None and e.get("date") is not None and e.get("type", "sgv") == "sgv"],
        columns=["time", "mgdl"],
    )
    g = g[(g.mgdl >= 39) & (g.mgdl <= 401)]
    # Several uploaders (e.g. the Dexcom app and Loop) can send the same readings seconds apart.
    g = g.sort_values("time")
    keep, last = [], -np.inf
    for t in g.time.values:
        keep.append(t - last >= 150)
        if keep[-1]:
            last = t
    g = g[np.array(keep, dtype=bool)]

    seen = set()
    boluses, temps, carbs, overrides, events = [], [], [], [], []
    for t in treatments:
        key = t.get("syncIdentifier") or t.get("_id")
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        ts = _time_of(t)
        if ts is None:
            continue
        et = t.get("eventType", "")
        dur_min = float(t.get("duration") or 0.0)
        if et == "Temp Basal":
            rate = t.get("absolute", t.get("rate"))
            if rate is not None:
                temps.append((ts, float(rate), dur_min * 60))
            continue
        if et == "Temporary Override":
            end = ts + dur_min * 60 if t.get("durationType") != "indefinite" and dur_min > 0 else np.nan
            scale = t.get("insulinNeedsScaleFactor")
            overrides.append((ts, end, 1.0 if scale is None else float(scale)))
            continue
        if et == "Sensor Start":
            events.append((ts, "sensor_start"))
        elif et == "Site Change":
            events.append((ts, "site_change"))
        if t.get("insulin"):
            boluses.append((ts, float(t["insulin"]), dur_min * 60, bool(t.get("automatic", False))))
        if t.get("carbs"):
            absorption = t.get("absorptionTime")
            carbs.append((ts, float(t["carbs"]), float(absorption) if absorption else np.nan))

    ov = pd.DataFrame(overrides, columns=["time", "end", "scale"]).sort_values("time")
    if len(ov):
        nxt = ov.time.shift(-1).fillna(ov.time.iloc[-1] + 12 * 3600)
        ov["end"] = np.where(ov.end.isna(), np.minimum(nxt, ov.time + 12 * 3600), ov.end)

    profiles = parse_profiles(profile_docs, default_tz)
    if not profiles:
        raise ValueError("ingen profil hittades i profile.json")
    # Profiles created after the data ends are irrelevant, and the last profile is used as current settings.
    end = g.time.max() if len(g) else np.inf
    profiles = [p for p in profiles if p.start <= end] or profiles[:1]

    return Dataset(
        glucose=g.reset_index(drop=True),
        boluses=pd.DataFrame(boluses, columns=["time", "units", "duration_s", "automatic"]).sort_values("time"),
        temp_basals=pd.DataFrame(temps, columns=["time", "rate", "duration_s"]).sort_values("time"),
        carbs=pd.DataFrame(carbs, columns=["time", "grams", "absorption_min"]).sort_values("time"),
        overrides=ov.reset_index(drop=True),
        events=pd.DataFrame(events, columns=["time", "kind"]).sort_values("time"),
        profiles=profiles,
    )
