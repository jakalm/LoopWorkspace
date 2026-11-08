"""Generates Nightscout style data from a simulated patient with known true settings.

The simulation deliberately differs from the estimator's model: a different insulin peak, a different
carb absorption curve and random absorption times, day to day sensitivity changes, drifting liver glucose
output, unlogged hypo treatments, exercise, CGM noise and compression lows. A simple Loop style controller
doses with the (wrong) profile settings, so insulin delivery reacts to glucose like in real closed loop data.
"""

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .models import MGDL_PER_MMOL, ExponentialInsulinModel, piecewise_linear_absorbed

# Hour of day -> value schedules.
TRUE = {
    "basal": [(0, 0.75), (3, 1.05), (8, 0.8), (18, 0.9)],
    "isf": [(0, 55.0), (6, 38.0), (11, 46.0)],
    "cr": [(0, 11.0), (6, 8.0), (11, 13.0), (17, 11.0)],
}
PROFILE = {
    "basal": [(0, 0.8), (6, 0.9), (12, 0.8), (18, 0.85)],
    "isf_mmol": [(0, 2.5)],
    "cr": [(0, 10.0), (11, 12.0), (17, 11.0)],
}


def _sched(items, hours):
    hs = np.array([h for h, _ in items], float)
    vs = np.array([v for _, v in items], float)
    return vs[np.clip(np.searchsorted(hs, hours, side="right") - 1, 0, len(vs) - 1)]


def true_values(data_dir, block_hours, tz="Europe/Stockholm"):
    """True block values comparable to the estimator's: ISF and basal averaged over the block, and carb
    sensitivity (CSF, mg/dL per g) at the logged meals in the block, weighted by grams. CR is block ISF / CSF;
    when the true ISF changes within a block, CSF is the quantity to compare, since that is what Loop applies."""
    edges = list(block_hours) + [24]
    out = {"isf": [], "basal": [], "cr": [], "csf": []}
    meals = pd.DataFrame(json.loads((Path(data_dir) / "truth.json").read_text())["meals"],
                         columns=["time", "grams", "csf"])
    meal_hour = pd.to_datetime(meals.time, unit="s", utc=True).dt.tz_convert(tz)
    meal_hour = (meal_hour.dt.hour + meal_hour.dt.minute / 60).values
    for a, b in zip(edges[:-1], edges[1:]):
        h = np.arange(a * 60, b * 60) / 60
        isf, basal = _sched(TRUE["isf"], h), _sched(TRUE["basal"], h)
        out["isf"].append(isf.mean())
        out["basal"].append((isf * basal).mean() / isf.mean())
        sel = (meal_hour >= a) & (meal_hour < b)
        csf = np.average(meals.csf[sel], weights=meals.grams[sel]) if sel.any() else np.nan
        out["cr"].append(isf.mean() / csf)
        out["csf"].append(csf)
    return {k: np.array(v) for k, v in out.items()}


def generate(out_dir, days=42, seed=1, tz="Europe/Stockholm", start="2026-09-01",
             egp_drift_sd=0.08, day_sd=0.08, absorption_jitter=(0.8, 1.3), carb_shape="gamma",
             controller="tempbasal", fat_protein=0.0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz=tz).tz_convert("UTC").timestamp()
    m = days * 1440
    tmin = t0 + 60 * np.arange(m)
    lt = pd.to_datetime(tmin, unit="s", utc=True).tz_convert(tz)
    hour = lt.hour.values + lt.minute.values / 60
    day_idx = ((tmin - t0) // 86400).astype(int)

    true_basal = _sched(TRUE["basal"], hour)
    true_isf = _sched(TRUE["isf"], hour) * np.exp(rng.normal(0, day_sd, days))[day_idx]
    true_cr = _sched(TRUE["cr"], hour)
    prof_basal = _sched(PROFILE["basal"], hour)
    prof_isf = _sched(PROFILE["isf_mmol"], hour) * MGDL_PER_MMOL
    prof_cr = _sched(PROFILE["cr"], hour)

    true_ins = ExponentialInsulinModel(360, 65, 10)
    ctrl_ins = ExponentialInsulinModel(360, 75, 10)
    act = true_ins.minute_kernel()
    per = ctrl_ins.percent_effect_remaining(np.arange(len(act)) + 0.5)
    L = len(act)

    ia = np.zeros(m + L)  # true insulin activity, U/min
    net_iob = np.zeros(m + L)  # controller's IOB above scheduled basal
    carb_rate = np.zeros(m + L)  # true carb absorption, g/min, with true CSF applied later
    carb_csf = np.zeros(m + L)  # mg/dL per minute from carbs
    cob = np.zeros(m + L)  # controller's carbs on board (logged carbs)

    treatments, sensitivity_boost, meal_truth = [], np.ones(m), []

    def deliver(i, units, sched_rate):
        ia[i:i + L] += units * act
        net_iob[i:i + L] += (units - sched_rate / 60) * per

    def eat(i, grams, absorb_min, logged, logged_absorb):
        actual = absorb_min * rng.uniform(*absorption_jitter)
        k = np.arange(L * 2)
        if carb_shape == "gamma":
            frac = stats.gamma.cdf(k + 1, 3, scale=actual / 6.3) - stats.gamma.cdf(k, 3, scale=actual / 6.3)
        else:
            frac = piecewise_linear_absorbed((k + 1) / actual) - piecewise_linear_absorbed(k / actual)
        hi = min(i + 10 + len(k), m + L)
        csf = true_isf[min(i, m - 1)] / true_cr[min(i, m - 1)]
        carb_csf[i + 10:hi] += grams * frac[: hi - i - 10] * csf
        if logged and rng.random() < fat_protein:
            # Unlogged slow glucose from fat and protein, a few hours after the meal.
            extra = np.zeros(len(k))
            start, span = int(rng.uniform(90, 180)), int(rng.uniform(180, 300))
            extra[start:start + span] = 1.0 / span
            carb_csf[i:min(i + len(k), m + L)] += 0.35 * grams * extra[: min(len(k), m + L - i)] * csf
        if logged:
            g = round(grams * (1 + rng.normal(0, 0.05)))
            meal_truth.append((float(tmin[i]), g, float(csf)))
            cob_curve = 1 - piecewise_linear_absorbed((k - 10) / logged_absorb)
            hi2 = min(i + len(k), m + L)
            cob[i:hi2] += g * cob_curve[: hi2 - i]
            treatments.append({"eventType": "Carb Correction", "carbs": g, "absorptionTime": logged_absorb,
                               "created_at": _iso(tmin[i]), "syncIdentifier": str(uuid.uuid4())})
            return g
        return 0

    # Meal plan.
    meals = {}
    for d in range(days):
        for mean_h, sd_min, lo, hi_g, absorb in ((7, 40, 40, 70, 180), (12.5, 45, 60, 90, 180),
                                               (18.5, 45, 60, 100, 180), (15.5, 30, 15, 30, 120)):
            if absorb == 120 and rng.random() < 0.5:
                continue
            i = int(d * 1440 + mean_h * 60 + rng.normal(0, sd_min))
            meals[max(i, 0)] = (rng.uniform(lo, hi_g), absorb)

    exercise_days = set(rng.choice(days, size=days * 2 // 7, replace=False).tolist())
    for d in exercise_days:
        a = d * 1440 + 17 * 60
        sensitivity_boost[a:a + 180] = 1.6
        treatments.append({"eventType": "Temporary Override", "reason": "Workout", "duration": 165,
                           "insulinNeedsScaleFactor": 0.6, "created_at": _iso(tmin[a - 15]),
                           "syncIdentifier": str(uuid.uuid4())})
    for d in range(0, days, 10):
        treatments.append({"eventType": "Sensor Start", "created_at": _iso(tmin[d * 1440 + 600])})
    for d in range(0, days, 3):
        treatments.append({"eventType": "Site Change", "created_at": _iso(tmin[d * 1440 + 1200])})
    compression = np.zeros(m)
    for d in range(days):
        if rng.random() < 0.2:
            a = d * 1440 + 3 * 60
            compression[a:a + 25] = 40 * np.sin(np.linspace(0, np.pi, 25))

    sensor_bias = np.zeros(m)
    for d in range(0, days, 10):
        a = d * 1440 + 600
        sensor_bias[a:a + 240] = 20 * np.exp(-np.arange(240) / 60)

    bg = 110.0
    egp_noise = 0.0
    cgm = np.full(m, np.nan)
    last_cgm, temp_rate, temp_start = bg, None, None
    pending_bolus, last_rescue = {}, -999
    for i in range(m):
        egp_noise = 0.999 * egp_noise + rng.normal(0, egp_drift_sd * np.sqrt(1 - 0.999**2))
        isf_now = true_isf[i] * sensitivity_boost[i]
        sched_true = true_basal[i] * true_isf[i] * (1 + egp_noise) / 60
        bg += -isf_now * ia[i] + sched_true + carb_csf[i]
        bg = max(bg, 40.0)

        if i % 5 == 0:
            reading = bg + rng.normal(0, 4) - compression[i] + sensor_bias[i]
            cgm[i] = reading
            trend = reading - last_cgm
            last_cgm = reading

            if i in meals or any(j in meals for j in range(i, i + 5)):
                mi = next(j for j in range(i, i + 5) if j in meals)
                grams, absorb = meals.pop(mi)
                g = eat(i, grams, absorb, True, absorb)
                units = round((g / prof_cr[i] + max(0, (reading - 120) / prof_isf[i]) * 0.5) / 0.05) * 0.05
                pending_bolus[i + int(rng.uniform(0, 3)) * 5] = (units, False)

            if reading < 70 and i - last_rescue > 20:
                eat(i, 15, 40, rng.random() < 1 / 3, 30)
                last_rescue = i

            scale = 0.6 if sensitivity_boost[i] > 1 else 1.0
            eventual = reading + 3 * trend - net_iob[i] * prof_isf[i] / scale + cob[i] * prof_isf[i] / prof_cr[i] / scale
            if controller == "autobolus":
                # Loop's automatic bolus strategy: 40 % of the correction as a bolus, temp basal only to reduce.
                need = (eventual - 110) / (prof_isf[i] / scale)
                if need > 0:
                    units = round(0.4 * need / 0.05) * 0.05
                    if units >= 0.05:
                        pending_bolus.setdefault(i, (units, True))
                    rate = prof_basal[i] * scale
                else:
                    rate = float(np.clip(round((prof_basal[i] * scale + need * 2) / 0.05) * 0.05, 0, 3.0))
                rate = 0.0 if eventual < 80 else rate
            else:
                rate = prof_basal[i] * scale + (eventual - 110) / (prof_isf[i] / scale) * 2
                rate = 0.0 if eventual < 80 else float(np.clip(round(rate / 0.05) * 0.05, 0, 3.0))
            cancel = abs(rate - prof_basal[i]) < 0.05
            new = None if cancel else rate
            if new != temp_rate:
                if temp_rate is not None:
                    treatments.append(_temp(tmin[temp_start], temp_rate, i - temp_start))
                temp_rate, temp_start = new, i

        if i in pending_bolus:
            units, automatic = pending_bolus.pop(i)
            treatments.append({"eventType": "Correction Bolus", "insulin": units, "programmed": units,
                               "duration": 0, "type": "normal", "automatic": automatic,
                               "created_at": _iso(tmin[i]), "syncIdentifier": str(uuid.uuid4())})
            deliver(i, units, 0.0)
        rate_now = prof_basal[i] if temp_rate is None else temp_rate
        deliver(i, rate_now / 60, prof_basal[i])

    if temp_rate is not None:
        treatments.append(_temp(tmin[temp_start], temp_rate, m - temp_start))

    entries = [{"type": "sgv", "date": int(tmin[i] * 1000), "sgv": int(round(np.clip(cgm[i], 40, 400)))}
               for i in range(m) if not np.isnan(cgm[i])]
    profile = [{
        "defaultProfile": "Default", "startDate": _iso(t0 - 86400), "units": "mmol/L",
        "store": {"Default": {
            "units": "mmol/L", "timezone": tz,
            "basal": [{"time": f"{h:02d}:00", "timeAsSeconds": h * 3600, "value": v} for h, v in PROFILE["basal"]],
            "sens": [{"time": f"{h:02d}:00", "timeAsSeconds": h * 3600, "value": v} for h, v in PROFILE["isf_mmol"]],
            "carbratio": [{"time": f"{h:02d}:00", "timeAsSeconds": h * 3600, "value": v} for h, v in PROFILE["cr"]],
        }},
    }]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "entries.json").write_text(json.dumps(entries))
    (out / "treatments.json").write_text(json.dumps(treatments))
    (out / "profile.json").write_text(json.dumps(profile))
    (out / "truth.json").write_text(json.dumps({"meals": meal_truth}))
    cgm_valid = cgm[~np.isnan(cgm)]
    return {"tir": float(np.mean((cgm_valid >= 70) & (cgm_valid <= 180))), "mean": float(cgm_valid.mean())}


def _iso(ts):
    return pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _temp(start, rate, minutes):
    return {"eventType": "Temp Basal", "temp": "absolute", "rate": rate, "absolute": rate,
            "duration": float(minutes), "automatic": True, "created_at": _iso(start),
            "syncIdentifier": str(uuid.uuid4())}
