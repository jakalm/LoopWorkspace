"""Model free checks that do not depend on the estimator's fit."""

import numpy as np
import pandas as pd

from .features import build_steps
from .models import MGDL_PER_MMOL


BLOCKS = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24))


def effective_carb_ratios(ds, cfg, steps=None, assumed_isf_mmol=(1.5, 2.5)):
    """Grams per unit of insulin actually given (manual and automatic boluses, basal above or below the
    schedule) for meals with no other meal 4 h before or 5 h after, corrected for the glucose change over
    5 h with an assumed ISF. Meals followed by a hypo are left out of the ratio, since hypo treatments
    are often not logged."""
    st = steps or build_steps(ds, cfg)
    t, bg = st.t[:-1], st.bg
    bol = np.zeros(len(t))
    idx = ((ds.boluses.time.values - t[0]) // 300).astype(int)
    ok = (idx >= 0) & (idx < len(t))
    np.add.at(bol, idx[ok], ds.boluses.units.values[ok])
    net = st.basal - st.sched + bol
    c = ds.carbs.sort_values("time")
    ct, cg = c.time.values, c.grams.values
    hours = pd.to_datetime(ct, unit="s", utc=True).tz_convert(cfg.timezone)
    hours = hours.hour.values + hours.minute.values / 60
    window = 5 * 12
    rows = []
    for j, (ts, g) in enumerate(zip(ct, cg)):
        if g < 15 or np.any((np.abs(ct - ts) > 0) & (ct > ts - 4 * 3600) & (ct < ts + 5 * 3600)):
            continue
        i = int((ts - t[0]) // 300)
        if i < 72 or i + window >= len(t) or np.isnan(bg[i]) or np.isnan(bg[i + window]):
            continue
        rows.append((hours[j], g, net[i - 6:i + window].sum(), bg[i + window] - bg[i],
                     np.nanmin(bg[i:i + window]) < cfg.hypo_mgdl))
    r = pd.DataFrame(rows, columns=["hour", "grams", "units", "dbg", "hypo"])
    out = []
    for a, b in BLOCKS:
        g = r[(r.hour >= a) & (r.hour < b)]
        ok = g[~g.hypo]
        ratios = [float(np.median(ok.grams / (ok.units + ok.dbg / (isf * MGDL_PER_MMOL)))) if len(ok) else np.nan
                  for isf in assumed_isf_mmol]
        out.append({"block": f"{a:02d}-{b:02d}", "meals": len(g), "without_hypo": len(ok),
                    "grams": float(g.grams.median()) if len(g) else np.nan, "ratio": ratios,
                    "hypo_share": float(g.hypo.mean()) if len(g) else np.nan})
    return out


def run_checks(ds, cfg):
    st = build_steps(ds, cfg)
    t, bg = st.t[:-1], st.bg
    lt = pd.to_datetime(t, unit="s", utc=True).tz_convert(cfg.timezone)
    hour = lt.hour.values
    profile = ds.profiles[-1]
    mmol = profile.units == "mmol"
    unit = "mmol/L" if mmol else "mg/dL"
    conv = MGDL_PER_MMOL if mmol else 1.0
    net = st.ia - st.sched
    carb_t = ds.carbs.time.values
    lines = []

    # Correction episodes: high glucose, no carbs 3 h before to 4 h after, extra insulin above the schedule.
    high, n_steps = 180, 48
    rows, i = [], 72
    while i < len(t) - n_steps:
        if (bg[i] > high and st.valid[i:i + n_steps].all()
                and not np.any((carb_t > t[i] - 3 * 3600) & (carb_t < t[i] + 4 * 3600))):
            ins = net[i:i + n_steps].sum()
            if ins > 0.5:
                rows.append((bg[i + n_steps] - bg[i], ins))
                i += n_steps
                continue
        i += 1
    if rows:
        r = np.array(rows)
        isf = -r[:, 0].sum() / r[:, 1].sum() / conv
        lines.append(f"Korrigeringar utan kolhydrater (start över {high / conv:.0f} {unit}, 4 h): {len(r)} st. "
                     f"Glukos {r[:, 0].mean() / conv:+.1f} {unit} med {r[:, 1].mean():.1f} E utöver basal, "
                     f"dvs ISF ≈ {isf:.1f} (inställning {profile.isf.values.mean() / conv:.1f}). "
                     "Underskattar troligen ISF, eftersom Loop ger mer insulin när glukos stiger av andra orsaker.")
    else:
        lines.append("Korrigeringar utan kolhydrater: inga hittades.")

    nights = []
    for d in np.unique(st.day):
        idx = np.flatnonzero((st.day == d) & (hour < 6))
        if len(idx) < 70 or not st.valid[idx].all():
            continue
        if np.any((carb_t > t[idx[0]] - 3 * 3600) & (carb_t < t[idx[-1]])):
            continue
        nights.append(((bg[idx[-1] + 1] - bg[idx[0]]) / conv, net[idx].sum()))
    if nights:
        n = np.array(nights)
        lines.append(f"Nätter 00–06 utan kolhydrater: {len(n)} st. Glukos {n[:, 0].mean():+.1f} {unit}, "
                     f"Loop gav {n[:, 1].mean():+.1f} E jämfört med schemat ({n[:, 1].mean() / 6:+.2f} E/h).")

    low = (bg[:-1] < cfg.hypo_mgdl) & ~np.isnan(bg[:-1])
    starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    since = np.array([(t[s] - carb_t[carb_t <= t[s]].max()) / 3600 if np.any(carb_t <= t[s]) else np.inf
                      for s in starts])
    valid_bg = bg[~np.isnan(bg)]
    lines.append(f"Tid i mål {np.mean((valid_bg >= 70) & (valid_bg <= 180)):.0%}, under {np.mean(valid_bg < 70):.1%}, "
                 f"över {np.mean(valid_bg > 180):.0%}. Känningar: {len(starts)} st, "
                 f"{np.mean(since < 4) if len(starts) else 0:.0%} inom 4 h efter loggad måltid.")
    lines.append("Faktisk kolhydratkvot (g/E) för fristående måltider, med antagen ISF 1,5 / 2,5 mmol/L/E:")
    for row in effective_carb_ratios(ds, cfg, st):
        if not row["meals"]:
            continue
        r15, r25 = row["ratio"]
        ratio = "–" if np.isnan(r15) else f"{r15:.1f} / {r25:.1f}"
        lines.append(f"  {row['block']}: {row['meals']:3d} måltider ({row['without_hypo']} utan känning), "
                     f"median {row['grams']:.0f} g, kvot {ratio}, känning efteråt {row['hypo_share']:.0%}")
    lines.append("Känningar per timme: " + " ".join(f"{h:02d}:{c}" for h, c in
                                                   enumerate(np.bincount(hour[starts], minlength=24)) if c))
    return "\n".join(lines)
