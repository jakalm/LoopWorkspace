"""End to end: tune the insulin peak, validate on held out days, and suggest settings."""

from dataclasses import replace

import numpy as np

from .features import ABSORPTION_BASIS, build_steps, build_windows
from .fit import DEFAULT_SHAPE, bootstrap_settings, default_shape, fit, prediction_errors, settings_point, theta_from_profile
from .models import MGDL_PER_MMOL, piecewise_linear_absorbed


def split_days(steps, test_fraction=0.25):
    days = np.unique(steps.day[steps.valid])
    n_test = max(1, int(round(len(days) * test_fraction)))
    return days[:-n_test], days[-n_test:]


def validate(ds, cfg, fit_kw, window_min):
    steps = build_steps(ds, cfg)
    train, test = split_days(steps)
    theta0 = theta_from_profile(ds.profiles[-1], cfg.block_hours, steps.n_segments)
    res = fit(build_windows(steps, window_min, window_min, train), theta0, cfg.n_blocks, **fit_kw)
    models = {"nuvarande": (theta0, default_shape(cfg.n_meal_groups)), "skattad": (res.theta, res.shape)}
    errs = {h: prediction_errors(steps, models, h, test) for h in (30, 60, 120)}
    return {"errors": errs, "train_days": train, "test_days": test}


def tune(ds, cfg, fit_kw, window_min, peaks):
    rows = []
    for peak in peaks:
        c = replace(cfg, insulin=(cfg.insulin[0], float(peak), cfg.insulin[2]))
        errs = validate(ds, c, fit_kw, window_min)["errors"]
        rows.append((peak, errs[60]["skattad"]["rmse"], errs[120]["skattad"]["rmse"]))
    best = min(rows, key=lambda r: r[1] + r[2])
    return rows, replace(cfg, insulin=(cfg.insulin[0], float(best[0]), cfg.insulin[2]))


def suggest(ds, cfg, fit_kw, window_min, max_step=0.2, max_rel_width=0.4, min_carbs_g=300):
    steps = build_steps(ds, cfg)
    windows = build_windows(steps, window_min, window_min)
    profile = ds.profiles[-1]
    theta0 = theta_from_profile(profile, cfg.block_hours, steps.n_segments)
    res = fit(windows, theta0, cfg.n_blocks, **fit_kw)
    k = cfg.n_blocks
    current = settings_point(theta0, k)
    point = settings_point(res.theta, k)
    samples = bootstrap_settings(windows, steps.day[windows.start], theta0, k, fit_kw)
    carbs_per_block = (windows.Xc @ res.shape).sum(0)
    windows_per_block = np.bincount(steps.block[windows.start], minlength=k)

    rows = []
    for b in range(k):
        row = {"block": b, "start_hour": cfg.block_hours[b], "windows": int(windows_per_block[b]),
               "carbs_g": float(carbs_per_block[b])}
        for name in ("isf", "basal", "cr"):
            lo, med, hi = np.percentile(samples[name][:, b], [10, 50, 90])
            cur, est = current[name][b], point[name][b]
            reasons = []
            if (hi - lo) / med > max_rel_width:
                reasons.append("för osäker")
            if name == "cr" and carbs_per_block[b] < min_carbs_g:
                reasons.append("för få måltider")
            sug = cur if reasons else float(np.clip(est, cur * (1 - max_step), cur * (1 + max_step)))
            row[name] = {"current": cur, "estimate": est, "p10": lo, "p90": hi, "suggested": sug,
                         "capped": not reasons and abs(sug - est) > 1e-9, "unchanged_reason": ", ".join(reasons)}
        rows.append(row)
    return {"rows": rows, "fit": res, "steps": steps, "windows": windows, "units": profile.units}


def absorbed_fraction(shape, p):
    """Fraction absorbed at p times the entered absorption time, for a learned shape."""
    return sum(w * piecewise_linear_absorbed(p / f) for w, f in zip(shape, ABSORPTION_BASIS))


def _round(name, v, units):
    if name == "basal":
        return round(v / 0.05) * 0.05
    if name == "isf":
        return round(v / MGDL_PER_MMOL, 1) if units == "mmol" else round(v)
    return round(v, 1)


def format_report(result, cfg, validation, tuning=None):
    units = result["units"]
    isf_unit = "mmol/L/E" if units == "mmol" else "mg/dL/E"
    lines = []
    st = result["steps"]
    lines.append(f"Data: {len(np.unique(st.day))} dagar, {len(st.ia)} steg à 5 min, {st.valid.sum()} användbara "
                 f"({100 * st.valid.mean():.0f} %).")
    lines.append("Bortfiltrerat (steg): " + ", ".join(f"{k} {v}" for k, v in st.exclusions.items() if v))
    lines.append(f"Insulinmodell: verkan {cfg.insulin[0]:.0f} min, topp {cfg.insulin[1]:.0f} min, "
                 f"fördröjning {cfg.insulin[2]:.0f} min.")
    if tuning:
        lines.append("\nTrimning av insulintopp (RMSE på testdagar, mg/dL):")
        lines.append("  topp  60 min  120 min")
        for peak, r60, r120 in tuning:
            lines.append(f"  {peak:4.0f}  {r60:6.1f}  {r120:7.1f}")

    lines.append(f"\nValidering på de sista {len(validation['test_days'])} dagarna "
                 f"(tränad på {len(validation['train_days'])}), prediktionsfel RMSE i mg/dL:")
    lines.append("  horisont   ingen förändring   nuvarande   skattad   antal")
    for h, e in validation["errors"].items():
        lines.append(f"  {h:4d} min   {e['ingen förändring']['rmse']:16.1f}   {e['nuvarande']['rmse']:9.1f}   "
                     f"{e['skattad']['rmse']:7.1f}   {e['n']:5d}")

    res = result["fit"]
    lines.append(f"\nAnpassning på all data: {res.n} fönster, residual {res.sigma:.1f} mg/dL, "
                 f"autokorrelation {res.rho:.2f}.")
    lines.append("Inlärd kolhydratabsorption, andel absorberad vid 25/50/100 % av angiven absorptionstid "
                 f"(Loops standardkurva: {absorbed_fraction(DEFAULT_SHAPE, 0.25):.0%} / "
                 f"{absorbed_fraction(DEFAULT_SHAPE, 0.5):.0%} / 100%):")
    groups = res.shape.reshape(-1, len(ABSORPTION_BASIS))
    hours = cfg.meal_group_hours or (0,)
    for g, shape in enumerate(groups):
        fr = [absorbed_fraction(shape, p) for p in (0.25, 0.5, 1.0)]
        label = "alla måltider" if len(groups) == 1 else f"måltider från {hours[g]:02d}:00"
        lines.append(f"  {label}: {fr[0]:.0%} / {fr[1]:.0%} / {fr[2]:.0%}")

    titles = {"basal": "Basal (E/h)", "isf": f"Insulinkänslighet ({isf_unit})", "cr": "Kolhydratkvot (g/E)"}
    edges = list(cfg.block_hours) + [24]
    for name in ("basal", "isf", "cr"):
        lines.append(f"\n{titles[name]}")
        lines.append("  tid         nu   skattat   80 % intervall   förslag")
        for r in result["rows"]:
            v = r[name]
            fmt = lambda x: f"{_round(name, x, units):g}"
            note = v["unchanged_reason"] or ("begränsat till ±20 %" if v["capped"] else "")
            interval = f"{fmt(v['p10'])} – {fmt(v['p90'])}"
            lines.append(f"  {edges[r['block']]:02d}-{edges[r['block'] + 1]:02d}  {fmt(v['current']):>6}  "
                         f"{fmt(v['estimate']):>8}   {interval:<15}  {fmt(v['suggested']):>7}  {note}")
    lines.append("\nFörslagen ändras högst ±20 % per körning. Granska dem innan du ändrar något i Loop.")
    return "\n".join(lines)


def run(ds, cfg, fit_kw, window_min=30, do_tune=False, peaks=(55, 65, 75)):
    tuning = None
    if do_tune:
        tuning, cfg = tune(ds, cfg, fit_kw, window_min, peaks)
    validation = validate(ds, cfg, fit_kw, window_min)
    result = suggest(ds, cfg, fit_kw, window_min)
    return cfg, result, validation, format_report(result, cfg, validation, tuning)
