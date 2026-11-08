"""Command line: fetch | estimate | synth."""

import argparse
import json
import os
import sys

from .features import Config
from .models import INSULIN_PRESETS


def main(argv=None):
    p = argparse.ArgumentParser(prog="therapy_estimator")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="hämta data från Nightscout")
    f.add_argument("--url", default=os.environ.get("NS_URL"))
    f.add_argument("--days", type=int, default=60)
    f.add_argument("--out", default="data")
    f.add_argument("--start", default=None, help="startdatum (UTC), t.ex. 2025-01-01; ersätter --days")
    f.add_argument("--end", default=None, help="slutdatum (UTC)")

    e = sub.add_parser("estimate", help="skatta inställningar från hämtad data")
    e.add_argument("--data", default="data")
    e.add_argument("--insulin", choices=sorted(INSULIN_PRESETS), default="rapid-adult")
    e.add_argument("--blocks", default="0,4,8,12,16,20", help="blockens starttimmar")
    e.add_argument("--window", type=int, default=30, help="fönsterlängd i minuter")
    e.add_argument("--prior", type=float, default=0.5, help="relativ osäkerhet i nuvarande inställningar")
    e.add_argument("--smooth", type=float, default=0.2, help="tillåten relativ skillnad mellan grannblock")
    e.add_argument("--tune", action="store_true", help="trimma insulintopp och absorptionstid")
    e.add_argument("--tz", default="Europe/Stockholm", help="lokal tidszon för dygnsblocken")
    e.add_argument("--meal-groups", default=None, metavar="TIMMAR",
                   help="egen absorptionskurva per måltidsgrupp, starttimmar t.ex. 4,10,15,21. Kan dra ner ISF; "
                        "använd för att se absorptionstider, inte för inställningar")
    e.add_argument("--json", default=None, help="spara resultatet som JSON")

    c = sub.add_parser("check", help="modellfria kontroller av datan")
    c.add_argument("--data", default="data")
    c.add_argument("--insulin", choices=sorted(INSULIN_PRESETS), default="rapid-adult")
    c.add_argument("--tz", default="Europe/Stockholm")

    pa = sub.add_parser("podage", help="insulineffekt per poddag")
    pa.add_argument("--data", default="data")
    pa.add_argument("--insulin", choices=sorted(INSULIN_PRESETS), default="rapid-adult")
    pa.add_argument("--tz", default="Europe/Stockholm")

    s = sub.add_parser("synth", help="generera syntetisk testdata")
    s.add_argument("--out", default="data_synth")
    s.add_argument("--days", type=int, default=42)
    s.add_argument("--seed", type=int, default=1)

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        from .nightscout import fetch
        if not a.url:
            sys.exit("ange --url eller NS_URL")
        fetch(a.url, a.out, a.days, token=os.environ.get("NS_TOKEN"), api_secret=os.environ.get("NS_API_SECRET"),
              start=a.start, end=a.end)
    elif a.cmd == "check":
        from .checks import run_checks
        from .nightscout import load
        print(run_checks(load(a.data), Config(insulin=INSULIN_PRESETS[a.insulin], timezone=a.tz)))
    elif a.cmd == "podage":
        from .nightscout import load
        from .podage import POD_DAYS, pod_age_effect
        r = pod_age_effect(load(a.data), Config(insulin=INSULIN_PRESETS[a.insulin], timezone=a.tz))
        print("Insulineffekt jämfört med poddag 1 (80 % intervall, andel av insulinet):")
        for i, name in enumerate(POD_DAYS[1:], start=1):
            print(f"  {name}: {r['factor'][i]:.2f} ({r['p10'][i]:.2f}–{r['p90'][i]:.2f}, {r['insulin_share'][i]:.0%})")
    elif a.cmd == "synth":
        from .synthetic import generate
        print(generate(a.out, a.days, a.seed))
    else:
        from .estimate import run
        from .nightscout import load
        ds = load(a.data)
        cfg = Config(block_hours=tuple(int(x) for x in a.blocks.split(",")), insulin=INSULIN_PRESETS[a.insulin],
                     timezone=a.tz,
                     meal_group_hours=tuple(int(x) for x in a.meal_groups.split(",")) if a.meal_groups else None)
        cfg, result, validation, report = run(ds, cfg, {"prior_rel": a.prior, "smooth_rel": a.smooth},
                                              a.window, a.tune)
        print(report)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump({"insulin": cfg.insulin, "carb_absorption_scale": cfg.carb_absorption_scale,
                           "units": result["units"], "blocks": result["rows"]}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
