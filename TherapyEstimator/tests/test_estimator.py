import json

import numpy as np
import pytest

from therapy_estimator.estimate import suggest, validate
from therapy_estimator.features import Config, build_steps
from therapy_estimator.models import ExponentialInsulinModel, piecewise_linear_absorbed
from therapy_estimator.nightscout import load
from therapy_estimator.synthetic import generate, true_values

BLOCKS = (0, 4, 8, 12, 16, 20)


def estimates(result):
    return {k: np.array([row[k]["estimate"] for row in result["rows"]]) for k in ("isf", "basal", "cr")}


def test_insulin_curve_matches_loop_shape():
    m = ExponentialInsulinModel(360, 75, 10)
    assert m.percent_effect_remaining(0) == 1.0
    assert m.percent_effect_remaining(10) == 1.0
    assert m.percent_effect_remaining(370) == 0.0
    assert m.minute_kernel().sum() == pytest.approx(1.0)
    activity = m.minute_kernel()
    assert 10 + 70 <= np.argmax(activity) <= 10 + 80


def test_carb_curve_is_monotonic_and_complete():
    p = np.linspace(-0.1, 1.1, 500)
    f = piecewise_linear_absorbed(p)
    assert f[0] == 0 and f[-1] == 1
    assert np.all(np.diff(f) >= -1e-12)


def write_ns(tmp_path, treatments, hours=12, basal=1.0):
    t0 = 1_780_000_000 // 300 * 300
    entries = [{"type": "sgv", "date": (t0 + 300 * i) * 1000, "sgv": 120} for i in range(hours * 12)]
    profile = [{"defaultProfile": "Default", "startDate": "2020-01-01T00:00:00Z", "store": {"Default": {
        "units": "mg/dL", "timezone": "UTC",
        "basal": [{"time": "00:00", "value": basal}], "sens": [{"time": "00:00", "value": 50}],
        "carbratio": [{"time": "00:00", "value": 10}]}}}]
    for name, data in (("entries", entries), ("treatments", treatments), ("profile", profile)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    return t0


def test_scheduled_basal_fills_gaps_between_temp_basals(tmp_path):
    t0 = write_ns(tmp_path, [])
    ds = load(tmp_path)
    steps = build_steps(ds, Config(block_hours=(0,), drift_hours=0))
    # Steady state on 1 U/h: insulin effect per 5 minutes equals what is delivered per 5 minutes.
    assert steps.ia[-12:] == pytest.approx(np.full(12, 1.0 / 12), rel=1e-3)


def test_suspend_temp_basal_stops_insulin(tmp_path):
    t0 = write_ns(tmp_path, [{"eventType": "Temp Basal", "rate": 0, "absolute": 0, "duration": 720,
                              "reason": "suspend", "created_at": "2026-05-28T00:00:00Z"}])
    ds = load(tmp_path)
    ds.temp_basals["time"] = float(t0)
    steps = build_steps(ds, Config(block_hours=(0,), drift_hours=0))
    assert steps.ia[-12:].sum() == pytest.approx(0.0, abs=1e-9)


@pytest.fixture(scope="module")
def clean_sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("clean")
    generate(d, egp_drift_sd=0.0, day_sd=0.0, absorption_jitter=(1, 1), carb_shape="loop")
    return d


@pytest.fixture(scope="module")
def realistic_sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("realistic")
    generate(d)
    return d


def test_recovers_true_settings_without_model_mismatch(clean_sim):
    r = suggest(load(clean_sim), Config(block_hours=BLOCKS, insulin=(360, 65, 10)), {}, 30)
    est, truth = estimates(r), true_values(clean_sim, BLOCKS)
    assert est["isf"] / truth["isf"] == pytest.approx(np.ones(6), abs=0.15)
    assert est["basal"] / truth["basal"] == pytest.approx(np.ones(6), abs=0.1)
    meal_blocks = [1, 3, 4]
    csf = est["isf"] / est["cr"]
    assert csf[meal_blocks] / truth["csf"][meal_blocks] == pytest.approx(np.ones(3), abs=0.1)


def test_realistic_data_close_to_truth_and_safe(realistic_sim):
    ds = load(realistic_sim)
    cfg = Config(block_hours=BLOCKS, insulin=(360, 65, 10))
    r = suggest(ds, cfg, {}, 30)
    est, truth = estimates(r), true_values(realistic_sim, BLOCKS)
    assert est["isf"] / truth["isf"] == pytest.approx(np.ones(6), abs=0.2)
    assert est["basal"] / truth["basal"] == pytest.approx(np.ones(6), abs=0.25)
    meal_blocks = [1, 3, 4]
    csf = est["isf"] / est["cr"]
    assert csf[meal_blocks] / truth["csf"][meal_blocks] == pytest.approx(np.ones(3), abs=0.15)
    for row in r["rows"]:
        for name in ("isf", "basal", "cr"):
            v = row[name]
            assert abs(v["suggested"] / v["current"] - 1) <= 0.2 + 1e-9


def test_beats_current_settings_on_held_out_days(realistic_sim):
    v = validate(load(realistic_sim), Config(block_hours=BLOCKS), {}, 30)
    for h in (60, 120):
        e = v["errors"][h]
        assert e["skattad"]["rmse"] < 0.9 * e["nuvarande"]["rmse"]
