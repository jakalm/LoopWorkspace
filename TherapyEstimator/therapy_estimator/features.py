"""Turns a Dataset into a 5 minute grid of model inputs and fit windows.

Model per 5 minute step, with parameters per time of day block k:
    dBG = -ISF_k * insulin_activity + (EGP_k + drift_s) * dt_hours + CSF_j * carbs_absorbed
where EGP_k = ISF_k * basal_k (the glucose rise per hour that the true basal rate exactly cancels),
CSF_j = ISF_j / CR_j for the block j the carbs were entered in, and drift_s is a slow offset per
segment of the data (illness, hormones, ...). This is linear in the parameters, so it can be fit
by least squares. Insulin activity uses all delivered insulin (scheduled basal, temp basals, boluses).

The carb absorption curve is a weighted mix of Loop's piecewise linear curve stretched to several
multiples of the entered absorption time. The weights are learned from the data (see fit.py), since a
wrong absorption shape otherwise leaks into the ISF estimate.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import ExponentialInsulinModel, piecewise_linear_absorbed

STEP_MIN = 5
ABSORPTION_BASIS = (0.33, 0.67, 1.0, 1.5, 2.0)


@dataclass
class Config:
    block_hours: tuple = (0, 4, 8, 12, 16, 20)
    insulin: tuple = (360.0, 75.0, 10.0)
    carb_delay_min: float = 10.0
    carb_absorption_scale: float = 1.0
    default_absorption_min: float = 180.0
    # Local start hours of meal groups that get their own absorption shape, e.g. (4, 10, 15, 21) for
    # breakfast, lunch, dinner and evening; None gives one shape for all meals.
    meal_group_hours: tuple | None = None
    drift_hours: int = 12  # 0 disables the drift offsets
    hypo_mgdl: float = 70.0
    post_hypo_min: float = 120.0
    max_roc_mgdl_per_min: float = 5.0
    exclude_after_event_min: dict = field(default_factory=lambda: {"sensor_start": 120, "site_change": 120})
    timezone: str = "Europe/Stockholm"  # local clock for the time of day blocks

    @property
    def n_blocks(self):
        return len(self.block_hours)

    @property
    def n_meal_groups(self):
        return len(self.meal_group_hours) if self.meal_group_hours else 1

    def meal_group_of(self, sec_of_day):
        if not self.meal_group_hours:
            return np.zeros(len(sec_of_day), dtype=np.int64)
        starts = np.array(self.meal_group_hours, dtype=float) * 3600
        g = np.searchsorted(starts, sec_of_day, side="right") - 1
        return np.where(g < 0, len(starts) - 1, g)


@dataclass
class Steps:
    t: np.ndarray  # step boundary times, epoch s (N + 1)
    bg: np.ndarray  # glucose at boundaries, mg/dL, NaN when missing (N + 1)
    ia: np.ndarray  # insulin effect during step, U (N)
    ia_pod: np.ndarray  # same, split by pod age at delivery: day 1, day 2, day 3+, unknown (N x 4)
    sched: np.ndarray  # scheduled basal delivered per step under the profile in effect, U (N)
    basal: np.ndarray  # basal actually delivered per step (scheduled or temp), U (N)
    carbs: np.ndarray  # carbs absorbed during step by entry block and (meal group, absorption basis), g (N x K x G*B)
    block: np.ndarray  # time of day block per step (N)
    segment: np.ndarray  # drift segment per step (N)
    n_segments: int
    day: np.ndarray  # local calendar day ordinal per step (N)
    valid: np.ndarray  # step usable for fitting (N)
    exclusions: dict  # reason -> number of excluded steps
    n_blocks: int

    @property
    def n_params(self):
        return 3 * self.n_blocks + self.n_segments

    def features(self):
        """Per step design matrix [ISF_0..K-1, EGP_0..K-1, CSF_0..K-1 (left zero), drift_0..S-1]
        and the carb columns per absorption basis (N x K x B)."""
        n, k = len(self.ia), self.n_blocks
        f = np.zeros((n, self.n_params))
        rows = np.arange(n)
        f[rows, self.block] = -self.ia
        f[rows, k + self.block] = STEP_MIN / 60.0
        if self.n_segments:
            f[rows, 3 * k + self.segment] = STEP_MIN / 60.0
        return f, self.carbs


def _local_seconds(times, tz):
    lt = pd.to_datetime(times, unit="s", utc=True).tz_convert(tz)
    sec = lt.hour.values * 3600 + lt.minute.values * 60 + lt.second.values
    day = lt.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64)
    return sec, day


def _pod_day(ds, times, min_life_h=40, max_life_h=84):
    """Pod age in whole days at each time (0, 1, 2 for day 3 and later), or 3 when unknown. Pods whose
    next site change is missing or implausibly far away count as unknown."""
    changes = np.sort(ds.events[ds.events.kind == "site_change"].time.values)
    out = np.full(len(times), 3, dtype=np.int64)
    if not len(changes):
        return out
    j = np.searchsorted(changes, times, side="right") - 1
    has = j >= 0
    last = changes[np.clip(j, 0, None)]
    nxt = np.append(changes, np.inf)[np.clip(j + 1, 0, len(changes))]
    age_h = (times - last) / 3600
    life_h = (nxt - last) / 3600
    ok = has & (age_h < max_life_h) & (((life_h >= min_life_h) & (life_h <= max_life_h)) | np.isinf(nxt))
    out[ok] = np.minimum(age_h[ok] // 24, 2).astype(np.int64)
    return out


def _block_of(sec, cfg):
    block_starts = np.array(cfg.block_hours, dtype=float) * 3600
    return np.clip(np.searchsorted(block_starts, sec, side="right") - 1, 0, cfg.n_blocks - 1)


def build_steps(ds, cfg: Config) -> Steps:
    tz = cfg.timezone
    t0 = np.floor(ds.glucose.time.min() / 300) * 300
    t_end = np.floor(ds.glucose.time.max() / 300) * 300
    n = int((t_end - t0) / 300)
    m = n * STEP_MIN
    k_blocks = cfg.n_blocks
    tmin = t0 + 60 * np.arange(m)

    # Scheduled basal per minute from the profile in effect, in that profile's (pump) time zone.
    starts = np.array([p.start for p in ds.profiles])
    pidx = np.clip(np.searchsorted(starts, tmin, side="right") - 1, 0, len(ds.profiles) - 1)
    sched = np.empty(m)
    for i, p in enumerate(ds.profiles):
        mask = pidx == i
        if mask.any():
            sched[mask] = p.basal.at(_local_seconds(tmin[mask], p.timezone)[0])

    temp = np.full(m, np.nan)
    tb = ds.temp_basals
    if len(tb):
        st = tb.time.values
        end = np.minimum(st + tb.duration_s.values, np.append(st[1:], np.inf))
        i0 = np.round((st - t0) / 60).astype(np.int64)
        i1 = np.round((end - t0) / 60).astype(np.int64)
        for a, b, r in zip(i0, i1, tb.rate.values):
            a, b = max(a, 0), min(b, m)
            if b > a:
                temp[a:b] = r
    delivered = np.where(np.isnan(temp), sched, temp) / 60.0

    for ts, units, dur in ds.boluses[["time", "units", "duration_s"]].itertuples(index=False):
        a = int(np.floor((ts - t0) / 60))
        span = max(int(round(dur / 60)), 1)
        if a + span <= 0 or a >= m:
            continue
        lo, hi = max(a, 0), min(a + span, m)
        delivered[lo:hi] += units / span

    model = ExponentialInsulinModel(*cfg.insulin)
    kernel = model.minute_kernel()
    ia_min = np.convolve(delivered, kernel)[:m]
    pod_day = _pod_day(ds, tmin)
    ia_pod = np.stack([np.convolve(delivered * (pod_day == d), kernel)[:m].reshape(n, STEP_MIN).sum(1)
                       for d in range(4)], axis=1)

    n_basis = len(ABSORPTION_BASIS)
    n_cols = cfg.n_meal_groups * n_basis
    carbs_min = np.zeros((m, k_blocks, n_cols))
    entry_block, entry_group = [], []
    if len(ds.carbs):
        entry_sec, _ = _local_seconds(ds.carbs.time.values, tz)
        entry_block, entry_group = _block_of(entry_sec, cfg), cfg.meal_group_of(entry_sec)
    for (ts, grams, absorption), b, grp in zip(ds.carbs[["time", "grams", "absorption_min"]].itertuples(index=False),
                                               entry_block, entry_group):
        if np.isnan(absorption):
            absorption = cfg.default_absorption_min
        s = (ts - t0) / 60 + cfg.carb_delay_min
        for j, factor in enumerate(ABSORPTION_BASIS):
            T = absorption * cfg.carb_absorption_scale * factor
            lo, hi = max(int(np.floor(s)), 0), min(int(np.ceil(s + T)) + 1, m)
            if hi > lo:
                k = np.arange(lo, hi)
                carbs_min[lo:hi, b, grp * n_basis + j] += grams * (piecewise_linear_absorbed((k + 1 - s) / T)
                                                   - piecewise_linear_absorbed((k - s) / T))

    ia = ia_min.reshape(n, STEP_MIN).sum(1)
    carbs = carbs_min.reshape(n, STEP_MIN, k_blocks, n_cols).sum(1)

    tb_bounds = t0 + 300 * np.arange(n + 1)
    gt, gv = ds.glucose.time.values, ds.glucose.mgdl.values
    j = np.searchsorted(gt, tb_bounds)
    has_both = (j > 0) & (j < len(gt))
    jl, jr = np.clip(j - 1, 0, len(gt) - 1), np.clip(j, 0, len(gt) - 1)
    exact = np.isclose(gt[jr], tb_bounds)
    ok = (has_both & (gt[jr] - gt[jl] <= 12 * 60)) | exact
    bg = np.where(ok, np.interp(tb_bounds, gt, gv), np.nan)

    step_sec, day = _local_seconds(tb_bounds[:-1] + 150, tz)
    block = _block_of(step_sec, cfg)
    if cfg.drift_hours:
        per_day = 24 // cfg.drift_hours
        raw = (day - day.min()) * per_day + step_sec // (cfg.drift_hours * 3600)
        segment = raw.astype(np.int64)
        n_segments = int(segment.max()) + 1
    else:
        segment, n_segments = np.zeros(n, dtype=np.int64), 0

    reasons = {}
    valid = np.ones(n, dtype=bool)

    def exclude(name, mask):
        reasons[name] = int((mask & valid).sum())
        valid[mask] = False

    exclude("saknar glukos", np.isnan(bg[:-1]) | np.isnan(bg[1:]))
    hist = np.zeros(n, dtype=bool)
    hist[: int(np.ceil(model.effect_duration / STEP_MIN))] = True
    exclude("för kort insulinhistorik", hist)

    ov_mask = np.zeros(m, dtype=bool)
    for ts, end, scale in ds.overrides[["time", "end", "scale"]].itertuples(index=False):
        if abs(scale - 1.0) > 1e-6:
            a, b = max(int((ts - t0) // 60), 0), min(int(np.ceil((end - t0) / 60)), m)
            ov_mask[a:b] = True
    exclude("override aktiv", ov_mask.reshape(n, STEP_MIN).any(1))

    low = np.nan_to_num(bg[:-1], nan=1e9) < cfg.hypo_mgdl
    w = int(cfg.post_hypo_min / STEP_MIN)
    c = np.concatenate([[0], np.cumsum(low)])
    idx = np.arange(n)
    exclude("efter känning", (c[idx + 1] - c[np.maximum(idx - w, 0)]) > 0)

    roc = np.abs(np.diff(bg)) / STEP_MIN > cfg.max_roc_mgdl_per_min
    roc = np.convolve(roc.astype(float), np.ones(5), mode="same") > 0
    exclude("orimlig förändringstakt", roc)

    ev_mask = np.zeros(n, dtype=bool)
    for ts, kind in ds.events[["time", "kind"]].itertuples(index=False):
        dur = cfg.exclude_after_event_min.get(kind, 0)
        a, b = int((ts - t0) // 300), int(np.ceil((ts + dur * 60 - t0) / 300))
        ev_mask[max(a, 0):max(min(b, n), 0)] = True
    exclude("sensorstart/podbyte", ev_mask)

    return Steps(t=tb_bounds, bg=bg, ia=ia, ia_pod=ia_pod, sched=sched.reshape(n, STEP_MIN).sum(1) / 60.0,
                 basal=np.where(np.isnan(temp), sched, temp).reshape(n, STEP_MIN).sum(1) / 60.0, carbs=carbs, block=block,
                 segment=segment, n_segments=n_segments, day=day, valid=valid, exclusions=reasons,
                 n_blocks=k_blocks)


@dataclass
class Windows:
    X: np.ndarray  # design matrix, carb columns zero until filled by fit
    Xc: np.ndarray  # carbs absorbed per entry block and (meal group, absorption basis) (n x K x G*B)
    y: np.ndarray
    start: np.ndarray  # step index of window start
    stride: int

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return Windows(X=self.X[idx], Xc=self.Xc[idx], y=self.y[idx], start=self.start[idx],
                       stride=self.stride)

    def design(self, weights):
        k = self.Xc.shape[1]
        X = self.X.copy()
        X[:, 2 * k:3 * k] = self.Xc @ weights
        return X


def build_windows(steps: Steps, length_min=30, stride_min=30, day_filter=None) -> Windows:
    L, s = length_min // STEP_MIN, stride_min // STEP_MIN
    F, Fc = steps.features()
    C = np.vstack([np.zeros(F.shape[1]), np.cumsum(F, axis=0)])
    Cc = np.concatenate([np.zeros((1,) + Fc.shape[1:]), np.cumsum(Fc, axis=0)])
    bad = np.concatenate([[0], np.cumsum(~steps.valid)])
    starts = np.arange(0, len(steps.ia) - L + 1, s)
    ok = (bad[starts + L] - bad[starts]) == 0
    if day_filter is not None:
        ok &= np.isin(steps.day[starts], day_filter)
    starts = starts[ok]
    return Windows(X=C[starts + L] - C[starts], Xc=Cc[starts + L] - Cc[starts],
                   y=steps.bg[starts + L] - steps.bg[starts], start=starts, stride=s)
