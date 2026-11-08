"""Insulin effect per pod day, relative to the first day of a pod."""

from dataclasses import replace

import numpy as np

from .features import build_steps, build_windows
from .fit import fit, theta_from_profile

POD_DAYS = ("dag 1", "dag 2", "dag 3", "okänd")


def _pod_windows(steps, windows, length_steps):
    """Window sums of the insulin columns split by pod day (n x K x 4), with the same sign as X."""
    n, k = len(steps.ia), steps.n_blocks
    f = np.zeros((n, k, 4))
    f[np.arange(n), steps.block, :] = -steps.ia_pod
    c = np.concatenate([np.zeros((1, k, 4)), np.cumsum(f, axis=0)])
    return c[windows.start + length_steps] - c[windows.start]


def _fit(windows, Xp, theta0, k, fit_kw, n_outer=8, ridge=1.0):
    m = np.ones(4)
    for _ in range(n_outer):
        X = windows.X.copy()
        X[:, :k] = Xp @ m
        w = replace(windows, X=X)
        res = fit(w, theta0, k, **fit_kw)
        Xo = w.design(res.shape)
        Xo[:, :k] = 0
        r0 = w.y - Xo @ res.theta
        Z = np.einsum("nkd,k->nd", Xp, res.theta[:k])
        target, A = r0 - Z[:, 0], Z[:, 1:]
        weights = np.ones(len(target))
        for _ in range(10):
            sw = np.sqrt(weights) / res.sigma
            lhs = np.vstack([A * sw[:, None], np.sqrt(ridge) * np.eye(3)])
            rhs = np.concatenate([target * sw, np.sqrt(ridge) * np.ones(3)])
            new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            r = target - A @ new
            weights = np.minimum(1.0, 1.345 * res.sigma / np.maximum(np.abs(r), 1e-9))
        done = np.max(np.abs(new - m[1:])) < 1e-3
        m[1:] = new
        if done:
            break
    return m, res


def pod_age_effect(ds, cfg, fit_kw=None, n_boot=20, window_min=30, seed=0):
    """Returns the insulin effect on pod day 2, day 3 and unknown relative to day 1, with 80 % intervals
    from refits on resampled days, and the share of insulin delivered per pod day."""
    fit_kw = fit_kw or {}
    steps = build_steps(ds, cfg)
    windows = build_windows(steps, window_min, window_min)
    L = window_min // 5
    Xp = _pod_windows(steps, windows, L)
    theta0 = theta_from_profile(ds.profiles[-1], cfg.block_hours, steps.n_segments)
    m, _ = _fit(windows, Xp, theta0, cfg.n_blocks, fit_kw)
    rng = np.random.default_rng(seed)
    day = steps.day[windows.start]
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, size=len(days))])
        idx = idx[np.argsort(windows.start[idx], kind="stable")]
        boot.append(_fit(windows.subset(idx), Xp[idx], theta0, cfg.n_blocks, fit_kw)[0])
    boot = np.array(boot)
    share = steps.ia_pod.sum(0) / steps.ia_pod.sum()
    return {"factor": m, "p10": np.percentile(boot, 10, axis=0), "p90": np.percentile(boot, 90, axis=0),
            "insulin_share": share}
