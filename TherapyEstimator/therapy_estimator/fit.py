"""Robust regularized least squares for (ISF, EGP, CSF) per time of day block, drift offsets, and the
carb absorption shape."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .features import ABSORPTION_BASIS, build_windows

DEFAULT_SHAPE = np.array([1.0 if f == 1.0 else 0.0 for f in ABSORPTION_BASIS])


@dataclass
class FitResult:
    theta: np.ndarray
    cov: np.ndarray  # conditional on the absorption shape
    shape: np.ndarray  # weights per ABSORPTION_BASIS entry, sum 1
    sigma: float  # robust residual scale, mg/dL
    rho: float  # lag 1 autocorrelation of residuals between adjacent windows
    n: int


def theta_from_profile(profile, block_hours, n_segments=0):
    """Block averages of (ISF, ISF * basal, ISF / CR) for a profile, with zero drift."""
    k = len(block_hours)
    edges = list(block_hours) + [24]
    isf, egp, csf = np.empty(k), np.empty(k), np.empty(k)
    for b in range(k):
        sec = np.arange(edges[b] * 3600, edges[b + 1] * 3600, 60)
        i, br, cr = profile.isf.at(sec), profile.basal.at(sec), profile.cr.at(sec)
        isf[b], egp[b], csf[b] = i.mean(), (i * br).mean(), (i / cr).mean()
    return np.concatenate([isf, egp, csf, np.zeros(n_segments)])


def _lag1(r, windows):
    adj = np.diff(windows.start) == windows.stride
    if adj.sum() < 10:
        return 0.0
    return float(np.corrcoef(r[:-1][adj], r[1:][adj])[0, 1])


def _robust_scale(r):
    return max(1.4826 * np.median(np.abs(r)), 1.0)


def _prior_precision(theta0, k, prior_rel, smooth_rel, drift_rel):
    p = len(theta0)
    prior_sd = np.concatenate([prior_rel * np.abs(theta0[:3 * k]),
                               np.full(p - 3 * k, drift_rel * theta0[k:2 * k].mean())])
    P = np.diag(1.0 / prior_sd**2)
    D = np.eye(k) - np.roll(np.eye(k), 1, axis=1)
    for j in range(3):
        sl = slice(j * k, (j + 1) * k)
        P[sl, sl] += D.T @ D / (smooth_rel * theta0[sl].mean()) ** 2
    return P, prior_sd


def _fit_theta(X, y, windows, theta0, P, prior_sd, huber_k, theta, n_iter=30):
    r = y - X @ theta
    sigma = _robust_scale(r)
    w = np.ones(len(y))
    f, rho = 1.0, 0.0
    for _ in range(n_iter):
        Xw = X * (w / sigma**2)[:, None]
        A = f * (X.T @ Xw) + P
        new = np.linalg.solve(A, f * (Xw.T @ y) + P @ theta0)
        r = y - X @ new
        sigma = _robust_scale(r)
        w = np.minimum(1.0, huber_k * sigma / np.maximum(np.abs(r), 1e-9))
        rho = _lag1(r, windows)
        f = float(np.clip((1 - rho) / (1 + rho), 0.05, 1.0))
        done = np.max(np.abs(new - theta) / prior_sd) < 1e-6
        theta = new
        if done:
            break
    return theta, A, sigma, w, rho


def _fit_shape(windows, theta, k, w, sigma, shape_prior_sd):
    """Absorption weights given theta: non negative, summing to one, shrunk towards Loop's default curve."""
    X0 = windows.X.copy()
    X0[:, 2 * k:3 * k] = 0
    r0 = windows.y - X0 @ theta
    Z = np.einsum("nkb,k->nb", windows.Xc, theta[2 * k:3 * k])
    sw = np.sqrt(w) / sigma
    nb = Z.shape[1]
    big = 1e3 * np.sqrt(len(r0))
    lam = 1.0 / shape_prior_sd
    A = np.vstack([Z * sw[:, None], big * np.ones((1, nb)), lam * np.eye(nb)])
    b = np.concatenate([r0 * sw, [big], lam * DEFAULT_SHAPE])
    s, _ = nnls(A, b)
    return s / s.sum()


def fit(windows, theta0, n_blocks, prior_rel=0.5, smooth_rel=0.2, drift_rel=0.1, huber_k=1.345,
        learn_shape=True, shape_prior_sd=0.3, n_outer=20):
    """Prior: block parameters ~ N(theta0, (prior_rel * theta0)^2) with a penalty on differences between
    neighbouring blocks (circular over the day); drift offsets ~ N(0, (drift_rel * mean EGP)^2).
    Residual autocorrelation shrinks the effective amount of data so the uncertainty is not overconfident.
    Alternates between the linear parameters and the absorption shape."""
    k = n_blocks
    P, prior_sd = _prior_precision(theta0, k, prior_rel, smooth_rel, drift_rel)
    shape = DEFAULT_SHAPE.copy()
    theta = theta0.copy()
    for _ in range(n_outer if learn_shape else 1):
        theta, A, sigma, w, rho = _fit_theta(windows.design(shape), windows.y, windows, theta0, P, prior_sd,
                                             huber_k, theta)
        if not learn_shape:
            break
        new_shape = _fit_shape(windows, theta, k, w, sigma, shape_prior_sd)
        done = np.max(np.abs(new_shape - shape)) < 1e-4
        shape = new_shape
        if done:
            break
    theta, A, sigma, w, rho = _fit_theta(windows.design(shape), windows.y, windows, theta0, P, prior_sd,
                                         huber_k, theta)
    return FitResult(theta=theta, cov=np.linalg.inv(A), shape=shape, sigma=sigma, rho=rho, n=len(windows))


def settings_point(theta, n_blocks):
    k = n_blocks
    isf, egp, csf = theta[:k], theta[k:2 * k], theta[2 * k:3 * k]
    return {"isf": isf, "basal": egp / isf, "cr": isf / csf}


def bootstrap_settings(windows, day, theta0, n_blocks, fit_kw, n_boot=40, seed=0):
    """Settings per block from refits on days resampled with replacement, (n_boot x K) per setting.
    Captures day to day variation and some model error, unlike the conditional covariance."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    out = {"isf": [], "basal": [], "cr": []}
    for _ in range(n_boot):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, size=len(days))])
        idx = idx[np.argsort(windows.start[idx], kind="stable")]
        res = fit(windows.subset(idx), theta0, n_blocks, **fit_kw)
        for name, v in settings_point(res.theta, n_blocks).items():
            out[name].append(v)
    return {name: np.array(v) for name, v in out.items()}


def prediction_errors(steps, models, horizon_min, day_filter, stride_min=30):
    """RMSE and MAE of the predicted glucose change over a horizon, using recorded insulin and carbs.
    `models` maps a name to (theta, absorption shape). Drift offsets are set to zero, since they are
    unknown for future days."""
    w = build_windows(steps, horizon_min, stride_min, day_filter)
    k3 = 3 * steps.n_blocks
    out = {"n": len(w.y)}
    preds = {name: w.design(shape)[:, :k3] @ th[:k3] for name, (th, shape) in models.items()}
    preds["ingen förändring"] = np.zeros(len(w.y))
    for name, pr in preds.items():
        e = w.y - pr
        out[name] = {"rmse": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                     "mae": float(np.mean(np.abs(e))) if len(e) else np.nan}
    return out
