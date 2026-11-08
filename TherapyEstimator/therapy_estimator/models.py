"""Insulin and carb absorption curves, matching LoopKit's implementations. Times are in minutes."""

import numpy as np

MGDL_PER_MMOL = 18.0156

# (action duration, peak activity, delay), same as LoopKit's ExponentialInsulinModelPreset.
INSULIN_PRESETS = {
    "rapid-adult": (360.0, 75.0, 10.0),
    "rapid-child": (360.0, 65.0, 10.0),
    "fiasp": (360.0, 55.0, 10.0),
    "lyumjev": (360.0, 55.0, 10.0),
}


class ExponentialInsulinModel:
    def __init__(self, action_duration=360.0, peak=75.0, delay=10.0):
        self.td = float(action_duration)
        self.tp = float(peak)
        self.delay = float(delay)
        self.tau = self.tp * (1 - self.tp / self.td) / (1 - 2 * self.tp / self.td)
        self.a = 2 * self.tau / self.td
        self.S = 1 / (1 - self.a + (1 + self.a) * np.exp(-self.td / self.tau))

    @property
    def effect_duration(self):
        return self.td + self.delay

    def percent_effect_remaining(self, t):
        t = np.asarray(t, dtype=float) - self.delay
        tc = np.clip(t, 0.0, self.td)
        tau, a, S, td = self.tau, self.a, self.S, self.td
        per = 1 - S * (1 - a) * ((tc**2 / (tau * td * (1 - a)) - tc / tau - 1) * np.exp(-tc / tau) + 1)
        return np.where(t <= 0, 1.0, np.where(t >= td, 0.0, per))

    def minute_kernel(self):
        """Fraction of a dose's total effect that occurs during minute m after delivery,
        for a dose delivered at the middle of minute 0."""
        m = np.arange(int(np.ceil(self.effect_duration)) + 2, dtype=float)
        return self.percent_effect_remaining(m - 0.5) - self.percent_effect_remaining(m + 0.5)


def piecewise_linear_absorbed(p):
    """Fraction of carbs absorbed at fraction p of the absorption time (LoopKit PiecewiseLinearAbsorption)."""
    rise, fall = 0.15, 0.5
    scale = 2.0 / (1.0 + fall - rise)
    p = np.asarray(p, dtype=float)
    return np.select(
        [p <= 0, p < rise, p < fall, p < 1],
        [
            0.0,
            0.5 * scale * p**2 / rise,
            scale * (p - 0.5 * rise),
            scale * (fall - 0.5 * rise + (p - fall) * (1 - 0.5 * (p - fall) / (1 - fall))),
        ],
        default=1.0,
    )
