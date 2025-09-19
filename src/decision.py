from __future__ import annotations
import numpy as np

OPT_BANDS = {
    "soft": (95.0, 110.0),
    "medium": (90.0, 106.0),
    "hard": (88.0, 104.0)
}

class DecisionEngine:
    def __init__(self, compound="medium"):
        self.band = OPT_BANDS.get(compound, OPT_BANDS["medium"])

    def forecast_naive(self, hist, steps=10):
        # simple exponential smoothing trend per state dim
        if len(hist) < 5:
            return np.tile(hist[-1], (steps,1))
        hist = np.array(hist[-30:])
        alpha = 0.35
        level = hist[0]
        for x in hist[1:]:
            level = alpha*x + (1-alpha)*level
        return np.vstack([level for _ in range(steps)])

    def actions(self, est_by_corner):
        lo, hi = self.band
        recs = []
        for corner, x in est_by_corner.items():
            Tt, Tc, Tr = x
            if Tt > hi:
                recs.append((corner, f"Tread {Tt:.1f}C > {hi}C: brake bias +1, gentler entry T7-T9"))
            elif Tt < lo:
                recs.append((corner, f"Tread {Tt:.1f}C < {lo}C: diff entry -1, short throttle bursts, push on exits"))
            if Tc > hi+2:
                recs.append((corner, f"Carcass {Tc:.1f}C high: consider +0.2 psi next stop"))
        return recs
