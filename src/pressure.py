from __future__ import annotations
import numpy as np

def suggest_pressure_delta(carcass_t: np.ndarray, target_band: tuple[float,float]) -> float:
    """
    Heuristic: carcass ΔT per 1 psi ≈ -6°C (raising pressure reduces carcass flex/heat).
    Clamp to ±0.6 psi. Return delta psi to move towards center of band.
    """
    if carcass_t.size == 0 or np.any(~np.isfinite(carcass_t)):
        return 0.0
    lo, hi = target_band
    target = 0.5*(lo + hi)
    avg = float(np.mean(carcass_t))
    k = -6.0  # °C per psi
    raw = (target - avg) / k  # psi
    return float(max(-0.6, min(0.6, raw)))
