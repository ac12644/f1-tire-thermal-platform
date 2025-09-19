from __future__ import annotations

def detect_lockup(prev_slip: float, slip: float, brake: float) -> bool:
    # lockup proxy: hard braking and sharp slip spike
    return (brake > 0.6) and (slip - prev_slip > 0.08)

def detect_slide(prev_sa: float, sa: float, latg: float) -> bool:
    # slide proxy: high lateral g and slip angle jump
    return (latg > 2.0) and (sa - prev_sa > 2.0)
