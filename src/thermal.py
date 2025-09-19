from __future__ import annotations
import numpy as np

class ThermalParams:
    def __init__(self, a1=1.8e-3, a2=8.5e-4, a3=0.12, a4=0.08, a5=0.04,
                       b1=0.07, b2=0.09, c1=0.012, c2=0.03):
        self.a1=a1; self.a2=a2; self.a3=a3; self.a4=a4; self.a5=a5
        self.b1=b1; self.b2=b2; self.c1=c1; self.c2=c2

class ThermalModel:
    """3-node tire thermal model (tread Tt, carcass Tc, rim Tr)."""
    def __init__(self, params: ThermalParams):
        self.p = params

    def airflow(self, V_kmh: float) -> float:
        return max(0.0, 0.4*np.log1p(V_kmh))

    def step(self, x, u, dt):
        Tt, Tc, Tr = x
        slip = abs(u.get('slip',0.07))
        slip_ang = abs(u.get('slip_ang',3.0))*np.pi/180
        N = max(1.0, u.get('load', 4000.0))
        V = max(0.0, u.get('speed_kmh', 150.0))
        brake = max(0.0, u.get('brake', 0.2))
        Ta = u.get('Ta', 25.0)
        Ttrack = u.get('Ttrack', 35.0)
        # cooling_factor encodes wake/wet effects (1.0 = baseline)
        cooling_factor = float(u.get("cooling_factor", 1.0))
        air = self.airflow(V) * cooling_factor

        p = self.p
        dTt = (p.a1*N*slip + p.a2*V*slip_ang + p.a3*(Tc-Tt) + p.a4*(Ttrack-Tt) - p.a5*air*(Tt - Ta))
        dTc = (p.b1*(Tt-Tc) + p.b2*(Tr-Tc))
        dTr = (p.c1*brake*N/6000.0 - p.c2*(Tr-Ta))
        return np.array([Tt + dt*dTt, Tc + dt*dTc, Tr + dt*dTr])
