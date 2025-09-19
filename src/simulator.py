from __future__ import annotations
import numpy as np

CORNER_KEYS = ["FL","FR","RL","RR"]

class TelemetrySim:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.lap_dist = 5_793  # example track length in meters
        self.pos = 0.0
        self.speed = 180.0
        self.ambient = 27.0
        self.track = 39.0
        self.lap_time_s = 83.0
        self.t = 0.0

    def sector_profile(self, s):
        # crude speed/lat-g/brake profile by normalized distance (0..1)
        V = 310 - 130*np.exp(-((s-0.15)/0.12)**2) - 120*np.exp(-((s-0.75)/0.08)**2)
        latg = 3.8*np.exp(-((s-0.32)/0.05)**2) + 3.2*np.exp(-((s-0.58)/0.06)**2)
        brake = 0.7*np.exp(-((s-0.14)/0.04)**2) + 0.8*np.exp(-((s-0.74)/0.03)**2)
        return max(100,V), latg, min(1.0, brake)

    def step(self, dt=0.2):
        self.t += dt
        s = (self.t % self.lap_time_s) / self.lap_time_s
        V, latg, brake = self.sector_profile(s)
        longg = 0.2 + 0.1*np.sin(2*np.pi*s)
        slip = 0.05 + 0.03*latg/4.5 + 0.02*self.rng.normal()
        slip_ang = 2.0 + 5.0*latg/4.5 + 0.5*self.rng.normal()
        base_load = 4000 + 1500*latg/4.5
        loads = {
            "FL": base_load*(1.0+0.18*latg/4.5),
            "FR": base_load*(1.0-0.16*latg/4.5),
            "RL": base_load*(0.95+0.12*longg),
            "RR": base_load*(0.95-0.12*longg),
        }
        u_common = dict(
            speed_kmh=V, Ta=self.ambient, Ttrack=self.track,
            slip=slip, slip_ang=slip_ang, brake=brake, latg=latg
        )
        # Sensor measurements (noisy): TPMS temp (proxy Tt) and hub temp (proxy Tr)
        sensors = {}
        for k in CORNER_KEYS:
            sensors[k] = {
                "tpms": 95 + 8*self.rng.normal(),
                "hub": 85 + 6*self.rng.normal(),
            }
        return u_common, loads, sensors
    
    def snapshot_controls(self, t: float):
        """
        Pure snapshot of controls at time t (seconds), without mutating internal state.
        No sensor noise.
        """
        s = (t % self.lap_time_s) / self.lap_time_s
        V, latg, brake = self.sector_profile(s)
        longg = 0.2 + 0.1*np.sin(2*np.pi*s)
        slip = 0.05 + 0.03*latg/4.5
        slip_ang = 2.0 + 5.0*latg/4.5
        base_load = 4000 + 1500*latg/4.5
        loads = {
            "FL": base_load*(1.0+0.18*latg/4.5),
            "FR": base_load*(1.0-0.16*latg/4.5),
            "RL": base_load*(0.95+0.12*longg),
            "RR": base_load*(0.95-0.12*longg),
        }
        u_common = dict(
            speed_kmh=max(100, V),
            Ta=self.ambient,
            Ttrack=self.track,
            slip=slip,
            slip_ang=slip_ang,
            brake=min(1.0, brake),
            latg=latg,
        )
        return u_common, loads

