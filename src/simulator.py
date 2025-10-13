from __future__ import annotations
import numpy as np
from typing import Optional
try:
    from .config import system_config
except ImportError:
    from config import system_config

CORNER_KEYS = ["FL","FR","RL","RR"]

class TelemetrySim:
    def __init__(self, seed=7, wear_model=None, weather_model=None):
        self.rng = np.random.default_rng(seed)
        track_config = system_config.track_config
        self.lap_dist = track_config.length_meters
        self.pos = 0.0
        self.speed = track_config.typical_speed_kmh
        self.ambient = track_config.typical_ambient_temp
        self.track = track_config.typical_track_temp
        self.lap_time_s = track_config.typical_lap_time
        self.t = 0.0
        self.wear_model = wear_model
        self.weather_model = weather_model
        self.current_compound = "medium"
        self.lap_count = 0

    def sector_profile(self, s):
        # crude speed/lat-g/brake profile by normalized distance (0..1)
        V = 310 - 130*np.exp(-((s-0.15)/0.12)**2) - 120*np.exp(-((s-0.75)/0.08)**2)
        latg = 3.8*np.exp(-((s-0.32)/0.05)**2) + 3.2*np.exp(-((s-0.58)/0.06)**2)
        brake = 0.7*np.exp(-((s-0.14)/0.04)**2) + 0.8*np.exp(-((s-0.74)/0.03)**2)
        return max(100,V), latg, min(1.0, brake)

    def step(self, dt=0.2):
        self.t += dt
        
        # Update weather model if available
        if self.weather_model is not None:
            self.weather_model.update_weather(dt, self.lap_count)
            weather_summary = self.weather_model.get_weather_summary()
            # Update ambient and track temperatures from weather
            self.ambient = weather_summary['ambient_temperature']
            self.track = weather_summary['track_temperature']
        
        # Track lap count for wear modeling
        if hasattr(self, '_last_lap_time'):
            if self.t - self._last_lap_time >= self.lap_time_s:
                self.lap_count += 1
        self._last_lap_time = self.t
        
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
        
        # Apply weather effects to loads (wet conditions reduce grip)
        if self.weather_model is not None:
            weather_summary = self.weather_model.get_weather_summary()
            grip_factor = weather_summary['grip_factor']
            # Reduce loads in wet conditions (less grip = less load transfer)
            for corner in loads:
                loads[corner] *= grip_factor
        
        u_common = dict(
            speed_kmh=V, Ta=self.ambient, Ttrack=self.track,
            slip=slip, slip_ang=slip_ang, brake=brake, latg=latg
        )
        
        # Sensor measurements (noisy): TPMS temp (proxy Tt) and hub temp (proxy Tr)
        # Apply wear and weather effects to sensor readings
        sensors = {}
        for k in CORNER_KEYS:
            base_tpms = 95 + 8*self.rng.normal()
            base_hub = 85 + 6*self.rng.normal()
            
            # Apply wear effects to sensor readings
            if self.wear_model is not None:
                wear_effects = self.wear_model.get_wear_effects(k)
                # Worn tires may show different thermal signatures
                wear_factor = 1.0 + 0.1 * wear_effects["wear_level"]  # Slight thermal signature change
                base_tpms *= wear_factor
                base_hub *= wear_factor
            
            # Apply weather effects to sensor readings
            if self.weather_model is not None:
                weather_summary = self.weather_model.get_weather_summary()
                thermal_factor = weather_summary['thermal_factor']
                # Wet conditions affect thermal signatures
                base_tpms *= thermal_factor
                base_hub *= thermal_factor
            
            sensors[k] = {
                "tpms": base_tpms,
                "hub": base_hub,
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
    
    def pit_stop(self, new_compound: str):
        """Simulate pit stop - reset wear and change compound."""
        self.current_compound = new_compound
        if self.wear_model is not None:
            self.wear_model.reset_wear()  # Reset all corners
        self.lap_count = 0
    
    def get_wear_summary(self):
        """Get current wear status for all corners."""
        if self.wear_model is not None:
            return self.wear_model.get_wear_summary()
        return {corner: {"wear_level": 0.0, "grip_factor": 1.0} for corner in CORNER_KEYS}
    
    def get_weather_summary(self):
        """Get current weather status."""
        if self.weather_model is not None:
            return self.weather_model.get_weather_summary()
        return {
            'current_condition': 'dry',
            'rain_probability': 0.0,
            'track_temperature': self.track,
            'ambient_temperature': self.ambient,
            'cooling_factor': 1.0,
            'grip_factor': 1.0,
            'thermal_factor': 1.0
        }
    
    def set_session_type(self, session_type: str):
        """Set session type (affects weather evolution)."""
        if self.weather_model is not None:
            from weather import SessionType
            session_enum = SessionType(session_type.lower())
            self.weather_model.set_session_type(session_enum)

