from __future__ import annotations
import numpy as np
from typing import Optional

class ThermalParams:
    def __init__(self, a1=1.8e-3, a2=8.5e-4, a3=0.12, a4=0.08, a5=0.04,
                       b1=0.07, b2=0.09, c1=0.012, c2=0.03):
        self.a1=a1; self.a2=a2; self.a3=a3; self.a4=a4; self.a5=a5
        self.b1=b1; self.b2=b2; self.c1=c1; self.c2=c2

class ThermalModel:
    """3-node tire thermal model (tread Tt, carcass Tc, rim Tr) with wear and weather effects."""
    def __init__(self, params: ThermalParams, wear_model=None, weather_model=None):
        self.p = params
        self.wear_model = wear_model
        self.weather_model = weather_model

    def airflow(self, V_kmh: float) -> float:
        return max(0.0, 0.4*np.log1p(V_kmh))

    def step(self, x, u, dt, corner: str = "FL", compound: str = "medium"):
        """
        Thermal step with wear and weather effects.
        
        Args:
            x: Current thermal state [Tt, Tc, Tr]
            u: Control inputs
            dt: Time step
            corner: Tire corner ("FL", "FR", "RL", "RR")
            compound: Tire compound ("soft", "medium", "hard")
        """
        Tt, Tc, Tr = x
        slip = abs(u.get('slip',0.07))
        slip_ang = abs(u.get('slip_ang',3.0))*np.pi/180
        N = max(1.0, u.get('load', 4000.0))
        V = max(0.0, u.get('speed_kmh', 150.0))
        brake = max(0.0, u.get('brake', 0.2))
        
        # Get environmental temperatures (weather model takes precedence)
        if self.weather_model is not None:
            weather_summary = self.weather_model.get_weather_summary()
            Ta = weather_summary['ambient_temperature']
            Ttrack = weather_summary['track_temperature']
            weather_cooling_factor = weather_summary['cooling_factor']
            weather_thermal_factor = weather_summary['thermal_factor']
        else:
            Ta = u.get('Ta', 25.0)
            Ttrack = u.get('Ttrack', 35.0)
            weather_cooling_factor = 1.0
            weather_thermal_factor = 1.0
        
        # cooling_factor encodes wake/wet effects (1.0 = baseline)
        cooling_factor = float(u.get("cooling_factor", 1.0)) * weather_cooling_factor
        air = self.airflow(V) * cooling_factor

        # Update wear model if available
        if self.wear_model is not None:
            self.wear_model.update_wear(corner, x, N, slip, slip_ang * 180/np.pi, compound, dt)
            wear_effects = self.wear_model.get_wear_effects(corner)
            
            # Apply wear effects to thermal parameters
            thermal_conductivity_factor = wear_effects["thermal_conductivity_factor"]
            thermal_capacity_factor = wear_effects["thermal_capacity_factor"]
        else:
            thermal_conductivity_factor = 1.0
            thermal_capacity_factor = 1.0

        p = self.p
        
        # Modified thermal equations with wear and weather effects
        # Weather affects thermal generation and cooling
        thermal_generation_factor = weather_thermal_factor
        
        dTt = (thermal_generation_factor * (p.a1*N*slip + p.a2*V*slip_ang) + 
               p.a3*thermal_conductivity_factor*(Tc-Tt) + 
               p.a4*(Ttrack-Tt) - 
               p.a5*air*(Tt - Ta)) / thermal_capacity_factor
        
        dTc = (p.b1*thermal_conductivity_factor*(Tt-Tc) + 
               p.b2*thermal_conductivity_factor*(Tr-Tc)) / thermal_capacity_factor
        
        dTr = (thermal_generation_factor * p.c1*brake*N/6000.0 - p.c2*(Tr-Ta)) / thermal_capacity_factor
        
        return np.array([Tt + dt*dTt, Tc + dt*dTc, Tr + dt*dTr])
