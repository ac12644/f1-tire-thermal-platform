from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class AeroMode(Enum):
    """Aerodynamic modes affecting tire cooling."""
    CLEAN_AIR = "clean_air"        # Clean air flow
    WAKE = "wake"                  # In wake of another car
    DRS_ACTIVE = "drs_active"      # DRS open (reduced downforce)
    DIRTY_AIR = "dirty_air"        # Dirty/turbulent air
    SLIPSTREAM = "slipstream"      # Beneficial slipstream

class CarPosition(Enum):
    """Car position relative to others."""
    LEADING = "leading"            # Leading car
    FOLLOWING = "following"        # Following car
    SIDE_BY_SIDE = "side_by_side"  # Side by side racing
    ISOLATED = "isolated"          # No other cars nearby

@dataclass
class AeroParams:
    """Parameters for aerodynamic modeling."""
    # Base aerodynamic properties
    base_downforce: float = 1.0        # Base downforce multiplier
    base_drag: float = 1.0             # Base drag multiplier
    base_cooling: float = 1.0          # Base cooling efficiency
    
    # Wake effects
    wake_cooling_reduction: float = 0.3  # Cooling reduction in wake
    wake_downforce_reduction: float = 0.2  # Downforce reduction in wake
    wake_distance_factor: float = 0.5   # Distance factor for wake effects
    
    # DRS effects
    drs_downforce_reduction: float = 0.4  # Downforce reduction with DRS
    drs_cooling_increase: float = 0.2    # Cooling increase with DRS
    drs_drag_reduction: float = 0.3      # Drag reduction with DRS
    
    # Slipstream effects
    slipstream_cooling_increase: float = 0.1  # Cooling increase in slipstream
    slipstream_downforce_reduction: float = 0.1  # Slight downforce reduction
    
    # Speed-dependent effects
    speed_cooling_factor: float = 0.01   # Cooling increase with speed
    speed_downforce_factor: float = 0.02  # Downforce increase with speed
    
    # Wind effects
    crosswind_cooling_factor: float = 0.05  # Crosswind cooling effect
    headwind_cooling_factor: float = 0.1    # Headwind cooling effect
    tailwind_cooling_factor: float = -0.05  # Tailwind cooling effect

class AerodynamicModel:
    """
    Advanced aerodynamic model for tire temperature management.
    
    Features:
    - Wake effects on tire cooling and downforce
    - DRS impact on aerodynamic balance
    - Slipstream effects on tire temperatures
    - Wind effects on cooling efficiency
    - Multi-car aerodynamic interactions
    - Speed-dependent aerodynamic effects
    """
    
    def __init__(self, params: AeroParams = None):
        self.p = params or AeroParams()
        
        # Aerodynamic state
        self.current_aero_mode = AeroMode.CLEAN_AIR
        self.car_position = CarPosition.ISOLATED
        self.drs_active = False
        
        # Environmental conditions
        self.wind_speed = 0.0
        self.wind_direction = 0.0  # Degrees from car direction
        self.ambient_pressure = 101325.0  # Pa
        
        # Car dynamics
        self.car_speed = 0.0  # m/s
        self.following_distance = 0.0  # m to car ahead
        self.side_distance = 0.0  # m to car beside
        
        # Aerodynamic effects
        self.downforce_multiplier = 1.0
        self.cooling_multiplier = 1.0
        self.drag_multiplier = 1.0
        
        # History for analysis
        self.aero_history = []
        self.cooling_history = []
        
    def update_aerodynamic_state(self, car_speed: float, following_distance: float = 0.0,
                               side_distance: float = 0.0, drs_active: bool = False,
                               wind_speed: float = 0.0, wind_direction: float = 0.0):
        """
        Update aerodynamic state based on car dynamics and environment.
        
        Args:
            car_speed: Car speed (m/s)
            following_distance: Distance to car ahead (m)
            side_distance: Distance to car beside (m)
            drs_active: Whether DRS is active
            wind_speed: Wind speed (m/s)
            wind_direction: Wind direction (degrees)
        """
        self.car_speed = car_speed
        self.following_distance = following_distance
        self.side_distance = side_distance
        self.drs_active = drs_active
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        
        # Determine aerodynamic mode
        self._determine_aero_mode()
        
        # Determine car position
        self._determine_car_position()
        
        # Calculate aerodynamic multipliers
        self._calculate_aerodynamic_multipliers()
        
        # Store history
        self._update_history()
    
    def _determine_aero_mode(self):
        """Determine current aerodynamic mode based on conditions."""
        if self.drs_active:
            self.current_aero_mode = AeroMode.DRS_ACTIVE
        elif self.following_distance > 0 and self.following_distance < 50:
            # Close following - wake effects
            if self.following_distance < 20:
                self.current_aero_mode = AeroMode.WAKE
            else:
                self.current_aero_mode = AeroMode.SLIPSTREAM
        elif self.side_distance > 0 and self.side_distance < 30:
            self.current_aero_mode = AeroMode.DIRTY_AIR
        else:
            self.current_aero_mode = AeroMode.CLEAN_AIR
    
    def _determine_car_position(self):
        """Determine car position relative to others."""
        if self.following_distance == 0 and self.side_distance == 0:
            self.car_position = CarPosition.ISOLATED
        elif self.following_distance > 0 and self.side_distance == 0:
            self.car_position = CarPosition.FOLLOWING
        elif self.following_distance == 0 and self.side_distance > 0:
            self.car_position = CarPosition.SIDE_BY_SIDE
        else:
            self.car_position = CarPosition.LEADING
    
    def _calculate_aerodynamic_multipliers(self):
        """Calculate aerodynamic multipliers based on current conditions."""
        # Start with base values
        downforce_mult = self.p.base_downforce
        cooling_mult = self.p.base_cooling
        drag_mult = self.p.base_drag
        
        # Speed effects
        speed_factor = self.car_speed / 100.0  # Normalize to 100 m/s
        downforce_mult *= (1.0 + speed_factor * self.p.speed_downforce_factor)
        cooling_mult *= (1.0 + speed_factor * self.p.speed_cooling_factor)
        
        # Aerodynamic mode effects
        if self.current_aero_mode == AeroMode.WAKE:
            downforce_mult *= (1.0 - self.p.wake_downforce_reduction)
            cooling_mult *= (1.0 - self.p.wake_cooling_reduction)
        elif self.current_aero_mode == AeroMode.DRS_ACTIVE:
            downforce_mult *= (1.0 - self.p.drs_downforce_reduction)
            cooling_mult *= (1.0 + self.p.drs_cooling_increase)
            drag_mult *= (1.0 - self.p.drs_drag_reduction)
        elif self.current_aero_mode == AeroMode.SLIPSTREAM:
            downforce_mult *= (1.0 - self.p.slipstream_downforce_reduction)
            cooling_mult *= (1.0 + self.p.slipstream_cooling_increase)
        elif self.current_aero_mode == AeroMode.DIRTY_AIR:
            cooling_mult *= (1.0 - 0.1)  # Slight cooling reduction
        
        # Wind effects
        wind_cooling_factor = self._calculate_wind_cooling_factor()
        cooling_mult *= wind_cooling_factor
        
        # Distance effects (for wake/slipstream)
        if self.following_distance > 0:
            distance_factor = self._calculate_distance_factor()
            if self.current_aero_mode == AeroMode.WAKE:
                cooling_mult *= (1.0 - distance_factor * self.p.wake_cooling_reduction)
            elif self.current_aero_mode == AeroMode.SLIPSTREAM:
                cooling_mult *= (1.0 + distance_factor * self.p.slipstream_cooling_increase)
        
        # Apply multipliers
        self.downforce_multiplier = max(0.3, min(2.0, downforce_mult))
        self.cooling_multiplier = max(0.2, min(3.0, cooling_mult))
        self.drag_multiplier = max(0.3, min(2.0, drag_mult))
    
    def _calculate_wind_cooling_factor(self) -> float:
        """Calculate wind effects on cooling."""
        if self.wind_speed == 0:
            return 1.0
        
        # Convert wind direction to relative angle
        relative_angle = abs(self.wind_direction) % 180
        
        if relative_angle < 30:  # Headwind
            return 1.0 + self.wind_speed * self.p.headwind_cooling_factor
        elif relative_angle > 150:  # Tailwind
            return 1.0 + self.wind_speed * self.p.tailwind_cooling_factor
        else:  # Crosswind
            return 1.0 + self.wind_speed * self.p.crosswind_cooling_factor
    
    def _calculate_distance_factor(self) -> float:
        """Calculate distance factor for aerodynamic effects."""
        if self.following_distance <= 0:
            return 0.0
        
        # Exponential decay with distance
        max_distance = 50.0  # Maximum effective distance
        distance_factor = np.exp(-self.following_distance / max_distance)
        
        return distance_factor
    
    def _update_history(self):
        """Update aerodynamic history."""
        self.aero_history.append({
            'mode': self.current_aero_mode.value,
            'position': self.car_position.value,
            'downforce_multiplier': self.downforce_multiplier,
            'cooling_multiplier': self.cooling_multiplier,
            'drag_multiplier': self.drag_multiplier,
            'car_speed': self.car_speed,
            'following_distance': self.following_distance
        })
        
        self.cooling_history.append(self.cooling_multiplier)
        
        # Keep only last 100 points
        if len(self.aero_history) > 100:
            self.aero_history = self.aero_history[-100:]
            self.cooling_history = self.cooling_history[-100:]
    
    def get_aerodynamic_effects(self) -> Dict[str, float]:
        """Get current aerodynamic effects."""
        return {
            'downforce_multiplier': self.downforce_multiplier,
            'cooling_multiplier': self.cooling_multiplier,
            'drag_multiplier': self.drag_multiplier,
            'aero_mode': self.current_aero_mode.value,
            'car_position': self.car_position.value,
            'drs_active': self.drs_active,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'car_speed': self.car_speed,
            'following_distance': self.following_distance,
            'side_distance': self.side_distance
        }
    
    def calculate_tire_cooling_effect(self, base_cooling: float, tire_position: str) -> float:
        """
        Calculate tire-specific cooling effect based on aerodynamic conditions.
        
        Args:
            base_cooling: Base cooling rate
            tire_position: Tire position ("FL", "FR", "RL", "RR")
            
        Returns:
            Aerodynamic cooling effect
        """
        # Base aerodynamic cooling
        aero_cooling = base_cooling * self.cooling_multiplier
        
        # Position-specific effects
        if tire_position in ["FL", "FR"]:  # Front tires
            # Front tires are more affected by wake
            if self.current_aero_mode == AeroMode.WAKE:
                aero_cooling *= 0.8  # Additional reduction for front tires
            elif self.current_aero_mode == AeroMode.DRS_ACTIVE:
                aero_cooling *= 1.1  # Additional cooling with DRS
        
        elif tire_position in ["RL", "RR"]:  # Rear tires
            # Rear tires are more affected by dirty air
            if self.current_aero_mode == AeroMode.DIRTY_AIR:
                aero_cooling *= 0.9  # Additional reduction for rear tires
        
        # Wind effects on specific tire positions
        if self.wind_speed > 0:
            wind_factor = self._calculate_tire_wind_factor(tire_position)
            aero_cooling *= wind_factor
        
        return aero_cooling
    
    def _calculate_tire_wind_factor(self, tire_position: str) -> float:
        """Calculate wind factor for specific tire position."""
        # Simplified wind effect based on tire position
        if tire_position == "FL":
            return 1.0 + self.wind_speed * 0.02  # Left front gets more cooling
        elif tire_position == "FR":
            return 1.0 - self.wind_speed * 0.01  # Right front gets less cooling
        elif tire_position == "RL":
            return 1.0 + self.wind_speed * 0.01  # Left rear gets slight cooling
        elif tire_position == "RR":
            return 1.0 - self.wind_speed * 0.02  # Right rear gets less cooling
        
        return 1.0
    
    def get_aerodynamic_recommendations(self, thermal_state: np.ndarray) -> List[Tuple[str, str]]:
        """
        Get aerodynamic-based recommendations.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            
        Returns:
            List of aerodynamic recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Wake-based recommendations
        if self.current_aero_mode == AeroMode.WAKE:
            if Tt > 100:
                recommendations.append(("AERO", f"In wake: reduced cooling - consider overtaking or increasing gap"))
            else:
                recommendations.append(("AERO", f"In wake: monitor tire temps closely, reduced cooling"))
        
        # DRS recommendations
        if self.drs_active:
            if Tt > 105:
                recommendations.append(("AERO", f"DRS active: increased cooling - good for high temps"))
            else:
                recommendations.append(("AERO", f"DRS active: monitor temps, increased cooling may overcool"))
        
        # Slipstream recommendations
        if self.current_aero_mode == AeroMode.SLIPSTREAM:
            if Tt < 90:
                recommendations.append(("AERO", f"Slipstream: increased cooling - may overcool tires"))
            else:
                recommendations.append(("AERO", f"Slipstream: beneficial cooling - maintain position"))
        
        # Wind recommendations
        if self.wind_speed > 15:
            if abs(self.wind_direction) < 30:  # Headwind
                recommendations.append(("AERO", f"Headwind {self.wind_speed:.0f}km/h: increased cooling"))
            elif abs(self.wind_direction) > 150:  # Tailwind
                recommendations.append(("AERO", f"Tailwind {self.wind_speed:.0f}km/h: reduced cooling"))
            else:  # Crosswind
                recommendations.append(("AERO", f"Crosswind {self.wind_speed:.0f}km/h: asymmetric cooling"))
        
        # Position-based recommendations
        if self.car_position == CarPosition.FOLLOWING and self.following_distance < 20:
            recommendations.append(("AERO", f"Close following {self.following_distance:.0f}m: reduced cooling"))
        elif self.car_position == CarPosition.LEADING:
            recommendations.append(("AERO", f"Leading car: clean air cooling"))
        
        return recommendations
    
    def get_aerodynamic_summary(self) -> Dict[str, float]:
        """Get comprehensive aerodynamic summary."""
        return {
            'current_mode': self.current_aero_mode.value,
            'car_position': self.car_position.value,
            'downforce_multiplier': self.downforce_multiplier,
            'cooling_multiplier': self.cooling_multiplier,
            'drag_multiplier': self.drag_multiplier,
            'drs_active': self.drs_active,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'car_speed': self.car_speed,
            'following_distance': self.following_distance,
            'side_distance': self.side_distance,
            'average_cooling': np.mean(self.cooling_history) if self.cooling_history else 1.0
        }
    
    def reset_aerodynamic_state(self):
        """Reset aerodynamic state for new session."""
        self.current_aero_mode = AeroMode.CLEAN_AIR
        self.car_position = CarPosition.ISOLATED
        self.drs_active = False
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.car_speed = 0.0
        self.following_distance = 0.0
        self.side_distance = 0.0
        self.downforce_multiplier = 1.0
        self.cooling_multiplier = 1.0
        self.drag_multiplier = 1.0
        self.aero_history = []
        self.cooling_history = []
