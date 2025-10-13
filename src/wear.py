from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class WearParams:
    """Parameters for tire wear modeling based on thermal history and load cycles."""
    # Thermal wear coefficients
    thermal_wear_rate: float = 0.001  # Wear per °C above optimal per second
    thermal_threshold: float = 105.0   # Temperature threshold for accelerated wear
    
    # Load cycle wear coefficients  
    load_wear_base: float = 0.0001    # Base wear per load cycle
    load_wear_factor: float = 0.00005  # Additional wear per N of load
    
    # Slip wear coefficients
    slip_wear_rate: float = 0.0002     # Wear per unit slip
    slip_angle_wear_rate: float = 0.0001  # Wear per degree slip angle
    
    # Compound-specific wear rates
    compound_wear_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.compound_wear_multipliers is None:
            self.compound_wear_multipliers = {
                "soft": 1.5,    # Soft tires wear fastest
                "medium": 1.0,  # Baseline
                "hard": 0.7    # Hard tires wear slowest
            }

class TireWearModel:
    """
    Models tire wear based on thermal history, load cycles, and slip conditions.
    
    Wear affects:
    1. Thermal conductivity (worn tires conduct heat differently)
    2. Grip levels (worn tires have reduced grip)
    3. Thermal capacity (worn tires heat/cool differently)
    """
    
    def __init__(self, params: WearParams = None):
        self.params = params or WearParams()
        
        # Wear state per corner
        self.wear_levels = {corner: 0.0 for corner in ["FL", "FR", "RL", "RR"]}
        
        # Thermal history tracking (for wear calculation)
        self.thermal_history = {corner: [] for corner in ["FL", "FR", "RL", "RR"]}
        self.load_history = {corner: [] for corner in ["FL", "FR", "RL", "RR"]}
        self.slip_history = {corner: [] for corner in ["FL", "FR", "RL", "RR"]}
        
        # Maximum history length (keep last 100 samples)
        self.max_history = 100
        
    def update_wear(self, corner: str, thermal_state: np.ndarray, 
                   load: float, slip: float, slip_angle: float, 
                   compound: str, dt: float):
        """
        Update wear level for a specific corner based on current conditions.
        
        Args:
            corner: Tire corner ("FL", "FR", "RL", "RR")
            thermal_state: [Tt, Tc, Tr] temperatures
            load: Vertical load in N
            slip: Longitudinal slip ratio
            slip_angle: Lateral slip angle in degrees
            compound: Tire compound ("soft", "medium", "hard")
            dt: Time step in seconds
        """
        Tt, Tc, Tr = thermal_state
        
        # Store history
        self.thermal_history[corner].append(thermal_state.copy())
        self.load_history[corner].append(load)
        self.slip_history[corner].append((slip, slip_angle))
        
        # Trim history to max length
        if len(self.thermal_history[corner]) > self.max_history:
            self.thermal_history[corner] = self.thermal_history[corner][-self.max_history:]
            self.load_history[corner] = self.load_history[corner][-self.max_history:]
            self.slip_history[corner] = self.slip_history[corner][-self.max_history:]
        
        # Calculate wear increment
        wear_increment = self._calculate_wear_increment(
            thermal_state, load, slip, slip_angle, compound, dt
        )
        
        # Update wear level (clamped between 0 and 1)
        self.wear_levels[corner] = max(0.0, min(1.0, self.wear_levels[corner] + wear_increment))
    
    def _calculate_wear_increment(self, thermal_state: np.ndarray, 
                                load: float, slip: float, slip_angle: float,
                                compound: str, dt: float) -> float:
        """Calculate wear increment for current conditions."""
        Tt, Tc, Tr = thermal_state
        p = self.params
        
        # Get compound-specific wear multiplier
        compound_mult = p.compound_wear_multipliers.get(compound, 1.0)
        
        # Thermal wear (accelerated above threshold)
        thermal_wear = 0.0
        if Tt > p.thermal_threshold:
            thermal_wear = p.thermal_wear_rate * (Tt - p.thermal_threshold) * dt
        
        # Load cycle wear
        load_wear = (p.load_wear_base + p.load_wear_factor * load) * dt
        
        # Slip wear
        slip_wear = p.slip_wear_rate * abs(slip) * dt
        slip_angle_wear = p.slip_angle_wear_rate * abs(slip_angle) * dt
        
        # Total wear increment
        total_wear = (thermal_wear + load_wear + slip_wear + slip_angle_wear) * compound_mult
        
        return total_wear
    
    def get_wear_effects(self, corner: str) -> Dict[str, float]:
        """
        Get the effects of wear on tire properties.
        
        Returns:
            Dictionary with wear effects on thermal and grip properties
        """
        wear = max(0.0, min(1.0, self.wear_levels[corner]))  # Clamp wear level
        
        # Thermal effects (worn tires conduct heat differently)
        thermal_conductivity_factor = 1.0 + 0.3 * wear  # Worn tires conduct heat better
        thermal_capacity_factor = 1.0 - 0.2 * wear     # Worn tires have less thermal mass
        
        # Grip effects (worn tires have reduced grip)
        grip_factor = 1.0 - 0.4 * wear  # Up to 40% grip loss at full wear
        
        # Stiffness effects (worn tires are less stiff)
        stiffness_factor = 1.0 - 0.25 * wear  # Up to 25% stiffness loss
        
        return {
            "thermal_conductivity_factor": thermal_conductivity_factor,
            "thermal_capacity_factor": thermal_capacity_factor,
            "grip_factor": grip_factor,
            "stiffness_factor": stiffness_factor,
            "wear_level": wear
        }
    
    def get_grip_degradation(self, corner: str, base_grip: float = 1.0) -> float:
        """Get current grip level accounting for wear degradation."""
        effects = self.get_wear_effects(corner)
        return base_grip * effects["grip_factor"]
    
    def reset_wear(self, corner: str = None):
        """Reset wear levels (e.g., after pit stop with new tires)."""
        if corner is None:
            # Reset all corners
            for c in self.wear_levels:
                self.wear_levels[c] = 0.0
                self.thermal_history[c] = []
                self.load_history[c] = []
                self.slip_history[c] = []
        else:
            # Reset specific corner
            self.wear_levels[corner] = 0.0
            self.thermal_history[corner] = []
            self.load_history[corner] = []
            self.slip_history[corner] = []
    
    def get_wear_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive wear summary for all corners."""
        summary = {}
        for corner in self.wear_levels:
            effects = self.get_wear_effects(corner)
            summary[corner] = {
                "wear_level": self.wear_levels[corner],
                "grip_factor": effects["grip_factor"],
                "thermal_conductivity_factor": effects["thermal_conductivity_factor"],
                "thermal_capacity_factor": effects["thermal_capacity_factor"],
                "stiffness_factor": effects["stiffness_factor"]
            }
        return summary
    
    def predict_wear_remaining(self, corner: str, current_lap: int, 
                             total_laps: int) -> float:
        """
        Predict remaining tire life based on current wear rate.
        
        Returns:
            Estimated laps remaining before critical wear (0.8 wear level)
        """
        if len(self.thermal_history[corner]) < 10:
            return total_laps - current_lap  # Not enough data
        
        # Calculate recent wear rate (last 10 samples)
        recent_wear = self.wear_levels[corner]
        if recent_wear < 0.01:
            return total_laps - current_lap  # Minimal wear so far
        
        # Estimate wear rate per lap
        wear_per_lap = recent_wear / max(1, current_lap)
        
        # Predict laps to critical wear (0.8)
        critical_wear = 0.8
        if wear_per_lap <= 0:
            return total_laps - current_lap
        
        laps_to_critical = (critical_wear - self.wear_levels[corner]) / wear_per_lap
        
        return max(0, min(laps_to_critical, total_laps - current_lap))
