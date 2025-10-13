from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class TireConstruction(Enum):
    """Tire construction types affecting structural behavior."""
    RADIAL = "radial"          # Standard radial construction
    BIAS_PLY = "bias_ply"      # Bias-ply construction (stiffer)
    CARBON_BELTED = "carbon_belted"  # Carbon fiber belted (high performance)

class ContactPatchShape(Enum):
    """Contact patch shape classifications."""
    ELLIPTICAL = "elliptical"  # Standard elliptical contact patch
    RECTANGULAR = "rectangular"  # Rectangular contact patch (high load)
    IRREGULAR = "irregular"    # Irregular shape (wear/deflection effects)

@dataclass
class StructuralParams:
    """Parameters for structural tire modeling."""
    # Tire dimensions
    tire_width: float = 245.0          # Tire width (mm)
    aspect_ratio: float = 0.4          # Aspect ratio (sidewall height / width)
    rim_diameter: float = 18.0          # Rim diameter (inches)
    
    # Structural properties
    sidewall_stiffness: float = 1.0     # Sidewall stiffness multiplier
    tread_stiffness: float = 1.0        # Tread stiffness multiplier
    carcass_stiffness: float = 1.0     # Carcass stiffness multiplier
    
    # Contact patch properties
    max_deflection: float = 0.15        # Maximum deflection ratio (15% of radius)
    contact_pressure_max: float = 1.5   # Maximum contact pressure (MPa)
    contact_patch_length: float = 0.2   # Contact patch length ratio
    
    # Dynamic properties
    damping_ratio: float = 0.1         # Structural damping ratio
    natural_frequency: float = 50.0     # Natural frequency (Hz)
    
    # Wear effects on structure
    wear_stiffness_reduction: float = 0.3  # Stiffness reduction at full wear
    wear_deflection_increase: float = 0.2  # Deflection increase at full wear

class StructuralTireModel:
    """
    Advanced structural tire model with deflection, contact patch dynamics, and pressure distribution.
    
    Features:
    - Tire deflection modeling based on load and inflation pressure
    - Contact patch shape and pressure distribution
    - Structural stiffness affected by temperature and wear
    - Dynamic response to load changes
    - Multi-physics coupling with thermal and aerodynamic models
    """
    
    def __init__(self, params: StructuralParams = None, construction: TireConstruction = TireConstruction.RADIAL):
        self.p = params or StructuralParams()
        self.construction = construction
        
        # Calculate derived parameters
        self.sidewall_height = self.p.tire_width * self.p.aspect_ratio
        self.tire_radius = self._calculate_tire_radius()
        
        # Structural state
        self.current_deflection = 0.0
        self.contact_patch_area = 0.0
        self.contact_patch_shape = ContactPatchShape.ELLIPTICAL
        
        # Pressure distribution
        self.pressure_distribution = np.zeros((10, 10))  # 10x10 grid
        
        # Dynamic state
        self.deflection_velocity = 0.0
        self.last_load = 0.0
        
        # History for analysis
        self.deflection_history = []
        self.pressure_history = []
        
    def _calculate_tire_radius(self) -> float:
        """Calculate tire radius from dimensions."""
        # Convert rim diameter to mm
        rim_radius_mm = (self.p.rim_diameter * 25.4) / 2.0
        # Add sidewall height
        tire_radius = rim_radius_mm + self.sidewall_height
        return tire_radius / 1000.0  # Convert to meters
    
    def calculate_deflection(self, vertical_load: float, inflation_pressure: float, 
                           temperature: float, wear_level: float) -> float:
        """
        Calculate tire deflection based on load, pressure, temperature, and wear.
        
        Args:
            vertical_load: Vertical load on tire (N)
            inflation_pressure: Tire inflation pressure (kPa)
            temperature: Tire temperature (°C)
            wear_level: Tire wear level (0-1)
            
        Returns:
            Deflection ratio (deflection / tire_radius)
        """
        # Base deflection from load and pressure
        # Simplified tire deflection model: deflection ∝ load / pressure
        base_deflection = (vertical_load / (inflation_pressure * 1000.0)) * 0.001
        
        # Temperature effects on stiffness
        temp_factor = self._calculate_temperature_stiffness_factor(temperature)
        
        # Wear effects on stiffness
        wear_factor = self._calculate_wear_stiffness_factor(wear_level)
        
        # Construction effects
        construction_factor = self._calculate_construction_factor()
        
        # Calculate effective deflection
        effective_deflection = base_deflection / (temp_factor * wear_factor * construction_factor)
        
        # Apply maximum deflection limit
        effective_deflection = min(effective_deflection, self.p.max_deflection)
        
        return effective_deflection
    
    def _calculate_temperature_stiffness_factor(self, temperature: float) -> float:
        """Calculate stiffness factor based on temperature."""
        # Tire stiffness generally decreases with temperature
        # Optimal temperature around 80-90°C
        optimal_temp = 85.0
        
        if temperature < optimal_temp:
            # Cold tires are stiffer
            temp_factor = 1.0 + (optimal_temp - temperature) / 100.0
        else:
            # Hot tires are softer
            temp_factor = 1.0 - (temperature - optimal_temp) / 200.0
        
        return max(0.5, min(2.0, temp_factor))
    
    def _calculate_wear_stiffness_factor(self, wear_level: float) -> float:
        """Calculate stiffness factor based on wear."""
        # Worn tires are less stiff
        stiffness_reduction = wear_level * self.p.wear_stiffness_reduction
        return 1.0 - stiffness_reduction
    
    def _calculate_construction_factor(self) -> float:
        """Calculate stiffness factor based on tire construction."""
        construction_factors = {
            TireConstruction.RADIAL: 1.0,
            TireConstruction.BIAS_PLY: 1.3,  # Stiffer
            TireConstruction.CARBON_BELTED: 1.1  # Slightly stiffer
        }
        return construction_factors.get(self.construction, 1.0)
    
    def calculate_contact_patch(self, deflection: float, vertical_load: float, 
                              slip_angle: float) -> Dict[str, float]:
        """
        Calculate contact patch properties based on deflection and load.
        
        Args:
            deflection: Tire deflection ratio
            vertical_load: Vertical load (N)
            slip_angle: Slip angle (radians)
            
        Returns:
            Dict with contact patch properties
        """
        # Contact patch length (simplified model)
        patch_length = deflection * self.tire_radius * 2.0
        
        # Contact patch width (based on tire width and load)
        patch_width = self.p.tire_width / 1000.0 * (1.0 + deflection * 0.5)
        
        # Contact patch area
        patch_area = patch_length * patch_width * np.pi / 4.0  # Elliptical area
        
        # Determine contact patch shape
        if deflection > 0.1:
            self.contact_patch_shape = ContactPatchShape.RECTANGULAR
        elif abs(slip_angle) > 0.1:
            self.contact_patch_shape = ContactPatchShape.IRREGULAR
        else:
            self.contact_patch_shape = ContactPatchShape.ELLIPTICAL
        
        # Contact pressure (average)
        avg_pressure = vertical_load / patch_area if patch_area > 0 else 0.0
        
        return {
            'length': patch_length,
            'width': patch_width,
            'area': patch_area,
            'average_pressure': avg_pressure,
            'shape': self.contact_patch_shape.value
        }
    
    def calculate_pressure_distribution(self, contact_patch: Dict[str, float], 
                                      slip_angle: float, camber: float = 0.0) -> np.ndarray:
        """
        Calculate pressure distribution within contact patch.
        
        Args:
            contact_patch: Contact patch properties
            slip_angle: Slip angle (radians)
            camber: Camber angle (radians)
            
        Returns:
            2D array representing pressure distribution
        """
        length = contact_patch['length']
        width = contact_patch['width']
        avg_pressure = contact_patch['average_pressure']
        
        # Create pressure distribution grid
        pressure_grid = np.zeros((10, 10))
        
        # Base pressure distribution (elliptical)
        for i in range(10):
            for j in range(10):
                # Normalized coordinates (-1 to 1)
                x = (i - 4.5) / 4.5
                y = (j - 4.5) / 4.5
                
                # Elliptical pressure distribution
                ellipse_factor = 1.0 - (x**2 + y**2)
                if ellipse_factor > 0:
                    pressure_grid[i, j] = avg_pressure * ellipse_factor
        
        # Apply slip angle effects (pressure shift)
        if abs(slip_angle) > 0.01:
            shift_factor = min(abs(slip_angle) * 2.0, 1.0)
            # Shift pressure towards slip direction
            for i in range(10):
                for j in range(10):
                    shift = int(slip_angle * 3.0)  # Shift in grid units
                    if 0 <= i + shift < 10:
                        pressure_grid[i, j] *= (1.0 - shift_factor * 0.3)
                        if pressure_grid[i + shift, j] == 0:
                            pressure_grid[i + shift, j] = pressure_grid[i, j] * shift_factor * 0.3
        
        # Apply camber effects (pressure bias)
        if abs(camber) > 0.01:
            camber_factor = abs(camber) * 2.0
            for i in range(10):
                for j in range(10):
                    # Bias pressure towards camber direction
                    bias = camber_factor * (j - 4.5) / 4.5
                    pressure_grid[i, j] *= (1.0 + bias * 0.2)
        
        # Normalize to maintain total load
        total_pressure = np.sum(pressure_grid)
        if total_pressure > 0:
            pressure_grid = pressure_grid * (avg_pressure * 100) / total_pressure
        
        return pressure_grid
    
    def calculate_structural_forces(self, deflection: float, deflection_velocity: float,
                                   vertical_load: float) -> Dict[str, float]:
        """
        Calculate structural forces based on deflection and dynamics.
        
        Args:
            deflection: Current deflection ratio
            deflection_velocity: Rate of deflection change
            vertical_load: Vertical load (N)
            
        Returns:
            Dict with structural forces
        """
        # Spring force (proportional to deflection)
        spring_force = deflection * vertical_load * 10.0
        
        # Damping force (proportional to velocity)
        damping_force = deflection_velocity * vertical_load * self.p.damping_ratio
        
        # Total structural force
        total_force = spring_force + damping_force
        
        # Calculate natural frequency response
        frequency_response = self._calculate_frequency_response(deflection_velocity)
        
        return {
            'spring_force': spring_force,
            'damping_force': damping_force,
            'total_structural_force': total_force,
            'frequency_response': frequency_response
        }
    
    def _calculate_frequency_response(self, deflection_velocity: float) -> float:
        """Calculate frequency response factor."""
        # Simple frequency response model
        velocity_factor = abs(deflection_velocity) / 10.0  # Normalize velocity
        frequency_factor = 1.0 + velocity_factor * 0.1  # Slight increase with velocity
        
        return min(frequency_factor, 1.5)  # Cap at 1.5x
    
    def update_structural_state(self, vertical_load: float, inflation_pressure: float,
                              temperature: float, wear_level: float, slip_angle: float,
                              camber: float = 0.0, dt: float = 0.1):
        """
        Update structural state with new conditions.
        
        Args:
            vertical_load: Vertical load (N)
            inflation_pressure: Inflation pressure (kPa)
            temperature: Tire temperature (°C)
            wear_level: Tire wear level (0-1)
            slip_angle: Slip angle (radians)
            camber: Camber angle (radians)
            dt: Time step (s)
        """
        # Calculate new deflection
        new_deflection = self.calculate_deflection(vertical_load, inflation_pressure, temperature, wear_level)
        
        # Calculate deflection velocity
        self.deflection_velocity = (new_deflection - self.current_deflection) / dt
        
        # Update deflection
        self.current_deflection = new_deflection
        
        # Calculate contact patch
        contact_patch = self.calculate_contact_patch(new_deflection, vertical_load, slip_angle)
        self.contact_patch_area = contact_patch['area']
        
        # Calculate pressure distribution
        self.pressure_distribution = self.calculate_pressure_distribution(
            contact_patch, slip_angle, camber
        )
        
        # Calculate structural forces
        structural_forces = self.calculate_structural_forces(
            new_deflection, self.deflection_velocity, vertical_load
        )
        
        # Store history
        self.deflection_history.append(new_deflection)
        self.pressure_history.append(np.mean(self.pressure_distribution))
        
        # Keep only last 100 points
        if len(self.deflection_history) > 100:
            self.deflection_history = self.deflection_history[-100:]
            self.pressure_history = self.pressure_history[-100:]
        
        # Update last load
        self.last_load = vertical_load
    
    def get_structural_summary(self) -> Dict[str, float]:
        """Get comprehensive structural summary."""
        return {
            'deflection_ratio': self.current_deflection,
            'deflection_velocity': self.deflection_velocity,
            'contact_patch_area': self.contact_patch_area,
            'contact_patch_shape': self.contact_patch_shape.value,
            'average_pressure': np.mean(self.pressure_distribution),
            'max_pressure': np.max(self.pressure_distribution),
            'pressure_variance': np.var(self.pressure_distribution),
            'tire_radius': self.tire_radius,
            'construction': self.construction.value
        }
    
    def get_structural_recommendations(self, thermal_state: np.ndarray, 
                                     wear_level: float) -> List[Tuple[str, str]]:
        """
        Get structural-based recommendations.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            wear_level: Current wear level (0-1)
            
        Returns:
            List of structural recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Deflection-based recommendations
        if self.current_deflection > 0.12:
            recommendations.append(("STRUCTURAL", f"High deflection {self.current_deflection:.1%}: consider increasing pressure"))
        elif self.current_deflection < 0.08:
            recommendations.append(("STRUCTURAL", f"Low deflection {self.current_deflection:.1%}: consider decreasing pressure"))
        
        # Contact patch recommendations
        if self.contact_patch_shape == ContactPatchShape.IRREGULAR:
            recommendations.append(("STRUCTURAL", "Irregular contact patch: check alignment and camber"))
        
        # Pressure distribution recommendations
        pressure_variance = np.var(self.pressure_distribution)
        if pressure_variance > 0.5:
            recommendations.append(("STRUCTURAL", f"High pressure variance {pressure_variance:.2f}: optimize contact patch"))
        
        # Temperature-structure coupling
        if Tt > 100 and self.current_deflection > 0.1:
            recommendations.append(("STRUCTURAL", "High temp + deflection: monitor structural integrity"))
        
        # Wear-structure coupling
        if wear_level > 0.5 and self.current_deflection > 0.1:
            recommendations.append(("STRUCTURAL", f"Wear {wear_level:.1%} + deflection: consider tire change"))
        
        return recommendations
    
    def reset_structural_state(self):
        """Reset structural state for new session."""
        self.current_deflection = 0.0
        self.deflection_velocity = 0.0
        self.contact_patch_area = 0.0
        self.pressure_distribution = np.zeros((10, 10))
        self.deflection_history = []
        self.pressure_history = []
        self.last_load = 0.0
