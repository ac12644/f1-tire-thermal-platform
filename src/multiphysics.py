from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from structural import StructuralTireModel, StructuralParams, TireConstruction
from aerodynamic import AerodynamicModel, AeroParams, AeroMode, CarPosition
from thermal import ThermalModel, ThermalParams
from wear import TireWearModel, WearParams
from weather import WeatherModel, WeatherParams

class CouplingMode(Enum):
    """Multi-physics coupling modes."""
    THERMAL_STRUCTURAL = "thermal_structural"      # Thermal-structural coupling
    THERMAL_AERO = "thermal_aero"                  # Thermal-aerodynamic coupling
    STRUCTURAL_AERO = "structural_aero"             # Structural-aerodynamic coupling
    FULL_COUPLING = "full_coupling"                # Full multi-physics coupling

@dataclass
class MultiPhysicsParams:
    """Parameters for multi-physics coupling."""
    # Coupling strengths
    thermal_structural_coupling: float = 0.3      # Thermal affects structural properties
    structural_thermal_coupling: float = 0.2      # Structural affects thermal generation
    thermal_aero_coupling: float = 0.1             # Thermal affects aerodynamic properties
    aero_thermal_coupling: float = 0.4             # Aerodynamic affects thermal cooling
    structural_aero_coupling: float = 0.1          # Structural affects aerodynamic properties
    
    # Temperature thresholds for coupling effects
    thermal_coupling_threshold: float = 80.0      # Temperature above which coupling is significant
    structural_coupling_threshold: float = 0.1    # Deflection threshold for coupling
    
    # Coupling time constants
    thermal_time_constant: float = 1.0            # Thermal coupling time constant
    structural_time_constant: float = 0.1          # Structural coupling time constant
    aero_time_constant: float = 0.05               # Aerodynamic coupling time constant

class MultiPhysicsCoupling:
    """
    Multi-physics coupling system integrating thermal, structural, and aerodynamic models.
    
    Features:
    - Thermal-structural coupling (temperature affects stiffness, deflection affects thermal generation)
    - Thermal-aerodynamic coupling (temperature affects aerodynamic properties, aero affects cooling)
    - Structural-aerodynamic coupling (deflection affects aerodynamic properties)
    - Full multi-physics integration with feedback loops
    - Adaptive coupling strength based on conditions
    """
    
    def __init__(self, thermal_model: ThermalModel, structural_model: StructuralTireModel,
                 aerodynamic_model: AerodynamicModel, params: MultiPhysicsParams = None):
        self.thermal_model = thermal_model
        self.structural_model = structural_model
        self.aerodynamic_model = aerodynamic_model
        self.p = params or MultiPhysicsParams()
        
        # Coupling state
        self.coupling_mode = CouplingMode.FULL_COUPLING
        self.coupling_active = True
        
        # Coupling history
        self.coupling_history = []
        self.thermal_coupling_history = []
        self.structural_coupling_history = []
        self.aero_coupling_history = []
        
        # Coupling effects tracking
        self.thermal_to_structural_effects = {}
        self.structural_to_thermal_effects = {}
        self.thermal_to_aero_effects = {}
        self.aero_to_thermal_effects = {}
        self.structural_to_aero_effects = {}
    
    def update_coupling(self, thermal_state: np.ndarray, structural_state: Dict[str, float],
                       aerodynamic_state: Dict[str, float], wear_level: float,
                       vertical_load: float, inflation_pressure: float, dt: float):
        """
        Update multi-physics coupling effects.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            structural_state: Current structural state
            aerodynamic_state: Current aerodynamic state
            wear_level: Current wear level (0-1)
            vertical_load: Vertical load (N)
            inflation_pressure: Inflation pressure (kPa)
            dt: Time step (s)
        """
        if not self.coupling_active:
            return
        
        Tt, Tc, Tr = thermal_state
        
        # Calculate coupling effects
        self._calculate_thermal_structural_coupling(Tt, structural_state, wear_level, dt)
        self._calculate_thermal_aero_coupling(Tt, aerodynamic_state, dt)
        self._calculate_structural_aero_coupling(structural_state, aerodynamic_state, dt)
        
        # Apply coupling effects
        self._apply_coupling_effects(thermal_state, structural_state, aerodynamic_state, dt)
        
        # Update coupling history
        self._update_coupling_history(thermal_state, structural_state, aerodynamic_state)
    
    def _calculate_thermal_structural_coupling(self, tread_temp: float, structural_state: Dict[str, float],
                                             wear_level: float, dt: float):
        """Calculate thermal-structural coupling effects."""
        # Thermal affects structural properties
        if tread_temp > self.p.thermal_coupling_threshold:
            # High temperature reduces stiffness
            temp_factor = (tread_temp - self.p.thermal_coupling_threshold) / 50.0
            stiffness_reduction = temp_factor * self.p.thermal_structural_coupling
            
            self.thermal_to_structural_effects = {
                'stiffness_reduction': stiffness_reduction,
                'deflection_increase': stiffness_reduction * 0.5,
                'pressure_distribution_change': stiffness_reduction * 0.3
            }
        else:
            self.thermal_to_structural_effects = {}
        
        # Structural affects thermal generation
        deflection = structural_state.get('deflection_ratio', 0.0)
        if deflection > self.p.structural_coupling_threshold:
            # High deflection increases thermal generation
            deflection_factor = deflection / self.p.structural_coupling_threshold
            thermal_increase = deflection_factor * self.p.structural_thermal_coupling
            
            self.structural_to_thermal_effects = {
                'thermal_generation_increase': thermal_increase,
                'heat_capacity_change': -thermal_increase * 0.2,
                'thermal_conductivity_change': thermal_increase * 0.1
            }
        else:
            self.structural_to_thermal_effects = {}
    
    def _calculate_thermal_aero_coupling(self, tread_temp: float, aerodynamic_state: Dict[str, float], dt: float):
        """Calculate thermal-aerodynamic coupling effects."""
        # Thermal affects aerodynamic properties
        if tread_temp > self.p.thermal_coupling_threshold:
            # High temperature affects aerodynamic properties
            temp_factor = (tread_temp - self.p.thermal_coupling_threshold) / 50.0
            
            self.thermal_to_aero_effects = {
                'downforce_reduction': temp_factor * self.p.thermal_aero_coupling * 0.1,
                'cooling_efficiency_change': temp_factor * self.p.thermal_aero_coupling * 0.2,
                'drag_change': temp_factor * self.p.thermal_aero_coupling * 0.05
            }
        else:
            self.thermal_to_aero_effects = {}
        
        # Aerodynamic affects thermal cooling
        cooling_multiplier = aerodynamic_state.get('cooling_multiplier', 1.0)
        if cooling_multiplier != 1.0:
            self.aero_to_thermal_effects = {
                'cooling_factor': cooling_multiplier,
                'thermal_generation_change': (cooling_multiplier - 1.0) * self.p.aero_thermal_coupling * 0.1,
                'heat_transfer_change': (cooling_multiplier - 1.0) * self.p.aero_thermal_coupling * 0.3
            }
        else:
            self.aero_to_thermal_effects = {}
    
    def _calculate_structural_aero_coupling(self, structural_state: Dict[str, float],
                                          aerodynamic_state: Dict[str, float], dt: float):
        """Calculate structural-aerodynamic coupling effects."""
        # Structural affects aerodynamic properties
        deflection = structural_state.get('deflection_ratio', 0.0)
        if deflection > self.p.structural_coupling_threshold:
            # High deflection affects aerodynamic properties
            deflection_factor = deflection / self.p.structural_coupling_threshold
            
            self.structural_to_aero_effects = {
                'downforce_change': deflection_factor * self.p.structural_aero_coupling * 0.05,
                'drag_change': deflection_factor * self.p.structural_aero_coupling * 0.1,
                'cooling_change': deflection_factor * self.p.structural_aero_coupling * 0.02
            }
        else:
            self.structural_to_aero_effects = {}
    
    def _apply_coupling_effects(self, thermal_state: np.ndarray, structural_state: Dict[str, float],
                              aerodynamic_state: Dict[str, float], dt: float):
        """Apply coupling effects to the models."""
        # Apply thermal-structural coupling
        if self.thermal_to_structural_effects:
            self._apply_thermal_to_structural_effects()
        
        if self.structural_to_thermal_effects:
            self._apply_structural_to_thermal_effects()
        
        # Apply thermal-aerodynamic coupling
        if self.thermal_to_aero_effects:
            self._apply_thermal_to_aero_effects()
        
        if self.aero_to_thermal_effects:
            self._apply_aero_to_thermal_effects()
        
        # Apply structural-aerodynamic coupling
        if self.structural_to_aero_effects:
            self._apply_structural_to_aero_effects()
    
    def _apply_thermal_to_structural_effects(self):
        """Apply thermal effects to structural model."""
        effects = self.thermal_to_structural_effects
        
        # Modify structural parameters based on thermal effects
        if 'stiffness_reduction' in effects:
            # This would modify the structural model's stiffness parameters
            # For now, we'll track the effects
            pass
    
    def _apply_structural_to_thermal_effects(self):
        """Apply structural effects to thermal model."""
        effects = self.structural_to_thermal_effects
        
        # Modify thermal parameters based on structural effects
        if 'thermal_generation_increase' in effects:
            # This would modify the thermal model's generation parameters
            # For now, we'll track the effects
            pass
    
    def _apply_thermal_to_aero_effects(self):
        """Apply thermal effects to aerodynamic model."""
        effects = self.thermal_to_aero_effects
        
        # Modify aerodynamic parameters based on thermal effects
        if 'downforce_reduction' in effects:
            # This would modify the aerodynamic model's downforce parameters
            # For now, we'll track the effects
            pass
    
    def _apply_aero_to_thermal_effects(self):
        """Apply aerodynamic effects to thermal model."""
        effects = self.aero_to_thermal_effects
        
        # Modify thermal parameters based on aerodynamic effects
        if 'cooling_factor' in effects:
            # This would modify the thermal model's cooling parameters
            # For now, we'll track the effects
            pass
    
    def _apply_structural_to_aero_effects(self):
        """Apply structural effects to aerodynamic model."""
        effects = self.structural_to_aero_effects
        
        # Modify aerodynamic parameters based on structural effects
        if 'downforce_change' in effects:
            # This would modify the aerodynamic model's parameters
            # For now, we'll track the effects
            pass
    
    def _update_coupling_history(self, thermal_state: np.ndarray, structural_state: Dict[str, float],
                                aerodynamic_state: Dict[str, float]):
        """Update coupling history."""
        coupling_data = {
            'thermal_state': thermal_state.copy(),
            'structural_state': structural_state.copy(),
            'aerodynamic_state': aerodynamic_state.copy(),
            'thermal_to_structural': self.thermal_to_structural_effects.copy(),
            'structural_to_thermal': self.structural_to_thermal_effects.copy(),
            'thermal_to_aero': self.thermal_to_aero_effects.copy(),
            'aero_to_thermal': self.aero_to_thermal_effects.copy(),
            'structural_to_aero': self.structural_to_aero_effects.copy()
        }
        
        self.coupling_history.append(coupling_data)
        
        # Keep only last 100 points
        if len(self.coupling_history) > 100:
            self.coupling_history = self.coupling_history[-100:]
    
    def get_coupling_summary(self) -> Dict[str, float]:
        """Get comprehensive coupling summary."""
        return {
            'coupling_mode': self.coupling_mode.value,
            'coupling_active': self.coupling_active,
            'thermal_structural_coupling_strength': self.p.thermal_structural_coupling,
            'structural_thermal_coupling_strength': self.p.structural_thermal_coupling,
            'thermal_aero_coupling_strength': self.p.thermal_aero_coupling,
            'aero_thermal_coupling_strength': self.p.aero_thermal_coupling,
            'structural_aero_coupling_strength': self.p.structural_aero_coupling,
            'thermal_to_structural_effects': len(self.thermal_to_structural_effects),
            'structural_to_thermal_effects': len(self.structural_to_thermal_effects),
            'thermal_to_aero_effects': len(self.thermal_to_aero_effects),
            'aero_to_thermal_effects': len(self.aero_to_thermal_effects),
            'structural_to_aero_effects': len(self.structural_to_aero_effects)
        }
    
    def get_coupling_recommendations(self, thermal_state: np.ndarray, structural_state: Dict[str, float],
                                  aerodynamic_state: Dict[str, float]) -> List[Tuple[str, str]]:
        """
        Get multi-physics coupling recommendations.
        
        Args:
            thermal_state: Current thermal state
            structural_state: Current structural state
            aerodynamic_state: Current aerodynamic state
            
        Returns:
            List of coupling recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Thermal-structural coupling recommendations
        if self.thermal_to_structural_effects:
            stiffness_reduction = self.thermal_to_structural_effects.get('stiffness_reduction', 0.0)
            if stiffness_reduction > 0.1:
                recommendations.append(("COUPLING", f"Thermal-structural: {stiffness_reduction:.1%} stiffness reduction from high temps"))
        
        if self.structural_to_thermal_effects:
            thermal_increase = self.structural_to_thermal_effects.get('thermal_generation_increase', 0.0)
            if thermal_increase > 0.1:
                recommendations.append(("COUPLING", f"Structural-thermal: {thermal_increase:.1%} thermal increase from deflection"))
        
        # Thermal-aerodynamic coupling recommendations
        if self.thermal_to_aero_effects:
            downforce_reduction = self.thermal_to_aero_effects.get('downforce_reduction', 0.0)
            if downforce_reduction > 0.05:
                recommendations.append(("COUPLING", f"Thermal-aero: {downforce_reduction:.1%} downforce reduction from high temps"))
        
        if self.aero_to_thermal_effects:
            cooling_factor = self.aero_to_thermal_effects.get('cooling_factor', 1.0)
            if cooling_factor > 1.2:
                recommendations.append(("COUPLING", f"Aero-thermal: {cooling_factor:.1f}x cooling increase from aero"))
            elif cooling_factor < 0.8:
                recommendations.append(("COUPLING", f"Aero-thermal: {cooling_factor:.1f}x cooling reduction from aero"))
        
        # Structural-aerodynamic coupling recommendations
        if self.structural_to_aero_effects:
            drag_change = self.structural_to_aero_effects.get('drag_change', 0.0)
            if drag_change > 0.05:
                recommendations.append(("COUPLING", f"Structural-aero: {drag_change:.1%} drag increase from deflection"))
        
        # Multi-physics stability recommendations
        if Tt > 110 and structural_state.get('deflection_ratio', 0.0) > 0.12:
            recommendations.append(("COUPLING", "Multi-physics: High temp + deflection - monitor stability"))
        
        if aerodynamic_state.get('cooling_multiplier', 1.0) < 0.7 and Tt > 100:
            recommendations.append(("COUPLING", "Multi-physics: Reduced aero cooling + high temp - critical"))
        
        return recommendations
    
    def set_coupling_mode(self, mode: CouplingMode):
        """Set coupling mode."""
        self.coupling_mode = mode
        
        # Adjust coupling parameters based on mode
        if mode == CouplingMode.THERMAL_STRUCTURAL:
            self.p.thermal_aero_coupling = 0.0
            self.p.structural_aero_coupling = 0.0
        elif mode == CouplingMode.THERMAL_AERO:
            self.p.thermal_structural_coupling = 0.0
            self.p.structural_aero_coupling = 0.0
        elif mode == CouplingMode.STRUCTURAL_AERO:
            self.p.thermal_structural_coupling = 0.0
            self.p.thermal_aero_coupling = 0.0
        elif mode == CouplingMode.FULL_COUPLING:
            # Restore full coupling
            self.p.thermal_structural_coupling = 0.3
            self.p.structural_thermal_coupling = 0.2
            self.p.thermal_aero_coupling = 0.1
            self.p.aero_thermal_coupling = 0.4
            self.p.structural_aero_coupling = 0.1
    
    def enable_coupling(self, enabled: bool = True):
        """Enable or disable coupling."""
        self.coupling_active = enabled
    
    def reset_coupling_state(self):
        """Reset coupling state for new session."""
        self.coupling_history = []
        self.thermal_coupling_history = []
        self.structural_coupling_history = []
        self.aero_coupling_history = []
        self.thermal_to_structural_effects = {}
        self.structural_to_thermal_effects = {}
        self.thermal_to_aero_effects = {}
        self.aero_to_thermal_effects = {}
        self.structural_to_aero_effects = {}
