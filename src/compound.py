from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class CompoundType(Enum):
    """Advanced tire compound types."""
    ULTRA_SOFT = "ultra_soft"      # Maximum grip, minimum durability
    SUPER_SOFT = "super_soft"      # High grip, low durability
    SOFT = "soft"                  # Good grip, moderate durability
    MEDIUM = "medium"              # Balanced grip and durability
    HARD = "hard"                  # Lower grip, high durability
    SUPER_HARD = "super_hard"      # Minimum grip, maximum durability
    INTERMEDIATE = "intermediate"  # Wet weather compound
    FULL_WET = "full_wet"         # Heavy rain compound

class MaterialProperty(Enum):
    """Material properties affecting tire behavior."""
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    THERMAL_CAPACITY = "thermal_capacity"
    ELASTIC_MODULUS = "elastic_modulus"
    VISCOELASTIC_MODULUS = "viscoelastic_modulus"
    FRICTION_COEFFICIENT = "friction_coefficient"
    WEAR_RESISTANCE = "wear_resistance"
    FATIGUE_RESISTANCE = "fatigue_resistance"

@dataclass
class CompoundParams:
    """Parameters for advanced tire compound modeling."""
    # Base compound properties
    compound_name: str = "medium"
    compound_type: CompoundType = CompoundType.MEDIUM
    
    # Thermal properties
    thermal_conductivity: float = 1.0      # Thermal conductivity multiplier
    thermal_capacity: float = 1.0          # Thermal capacity multiplier
    thermal_diffusivity: float = 1.0       # Thermal diffusivity multiplier
    
    # Mechanical properties
    elastic_modulus: float = 1.0           # Elastic modulus multiplier
    viscoelastic_modulus: float = 1.0     # Viscoelastic modulus multiplier
    poisson_ratio: float = 0.5            # Poisson's ratio
    density: float = 1.0                  # Material density multiplier
    
    # Friction properties
    static_friction: float = 1.0          # Static friction coefficient
    dynamic_friction: float = 0.9         # Dynamic friction coefficient
    friction_temperature_sensitivity: float = 0.1  # Friction change with temperature
    
    # Wear properties
    wear_resistance: float = 1.0          # Wear resistance multiplier
    fatigue_resistance: float = 1.0       # Fatigue resistance multiplier
    abrasion_resistance: float = 1.0      # Abrasion resistance multiplier
    
    # Temperature characteristics
    glass_transition_temp: float = -50.0  # Glass transition temperature (°C)
    optimal_temp_range: Tuple[float, float] = (80.0, 100.0)  # Optimal temperature range
    max_operating_temp: float = 120.0     # Maximum operating temperature
    
    # Pressure sensitivity
    pressure_sensitivity: float = 0.1     # Pressure sensitivity factor
    inflation_pressure_optimal: float = 1.5  # Optimal inflation pressure (bar)
    
    # Ageing properties
    ageing_rate: float = 0.001           # Ageing rate per hour
    uv_resistance: float = 1.0           # UV resistance multiplier
    ozone_resistance: float = 1.0        # Ozone resistance multiplier

class AdvancedCompoundModel:
    """
    Advanced tire compound model with complex material properties and behavior.
    
    Features:
    - Temperature-dependent material properties
    - Pressure-sensitive behavior
    - Ageing and degradation modeling
    - Multi-property coupling effects
    - Compound-specific optimization
    - Real-time property adaptation
    """
    
    def __init__(self, params: CompoundParams = None):
        self.p = params or CompoundParams()
        
        # Current state
        self.current_temperature = 25.0
        self.current_pressure = 1.5
        self.current_wear = 0.0
        self.ageing_level = 0.0
        
        # Material properties (temperature-dependent)
        self.material_properties = self._initialize_material_properties()
        
        # Property history
        self.property_history = []
        self.temperature_history = []
        self.pressure_history = []
        
        # Compound-specific characteristics
        self.compound_characteristics = self._get_compound_characteristics()
    
    def _initialize_material_properties(self) -> Dict[str, float]:
        """Initialize material properties based on compound type."""
        base_properties = {
            'thermal_conductivity': self.p.thermal_conductivity,
            'thermal_capacity': self.p.thermal_capacity,
            'elastic_modulus': self.p.elastic_modulus,
            'viscoelastic_modulus': self.p.viscoelastic_modulus,
            'static_friction': self.p.static_friction,
            'dynamic_friction': self.p.dynamic_friction,
            'wear_resistance': self.p.wear_resistance,
            'fatigue_resistance': self.p.fatigue_resistance
        }
        
        return base_properties
    
    def _get_compound_characteristics(self) -> Dict[str, float]:
        """Get compound-specific characteristics."""
        characteristics = {
            CompoundType.ULTRA_SOFT: {
                'thermal_conductivity': 0.8,
                'thermal_capacity': 1.2,
                'elastic_modulus': 0.7,
                'static_friction': 1.4,
                'wear_resistance': 0.6,
                'optimal_temp_range': (75.0, 95.0),
                'max_operating_temp': 110.0
            },
            CompoundType.SUPER_SOFT: {
                'thermal_conductivity': 0.9,
                'thermal_capacity': 1.1,
                'elastic_modulus': 0.8,
                'static_friction': 1.3,
                'wear_resistance': 0.7,
                'optimal_temp_range': (80.0, 100.0),
                'max_operating_temp': 115.0
            },
            CompoundType.SOFT: {
                'thermal_conductivity': 1.0,
                'thermal_capacity': 1.0,
                'elastic_modulus': 0.9,
                'static_friction': 1.2,
                'wear_resistance': 0.8,
                'optimal_temp_range': (85.0, 105.0),
                'max_operating_temp': 120.0
            },
            CompoundType.MEDIUM: {
                'thermal_conductivity': 1.0,
                'thermal_capacity': 1.0,
                'elastic_modulus': 1.0,
                'static_friction': 1.0,
                'wear_resistance': 1.0,
                'optimal_temp_range': (90.0, 110.0),
                'max_operating_temp': 125.0
            },
            CompoundType.HARD: {
                'thermal_conductivity': 1.1,
                'thermal_capacity': 0.9,
                'elastic_modulus': 1.1,
                'static_friction': 0.9,
                'wear_resistance': 1.2,
                'optimal_temp_range': (95.0, 115.0),
                'max_operating_temp': 130.0
            },
            CompoundType.SUPER_HARD: {
                'thermal_conductivity': 1.2,
                'thermal_capacity': 0.8,
                'elastic_modulus': 1.2,
                'static_friction': 0.8,
                'wear_resistance': 1.4,
                'optimal_temp_range': (100.0, 120.0),
                'max_operating_temp': 135.0
            },
            CompoundType.INTERMEDIATE: {
                'thermal_conductivity': 1.3,
                'thermal_capacity': 1.1,
                'elastic_modulus': 0.8,
                'static_friction': 1.1,
                'wear_resistance': 1.1,
                'optimal_temp_range': (70.0, 90.0),
                'max_operating_temp': 100.0
            },
            CompoundType.FULL_WET: {
                'thermal_conductivity': 1.5,
                'thermal_capacity': 1.2,
                'elastic_modulus': 0.7,
                'static_friction': 1.0,
                'wear_resistance': 1.3,
                'optimal_temp_range': (60.0, 80.0),
                'max_operating_temp': 90.0
            }
        }
        
        return characteristics.get(self.p.compound_type, characteristics[CompoundType.MEDIUM])
    
    def update_compound_state(self, temperature: float, pressure: float, wear_level: float,
                            ageing_time: float = 0.0, dt: float = 0.1):
        """
        Update compound state and recalculate material properties.
        
        Args:
            temperature: Current tire temperature (°C)
            pressure: Current inflation pressure (bar)
            wear_level: Current wear level (0-1)
            ageing_time: Ageing time (hours)
            dt: Time step (s)
        """
        self.current_temperature = temperature
        self.current_pressure = pressure
        self.current_wear = wear_level
        
        # Update ageing
        self.ageing_level += ageing_time * self.p.ageing_rate
        
        # Recalculate material properties
        self._update_material_properties()
        
        # Store history
        self._update_property_history()
    
    def _update_material_properties(self):
        """Update material properties based on current state."""
        temp = self.current_temperature
        pressure = self.current_pressure
        wear = self.current_wear
        ageing = self.ageing_level
        
        # Get compound characteristics
        char = self.compound_characteristics
        
        # Temperature effects
        temp_factor = self._calculate_temperature_factor(temp)
        
        # Pressure effects
        pressure_factor = self._calculate_pressure_factor(pressure)
        
        # Wear effects
        wear_factor = self._calculate_wear_factor(wear)
        
        # Ageing effects
        ageing_factor = self._calculate_ageing_factor(ageing)
        
        # Update properties
        self.material_properties['thermal_conductivity'] = (
            char['thermal_conductivity'] * temp_factor * pressure_factor * wear_factor * ageing_factor
        )
        
        self.material_properties['thermal_capacity'] = (
            char['thermal_capacity'] * temp_factor * pressure_factor * wear_factor * ageing_factor
        )
        
        self.material_properties['elastic_modulus'] = (
            char['elastic_modulus'] * temp_factor * pressure_factor * wear_factor * ageing_factor
        )
        
        self.material_properties['static_friction'] = (
            char['static_friction'] * temp_factor * pressure_factor * wear_factor * ageing_factor
        )
        
        self.material_properties['wear_resistance'] = (
            char['wear_resistance'] * temp_factor * pressure_factor * wear_factor * ageing_factor
        )
    
    def _calculate_temperature_factor(self, temperature: float) -> float:
        """Calculate temperature factor for material properties."""
        optimal_range = self.compound_characteristics['optimal_temp_range']
        min_temp, max_temp = optimal_range
        
        if min_temp <= temperature <= max_temp:
            # Optimal temperature range
            return 1.0
        elif temperature < min_temp:
            # Below optimal - properties degrade
            factor = 1.0 - (min_temp - temperature) / 100.0
            return max(0.5, factor)
        else:
            # Above optimal - properties degrade
            factor = 1.0 - (temperature - max_temp) / 100.0
            return max(0.3, factor)
    
    def _calculate_pressure_factor(self, pressure: float) -> float:
        """Calculate pressure factor for material properties."""
        optimal_pressure = self.p.inflation_pressure_optimal
        pressure_diff = abs(pressure - optimal_pressure)
        
        # Pressure sensitivity
        factor = 1.0 - pressure_diff * self.p.pressure_sensitivity
        return max(0.7, min(1.3, factor))
    
    def _calculate_wear_factor(self, wear_level: float) -> float:
        """Calculate wear factor for material properties."""
        # Wear reduces most properties
        wear_reduction = wear_level * 0.5  # Up to 50% reduction at full wear
        return 1.0 - wear_reduction
    
    def _calculate_ageing_factor(self, ageing_level: float) -> float:
        """Calculate ageing factor for material properties."""
        # Ageing reduces properties over time
        ageing_reduction = min(ageing_level * 0.1, 0.3)  # Up to 30% reduction
        return 1.0 - ageing_reduction
    
    def _update_property_history(self):
        """Update property history."""
        self.property_history.append(self.material_properties.copy())
        self.temperature_history.append(self.current_temperature)
        self.pressure_history.append(self.current_pressure)
        
        # Keep only last 100 points
        if len(self.property_history) > 100:
            self.property_history = self.property_history[-100:]
            self.temperature_history = self.temperature_history[-100:]
            self.pressure_history = self.pressure_history[-100:]
    
    def get_thermal_properties(self) -> Dict[str, float]:
        """Get current thermal properties."""
        return {
            'thermal_conductivity': self.material_properties['thermal_conductivity'],
            'thermal_capacity': self.material_properties['thermal_capacity'],
            'thermal_diffusivity': self.material_properties['thermal_conductivity'] / 
                                 self.material_properties['thermal_capacity'],
            'optimal_temp_range': self.compound_characteristics['optimal_temp_range'],
            'max_operating_temp': self.compound_characteristics['max_operating_temp']
        }
    
    def get_mechanical_properties(self) -> Dict[str, float]:
        """Get current mechanical properties."""
        return {
            'elastic_modulus': self.material_properties['elastic_modulus'],
            'viscoelastic_modulus': self.material_properties['viscoelastic_modulus'],
            'poisson_ratio': self.p.poisson_ratio,
            'density': self.p.density,
            'static_friction': self.material_properties['static_friction'],
            'dynamic_friction': self.material_properties['dynamic_friction']
        }
    
    def get_wear_properties(self) -> Dict[str, float]:
        """Get current wear properties."""
        return {
            'wear_resistance': self.material_properties['wear_resistance'],
            'fatigue_resistance': self.material_properties['fatigue_resistance'],
            'abrasion_resistance': self.p.abrasion_resistance,
            'current_wear': self.current_wear,
            'ageing_level': self.ageing_level
        }
    
    def get_compound_recommendations(self, thermal_state: np.ndarray, 
                                   structural_state: Dict[str, float]) -> List[Tuple[str, str]]:
        """
        Get compound-specific recommendations.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            structural_state: Current structural state
            
        Returns:
            List of compound recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Temperature-based recommendations
        optimal_range = self.compound_characteristics['optimal_temp_range']
        min_temp, max_temp = optimal_range
        
        if Tt < min_temp:
            recommendations.append(("COMPOUND", f"Below optimal temp {Tt:.1f}°C < {min_temp}°C: warm up tires"))
        elif Tt > max_temp:
            recommendations.append(("COMPOUND", f"Above optimal temp {Tt:.1f}°C > {max_temp}°C: cool down tires"))
        
        # Pressure-based recommendations
        pressure_diff = abs(self.current_pressure - self.p.inflation_pressure_optimal)
        if pressure_diff > 0.2:
            recommendations.append(("COMPOUND", f"Pressure deviation {pressure_diff:.1f}bar: adjust to {self.p.inflation_pressure_optimal}bar"))
        
        # Wear-based recommendations
        if self.current_wear > 0.7:
            recommendations.append(("COMPOUND", f"High wear {self.current_wear:.1%}: consider tire change"))
        elif self.current_wear > 0.5:
            recommendations.append(("COMPOUND", f"Moderate wear {self.current_wear:.1%}: monitor closely"))
        
        # Ageing-based recommendations
        if self.ageing_level > 0.5:
            recommendations.append(("COMPOUND", f"High ageing {self.ageing_level:.1%}: consider fresh tires"))
        
        # Compound-specific recommendations
        if self.p.compound_type == CompoundType.ULTRA_SOFT:
            if Tt > 100:
                recommendations.append(("COMPOUND", "Ultra-soft: high temp risk - manage thermal load"))
        elif self.p.compound_type == CompoundType.SUPER_HARD:
            if Tt < 90:
                recommendations.append(("COMPOUND", "Super-hard: low temp - may struggle for grip"))
        elif self.p.compound_type in [CompoundType.INTERMEDIATE, CompoundType.FULL_WET]:
            if Tt > 80:
                recommendations.append(("COMPOUND", "Wet compound: high temp - may overheat"))
        
        return recommendations
    
    def get_compound_summary(self) -> Dict[str, float]:
        """Get comprehensive compound summary."""
        return {
            'compound_name': self.p.compound_name,
            'compound_type': self.p.compound_type.value,
            'current_temperature': self.current_temperature,
            'current_pressure': self.current_pressure,
            'current_wear': self.current_wear,
            'ageing_level': self.ageing_level,
            'thermal_properties': self.get_thermal_properties(),
            'mechanical_properties': self.get_mechanical_properties(),
            'wear_properties': self.get_wear_properties(),
            'optimal_temp_range': self.compound_characteristics['optimal_temp_range'],
            'max_operating_temp': self.compound_characteristics['max_operating_temp']
        }
    
    def reset_compound_state(self):
        """Reset compound state for new session."""
        self.current_temperature = 25.0
        self.current_pressure = 1.5
        self.current_wear = 0.0
        self.ageing_level = 0.0
        self.property_history = []
        self.temperature_history = []
        self.pressure_history = []
        self.material_properties = self._initialize_material_properties()
    
    def set_compound_type(self, compound_type: CompoundType):
        """Set compound type and update characteristics."""
        self.p.compound_type = compound_type
        self.compound_characteristics = self._get_compound_characteristics()
        self.material_properties = self._initialize_material_properties()
