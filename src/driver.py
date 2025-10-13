from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class DrivingStyle(Enum):
    """Driving style classifications affecting tire management."""
    AGGRESSIVE = "aggressive"      # High thermal generation, fast lap times
    CONSERVATIVE = "conservative"  # Low thermal generation, consistent lap times
    BALANCED = "balanced"          # Moderate thermal generation, adaptable
    SMOOTH = "smooth"             # Very low thermal generation, gentle inputs
    BRAVE = "brave"               # High risk, high reward driving

class DriverExperience(Enum):
    """Driver experience levels affecting tire management."""
    ROOKIE = "rookie"             # Less tire management awareness
    EXPERIENCED = "experienced"    # Good tire management skills
    VETERAN = "veteran"           # Expert tire management
    CHAMPION = "champion"         # Master-level tire management

@dataclass
class DriverParams:
    """Parameters defining a driver's characteristics."""
    # Thermal signature multipliers
    thermal_aggression: float = 1.0      # How aggressively driver heats tires (0.5-2.0)
    thermal_efficiency: float = 1.0      # How efficiently driver manages tire temps (0.5-1.5)
    thermal_adaptation: float = 1.0      # How well driver adapts to changing conditions (0.5-1.5)
    
    # Driving style characteristics
    brake_aggression: float = 1.0        # Brake application intensity (0.5-2.0)
    throttle_aggression: float = 1.0     # Throttle application intensity (0.5-2.0)
    steering_aggression: float = 1.0     # Steering input intensity (0.5-2.0)
    
    # Tire management skills
    tire_awareness: float = 1.0          # Awareness of tire condition (0.5-1.5)
    pressure_management: float = 1.0     # Skill in managing tire pressure (0.5-1.5)
    compound_adaptation: float = 1.0     # Ability to adapt to different compounds (0.5-1.5)
    
    # Racecraft characteristics
    overtaking_aggression: float = 1.0   # Aggressiveness in overtaking (0.5-2.0)
    defensive_skill: float = 1.0         # Defensive driving ability (0.5-1.5)
    race_pace: float = 1.0              # Race pace management (0.5-1.5)
    
    # Weather adaptation
    wet_weather_skill: float = 1.0       # Skill in wet conditions (0.5-1.5)
    temperature_adaptation: float = 1.0   # Adaptation to temperature changes (0.5-1.5)

class DriverProfile:
    """
    Models individual driver characteristics and their effects on tire management.
    
    Features:
    - Thermal signature modeling (how driver heats tires)
    - Driving style classification (aggressive, conservative, balanced, etc.)
    - Experience-based tire management skills
    - Personalized recommendations based on driver characteristics
    - Adaptation to different conditions and compounds
    """
    
    def __init__(self, name: str, params: DriverParams = None, 
                 style: DrivingStyle = DrivingStyle.BALANCED,
                 experience: DriverExperience = DriverExperience.EXPERIENCED):
        self.name = name
        self.params = params or DriverParams()
        self.style = style
        self.experience = experience
        
        # Driver-specific thermal characteristics
        self.thermal_signature = self._calculate_thermal_signature()
        
        # Performance history tracking
        self.lap_history = []
        self.thermal_history = []
        self.recommendation_history = []
        
        # Current session state
        self.current_compound = "medium"
        self.session_laps = 0
        self.total_laps = 0
        
    def _calculate_thermal_signature(self) -> Dict[str, float]:
        """Calculate driver's thermal signature based on characteristics."""
        # Base thermal generation multiplier
        thermal_gen = (self.params.thermal_aggression * 
                      self.params.brake_aggression * 
                      self.params.throttle_aggression * 
                      self.params.steering_aggression) ** 0.25
        
        # Thermal efficiency (how well driver manages heat)
        thermal_eff = (self.params.thermal_efficiency * 
                      self.params.tire_awareness * 
                      self.params.pressure_management) ** 0.33
        
        # Adaptation capability
        adaptation = (self.params.thermal_adaptation * 
                     self.params.compound_adaptation * 
                     self.params.temperature_adaptation) ** 0.33
        
        # Style-based adjustments
        style_multipliers = {
            DrivingStyle.AGGRESSIVE: 1.3,
            DrivingStyle.CONSERVATIVE: 0.7,
            DrivingStyle.BALANCED: 1.0,
            DrivingStyle.SMOOTH: 0.6,
            DrivingStyle.BRAVE: 1.4
        }
        
        style_mult = style_multipliers.get(self.style, 1.0)
        
        # Experience-based adjustments
        experience_multipliers = {
            DriverExperience.ROOKIE: 0.8,
            DriverExperience.EXPERIENCED: 1.0,
            DriverExperience.VETERAN: 1.2,
            DriverExperience.CHAMPION: 1.3
        }
        
        exp_mult = experience_multipliers.get(self.experience, 1.0)
        
        return {
            'thermal_generation': thermal_gen * style_mult * exp_mult,
            'thermal_efficiency': thermal_eff * exp_mult,
            'adaptation_capability': adaptation * exp_mult,
            'style_multiplier': style_mult,
            'experience_multiplier': exp_mult
        }
    
    def get_thermal_multipliers(self, conditions: Dict) -> Dict[str, float]:
        """
        Get thermal multipliers based on current conditions and driver characteristics.
        
        Args:
            conditions: Dict with current conditions (temperature, weather, etc.)
            
        Returns:
            Dict with thermal multipliers for different aspects
        """
        base_signature = self.thermal_signature
        
        # Temperature adaptation
        track_temp = conditions.get('track_temperature', 35.0)
        temp_factor = self._calculate_temperature_factor(track_temp)
        
        # Weather adaptation
        weather_factor = self._calculate_weather_factor(conditions)
        
        # Compound adaptation
        compound_factor = self._calculate_compound_factor(conditions.get('compound', 'medium'))
        
        return {
            'thermal_generation': base_signature['thermal_generation'] * temp_factor * weather_factor,
            'thermal_efficiency': base_signature['thermal_efficiency'] * compound_factor,
            'adaptation_capability': base_signature['adaptation_capability'] * weather_factor,
            'temperature_factor': temp_factor,
            'weather_factor': weather_factor,
            'compound_factor': compound_factor
        }
    
    def _calculate_temperature_factor(self, track_temp: float) -> float:
        """Calculate temperature adaptation factor."""
        # Drivers have different optimal temperature ranges
        optimal_temp = 35.0  # Base optimal temperature
        
        # Temperature adaptation skill affects how well driver performs in non-optimal temps
        adaptation_skill = self.params.temperature_adaptation
        
        # Calculate performance factor based on temperature deviation
        temp_deviation = abs(track_temp - optimal_temp)
        temp_factor = 1.0 - (temp_deviation / 50.0) * (1.0 - adaptation_skill)
        
        return max(0.5, min(1.5, temp_factor))
    
    def _calculate_weather_factor(self, conditions: Dict) -> float:
        """Calculate weather adaptation factor."""
        rain_probability = conditions.get('rain_probability', 0.0)
        wind_speed = conditions.get('wind_speed', 0.0)
        
        # Wet weather skill affects performance in rain
        wet_factor = 1.0 - (rain_probability * (1.0 - self.params.wet_weather_skill))
        
        # Wind affects all drivers similarly, but some adapt better
        wind_factor = 1.0 - (wind_speed / 100.0) * (1.0 - self.params.thermal_adaptation)
        
        return max(0.5, min(1.2, wet_factor * wind_factor))
    
    def _calculate_compound_factor(self, compound: str) -> float:
        """Calculate compound adaptation factor."""
        # Different drivers have different compound preferences
        compound_preferences = {
            'soft': 1.0 + 0.1 * self.params.thermal_aggression,    # Aggressive drivers like soft
            'medium': 1.0,                                          # Baseline
            'hard': 1.0 - 0.1 * (self.params.thermal_aggression - 1.0)  # Conservative drivers like hard
        }
        
        base_factor = compound_preferences.get(compound, 1.0)
        adaptation_skill = self.params.compound_adaptation
        
        return max(0.7, min(1.3, base_factor * adaptation_skill))
    
    def get_personalized_recommendations(self, thermal_state: np.ndarray, 
                                       conditions: Dict, wear_summary: Dict) -> List[Tuple[str, str]]:
        """
        Get personalized recommendations based on driver characteristics.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            conditions: Current environmental conditions
            wear_summary: Current wear status
            
        Returns:
            List of personalized recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Get driver-specific thermal multipliers
        thermal_mult = self.get_thermal_multipliers(conditions)
        
        # Thermal management recommendations based on driver style
        if self.style == DrivingStyle.AGGRESSIVE:
            if Tt > 105:
                recommendations.append(("DRIVING", f"Aggressive style: Reduce brake bias, smoother entries"))
            elif Tt < 95:
                recommendations.append(("DRIVING", f"Aggressive style: Push harder, more aggressive braking"))
        
        elif self.style == DrivingStyle.CONSERVATIVE:
            if Tt > 100:
                recommendations.append(("DRIVING", f"Conservative style: Already managing well, maintain pace"))
            elif Tt < 90:
                recommendations.append(("DRIVING", f"Conservative style: Can push more, increase pace"))
        
        elif self.style == DrivingStyle.SMOOTH:
            if Tt > 102:
                recommendations.append(("DRIVING", f"Smooth style: Excellent thermal management, maintain"))
            elif Tt < 88:
                recommendations.append(("DRIVING", f"Smooth style: Can be more aggressive, increase pace"))
        
        # Experience-based recommendations
        if self.experience == DriverExperience.ROOKIE:
            if Tt > 100:
                recommendations.append(("COACHING", f"Rookie: Focus on smoother inputs, less aggressive braking"))
            recommendations.append(("COACHING", f"Rookie: Monitor tire temps closely, learn thermal management"))
        
        elif self.experience == DriverExperience.CHAMPION:
            if Tt > 108:
                recommendations.append(("CHAMPION", f"Champion level: Push boundaries, aggressive strategy"))
            recommendations.append(("CHAMPION", f"Champion level: Trust your instincts, adapt quickly"))
        
        # Driver-specific pressure recommendations
        pressure_skill = self.params.pressure_management
        if pressure_skill < 1.0:
            recommendations.append(("PRESSURE", f"Pressure management: Focus on consistent pressure, avoid big changes"))
        elif pressure_skill > 1.2:
            recommendations.append(("PRESSURE", f"Pressure management: Use pressure changes strategically"))
        
        # Weather-specific recommendations
        if conditions.get('rain_probability', 0.0) > 0.5:
            wet_skill = self.params.wet_weather_skill
            if wet_skill < 1.0:
                recommendations.append(("WEATHER", f"Wet conditions: Extra caution needed, reduce pace"))
            elif wet_skill > 1.2:
                recommendations.append(("WEATHER", f"Wet conditions: Use your wet weather skills, push when safe"))
        
        return recommendations
    
    def update_session_data(self, thermal_state: np.ndarray, lap_time: float, 
                           conditions: Dict, recommendations: List[Tuple[str, str]]):
        """Update driver's session data for analysis."""
        self.lap_history.append({
            'lap': self.session_laps,
            'lap_time': lap_time,
            'thermal_state': thermal_state.copy(),
            'conditions': conditions.copy(),
            'timestamp': len(self.lap_history)
        })
        
        self.thermal_history.append(thermal_state.copy())
        self.recommendation_history.extend(recommendations)
        
        self.session_laps += 1
        self.total_laps += 1
        
        # Keep only last 100 laps
        if len(self.lap_history) > 100:
            self.lap_history = self.lap_history[-100:]
            self.thermal_history = self.thermal_history[-100:]
    
    def get_driver_summary(self) -> Dict:
        """Get comprehensive driver summary."""
        return {
            'name': self.name,
            'style': self.style.value,
            'experience': self.experience.value,
            'thermal_signature': self.thermal_signature,
            'current_compound': self.current_compound,
            'session_laps': self.session_laps,
            'total_laps': self.total_laps,
            'average_lap_time': np.mean([lap['lap_time'] for lap in self.lap_history[-10:]]) if self.lap_history else 0.0,
            'thermal_consistency': self._calculate_thermal_consistency(),
            'recommendation_follow_rate': self._calculate_recommendation_follow_rate()
        }
    
    def _calculate_thermal_consistency(self) -> float:
        """Calculate how consistent driver's thermal management is."""
        if len(self.thermal_history) < 5:
            return 1.0
        
        # Calculate coefficient of variation for tread temperatures
        tread_temps = [state[0] for state in self.thermal_history[-20:]]
        if len(tread_temps) < 2:
            return 1.0
        
        mean_temp = np.mean(tread_temps)
        std_temp = np.std(tread_temps)
        
        if mean_temp == 0:
            return 1.0
        
        cv = std_temp / mean_temp
        consistency = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
        
        return consistency
    
    def _calculate_recommendation_follow_rate(self) -> float:
        """Calculate how well driver follows recommendations."""
        if len(self.recommendation_history) < 5:
            return 1.0
        
        # Simple heuristic: more experienced drivers follow recommendations better
        base_rate = 0.7  # Base follow rate
        
        experience_bonus = {
            DriverExperience.ROOKIE: -0.2,
            DriverExperience.EXPERIENCED: 0.0,
            DriverExperience.VETERAN: 0.1,
            DriverExperience.CHAMPION: 0.2
        }
        
        return max(0.0, min(1.0, base_rate + experience_bonus.get(self.experience, 0.0)))
    
    def reset_session(self):
        """Reset driver data for new session."""
        self.session_laps = 0
        self.lap_history = []
        self.thermal_history = []
        self.recommendation_history = []
    
    def set_compound(self, compound: str):
        """Set current tire compound."""
        self.current_compound = compound
