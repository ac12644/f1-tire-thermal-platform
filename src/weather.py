from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time

class WeatherCondition(Enum):
    """Weather conditions affecting tire performance."""
    DRY = "dry"
    DAMP = "damp"
    WET = "wet"
    HEAVY_RAIN = "heavy_rain"

class SessionType(Enum):
    """F1 session types with different characteristics."""
    FP1 = "fp1"  # Free Practice 1
    FP2 = "fp2"  # Free Practice 2
    FP3 = "fp3"  # Free Practice 3
    QUALIFYING = "qualifying"
    RACE = "race"

@dataclass
class WeatherParams:
    """Parameters for weather modeling."""
    # Rain probability modeling
    rain_probability: float = 0.0  # 0.0 to 1.0
    rain_intensity: float = 0.0   # 0.0 to 1.0 (light to heavy)
    
    # Environmental factors
    humidity: float = 0.5         # 0.0 to 1.0 (relative humidity)
    wind_speed: float = 0.0       # km/h
    wind_direction: float = 0.0   # degrees (0 = headwind, 180 = tailwind)
    solar_radiation: float = 0.8   # 0.0 to 1.0 (cloud cover factor)
    
    # Track evolution parameters
    rubber_buildup_rate: float = 0.001  # Per lap
    track_evolution_factor: float = 1.0  # Multiplier for track changes
    
    # Session-specific parameters
    session_duration_minutes: float = 60.0
    track_temperature_base: float = 35.0  # Base track temperature
    ambient_temperature_base: float = 25.0  # Base ambient temperature

class WeatherModel:
    """
    Models weather conditions and their effects on tire performance.
    
    Features:
    - Rain probability tracking and intensity
    - Track temperature evolution throughout session
    - Environmental factor effects (humidity, wind, solar)
    - Session progression modeling (FP1 → FP2 → FP3 → Qual → Race)
    - Rubber buildup effects on grip and thermal behavior
    """
    
    def __init__(self, params: WeatherParams = None):
        self.params = params or WeatherParams()
        
        # Current weather state
        self.current_condition = WeatherCondition.DRY
        self.session_type = SessionType.FP1
        self.session_start_time = time.time()
        self.session_elapsed_minutes = 0.0
        
        # Track evolution state
        self.rubber_buildup = 0.0  # 0.0 to 1.0
        self.track_temperature = self.params.track_temperature_base
        self.ambient_temperature = self.params.ambient_temperature_base
        
        # Weather history for trend analysis
        self.weather_history = []
        self.track_temp_history = []
        self.ambient_temp_history = []
        
        # Environmental effects
        self.cooling_factor = 1.0  # Multiplier for tire cooling
        self.grip_factor = 1.0     # Multiplier for tire grip
        self.thermal_factor = 1.0  # Multiplier for thermal behavior
        
    def update_weather(self, dt: float, lap_count: int = 0):
        """
        Update weather conditions and environmental effects.
        
        Args:
            dt: Time step in seconds
            lap_count: Current lap number for track evolution
        """
        self.session_elapsed_minutes += dt / 60.0
        
        # Update rain probability (can change during session)
        self._update_rain_probability()
        
        # Update track temperature evolution
        self._update_track_temperature()
        
        # Update ambient temperature
        self._update_ambient_temperature()
        
        # Update rubber buildup
        self._update_rubber_buildup(lap_count)
        
        # Calculate environmental effects
        self._calculate_environmental_effects()
        
        # Store history
        self._store_weather_history()
    
    def _update_rain_probability(self):
        """Update rain probability based on session progression and conditions."""
        # Rain probability can increase during session (realistic weather change)
        session_progress = self.session_elapsed_minutes / self.params.session_duration_minutes
        
        # Simulate weather front moving in (example: increasing rain probability)
        if session_progress > 0.3:  # After 30% of session
            self.params.rain_probability = min(1.0, 
                self.params.rain_probability + 0.001 * np.random.uniform(0, 1))
        
        # Determine current weather condition
        if self.params.rain_probability > 0.8:
            self.current_condition = WeatherCondition.HEAVY_RAIN
            self.params.rain_intensity = 0.8 + 0.2 * np.random.uniform(0, 1)
        elif self.params.rain_probability > 0.5:
            self.current_condition = WeatherCondition.WET
            self.params.rain_intensity = 0.4 + 0.4 * np.random.uniform(0, 1)
        elif self.params.rain_probability > 0.2:
            self.current_condition = WeatherCondition.DAMP
            self.params.rain_intensity = 0.1 + 0.3 * np.random.uniform(0, 1)
        else:
            self.current_condition = WeatherCondition.DRY
            self.params.rain_intensity = 0.0
    
    def _update_track_temperature(self):
        """Update track temperature based on session progression and weather."""
        base_temp = self.params.track_temperature_base
        
        # Session progression effect (track heats up during session)
        session_progress = self.session_elapsed_minutes / self.params.session_duration_minutes
        session_heating = 5.0 * session_progress  # Up to 5°C heating
        
        # Solar radiation effect
        solar_effect = 3.0 * self.params.solar_radiation
        
        # Rain cooling effect
        rain_cooling = -8.0 * self.params.rain_intensity
        
        # Wind cooling effect
        wind_cooling = -0.5 * self.params.wind_speed / 10.0  # Wind speed in km/h
        
        # Calculate final track temperature
        self.track_temperature = base_temp + session_heating + solar_effect + rain_cooling + wind_cooling
        self.track_temperature = max(15.0, min(60.0, self.track_temperature))  # Clamp to realistic range
    
    def _update_ambient_temperature(self):
        """Update ambient temperature based on weather conditions."""
        base_temp = self.params.ambient_temperature_base
        
        # Rain cooling effect
        rain_cooling = -3.0 * self.params.rain_intensity
        
        # Cloud cover effect (reduces solar heating)
        cloud_effect = -2.0 * (1.0 - self.params.solar_radiation)
        
        # Calculate final ambient temperature
        self.ambient_temperature = base_temp + rain_cooling + cloud_effect
        self.ambient_temperature = max(10.0, min(40.0, self.ambient_temperature))  # Clamp to realistic range
    
    def _update_rubber_buildup(self, lap_count: int):
        """Update rubber buildup on track surface."""
        if lap_count > 0:
            # Rubber builds up with each lap
            self.rubber_buildup += self.params.rubber_buildup_rate * self.params.track_evolution_factor
            
            # Rain washes away rubber
            if self.current_condition in [WeatherCondition.WET, WeatherCondition.HEAVY_RAIN]:
                wash_rate = 0.01 * self.params.rain_intensity
                self.rubber_buildup = max(0.0, self.rubber_buildup - wash_rate)
            
            # Clamp rubber buildup
            self.rubber_buildup = min(1.0, self.rubber_buildup)
    
    def _calculate_environmental_effects(self):
        """Calculate environmental effects on tire performance."""
        # Cooling factor (affects tire cooling rate)
        self.cooling_factor = 1.0
        
        # Rain increases cooling
        if self.current_condition != WeatherCondition.DRY:
            self.cooling_factor += 0.5 * self.params.rain_intensity
        
        # Wind increases cooling
        self.cooling_factor += 0.1 * self.params.wind_speed / 20.0
        
        # Humidity affects cooling (high humidity = less effective cooling)
        self.cooling_factor *= (1.0 - 0.2 * self.params.humidity)
        
        # Grip factor (affects tire grip)
        self.grip_factor = 1.0
        
        # Rain reduces grip
        if self.current_condition != WeatherCondition.DRY:
            grip_reduction = 0.3 + 0.4 * self.params.rain_intensity
            self.grip_factor = max(0.1, 1.0 - grip_reduction)
        
        # Rubber buildup increases grip (up to a point)
        rubber_grip_boost = 0.1 * self.rubber_buildup
        self.grip_factor = min(1.2, self.grip_factor + rubber_grip_boost)
        
        # Thermal factor (affects thermal behavior)
        self.thermal_factor = 1.0
        
        # Rain reduces thermal generation
        if self.current_condition != WeatherCondition.DRY:
            thermal_reduction = 0.2 + 0.3 * self.params.rain_intensity
            self.thermal_factor = max(0.3, 1.0 - thermal_reduction)
        
        # Humidity affects thermal behavior
        self.thermal_factor *= (1.0 + 0.1 * self.params.humidity)
    
    def _store_weather_history(self):
        """Store current weather state in history."""
        self.weather_history.append({
            'time': self.session_elapsed_minutes,
            'condition': self.current_condition.value,
            'rain_probability': self.params.rain_probability,
            'rain_intensity': self.params.rain_intensity,
            'cooling_factor': self.cooling_factor,
            'grip_factor': self.grip_factor,
            'thermal_factor': self.thermal_factor
        })
        
        self.track_temp_history.append(self.track_temperature)
        self.ambient_temp_history.append(self.ambient_temperature)
        
        # Keep only last 1000 samples
        if len(self.weather_history) > 1000:
            self.weather_history = self.weather_history[-1000:]
            self.track_temp_history = self.track_temp_history[-1000:]
            self.ambient_temp_history = self.ambient_temp_history[-1000:]
    
    def get_weather_summary(self) -> Dict:
        """Get comprehensive weather summary."""
        return {
            'current_condition': self.current_condition.value,
            'rain_probability': self.params.rain_probability,
            'rain_intensity': self.params.rain_intensity,
            'track_temperature': self.track_temperature,
            'ambient_temperature': self.ambient_temperature,
            'rubber_buildup': self.rubber_buildup,
            'cooling_factor': self.cooling_factor,
            'grip_factor': self.grip_factor,
            'thermal_factor': self.thermal_factor,
            'session_elapsed_minutes': self.session_elapsed_minutes,
            'session_progress': self.session_elapsed_minutes / self.params.session_duration_minutes
        }
    
    def get_weather_recommendations(self) -> List[Tuple[str, str]]:
        """Get weather-based recommendations."""
        recommendations = []
        
        # Rain-related recommendations
        if self.params.rain_probability > 0.7:
            recommendations.append(("WEATHER", f"High rain probability ({self.params.rain_probability:.1%}) - consider wet tires"))
        elif self.params.rain_probability > 0.3:
            recommendations.append(("WEATHER", f"Moderate rain risk ({self.params.rain_probability:.1%}) - monitor conditions"))
        
        # Track temperature recommendations
        if self.track_temperature > 50:
            recommendations.append(("TRACK", f"Hot track ({self.track_temperature:.1f}°C) - monitor tire temperatures"))
        elif self.track_temperature < 20:
            recommendations.append(("TRACK", f"Cold track ({self.track_temperature:.1f}°C) - warm-up laps needed"))
        
        # Rubber buildup recommendations
        if self.rubber_buildup > 0.7:
            recommendations.append(("TRACK", f"High rubber buildup ({self.rubber_buildup:.1%}) - expect higher grip"))
        elif self.rubber_buildup < 0.1:
            recommendations.append(("TRACK", f"Low rubber buildup ({self.rubber_buildup:.1%}) - green track conditions"))
        
        # Wind recommendations
        if self.params.wind_speed > 30:
            recommendations.append(("WIND", f"Strong winds ({self.params.wind_speed:.0f} km/h) - affects cooling and stability"))
        
        return recommendations
    
    def set_session_type(self, session_type: SessionType):
        """Set the current session type."""
        self.session_type = session_type
        
        # Adjust parameters based on session type
        if session_type == SessionType.RACE:
            self.params.session_duration_minutes = 120.0  # 2-hour race window
            self.params.track_evolution_factor = 1.5  # More track evolution in race
        elif session_type == SessionType.QUALIFYING:
            self.params.session_duration_minutes = 20.0  # 20-minute qualifying
            self.params.track_evolution_factor = 0.8  # Less track evolution
        else:  # Practice sessions
            self.params.session_duration_minutes = 60.0
            self.params.track_evolution_factor = 1.0
    
    def reset_session(self):
        """Reset weather model for new session."""
        self.session_start_time = time.time()
        self.session_elapsed_minutes = 0.0
        self.rubber_buildup = 0.0
        self.track_temperature = self.params.track_temperature_base
        self.ambient_temperature = self.params.ambient_temperature_base
        self.weather_history = []
        self.track_temp_history = []
        self.ambient_temp_history = []
        self.current_condition = WeatherCondition.DRY
        self.params.rain_probability = 0.0
        self.params.rain_intensity = 0.0
