"""
Configuration system for F1 Tire Temperature Management System.

This module centralizes all configuration values, replacing hardcoded magic numbers
throughout the codebase with a maintainable configuration system.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
from enum import Enum
import json
from pathlib import Path


class CompoundType(Enum):
    """Tire compound types."""
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class TrackType(Enum):
    """Track types for different configurations."""
    MONACO = "monaco"
    MONZA = "monza"
    SILVERSTONE = "silverstone"
    SPA = "spa"
    BAHRAIN = "bahrain"


@dataclass
class TireConfig:
    """Tire-specific configuration."""
    # Optimal temperature bands by compound
    optimal_bands: Dict[str, Tuple[float, float]]
    
    # Grip degradation thresholds
    grip_degradation_thresholds: Dict[str, float]
    
    # Wear rate multipliers
    wear_rate_multipliers: Dict[str, float]


@dataclass
class TrackConfig:
    """Track-specific configuration."""
    # Track dimensions
    length_meters: float
    width_meters: float
    
    # Typical lap characteristics
    typical_lap_time: float
    typical_speed_kmh: float
    
    # Environmental characteristics
    typical_ambient_temp: float
    typical_track_temp: float
    
    # Sector profiles (normalized 0-1)
    speed_profile: List[float]
    lateral_g_profile: List[float]
    brake_profile: List[float]


@dataclass
class SimulationConfig:
    """Simulation-specific configuration."""
    # Time parameters
    default_time_step: float
    max_simulation_time: float
    
    # Race parameters
    typical_race_laps: int
    pit_lane_time: float
    
    # Safety parameters
    safety_car_probability: float
    red_flag_probability: float
    
    # Weather parameters
    weather_change_probability: float
    rain_intensity_range: Tuple[float, float]
    temperature_variation: float


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    # Neural network parameters
    input_dimensions: int
    hidden_dimensions: List[int]
    output_dimensions: int
    learning_rate: float
    batch_size: int
    epochs: int
    
    # Training parameters
    validation_split: float
    early_stopping_patience: int
    
    # Data parameters
    sequence_length: int
    prediction_horizon: int


@dataclass
class SystemConfig:
    """Main system configuration."""
    tire_config: TireConfig
    track_config: TrackConfig
    simulation_config: SimulationConfig
    ml_config: MLConfig
    
    # Performance parameters
    max_concurrent_simulations: int
    simulation_timeout: int
    cache_results: bool
    
    # Data retention
    max_history_points: int
    data_retention_days: int


# Default configurations
DEFAULT_TIRE_CONFIG = TireConfig(
    optimal_bands={
        "soft": (95.0, 110.0),
        "medium": (90.0, 106.0),
        "hard": (88.0, 104.0)
    },
    grip_degradation_thresholds={
        "soft": 0.3,
        "medium": 0.4,
        "hard": 0.5
    },
    wear_rate_multipliers={
        "soft": 1.2,
        "medium": 1.0,
        "hard": 0.8
    }
)

DEFAULT_TRACK_CONFIG = TrackConfig(
    length_meters=5793,  # Monaco length
    width_meters=12.0,
    typical_lap_time=83.0,
    typical_speed_kmh=180.0,
    typical_ambient_temp=27.0,
    typical_track_temp=39.0,
    speed_profile=[310, 280, 250, 220, 200, 180, 160, 140, 120, 100],
    lateral_g_profile=[3.8, 3.2, 2.8, 2.4, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0],
    brake_profile=[0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
)

DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    default_time_step=0.2,
    max_simulation_time=7200.0,  # 2 hours
    typical_race_laps=58,
    pit_lane_time=20.0,
    safety_car_probability=0.1,
    red_flag_probability=0.02,
    weather_change_probability=0.05,
    rain_intensity_range=(0.1, 1.0),
    temperature_variation=5.0
)

DEFAULT_ML_CONFIG = MLConfig(
    input_dimensions=15,
    hidden_dimensions=[64, 32, 16],
    output_dimensions=5,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    early_stopping_patience=10,
    sequence_length=10,
    prediction_horizon=5
)

DEFAULT_SYSTEM_CONFIG = SystemConfig(
    tire_config=DEFAULT_TIRE_CONFIG,
    track_config=DEFAULT_TRACK_CONFIG,
    simulation_config=DEFAULT_SIMULATION_CONFIG,
    ml_config=DEFAULT_ML_CONFIG,
    max_concurrent_simulations=10,
    simulation_timeout=300,
    cache_results=True,
    max_history_points=1000,
    data_retention_days=30
)


class ConfigManager:
    """Configuration manager for loading and saving configurations."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = DEFAULT_SYSTEM_CONFIG
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                # Convert dict back to SystemConfig
                self.config = self._dict_to_config(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
        return self.config
    
    def save_config(self, config: SystemConfig = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            config_dict = self._config_to_dict(config)
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _config_to_dict(self, config: SystemConfig) -> dict:
        """Convert SystemConfig to dictionary for JSON serialization."""
        return {
            "tire_config": {
                "optimal_bands": config.tire_config.optimal_bands,
                "grip_degradation_thresholds": config.tire_config.grip_degradation_thresholds,
                "wear_rate_multipliers": config.tire_config.wear_rate_multipliers
            },
            "track_config": {
                "length_meters": config.track_config.length_meters,
                "width_meters": config.track_config.width_meters,
                "typical_lap_time": config.track_config.typical_lap_time,
                "typical_speed_kmh": config.track_config.typical_speed_kmh,
                "typical_ambient_temp": config.track_config.typical_ambient_temp,
                "typical_track_temp": config.track_config.typical_track_temp,
                "speed_profile": config.track_config.speed_profile,
                "lateral_g_profile": config.track_config.lateral_g_profile,
                "brake_profile": config.track_config.brake_profile
            },
            "simulation_config": {
                "default_time_step": config.simulation_config.default_time_step,
                "max_simulation_time": config.simulation_config.max_simulation_time,
                "typical_race_laps": config.simulation_config.typical_race_laps,
                "pit_lane_time": config.simulation_config.pit_lane_time,
                "safety_car_probability": config.simulation_config.safety_car_probability,
                "red_flag_probability": config.simulation_config.red_flag_probability,
                "weather_change_probability": config.simulation_config.weather_change_probability,
                "rain_intensity_range": config.simulation_config.rain_intensity_range,
                "temperature_variation": config.simulation_config.temperature_variation
            },
            "ml_config": {
                "input_dimensions": config.ml_config.input_dimensions,
                "hidden_dimensions": config.ml_config.hidden_dimensions,
                "output_dimensions": config.ml_config.output_dimensions,
                "learning_rate": config.ml_config.learning_rate,
                "batch_size": config.ml_config.batch_size,
                "epochs": config.ml_config.epochs,
                "validation_split": config.ml_config.validation_split,
                "early_stopping_patience": config.ml_config.early_stopping_patience,
                "sequence_length": config.ml_config.sequence_length,
                "prediction_horizon": config.ml_config.prediction_horizon
            },
            "max_concurrent_simulations": config.max_concurrent_simulations,
            "simulation_timeout": config.simulation_timeout,
            "cache_results": config.cache_results,
            "max_history_points": config.max_history_points,
            "data_retention_days": config.data_retention_days
        }
    
    def _dict_to_config(self, config_dict: dict) -> SystemConfig:
        """Convert dictionary to SystemConfig."""
        return SystemConfig(
            tire_config=TireConfig(**config_dict["tire_config"]),
            track_config=TrackConfig(**config_dict["track_config"]),
            simulation_config=SimulationConfig(**config_dict["simulation_config"]),
            ml_config=MLConfig(**config_dict["ml_config"]),
            max_concurrent_simulations=config_dict["max_concurrent_simulations"],
            simulation_timeout=config_dict["simulation_timeout"],
            cache_results=config_dict["cache_results"],
            max_history_points=config_dict["max_history_points"],
            data_retention_days=config_dict["data_retention_days"]
        )
    
    def get_tire_config(self) -> TireConfig:
        """Get tire configuration."""
        return self.config.tire_config
    
    def get_track_config(self) -> TrackConfig:
        """Get track configuration."""
        return self.config.track_config
    
    def get_simulation_config(self) -> SimulationConfig:
        """Get simulation configuration."""
        return self.config.simulation_config
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration."""
        return self.config.ml_config


# Global configuration instance
config_manager = ConfigManager()
system_config = config_manager.load_config()
