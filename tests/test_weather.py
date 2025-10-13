import numpy as np
import pytest
from weather import WeatherModel, WeatherParams, WeatherCondition, SessionType
from thermal import ThermalModel, ThermalParams
from simulator import TelemetrySim
from wear import TireWearModel, WearParams

def test_weather_model_initialization():
    """Test weather model initializes correctly with default parameters."""
    weather_model = WeatherModel()
    assert weather_model.current_condition == WeatherCondition.DRY
    assert weather_model.session_type == SessionType.FP1
    assert weather_model.session_elapsed_minutes == 0.0
    assert weather_model.rubber_buildup == 0.0
    assert len(weather_model.weather_history) == 0

def test_weather_params_initialization():
    """Test weather parameters initialize with correct defaults."""
    params = WeatherParams()
    assert params.rain_probability == 0.0
    assert params.rain_intensity == 0.0
    assert params.humidity == 0.5
    assert params.wind_speed == 0.0
    assert params.track_temperature_base == 35.0
    assert params.ambient_temperature_base == 25.0

def test_weather_update_basic():
    """Test basic weather update functionality."""
    weather_model = WeatherModel()
    
    # Update weather for a few time steps
    for _ in range(10):
        weather_model.update_weather(0.2, 0)
    
    # Check that time has progressed
    assert weather_model.session_elapsed_minutes > 0.0
    
    # Check that history was stored
    assert len(weather_model.weather_history) > 0
    assert len(weather_model.track_temp_history) > 0
    assert len(weather_model.ambient_temp_history) > 0

def test_rain_probability_effects():
    """Test that rain probability affects weather conditions."""
    weather_model = WeatherModel()
    
    # Set high rain probability
    weather_model.params.rain_probability = 0.8
    weather_model.update_weather(0.2, 0)
    
    # Should be wet or heavy rain
    assert weather_model.current_condition in [WeatherCondition.WET, WeatherCondition.HEAVY_RAIN]
    assert weather_model.params.rain_intensity > 0.0

def test_track_temperature_evolution():
    """Test track temperature evolution during session."""
    weather_model = WeatherModel()
    
    # Simulate session progression
    for i in range(50):
        weather_model.update_weather(0.2, i)
    
    # Track temperature should evolve (could go up or down depending on conditions)
    assert len(weather_model.track_temp_history) == 50
    assert all(15.0 <= temp <= 60.0 for temp in weather_model.track_temp_history)

def test_ambient_temperature_evolution():
    """Test ambient temperature evolution."""
    weather_model = WeatherModel()
    
    # Simulate with rain (should cool ambient temperature)
    weather_model.params.rain_probability = 0.6
    weather_model.params.rain_intensity = 0.5
    
    for _ in range(20):
        weather_model.update_weather(0.2, 0)
    
    # Ambient temperature should be affected by rain
    assert len(weather_model.ambient_temp_history) == 20
    assert all(10.0 <= temp <= 40.0 for temp in weather_model.ambient_temp_history)

def test_rubber_buildup():
    """Test rubber buildup on track surface."""
    weather_model = WeatherModel()
    
    # Simulate laps (rubber should build up)
    for lap in range(20):
        weather_model.update_weather(0.2, lap)
    
    # Rubber should have built up
    assert weather_model.rubber_buildup > 0.0
    assert weather_model.rubber_buildup <= 1.0

def test_rain_washes_rubber():
    """Test that rain washes away rubber buildup."""
    weather_model = WeatherModel()
    
    # Build up some rubber first
    for lap in range(10):
        weather_model.update_weather(0.2, lap)
    
    initial_rubber = weather_model.rubber_buildup
    assert initial_rubber > 0.0
    
    # Add rain
    weather_model.params.rain_probability = 0.8
    weather_model.params.rain_intensity = 0.7
    
    # Continue with rain
    for lap in range(10, 20):
        weather_model.update_weather(0.2, lap)
    
    # Rubber should be reduced by rain
    assert weather_model.rubber_buildup < initial_rubber

def test_environmental_effects():
    """Test environmental effects on tire performance."""
    weather_model = WeatherModel()
    
    # Test dry conditions
    weather_model.update_weather(0.2, 0)
    dry_cooling = weather_model.cooling_factor
    dry_grip = weather_model.grip_factor
    dry_thermal = weather_model.thermal_factor
    
    # Test wet conditions
    weather_model.params.rain_probability = 0.8
    weather_model.params.rain_intensity = 0.6
    weather_model.update_weather(0.2, 0)
    wet_cooling = weather_model.cooling_factor
    wet_grip = weather_model.grip_factor
    wet_thermal = weather_model.thermal_factor
    
    # Wet conditions should increase cooling, reduce grip and thermal generation
    assert wet_cooling > dry_cooling
    assert wet_grip < dry_grip
    assert wet_thermal < dry_thermal

def test_session_type_effects():
    """Test different session types affect parameters."""
    weather_model = WeatherModel()
    
    # Test race session
    weather_model.set_session_type(SessionType.RACE)
    assert weather_model.session_type == SessionType.RACE
    assert weather_model.params.session_duration_minutes == 120.0
    assert weather_model.params.track_evolution_factor == 1.5
    
    # Test qualifying session
    weather_model.set_session_type(SessionType.QUALIFYING)
    assert weather_model.session_type == SessionType.QUALIFYING
    assert weather_model.params.session_duration_minutes == 20.0
    assert weather_model.params.track_evolution_factor == 0.8

def test_weather_summary():
    """Test weather summary generation."""
    weather_model = WeatherModel()
    weather_model.update_weather(0.2, 5)
    
    summary = weather_model.get_weather_summary()
    
    # Check that summary contains all required keys
    required_keys = [
        'current_condition', 'rain_probability', 'rain_intensity',
        'track_temperature', 'ambient_temperature', 'rubber_buildup',
        'cooling_factor', 'grip_factor', 'thermal_factor',
        'session_elapsed_minutes', 'session_progress'
    ]
    assert all(key in summary for key in required_keys)
    
    # Check value ranges
    assert 0.0 <= summary['rain_probability'] <= 1.0
    assert 0.0 <= summary['rain_intensity'] <= 1.0
    assert 15.0 <= summary['track_temperature'] <= 60.0
    assert 10.0 <= summary['ambient_temperature'] <= 40.0
    assert 0.0 <= summary['rubber_buildup'] <= 1.0

def test_weather_recommendations():
    """Test weather-based recommendations."""
    weather_model = WeatherModel()
    
    # Test high rain probability
    weather_model.params.rain_probability = 0.8
    weather_model.update_weather(0.2, 0)
    
    recommendations = weather_model.get_weather_recommendations()
    assert len(recommendations) > 0
    
    # Should have rain-related recommendations
    rain_recs = [rec for _, rec in recommendations if "rain" in rec.lower()]
    assert len(rain_recs) > 0

def test_weather_history_management():
    """Test weather history management and trimming."""
    weather_model = WeatherModel()
    
    # Add more samples than max history
    for i in range(1200):  # More than max_history (1000)
        weather_model.update_weather(0.2, i)
    
    # History should be trimmed to max_history
    assert len(weather_model.weather_history) <= 1000
    assert len(weather_model.track_temp_history) <= 1000
    assert len(weather_model.ambient_temp_history) <= 1000

def test_reset_session():
    """Test session reset functionality."""
    weather_model = WeatherModel()
    
    # Add some data
    for i in range(10):
        weather_model.update_weather(0.2, i)
    
    assert weather_model.session_elapsed_minutes > 0.0
    assert len(weather_model.weather_history) > 0
    
    # Reset session
    weather_model.reset_session()
    
    assert weather_model.session_elapsed_minutes == 0.0
    assert len(weather_model.weather_history) == 0
    assert weather_model.rubber_buildup == 0.0
    assert weather_model.current_condition == WeatherCondition.DRY

def test_thermal_model_weather_integration():
    """Test thermal model integration with weather."""
    weather_model = WeatherModel()
    thermal_model = ThermalModel(ThermalParams(), weather_model=weather_model)
    
    # Set wet conditions
    weather_model.params.rain_probability = 0.7
    weather_model.params.rain_intensity = 0.5
    weather_model.update_weather(0.2, 0)
    
    # Test thermal step with weather effects
    x = np.array([100.0, 95.0, 90.0])
    u = dict(slip=0.05, slip_ang=3.0, load=4000.0, speed_kmh=180.0, brake=0.2)
    
    result = thermal_model.step(x, u, 0.2, "FL", "medium")
    
    # Should return valid thermal state
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.isfinite(result).all()

def test_simulator_weather_integration():
    """Test simulator integration with weather."""
    weather_model = WeatherModel()
    wear_model = TireWearModel()
    sim = TelemetrySim(wear_model=wear_model, weather_model=weather_model)
    
    # Set weather conditions
    weather_model.params.rain_probability = 0.6
    weather_model.params.wind_speed = 25.0
    weather_model.params.humidity = 0.8
    
    # Run simulation step
    u_common, loads, sensors = sim.step(0.2)
    
    # Check that weather affected simulation
    assert 'speed_kmh' in u_common
    assert 'Ta' in u_common
    assert 'Ttrack' in u_common
    
    # Weather should have updated ambient and track temperatures
    weather_summary = weather_model.get_weather_summary()
    assert u_common['Ta'] == weather_summary['ambient_temperature']
    assert u_common['Ttrack'] == weather_summary['track_temperature']

def test_weather_condition_enum():
    """Test weather condition enum values."""
    assert WeatherCondition.DRY.value == "dry"
    assert WeatherCondition.DAMP.value == "damp"
    assert WeatherCondition.WET.value == "wet"
    assert WeatherCondition.HEAVY_RAIN.value == "heavy_rain"

def test_session_type_enum():
    """Test session type enum values."""
    assert SessionType.FP1.value == "fp1"
    assert SessionType.FP2.value == "fp2"
    assert SessionType.FP3.value == "fp3"
    assert SessionType.QUALIFYING.value == "qualifying"
    assert SessionType.RACE.value == "race"

def test_wind_effects():
    """Test wind effects on cooling and thermal behavior."""
    weather_model = WeatherModel()
    
    # Test no wind
    weather_model.params.wind_speed = 0.0
    weather_model.update_weather(0.2, 0)
    no_wind_cooling = weather_model.cooling_factor
    
    # Test strong wind
    weather_model.params.wind_speed = 40.0
    weather_model.update_weather(0.2, 0)
    strong_wind_cooling = weather_model.cooling_factor
    
    # Strong wind should increase cooling
    assert strong_wind_cooling > no_wind_cooling

def test_humidity_effects():
    """Test humidity effects on thermal behavior."""
    weather_model = WeatherModel()
    
    # Test low humidity
    weather_model.params.humidity = 0.2
    weather_model.update_weather(0.2, 0)
    low_humidity_cooling = weather_model.cooling_factor
    low_humidity_thermal = weather_model.thermal_factor
    
    # Test high humidity
    weather_model.params.humidity = 0.9
    weather_model.update_weather(0.2, 0)
    high_humidity_cooling = weather_model.cooling_factor
    high_humidity_thermal = weather_model.thermal_factor
    
    # High humidity should reduce cooling effectiveness and increase thermal factor
    assert high_humidity_cooling < low_humidity_cooling
    assert high_humidity_thermal > low_humidity_thermal
