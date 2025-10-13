import numpy as np
import pytest
from driver import DriverProfile, DriverParams, DrivingStyle, DriverExperience
from driver_profiles import DriverProfiles
from decision import DecisionEngine
from wear import TireWearModel, WearParams
from weather import WeatherModel, WeatherParams

def test_driver_profile_initialization():
    """Test driver profile initializes correctly."""
    params = DriverParams()
    driver = DriverProfile("Test Driver", params, DrivingStyle.BALANCED, DriverExperience.EXPERIENCED)
    
    assert driver.name == "Test Driver"
    assert driver.style == DrivingStyle.BALANCED
    assert driver.experience == DriverExperience.EXPERIENCED
    assert driver.params.thermal_aggression == 1.0
    assert len(driver.lap_history) == 0

def test_driver_params_initialization():
    """Test driver parameters initialize with correct defaults."""
    params = DriverParams()
    assert params.thermal_aggression == 1.0
    assert params.thermal_efficiency == 1.0
    assert params.brake_aggression == 1.0
    assert params.throttle_aggression == 1.0
    assert params.tire_awareness == 1.0
    assert params.wet_weather_skill == 1.0

def test_thermal_signature_calculation():
    """Test thermal signature calculation based on driver characteristics."""
    # Aggressive driver
    aggressive_params = DriverParams(
        thermal_aggression=1.4,
        brake_aggression=1.3,
        throttle_aggression=1.4,
        steering_aggression=1.2
    )
    aggressive_driver = DriverProfile("Aggressive", aggressive_params, DrivingStyle.AGGRESSIVE)
    
    # Conservative driver
    conservative_params = DriverParams(
        thermal_aggression=0.7,
        brake_aggression=0.8,
        throttle_aggression=0.7,
        steering_aggression=0.9
    )
    conservative_driver = DriverProfile("Conservative", conservative_params, DrivingStyle.CONSERVATIVE)
    
    # Aggressive driver should have higher thermal generation
    assert aggressive_driver.thermal_signature['thermal_generation'] > conservative_driver.thermal_signature['thermal_generation']

def test_thermal_multipliers():
    """Test thermal multipliers based on conditions."""
    driver = DriverProfile("Test Driver")
    
    # Test temperature adaptation
    conditions = {'track_temperature': 45.0}
    multipliers = driver.get_thermal_multipliers(conditions)
    
    assert 'thermal_generation' in multipliers
    assert 'thermal_efficiency' in multipliers
    assert 'adaptation_capability' in multipliers
    assert all(isinstance(v, float) for v in multipliers.values())

def test_weather_adaptation():
    """Test weather adaptation factors."""
    # Create driver with different wet weather skill
    params = DriverParams(wet_weather_skill=0.5)  # Lower wet weather skill
    driver = DriverProfile("Test Driver", params)
    
    # Dry conditions
    dry_conditions = {'rain_probability': 0.0, 'wind_speed': 0.0}
    dry_multipliers = driver.get_thermal_multipliers(dry_conditions)
    
    # Wet conditions
    wet_conditions = {'rain_probability': 0.8, 'wind_speed': 20.0}
    wet_multipliers = driver.get_thermal_multipliers(wet_conditions)
    
    # Wet conditions should affect thermal behavior
    assert wet_multipliers['weather_factor'] != dry_multipliers['weather_factor']

def test_compound_adaptation():
    """Test compound adaptation factors."""
    driver = DriverProfile("Test Driver")
    
    # Test different compounds
    soft_conditions = {'compound': 'soft'}
    hard_conditions = {'compound': 'hard'}
    
    soft_multipliers = driver.get_thermal_multipliers(soft_conditions)
    hard_multipliers = driver.get_thermal_multipliers(hard_conditions)
    
    # Different compounds should have different adaptation factors
    assert soft_multipliers['compound_factor'] != hard_multipliers['compound_factor']

def test_personalized_recommendations():
    """Test personalized recommendations generation."""
    params = DriverParams()
    driver = DriverProfile("Test Driver", params, DrivingStyle.AGGRESSIVE, DriverExperience.ROOKIE)
    
    thermal_state = np.array([105.0, 100.0, 95.0])  # High temperatures
    conditions = {'rain_probability': 0.3}
    wear_summary = {}
    
    recommendations = driver.get_personalized_recommendations(thermal_state, conditions, wear_summary)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_driver_profiles_initialization():
    """Test driver profiles manager initializes correctly."""
    profiles = DriverProfiles()
    
    assert len(profiles.drivers) > 0
    assert profiles.active_driver is not None
    assert profiles.active_driver in profiles.drivers

def test_driver_profiles_add_driver():
    """Test adding new driver profiles."""
    profiles = DriverProfiles()
    initial_count = len(profiles.drivers)
    
    params = DriverParams(thermal_aggression=1.2)
    profiles.add_driver("New Driver", params, DrivingStyle.BALANCED, DriverExperience.EXPERIENCED)
    
    assert len(profiles.drivers) == initial_count + 1
    assert "New Driver" in profiles.drivers

def test_driver_profiles_get_driver():
    """Test getting driver profiles."""
    profiles = DriverProfiles()
    
    driver = profiles.get_driver("Max Verstappen")
    assert driver is not None
    assert driver.name == "Max Verstappen"
    
    # Test non-existent driver
    non_existent = profiles.get_driver("Non Existent")
    assert non_existent is None

def test_driver_profiles_set_active():
    """Test setting active driver."""
    profiles = DriverProfiles()
    
    profiles.set_active_driver("Lewis Hamilton")
    assert profiles.active_driver == "Lewis Hamilton"
    
    active_driver = profiles.get_active_driver()
    assert active_driver.name == "Lewis Hamilton"

def test_driver_comparison():
    """Test driver comparison functionality."""
    profiles = DriverProfiles()
    
    comparison = profiles.compare_drivers()
    
    assert isinstance(comparison, dict)
    assert len(comparison) > 0
    
    # Check that each driver has required keys
    for driver_name, data in comparison.items():
        required_keys = ['style', 'experience', 'thermal_signature', 'thermal_consistency']
        assert all(key in data for key in required_keys)

def test_driver_rankings():
    """Test driver rankings functionality."""
    profiles = DriverProfiles()
    
    rankings = profiles.get_driver_rankings('thermal_consistency')
    
    assert isinstance(rankings, list)
    assert len(rankings) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in rankings)
    
    # Rankings should be sorted (descending)
    values = [item[1] for item in rankings]
    assert values == sorted(values, reverse=True)

def test_personalized_recommendations_integration():
    """Test personalized recommendations integration."""
    profiles = DriverProfiles()
    
    thermal_state = np.array([100.0, 95.0, 90.0])
    conditions = {'rain_probability': 0.5}
    wear_summary = {}
    
    recommendations = profiles.get_personalized_recommendations(
        "Max Verstappen", thermal_state, conditions, wear_summary
    )
    
    assert isinstance(recommendations, list)

def test_multi_driver_race_simulation():
    """Test multi-driver race simulation."""
    profiles = DriverProfiles()
    
    conditions = {'rain_probability': 0.2, 'track_temperature': 35.0}
    simulation_results = profiles.simulate_multi_driver_race(conditions, laps=3)
    
    assert isinstance(simulation_results, dict)
    assert len(simulation_results) > 0
    
    # Check that each driver has simulation results
    for driver_name, results in simulation_results.items():
        assert 'lap_times' in results
        assert 'thermal_states' in results
        assert 'recommendations' in results
        assert len(results['lap_times']) == 3
        assert len(results['thermal_states']) == 3

def test_race_strategy_recommendations():
    """Test race strategy recommendations."""
    profiles = DriverProfiles()
    
    conditions = {'rain_probability': 0.6}
    strategy_recs = profiles.get_race_strategy_recommendations("Max Verstappen", conditions)
    
    assert isinstance(strategy_recs, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in strategy_recs)

def test_driver_development_insights():
    """Test driver development insights."""
    profiles = DriverProfiles()
    
    insights = profiles.get_driver_development_insights("Max Verstappen")
    
    assert isinstance(insights, list)
    assert all(isinstance(insight, tuple) and len(insight) == 2 for insight in insights)

def test_driver_session_data_update():
    """Test driver session data update."""
    driver = DriverProfile("Test Driver")
    
    thermal_state = np.array([100.0, 95.0, 90.0])
    lap_time = 85.5
    conditions = {'track_temperature': 35.0}
    recommendations = [("DRIVING", "Test recommendation")]
    
    driver.update_session_data(thermal_state, lap_time, conditions, recommendations)
    
    assert len(driver.lap_history) == 1
    assert len(driver.thermal_history) == 1
    assert len(driver.recommendation_history) == 1
    assert driver.session_laps == 1

def test_driver_summary():
    """Test driver summary generation."""
    driver = DriverProfile("Test Driver")
    
    # Add some session data
    for i in range(5):
        thermal_state = np.array([100.0 + i, 95.0 + i, 90.0 + i])
        driver.update_session_data(thermal_state, 85.0 + i, {}, [])
    
    summary = driver.get_driver_summary()
    
    required_keys = ['name', 'style', 'experience', 'thermal_signature', 'session_laps']
    assert all(key in summary for key in required_keys)
    assert summary['name'] == "Test Driver"
    assert summary['session_laps'] == 5

def test_driver_reset_session():
    """Test driver session reset."""
    driver = DriverProfile("Test Driver")
    
    # Add some data
    driver.update_session_data(np.array([100.0, 95.0, 90.0]), 85.0, {}, [])
    assert driver.session_laps == 1
    
    # Reset session
    driver.reset_session()
    assert driver.session_laps == 0
    assert len(driver.lap_history) == 0
    assert len(driver.thermal_history) == 0

def test_driver_thermal_consistency():
    """Test thermal consistency calculation."""
    driver = DriverProfile("Test Driver")
    
    # Add consistent thermal data
    for i in range(10):
        thermal_state = np.array([100.0, 95.0, 90.0])  # Consistent temperatures
        driver.update_session_data(thermal_state, 85.0, {}, [])
    
    summary = driver.get_driver_summary()
    consistency = summary['thermal_consistency']
    
    # Should have high consistency
    assert consistency > 0.8

def test_driver_recommendation_follow_rate():
    """Test recommendation follow rate calculation."""
    driver = DriverProfile("Test Driver", experience=DriverExperience.CHAMPION)
    
    summary = driver.get_driver_summary()
    follow_rate = summary['recommendation_follow_rate']
    
    # Champion should have high follow rate
    assert follow_rate > 0.8

def test_decision_engine_driver_integration():
    """Test decision engine integration with driver profiles."""
    profiles = DriverProfiles()
    wear_model = TireWearModel()
    engine = DecisionEngine("medium", wear_model, profiles, "Max Verstappen")
    
    # Use extreme thermal conditions to trigger driver-specific recommendations
    est_by_corner = {"FL": np.array([120.0, 115.0, 110.0])}  # Very high temperatures
    wear_summary = {}
    weather_summary = {'rain_probability': 0.8}  # Wet conditions
    
    actions = engine.actions(est_by_corner, wear_summary, weather_summary)
    
    assert isinstance(actions, list)
    # Should include driver-specific recommendations (may be empty if conditions are normal)
    # Just check that the method works without errors

def test_driver_specific_temperature_bands():
    """Test driver-specific temperature bands."""
    profiles = DriverProfiles()
    engine = DecisionEngine("medium", None, profiles, "Max Verstappen")
    
    driver_bands = engine._get_driver_specific_bands()
    
    assert isinstance(driver_bands, dict)
    assert all(corner in driver_bands for corner in ["FL", "FR", "RL", "RR"])
    assert all(isinstance(band, tuple) and len(band) == 2 for band in driver_bands.values())

def test_driving_style_enum():
    """Test driving style enum values."""
    assert DrivingStyle.AGGRESSIVE.value == "aggressive"
    assert DrivingStyle.CONSERVATIVE.value == "conservative"
    assert DrivingStyle.BALANCED.value == "balanced"
    assert DrivingStyle.SMOOTH.value == "smooth"
    assert DrivingStyle.BRAVE.value == "brave"

def test_driver_experience_enum():
    """Test driver experience enum values."""
    assert DriverExperience.ROOKIE.value == "rookie"
    assert DriverExperience.EXPERIENCED.value == "experienced"
    assert DriverExperience.VETERAN.value == "veteran"
    assert DriverExperience.CHAMPION.value == "champion"

def test_driver_statistics():
    """Test driver statistics generation."""
    profiles = DriverProfiles()
    
    stats = profiles.get_driver_statistics()
    
    assert isinstance(stats, dict)
    assert len(stats) > 0
    
    # Check that each driver has required statistics
    for driver_name, driver_stats in stats.items():
        assert 'basic_info' in driver_stats
        assert 'performance' in driver_stats
        assert 'characteristics' in driver_stats
        assert 'session_data' in driver_stats
