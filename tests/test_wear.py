import numpy as np
import pytest
from wear import TireWearModel, WearParams
from thermal import ThermalModel, ThermalParams
from decision import DecisionEngine, GRIP_DEGRADATION_THRESHOLDS

def test_wear_model_initialization():
    """Test wear model initializes correctly with default parameters."""
    wear_model = TireWearModel()
    assert len(wear_model.wear_levels) == 4
    assert all(wear_level == 0.0 for wear_level in wear_model.wear_levels.values())
    assert all(corner in wear_model.wear_levels for corner in ["FL", "FR", "RL", "RR"])

def test_wear_params_initialization():
    """Test wear parameters initialize with correct defaults."""
    params = WearParams()
    assert params.thermal_wear_rate == 0.001
    assert params.thermal_threshold == 105.0
    assert params.load_wear_base == 0.0001
    assert params.compound_wear_multipliers["soft"] == 1.5
    assert params.compound_wear_multipliers["medium"] == 1.0
    assert params.compound_wear_multipliers["hard"] == 0.7

def test_wear_update_basic():
    """Test basic wear update functionality."""
    wear_model = TireWearModel()
    thermal_state = np.array([100.0, 95.0, 90.0])
    
    # Update wear for FL corner
    wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    # Check that wear level increased
    assert wear_model.wear_levels["FL"] > 0.0
    assert wear_model.wear_levels["FL"] <= 1.0  # Should be clamped
    
    # Check that history was stored
    assert len(wear_model.thermal_history["FL"]) == 1
    assert len(wear_model.load_history["FL"]) == 1
    assert len(wear_model.slip_history["FL"]) == 1

def test_wear_update_thermal_effects():
    """Test that high temperatures increase wear rate."""
    wear_model = TireWearModel()
    
    # Low temperature scenario
    low_temp_state = np.array([95.0, 90.0, 85.0])
    wear_model.update_wear("FL", low_temp_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    low_wear = wear_model.wear_levels["FL"]
    
    # Reset and test high temperature scenario
    wear_model.reset_wear("FL")
    high_temp_state = np.array([110.0, 105.0, 100.0])  # Above thermal threshold
    wear_model.update_wear("FL", high_temp_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    high_wear = wear_model.wear_levels["FL"]
    
    # High temperature should cause more wear
    assert high_wear > low_wear

def test_wear_update_compound_effects():
    """Test that different compounds have different wear rates."""
    wear_model = TireWearModel()
    thermal_state = np.array([100.0, 95.0, 90.0])
    
    # Test soft compound (should wear fastest)
    wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "soft", 0.2)
    soft_wear = wear_model.wear_levels["FL"]
    
    # Reset and test hard compound
    wear_model.reset_wear("FL")
    wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "hard", 0.2)
    hard_wear = wear_model.wear_levels["FL"]
    
    # Soft should wear faster than hard
    assert soft_wear > hard_wear

def test_wear_effects_calculation():
    """Test wear effects on tire properties."""
    wear_model = TireWearModel()
    
    # Set some wear level
    wear_model.wear_levels["FL"] = 0.5  # 50% wear
    
    effects = wear_model.get_wear_effects("FL")
    
    # Check that effects are calculated correctly
    assert effects["wear_level"] == 0.5
    assert effects["thermal_conductivity_factor"] > 1.0  # Worn tires conduct heat better
    assert effects["thermal_capacity_factor"] < 1.0     # Worn tires have less thermal mass
    assert effects["grip_factor"] < 1.0                 # Worn tires have less grip
    assert effects["stiffness_factor"] < 1.0            # Worn tires are less stiff

def test_wear_reset():
    """Test wear reset functionality."""
    wear_model = TireWearModel()
    
    # Add some wear
    thermal_state = np.array([100.0, 95.0, 90.0])
    wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    assert wear_model.wear_levels["FL"] > 0.0
    
    # Reset specific corner
    wear_model.reset_wear("FL")
    assert wear_model.wear_levels["FL"] == 0.0
    assert len(wear_model.thermal_history["FL"]) == 0
    
    # Add wear to all corners
    for corner in ["FL", "FR", "RL", "RR"]:
        wear_model.update_wear(corner, thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    # Reset all corners
    wear_model.reset_wear()
    assert all(wear_level == 0.0 for wear_level in wear_model.wear_levels.values())

def test_wear_summary():
    """Test wear summary generation."""
    wear_model = TireWearModel()
    
    # Add wear to all corners
    thermal_state = np.array([100.0, 95.0, 90.0])
    for corner in ["FL", "FR", "RL", "RR"]:
        wear_model.update_wear(corner, thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    summary = wear_model.get_wear_summary()
    
    # Check that summary contains all corners
    assert len(summary) == 4
    assert all(corner in summary for corner in ["FL", "FR", "RL", "RR"])
    
    # Check that each corner has required keys
    for corner_data in summary.values():
        required_keys = ["wear_level", "grip_factor", "thermal_conductivity_factor", 
                        "thermal_capacity_factor", "stiffness_factor"]
        assert all(key in corner_data for key in required_keys)

def test_grip_degradation_calculation():
    """Test grip degradation calculation."""
    wear_model = TireWearModel()
    
    # Test different wear levels
    test_cases = [
        (0.0, 1.0),    # No wear = full grip
        (0.2, 0.92),   # 20% wear = 92% grip
        (0.5, 0.8),    # 50% wear = 80% grip
        (1.0, 0.6),    # 100% wear = 60% grip
    ]
    
    for wear_level, expected_grip in test_cases:
        wear_model.wear_levels["FL"] = wear_level
        grip_factor = wear_model.get_grip_degradation("FL")
        assert abs(grip_factor - expected_grip) < 0.01

def test_wear_prediction():
    """Test wear prediction functionality."""
    wear_model = TireWearModel()
    
    # Simulate some wear over multiple updates
    thermal_state = np.array([100.0, 95.0, 90.0])
    for _ in range(10):
        wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    # Predict remaining laps
    remaining_laps = wear_model.predict_wear_remaining("FL", 10, 50)
    
    # Should return a reasonable prediction
    assert isinstance(remaining_laps, float)
    assert remaining_laps >= 0
    assert remaining_laps <= 40  # Should be less than total remaining laps

def test_decision_engine_wear_integration():
    """Test decision engine integration with wear model."""
    wear_model = TireWearModel()
    engine = DecisionEngine("medium", wear_model)
    
    # Set up some wear
    thermal_state = np.array([100.0, 95.0, 90.0])
    wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    # Generate recommendations with wear data
    est_by_corner = {"FL": thermal_state}
    wear_summary = wear_model.get_wear_summary()
    
    actions = engine.actions(est_by_corner, wear_summary)
    
    # Should generate some recommendations
    assert isinstance(actions, list)
    
    # Check for wear-specific recommendations
    wear_recs = engine.get_wear_recommendations(wear_summary)
    assert isinstance(wear_recs, list)

def test_pit_window_prediction():
    """Test pit window prediction functionality."""
    wear_model = TireWearModel()
    engine = DecisionEngine("medium", wear_model)
    
    # Set up moderate wear
    thermal_state = np.array([100.0, 95.0, 90.0])
    for _ in range(5):
        wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    wear_summary = wear_model.get_wear_summary()
    pit_predictions = engine.predict_pit_window(wear_summary, 5, 50)
    
    # Should have predictions for all corners
    assert len(pit_predictions) == 4
    assert all(corner in pit_predictions for corner in ["FL", "FR", "RL", "RR"])
    
    # Each prediction should have required keys
    for pred in pit_predictions.values():
        assert "recommended_pit_lap" in pred
        assert "laps_to_critical" in pred
        assert "urgency" in pred
        assert pred["urgency"] in ["HIGH", "MEDIUM", "LOW"]

def test_grip_degradation_thresholds():
    """Test grip degradation thresholds for different compounds."""
    assert GRIP_DEGRADATION_THRESHOLDS["soft"] == 0.3
    assert GRIP_DEGRADATION_THRESHOLDS["medium"] == 0.4
    assert GRIP_DEGRADATION_THRESHOLDS["hard"] == 0.5
    
    # Soft tires should degrade earliest, hard tires latest
    assert GRIP_DEGRADATION_THRESHOLDS["soft"] < GRIP_DEGRADATION_THRESHOLDS["medium"]
    assert GRIP_DEGRADATION_THRESHOLDS["medium"] < GRIP_DEGRADATION_THRESHOLDS["hard"]

def test_wear_history_management():
    """Test wear history management and trimming."""
    wear_model = TireWearModel()
    
    # Add more samples than max_history
    thermal_state = np.array([100.0, 95.0, 90.0])
    for i in range(150):  # More than max_history (100)
        wear_model.update_wear("FL", thermal_state, 4000.0, 0.05, 2.0, "medium", 0.2)
    
    # History should be trimmed to max_history
    assert len(wear_model.thermal_history["FL"]) == wear_model.max_history
    assert len(wear_model.load_history["FL"]) == wear_model.max_history
    assert len(wear_model.slip_history["FL"]) == wear_model.max_history

def test_wear_clamping():
    """Test that wear levels are properly clamped between 0 and 1."""
    wear_model = TireWearModel()
    
    # Force extreme wear conditions
    thermal_state = np.array([120.0, 115.0, 110.0])  # Very high temps
    for _ in range(100):  # Many updates
        wear_model.update_wear("FL", thermal_state, 6000.0, 0.2, 10.0, "soft", 0.2)
    
    # Wear should be clamped to 1.0
    assert wear_model.wear_levels["FL"] <= 1.0
    
    # Reset and test negative wear (shouldn't happen but test clamping)
    wear_model.wear_levels["FL"] = -0.1
    effects = wear_model.get_wear_effects("FL")
    assert effects["wear_level"] >= 0.0
