import numpy as np
import pytest
from ml_strategy import MLStrategyOptimizer, MLStrategyParams, StrategyAction, RacePhase
from ml_degradation import MLDegradationPredictor, MLDegradationParams, DegradationType, PredictionHorizon
from ml_driver_profiling import MLDriverProfiler, MLDriverProfilingParams, DriverBehaviorPattern, LearningPhase
from ml_recommendations import MLRecommendationEngine, MLRecommendationParams, RecommendationType, RecommendationPriority
from ml_patterns import MLPatternRecognizer, MLPatternRecognitionParams, PatternType, PatternComplexity

def test_ml_strategy_optimizer_initialization():
    """Test ML strategy optimizer initialization."""
    params = MLStrategyParams()
    optimizer = MLStrategyOptimizer(params)
    
    assert optimizer.p.learning_rate == 0.01
    assert optimizer.p.discount_factor == 0.95
    assert optimizer.current_exploration_rate == 0.1
    assert optimizer.training_step == 0

def test_ml_strategy_state_representation():
    """Test state representation generation."""
    optimizer = MLStrategyOptimizer()
    
    thermal_state = np.array([100.0, 95.0, 90.0])
    wear_summary = {'FL': {'wear_level': 0.2}, 'FR': {'wear_level': 0.3}}
    weather_summary = {'rain_probability': 0.1}
    race_context = {'current_lap': 15, 'position': 3}
    
    state = optimizer.get_state_representation(thermal_state, wear_summary, weather_summary, race_context)
    
    assert isinstance(state, str)
    assert 'T' in state and 'W' in state

def test_ml_strategy_action_selection():
    """Test action selection using epsilon-greedy policy."""
    optimizer = MLStrategyOptimizer()
    
    state = "T20_W2_WX1_L1_P1"
    action = optimizer.select_action(state)
    
    assert action in StrategyAction
    assert isinstance(action, StrategyAction)

def test_ml_strategy_reward_calculation():
    """Test reward calculation for strategy actions."""
    optimizer = MLStrategyOptimizer()
    
    thermal_state = np.array([95.0, 90.0, 85.0])
    wear_summary = {'FL': {'wear_level': 0.2}}
    weather_summary = {}
    race_context = {'position': 5, 'total_cars': 20}
    action = StrategyAction.MAINTAIN_PACE
    
    reward = optimizer.calculate_reward(thermal_state, wear_summary, weather_summary, race_context, action)
    
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0

def test_ml_strategy_q_learning_update():
    """Test Q-learning update mechanism."""
    optimizer = MLStrategyOptimizer()
    
    state = "T20_W2_WX1_L1_P1"
    action = StrategyAction.MAINTAIN_PACE
    reward = 0.8
    next_state = "T21_W2_WX1_L1_P1"
    
    optimizer.update_q_value(state, action, reward, next_state)
    
    assert state in optimizer.q_table
    assert action in optimizer.q_table[state]

def test_ml_strategy_recommendations():
    """Test strategy recommendations generation."""
    optimizer = MLStrategyOptimizer()
    
    thermal_state = np.array([105.0, 100.0, 95.0])
    wear_summary = {'FL': {'wear_level': 0.3}}
    weather_summary = {'rain_probability': 0.2}
    race_context = {'current_lap': 20, 'position': 3}
    
    recommendations = optimizer.get_strategy_recommendations(thermal_state, wear_summary, weather_summary, race_context)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_ml_degradation_predictor_initialization():
    """Test ML degradation predictor initialization."""
    params = MLDegradationParams()
    predictor = MLDegradationPredictor(params)
    
    assert predictor.p.input_dimensions == 15
    assert predictor.p.output_dimensions == 5
    assert predictor.p.learning_rate == 0.001
    assert not predictor.is_trained

def test_ml_degradation_feature_extraction():
    """Test feature extraction for degradation prediction."""
    predictor = MLDegradationPredictor()
    
    thermal_state = np.array([90.0, 85.0, 80.0])
    wear_summary = {'FL': {'wear_level': 0.2}}
    weather_summary = {'track_temperature': 30.0}
    structural_state = {'deflection_ratio': 0.1}
    compound_state = {'current_temperature': 85.0}
    race_context = {'current_lap': 10}
    
    features = predictor.extract_features(thermal_state, wear_summary, weather_summary,
                                       structural_state, compound_state, race_context)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == predictor.p.input_dimensions

def test_ml_degradation_prediction():
    """Test degradation prediction."""
    predictor = MLDegradationPredictor()
    
    # Mock training data
    historical_data = []
    for i in range(100):
        data_point = {
            'thermal_state': np.array([90 + i*0.1, 85 + i*0.1, 80 + i*0.1]),
            'wear_summary': {'FL': {'wear_level': i*0.01}},
            'weather_summary': {'track_temperature': 30.0},
            'structural_state': {'deflection_ratio': 0.1},
            'compound_state': {'current_temperature': 85.0},
            'race_context': {'current_lap': i}
        }
        historical_data.append(data_point)
    
    predictor.prepare_training_data(historical_data)
    predictor.train_model(epochs=10)
    
    # Test prediction
    thermal_state = np.array([95.0, 90.0, 85.0])
    wear_summary = {'FL': {'wear_level': 0.3}}
    weather_summary = {'track_temperature': 35.0}
    structural_state = {'deflection_ratio': 0.12}
    compound_state = {'current_temperature': 90.0}
    race_context = {'current_lap': 15}
    
    prediction = predictor.predict_degradation(thermal_state, wear_summary, weather_summary,
                                             structural_state, compound_state, race_context)
    
    assert isinstance(prediction, dict)
    assert 'total_degradation' in prediction
    assert 'confidence' in prediction

def test_ml_degradation_recommendations():
    """Test degradation-based recommendations."""
    predictor = MLDegradationPredictor()
    
    degradation_prediction = {
        'total_degradation': 0.8,
        'thermal_degradation': 0.6,
        'wear_degradation': 0.5,
        'confidence': 0.9
    }
    thermal_state = np.array([110.0, 105.0, 100.0])
    wear_summary = {'FL': {'wear_level': 0.7}}
    
    recommendations = predictor.get_degradation_recommendations(degradation_prediction, thermal_state, wear_summary)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)

def test_ml_driver_profiler_initialization():
    """Test ML driver profiler initialization."""
    params = MLDriverProfilingParams()
    profiler = MLDriverProfiler(params)
    
    assert profiler.p.n_clusters == 7
    assert profiler.p.min_samples == 50
    assert profiler.learning_phase == LearningPhase.INITIAL
    assert profiler.model_confidence == 0.0

def test_ml_driver_feature_extraction():
    """Test driver feature extraction."""
    profiler = MLDriverProfiler()
    
    thermal_state = np.array([90.0, 85.0, 80.0])
    wear_summary = {'FL': {'wear_level': 0.2}}
    weather_summary = {'track_temperature': 30.0}
    structural_state = {'deflection_ratio': 0.1}
    compound_state = {'current_temperature': 85.0}
    race_context = {'speed': 50.0}
    driver_actions = ['maintain_pace', 'increase_pace']
    performance_metrics = {'lap_time': 90.0}
    
    features = profiler.extract_driver_features(thermal_state, wear_summary, weather_summary,
                                              structural_state, compound_state, race_context, driver_actions)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == profiler.p.feature_dimensions

def test_ml_driver_telemetry_data():
    """Test adding telemetry data for driver profiling."""
    profiler = MLDriverProfiler()
    
    thermal_state = np.array([90.0, 85.0, 80.0])
    wear_summary = {'FL': {'wear_level': 0.2}}
    weather_summary = {'track_temperature': 30.0}
    structural_state = {'deflection_ratio': 0.1}
    compound_state = {'current_temperature': 85.0}
    race_context = {'speed': 50.0}
    driver_actions = ['maintain_pace']
    performance_metrics = {'lap_time': 90.0}
    
    profiler.add_telemetry_data('driver_1', thermal_state, wear_summary, weather_summary,
                               structural_state, compound_state, race_context, driver_actions, performance_metrics)
    
    assert 'driver_1' in profiler.driver_profiles
    assert len(profiler.telemetry_data) == 1

def test_ml_driver_profile():
    """Test driver profile generation."""
    profiler = MLDriverProfiler()
    
    # Add some telemetry data
    for i in range(60):  # More than min_samples
        thermal_state = np.array([90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1])
        wear_summary = {'FL': {'wear_level': i*0.01}}
        weather_summary = {'track_temperature': 30.0}
        structural_state = {'deflection_ratio': 0.1}
        compound_state = {'current_temperature': 85.0}
        race_context = {'speed': 50.0}
        driver_actions = ['maintain_pace']
        performance_metrics = {'lap_time': 90.0}
        
        profiler.add_telemetry_data('driver_1', thermal_state, wear_summary, weather_summary,
                                   structural_state, compound_state, race_context, driver_actions, performance_metrics)
    
    profile = profiler.get_driver_profile('driver_1')
    
    assert isinstance(profile, dict)
    assert 'driver_id' in profile
    assert 'behavior_pattern' in profile
    assert 'confidence' in profile

def test_ml_driver_performance_prediction():
    """Test driver performance prediction."""
    profiler = MLDriverProfiler()
    
    # Add telemetry data
    for i in range(60):
        thermal_state = np.array([90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1])
        wear_summary = {'FL': {'wear_level': i*0.01}}
        weather_summary = {'track_temperature': 30.0}
        structural_state = {'deflection_ratio': 0.1}
        compound_state = {'current_temperature': 85.0}
        race_context = {'speed': 50.0}
        driver_actions = ['maintain_pace']
        performance_metrics = {'lap_time': 90.0}
        
        profiler.add_telemetry_data('driver_1', thermal_state, wear_summary, weather_summary,
                                   structural_state, compound_state, race_context, driver_actions, performance_metrics)
    
    conditions = {'high_temperature': True, 'wet_conditions': False}
    prediction = profiler.predict_driver_performance('driver_1', conditions)
    
    assert isinstance(prediction, dict)
    assert 'predicted_performance' in prediction
    assert 'confidence' in prediction

def test_ml_recommendation_engine_initialization():
    """Test ML recommendation engine initialization."""
    params = MLRecommendationParams()
    engine = MLRecommendationEngine(params)
    
    assert len(engine.models) == 4
    assert 'random_forest' in engine.models
    assert 'gradient_boosting' in engine.models
    assert 'svm' in engine.models
    assert 'logistic_regression' in engine.models

def test_ml_recommendation_feature_extraction():
    """Test recommendation feature extraction."""
    engine = MLRecommendationEngine()
    
    thermal_state = np.array([90.0, 85.0, 80.0])
    wear_summary = {'FL': {'wear_level': 0.2}}
    weather_summary = {'track_temperature': 30.0}
    structural_state = {'deflection_ratio': 0.1}
    compound_state = {'current_temperature': 85.0}
    race_context = {'current_lap': 10, 'position': 3}
    driver_profile = {'confidence': 0.8, 'behavior_pattern': 'thermal_aggressor'}
    ml_predictions = {'degradation_prediction': {'total_degradation': 0.3, 'confidence': 0.7}}
    
    features = engine.extract_recommendation_features(thermal_state, wear_summary, weather_summary,
                                                    structural_state, compound_state, race_context,
                                                    driver_profile, ml_predictions)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == engine.p.feature_dimensions

def test_ml_recommendation_generation():
    """Test recommendation generation."""
    engine = MLRecommendationEngine()
    
    thermal_state = np.array([110.0, 105.0, 100.0])
    wear_summary = {'FL': {'wear_level': 0.7}}
    weather_summary = {'track_temperature': 40.0, 'rain_probability': 0.8}
    structural_state = {'deflection_ratio': 0.15}
    compound_state = {'current_temperature': 100.0}
    race_context = {'current_lap': 20, 'position': 5}
    driver_profile = {'confidence': 0.8, 'behavior_pattern': 'thermal_aggressor'}
    ml_predictions = {'degradation_prediction': {'total_degradation': 0.8, 'confidence': 0.9}}
    
    recommendations = engine.generate_recommendations(thermal_state, wear_summary, weather_summary,
                                                    structural_state, compound_state, race_context,
                                                    driver_profile, ml_predictions)
    
    assert isinstance(recommendations, list)
    for rec in recommendations:
        assert isinstance(rec, dict)
        assert 'type' in rec
        assert 'priority' in rec
        assert 'message' in rec
        assert 'confidence' in rec

def test_ml_recommendation_feedback():
    """Test recommendation feedback system."""
    engine = MLRecommendationEngine()
    
    engine.update_recommendation_feedback(1, True, 0.1)
    engine.update_recommendation_feedback(2, False, -0.05)
    
    assert len(engine.recommendation_feedback) == 2
    assert len(engine.success_rate_history) == 2

def test_ml_pattern_recognizer_initialization():
    """Test ML pattern recognizer initialization."""
    params = MLPatternRecognitionParams()
    recognizer = MLPatternRecognizer(params)
    
    assert recognizer.p.min_pattern_length == 10
    assert recognizer.p.max_pattern_length == 100
    assert recognizer.p.pattern_similarity_threshold == 0.8

def test_ml_pattern_feature_extraction():
    """Test pattern feature extraction."""
    recognizer = MLPatternRecognizer()
    
    time_series_data = []
    for i in range(20):
        data_point = {
            'thermal_state': [90.0 + i*0.5, 85.0 + i*0.5, 80.0 + i*0.5],
            'wear_levels': [0.1 + i*0.01, 0.1 + i*0.01, 0.1 + i*0.01, 0.1 + i*0.01],
            'outcome': {'success': i % 2 == 0},
            'performance_metrics': {'lap_time': 90.0 - i*0.1}
        }
        time_series_data.append(data_point)
    
    features = recognizer.extract_pattern_features(time_series_data)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == recognizer.p.feature_dimensions

def test_ml_pattern_discovery():
    """Test pattern discovery in historical data."""
    recognizer = MLPatternRecognizer()
    
    historical_data = []
    for i in range(100):
        data_point = {
            'thermal_state': [90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1],
            'wear_levels': [0.1 + i*0.005, 0.1 + i*0.005, 0.1 + i*0.005, 0.1 + i*0.005],
            'outcome': {'success': i % 3 == 0},
            'performance_metrics': {'lap_time': 90.0 - i*0.05}
        }
        historical_data.append(data_point)
    
    patterns = recognizer.discover_patterns(historical_data)
    
    assert isinstance(patterns, dict)

def test_ml_pattern_classification():
    """Test pattern classification."""
    recognizer = MLPatternRecognizer()
    
    # Add some patterns to the database
    recognizer.pattern_database = {
        'pattern_1': {
            'pattern_id': 'pattern_1',
            'pattern_type': PatternType.THERMAL_PATTERN.value,
            'success_rate': 0.8,
            'recommendations': []
        }
    }
    
    current_data = []
    for i in range(15):
        data_point = {
            'thermal_state': [90.0 + i*0.2, 85.0 + i*0.2, 80.0 + i*0.2],
            'wear_levels': [0.1 + i*0.01, 0.1 + i*0.01, 0.1 + i*0.01, 0.1 + i*0.01],
            'outcome': {'success': True},
            'performance_metrics': {'lap_time': 90.0}
        }
        current_data.append(data_point)
    
    classification = recognizer.classify_current_pattern(current_data)
    
    assert isinstance(classification, dict)
    assert 'classification' in classification
    assert 'confidence' in classification

def test_ml_strategy_training():
    """Test ML strategy training."""
    optimizer = MLStrategyOptimizer()
    
    # Simulate training episodes
    for episode in range(50):
        state = f"T{20 + episode}_W2_WX1_L1_P1"
        action = optimizer.select_action(state)
        reward = 0.8 if action == StrategyAction.MAINTAIN_PACE else 0.6
        next_state = f"T{21 + episode}_W2_WX1_L1_P1"
        
        optimizer.update_q_value(state, action, reward, next_state)
        optimizer.train_model()
    
    assert optimizer.training_step > 0
    assert len(optimizer.q_table) > 0

def test_ml_degradation_training():
    """Test ML degradation model training."""
    predictor = MLDegradationPredictor()
    
    # Prepare training data
    historical_data = []
    for i in range(100):
        data_point = {
            'thermal_state': np.array([90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1]),
            'wear_summary': {'FL': {'wear_level': i*0.01}},
            'weather_summary': {'track_temperature': 30.0},
            'structural_state': {'deflection_ratio': 0.1},
            'compound_state': {'current_temperature': 85.0},
            'race_context': {'current_lap': i}
        }
        historical_data.append(data_point)
    
    predictor.prepare_training_data(historical_data)
    predictor.train_model(epochs=20)
    
    assert predictor.is_trained
    assert len(predictor.training_history) > 0

def test_ml_driver_profiling_training():
    """Test ML driver profiling training."""
    profiler = MLDriverProfiler()
    
    # Add telemetry data for multiple drivers
    for driver_id in ['driver_1', 'driver_2', 'driver_3']:
        for i in range(60):
            thermal_state = np.array([90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1])
            wear_summary = {'FL': {'wear_level': i*0.01}}
            weather_summary = {'track_temperature': 30.0}
            structural_state = {'deflection_ratio': 0.1}
            compound_state = {'current_temperature': 85.0}
            race_context = {'speed': 50.0}
            driver_actions = ['maintain_pace']
            performance_metrics = {'lap_time': 90.0}
            
            profiler.add_telemetry_data(driver_id, thermal_state, wear_summary, weather_summary,
                                       structural_state, compound_state, race_context, driver_actions, performance_metrics)
    
    assert len(profiler.driver_profiles) == 3
    assert profiler.model_confidence > 0

def test_ml_recommendation_training():
    """Test ML recommendation engine training."""
    engine = MLRecommendationEngine()
    
    # Prepare training data
    historical_data = []
    for i in range(100):
        data_point = {
            'thermal_state': np.array([90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1]),
            'wear_summary': {'FL': {'wear_level': i*0.01}},
            'weather_summary': {'track_temperature': 30.0},
            'structural_state': {'deflection_ratio': 0.1},
            'compound_state': {'current_temperature': 85.0},
            'race_context': {'current_lap': i, 'position': 3},
            'driver_profile': {'confidence': 0.8},
            'ml_predictions': {'degradation_prediction': {'total_degradation': 0.3}},
            'outcome': {'success': i % 2 == 0}
        }
        historical_data.append(data_point)
    
    engine.prepare_training_data(historical_data)
    engine.train_ensemble_models()
    
    assert len(engine.model_performance) > 0
    assert all('accuracy' in perf for perf in engine.model_performance.values())

def test_ml_pattern_training():
    """Test ML pattern recognition training."""
    recognizer = MLPatternRecognizer()
    
    # Prepare historical data
    historical_data = []
    for i in range(100):
        data_point = {
            'thermal_state': [90.0 + i*0.1, 85.0 + i*0.1, 80.0 + i*0.1],
            'wear_levels': [0.1 + i*0.005, 0.1 + i*0.005, 0.1 + i*0.005, 0.1 + i*0.005],
            'outcome': {'success': i % 3 == 0},
            'performance_metrics': {'lap_time': 90.0 - i*0.05}
        }
        historical_data.append(data_point)
    
    patterns = recognizer.discover_patterns(historical_data)
    
    assert isinstance(patterns, dict)

def test_ml_model_save_load():
    """Test ML model save and load functionality."""
    optimizer = MLStrategyOptimizer()
    
    # Add some data
    optimizer.update_q_value("state1", StrategyAction.MAINTAIN_PACE, 0.8, "state2")
    optimizer.training_step = 10
    
    # Save model
    optimizer.save_model("test_model.pkl")
    
    # Create new optimizer and load model
    new_optimizer = MLStrategyOptimizer()
    new_optimizer.load_model("test_model.pkl")
    
    assert new_optimizer.training_step == 10
    assert "state1" in new_optimizer.q_table

def test_ml_model_reset():
    """Test ML model reset functionality."""
    optimizer = MLStrategyOptimizer()
    
    # Add some data
    optimizer.update_q_value("state1", StrategyAction.MAINTAIN_PACE, 0.8, "state2")
    optimizer.training_step = 10
    
    # Reset model
    optimizer.reset_model()
    
    assert optimizer.training_step == 0
    assert len(optimizer.q_table) == 0
    assert optimizer.current_exploration_rate == optimizer.p.exploration_rate

def test_ml_integration():
    """Test integration between ML components."""
    # Initialize all ML components
    strategy_optimizer = MLStrategyOptimizer()
    degradation_predictor = MLDegradationPredictor()
    driver_profiler = MLDriverProfiler()
    recommendation_engine = MLRecommendationEngine()
    pattern_recognizer = MLPatternRecognizer()
    
    # Test data flow between components
    thermal_state = np.array([95.0, 90.0, 85.0])
    wear_summary = {'FL': {'wear_level': 0.3}}
    weather_summary = {'track_temperature': 35.0}
    structural_state = {'deflection_ratio': 0.12}
    compound_state = {'current_temperature': 90.0}
    race_context = {'current_lap': 15, 'position': 3}
    driver_profile = {'confidence': 0.8, 'behavior_pattern': 'thermal_aggressor'}
    
    # Get predictions from different components
    strategy_recs = strategy_optimizer.get_strategy_recommendations(thermal_state, wear_summary, weather_summary, race_context)
    degradation_pred = degradation_predictor.predict_degradation(thermal_state, wear_summary, weather_summary, structural_state, compound_state, race_context)
    driver_profile_data = driver_profiler.get_driver_profile('driver_1')
    
    # Test that all components work together
    assert isinstance(strategy_recs, list)
    assert isinstance(degradation_pred, dict)
    assert isinstance(driver_profile_data, dict)
    
    print("ML Integration Test Passed!")

def test_ml_performance_metrics():
    """Test ML performance metrics calculation."""
    optimizer = MLStrategyOptimizer()
    
    # Add some training data
    for i in range(100):
        state = f"state_{i}"
        action = StrategyAction.MAINTAIN_PACE
        reward = 0.8
        next_state = f"state_{i+1}"
        optimizer.update_q_value(state, action, reward, next_state)
    
    summary = optimizer.get_ml_summary()
    
    assert 'training_step' in summary
    assert 'q_table_size' in summary
    assert 'learning_progress' in summary

def test_ml_error_handling():
    """Test ML error handling."""
    optimizer = MLStrategyOptimizer()
    
    # Test with invalid data
    thermal_state = np.array([100.0, 95.0, 90.0])
    wear_summary = {}  # Empty wear summary
    weather_summary = {}
    race_context = {}
    
    # Should handle gracefully
    recommendations = optimizer.get_strategy_recommendations(thermal_state, wear_summary, weather_summary, race_context)
    
    assert isinstance(recommendations, list)
