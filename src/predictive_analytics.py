from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class PredictionType(Enum):
    """Types of predictions."""
    LAP_TIME = "lap_time"
    TIRE_DEGRADATION = "tire_degradation"
    PIT_WINDOW = "pit_window"
    STRATEGY_SUCCESS = "strategy_success"
    WEATHER_IMPACT = "weather_impact"
    DRIVER_PERFORMANCE = "driver_performance"

class ModelType(Enum):
    """Types of ML models."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    ENSEMBLE = "ensemble"

@dataclass
class PredictiveAnalyticsParams:
    """Parameters for predictive analytics."""
    # Model parameters
    model_type: ModelType = ModelType.ENSEMBLE
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    feature_window: int = 10  # Number of previous data points to use
    lag_features: int = 5    # Number of lag features
    rolling_window: int = 5  # Rolling window for features
    
    # Model training
    max_iterations: int = 1000
    early_stopping_rounds: int = 50
    validation_split: float = 0.1
    
    # Prediction parameters
    prediction_horizon: int = 10  # Number of future predictions
    confidence_threshold: float = 0.7
    uncertainty_quantification: bool = True
    
    # Performance parameters
    min_training_samples: int = 100
    model_update_frequency: int = 24  # hours
    cache_predictions: bool = True

class PredictiveAnalytics:
    """
    Predictive analytics system for F1 race strategy optimization.
    
    Features:
    - Multi-model ensemble for robust predictions
    - Lap time prediction with confidence intervals
    - Tire degradation forecasting
    - Pit window optimization
    - Strategy success prediction
    - Weather impact modeling
    - Driver performance prediction
    - Real-time model updates and adaptation
    """
    
    def __init__(self, params: PredictiveAnalyticsParams = None):
        self.p = params or PredictiveAnalyticsParams()
        
        # Models
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Training data
        self.training_data = {}
        self.feature_columns = {}
        
        # Predictions cache
        self.prediction_cache = {}
        self.model_performance = {}
        
        # Model metadata
        self.model_metadata = {
            'last_trained': None,
            'training_samples': 0,
            'model_versions': {},
            'performance_history': []
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different prediction types."""
        for prediction_type in PredictionType:
            self.models[prediction_type] = {}
            self.scalers[prediction_type] = StandardScaler()
            
            if self.p.model_type == ModelType.ENSEMBLE:
                self.models[prediction_type] = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.p.random_state
                    ),
                    'gradient_boosting': GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.p.random_state
                    ),
                    'linear_regression': LinearRegression(),
                    'ridge_regression': Ridge(alpha=1.0)
                }
            else:
                model_class = self._get_model_class(self.p.model_type)
                self.models[prediction_type] = model_class
    
    def _get_model_class(self, model_type: ModelType):
        """Get model class for specified model type."""
        model_classes = {
            ModelType.RANDOM_FOREST: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor,
            ModelType.LINEAR_REGRESSION: LinearRegression,
            ModelType.RIDGE_REGRESSION: Ridge
        }
        return model_classes[model_type]
    
    def prepare_training_data(self, data: Dict[str, Any], prediction_type: PredictionType):
        """
        Prepare training data for a specific prediction type.
        
        Args:
            data: Training data dictionary
            prediction_type: Type of prediction to prepare data for
        """
        # Extract features and target
        features, target = self._extract_features_target(data, prediction_type)
        
        if len(features) < self.p.min_training_samples:
            warnings.warn(f"Insufficient training data: {len(features)} samples")
            return
        
        # Store training data
        self.training_data[prediction_type] = {
            'features': features,
            'target': target,
            'feature_names': self._get_feature_names(prediction_type)
        }
        
        # Update feature columns
        self.feature_columns[prediction_type] = features.columns.tolist()
    
    def _extract_features_target(self, data: Dict[str, Any], prediction_type: PredictionType) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target for specific prediction type."""
        if prediction_type == PredictionType.LAP_TIME:
            return self._extract_lap_time_features(data)
        elif prediction_type == PredictionType.TIRE_DEGRADATION:
            return self._extract_tire_degradation_features(data)
        elif prediction_type == PredictionType.PIT_WINDOW:
            return self._extract_pit_window_features(data)
        elif prediction_type == PredictionType.STRATEGY_SUCCESS:
            return self._extract_strategy_success_features(data)
        elif prediction_type == PredictionType.WEATHER_IMPACT:
            return self._extract_weather_impact_features(data)
        elif prediction_type == PredictionType.DRIVER_PERFORMANCE:
            return self._extract_driver_performance_features(data)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def _extract_lap_time_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for lap time prediction."""
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Create features
        features = pd.DataFrame()
        
        # Temperature features
        features['tread_temp'] = df['tread_temp']
        features['carcass_temp'] = df['carcass_temp']
        features['rim_temp'] = df['rim_temp']
        features['track_temp'] = df['track_temp']
        features['ambient_temp'] = df['ambient_temp']
        
        # Tire condition features
        features['wear_level'] = df['wear_level']
        features['pressure'] = df['pressure']
        features['tire_age'] = df['tire_age']
        
        # Weather features
        features['humidity'] = df['humidity']
        features['wind_speed'] = df['wind_speed']
        features['rain_probability'] = df['rain_probability']
        
        # Driver and position features
        features['position'] = df['position']
        features['fuel_level'] = df['fuel_level']
        
        # Add lag features
        for lag in range(1, self.p.lag_features + 1):
            features[f'tread_temp_lag_{lag}'] = df['tread_temp'].shift(lag)
            features[f'wear_level_lag_{lag}'] = df['wear_level'].shift(lag)
            features[f'lap_time_lag_{lag}'] = df['lap_time'].shift(lag)
        
        # Add rolling features
        for window in [3, 5, 10]:
            features[f'tread_temp_rolling_{window}'] = df['tread_temp'].rolling(window).mean()
            features[f'wear_level_rolling_{window}'] = df['wear_level'].rolling(window).mean()
            features[f'lap_time_rolling_{window}'] = df['lap_time'].rolling(window).mean()
        
        # Target variable
        target = df['lap_time']
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _extract_tire_degradation_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for tire degradation prediction."""
        df = pd.DataFrame(data)
        
        features = pd.DataFrame()
        
        # Temperature features
        features['tread_temp'] = df['tread_temp']
        features['carcass_temp'] = df['carcass_temp']
        features['rim_temp'] = df['rim_temp']
        features['track_temp'] = df['track_temp']
        
        # Load and stress features
        features['lateral_load'] = df.get('lateral_load', 0)
        features['longitudinal_load'] = df.get('longitudinal_load', 0)
        features['slip_angle'] = df.get('slip_angle', 0)
        features['slip_ratio'] = df.get('slip_ratio', 0)
        
        # Tire condition features
        features['pressure'] = df['pressure']
        features['tire_age'] = df['tire_age']
        features['compound_type'] = df.get('compound_type', 0)
        
        # Weather features
        features['humidity'] = df['humidity']
        features['wind_speed'] = df['wind_speed']
        
        # Add lag features
        for lag in range(1, self.p.lag_features + 1):
            features[f'wear_level_lag_{lag}'] = df['wear_level'].shift(lag)
            features[f'tread_temp_lag_{lag}'] = df['tread_temp'].shift(lag)
        
        # Target variable (wear level change)
        target = df['wear_level'].diff().fillna(0)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _extract_pit_window_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for pit window prediction."""
        df = pd.DataFrame(data)
        
        features = pd.DataFrame()
        
        # Tire condition features
        features['wear_level'] = df['wear_level']
        features['tread_temp'] = df['tread_temp']
        features['pressure'] = df['pressure']
        features['tire_age'] = df['tire_age']
        
        # Performance features
        features['lap_time'] = df['lap_time']
        features['position'] = df['position']
        features['gap_to_leader'] = df.get('gap_to_leader', 0)
        
        # Weather features
        features['track_temp'] = df['track_temp']
        features['rain_probability'] = df['rain_probability']
        
        # Add lag features
        for lag in range(1, self.p.lag_features + 1):
            features[f'wear_level_lag_{lag}'] = df['wear_level'].shift(lag)
            features[f'lap_time_lag_{lag}'] = df['lap_time'].shift(lag)
        
        # Target variable (pit window in laps)
        target = df.get('pit_window', 0)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _extract_strategy_success_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for strategy success prediction."""
        df = pd.DataFrame(data)
        
        features = pd.DataFrame()
        
        # Strategy features
        features['strategy_type'] = df.get('strategy_type', 0)
        features['pit_stops'] = df.get('pit_stops', 0)
        features['compound_sequence'] = df.get('compound_sequence', 0)
        
        # Performance features
        features['lap_time'] = df['lap_time']
        features['position'] = df['position']
        features['tire_management_score'] = df.get('tire_management_score', 0)
        
        # Weather features
        features['track_temp'] = df['track_temp']
        features['rain_probability'] = df['rain_probability']
        
        # Target variable (strategy success score)
        target = df.get('strategy_success', 0)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _extract_weather_impact_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for weather impact prediction."""
        df = pd.DataFrame(data)
        
        features = pd.DataFrame()
        
        # Weather features
        features['track_temp'] = df['track_temp']
        features['ambient_temp'] = df['ambient_temp']
        features['humidity'] = df['humidity']
        features['wind_speed'] = df['wind_speed']
        features['rain_probability'] = df['rain_probability']
        features['cloud_cover'] = df.get('cloud_cover', 0)
        features['visibility'] = df.get('visibility', 0)
        
        # Tire features
        features['tread_temp'] = df['tread_temp']
        features['wear_level'] = df['wear_level']
        features['pressure'] = df['pressure']
        
        # Target variable (weather impact on lap time)
        target = df.get('weather_impact', 0)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _extract_driver_performance_features(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for driver performance prediction."""
        df = pd.DataFrame(data)
        
        features = pd.DataFrame()
        
        # Driver features
        features['driver_id'] = df.get('driver_id', 0)
        features['experience_level'] = df.get('experience_level', 0)
        features['wet_weather_skill'] = df.get('wet_weather_skill', 0)
        
        # Performance features
        features['lap_time'] = df['lap_time']
        features['tire_management_score'] = df.get('tire_management_score', 0)
        features['thermal_efficiency'] = df.get('thermal_efficiency', 0)
        features['consistency_score'] = df.get('consistency_score', 0)
        
        # Conditions
        features['track_temp'] = df['track_temp']
        features['rain_probability'] = df['rain_probability']
        
        # Target variable (driver performance score)
        target = df.get('driver_performance', 0)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features, target
    
    def _get_feature_names(self, prediction_type: PredictionType) -> List[str]:
        """Get feature names for specific prediction type."""
        # This would be dynamically generated based on the features
        # For now, return a placeholder
        return [f"feature_{i}" for i in range(20)]
    
    def train_models(self, prediction_type: PredictionType = None):
        """
        Train models for specified prediction type or all types.
        
        Args:
            prediction_type: Specific prediction type to train, or None for all
        """
        if prediction_type:
            prediction_types = [prediction_type]
        else:
            prediction_types = list(PredictionType)
        
        for pred_type in prediction_types:
            if pred_type not in self.training_data:
                continue
            
            training_data = self.training_data[pred_type]
            features = training_data['features']
            target = training_data['target']
            
            if len(features) < self.p.min_training_samples:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=self.p.test_size, random_state=self.p.random_state
            )
            
            # Scale features
            X_train_scaled = self.scalers[pred_type].fit_transform(X_train)
            X_test_scaled = self.scalers[pred_type].transform(X_test)
            
            # Train models
            if isinstance(self.models[pred_type], dict):
                # Ensemble models
                for model_name, model in self.models[pred_type].items():
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store performance
                    if pred_type not in self.model_performance:
                        self.model_performance[pred_type] = {}
                    
                    self.model_performance[pred_type][model_name] = {
                        'mse': mse,
                        'r2': r2,
                        'mae': mae,
                        'training_samples': len(X_train)
                    }
            else:
                # Single model
                model = self.models[pred_type]
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Store performance
                self.model_performance[pred_type] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'training_samples': len(X_train)
                }
            
            # Update metadata
            self.model_metadata['last_trained'] = datetime.now()
            self.model_metadata['training_samples'] = len(features)
    
    def predict(self, features: Dict[str, Any], prediction_type: PredictionType) -> Dict[str, Any]:
        """
        Make predictions for specified features and prediction type.
        
        Args:
            features: Input features for prediction
            prediction_type: Type of prediction to make
            
        Returns:
            Dictionary with prediction results
        """
        if prediction_type not in self.models:
            return {'error': 'Model not trained for this prediction type'}
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Scale features
        try:
            features_scaled = self.scalers[prediction_type].transform(feature_df)
        except:
            return {'error': 'Feature scaling failed'}
        
        # Make predictions
        if isinstance(self.models[prediction_type], dict):
            # Ensemble predictions
            predictions = {}
            for model_name, model in self.models[prediction_type].items():
                pred = model.predict(features_scaled)[0]
                predictions[model_name] = pred
            
            # Calculate ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            
            # Calculate confidence (standard deviation of predictions)
            confidence = 1.0 - (np.std(list(predictions.values())) / np.mean(list(predictions.values())))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'prediction': ensemble_pred,
                'confidence': confidence,
                'individual_predictions': predictions,
                'prediction_type': prediction_type.value
            }
        else:
            # Single model prediction
            model = self.models[prediction_type]
            prediction = model.predict(features_scaled)[0]
            
            return {
                'prediction': prediction,
                'confidence': 0.8,  # Default confidence for single model
                'prediction_type': prediction_type.value
            }
    
    def predict_lap_time(self, thermal_state: np.ndarray, wear_summary: Dict, 
                        weather_summary: Dict, race_context: Dict) -> Dict[str, Any]:
        """Predict lap time based on current conditions."""
        features = {
            'tread_temp': thermal_state[0],
            'carcass_temp': thermal_state[1],
            'rim_temp': thermal_state[2],
            'wear_level': wear_summary.get('FL', {}).get('wear_level', 0.0),
            'pressure': 1.5,  # Default pressure
            'track_temp': weather_summary.get('track_temperature', 30.0),
            'ambient_temp': weather_summary.get('ambient_temperature', 25.0),
            'humidity': weather_summary.get('humidity', 50.0),
            'wind_speed': weather_summary.get('wind_speed', 0.0),
            'rain_probability': weather_summary.get('rain_probability', 0.0),
            'position': race_context.get('position', 1),
            'fuel_level': race_context.get('fuel_level', 100.0),
            'tire_age': race_context.get('tire_age', 0),
            'lap_time': 90.0  # Default lap time
        }
        
        return self.predict(features, PredictionType.LAP_TIME)
    
    def predict_tire_degradation(self, thermal_state: np.ndarray, wear_summary: Dict,
                               weather_summary: Dict, structural_state: Dict) -> Dict[str, Any]:
        """Predict tire degradation based on current conditions."""
        features = {
            'tread_temp': thermal_state[0],
            'carcass_temp': thermal_state[1],
            'rim_temp': thermal_state[2],
            'track_temp': weather_summary.get('track_temperature', 30.0),
            'lateral_load': structural_state.get('lateral_load', 0.0),
            'longitudinal_load': structural_state.get('longitudinal_load', 0.0),
            'slip_angle': structural_state.get('slip_angle', 0.0),
            'slip_ratio': structural_state.get('slip_ratio', 0.0),
            'pressure': 1.5,
            'tire_age': 0,
            'compound_type': 0,
            'humidity': weather_summary.get('humidity', 50.0),
            'wind_speed': weather_summary.get('wind_speed', 0.0),
            'wear_level': wear_summary.get('FL', {}).get('wear_level', 0.0)
        }
        
        return self.predict(features, PredictionType.TIRE_DEGRADATION)
    
    def predict_pit_window(self, wear_summary: Dict, thermal_state: np.ndarray,
                          weather_summary: Dict, race_context: Dict) -> Dict[str, Any]:
        """Predict optimal pit window based on current conditions."""
        features = {
            'wear_level': wear_summary.get('FL', {}).get('wear_level', 0.0),
            'tread_temp': thermal_state[0],
            'pressure': 1.5,
            'tire_age': race_context.get('tire_age', 0),
            'lap_time': race_context.get('lap_time', 90.0),
            'position': race_context.get('position', 1),
            'gap_to_leader': race_context.get('gap_to_leader', 0.0),
            'track_temp': weather_summary.get('track_temperature', 30.0),
            'rain_probability': weather_summary.get('rain_probability', 0.0)
        }
        
        return self.predict(features, PredictionType.PIT_WINDOW)
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return {
            'model_performance': self.model_performance,
            'model_metadata': self.model_metadata,
            'training_data_status': {
                pred_type: len(data['features']) if pred_type in self.training_data else 0
                for pred_type in PredictionType
            }
        }
    
    def save_models(self, filepath: str):
        """Save trained models to file."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'model_performance': self.model_performance,
            'model_metadata': self.model_metadata,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str):
        """Load trained models from file."""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_performance = model_data['model_performance']
            self.model_metadata = model_data['model_metadata']
            self.feature_columns = model_data['feature_columns']
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return {
            'prediction_types': [pt.value for pt in PredictionType],
            'model_type': self.p.model_type.value,
            'model_performance': self.model_performance,
            'model_metadata': self.model_metadata,
            'training_data_status': {
                pred_type: len(data['features']) if pred_type in self.training_data else 0
                for pred_type in PredictionType
            },
            'prediction_cache_size': len(self.prediction_cache)
        }
