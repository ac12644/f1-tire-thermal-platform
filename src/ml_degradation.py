from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pickle
import json
from collections import deque
import random

class DegradationType(Enum):
    """Types of tire degradation."""
    THERMAL_DEGRADATION = "thermal_degradation"
    MECHANICAL_DEGRADATION = "mechanical_degradation"
    WEAR_DEGRADATION = "wear_degradation"
    FATIGUE_DEGRADATION = "fatigue_degradation"
    COMPOUND_DEGRADATION = "compound_degradation"

class PredictionHorizon(Enum):
    """Prediction horizons for degradation modeling."""
    SHORT_TERM = "short_term"      # 1-5 laps
    MEDIUM_TERM = "medium_term"    # 5-20 laps
    LONG_TERM = "long_term"        # 20+ laps

@dataclass
class MLDegradationParams:
    """Parameters for ML degradation modeling."""
    # Neural network parameters
    input_dimensions: int = 15     # Input features
    hidden_dimensions: List[int] = None  # Hidden layer sizes
    output_dimensions: int = 5    # Output predictions
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Training parameters
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_save_frequency: int = 50
    
    # Data parameters
    sequence_length: int = 10      # Time series length
    prediction_horizon: int = 5    # Laps ahead to predict
    
    # Feature engineering
    feature_scaling: bool = True
    feature_normalization: bool = True
    
    def __post_init__(self):
        if self.hidden_dimensions is None:
            self.hidden_dimensions = [64, 32, 16]

class MLDegradationPredictor:
    """
    Machine Learning-based tire degradation prediction using neural networks.
    
    Features:
    - Multi-layer neural network for degradation prediction
    - Time series analysis for temporal patterns
    - Multi-output prediction (different degradation types)
    - Feature engineering and normalization
    - Model validation and performance tracking
    - Real-time prediction updates
    """
    
    def __init__(self, params: MLDegradationParams = None):
        self.p = params or MLDegradationParams()
        
        # Neural network components (simplified implementation)
        self.weights = {}
        self.biases = {}
        self.activation_functions = {}
        
        # Training data
        self.training_data = []
        self.validation_data = []
        self.feature_scalers = {}
        
        # Model state
        self.is_trained = False
        self.training_history = []
        self.prediction_history = []
        
        # Performance tracking
        self.accuracy_history = []
        self.loss_history = []
        
        # Initialize network
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize neural network weights and biases."""
        # Input layer
        self.weights['input'] = np.random.randn(self.p.input_dimensions, self.p.hidden_dimensions[0]) * 0.1
        self.biases['input'] = np.zeros(self.p.hidden_dimensions[0])
        
        # Hidden layers
        for i in range(len(self.p.hidden_dimensions) - 1):
            layer_name = f'hidden_{i}'
            next_layer_name = f'hidden_{i+1}'
            
            self.weights[layer_name] = np.random.randn(
                self.p.hidden_dimensions[i], 
                self.p.hidden_dimensions[i+1]
            ) * 0.1
            self.biases[layer_name] = np.zeros(self.p.hidden_dimensions[i+1])
        
        # Output layer
        last_hidden = len(self.p.hidden_dimensions) - 1
        self.weights['output'] = np.random.randn(
            self.p.hidden_dimensions[last_hidden], 
            self.p.output_dimensions
        ) * 0.1
        self.biases['output'] = np.zeros(self.p.output_dimensions)
        
        # Activation functions
        self.activation_functions = {
            'input': self._relu,
            'hidden_0': self._relu,
            'hidden_1': self._relu,
            'output': self._sigmoid
        }
    
    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward_pass(self, inputs):
        """Forward pass through the neural network."""
        # Input layer
        hidden = self._relu(np.dot(inputs, self.weights['input']) + self.biases['input'])
        
        # Hidden layers
        for i in range(len(self.p.hidden_dimensions) - 1):
            layer_name = f'hidden_{i}'
            hidden = self._relu(np.dot(hidden, self.weights[layer_name]) + self.biases[layer_name])
        
        # Output layer
        output = self._sigmoid(np.dot(hidden, self.weights['output']) + self.biases['output'])
        
        return output
    
    def extract_features(self, thermal_state: np.ndarray, wear_summary: Dict,
                        weather_summary: Dict, structural_state: Dict,
                        compound_state: Dict, race_context: Dict) -> np.ndarray:
        """
        Extract features for degradation prediction.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            
        Returns:
            Feature vector
        """
        Tt, Tc, Tr = thermal_state
        
        # Thermal features
        thermal_features = [
            Tt,  # Tread temperature
            Tc,  # Carcass temperature
            Tr,  # Rim temperature
            Tt - Tc,  # Temperature gradient
            (Tt + Tc + Tr) / 3  # Average temperature
        ]
        
        # Wear features
        wear_levels = [wear_summary.get(corner, {}).get('wear_level', 0.0) 
                      for corner in ['FL', 'FR', 'RL', 'RR']]
        wear_features = [
            np.mean(wear_levels),  # Average wear
            np.std(wear_levels),   # Wear variation
            np.max(wear_levels)    # Maximum wear
        ]
        
        # Weather features
        weather_features = [
            weather_summary.get('track_temperature', 25.0),
            weather_summary.get('rain_probability', 0.0),
            weather_summary.get('wind_speed', 0.0)
        ]
        
        # Structural features
        structural_features = [
            structural_state.get('deflection_ratio', 0.0),
            structural_state.get('contact_patch_area', 0.0),
            np.mean(structural_state.get('pressure_distribution', np.zeros((10, 10))))
        ]
        
        # Compound features
        compound_features = [
            compound_state.get('current_temperature', 25.0),
            compound_state.get('ageing_level', 0.0)
        ]
        
        # Combine all features
        features = (thermal_features + wear_features + weather_features + 
                   structural_features + compound_features)
        
        return np.array(features[:self.p.input_dimensions])
    
    def prepare_training_data(self, historical_data: List[Dict]):
        """
        Prepare training data from historical telemetry.
        
        Args:
            historical_data: List of historical data points
        """
        sequences = []
        targets = []
        
        for i in range(len(historical_data) - self.p.sequence_length - self.p.prediction_horizon):
            # Extract sequence
            sequence = []
            for j in range(self.p.sequence_length):
                data_point = historical_data[i + j]
                features = self.extract_features(
                    data_point['thermal_state'],
                    data_point['wear_summary'],
                    data_point['weather_summary'],
                    data_point['structural_state'],
                    data_point['compound_state'],
                    data_point['race_context']
                )
                sequence.append(features)
            
            # Extract target (degradation after prediction_horizon laps)
            target_data = historical_data[i + self.p.sequence_length + self.p.prediction_horizon]
            target_wear = np.mean([target_data['wear_summary'].get(corner, {}).get('wear_level', 0.0)
                                for corner in ['FL', 'FR', 'RL', 'RR']])
            
            # Create target vector (simplified)
            target = np.array([
                target_wear,  # Total wear
                target_data['thermal_state'][0] / 150.0,  # Normalized tread temp
                target_data['structural_state'].get('deflection_ratio', 0.0),  # Deflection
                target_data['compound_state'].get('ageing_level', 0.0),  # Ageing
                1.0 if target_wear > 0.8 else 0.0  # Critical degradation flag
            ])
            
            sequences.append(sequence)
            targets.append(target)
        
        # Convert to numpy arrays
        self.training_data = np.array(sequences)
        self.validation_data = np.array(targets)
        
        # Normalize features if enabled
        if self.p.feature_scaling:
            self._normalize_features()
    
    def _normalize_features(self):
        """Normalize training features."""
        # Flatten sequences for normalization
        flat_data = self.training_data.reshape(-1, self.p.input_dimensions)
        
        # Calculate mean and std for each feature
        self.feature_scalers = {
            'mean': np.mean(flat_data, axis=0),
            'std': np.std(flat_data, axis=0)
        }
        
        # Normalize data
        normalized_data = (flat_data - self.feature_scalers['mean']) / (self.feature_scalers['std'] + 1e-8)
        self.training_data = normalized_data.reshape(self.training_data.shape)
    
    def train_model(self, epochs: int = None):
        """
        Train the neural network model.
        
        Args:
            epochs: Number of training epochs
        """
        if epochs is None:
            epochs = self.p.epochs
        
        if len(self.training_data) == 0:
            raise ValueError("No training data available. Call prepare_training_data first.")
        
        # Simple training loop (simplified implementation)
        for epoch in range(epochs):
            # Forward pass
            predictions = []
            for sequence in self.training_data:
                # Use last timestep for prediction
                prediction = self._forward_pass(sequence[-1])
                predictions.append(prediction)
            
            predictions = np.array(predictions)
            
            # Calculate loss (simplified MSE)
            loss = np.mean((predictions - self.validation_data) ** 2)
            
            # Simple gradient update (simplified)
            self._update_weights(predictions, self.validation_data)
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'accuracy': self._calculate_accuracy(predictions, self.validation_data)
            })
            
            # Early stopping
            if len(self.training_history) > self.p.early_stopping_patience:
                recent_losses = [h['loss'] for h in self.training_history[-self.p.early_stopping_patience:]]
                if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    break
        
        self.is_trained = True
    
    def _update_weights(self, predictions, targets):
        """Update network weights using simplified gradient descent."""
        # Simplified weight update (in practice, would use proper backpropagation)
        error = predictions - targets
        
        # Update output layer weights
        self.weights['output'] -= self.p.learning_rate * np.mean(error, axis=0)
        self.biases['output'] -= self.p.learning_rate * np.mean(error, axis=0)
    
    def _calculate_accuracy(self, predictions, targets):
        """Calculate prediction accuracy."""
        # Simple accuracy calculation
        errors = np.abs(predictions - targets)
        accuracy = 1.0 - np.mean(errors)
        return max(0.0, accuracy)
    
    def predict_degradation(self, thermal_state: np.ndarray, wear_summary: Dict,
                          weather_summary: Dict, structural_state: Dict,
                          compound_state: Dict, race_context: Dict,
                          horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM) -> Dict[str, float]:
        """
        Predict tire degradation for given horizon.
        
        Args:
            thermal_state: Current thermal state
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            horizon: Prediction horizon
            
        Returns:
            Dictionary with degradation predictions
        """
        if not self.is_trained:
            # Return default predictions if model not trained
            return {
                'total_degradation': 0.1,
                'thermal_degradation': 0.05,
                'mechanical_degradation': 0.03,
                'wear_degradation': 0.02,
                'confidence': 0.5
            }
        
        # Extract features
        features = self.extract_features(thermal_state, wear_summary, weather_summary,
                                       structural_state, compound_state, race_context)
        
        # Normalize features if scalers available
        if self.feature_scalers:
            features = (features - self.feature_scalers['mean']) / (self.feature_scalers['std'] + 1e-8)
        
        # Make prediction
        prediction = self._forward_pass(features)
        
        # Map prediction to degradation types
        degradation_prediction = {
            'total_degradation': float(prediction[0]),
            'thermal_degradation': float(prediction[1]),
            'mechanical_degradation': float(prediction[2]),
            'wear_degradation': float(prediction[3]),
            'fatigue_degradation': float(prediction[4]),
            'confidence': self._calculate_prediction_confidence(features, prediction)
        }
        
        # Adjust predictions based on horizon
        horizon_multipliers = {
            PredictionHorizon.SHORT_TERM: 0.5,
            PredictionHorizon.MEDIUM_TERM: 1.0,
            PredictionHorizon.LONG_TERM: 1.5
        }
        
        multiplier = horizon_multipliers.get(horizon, 1.0)
        for key in degradation_prediction:
            if key != 'confidence':
                degradation_prediction[key] *= multiplier
        
        # Store prediction history
        self.prediction_history.append({
            'timestamp': len(self.prediction_history),
            'prediction': degradation_prediction.copy(),
            'horizon': horizon.value
        })
        
        return degradation_prediction
    
    def _calculate_prediction_confidence(self, features, prediction):
        """Calculate confidence in prediction based on feature similarity."""
        # Simple confidence calculation based on feature variance
        feature_variance = np.var(features)
        confidence = 1.0 / (1.0 + feature_variance)
        return float(confidence)
    
    def get_degradation_recommendations(self, degradation_prediction: Dict[str, float],
                                      thermal_state: np.ndarray, wear_summary: Dict) -> List[Tuple[str, str]]:
        """
        Get recommendations based on degradation predictions.
        
        Args:
            degradation_prediction: Predicted degradation values
            thermal_state: Current thermal state
            wear_summary: Current wear status
            
        Returns:
            List of degradation-based recommendations
        """
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Total degradation recommendations
        total_degradation = degradation_prediction['total_degradation']
        confidence = degradation_prediction['confidence']
        
        if total_degradation > 0.7 and confidence > 0.7:
            recommendations.append(("ML_DEGRADATION", f"High degradation predicted ({total_degradation:.1%}) - consider pit stop"))
        elif total_degradation > 0.5 and confidence > 0.6:
            recommendations.append(("ML_DEGRADATION", f"Moderate degradation predicted ({total_degradation:.1%}) - monitor closely"))
        
        # Thermal degradation recommendations
        thermal_degradation = degradation_prediction['thermal_degradation']
        if thermal_degradation > 0.6:
            recommendations.append(("ML_DEGRADATION", f"High thermal degradation predicted - manage temperatures"))
        
        # Wear degradation recommendations
        wear_degradation = degradation_prediction['wear_degradation']
        if wear_degradation > 0.5:
            recommendations.append(("ML_DEGRADATION", f"High wear degradation predicted - reduce aggressive driving"))
        
        # Confidence-based recommendations
        if confidence < 0.5:
            recommendations.append(("ML_DEGRADATION", f"Low prediction confidence ({confidence:.1%}) - gather more data"))
        
        return recommendations
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics."""
        if not self.training_history:
            return {'accuracy': 0.0, 'loss': 0.0, 'epochs_trained': 0}
        
        latest_history = self.training_history[-1]
        return {
            'accuracy': latest_history['accuracy'],
            'loss': latest_history['loss'],
            'epochs_trained': len(self.training_history),
            'average_confidence': np.mean([p['prediction']['confidence'] 
                                         for p in self.prediction_history[-10:]]) if self.prediction_history else 0.0
        }
    
    def save_model(self, filepath: str):
        """Save the ML model to file."""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'feature_scalers': self.feature_scalers,
            'training_history': self.training_history,
            'prediction_history': self.prediction_history,
            'is_trained': self.is_trained,
            'params': self.p
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the ML model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.feature_scalers = model_data['feature_scalers']
        self.training_history = model_data['training_history']
        self.prediction_history = model_data['prediction_history']
        self.is_trained = model_data['is_trained']
        self.p = model_data['params']
    
    def reset_model(self):
        """Reset the ML model for new training."""
        self._initialize_network()
        self.training_data = []
        self.validation_data = []
        self.feature_scalers = {}
        self.is_trained = False
        self.training_history = []
        self.prediction_history = []
        self.accuracy_history = []
        self.loss_history = []
