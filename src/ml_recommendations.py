from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pickle
import json
from collections import deque
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

class RecommendationType(Enum):
    """Types of ML recommendations."""
    THERMAL_MANAGEMENT = "thermal_management"
    WEAR_OPTIMIZATION = "wear_optimization"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    COMPOUND_SELECTION = "compound_selection"
    PRESSURE_ADJUSTMENT = "pressure_adjustment"
    DRIVING_STYLE = "driving_style"
    PIT_STOP_TIMING = "pit_stop_timing"
    RISK_ASSESSMENT = "risk_assessment"

class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Important action needed
    MEDIUM = "medium"         # Recommended action
    LOW = "low"               # Optional action
    INFO = "info"             # Informational only

@dataclass
class MLRecommendationParams:
    """Parameters for ML recommendation system."""
    # Ensemble parameters
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    random_state: int = 42
    
    # Training parameters
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    min_samples_for_training: int = 100
    
    # Recommendation parameters
    confidence_threshold: float = 0.7
    priority_threshold: float = 0.8
    max_recommendations: int = 5
    
    # Feature engineering
    feature_scaling: bool = True
    feature_selection: bool = True
    feature_dimensions: int = 30
    
    # Model update parameters
    update_frequency: int = 50
    performance_threshold: float = 0.8

class MLRecommendationEngine:
    """
    Real-time ML recommendation engine using ensemble models.
    
    Features:
    - Ensemble of multiple ML models (Random Forest, Gradient Boosting, SVM, Logistic Regression)
    - Real-time recommendation generation
    - Confidence-based recommendation filtering
    - Priority-based recommendation ranking
    - Continuous learning and model updates
    - Performance tracking and validation
    """
    
    def __init__(self, params: MLRecommendationParams = None):
        self.p = params or MLRecommendationParams()
        
        # Ensemble models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.p.n_estimators,
                max_depth=self.p.max_depth,
                random_state=self.p.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=self.p.n_estimators,
                learning_rate=self.p.learning_rate,
                random_state=self.p.random_state
            ),
            'svm': SVC(probability=True, random_state=self.p.random_state),
            'logistic_regression': LogisticRegression(random_state=self.p.random_state)
        }
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.feature_selector = None
        
        # Training data
        self.training_data = []
        self.validation_data = []
        self.feature_history = []
        self.recommendation_history = []
        
        # Model performance
        self.model_performance = {}
        self.model_weights = {}
        
        # Recommendation tracking
        self.recommendation_feedback = []
        self.success_rate_history = []
        
        # Initialize model weights
        self._initialize_model_weights()
    
    def _initialize_model_weights(self):
        """Initialize model weights for ensemble voting."""
        for model_name in self.models:
            self.model_weights[model_name] = 1.0 / len(self.models)
    
    def extract_recommendation_features(self, thermal_state: np.ndarray, wear_summary: Dict,
                                      weather_summary: Dict, structural_state: Dict,
                                      compound_state: Dict, race_context: Dict,
                                      driver_profile: Dict, ml_predictions: Dict) -> np.ndarray:
        """
        Extract features for recommendation generation.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            driver_profile: Driver behavior profile
            ml_predictions: ML model predictions
            
        Returns:
            Feature vector for recommendation generation
        """
        Tt, Tc, Tr = thermal_state
        
        # Thermal features
        thermal_features = [
            Tt,  # Tread temperature
            Tc,  # Carcass temperature
            Tr,  # Rim temperature
            Tt - Tc,  # Temperature gradient
            (Tt + Tc + Tr) / 3,  # Average temperature
            np.std([Tt, Tc, Tr])  # Temperature variability
        ]
        
        # Wear features
        wear_levels = [wear_summary.get(corner, {}).get('wear_level', 0.0) 
                      for corner in ['FL', 'FR', 'RL', 'RR']]
        wear_features = [
            np.mean(wear_levels),  # Average wear
            np.std(wear_levels),   # Wear variation
            np.max(wear_levels),   # Maximum wear
            np.min(wear_levels)    # Minimum wear
        ]
        
        # Weather features
        weather_features = [
            weather_summary.get('track_temperature', 25.0) / 50.0,
            weather_summary.get('rain_probability', 0.0),
            weather_summary.get('wind_speed', 0.0) / 50.0,
            weather_summary.get('humidity', 0.5)
        ]
        
        # Structural features
        structural_features = [
            structural_state.get('deflection_ratio', 0.0),
            structural_state.get('contact_patch_area', 0.0),
            np.mean(structural_state.get('pressure_distribution', np.zeros((10, 10))))
        ]
        
        # Compound features
        compound_features = [
            compound_state.get('current_temperature', 25.0) / 150.0,
            compound_state.get('ageing_level', 0.0),
            compound_state.get('current_pressure', 1.5) / 3.0
        ]
        
        # Race context features
        race_features = [
            race_context.get('current_lap', 0) / 100.0,
            race_context.get('total_laps', 50) / 100.0,
            race_context.get('position', 1) / 20.0,
            race_context.get('gap_to_leader', 0.0) / 100.0
        ]
        
        # Driver profile features
        driver_features = [
            driver_profile.get('confidence', 0.0),
            1.0 if driver_profile.get('behavior_pattern') == 'thermal_aggressor' else 0.0,
            1.0 if driver_profile.get('behavior_pattern') == 'wear_optimizer' else 0.0,
            1.0 if driver_profile.get('behavior_pattern') == 'adaptive_driver' else 0.0
        ]
        
        # ML prediction features
        ml_features = [
            ml_predictions.get('degradation_prediction', {}).get('total_degradation', 0.0),
            ml_predictions.get('degradation_prediction', {}).get('confidence', 0.0),
            ml_predictions.get('strategy_prediction', {}).get('confidence', 0.0)
        ]
        
        # Combine all features
        features = (thermal_features + wear_features + weather_features + 
                   structural_features + compound_features + race_features + 
                   driver_features + ml_features)
        
        return np.array(features[:self.p.feature_dimensions])
    
    def prepare_training_data(self, historical_data: List[Dict]):
        """
        Prepare training data from historical recommendations and outcomes.
        
        Args:
            historical_data: List of historical data with recommendations and outcomes
        """
        features = []
        labels = []
        
        for data_point in historical_data:
            # Extract features
            feature_vector = self.extract_recommendation_features(
                data_point['thermal_state'],
                data_point['wear_summary'],
                data_point['weather_summary'],
                data_point['structural_state'],
                data_point['compound_state'],
                data_point['race_context'],
                data_point['driver_profile'],
                data_point['ml_predictions']
            )
            
            # Extract label (success/failure of recommendation)
            outcome = data_point.get('outcome', {})
            success = outcome.get('success', False)
            label = 1 if success else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        # Convert to numpy arrays
        self.training_data = np.array(features)
        self.validation_data = np.array(labels)
        
        # Scale features
        if self.p.feature_scaling:
            self.training_data = self.feature_scaler.fit_transform(self.training_data)
    
    def train_ensemble_models(self):
        """Train all ensemble models."""
        if len(self.training_data) < self.p.min_samples_for_training:
            return
        
        # Split data for training and validation
        split_idx = int(len(self.training_data) * self.p.train_test_split)
        X_train = self.training_data[:split_idx]
        y_train = self.validation_data[:split_idx]
        X_val = self.training_data[split_idx:]
        y_val = self.validation_data[split_idx:]
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                
                # Evaluate model performance
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                
                # Store performance metrics
                self.model_performance[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                }
                
                # Update model weights based on performance
                self.model_weights[model_name] = accuracy
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Normalize model weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
    
    def generate_recommendations(self, thermal_state: np.ndarray, wear_summary: Dict,
                               weather_summary: Dict, structural_state: Dict,
                               compound_state: Dict, race_context: Dict,
                               driver_profile: Dict, ml_predictions: Dict) -> List[Dict[str, Any]]:
        """
        Generate real-time ML recommendations using ensemble models.
        
        Args:
            thermal_state: Current thermal state
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            driver_profile: Driver behavior profile
            ml_predictions: ML model predictions
            
        Returns:
            List of recommendations with confidence and priority
        """
        # Extract features
        features = self.extract_recommendation_features(
            thermal_state, wear_summary, weather_summary, structural_state,
            compound_state, race_context, driver_profile, ml_predictions
        )
        
        # Scale features
        if hasattr(self.feature_scaler, 'mean_'):
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Generate predictions from all models
        ensemble_predictions = {}
        ensemble_probabilities = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    ensemble_probabilities[model_name] = proba
                else:
                    prediction = model.predict(features_scaled)[0]
                    ensemble_probabilities[model_name] = [1 - prediction, prediction]
                
                ensemble_predictions[model_name] = np.argmax(ensemble_probabilities[model_name])
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                ensemble_probabilities[model_name] = [0.5, 0.5]
                ensemble_predictions[model_name] = 0
        
        # Calculate ensemble prediction
        weighted_probabilities = np.zeros(2)
        for model_name, proba in ensemble_probabilities.items():
            weight = self.model_weights.get(model_name, 0.0)
            weighted_probabilities += weight * np.array(proba)
        
        ensemble_confidence = np.max(weighted_probabilities)
        ensemble_prediction = np.argmax(weighted_probabilities)
        
        # Generate recommendations based on prediction
        recommendations = []
        
        if ensemble_confidence > self.p.confidence_threshold:
            recommendations = self._generate_specific_recommendations(
                thermal_state, wear_summary, weather_summary, structural_state,
                compound_state, race_context, driver_profile, ml_predictions,
                ensemble_confidence
            )
        
        # Store recommendation history
        self.recommendation_history.append({
            'timestamp': len(self.recommendation_history),
            'features': features,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': ensemble_confidence,
            'recommendations': recommendations
        })
        
        return recommendations
    
    def _generate_specific_recommendations(self, thermal_state: np.ndarray, wear_summary: Dict,
                                         weather_summary: Dict, structural_state: Dict,
                                         compound_state: Dict, race_context: Dict,
                                         driver_profile: Dict, ml_predictions: Dict,
                                         confidence: float) -> List[Dict[str, Any]]:
        """Generate specific recommendations based on current state."""
        recommendations = []
        Tt, Tc, Tr = thermal_state
        
        # Thermal management recommendations
        if Tt > 110:
            recommendations.append({
                'type': RecommendationType.THERMAL_MANAGEMENT.value,
                'priority': RecommendationPriority.CRITICAL.value,
                'message': f"Critical: Tread temperature {Tt:.1f}°C - immediate cooling required",
                'confidence': confidence,
                'action': 'reduce_pace'
            })
        elif Tt > 105:
            recommendations.append({
                'type': RecommendationType.THERMAL_MANAGEMENT.value,
                'priority': RecommendationPriority.HIGH.value,
                'message': f"High: Tread temperature {Tt:.1f}°C - manage thermal load",
                'confidence': confidence,
                'action': 'conserve_tires'
            })
        
        # Wear optimization recommendations
        avg_wear = np.mean([wear_summary.get(corner, {}).get('wear_level', 0.0) 
                           for corner in ['FL', 'FR', 'RL', 'RR']])
        
        if avg_wear > 0.8:
            recommendations.append({
                'type': RecommendationType.WEAR_OPTIMIZATION.value,
                'priority': RecommendationPriority.CRITICAL.value,
                'message': f"Critical: Average wear {avg_wear:.1%} - pit stop required",
                'confidence': confidence,
                'action': 'pit_stop'
            })
        elif avg_wear > 0.6:
            recommendations.append({
                'type': RecommendationType.WEAR_OPTIMIZATION.value,
                'priority': RecommendationPriority.HIGH.value,
                'message': f"High: Average wear {avg_wear:.1%} - monitor closely",
                'confidence': confidence,
                'action': 'reduce_aggression'
            })
        
        # Strategy adjustment recommendations
        if ml_predictions.get('degradation_prediction', {}).get('total_degradation', 0.0) > 0.7:
            recommendations.append({
                'type': RecommendationType.STRATEGY_ADJUSTMENT.value,
                'priority': RecommendationPriority.HIGH.value,
                'message': "High degradation predicted - adjust strategy",
                'confidence': confidence,
                'action': 'strategy_change'
            })
        
        # Driver-specific recommendations
        behavior_pattern = driver_profile.get('behavior_pattern')
        if behavior_pattern == 'thermal_aggressor' and Tt > 100:
            recommendations.append({
                'type': RecommendationType.DRIVING_STYLE.value,
                'priority': RecommendationPriority.MEDIUM.value,
                'message': "Thermal aggressor: reduce aggressive driving",
                'confidence': confidence,
                'action': 'smooth_inputs'
            })
        
        # Weather-based recommendations
        if weather_summary.get('rain_probability', 0.0) > 0.5:
            recommendations.append({
                'type': RecommendationType.COMPOUND_SELECTION.value,
                'priority': RecommendationPriority.HIGH.value,
                'message': "High rain probability - consider wet compound",
                'confidence': confidence,
                'action': 'compound_change'
            })
        
        # Limit number of recommendations
        recommendations = recommendations[:self.p.max_recommendations]
        
        return recommendations
    
    def update_recommendation_feedback(self, recommendation_id: int, success: bool, 
                                     performance_improvement: float = 0.0):
        """
        Update recommendation feedback for continuous learning.
        
        Args:
            recommendation_id: ID of the recommendation
            success: Whether the recommendation was successful
            performance_improvement: Performance improvement achieved
        """
        feedback = {
            'recommendation_id': recommendation_id,
            'success': success,
            'performance_improvement': performance_improvement,
            'timestamp': len(self.recommendation_feedback)
        }
        
        self.recommendation_feedback.append(feedback)
        
        # Update success rate
        recent_feedback = self.recommendation_feedback[-100:]  # Last 100 feedbacks
        success_rate = sum(f['success'] for f in recent_feedback) / len(recent_feedback)
        self.success_rate_history.append(success_rate)
        
        # Retrain models if performance drops
        if len(self.success_rate_history) > 10:
            recent_success_rate = np.mean(self.success_rate_history[-10:])
            if recent_success_rate < self.p.performance_threshold:
                self.train_ensemble_models()
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get comprehensive recommendation system summary."""
        return {
            'total_recommendations': len(self.recommendation_history),
            'model_performance': self.model_performance,
            'model_weights': self.model_weights,
            'success_rate': np.mean(self.success_rate_history[-50:]) if self.success_rate_history else 0.0,
            'average_confidence': np.mean([r['ensemble_confidence'] for r in self.recommendation_history[-50:]]) if self.recommendation_history else 0.0,
            'recommendation_types': self._get_recommendation_type_distribution(),
            'priority_distribution': self._get_priority_distribution()
        }
    
    def _get_recommendation_type_distribution(self) -> Dict[str, int]:
        """Get distribution of recommendation types."""
        distribution = {}
        for rec_history in self.recommendation_history:
            for rec in rec_history['recommendations']:
                rec_type = rec['type']
                distribution[rec_type] = distribution.get(rec_type, 0) + 1
        return distribution
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of recommendation priorities."""
        distribution = {}
        for rec_history in self.recommendation_history:
            for rec in rec_history['recommendations']:
                priority = rec['priority']
                distribution[priority] = distribution.get(priority, 0) + 1
        return distribution
    
    def save_model(self, filepath: str):
        """Save the ML recommendation model to file."""
        model_data = {
            'models': self.models,
            'feature_scaler': self.feature_scaler,
            'model_performance': self.model_performance,
            'model_weights': self.model_weights,
            'training_data': self.training_data,
            'validation_data': self.validation_data,
            'recommendation_history': self.recommendation_history,
            'recommendation_feedback': self.recommendation_feedback,
            'success_rate_history': self.success_rate_history,
            'params': self.p
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the ML recommendation model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.feature_scaler = model_data['feature_scaler']
        self.model_performance = model_data['model_performance']
        self.model_weights = model_data['model_weights']
        self.training_data = model_data['training_data']
        self.validation_data = model_data['validation_data']
        self.recommendation_history = model_data['recommendation_history']
        self.recommendation_feedback = model_data['recommendation_feedback']
        self.success_rate_history = model_data['success_rate_history']
        self.p = model_data['params']
    
    def reset_model(self):
        """Reset the ML recommendation model for new training."""
        # Reinitialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.p.n_estimators,
                max_depth=self.p.max_depth,
                random_state=self.p.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=self.p.n_estimators,
                learning_rate=self.p.learning_rate,
                random_state=self.p.random_state
            ),
            'svm': SVC(probability=True, random_state=self.p.random_state),
            'logistic_regression': LogisticRegression(random_state=self.p.random_state)
        }
        
        self.feature_scaler = StandardScaler()
        self.training_data = []
        self.validation_data = []
        self.recommendation_history = []
        self.recommendation_feedback = []
        self.success_rate_history = []
        self.model_performance = {}
        self._initialize_model_weights()
