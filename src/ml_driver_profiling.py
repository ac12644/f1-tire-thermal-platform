from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pickle
import json
from collections import deque
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class DriverBehaviorPattern(Enum):
    """Driver behavior patterns identified by ML."""
    THERMAL_AGGRESSOR = "thermal_aggressor"      # High thermal generation
    THERMAL_CONSERVATOR = "thermal_conservator"  # Low thermal generation
    WEAR_OPTIMIZER = "wear_optimizer"           # Excellent wear management
    WEAR_AGGRESSOR = "wear_aggressor"           # High wear generation
    ADAPTIVE_DRIVER = "adaptive_driver"         # Adapts well to conditions
    CONSISTENT_DRIVER = "consistent_driver"     # Very consistent patterns
    VARIABLE_DRIVER = "variable_driver"         # High variability in behavior

class LearningPhase(Enum):
    """Learning phases for adaptive profiling."""
    INITIAL = "initial"           # Initial data collection
    LEARNING = "learning"        # Active learning phase
    MATURE = "mature"            # Mature model with good accuracy
    ADAPTING = "adapting"        # Adapting to new patterns

@dataclass
class MLDriverProfilingParams:
    """Parameters for ML driver profiling."""
    # Clustering parameters
    n_clusters: int = 7          # Number of behavior clusters
    min_samples: int = 50        # Minimum samples for pattern recognition
    cluster_update_frequency: int = 100  # Update clusters every N samples
    
    # Feature extraction
    feature_window_size: int = 20  # Window size for feature extraction
    feature_dimensions: int = 25   # Number of features
    
    # Learning parameters
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.1  # Threshold for pattern adaptation
    confidence_threshold: float = 0.7  # Minimum confidence for predictions
    
    # Pattern recognition
    pattern_similarity_threshold: float = 0.8
    pattern_update_frequency: int = 50
    
    # Performance tracking
    performance_window_size: int = 100
    adaptation_rate: float = 0.1

class MLDriverProfiler:
    """
    Machine Learning-based adaptive driver profiling using telemetry data analysis.
    
    Features:
    - Unsupervised learning for behavior pattern recognition
    - Adaptive clustering for driver behavior classification
    - Real-time pattern adaptation based on new data
    - Performance prediction based on behavior patterns
    - Confidence-based recommendations
    - Historical pattern analysis
    """
    
    def __init__(self, params: MLDriverProfilingParams = None):
        self.p = params or MLDriverProfilingParams()
        
        # ML components
        self.kmeans_model = KMeans(n_clusters=self.p.n_clusters, random_state=42)
        self.feature_scaler = StandardScaler()
        
        # Data storage
        self.telemetry_data = deque(maxlen=10000)
        self.feature_history = deque(maxlen=1000)
        self.pattern_history = deque(maxlen=500)
        
        # Driver profiles
        self.driver_profiles = {}
        self.behavior_patterns = {}
        self.performance_patterns = {}
        
        # Learning state
        self.learning_phase = LearningPhase.INITIAL
        self.model_confidence = 0.0
        self.adaptation_count = 0
        
        # Performance tracking
        self.accuracy_history = []
        self.pattern_confidence_history = []
        
    def extract_driver_features(self, thermal_state: np.ndarray, wear_summary: Dict,
                              weather_summary: Dict, structural_state: Dict,
                              compound_state: Dict, race_context: Dict,
                              driver_actions: List[str]) -> np.ndarray:
        """
        Extract features for driver behavior analysis.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            driver_actions: Recent driver actions
            
        Returns:
            Feature vector for driver behavior
        """
        Tt, Tc, Tr = thermal_state
        
        # Thermal behavior features
        thermal_features = [
            Tt,  # Tread temperature
            Tc,  # Carcass temperature
            Tr,  # Rim temperature
            Tt - Tc,  # Temperature gradient
            (Tt + Tc + Tr) / 3,  # Average temperature
            np.std([Tt, Tc, Tr])  # Temperature variability
        ]
        
        # Wear behavior features
        wear_levels = [wear_summary.get(corner, {}).get('wear_level', 0.0) 
                      for corner in ['FL', 'FR', 'RL', 'RR']]
        wear_features = [
            np.mean(wear_levels),  # Average wear
            np.std(wear_levels),   # Wear variation
            np.max(wear_levels),   # Maximum wear
            np.min(wear_levels)    # Minimum wear
        ]
        
        # Driving behavior features
        driving_features = [
            race_context.get('speed', 0.0) / 100.0,  # Normalized speed
            race_context.get('lateral_g', 0.0),       # Lateral acceleration
            race_context.get('longitudinal_g', 0.0),  # Longitudinal acceleration
            len(driver_actions) / 10.0  # Action frequency
        ]
        
        # Environmental adaptation features
        env_features = [
            weather_summary.get('track_temperature', 25.0) / 50.0,  # Normalized track temp
            weather_summary.get('rain_probability', 0.0),           # Rain probability
            weather_summary.get('wind_speed', 0.0) / 50.0          # Normalized wind speed
        ]
        
        # Structural behavior features
        structural_features = [
            structural_state.get('deflection_ratio', 0.0),
            structural_state.get('contact_patch_area', 0.0),
            np.mean(structural_state.get('pressure_distribution', np.zeros((10, 10))))
        ]
        
        # Compound management features
        compound_features = [
            compound_state.get('current_temperature', 25.0) / 150.0,
            compound_state.get('ageing_level', 0.0),
            compound_state.get('current_pressure', 1.5) / 3.0
        ]
        
        # Action pattern features
        action_features = [
            driver_actions.count('increase_pace') / max(1, len(driver_actions)),
            driver_actions.count('decrease_pace') / max(1, len(driver_actions)),
            driver_actions.count('pit_stop') / max(1, len(driver_actions)),
            driver_actions.count('conserve_tires') / max(1, len(driver_actions))
        ]
        
        # Combine all features
        features = (thermal_features + wear_features + driving_features + 
                   env_features + structural_features + compound_features + action_features)
        
        return np.array(features[:self.p.feature_dimensions])
    
    def add_telemetry_data(self, driver_id: str, thermal_state: np.ndarray, wear_summary: Dict,
                          weather_summary: Dict, structural_state: Dict, compound_state: Dict,
                          race_context: Dict, driver_actions: List[str], performance_metrics: Dict):
        """
        Add new telemetry data for driver profiling.
        
        Args:
            driver_id: Driver identifier
            thermal_state: Current thermal state
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            structural_state: Current structural state
            compound_state: Current compound state
            race_context: Race context
            driver_actions: Recent driver actions
            performance_metrics: Performance metrics
        """
        # Extract features
        features = self.extract_driver_features(thermal_state, wear_summary, weather_summary,
                                              structural_state, compound_state, race_context, driver_actions)
        
        # Store data
        data_point = {
            'driver_id': driver_id,
            'timestamp': len(self.telemetry_data),
            'features': features,
            'thermal_state': thermal_state,
            'wear_summary': wear_summary,
            'weather_summary': weather_summary,
            'structural_state': structural_state,
            'compound_state': compound_state,
            'race_context': race_context,
            'driver_actions': driver_actions,
            'performance_metrics': performance_metrics
        }
        
        self.telemetry_data.append(data_point)
        self.feature_history.append(features)
        
        # Update learning phase
        self._update_learning_phase()
        
        # Update driver profile
        self._update_driver_profile(driver_id, data_point)
        
        # Check for pattern adaptation
        if len(self.feature_history) % self.p.pattern_update_frequency == 0:
            self._adapt_patterns()
    
    def _update_learning_phase(self):
        """Update learning phase based on data availability."""
        data_size = len(self.telemetry_data)
        
        if data_size < self.p.min_samples:
            self.learning_phase = LearningPhase.INITIAL
        elif data_size < self.p.min_samples * 2:
            self.learning_phase = LearningPhase.LEARNING
        elif self.model_confidence > self.p.confidence_threshold:
            self.learning_phase = LearningPhase.MATURE
        else:
            self.learning_phase = LearningPhase.ADAPTING
    
    def _update_driver_profile(self, driver_id: str, data_point: Dict):
        """Update individual driver profile."""
        if driver_id not in self.driver_profiles:
            self.driver_profiles[driver_id] = {
                'data_points': [],
                'behavior_pattern': None,
                'confidence': 0.0,
                'adaptation_count': 0,
                'performance_history': []
            }
        
        profile = self.driver_profiles[driver_id]
        profile['data_points'].append(data_point)
        
        # Keep only recent data points
        if len(profile['data_points']) > self.p.performance_window_size:
            profile['data_points'] = profile['data_points'][-self.p.performance_window_size:]
        
        # Update behavior pattern
        if len(profile['data_points']) >= self.p.min_samples:
            self._classify_driver_behavior(driver_id)
    
    def _classify_driver_behavior(self, driver_id: str):
        """Classify driver behavior using clustering."""
        profile = self.driver_profiles[driver_id]
        
        if len(profile['data_points']) < self.p.min_samples:
            return
        
        # Extract features for clustering
        features = np.array([dp['features'] for dp in profile['data_points']])
        
        # Scale features
        if hasattr(self.feature_scaler, 'mean_'):
            features_scaled = self.feature_scaler.transform(features)
        else:
            features_scaled = features
        
        # Perform clustering
        if len(features_scaled) >= self.p.n_clusters:
            try:
                clusters = self.kmeans_model.fit_predict(features_scaled)
                
                # Determine dominant behavior pattern
                cluster_counts = np.bincount(clusters)
                dominant_cluster = np.argmax(cluster_counts)
                
                # Map cluster to behavior pattern
                behavior_pattern = self._cluster_to_behavior_pattern(dominant_cluster)
                
                # Update profile
                profile['behavior_pattern'] = behavior_pattern
                profile['confidence'] = cluster_counts[dominant_cluster] / len(clusters)
                
                # Store pattern history
                self.pattern_history.append({
                    'driver_id': driver_id,
                    'pattern': behavior_pattern,
                    'confidence': profile['confidence'],
                    'timestamp': len(self.pattern_history)
                })
                
            except Exception as e:
                print(f"Error in clustering for driver {driver_id}: {e}")
    
    def _cluster_to_behavior_pattern(self, cluster_id: int) -> DriverBehaviorPattern:
        """Map cluster ID to behavior pattern."""
        pattern_mapping = {
            0: DriverBehaviorPattern.THERMAL_AGGRESSOR,
            1: DriverBehaviorPattern.THERMAL_CONSERVATOR,
            2: DriverBehaviorPattern.WEAR_OPTIMIZER,
            3: DriverBehaviorPattern.WEAR_AGGRESSOR,
            4: DriverBehaviorPattern.ADAPTIVE_DRIVER,
            5: DriverBehaviorPattern.CONSISTENT_DRIVER,
            6: DriverBehaviorPattern.VARIABLE_DRIVER
        }
        
        return pattern_mapping.get(cluster_id, DriverBehaviorPattern.CONSISTENT_DRIVER)
    
    def _adapt_patterns(self):
        """Adapt patterns based on new data."""
        if len(self.feature_history) < self.p.min_samples:
            return
        
        # Extract all features
        all_features = np.array(list(self.feature_history))
        
        # Update scaler
        self.feature_scaler.fit(all_features)
        
        # Update clustering model
        features_scaled = self.feature_scaler.transform(all_features)
        
        if len(features_scaled) >= self.p.n_clusters:
            self.kmeans_model.fit(features_scaled)
            
            # Update model confidence
            self.model_confidence = self._calculate_model_confidence()
            
            # Reclassify all drivers
            for driver_id in self.driver_profiles:
                self._classify_driver_behavior(driver_id)
            
            self.adaptation_count += 1
    
    def _calculate_model_confidence(self) -> float:
        """Calculate model confidence based on clustering quality."""
        if len(self.feature_history) < self.p.min_samples:
            return 0.0
        
        # Calculate silhouette score (simplified)
        features_scaled = self.feature_scaler.transform(np.array(list(self.feature_history)))
        
        if len(features_scaled) >= self.p.n_clusters:
            try:
                clusters = self.kmeans_model.predict(features_scaled)
                
                # Calculate intra-cluster distance
                intra_cluster_distances = []
                for i in range(self.p.n_clusters):
                    cluster_points = features_scaled[clusters == i]
                    if len(cluster_points) > 1:
                        centroid = np.mean(cluster_points, axis=0)
                        distances = np.linalg.norm(cluster_points - centroid, axis=1)
                        intra_cluster_distances.extend(distances)
                
                # Calculate confidence based on cluster compactness
                if intra_cluster_distances:
                    avg_intra_distance = np.mean(intra_cluster_distances)
                    confidence = 1.0 / (1.0 + avg_intra_distance)
                    return min(1.0, confidence)
            
            except Exception:
                pass
        
        return 0.5  # Default confidence
    
    def get_driver_profile(self, driver_id: str) -> Dict[str, Any]:
        """Get comprehensive driver profile."""
        if driver_id not in self.driver_profiles:
            return {
                'driver_id': driver_id,
                'behavior_pattern': None,
                'confidence': 0.0,
                'learning_phase': self.learning_phase.value,
                'data_points': 0,
                'recommendations': []
            }
        
        profile = self.driver_profiles[driver_id]
        
        return {
            'driver_id': driver_id,
            'behavior_pattern': profile['behavior_pattern'].value if profile['behavior_pattern'] else None,
            'confidence': profile['confidence'],
            'learning_phase': self.learning_phase.value,
            'data_points': len(profile['data_points']),
            'adaptation_count': profile['adaptation_count'],
            'recommendations': self._get_driver_recommendations(driver_id)
        }
    
    def _get_driver_recommendations(self, driver_id: str) -> List[Tuple[str, str]]:
        """Get recommendations based on driver behavior pattern."""
        recommendations = []
        
        if driver_id not in self.driver_profiles:
            return recommendations
        
        profile = self.driver_profiles[driver_id]
        behavior_pattern = profile['behavior_pattern']
        confidence = profile['confidence']
        
        if behavior_pattern is None or confidence < self.p.confidence_threshold:
            recommendations.append(("ML_PROFILING", "Insufficient data for behavior analysis - continue data collection"))
            return recommendations
        
        # Pattern-specific recommendations
        if behavior_pattern == DriverBehaviorPattern.THERMAL_AGGRESSOR:
            recommendations.append(("ML_PROFILING", f"Thermal aggressor detected (confidence: {confidence:.1%}) - focus on temperature management"))
        elif behavior_pattern == DriverBehaviorPattern.THERMAL_CONSERVATOR:
            recommendations.append(("ML_PROFILING", f"Thermal conservator detected (confidence: {confidence:.1%}) - can push harder for performance"))
        elif behavior_pattern == DriverBehaviorPattern.WEAR_OPTIMIZER:
            recommendations.append(("ML_PROFILING", f"Wear optimizer detected (confidence: {confidence:.1%}) - excellent tire management"))
        elif behavior_pattern == DriverBehaviorPattern.WEAR_AGGRESSOR:
            recommendations.append(("ML_PROFILING", f"Wear aggressor detected (confidence: {confidence:.1%}) - focus on tire conservation"))
        elif behavior_pattern == DriverBehaviorPattern.ADAPTIVE_DRIVER:
            recommendations.append(("ML_PROFILING", f"Adaptive driver detected (confidence: {confidence:.1%}) - versatile driving style"))
        elif behavior_pattern == DriverBehaviorPattern.CONSISTENT_DRIVER:
            recommendations.append(("ML_PROFILING", f"Consistent driver detected (confidence: {confidence:.1%}) - reliable performance"))
        elif behavior_pattern == DriverBehaviorPattern.VARIABLE_DRIVER:
            recommendations.append(("ML_PROFILING", f"Variable driver detected (confidence: {confidence:.1%}) - work on consistency"))
        
        # Learning phase recommendations
        if self.learning_phase == LearningPhase.INITIAL:
            recommendations.append(("ML_PROFILING", "Initial learning phase - collecting baseline data"))
        elif self.learning_phase == LearningPhase.LEARNING:
            recommendations.append(("ML_PROFILING", "Active learning phase - patterns emerging"))
        elif self.learning_phase == LearningPhase.MATURE:
            recommendations.append(("ML_PROFILING", "Mature model - high confidence predictions"))
        elif self.learning_phase == LearningPhase.ADAPTING:
            recommendations.append(("ML_PROFILING", "Adapting to new patterns - model updating"))
        
        return recommendations
    
    def predict_driver_performance(self, driver_id: str, conditions: Dict) -> Dict[str, float]:
        """Predict driver performance based on behavior pattern and conditions."""
        if driver_id not in self.driver_profiles:
            return {'predicted_performance': 0.5, 'confidence': 0.0}
        
        profile = self.driver_profiles[driver_id]
        behavior_pattern = profile['behavior_pattern']
        
        if behavior_pattern is None:
            return {'predicted_performance': 0.5, 'confidence': 0.0}
        
        # Pattern-based performance prediction
        base_performance = {
            DriverBehaviorPattern.THERMAL_AGGRESSOR: 0.7,
            DriverBehaviorPattern.THERMAL_CONSERVATOR: 0.6,
            DriverBehaviorPattern.WEAR_OPTIMIZER: 0.8,
            DriverBehaviorPattern.WEAR_AGGRESSOR: 0.5,
            DriverBehaviorPattern.ADAPTIVE_DRIVER: 0.75,
            DriverBehaviorPattern.CONSISTENT_DRIVER: 0.7,
            DriverBehaviorPattern.VARIABLE_DRIVER: 0.6
        }
        
        predicted_performance = base_performance.get(behavior_pattern, 0.5)
        
        # Adjust based on conditions
        if conditions.get('high_temperature', False):
            if behavior_pattern == DriverBehaviorPattern.THERMAL_AGGRESSOR:
                predicted_performance *= 0.8  # Aggressor struggles in heat
            elif behavior_pattern == DriverBehaviorPattern.THERMAL_CONSERVATOR:
                predicted_performance *= 1.1  # Conservator excels in heat
        
        if conditions.get('wet_conditions', False):
            if behavior_pattern == DriverBehaviorPattern.ADAPTIVE_DRIVER:
                predicted_performance *= 1.2  # Adaptive driver excels in changing conditions
        
        return {
            'predicted_performance': min(1.0, predicted_performance),
            'confidence': profile['confidence']
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'learning_phase': self.learning_phase.value,
            'model_confidence': self.model_confidence,
            'total_drivers': len(self.driver_profiles),
            'total_data_points': len(self.telemetry_data),
            'adaptation_count': self.adaptation_count,
            'pattern_distribution': self._get_pattern_distribution(),
            'average_confidence': np.mean([p['confidence'] for p in self.driver_profiles.values()]) if self.driver_profiles else 0.0
        }
    
    def _get_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of behavior patterns."""
        distribution = {}
        for profile in self.driver_profiles.values():
            pattern = profile['behavior_pattern']
            if pattern:
                pattern_name = pattern.value
                distribution[pattern_name] = distribution.get(pattern_name, 0) + 1
        return distribution
    
    def save_model(self, filepath: str):
        """Save the ML model to file."""
        model_data = {
            'kmeans_model': self.kmeans_model,
            'feature_scaler': self.feature_scaler,
            'driver_profiles': self.driver_profiles,
            'learning_phase': self.learning_phase,
            'model_confidence': self.model_confidence,
            'adaptation_count': self.adaptation_count,
            'telemetry_data': list(self.telemetry_data),
            'feature_history': list(self.feature_history),
            'pattern_history': list(self.pattern_history),
            'params': self.p
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the ML model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans_model = model_data['kmeans_model']
        self.feature_scaler = model_data['feature_scaler']
        self.driver_profiles = model_data['driver_profiles']
        self.learning_phase = model_data['learning_phase']
        self.model_confidence = model_data['model_confidence']
        self.adaptation_count = model_data['adaptation_count']
        self.telemetry_data = deque(model_data['telemetry_data'], maxlen=10000)
        self.feature_history = deque(model_data['feature_history'], maxlen=1000)
        self.pattern_history = deque(model_data['pattern_history'], maxlen=500)
        self.p = model_data['params']
    
    def reset_model(self):
        """Reset the ML model for new training."""
        self.kmeans_model = KMeans(n_clusters=self.p.n_clusters, random_state=42)
        self.feature_scaler = StandardScaler()
        self.telemetry_data.clear()
        self.feature_history.clear()
        self.pattern_history.clear()
        self.driver_profiles = {}
        self.learning_phase = LearningPhase.INITIAL
        self.model_confidence = 0.0
        self.adaptation_count = 0
        self.accuracy_history = []
        self.pattern_confidence_history = []
