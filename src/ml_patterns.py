from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pickle
import json
from collections import deque
import random
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

class PatternType(Enum):
    """Types of patterns recognized."""
    THERMAL_PATTERN = "thermal_pattern"
    WEAR_PATTERN = "wear_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    STRATEGY_PATTERN = "strategy_pattern"
    ENVIRONMENTAL_PATTERN = "environmental_pattern"
    DRIVER_PATTERN = "driver_pattern"
    COMPOUND_PATTERN = "compound_pattern"
    STRUCTURAL_PATTERN = "structural_pattern"

class PatternComplexity(Enum):
    """Pattern complexity levels."""
    SIMPLE = "simple"        # Basic linear patterns
    MODERATE = "moderate"    # Non-linear patterns
    COMPLEX = "complex"      # Multi-dimensional patterns
    ADVANCED = "advanced"    # Deep learning patterns

@dataclass
class MLPatternRecognitionParams:
    """Parameters for ML pattern recognition."""
    # Pattern detection parameters
    min_pattern_length: int = 10
    max_pattern_length: int = 100
    pattern_similarity_threshold: float = 0.8
    pattern_confidence_threshold: float = 0.7
    
    # Clustering parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    kmeans_clusters: int = 8
    
    # Time series analysis
    window_size: int = 20
    overlap_ratio: float = 0.5
    trend_detection_threshold: float = 0.1
    
    # Feature extraction
    feature_dimensions: int = 15
    feature_scaling: bool = True
    
    # Pattern validation
    validation_window: int = 50
    min_pattern_frequency: int = 3
    pattern_decay_rate: float = 0.95

class MLPatternRecognizer:
    """
    Machine Learning-based pattern recognition for optimal tire management.
    
    Features:
    - Multi-dimensional pattern recognition
    - Time series analysis and trend detection
    - Clustering-based pattern discovery
    - Pattern similarity matching
    - Real-time pattern classification
    - Pattern-based optimization recommendations
    - Historical pattern analysis
    """
    
    def __init__(self, params: MLPatternRecognitionParams = None):
        self.p = params or MLPatternRecognitionParams()
        
        # Pattern storage
        self.recognized_patterns = {}
        self.pattern_database = {}
        self.pattern_history = deque(maxlen=1000)
        
        # Clustering models
        self.dbscan_model = DBSCAN(eps=self.p.dbscan_eps, min_samples=self.p.dbscan_min_samples)
        self.kmeans_model = KMeans(n_clusters=self.p.kmeans_clusters, random_state=42)
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.feature_extractor = None
        
        # Pattern analysis
        self.pattern_frequencies = {}
        self.pattern_success_rates = {}
        self.pattern_performance_metrics = {}
        
        # Time series analysis
        self.trend_analyzer = None
        self.seasonality_detector = None
        
        # Pattern validation
        self.pattern_validator = None
        self.validation_results = {}
        
    def extract_pattern_features(self, time_series_data: List[Dict]) -> np.ndarray:
        """
        Extract features from time series data for pattern recognition.
        
        Args:
            time_series_data: List of time series data points
            
        Returns:
            Feature vector for pattern analysis
        """
        if len(time_series_data) < self.p.min_pattern_length:
            return np.zeros(self.p.feature_dimensions)
        
        # Extract time series values
        thermal_values = [point.get('thermal_state', [0, 0, 0]) for point in time_series_data]
        wear_values = [point.get('wear_levels', [0, 0, 0, 0]) for point in time_series_data]
        
        # Convert to numpy arrays
        thermal_array = np.array(thermal_values)
        wear_array = np.array(wear_values)
        
        # Statistical features
        thermal_mean = np.mean(thermal_array, axis=0)
        thermal_std = np.std(thermal_array, axis=0)
        thermal_trend = self._calculate_trend(thermal_array)
        
        wear_mean = np.mean(wear_array, axis=0)
        wear_std = np.std(wear_array, axis=0)
        wear_trend = self._calculate_trend(wear_array)
        
        # Frequency domain features
        thermal_fft = np.fft.fft(thermal_array[:, 0])  # Use tread temperature
        thermal_freq_features = [
            np.mean(np.abs(thermal_fft)),
            np.std(np.abs(thermal_fft)),
            np.max(np.abs(thermal_fft))
        ]
        
        # Pattern complexity features
        complexity_features = [
            self._calculate_pattern_complexity(thermal_array),
            self._calculate_pattern_complexity(wear_array),
            self._calculate_pattern_volatility(thermal_array),
            self._calculate_pattern_volatility(wear_array)
        ]
        
        # Combine all features
        features = np.concatenate([
            thermal_mean, thermal_std, thermal_trend,
            wear_mean, wear_std, wear_trend,
            thermal_freq_features, complexity_features
        ])
        
        # Ensure correct dimensions
        if len(features) > self.p.feature_dimensions:
            features = features[:self.p.feature_dimensions]
        elif len(features) < self.p.feature_dimensions:
            features = np.pad(features, (0, self.p.feature_dimensions - len(features)))
        
        return features
    
    def _calculate_trend(self, data: np.ndarray) -> np.ndarray:
        """Calculate trend for each dimension of the data."""
        trends = []
        for i in range(data.shape[1]):
            if len(data) > 1:
                trend = np.polyfit(range(len(data)), data[:, i], 1)[0]
            else:
                trend = 0.0
            trends.append(trend)
        return np.array(trends)
    
    def _calculate_pattern_complexity(self, data: np.ndarray) -> float:
        """Calculate pattern complexity using entropy."""
        if len(data) == 0:
            return 0.0
        
        # Calculate entropy
        hist, _ = np.histogram(data.flatten(), bins=10)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def _calculate_pattern_volatility(self, data: np.ndarray) -> float:
        """Calculate pattern volatility."""
        if len(data) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if mean_val == 0:
            return 0.0
        
        return std_val / mean_val
    
    def discover_patterns(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Discover patterns in historical data using clustering.
        
        Args:
            historical_data: Historical telemetry data
            
        Returns:
            Dictionary of discovered patterns
        """
        if len(historical_data) < self.p.min_pattern_length:
            return {}
        
        # Extract features for all time windows
        pattern_features = []
        pattern_windows = []
        
        window_size = self.p.window_size
        step_size = int(window_size * (1 - self.p.overlap_ratio))
        
        for i in range(0, len(historical_data) - window_size, step_size):
            window_data = historical_data[i:i + window_size]
            features = self.extract_pattern_features(window_data)
            
            pattern_features.append(features)
            pattern_windows.append(window_data)
        
        if len(pattern_features) < self.p.dbscan_min_samples:
            return {}
        
        # Convert to numpy array
        pattern_features = np.array(pattern_features)
        
        # Scale features
        if self.p.feature_scaling:
            pattern_features_scaled = self.feature_scaler.fit_transform(pattern_features)
        else:
            pattern_features_scaled = pattern_features
        
        # Apply clustering
        dbscan_labels = self.dbscan_model.fit_predict(pattern_features_scaled)
        kmeans_labels = self.kmeans_model.fit_predict(pattern_features_scaled)
        
        # Analyze clusters
        discovered_patterns = {}
        
        # DBSCAN patterns (density-based)
        unique_dbscan_labels = np.unique(dbscan_labels)
        for label in unique_dbscan_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_indices = np.where(dbscan_labels == label)[0]
            if len(cluster_indices) >= self.p.min_pattern_frequency:
                pattern = self._analyze_cluster(pattern_windows, cluster_indices, label, 'dbscan')
                discovered_patterns[f'dbscan_pattern_{label}'] = pattern
        
        # K-means patterns (centroid-based)
        unique_kmeans_labels = np.unique(kmeans_labels)
        for label in unique_kmeans_labels:
            cluster_indices = np.where(kmeans_labels == label)[0]
            if len(cluster_indices) >= self.p.min_pattern_frequency:
                pattern = self._analyze_cluster(pattern_windows, cluster_indices, label, 'kmeans')
                discovered_patterns[f'kmeans_pattern_{label}'] = pattern
        
        # Store discovered patterns
        self.pattern_database.update(discovered_patterns)
        
        return discovered_patterns
    
    def _analyze_cluster(self, pattern_windows: List[List[Dict]], cluster_indices: np.ndarray, 
                        label: int, method: str) -> Dict[str, Any]:
        """Analyze a cluster to extract pattern characteristics."""
        cluster_windows = [pattern_windows[i] for i in cluster_indices]
        
        # Calculate pattern statistics
        pattern_stats = {
            'pattern_id': f'{method}_pattern_{label}',
            'method': method,
            'frequency': len(cluster_indices),
            'cluster_indices': cluster_indices.tolist(),
            'pattern_type': self._classify_pattern_type(cluster_windows),
            'complexity': self._calculate_pattern_complexity_score(cluster_windows),
            'success_rate': self._calculate_pattern_success_rate(cluster_windows),
            'performance_metrics': self._calculate_pattern_performance(cluster_windows)
        }
        
        # Calculate pattern characteristics
        pattern_stats['characteristics'] = self._extract_pattern_characteristics(cluster_windows)
        
        # Calculate pattern recommendations
        pattern_stats['recommendations'] = self._generate_pattern_recommendations(pattern_stats)
        
        return pattern_stats
    
    def _classify_pattern_type(self, cluster_windows: List[List[Dict]]) -> str:
        """Classify the type of pattern based on cluster characteristics."""
        # Analyze thermal patterns
        thermal_variations = []
        wear_variations = []
        
        for window in cluster_windows:
            thermal_states = [point.get('thermal_state', [0, 0, 0]) for point in window]
            wear_levels = [point.get('wear_levels', [0, 0, 0, 0]) for point in window]
            
            thermal_variations.append(np.std(thermal_states))
            wear_variations.append(np.std(wear_levels))
        
        avg_thermal_variation = np.mean(thermal_variations)
        avg_wear_variation = np.mean(wear_variations)
        
        # Classify based on variations
        if avg_thermal_variation > 10:
            return PatternType.THERMAL_PATTERN.value
        elif avg_wear_variation > 0.1:
            return PatternType.WEAR_PATTERN.value
        else:
            return PatternType.PERFORMANCE_PATTERN.value
    
    def _calculate_pattern_complexity_score(self, cluster_windows: List[List[Dict]]) -> float:
        """Calculate complexity score for the pattern."""
        complexity_scores = []
        
        for window in cluster_windows:
            thermal_states = [point.get('thermal_state', [0, 0, 0]) for point in window]
            thermal_array = np.array(thermal_states)
            
            complexity = self._calculate_pattern_complexity(thermal_array)
            complexity_scores.append(complexity)
        
        return np.mean(complexity_scores)
    
    def _calculate_pattern_success_rate(self, cluster_windows: List[List[Dict]]) -> float:
        """Calculate success rate for the pattern."""
        success_rates = []
        
        for window in cluster_windows:
            outcomes = [point.get('outcome', {}).get('success', False) for point in window]
            success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
            success_rates.append(success_rate)
        
        return np.mean(success_rates)
    
    def _calculate_pattern_performance(self, cluster_windows: List[List[Dict]]) -> Dict[str, float]:
        """Calculate performance metrics for the pattern."""
        performance_metrics = {
            'average_lap_time': 0.0,
            'average_tire_life': 0.0,
            'average_fuel_efficiency': 0.0,
            'consistency_score': 0.0
        }
        
        lap_times = []
        tire_lives = []
        fuel_efficiencies = []
        
        for window in cluster_windows:
            for point in window:
                metrics = point.get('performance_metrics', {})
                lap_times.append(metrics.get('lap_time', 0.0))
                tire_lives.append(metrics.get('tire_life', 0.0))
                fuel_efficiencies.append(metrics.get('fuel_efficiency', 0.0))
        
        if lap_times:
            performance_metrics['average_lap_time'] = np.mean(lap_times)
            performance_metrics['consistency_score'] = 1.0 / (1.0 + np.std(lap_times))
        
        if tire_lives:
            performance_metrics['average_tire_life'] = np.mean(tire_lives)
        
        if fuel_efficiencies:
            performance_metrics['average_fuel_efficiency'] = np.mean(fuel_efficiencies)
        
        return performance_metrics
    
    def _extract_pattern_characteristics(self, cluster_windows: List[List[Dict]]) -> Dict[str, Any]:
        """Extract detailed characteristics of the pattern."""
        characteristics = {
            'thermal_profile': {},
            'wear_profile': {},
            'environmental_profile': {},
            'driver_profile': {}
        }
        
        # Analyze thermal characteristics
        thermal_trends = []
        thermal_peaks = []
        
        for window in cluster_windows:
            thermal_states = [point.get('thermal_state', [0, 0, 0]) for point in window]
            thermal_array = np.array(thermal_states)
            
            # Calculate trend
            trend = self._calculate_trend(thermal_array)
            thermal_trends.append(trend)
            
            # Find peaks
            peaks, _ = signal.find_peaks(thermal_array[:, 0])
            thermal_peaks.extend(thermal_array[peaks, 0].tolist())
        
        characteristics['thermal_profile'] = {
            'average_trend': np.mean(thermal_trends, axis=0).tolist(),
            'trend_consistency': 1.0 / (1.0 + np.std(thermal_trends)),
            'peak_frequency': len(thermal_peaks) / len(cluster_windows),
            'average_peak_value': np.mean(thermal_peaks) if thermal_peaks else 0.0
        }
        
        # Analyze wear characteristics
        wear_rates = []
        for window in cluster_windows:
            wear_levels = [point.get('wear_levels', [0, 0, 0, 0]) for point in window]
            wear_array = np.array(wear_levels)
            
            if len(wear_array) > 1:
                wear_rate = np.mean(np.diff(wear_array, axis=0))
                wear_rates.append(wear_rate)
        
        characteristics['wear_profile'] = {
            'average_wear_rate': np.mean(wear_rates).tolist() if wear_rates else [0.0, 0.0, 0.0, 0.0],
            'wear_consistency': 1.0 / (1.0 + np.std(wear_rates)) if wear_rates else 0.0
        }
        
        return characteristics
    
    def _generate_pattern_recommendations(self, pattern_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        pattern_type = pattern_stats['pattern_type']
        success_rate = pattern_stats['success_rate']
        complexity = pattern_stats['complexity']
        
        # Pattern-specific recommendations
        if pattern_type == PatternType.THERMAL_PATTERN.value:
            if success_rate > 0.7:
                recommendations.append({
                    'type': 'optimization',
                    'message': f"High-performing thermal pattern detected (success rate: {success_rate:.1%})",
                    'action': 'maintain_current_thermal_management'
                })
            else:
                recommendations.append({
                    'type': 'improvement',
                    'message': f"Thermal pattern needs optimization (success rate: {success_rate:.1%})",
                    'action': 'improve_thermal_management'
                })
        
        elif pattern_type == PatternType.WEAR_PATTERN.value:
            if success_rate > 0.7:
                recommendations.append({
                    'type': 'optimization',
                    'message': f"Excellent wear management pattern (success rate: {success_rate:.1%})",
                    'action': 'continue_wear_optimization'
                })
            else:
                recommendations.append({
                    'type': 'improvement',
                    'message': f"Wear pattern needs attention (success rate: {success_rate:.1%})",
                    'action': 'improve_wear_management'
                })
        
        # Complexity-based recommendations
        if complexity > 0.8:
            recommendations.append({
                'type': 'complexity',
                'message': f"High complexity pattern detected (complexity: {complexity:.2f})",
                'action': 'simplify_approach'
            })
        
        return recommendations
    
    def classify_current_pattern(self, current_data: List[Dict]) -> Dict[str, Any]:
        """
        Classify current data against known patterns.
        
        Args:
            current_data: Current time series data
            
        Returns:
            Pattern classification results
        """
        if len(current_data) < self.p.min_pattern_length:
            return {'classification': 'insufficient_data', 'confidence': 0.0}
        
        # Extract features from current data
        current_features = self.extract_pattern_features(current_data)
        
        # Scale features
        if hasattr(self.feature_scaler, 'mean_'):
            current_features_scaled = self.feature_scaler.transform(current_features.reshape(1, -1))
        else:
            current_features_scaled = current_features.reshape(1, -1)
        
        # Find most similar pattern
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern_data in self.pattern_database.items():
            # Calculate similarity (simplified)
            similarity = self._calculate_pattern_similarity(current_features_scaled, pattern_data)
            
            if similarity > best_similarity and similarity > self.p.pattern_similarity_threshold:
                best_similarity = similarity
                best_match = pattern_data
        
        if best_match:
            return {
                'classification': best_match['pattern_id'],
                'confidence': best_similarity,
                'pattern_type': best_match['pattern_type'],
                'success_rate': best_match['success_rate'],
                'recommendations': best_match['recommendations']
            }
        else:
            return {
                'classification': 'new_pattern',
                'confidence': 0.0,
                'message': 'No matching pattern found - new pattern detected'
            }
    
    def _calculate_pattern_similarity(self, current_features: np.ndarray, pattern_data: Dict[str, Any]) -> float:
        """Calculate similarity between current features and pattern."""
        # Simplified similarity calculation
        # In practice, would use more sophisticated similarity metrics
        
        # Extract pattern characteristics
        characteristics = pattern_data.get('characteristics', {})
        
        # Calculate similarity based on thermal profile
        thermal_profile = characteristics.get('thermal_profile', {})
        if thermal_profile:
            # Simplified similarity calculation
            similarity = 0.8  # Placeholder
        else:
            similarity = 0.5
        
        return similarity
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get comprehensive pattern recognition summary."""
        return {
            'total_patterns': len(self.pattern_database),
            'pattern_types': self._get_pattern_type_distribution(),
            'pattern_frequencies': self.pattern_frequencies,
            'average_success_rate': np.mean([p['success_rate'] for p in self.pattern_database.values()]) if self.pattern_database else 0.0,
            'pattern_complexity_distribution': self._get_complexity_distribution(),
            'recent_classifications': len(self.pattern_history)
        }
    
    def _get_pattern_type_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern types."""
        distribution = {}
        for pattern_data in self.pattern_database.values():
            pattern_type = pattern_data['pattern_type']
            distribution[pattern_type] = distribution.get(pattern_type, 0) + 1
        return distribution
    
    def _get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern complexity."""
        distribution = {'simple': 0, 'moderate': 0, 'complex': 0, 'advanced': 0}
        
        for pattern_data in self.pattern_database.values():
            complexity = pattern_data['complexity']
            
            if complexity < 0.3:
                distribution['simple'] += 1
            elif complexity < 0.6:
                distribution['moderate'] += 1
            elif complexity < 0.8:
                distribution['complex'] += 1
            else:
                distribution['advanced'] += 1
        
        return distribution
    
    def save_model(self, filepath: str):
        """Save the pattern recognition model to file."""
        model_data = {
            'pattern_database': self.pattern_database,
            'feature_scaler': self.feature_scaler,
            'dbscan_model': self.dbscan_model,
            'kmeans_model': self.kmeans_model,
            'pattern_frequencies': self.pattern_frequencies,
            'pattern_success_rates': self.pattern_success_rates,
            'pattern_performance_metrics': self.pattern_performance_metrics,
            'pattern_history': list(self.pattern_history),
            'params': self.p
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the pattern recognition model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pattern_database = model_data['pattern_database']
        self.feature_scaler = model_data['feature_scaler']
        self.dbscan_model = model_data['dbscan_model']
        self.kmeans_model = model_data['kmeans_model']
        self.pattern_frequencies = model_data['pattern_frequencies']
        self.pattern_success_rates = model_data['pattern_success_rates']
        self.pattern_performance_metrics = model_data['pattern_performance_metrics']
        self.pattern_history = deque(model_data['pattern_history'], maxlen=1000)
        self.p = model_data['params']
    
    def reset_model(self):
        """Reset the pattern recognition model for new training."""
        self.pattern_database = {}
        self.pattern_history.clear()
        self.pattern_frequencies = {}
        self.pattern_success_rates = {}
        self.pattern_performance_metrics = {}
        self.feature_scaler = StandardScaler()
        self.dbscan_model = DBSCAN(eps=self.p.dbscan_eps, min_samples=self.p.dbscan_min_samples)
        self.kmeans_model = KMeans(n_clusters=self.p.kmeans_clusters, random_state=42)
