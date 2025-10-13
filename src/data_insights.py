from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict
import statistics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

class InsightType(Enum):
    """Types of data-driven insights."""
    PERFORMANCE_TREND = "performance_trend"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_INSIGHT = "predictive_insight"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"
    RISK_ASSESSMENT = "risk_assessment"

class InsightPriority(Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class InsightCategory(Enum):
    """Categories of insights."""
    THERMAL_MANAGEMENT = "thermal_management"
    TIRE_WEAR = "tire_wear"
    DRIVER_PERFORMANCE = "driver_performance"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    WEATHER_ADAPTATION = "weather_adaptation"
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE_ANALYSIS = "performance_analysis"

@dataclass
class DataInsightsParams:
    """Parameters for data-driven insights."""
    # Insight generation parameters
    min_data_points: int = 50
    confidence_threshold: float = 0.7
    anomaly_threshold: float = 2.0
    
    # Trend analysis parameters
    trend_window_days: int = 7
    trend_significance_threshold: float = 0.05
    
    # Pattern recognition parameters
    pattern_min_support: float = 0.1
    pattern_min_confidence: float = 0.6
    
    # Clustering parameters
    max_clusters: int = 10
    min_cluster_size: int = 5
    
    # Performance parameters
    insight_cache_size: int = 1000
    insight_retention_days: int = 30
    
    # Alert parameters
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'temperature_deviation': 10.0,
                'wear_rate_increase': 0.2,
                'performance_degradation': 0.15,
                'consistency_drop': 0.1
            }

class DataDrivenInsights:
    """
    Data-driven insights system for continuous F1 tire temperature management improvement.
    
    Features:
    - Performance trend analysis and prediction
    - Optimization opportunity identification
    - Anomaly detection and alerting
    - Pattern recognition and correlation analysis
    - Predictive insights for strategy optimization
    - Risk assessment and mitigation recommendations
    - Continuous learning and adaptation
    - Insight prioritization and actionability scoring
    """
    
    def __init__(self, params: DataInsightsParams = None):
        self.p = params or DataInsightsParams()
        
        # Data storage
        self.insights_data = {}
        self.performance_data = {}
        self.anomaly_data = {}
        self.pattern_data = {}
        
        # Insight generation
        self.insights_history = []
        self.insight_cache = {}
        self.insight_priorities = {}
        
        # Analysis models
        self.clustering_models = {}
        self.trend_models = {}
        self.correlation_matrices = {}
        
        # Alert system
        self.alert_history = []
        self.alert_thresholds = self.p.alert_thresholds
        
        # Performance tracking
        self.insight_accuracy = {}
        self.insight_actionability = {}
        
        # Metadata
        self.insights_metadata = {
            'last_analysis': None,
            'total_insights': 0,
            'insight_categories': {},
            'data_quality_scores': {}
        }
    
    def add_performance_data(self, data: Dict[str, Any]):
        """
        Add performance data for insight generation.
        
        Args:
            data: Performance data dictionary
        """
        timestamp = data.get('timestamp', datetime.now())
        
        # Store performance data
        if 'performance_data' not in self.insights_data:
            self.insights_data['performance_data'] = []
        
        self.insights_data['performance_data'].append({
            'timestamp': timestamp,
            'data': data
        })
        
        # Update metadata
        self.insights_metadata['last_analysis'] = timestamp
        self.insights_metadata['total_insights'] += 1
    
    def generate_performance_trend_insights(self, start_date: datetime = None, 
                                         end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Generate performance trend insights.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of trend insights
        """
        if 'performance_data' not in self.insights_data:
            return []
        
        # Filter data by date range
        performance_data = self._filter_data_by_date(
            self.insights_data['performance_data'], start_date, end_date
        )
        
        if len(performance_data) < self.p.min_data_points:
            return []
        
        insights = []
        
        # Analyze temperature trends
        temp_insights = self._analyze_temperature_trends(performance_data)
        insights.extend(temp_insights)
        
        # Analyze wear trends
        wear_insights = self._analyze_wear_trends(performance_data)
        insights.extend(wear_insights)
        
        # Analyze performance trends
        perf_insights = self._analyze_performance_trends(performance_data)
        insights.extend(perf_insights)
        
        # Store insights
        self.insights_history.extend(insights)
        
        return insights
    
    def _filter_data_by_date(self, data: List[Dict], start_date: datetime = None, 
                           end_date: datetime = None) -> List[Dict]:
        """Filter data by date range."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.p.trend_window_days)
        if end_date is None:
            end_date = datetime.now()
        
        filtered_data = []
        for entry in data:
            timestamp = entry['timestamp']
            if start_date <= timestamp <= end_date:
                filtered_data.append(entry)
        
        return filtered_data
    
    def _analyze_temperature_trends(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze temperature trends."""
        insights = []
        
        # Extract temperature data
        timestamps = [entry['timestamp'] for entry in data]
        tread_temps = [entry.get('tread_temp', 0) for entry in data]
        carcass_temps = [entry.get('carcass_temp', 0) for entry in data]
        rim_temps = [entry.get('rim_temp', 0) for entry in data]
        
        # Calculate trends
        tread_trend = self._calculate_trend(tread_temps)
        carcass_trend = self._calculate_trend(carcass_temps)
        rim_trend = self._calculate_trend(rim_temps)
        
        # Generate insights
        if tread_trend['slope'] > 0.5:
            insights.append({
                'type': InsightType.PERFORMANCE_TREND.value,
                'category': InsightCategory.THERMAL_MANAGEMENT.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'Increasing Tread Temperature Trend',
                'description': f'Tread temperature is increasing at {tread_trend["slope"]:.2f}°C per data point',
                'confidence': min(1.0, abs(tread_trend['r_squared'])),
                'actionable': True,
                'recommendations': [
                    'Monitor tire pressure more closely',
                    'Consider earlier pit stops',
                    'Adjust driving style to reduce thermal load'
                ],
                'timestamp': datetime.now()
            })
        
        if carcass_trend['slope'] > 0.3:
            insights.append({
                'type': InsightType.PERFORMANCE_TREND.value,
                'category': InsightCategory.THERMAL_MANAGEMENT.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'Increasing Carcass Temperature Trend',
                'description': f'Carcass temperature is increasing at {carcass_trend["slope"]:.2f}°C per data point',
                'confidence': min(1.0, abs(carcass_trend['r_squared'])),
                'actionable': True,
                'recommendations': [
                    'Check tire construction integrity',
                    'Monitor for potential tire failure',
                    'Consider compound change'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_wear_trends(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze wear trends."""
        insights = []
        
        # Extract wear data
        wear_levels = [entry.get('wear_level', 0) for entry in data]
        
        # Calculate wear trend
        wear_trend = self._calculate_trend(wear_levels)
        
        # Generate insights
        if wear_trend['slope'] > 0.01:
            insights.append({
                'type': InsightType.PERFORMANCE_TREND.value,
                'category': InsightCategory.TIRE_WEAR.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'Accelerating Wear Rate',
                'description': f'Wear rate is increasing at {wear_trend["slope"]:.3f} per data point',
                'confidence': min(1.0, abs(wear_trend['r_squared'])),
                'actionable': True,
                'recommendations': [
                    'Plan pit stop strategy',
                    'Reduce aggressive driving',
                    'Monitor for tire failure risk'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_performance_trends(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze performance trends."""
        insights = []
        
        # Extract performance data
        lap_times = [entry.get('lap_time', 0) for entry in data]
        consistency_scores = [entry.get('consistency_score', 0) for entry in data]
        
        # Calculate performance trends
        lap_time_trend = self._calculate_trend(lap_times)
        consistency_trend = self._calculate_trend(consistency_scores)
        
        # Generate insights
        if lap_time_trend['slope'] > 0.1:
            insights.append({
                'type': InsightType.PERFORMANCE_TREND.value,
                'category': InsightCategory.PERFORMANCE_ANALYSIS.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'Lap Time Degradation',
                'description': f'Lap times are increasing at {lap_time_trend["slope"]:.2f} seconds per data point',
                'confidence': min(1.0, abs(lap_time_trend['r_squared'])),
                'actionable': True,
                'recommendations': [
                    'Check tire condition',
                    'Monitor fuel load',
                    'Review driving strategy'
                ],
                'timestamp': datetime.now()
            })
        
        if consistency_trend['slope'] < -0.05:
            insights.append({
                'type': InsightType.PERFORMANCE_TREND.value,
                'category': InsightCategory.DRIVER_PERFORMANCE.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'Consistency Decline',
                'description': f'Consistency score is decreasing at {consistency_trend["slope"]:.3f} per data point',
                'confidence': min(1.0, abs(consistency_trend['r_squared'])),
                'actionable': True,
                'recommendations': [
                    'Focus on driving consistency',
                    'Review tire management approach',
                    'Consider strategy adjustment'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for a series of values."""
        if len(values) < 2:
            return {'slope': 0.0, 'r_squared': 0.0, 'direction': 'stable'}
        
        # Linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and R-squared
        slope = np.polyfit(x, y, 1)[0]
        y_pred = np.polyval([slope, np.mean(y) - slope * np.mean(x)], x)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'direction': direction,
            'change_percentage': (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        }
    
    def generate_optimization_insights(self, performance_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate optimization opportunity insights.
        
        Args:
            performance_data: Performance data for analysis
            
        Returns:
            List of optimization insights
        """
        insights = []
        
        # Analyze thermal efficiency
        thermal_insights = self._analyze_thermal_efficiency(performance_data)
        insights.extend(thermal_insights)
        
        # Analyze wear optimization
        wear_insights = self._analyze_wear_optimization(performance_data)
        insights.extend(wear_insights)
        
        # Analyze strategy optimization
        strategy_insights = self._analyze_strategy_optimization(performance_data)
        insights.extend(strategy_insights)
        
        return insights
    
    def _analyze_thermal_efficiency(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze thermal efficiency optimization opportunities."""
        insights = []
        
        # Extract thermal data
        tread_temps = [entry.get('tread_temp', 0) for entry in data]
        carcass_temps = [entry.get('carcass_temp', 0) for entry in data]
        rim_temps = [entry.get('rim_temp', 0) for entry in data]
        
        # Calculate thermal efficiency metrics
        avg_tread_temp = statistics.mean(tread_temps)
        temp_variance = statistics.variance(tread_temps) if len(tread_temps) > 1 else 0
        
        # Generate insights
        if avg_tread_temp > 100:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.THERMAL_MANAGEMENT.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'High Average Tread Temperature',
                'description': f'Average tread temperature is {avg_tread_temp:.1f}°C, above optimal range',
                'confidence': 0.9,
                'actionable': True,
                'recommendations': [
                    'Reduce tire pressure',
                    'Adjust driving style',
                    'Consider earlier pit stops'
                ],
                'timestamp': datetime.now()
            })
        
        if temp_variance > 50:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.THERMAL_MANAGEMENT.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'High Temperature Variance',
                'description': f'Temperature variance is {temp_variance:.1f}°C², indicating inconsistent thermal management',
                'confidence': 0.8,
                'actionable': True,
                'recommendations': [
                    'Improve driving consistency',
                    'Monitor tire pressure more closely',
                    'Review thermal management strategy'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_wear_optimization(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze wear optimization opportunities."""
        insights = []
        
        # Extract wear data
        wear_levels = [entry.get('wear_level', 0) for entry in data]
        
        # Calculate wear metrics
        avg_wear = statistics.mean(wear_levels)
        max_wear = max(wear_levels)
        wear_rate = self._calculate_wear_rate(wear_levels)
        
        # Generate insights
        if avg_wear > 0.7:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.TIRE_WEAR.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'High Average Wear Level',
                'description': f'Average wear level is {avg_wear:.2f}, approaching critical levels',
                'confidence': 0.9,
                'actionable': True,
                'recommendations': [
                    'Plan immediate pit stop',
                    'Reduce aggressive driving',
                    'Monitor for tire failure'
                ],
                'timestamp': datetime.now()
            })
        
        if wear_rate > 0.05:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.TIRE_WEAR.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'High Wear Rate',
                'description': f'Wear rate is {wear_rate:.3f} per data point, indicating aggressive wear',
                'confidence': 0.8,
                'actionable': True,
                'recommendations': [
                    'Adjust driving style',
                    'Monitor tire pressure',
                    'Consider compound change'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_strategy_optimization(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze strategy optimization opportunities."""
        insights = []
        
        # Extract strategy data
        strategy_successes = [entry.get('strategy_success', 0) for entry in data]
        performance_gains = [entry.get('performance_gain', 0) for entry in data]
        
        # Calculate strategy metrics
        avg_success = statistics.mean(strategy_successes)
        avg_gain = statistics.mean(performance_gains)
        
        # Generate insights
        if avg_success < 0.6:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.STRATEGY_OPTIMIZATION.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'Low Strategy Success Rate',
                'description': f'Strategy success rate is {avg_success:.2f}, below optimal threshold',
                'confidence': 0.8,
                'actionable': True,
                'recommendations': [
                    'Review strategy selection criteria',
                    'Improve strategy execution',
                    'Consider alternative strategies'
                ],
                'timestamp': datetime.now()
            })
        
        if avg_gain < 0.1:
            insights.append({
                'type': InsightType.OPTIMIZATION_OPPORTUNITY.value,
                'category': InsightCategory.STRATEGY_OPTIMIZATION.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'Low Performance Gain',
                'description': f'Average performance gain is {avg_gain:.2f}, indicating suboptimal strategy',
                'confidence': 0.7,
                'actionable': True,
                'recommendations': [
                    'Optimize strategy parameters',
                    'Review performance metrics',
                    'Consider strategy alternatives'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _calculate_wear_rate(self, wear_levels: List[float]) -> float:
        """Calculate wear rate from wear levels."""
        if len(wear_levels) < 2:
            return 0.0
        
        # Calculate wear rate as slope of wear progression
        x = np.arange(len(wear_levels))
        y = np.array(wear_levels)
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def generate_anomaly_insights(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate anomaly detection insights.
        
        Args:
            data: Data for anomaly detection
            
        Returns:
            List of anomaly insights
        """
        insights = []
        
        # Detect temperature anomalies
        temp_anomalies = self._detect_temperature_anomalies(data)
        insights.extend(temp_anomalies)
        
        # Detect wear anomalies
        wear_anomalies = self._detect_wear_anomalies(data)
        insights.extend(wear_anomalies)
        
        # Detect performance anomalies
        perf_anomalies = self._detect_performance_anomalies(data)
        insights.extend(perf_anomalies)
        
        return insights
    
    def _detect_temperature_anomalies(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect temperature anomalies."""
        insights = []
        
        # Extract temperature data
        tread_temps = [entry.get('tread_temp', 0) for entry in data]
        carcass_temps = [entry.get('carcass_temp', 0) for entry in data]
        rim_temps = [entry.get('rim_temp', 0) for entry in data]
        
        # Detect anomalies using statistical methods
        for i, (tread, carcass, rim) in enumerate(zip(tread_temps, carcass_temps, rim_temps)):
            # Check for extreme temperatures
            if tread > 120 or tread < 60:
                insights.append({
                    'type': InsightType.ANOMALY_DETECTION.value,
                    'category': InsightCategory.THERMAL_MANAGEMENT.value,
                    'priority': InsightPriority.CRITICAL.value,
                    'title': 'Extreme Tread Temperature',
                    'description': f'Tread temperature {tread:.1f}°C is outside normal range',
                    'confidence': 0.95,
                    'actionable': True,
                    'recommendations': [
                        'Immediate temperature monitoring',
                        'Check for tire failure risk',
                        'Consider emergency pit stop'
                    ],
                    'timestamp': data[i]['timestamp']
                })
            
            if carcass > 110 or carcass < 50:
                insights.append({
                    'type': InsightType.ANOMALY_DETECTION.value,
                    'category': InsightCategory.THERMAL_MANAGEMENT.value,
                    'priority': InsightPriority.HIGH.value,
                    'title': 'Extreme Carcass Temperature',
                    'description': f'Carcass temperature {carcass:.1f}°C is outside normal range',
                    'confidence': 0.9,
                    'actionable': True,
                    'recommendations': [
                        'Monitor tire integrity',
                        'Check for structural damage',
                        'Consider tire replacement'
                    ],
                    'timestamp': data[i]['timestamp']
                })
        
        return insights
    
    def _detect_wear_anomalies(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect wear anomalies."""
        insights = []
        
        # Extract wear data
        wear_levels = [entry.get('wear_level', 0) for entry in data]
        
        # Detect anomalies
        for i, wear in enumerate(wear_levels):
            if wear > 0.9:
                insights.append({
                    'type': InsightType.ANOMALY_DETECTION.value,
                    'category': InsightCategory.TIRE_WEAR.value,
                    'priority': InsightPriority.CRITICAL.value,
                    'title': 'Critical Wear Level',
                    'description': f'Wear level {wear:.2f} is at critical threshold',
                    'confidence': 0.95,
                    'actionable': True,
                    'recommendations': [
                        'Immediate pit stop required',
                        'Monitor for tire failure',
                        'Reduce driving aggression'
                    ],
                    'timestamp': data[i]['timestamp']
                })
        
        return insights
    
    def _detect_performance_anomalies(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        insights = []
        
        # Extract performance data
        lap_times = [entry.get('lap_time', 0) for entry in data]
        
        # Calculate performance statistics
        avg_lap_time = statistics.mean(lap_times)
        std_lap_time = statistics.stdev(lap_times) if len(lap_times) > 1 else 0
        
        # Detect anomalies
        for i, lap_time in enumerate(lap_times):
            if abs(lap_time - avg_lap_time) > 2 * std_lap_time:
                insights.append({
                    'type': InsightType.ANOMALY_DETECTION.value,
                    'category': InsightCategory.PERFORMANCE_ANALYSIS.value,
                    'priority': InsightPriority.MEDIUM.value,
                    'title': 'Performance Anomaly',
                    'description': f'Lap time {lap_time:.2f}s is significantly different from average',
                    'confidence': 0.8,
                    'actionable': True,
                    'recommendations': [
                        'Investigate performance cause',
                        'Check for technical issues',
                        'Review driving strategy'
                    ],
                    'timestamp': data[i]['timestamp']
                })
        
        return insights
    
    def generate_pattern_insights(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate pattern recognition insights.
        
        Args:
            data: Data for pattern analysis
            
        Returns:
            List of pattern insights
        """
        insights = []
        
        # Analyze temperature patterns
        temp_patterns = self._analyze_temperature_patterns(data)
        insights.extend(temp_patterns)
        
        # Analyze wear patterns
        wear_patterns = self._analyze_wear_patterns(data)
        insights.extend(wear_patterns)
        
        # Analyze performance patterns
        perf_patterns = self._analyze_performance_patterns(data)
        insights.extend(perf_patterns)
        
        return insights
    
    def _analyze_temperature_patterns(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze temperature patterns."""
        insights = []
        
        # Extract temperature data
        tread_temps = [entry.get('tread_temp', 0) for entry in data]
        carcass_temps = [entry.get('carcass_temp', 0) for entry in data]
        rim_temps = [entry.get('rim_temp', 0) for entry in data]
        
        # Analyze temperature correlation
        tread_carcass_corr = np.corrcoef(tread_temps, carcass_temps)[0, 1]
        tread_rim_corr = np.corrcoef(tread_temps, rim_temps)[0, 1]
        
        # Generate insights
        if tread_carcass_corr > 0.8:
            insights.append({
                'type': InsightType.PATTERN_RECOGNITION.value,
                'category': InsightCategory.THERMAL_MANAGEMENT.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'Strong Tread-Carcass Temperature Correlation',
                'description': f'Tread and carcass temperatures are highly correlated ({tread_carcass_corr:.2f})',
                'confidence': 0.8,
                'actionable': True,
                'recommendations': [
                    'Monitor thermal coupling',
                    'Optimize thermal management',
                    'Consider thermal isolation'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_wear_patterns(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze wear patterns."""
        insights = []
        
        # Extract wear data
        wear_levels = [entry.get('wear_level', 0) for entry in data]
        
        # Analyze wear progression pattern
        wear_progression = self._analyze_wear_progression(wear_levels)
        
        # Generate insights
        if wear_progression['pattern'] == 'exponential':
            insights.append({
                'type': InsightType.PATTERN_RECOGNITION.value,
                'category': InsightCategory.TIRE_WEAR.value,
                'priority': InsightPriority.HIGH.value,
                'title': 'Exponential Wear Progression',
                'description': 'Wear progression follows exponential pattern, indicating accelerating degradation',
                'confidence': 0.8,
                'actionable': True,
                'recommendations': [
                    'Plan early pit stops',
                    'Monitor wear acceleration',
                    'Consider preventive measures'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_performance_patterns(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze performance patterns."""
        insights = []
        
        # Extract performance data
        lap_times = [entry.get('lap_time', 0) for entry in data]
        
        # Analyze performance consistency
        performance_consistency = self._analyze_performance_consistency(lap_times)
        
        # Generate insights
        if performance_consistency['consistency_score'] < 0.7:
            insights.append({
                'type': InsightType.PATTERN_RECOGNITION.value,
                'category': InsightCategory.PERFORMANCE_ANALYSIS.value,
                'priority': InsightPriority.MEDIUM.value,
                'title': 'Inconsistent Performance Pattern',
                'description': f'Performance consistency score is {performance_consistency["consistency_score"]:.2f}',
                'confidence': 0.7,
                'actionable': True,
                'recommendations': [
                    'Improve driving consistency',
                    'Review performance factors',
                    'Optimize strategy execution'
                ],
                'timestamp': datetime.now()
            })
        
        return insights
    
    def _analyze_wear_progression(self, wear_levels: List[float]) -> Dict[str, Any]:
        """Analyze wear progression pattern."""
        if len(wear_levels) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Fit different models
        x = np.arange(len(wear_levels))
        y = np.array(wear_levels)
        
        # Linear model
        linear_slope = np.polyfit(x, y, 1)[0]
        
        # Exponential model (log-linear)
        log_y = np.log(y + 1e-6)  # Add small value to avoid log(0)
        exp_slope = np.polyfit(x, log_y, 1)[0]
        
        # Determine pattern
        if abs(exp_slope) > abs(linear_slope) * 2:
            pattern = 'exponential'
        elif abs(linear_slope) > 0.01:
            pattern = 'linear'
        else:
            pattern = 'stable'
        
        return {
            'pattern': pattern,
            'linear_slope': linear_slope,
            'exponential_slope': exp_slope
        }
    
    def _analyze_performance_consistency(self, lap_times: List[float]) -> Dict[str, Any]:
        """Analyze performance consistency."""
        if len(lap_times) < 2:
            return {'consistency_score': 0.0}
        
        # Calculate consistency score
        mean_lap_time = statistics.mean(lap_times)
        std_lap_time = statistics.stdev(lap_times)
        
        # Consistency score (lower std = higher consistency)
        consistency_score = max(0, 1 - (std_lap_time / mean_lap_time))
        
        return {
            'consistency_score': consistency_score,
            'mean_lap_time': mean_lap_time,
            'std_lap_time': std_lap_time
        }
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get comprehensive insights summary."""
        return {
            'insights_metadata': self.insights_metadata,
            'total_insights': len(self.insights_history),
            'insights_by_type': self._count_insights_by_type(),
            'insights_by_category': self._count_insights_by_category(),
            'insights_by_priority': self._count_insights_by_priority(),
            'actionable_insights': len([i for i in self.insights_history if i.get('actionable', False)]),
            'high_priority_insights': len([i for i in self.insights_history if i.get('priority') == InsightPriority.HIGH.value]),
            'critical_insights': len([i for i in self.insights_history if i.get('priority') == InsightPriority.CRITICAL.value])
        }
    
    def _count_insights_by_type(self) -> Dict[str, int]:
        """Count insights by type."""
        counts = defaultdict(int)
        for insight in self.insights_history:
            counts[insight.get('type', 'unknown')] += 1
        return dict(counts)
    
    def _count_insights_by_category(self) -> Dict[str, int]:
        """Count insights by category."""
        counts = defaultdict(int)
        for insight in self.insights_history:
            counts[insight.get('category', 'unknown')] += 1
        return dict(counts)
    
    def _count_insights_by_priority(self) -> Dict[str, int]:
        """Count insights by priority."""
        counts = defaultdict(int)
        for insight in self.insights_history:
            counts[insight.get('priority', 'unknown')] += 1
        return dict(counts)
