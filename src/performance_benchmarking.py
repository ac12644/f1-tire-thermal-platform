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

class BenchmarkType(Enum):
    """Types of performance benchmarks."""
    DRIVER_PERFORMANCE = "driver_performance"
    TIRE_MANAGEMENT = "tire_management"
    THERMAL_EFFICIENCY = "thermal_efficiency"
    WEAR_OPTIMIZATION = "wear_optimization"
    STRATEGY_EFFECTIVENESS = "strategy_effectiveness"
    WEATHER_ADAPTATION = "weather_adaptation"
    CONSISTENCY = "consistency"
    ADAPTATION_SPEED = "adaptation_speed"

class ConditionType(Enum):
    """Types of racing conditions."""
    DRY_CONDITIONS = "dry_conditions"
    WET_CONDITIONS = "wet_conditions"
    MIXED_CONDITIONS = "mixed_conditions"
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_HUMIDITY = "high_humidity"
    WINDY_CONDITIONS = "windy_conditions"

class MetricType(Enum):
    """Types of performance metrics."""
    LAP_TIME = "lap_time"
    TIRE_LIFE = "tire_life"
    THERMAL_STABILITY = "thermal_stability"
    WEAR_RATE = "wear_rate"
    CONSISTENCY_SCORE = "consistency_score"
    ADAPTATION_TIME = "adaptation_time"
    STRATEGY_SUCCESS = "strategy_success"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class PerformanceBenchmarkParams:
    """Parameters for performance benchmarking."""
    # Benchmarking parameters
    benchmark_window_days: int = 30
    min_samples_per_benchmark: int = 10
    confidence_level: float = 0.95
    
    # Performance thresholds
    excellent_threshold: float = 0.9
    good_threshold: float = 0.7
    average_threshold: float = 0.5
    
    # Comparison parameters
    comparison_drivers: int = 5  # Number of drivers to compare against
    percentile_ranks: List[int] = None
    
    # Analysis parameters
    trend_analysis_window: int = 7  # days
    seasonality_detection: bool = True
    outlier_detection: bool = True
    
    def __post_init__(self):
        if self.percentile_ranks is None:
            self.percentile_ranks = [25, 50, 75, 90, 95]

class PerformanceBenchmarking:
    """
    Performance benchmarking system for F1 tire temperature management.
    
    Features:
    - Multi-dimensional performance benchmarking
    - Driver comparison across different conditions
    - Tire management effectiveness analysis
    - Thermal efficiency benchmarking
    - Strategy effectiveness evaluation
    - Weather adaptation assessment
    - Consistency and reliability metrics
    - Performance trend analysis
    """
    
    def __init__(self, params: PerformanceBenchmarkParams = None):
        self.p = params or PerformanceBenchmarkParams()
        
        # Benchmark data storage
        self.benchmark_data = {}
        self.driver_benchmarks = {}
        self.condition_benchmarks = {}
        self.metric_benchmarks = {}
        
        # Performance rankings
        self.driver_rankings = {}
        self.condition_rankings = {}
        self.metric_rankings = {}
        
        # Historical performance
        self.performance_history = {}
        self.trend_analysis = {}
        
        # Benchmark metadata
        self.benchmark_metadata = {
            'last_updated': None,
            'total_benchmarks': 0,
            'benchmark_coverage': {},
            'data_quality_scores': {}
        }
    
    def add_performance_data(self, driver_id: str, performance_data: Dict[str, Any]):
        """
        Add performance data for benchmarking.
        
        Args:
            driver_id: Driver identifier
            performance_data: Performance metrics and conditions
        """
        timestamp = performance_data.get('timestamp', datetime.now())
        
        # Store driver performance data
        if driver_id not in self.benchmark_data:
            self.benchmark_data[driver_id] = []
        
        self.benchmark_data[driver_id].append({
            'timestamp': timestamp,
            'data': performance_data
        })
        
        # Update metadata
        self.benchmark_metadata['last_updated'] = timestamp
        self.benchmark_metadata['total_benchmarks'] += 1
    
    def calculate_driver_benchmark(self, driver_id: str, start_date: datetime = None, 
                                 end_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate comprehensive driver benchmark.
        
        Args:
            driver_id: Driver identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with driver benchmark results
        """
        if driver_id not in self.benchmark_data:
            return {'error': 'No data available for driver'}
        
        # Filter data by date range
        driver_data = self._filter_data_by_date(
            self.benchmark_data[driver_id], start_date, end_date
        )
        
        if len(driver_data) < self.p.min_samples_per_benchmark:
            return {'error': 'Insufficient data for benchmark'}
        
        # Calculate benchmark metrics
        benchmark_metrics = self._calculate_benchmark_metrics(driver_data)
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(benchmark_metrics)
        
        # Calculate rankings
        rankings = self._calculate_driver_rankings(driver_id, benchmark_metrics)
        
        # Generate insights
        insights = self._generate_driver_insights(driver_id, benchmark_metrics, performance_scores)
        
        driver_benchmark = {
            'driver_id': driver_id,
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(driver_data)
            },
            'benchmark_metrics': benchmark_metrics,
            'performance_scores': performance_scores,
            'rankings': rankings,
            'insights': insights,
            'benchmark_timestamp': datetime.now()
        }
        
        # Store benchmark
        self.driver_benchmarks[driver_id] = driver_benchmark
        
        return driver_benchmark
    
    def _filter_data_by_date(self, data: List[Dict], start_date: datetime = None, 
                           end_date: datetime = None) -> List[Dict]:
        """Filter data by date range."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.p.benchmark_window_days)
        if end_date is None:
            end_date = datetime.now()
        
        filtered_data = []
        for entry in data:
            timestamp = entry['timestamp']
            if start_date <= timestamp <= end_date:
                filtered_data.append(entry)
        
        return filtered_data
    
    def _calculate_benchmark_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate benchmark metrics from performance data."""
        metrics = {}
        
        # Extract metrics from data
        lap_times = []
        tire_lives = []
        thermal_stabilities = []
        wear_rates = []
        consistency_scores = []
        adaptation_times = []
        strategy_successes = []
        risk_scores = []
        
        for entry in data:
            perf_data = entry['data']
            
            lap_times.append(perf_data.get('lap_time', 0))
            tire_lives.append(perf_data.get('tire_life', 0))
            thermal_stabilities.append(perf_data.get('thermal_stability', 0))
            wear_rates.append(perf_data.get('wear_rate', 0))
            consistency_scores.append(perf_data.get('consistency_score', 0))
            adaptation_times.append(perf_data.get('adaptation_time', 0))
            strategy_successes.append(perf_data.get('strategy_success', 0))
            risk_scores.append(perf_data.get('risk_score', 0))
        
        # Calculate statistical metrics
        metrics['lap_time'] = {
            'mean': statistics.mean(lap_times),
            'median': statistics.median(lap_times),
            'std': statistics.stdev(lap_times) if len(lap_times) > 1 else 0,
            'min': min(lap_times),
            'max': max(lap_times),
            'percentiles': self._calculate_percentiles(lap_times)
        }
        
        metrics['tire_life'] = {
            'mean': statistics.mean(tire_lives),
            'median': statistics.median(tire_lives),
            'std': statistics.stdev(tire_lives) if len(tire_lives) > 1 else 0,
            'min': min(tire_lives),
            'max': max(tire_lives),
            'percentiles': self._calculate_percentiles(tire_lives)
        }
        
        metrics['thermal_stability'] = {
            'mean': statistics.mean(thermal_stabilities),
            'median': statistics.median(thermal_stabilities),
            'std': statistics.stdev(thermal_stabilities) if len(thermal_stabilities) > 1 else 0,
            'min': min(thermal_stabilities),
            'max': max(thermal_stabilities),
            'percentiles': self._calculate_percentiles(thermal_stabilities)
        }
        
        metrics['wear_rate'] = {
            'mean': statistics.mean(wear_rates),
            'median': statistics.median(wear_rates),
            'std': statistics.stdev(wear_rates) if len(wear_rates) > 1 else 0,
            'min': min(wear_rates),
            'max': max(wear_rates),
            'percentiles': self._calculate_percentiles(wear_rates)
        }
        
        metrics['consistency_score'] = {
            'mean': statistics.mean(consistency_scores),
            'median': statistics.median(consistency_scores),
            'std': statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0,
            'min': min(consistency_scores),
            'max': max(consistency_scores),
            'percentiles': self._calculate_percentiles(consistency_scores)
        }
        
        metrics['adaptation_time'] = {
            'mean': statistics.mean(adaptation_times),
            'median': statistics.median(adaptation_times),
            'std': statistics.stdev(adaptation_times) if len(adaptation_times) > 1 else 0,
            'min': min(adaptation_times),
            'max': max(adaptation_times),
            'percentiles': self._calculate_percentiles(adaptation_times)
        }
        
        metrics['strategy_success'] = {
            'mean': statistics.mean(strategy_successes),
            'median': statistics.median(strategy_successes),
            'std': statistics.stdev(strategy_successes) if len(strategy_successes) > 1 else 0,
            'min': min(strategy_successes),
            'max': max(strategy_successes),
            'percentiles': self._calculate_percentiles(strategy_successes)
        }
        
        metrics['risk_score'] = {
            'mean': statistics.mean(risk_scores),
            'median': statistics.median(risk_scores),
            'std': statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0,
            'min': min(risk_scores),
            'max': max(risk_scores),
            'percentiles': self._calculate_percentiles(risk_scores)
        }
        
        return metrics
    
    def _calculate_percentiles(self, data: List[float]) -> Dict[int, float]:
        """Calculate percentiles for data."""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        percentiles = {}
        
        for percentile in self.p.percentile_ranks:
            index = int((percentile / 100) * (len(sorted_data) - 1))
            percentiles[percentile] = sorted_data[index]
        
        return percentiles
    
    def _calculate_performance_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance scores from metrics."""
        scores = {}
        
        # Lap time score (lower is better)
        lap_time_mean = metrics['lap_time']['mean']
        lap_time_std = metrics['lap_time']['std']
        scores['lap_time'] = max(0, 1 - (lap_time_std / lap_time_mean)) if lap_time_mean > 0 else 0
        
        # Tire life score (higher is better)
        tire_life_mean = metrics['tire_life']['mean']
        scores['tire_life'] = min(1, tire_life_mean / 100)  # Assuming max tire life of 100
        
        # Thermal stability score (higher is better)
        thermal_stability_mean = metrics['thermal_stability']['mean']
        scores['thermal_stability'] = min(1, thermal_stability_mean / 100)
        
        # Wear rate score (lower is better)
        wear_rate_mean = metrics['wear_rate']['mean']
        scores['wear_rate'] = max(0, 1 - wear_rate_mean)
        
        # Consistency score (higher is better)
        consistency_mean = metrics['consistency_score']['mean']
        scores['consistency'] = min(1, consistency_mean / 100)
        
        # Adaptation time score (lower is better)
        adaptation_mean = metrics['adaptation_time']['mean']
        scores['adaptation'] = max(0, 1 - (adaptation_mean / 100))
        
        # Strategy success score (higher is better)
        strategy_mean = metrics['strategy_success']['mean']
        scores['strategy_success'] = min(1, strategy_mean / 100)
        
        # Risk score (lower is better)
        risk_mean = metrics['risk_score']['mean']
        scores['risk_management'] = max(0, 1 - risk_mean)
        
        # Overall performance score
        scores['overall'] = statistics.mean(list(scores.values()))
        
        return scores
    
    def _calculate_driver_rankings(self, driver_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate driver rankings compared to other drivers."""
        rankings = {}
        
        # Get all driver metrics for comparison
        all_driver_metrics = {}
        for other_driver_id, other_data in self.benchmark_data.items():
            if other_driver_id != driver_id and len(other_data) >= self.p.min_samples_per_benchmark:
                other_metrics = self._calculate_benchmark_metrics(other_data)
                all_driver_metrics[other_driver_id] = other_metrics
        
        if not all_driver_metrics:
            return {'error': 'Insufficient data for ranking comparison'}
        
        # Calculate rankings for each metric
        for metric_name, metric_data in metrics.items():
            if metric_name in ['lap_time', 'wear_rate', 'adaptation_time', 'risk_score']:
                # Lower is better
                ranking = self._calculate_percentile_ranking(
                    metric_data['mean'], 
                    [other_metrics[metric_name]['mean'] for other_metrics in all_driver_metrics.values()],
                    reverse=True
                )
            else:
                # Higher is better
                ranking = self._calculate_percentile_ranking(
                    metric_data['mean'],
                    [other_metrics[metric_name]['mean'] for other_metrics in all_driver_metrics.values()],
                    reverse=False
                )
            
            rankings[metric_name] = {
                'percentile_rank': ranking,
                'rank_description': self._get_rank_description(ranking)
            }
        
        return rankings
    
    def _calculate_percentile_ranking(self, value: float, comparison_values: List[float], 
                                    reverse: bool = False) -> float:
        """Calculate percentile ranking of a value compared to other values."""
        if not comparison_values:
            return 50.0
        
        sorted_values = sorted(comparison_values, reverse=reverse)
        
        # Count values better than the given value
        if not reverse:
            better_count = sum(1 for v in sorted_values if v > value)
        else:
            better_count = sum(1 for v in sorted_values if v < value)
        
        # Calculate percentile rank
        percentile_rank = (better_count / len(sorted_values)) * 100
        
        return percentile_rank
    
    def _get_rank_description(self, percentile_rank: float) -> str:
        """Get description for percentile rank."""
        if percentile_rank >= 90:
            return 'excellent'
        elif percentile_rank >= 75:
            return 'very_good'
        elif percentile_rank >= 50:
            return 'good'
        elif percentile_rank >= 25:
            return 'average'
        else:
            return 'needs_improvement'
    
    def _generate_driver_insights(self, driver_id: str, metrics: Dict[str, Any], 
                                scores: Dict[str, float]) -> List[str]:
        """Generate insights for driver performance."""
        insights = []
        
        # Overall performance insight
        overall_score = scores['overall']
        if overall_score >= self.p.excellent_threshold:
            insights.append(f"Driver {driver_id} shows excellent overall performance ({overall_score:.2f})")
        elif overall_score >= self.p.good_threshold:
            insights.append(f"Driver {driver_id} shows good overall performance ({overall_score:.2f})")
        else:
            insights.append(f"Driver {driver_id} shows average performance ({overall_score:.2f})")
        
        # Specific metric insights
        if scores['lap_time'] >= 0.8:
            insights.append("Excellent lap time consistency")
        elif scores['lap_time'] < 0.5:
            insights.append("Lap time consistency needs improvement")
        
        if scores['tire_life'] >= 0.8:
            insights.append("Excellent tire life management")
        elif scores['tire_life'] < 0.5:
            insights.append("Tire life management needs improvement")
        
        if scores['thermal_stability'] >= 0.8:
            insights.append("Excellent thermal stability")
        elif scores['thermal_stability'] < 0.5:
            insights.append("Thermal stability needs improvement")
        
        if scores['consistency'] >= 0.8:
            insights.append("Excellent performance consistency")
        elif scores['consistency'] < 0.5:
            insights.append("Performance consistency needs improvement")
        
        return insights
    
    def compare_drivers(self, driver_ids: List[str], start_date: datetime = None, 
                       end_date: datetime = None) -> Dict[str, Any]:
        """
        Compare multiple drivers across different metrics.
        
        Args:
            driver_ids: List of driver identifiers
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with driver comparison results
        """
        comparison_results = {}
        
        # Calculate benchmarks for each driver
        driver_benchmarks = {}
        for driver_id in driver_ids:
            benchmark = self.calculate_driver_benchmark(driver_id, start_date, end_date)
            if 'error' not in benchmark:
                driver_benchmarks[driver_id] = benchmark
        
        if len(driver_benchmarks) < 2:
            return {'error': 'Insufficient data for comparison'}
        
        # Compare drivers across metrics
        metric_comparisons = {}
        for metric_name in ['lap_time', 'tire_life', 'thermal_stability', 'wear_rate', 
                          'consistency_score', 'adaptation_time', 'strategy_success', 'risk_score']:
            
            metric_values = {}
            for driver_id, benchmark in driver_benchmarks.items():
                metric_values[driver_id] = benchmark['benchmark_metrics'][metric_name]['mean']
            
            # Sort drivers by metric performance
            sorted_drivers = sorted(metric_values.items(), key=lambda x: x[1])
            
            metric_comparisons[metric_name] = {
                'driver_rankings': sorted_drivers,
                'best_driver': sorted_drivers[0][0],
                'worst_driver': sorted_drivers[-1][0],
                'performance_spread': sorted_drivers[-1][1] - sorted_drivers[0][1]
            }
        
        # Overall comparison
        overall_scores = {}
        for driver_id, benchmark in driver_benchmarks.items():
            overall_scores[driver_id] = benchmark['performance_scores']['overall']
        
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison_results = {
            'comparison_period': {
                'start_date': start_date,
                'end_date': end_date,
                'drivers_compared': len(driver_benchmarks)
            },
            'overall_rankings': sorted_overall,
            'metric_comparisons': metric_comparisons,
            'driver_benchmarks': driver_benchmarks,
            'comparison_insights': self._generate_comparison_insights(metric_comparisons, sorted_overall)
        }
        
        return comparison_results
    
    def _generate_comparison_insights(self, metric_comparisons: Dict[str, Any], 
                                    overall_rankings: List[Tuple[str, float]]) -> List[str]:
        """Generate insights from driver comparison."""
        insights = []
        
        # Overall performance insight
        best_driver = overall_rankings[0][0]
        best_score = overall_rankings[0][1]
        insights.append(f"{best_driver} shows the best overall performance ({best_score:.2f})")
        
        # Metric-specific insights
        for metric_name, comparison in metric_comparisons.items():
            best_driver_metric = comparison['best_driver']
            worst_driver_metric = comparison['worst_driver']
            spread = comparison['performance_spread']
            
            insights.append(f"{best_driver_metric} leads in {metric_name} with {spread:.2f} advantage over {worst_driver_metric}")
        
        return insights
    
    def analyze_performance_trends(self, driver_id: str, start_date: datetime = None, 
                                 end_date: datetime = None) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            driver_id: Driver identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with trend analysis results
        """
        if driver_id not in self.benchmark_data:
            return {'error': 'No data available for driver'}
        
        # Filter data by date range
        driver_data = self._filter_data_by_date(
            self.benchmark_data[driver_id], start_date, end_date
        )
        
        if len(driver_data) < self.p.min_samples_per_benchmark:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Group data by time periods
        time_periods = self._group_data_by_time_periods(driver_data)
        
        # Calculate trends for each metric
        trend_analysis = {}
        for metric_name in ['lap_time', 'tire_life', 'thermal_stability', 'wear_rate', 
                          'consistency_score', 'adaptation_time', 'strategy_success', 'risk_score']:
            
            metric_trends = []
            for period_data in time_periods:
                period_metrics = self._calculate_benchmark_metrics(period_data)
                metric_trends.append(period_metrics[metric_name]['mean'])
            
            # Calculate trend statistics
            trend_stats = self._calculate_trend_statistics(metric_trends)
            trend_analysis[metric_name] = trend_stats
        
        # Generate trend insights
        trend_insights = self._generate_trend_insights(trend_analysis)
        
        trend_results = {
            'driver_id': driver_id,
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'time_periods': len(time_periods)
            },
            'trend_analysis': trend_analysis,
            'trend_insights': trend_insights,
            'trend_timestamp': datetime.now()
        }
        
        return trend_results
    
    def _group_data_by_time_periods(self, data: List[Dict]) -> List[List[Dict]]:
        """Group data by time periods for trend analysis."""
        # Group by days
        time_periods = defaultdict(list)
        
        for entry in data:
            date = entry['timestamp'].date()
            time_periods[date].append(entry)
        
        # Convert to list of lists
        return list(time_periods.values())
    
    def _calculate_trend_statistics(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a series of values."""
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
        
        # Linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and R-squared
        slope = np.polyfit(x, y, 1)[0]
        y_pred = np.polyval([slope, np.mean(y) - slope * np.mean(x)], x)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared,
            'change_percentage': (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0,
            'values': values
        }
    
    def _generate_trend_insights(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        for metric_name, trend_stats in trend_analysis.items():
            trend = trend_stats['trend']
            change_pct = trend_stats['change_percentage']
            
            if trend == 'improving':
                insights.append(f"{metric_name} shows improving trend ({change_pct:.1f}% change)")
            elif trend == 'declining':
                insights.append(f"{metric_name} shows declining trend ({change_pct:.1f}% change)")
            else:
                insights.append(f"{metric_name} shows stable trend")
        
        return insights
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get comprehensive benchmark summary."""
        return {
            'benchmark_metadata': self.benchmark_metadata,
            'driver_benchmarks': len(self.driver_benchmarks),
            'benchmark_data_status': {
                driver_id: len(data) for driver_id, data in self.benchmark_data.items()
            },
            'performance_history': len(self.performance_history),
            'trend_analysis': len(self.trend_analysis)
        }
