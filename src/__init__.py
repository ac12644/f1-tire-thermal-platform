"""
F1 Tire Thermal Platform - Core Package

This package contains all the core components for the F1 Tire Thermal Management System.
"""

__version__ = "2.0.0"
__author__ = "Abhishek Chauhan"
__email__ = "your-email@example.com"

# Core thermal modeling
from .thermal import ThermalModel, ThermalParams
from .wear import TireWearModel, WearParams
from .weather import WeatherModel, WeatherParams, SessionType
from .decision import DecisionEngine

# Driver profiling
from .driver import DriverProfile, DriverParams, DrivingStyle, ThermalSignature
from .driver_profiles import DriverProfiles

# Advanced physics
from .structural import StructuralTireModel, StructuralParams, TireConstruction
from .aerodynamic import AerodynamicModel, AerodynamicParams
from .multiphysics import MultiPhysicsCoupling, MultiPhysicsParams
from .compound import AdvancedCompoundModel, CompoundParams, CompoundType

# Machine learning
from .ml_strategy import MLStrategyOptimizer, MLStrategyParams
from .ml_degradation import MLDegradationPredictor, MLDegradationParams
from .ml_driver_profiling import MLDriverProfiler, MLDriverProfilingParams
from .ml_recommendations import MLRecommendationEngine, MLRecommendationParams
from .ml_patterns import MLPatternRecognizer, MLPatternRecognitionParams

# Analytics and visualization
from .big_data import BigDataAnalytics, BigDataParams
from .predictive_analytics import PredictiveAnalytics, PredictiveAnalyticsParams
from .performance_benchmarking import PerformanceBenchmarking, PerformanceBenchmarkParams
from .advanced_visualization import AdvancedVisualization, AdvancedVisualizationParams
from .data_insights import DataDrivenInsights, DataInsightsParams

# Advanced features
from .simulation_engine import RaceSimulation, SimulationParams
from .strategy_optimization import StrategyOptimizer, StrategyOptimizationParams
from .real_time_collaboration import RealTimeCollaboration, CollaborationParams
from .advanced_reporting import ReportGenerator, ReportParams
from .integration_testing import IntegrationTester, IntegrationTestParams

# Configuration
from .config import system_config

__all__ = [
    # Core thermal modeling
    'ThermalModel', 'ThermalParams',
    'TireWearModel', 'WearParams',
    'WeatherModel', 'WeatherParams', 'SessionType',
    'DecisionEngine',
    
    # Driver profiling
    'DriverProfile', 'DriverParams', 'DrivingStyle', 'ThermalSignature',
    'DriverProfiles',
    
    # Advanced physics
    'StructuralTireModel', 'StructuralParams', 'TireConstruction',
    'AerodynamicModel', 'AerodynamicParams',
    'MultiPhysicsCoupling', 'MultiPhysicsParams',
    'AdvancedCompoundModel', 'CompoundParams', 'CompoundType',
    
    # Machine learning
    'MLStrategyOptimizer', 'MLStrategyParams',
    'MLDegradationPredictor', 'MLDegradationParams',
    'MLDriverProfiler', 'MLDriverProfilingParams',
    'MLRecommendationEngine', 'MLRecommendationParams',
    'MLPatternRecognizer', 'MLPatternRecognitionParams',
    
    # Analytics and visualization
    'BigDataAnalytics', 'BigDataParams',
    'PredictiveAnalytics', 'PredictiveAnalyticsParams',
    'PerformanceBenchmarking', 'PerformanceBenchmarkParams',
    'AdvancedVisualization', 'AdvancedVisualizationParams',
    'DataDrivenInsights', 'DataInsightsParams',
    
    # Advanced features
    'RaceSimulation', 'SimulationParams',
    'StrategyOptimizer', 'StrategyOptimizationParams',
    'RealTimeCollaboration', 'CollaborationParams',
    'ReportGenerator', 'ReportParams',
    'IntegrationTester', 'IntegrationTestParams',
    
    # Configuration
    'system_config',
]
