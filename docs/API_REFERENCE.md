# 🏎️ F1 Tire Thermal Platform

## API Reference Documentation

**Version**: 2.0  
**Last Updated**: September 2025  
**Target Audience**: Software Engineers, Data Scientists, F1 Engineers

---

## 📋 **Overview**

This document provides comprehensive API reference for the F1 Tire Temperature Management System. All classes, methods, and data structures are documented with examples and usage patterns.

---

## 🔧 **Core API Classes**

### **ThermalModel**

The core thermal modeling engine implementing 3-node thermal system with Extended Kalman Filtering.

```python
class ThermalModel:
    def __init__(self, params: ThermalParams, wear_model=None, weather_model=None):
        """
        Initialize thermal model with optional integrations.

        Args:
            params (ThermalParams): Thermal parameters configuration
            wear_model (TireWearModel, optional): Wear modeling integration
            weather_model (WeatherModel, optional): Weather integration

        Example:
            >>> params = ThermalParams()
            >>> thermal_model = ThermalModel(params)
            >>> # With integrations
            >>> wear_model = TireWearModel(WearParams())
            >>> weather_model = WeatherModel(WeatherParams())
            >>> thermal_model = ThermalModel(params, wear_model, weather_model)
        """

    def step(self, dt: float, loads: Dict, corner: str, compound: str) -> np.ndarray:
        """
        Advance thermal simulation by dt seconds.

        Args:
            dt (float): Time step in seconds
            loads (Dict): Dictionary containing load inputs
                - 'Fx': Longitudinal force (N)
                - 'Fy': Lateral force (N)
                - 'Fz': Vertical force (N)
                - 'Mz': Aligning moment (Nm)
            corner (str): Tire corner identifier ('FL', 'FR', 'RL', 'RR')
            compound (str): Tire compound ('soft', 'medium', 'hard')

        Returns:
            np.ndarray: Temperature array [Tt, Tc, Tr] in °C

        Example:
            >>> loads = {'Fx': 1000, 'Fy': 500, 'Fz': 4000, 'Mz': 50}
            >>> temps = thermal_model.step(0.1, loads, 'FL', 'medium')
            >>> print(f"Tread: {temps[0]:.1f}°C, Carcass: {temps[1]:.1f}°C, Rim: {temps[2]:.1f}°C")
        """

    def get_thermal_state(self, corner: str) -> np.ndarray:
        """
        Get current thermal state for specified corner.

        Args:
            corner (str): Tire corner identifier

        Returns:
            np.ndarray: Current temperature state [Tt, Tc, Tr]
        """

    def reset_thermal_state(self, corner: str, initial_temps: np.ndarray = None):
        """
        Reset thermal state for specified corner.

        Args:
            corner (str): Tire corner identifier
            initial_temps (np.ndarray, optional): Initial temperatures [Tt, Tc, Tr]
        """
```

### **TireWearModel**

Advanced tire wear modeling with thermal history tracking and compound-specific degradation.

```python
class TireWearModel:
    def __init__(self, params: WearParams):
        """
        Initialize wear model with thermal history tracking.

        Args:
            params (WearParams): Wear modeling parameters

        Example:
            >>> params = WearParams()
            >>> wear_model = TireWearModel(params)
        """

    def update_wear(self, thermal_state: np.ndarray, loads: Dict, dt: float):
        """
        Update wear levels based on thermal state and loads.

        Args:
            thermal_state (np.ndarray): Current temperature state [Tt, Tc, Tr]
            loads (Dict): Load inputs dictionary
            dt (float): Time step for integration

        Example:
            >>> thermal_state = np.array([95.0, 88.0, 45.0])
            >>> loads = {'Fx': 1000, 'Fy': 500, 'Fz': 4000}
            >>> wear_model.update_wear(thermal_state, loads, 0.1)
        """

    def get_wear_effects(self, corner: str) -> Dict:
        """
        Get wear effects on grip and stiffness.

        Args:
            corner (str): Tire corner identifier

        Returns:
            Dict: Dictionary containing:
                - 'grip_factor': Grip reduction factor (0.0-1.0)
                - 'stiffness_factor': Stiffness reduction factor (0.0-1.0)
                - 'wear_level': Current wear level (0.0-1.0)

        Example:
            >>> effects = wear_model.get_wear_effects('FL')
            >>> print(f"Grip factor: {effects['grip_factor']:.3f}")
            >>> print(f"Wear level: {effects['wear_level']:.3f}")
        """

    def reset_wear(self, corner: str = None):
        """
        Reset wear levels for specified corner or all corners.

        Args:
            corner (str, optional): Specific corner to reset, None for all corners
        """

    def get_wear_summary(self) -> Dict:
        """
        Get comprehensive wear summary for all corners.

        Returns:
            Dict: Wear summary with corner-specific data
        """
```

### **WeatherModel**

Environmental intelligence system for weather integration and track evolution modeling.

```python
class WeatherModel:
    def __init__(self, params: WeatherParams):
        """
        Initialize weather model with environmental intelligence.

        Args:
            params (WeatherParams): Weather modeling parameters

        Example:
            >>> params = WeatherParams()
            >>> weather_model = WeatherModel(params)
        """

    def update_weather(self, dt: float, session_type: SessionType):
        """
        Update weather conditions and track evolution.

        Args:
            dt (float): Time step for weather evolution
            session_type (SessionType): Current session type

        Example:
            >>> from src.weather import SessionType
            >>> weather_model.update_weather(0.1, SessionType.RACE)
        """

    def get_weather_effects(self) -> Dict:
        """
        Get current weather effects on thermal behavior.

        Returns:
            Dict: Weather effects dictionary containing:
                - 'thermal_factor': Thermal behavior modification factor
                - 'grip_factor': Grip modification factor
                - 'cooling_factor': Cooling enhancement factor
                - 'rain_probability': Current rain probability

        Example:
            >>> effects = weather_model.get_weather_effects()
            >>> print(f"Rain probability: {effects['rain_probability']:.1%}")
        """

    def get_weather_summary(self) -> Dict:
        """
        Get comprehensive weather summary.

        Returns:
            Dict: Complete weather state and conditions
        """

    def set_session_type(self, session_type: SessionType):
        """
        Set current session type for weather modeling.

        Args:
            session_type (SessionType): Session type to set
        """
```

### **DriverProfile**

Driver profiling system for personalized recommendations and behavior analysis.

```python
class DriverProfile:
    def __init__(self, name: str, params: DriverParams):
        """
        Initialize driver profile with personalization parameters.

        Args:
            name (str): Driver name
            params (DriverParams): Driver-specific parameters

        Example:
            >>> params = DriverParams(thermal_aggression=0.7, tire_awareness=0.8)
            >>> driver = DriverProfile("Lewis Hamilton", params)
        """

    def get_personalized_recommendations(self, thermal_state: np.ndarray,
                                       weather_summary: Dict) -> List[str]:
        """
        Get personalized recommendations based on driver profile.

        Args:
            thermal_state (np.ndarray): Current thermal state
            weather_summary (Dict): Current weather conditions

        Returns:
            List[str]: List of personalized recommendations

        Example:
            >>> thermal_state = np.array([95.0, 88.0, 45.0])
            >>> weather_summary = {'rain_probability': 0.2}
            >>> recommendations = driver.get_personalized_recommendations(thermal_state, weather_summary)
        """

    def update_performance_data(self, performance_data: Dict):
        """
        Update driver performance data for adaptive profiling.

        Args:
            performance_data (Dict): Performance metrics and telemetry data
        """

    def get_thermal_signature(self) -> Dict[str, float]:
        """
        Get driver's thermal signature preferences.

        Returns:
            Dict[str, float]: Thermal signature parameters
        """
```

---

## 🤖 **Machine Learning API**

### **MLStrategyOptimizer**

Machine learning-based strategy optimization using reinforcement learning and genetic algorithms.

```python
class MLStrategyOptimizer:
    def __init__(self, params: StrategyOptimizationParams):
        """
        Initialize ML strategy optimizer.

        Args:
            params (StrategyOptimizationParams): Optimization parameters

        Example:
            >>> params = StrategyOptimizationParams()
            >>> optimizer = MLStrategyOptimizer(params)
        """

    def optimize_strategy(self, race_context: Dict) -> Dict:
        """
        Optimize race strategy using machine learning.

        Args:
            race_context (Dict): Race context including:
                - 'track_length': Track length in meters
                - 'race_laps': Total race laps
                - 'weather_forecast': Weather conditions
                - 'competitor_data': Competitor information

        Returns:
            Dict: Optimized strategy containing:
                - 'pit_windows': Optimal pit stop windows
                - 'tire_strategy': Recommended tire compounds
                - 'driving_style': Optimal driving style parameters
                - 'confidence': Strategy confidence score

        Example:
            >>> race_context = {
            ...     'track_length': 5793,
            ...     'race_laps': 58,
            ...     'weather_forecast': {'rain_probability': 0.3}
            ... }
            >>> strategy = optimizer.optimize_strategy(race_context)
            >>> print(f"Confidence: {strategy['confidence']:.2f}")
        """

    def train_model(self, training_data: List[Dict]):
        """
        Train the ML model with historical race data.

        Args:
            training_data (List[Dict]): Historical race data for training
        """

    def evaluate_strategy(self, strategy: Dict, race_context: Dict) -> float:
        """
        Evaluate strategy performance.

        Args:
            strategy (Dict): Strategy to evaluate
            race_context (Dict): Race context

        Returns:
            float: Strategy fitness score
        """
```

### **PredictiveAnalytics**

Advanced predictive analytics engine with ensemble models for multiple prediction types.

```python
class PredictiveAnalytics:
    def __init__(self, params: PredictiveAnalyticsParams):
        """
        Initialize predictive analytics engine.

        Args:
            params (PredictiveAnalyticsParams): Analytics parameters
        """

    def predict_lap_time(self, thermal_state: np.ndarray, wear_summary: Dict,
                        weather_summary: Dict, race_context: Dict) -> Dict:
        """
        Predict lap time using ensemble models.

        Args:
            thermal_state (np.ndarray): Current thermal state
            wear_summary (Dict): Current wear levels
            weather_summary (Dict): Current weather conditions
            race_context (Dict): Race context information

        Returns:
            Dict: Prediction results containing:
                - 'predicted_lap_time': Predicted lap time in seconds
                - 'confidence': Prediction confidence (0.0-1.0)
                - 'factors': Contributing factors to prediction

        Example:
            >>> thermal_state = np.array([95.0, 88.0, 45.0])
            >>> wear_summary = {'FL': 0.1, 'FR': 0.12, 'RL': 0.08, 'RR': 0.09}
            >>> weather_summary = {'track_temp': 39.0, 'ambient_temp': 27.0}
            >>> race_context = {'track_length': 5793, 'fuel_load': 100}
            >>> prediction = analytics.predict_lap_time(thermal_state, wear_summary, weather_summary, race_context)
        """

    def predict_degradation(self, thermal_state: np.ndarray,
                          wear_summary: Dict, laps_ahead: int) -> Dict:
        """
        Predict tire degradation for specified laps ahead.

        Args:
            thermal_state (np.ndarray): Current thermal state
            wear_summary (Dict): Current wear levels
            laps_ahead (int): Number of laps to predict ahead

        Returns:
            Dict: Degradation prediction results
        """

    def train_models(self, training_data: List[Dict]):
        """
        Train predictive models with historical data.

        Args:
            training_data (List[Dict]): Historical training data
        """
```

---

## 📊 **Analytics API**

### **BigDataAnalytics**

Big data analytics engine for multi-source data integration and analysis.

```python
class BigDataAnalytics:
    def __init__(self, params: BigDataAnalyticsParams):
        """
        Initialize big data analytics engine.

        Args:
            params (BigDataAnalyticsParams): Analytics parameters
        """

    def store_telemetry_data(self, telemetry_data: Dict):
        """
        Store telemetry data for analysis.

        Args:
            telemetry_data (Dict): Telemetry data to store
        """

    def analyze_correlations(self, metric1: str, metric2: str) -> Dict:
        """
        Analyze correlations between two metrics.

        Args:
            metric1 (str): First metric name
            metric2 (str): Second metric name

        Returns:
            Dict: Correlation analysis results
        """

    def detect_anomalies(self, data: List[Dict]) -> List[Dict]:
        """
        Detect anomalies in telemetry data.

        Args:
            data (List[Dict]): Data to analyze for anomalies

        Returns:
            List[Dict]: Detected anomalies with details
        """

    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.

        Returns:
            Dict: Performance analysis report
        """
```

### **DataDrivenInsights**

Data-driven insights engine for automated insight generation and optimization recommendations.

```python
class DataDrivenInsights:
    def __init__(self, params: DataInsightsParams):
        """
        Initialize data-driven insights engine.

        Args:
            params (DataInsightsParams): Insights parameters
        """

    def generate_optimization_insights(self, performance_data: List[Dict]) -> List[Dict]:
        """
        Generate optimization insights from performance data.

        Args:
            performance_data (List[Dict]): Performance data to analyze

        Returns:
            List[Dict]: Optimization insights with recommendations
        """

    def generate_anomaly_insights(self, performance_data: List[Dict]) -> List[Dict]:
        """
        Generate anomaly detection insights.

        Args:
            performance_data (List[Dict]): Performance data to analyze

        Returns:
            List[Dict]: Anomaly insights with explanations
        """

    def generate_pattern_insights(self, performance_data: List[Dict]) -> List[Dict]:
        """
        Generate pattern recognition insights.

        Args:
            performance_data (List[Dict]): Performance data to analyze

        Returns:
            List[Dict]: Pattern insights with trends
        """
```

---

## 🎮 **Simulation API**

### **RaceSimulation**

Race simulation engine for scenario modeling and strategy testing.

```python
class RaceSimulation:
    def __init__(self, params: SimulationParams):
        """
        Initialize race simulation engine.

        Args:
            params (SimulationParams): Simulation parameters
        """

    async def run_simulation(self, scenario: Dict) -> Dict:
        """
        Run race simulation for specified scenario.

        Args:
            scenario (Dict): Simulation scenario parameters

        Returns:
            Dict: Simulation results containing:
                - 'lap_data': Lap-by-lap simulation data
                - 'performance_metrics': Performance summary
                - 'duration': Simulation duration

        Example:
            >>> scenario = {
            ...     'duration_laps': 58,
            ...     'weather_conditions': {'rain_probability': 0.2},
            ...     'tire_strategy': 'medium_start'
            ... }
            >>> results = await simulation.run_simulation(scenario)
        """

    def compare_scenarios(self, scenarios: List[Dict]) -> Dict:
        """
        Compare multiple simulation scenarios.

        Args:
            scenarios (List[Dict]): List of scenarios to compare

        Returns:
            Dict: Comparative analysis results
        """
```

---

## 📈 **Data Structures**

### **Core Data Types**

```python
@dataclass
class ThermalParams:
    """Thermal modeling parameters"""
    a1: float = 0.15  # Heat generation coefficient
    a2: float = 0.08  # Convective cooling coefficient
    a3: float = 0.12  # Conduction coefficient
    a4: float = 0.05  # Rim cooling coefficient
    a5: float = 0.03  # Environmental coupling
    b1: float = 0.02  # Thermal mass factor
    b2: float = 0.01  # Heat capacity factor
    c1: float = 0.10  # Load-dependent heating
    c2: float = 0.06  # Slip-dependent heating

@dataclass
class WearParams:
    """Wear modeling parameters"""
    base_wear_rate: float = 0.001
    thermal_factor: float = 1.5
    load_factor: float = 1.2
    slip_factor: float = 2.0
    compound_factors: Dict[str, float] = field(default_factory=lambda: {
        'soft': 1.5, 'medium': 1.0, 'hard': 0.7
    })

@dataclass
class WeatherParams:
    """Weather modeling parameters"""
    rain_probability: float = 0.1
    wind_speed: float = 5.0
    humidity: float = 60.0
    temperature_variation: float = 2.0
    track_evolution_rate: float = 0.01

@dataclass
class DriverParams:
    """Driver-specific parameters"""
    thermal_aggression: float = 0.5
    tire_awareness: float = 0.5
    wet_weather_skill: float = 0.5
    experience_level: float = 0.5
```

### **Enums**

```python
class SessionType(Enum):
    """Session type enumeration"""
    FP1 = "fp1"
    FP2 = "fp2"
    FP3 = "fp3"
    QUALIFYING = "qualifying"
    RACE = "race"

class DrivingStyle(Enum):
    """Driving style enumeration"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

class ExperienceLevel(Enum):
    """Experience level enumeration"""
    ROOKIE = "rookie"
    INTERMEDIATE = "intermediate"
    VETERAN = "veteran"
```

---

## 🔧 **Usage Examples**

### **Basic Thermal Modeling**

```python
from src.thermal import ThermalModel, ThermalParams
from src.wear import TireWearModel, WearParams
from src.weather import WeatherModel, WeatherParams

# Initialize components
thermal_params = ThermalParams()
wear_params = WearParams()
weather_params = WeatherParams()

thermal_model = ThermalModel(thermal_params)
wear_model = TireWearModel(wear_params)
weather_model = WeatherModel(weather_params)

# Integrated thermal model
integrated_model = ThermalModel(thermal_params, wear_model, weather_model)

# Simulate one time step
loads = {'Fx': 1000, 'Fy': 500, 'Fz': 4000, 'Mz': 50}
temperatures = integrated_model.step(0.1, loads, 'FL', 'medium')
print(f"Temperatures: {temperatures}")
```

### **Machine Learning Strategy Optimization**

```python
from src.strategy_optimization import StrategyOptimizer, StrategyOptimizationParams

# Initialize optimizer
params = StrategyOptimizationParams()
optimizer = StrategyOptimizer(params)

# Define race context
race_context = {
    'track_length': 5793,
    'race_laps': 58,
    'weather_forecast': {'rain_probability': 0.3},
    'competitor_data': {'avg_pit_time': 20.5}
}

# Optimize strategy
strategy = optimizer.optimize_strategy(race_context)
print(f"Optimal strategy: {strategy}")
```

### **Predictive Analytics**

```python
from src.predictive_analytics import PredictiveAnalytics, PredictiveAnalyticsParams

# Initialize analytics
params = PredictiveAnalyticsParams()
analytics = PredictiveAnalytics(params)

# Prepare prediction inputs
thermal_state = np.array([95.0, 88.0, 45.0])
wear_summary = {'FL': 0.1, 'FR': 0.12, 'RL': 0.08, 'RR': 0.09}
weather_summary = {'track_temp': 39.0, 'ambient_temp': 27.0, 'rain_probability': 0.2}
race_context = {'track_length': 5793, 'fuel_load': 100}

# Predict lap time
prediction = analytics.predict_lap_time(thermal_state, wear_summary, weather_summary, race_context)
print(f"Predicted lap time: {prediction['predicted_lap_time']:.2f}s")
print(f"Confidence: {prediction['confidence']:.2f}")
```

---

## 🚨 **Error Handling**

### **Common Exceptions**

```python
class ThermalModelError(Exception):
    """Thermal model specific errors"""
    pass

class WearModelError(Exception):
    """Wear model specific errors"""
    pass

class WeatherModelError(Exception):
    """Weather model specific errors"""
    pass

class MLModelError(Exception):
    """Machine learning model errors"""
    pass

class AnalyticsError(Exception):
    """Analytics engine errors"""
    pass
```

### **Error Handling Examples**

```python
try:
    temperatures = thermal_model.step(0.1, loads, 'FL', 'medium')
except ThermalModelError as e:
    print(f"Thermal model error: {e}")
    # Handle error appropriately

try:
    strategy = optimizer.optimize_strategy(race_context)
except MLModelError as e:
    print(f"ML optimization error: {e}")
    # Fallback to default strategy
```

---

## 📊 **Performance Considerations**

### **Optimization Tips**

1. **Batch Processing**: Process multiple corners simultaneously
2. **Caching**: Cache frequently accessed data
3. **Memory Management**: Use appropriate data types
4. **Parallel Processing**: Utilize multi-core processing for ML tasks

### **Performance Monitoring**

```python
import time
import psutil

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        print(f"Execution time: {end_time - start_time:.3f}s")
        print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")

        return result
    return wrapper

# Usage
@monitor_performance
def thermal_step():
    return thermal_model.step(0.1, loads, 'FL', 'medium')
```

---

## 📞 **Support**

For API-related questions and issues:

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this reference and technical specifications
- **Code Examples**: See `/examples` directory for usage patterns
- **Community**: Join discussions in GitHub discussions

---

**API Reference Version**: 2.0  
**Last Updated**: September 2025  
**Next Review**: September 2026

_This API reference provides comprehensive documentation for all classes, methods, and data structures in the F1 Tire Temperature Management System._
