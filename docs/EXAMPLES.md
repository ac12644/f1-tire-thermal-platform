# 🏎️ F1 Tire Thermal Platform

## Examples & Use Cases

**Version**: 2.0  
**Last Updated**: September 2025  
**Target Audience**: Developers, Data Scientists, F1 Engineers

---

## 📋 **Table of Contents**

1. [Basic Usage Examples](#basic-usage-examples)
2. [Advanced Analytics Examples](#advanced-analytics-examples)
3. [Machine Learning Examples](#machine-learning-examples)
4. [Multi-Driver Scenarios](#multi-driver-scenarios)
5. [Weather Integration Examples](#weather-integration-examples)
6. [Strategy Optimization Examples](#strategy-optimization-examples)
7. [Data Export Examples](#data-export-examples)
8. [Custom Integration Examples](#custom-integration-examples)

---

## 🚀 **Basic Usage Examples**

### **Simple Thermal Modeling**

```python
from src.thermal import ThermalModel, ThermalParams
from src.wear import TireWearModel, WearParams
import numpy as np

# Initialize thermal model
thermal_params = ThermalParams()
thermal_model = ThermalModel(thermal_params)

# Define load inputs
loads = {
    'Fx': 1000,  # Longitudinal force (N)
    'Fy': 500,   # Lateral force (N)
    'Fz': 4000,  # Vertical force (N)
    'Mz': 50     # Aligning moment (Nm)
}

# Simulate thermal behavior
temperatures = thermal_model.step(0.1, loads, 'FL', 'medium')
print(f"Temperatures - Tread: {temperatures[0]:.1f}°C, "
      f"Carcass: {temperatures[1]:.1f}°C, Rim: {temperatures[2]:.1f}°C")

# With wear integration
wear_params = WearParams()
wear_model = TireWearModel(wear_params)
integrated_model = ThermalModel(thermal_params, wear_model)

# Simulate with wear effects
for i in range(100):
    temps = integrated_model.step(0.1, loads, 'FL', 'medium')
    wear_model.update_wear(temps, loads, 0.1)

    if i % 20 == 0:
        effects = wear_model.get_wear_effects('FL')
        print(f"Step {i}: Wear level: {effects['wear_level']:.3f}, "
              f"Grip factor: {effects['grip_factor']:.3f}")
```

### **Weather Integration**

```python
from src.weather import WeatherModel, WeatherParams, SessionType
from src.thermal import ThermalModel, ThermalParams

# Initialize weather model
weather_params = WeatherParams(
    rain_probability=0.3,
    wind_speed=15.0,
    humidity=70.0
)
weather_model = WeatherModel(weather_params)

# Initialize thermal model with weather integration
thermal_params = ThermalParams()
thermal_model = ThermalModel(thermal_params, weather_model=weather_model)

# Simulate weather evolution
for i in range(50):
    weather_model.update_weather(0.1, SessionType.RACE)
    weather_effects = weather_model.get_weather_effects()

    loads = {'Fx': 1000, 'Fy': 500, 'Fz': 4000, 'Mz': 50}
    temps = thermal_model.step(0.1, loads, 'FL', 'medium')

    if i % 10 == 0:
        print(f"Time {i*0.1:.1f}s: Rain prob: {weather_effects['rain_probability']:.1%}, "
              f"Track temp: {weather_model.track_temp:.1f}°C, "
              f"Tread temp: {temps[0]:.1f}°C")
```

---

## 📊 **Advanced Analytics Examples**

### **Big Data Analytics**

```python
from src.big_data import BigDataAnalytics, BigDataAnalyticsParams
import numpy as np
from datetime import datetime, timedelta

# Initialize analytics engine
params = BigDataAnalyticsParams()
analytics = BigDataAnalytics(params)

# Generate sample telemetry data
base_time = datetime.now()
for i in range(1000):
    timestamp = base_time + timedelta(seconds=i*0.1)
    telemetry_data = {
        'timestamp': timestamp,
        'corner': 'FL',
        'tread_temp': 95.0 + np.random.normal(0, 2),
        'carcass_temp': 88.0 + np.random.normal(0, 1.5),
        'rim_temp': 45.0 + np.random.normal(0, 1),
        'wear_level': min(1.0, i * 0.001),
        'compound': 'medium',
        'track_temp': 39.0 + np.random.normal(0, 1),
        'ambient_temp': 27.0 + np.random.normal(0, 0.5),
        'speed': 180.0 + np.random.normal(0, 5),
        'loads': {'Fx': 1000, 'Fy': 500, 'Fz': 4000, 'Mz': 50}
    }
    analytics.store_telemetry_data(telemetry_data)

# Analyze correlations
correlation_result = analytics.analyze_correlations('tread_temp', 'wear_level')
print(f"Temperature-Wear Correlation: {correlation_result['correlation']:.3f}")

# Detect anomalies
anomalies = analytics.detect_anomalies(analytics.telemetry_cache[-100:])
print(f"Detected {len(anomalies)} anomalies")

# Generate performance report
report = analytics.generate_performance_report()
print(f"Performance report generated: {report['status']}")
```

### **Predictive Analytics**

```python
from src.predictive_analytics import PredictiveAnalytics, PredictiveAnalyticsParams
import numpy as np

# Initialize predictive analytics
params = PredictiveAnalyticsParams()
analytics = PredictiveAnalytics(params)

# Prepare training data
training_data = []
for i in range(500):
    thermal_state = np.array([
        95.0 + np.random.normal(0, 3),  # Tread temp
        88.0 + np.random.normal(0, 2),  # Carcass temp
        45.0 + np.random.normal(0, 1)   # Rim temp
    ])

    wear_summary = {
        'FL': min(1.0, np.random.exponential(0.1)),
        'FR': min(1.0, np.random.exponential(0.1)),
        'RL': min(1.0, np.random.exponential(0.1)),
        'RR': min(1.0, np.random.exponential(0.1))
    }

    weather_summary = {
        'track_temp': 39.0 + np.random.normal(0, 2),
        'ambient_temp': 27.0 + np.random.normal(0, 1),
        'rain_probability': np.random.uniform(0, 0.5)
    }

    race_context = {
        'track_length': 5793,
        'fuel_load': 100 - i * 0.5,
        'lap_number': i
    }

    # Simulate lap time based on conditions
    base_lap_time = 83.0
    temp_factor = 1.0 + (thermal_state[0] - 95.0) * 0.001
    wear_factor = 1.0 + sum(wear_summary.values()) * 0.01
    weather_factor = 1.0 + weather_summary['rain_probability'] * 0.05

    lap_time = base_lap_time * temp_factor * wear_factor * weather_factor

    training_data.append({
        'thermal_state': thermal_state,
        'wear_summary': wear_summary,
        'weather_summary': weather_summary,
        'race_context': race_context,
        'lap_time': lap_time
    })

# Train models
analytics.train_models(training_data)

# Make predictions
current_thermal = np.array([97.0, 89.0, 46.0])
current_wear = {'FL': 0.15, 'FR': 0.12, 'RL': 0.18, 'RR': 0.14}
current_weather = {'track_temp': 41.0, 'ambient_temp': 28.0, 'rain_probability': 0.1}
current_context = {'track_length': 5793, 'fuel_load': 50, 'lap_number': 25}

prediction = analytics.predict_lap_time(
    current_thermal, current_wear, current_weather, current_context
)

print(f"Predicted lap time: {prediction['predicted_lap_time']:.2f}s")
print(f"Confidence: {prediction['confidence']:.2f}")
```

---

## 🤖 **Machine Learning Examples**

### **Strategy Optimization**

```python
from src.strategy_optimization import StrategyOptimizer, StrategyOptimizationParams
from src.strategy_optimization import StrategyChromosome
import asyncio

# Initialize strategy optimizer
params = StrategyOptimizationParams()
optimizer = StrategyOptimizer(params)

# Define race context
race_context = {
    'track_length': 5793,  # Monaco
    'race_laps': 58,
    'weather_forecast': {
        'rain_probability': 0.3,
        'temperature_range': (25, 35),
        'humidity': 70
    },
    'competitor_data': {
        'avg_pit_time': 20.5,
        'fastest_lap': 82.5,
        'strategy_variants': ['soft_start', 'medium_start', 'hard_start']
    },
    'fuel_consumption': 2.5,  # kg/lap
    'tire_degradation': {
        'soft': 0.008,
        'medium': 0.005,
        'hard': 0.003
    }
}

# Optimize strategy
async def optimize_race_strategy():
    strategy = optimizer.optimize_strategy(race_context)
    return strategy

# Run optimization
strategy_result = asyncio.run(optimize_race_strategy())

print("Optimal Race Strategy:")
print(f"Pit Windows: {strategy_result['pit_windows']}")
print(f"Tire Strategy: {strategy_result['tire_strategy']}")
print(f"Driving Style: {strategy_result['driving_style']}")
print(f"Confidence: {strategy_result['confidence']:.2f}")
print(f"Expected Race Time: {strategy_result['expected_race_time']:.2f}s")
```

### **Driver Profiling**

```python
from src.driver import DriverProfile, DriverParams, DrivingStyle, ExperienceLevel
from src.driver_profiles import DriverProfiles

# Initialize driver profiles manager
driver_profiles = DriverProfiles()

# Create custom driver profile
custom_params = DriverParams(
    thermal_aggression=0.8,
    tire_awareness=0.9,
    wet_weather_skill=0.7,
    experience_level=0.8
)

custom_driver = DriverProfile("Custom Driver", custom_params)
driver_profiles.add_driver(custom_driver)

# Set as active driver
driver_profiles.set_active_driver("Custom Driver")

# Get personalized recommendations
thermal_state = np.array([98.0, 90.0, 47.0])
weather_summary = {'rain_probability': 0.2, 'track_temp': 40.0}

recommendations = custom_driver.get_personalized_recommendations(
    thermal_state, weather_summary
)

print("Personalized Recommendations:")
for rec in recommendations:
    print(f"- {rec}")

# Compare drivers
comparison = driver_profiles.compare_drivers(['Lewis Hamilton', 'Max Verstappen', 'Custom Driver'])
print("\nDriver Comparison:")
for driver_name, metrics in comparison.items():
    print(f"{driver_name}: {metrics}")
```

---

## 👥 **Multi-Driver Scenarios**

### **Multi-Driver Race Simulation**

```python
from src.driver_profiles import DriverProfiles
from src.simulation_engine import RaceSimulation, SimulationParams
import asyncio

# Initialize components
driver_profiles = DriverProfiles()
simulation_params = SimulationParams(
    duration_laps=10,
    time_step=0.1,
    track_length=5793
)
race_simulation = RaceSimulation(simulation_params)

# Define multi-driver scenario
multi_driver_scenario = {
    'drivers': ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc'],
    'starting_positions': [1, 2, 3],
    'tire_strategies': {
        'Lewis Hamilton': 'soft_start',
        'Max Verstappen': 'medium_start',
        'Charles Leclerc': 'hard_start'
    },
    'weather_conditions': {
        'rain_probability': 0.1,
        'temperature_evolution': 'increasing'
    },
    'race_dynamics': {
        'overtaking_probability': 0.3,
        'safety_car_probability': 0.2
    }
}

# Run multi-driver simulation
async def run_multi_driver_simulation():
    results = await race_simulation.run_simulation(multi_driver_scenario)
    return results

simulation_results = asyncio.run(run_multi_driver_simulation())

print("Multi-Driver Simulation Results:")
for driver, data in simulation_results['driver_results'].items():
    print(f"\n{driver}:")
    print(f"  Final Position: {data['final_position']}")
    print(f"  Average Lap Time: {data['avg_lap_time']:.2f}s")
    print(f"  Tire Strategy: {data['tire_strategy']}")
    print(f"  Pit Stops: {data['pit_stops']}")
```

### **Driver Performance Analysis**

```python
from src.performance_benchmarking import PerformanceBenchmarking, BenchmarkParams

# Initialize performance benchmarking
benchmark_params = BenchmarkParams()
benchmarking = PerformanceBenchmarking(benchmark_params)

# Generate performance data for multiple drivers
performance_data = []
drivers = ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc', 'Lando Norris']

for driver in drivers:
    # Simulate performance metrics
    driver_data = {
        'driver_name': driver,
        'thermal_efficiency': np.random.uniform(0.7, 0.95),
        'wear_management': np.random.uniform(0.6, 0.9),
        'consistency': np.random.uniform(0.8, 0.98),
        'adaptability': np.random.uniform(0.7, 0.9),
        'lap_times': [83.0 + np.random.normal(0, 0.5) for _ in range(20)],
        'temperature_management': np.random.uniform(0.75, 0.92)
    }
    performance_data.append(driver_data)

# Run benchmarking analysis
benchmark_results = benchmarking.benchmark_performance(performance_data)

print("Driver Performance Benchmarking:")
for driver, metrics in benchmark_results.items():
    print(f"\n{driver}:")
    print(f"  Overall Score: {metrics['overall_score']:.2f}")
    print(f"  Percentile Rank: {metrics['percentile_rank']:.1f}%")
    print(f"  Strengths: {metrics['strengths']}")
    print(f"  Areas for Improvement: {metrics['improvement_areas']}")
```

---

## 🌤️ **Weather Integration Examples**

### **Dynamic Weather Simulation**

```python
from src.weather import WeatherModel, WeatherParams, SessionType
import matplotlib.pyplot as plt

# Initialize weather model with dynamic parameters
weather_params = WeatherParams(
    rain_probability=0.2,
    wind_speed=10.0,
    humidity=60.0,
    temperature_variation=3.0,
    track_evolution_rate=0.02
)
weather_model = WeatherModel(weather_params)

# Simulate weather evolution over a race session
time_points = []
track_temps = []
ambient_temps = []
rain_probs = []

session_duration = 3600  # 1 hour session
time_step = 10  # 10 second intervals

for i in range(session_duration // time_step):
    weather_model.update_weather(time_step, SessionType.RACE)

    time_points.append(i * time_step)
    track_temps.append(weather_model.track_temp)
    ambient_temps.append(weather_model.ambient_temp)
    rain_probs.append(weather_model.rain_probability)

# Plot weather evolution
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time_points, track_temps, 'r-', label='Track Temperature')
plt.plot(time_points, ambient_temps, 'b-', label='Ambient Temperature')
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Evolution')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(time_points, rain_probs, 'g-', label='Rain Probability')
plt.xlabel('Time (seconds)')
plt.ylabel('Rain Probability')
plt.title('Rain Probability Evolution')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(time_points, [weather_model.wind_speed] * len(time_points), 'purple', label='Wind Speed')
plt.xlabel('Time (seconds)')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(time_points, [weather_model.humidity] * len(time_points), 'orange', label='Humidity')
plt.xlabel('Time (seconds)')
plt.ylabel('Humidity (%)')
plt.title('Humidity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Get weather effects on tire performance
weather_effects = weather_model.get_weather_effects()
print(f"Weather Effects:")
print(f"  Thermal Factor: {weather_effects['thermal_factor']:.3f}")
print(f"  Grip Factor: {weather_effects['grip_factor']:.3f}")
print(f"  Cooling Factor: {weather_effects['cooling_factor']:.3f}")
```

### **Weather-Based Strategy Adaptation**

```python
from src.weather import WeatherModel, WeatherParams, SessionType
from src.strategy_optimization import StrategyOptimizer, StrategyOptimizationParams

# Initialize weather and strategy components
weather_params = WeatherParams(rain_probability=0.4)
weather_model = WeatherModel(weather_params)

strategy_params = StrategyOptimizationParams()
strategy_optimizer = StrategyOptimizer(strategy_params)

# Simulate strategy adaptation to changing weather
strategies = []
weather_conditions = []

for lap in range(58):  # Full race distance
    # Update weather (simulating changing conditions)
    weather_model.update_weather(83.0, SessionType.RACE)  # 83s lap time

    # Get current weather conditions
    weather_summary = weather_model.get_weather_summary()
    weather_conditions.append(weather_summary.copy())

    # Adapt strategy based on weather
    race_context = {
        'track_length': 5793,
        'race_laps': 58,
        'current_lap': lap,
        'weather_forecast': weather_summary,
        'remaining_laps': 58 - lap
    }

    # Get weather-adapted strategy
    strategy = strategy_optimizer.optimize_strategy(race_context)
    strategies.append(strategy)

# Analyze strategy adaptation
print("Weather-Based Strategy Adaptation:")
for i in range(0, len(strategies), 10):  # Every 10 laps
    lap = i
    weather = weather_conditions[i]
    strategy = strategies[i]

    print(f"\nLap {lap + 1}:")
    print(f"  Rain Probability: {weather['rain_probability']:.1%}")
    print(f"  Track Temperature: {weather['track_temp']:.1f}°C")
    print(f"  Recommended Strategy: {strategy['tire_strategy']}")
    print(f"  Pit Window: {strategy['pit_windows']}")
    print(f"  Confidence: {strategy['confidence']:.2f}")
```

---

## 📊 **Data Export Examples**

### **Comprehensive Data Export**

```python
import pandas as pd
import json
from datetime import datetime
from src.big_data import BigDataAnalytics, BigDataAnalyticsParams

# Initialize analytics for data export
params = BigDataAnalyticsParams()
analytics = BigDataAnalytics(params)

# Generate comprehensive dataset
base_time = datetime.now()
telemetry_data = []

for i in range(1000):
    timestamp = base_time + timedelta(seconds=i*0.1)

    # Generate realistic telemetry data
    corner_data = {}
    for corner in ['FL', 'FR', 'RL', 'RR']:
        corner_data[corner] = {
            'tread_temp': 95.0 + np.random.normal(0, 2),
            'carcass_temp': 88.0 + np.random.normal(0, 1.5),
            'rim_temp': 45.0 + np.random.normal(0, 1),
            'wear_level': min(1.0, i * 0.001 + np.random.normal(0, 0.01))
        }

    telemetry_entry = {
        'timestamp': timestamp,
        'lap_number': i // 100,
        'lap_time': 83.0 + np.random.normal(0, 0.5),
        'speed': 180.0 + np.random.normal(0, 5),
        'track_temp': 39.0 + np.random.normal(0, 1),
        'ambient_temp': 27.0 + np.random.normal(0, 0.5),
        'rain_probability': np.random.uniform(0, 0.3),
        'compound': 'medium',
        'corners': corner_data
    }

    telemetry_data.append(telemetry_entry)
    analytics.store_telemetry_data(telemetry_entry)

# Export to different formats
def export_to_csv():
    # Flatten data for CSV export
    csv_data = []
    for entry in telemetry_data:
        for corner, temps in entry['corners'].items():
            csv_data.append({
                'timestamp': entry['timestamp'],
                'lap_number': entry['lap_number'],
                'corner': corner,
                'tread_temp': temps['tread_temp'],
                'carcass_temp': temps['carcass_temp'],
                'rim_temp': temps['rim_temp'],
                'wear_level': temps['wear_level'],
                'lap_time': entry['lap_time'],
                'speed': entry['speed'],
                'track_temp': entry['track_temp'],
                'ambient_temp': entry['ambient_temp'],
                'rain_probability': entry['rain_probability'],
                'compound': entry['compound']
            })

    df = pd.DataFrame(csv_data)
    df.to_csv('f1_telemetry_export.csv', index=False)
    print(f"Exported {len(csv_data)} records to CSV")

def export_to_json():
    # Export structured JSON
    export_data = {
        'export_info': {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(telemetry_data),
            'export_format': 'structured_json'
        },
        'telemetry_data': telemetry_data
    }

    with open('f1_telemetry_export.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"Exported {len(telemetry_data)} records to JSON")

def export_analytics_report():
    # Generate and export analytics report
    report = analytics.generate_performance_report()

    report_data = {
        'report_info': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'performance_analysis'
        },
        'report_data': report
    }

    with open('f1_analytics_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    print("Analytics report exported to JSON")

# Execute exports
export_to_csv()
export_to_json()
export_analytics_report()

# Display summary statistics
df = pd.read_csv('f1_telemetry_export.csv')
print(f"\nData Summary:")
print(f"Total Records: {len(df)}")
print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Average Tread Temperature: {df['tread_temp'].mean():.1f}°C")
print(f"Average Wear Level: {df['wear_level'].mean():.3f}")
print(f"Temperature Range: {df['tread_temp'].min():.1f}°C to {df['tread_temp'].max():.1f}°C")
```

---

## 🔗 **Custom Integration Examples**

### **REST API Integration**

```python
from flask import Flask, request, jsonify
from src.thermal import ThermalModel, ThermalParams
from src.wear import TireWearModel, WearParams
import json

# Initialize Flask app
app = Flask(__name__)

# Initialize F1 system components
thermal_params = ThermalParams()
wear_params = WearParams()
thermal_model = ThermalModel(thermal_params)
wear_model = TireWearModel(wear_params)

@app.route('/api/thermal/step', methods=['POST'])
def thermal_step():
    """API endpoint for thermal simulation step"""
    try:
        data = request.get_json()

        # Extract parameters
        dt = data.get('dt', 0.1)
        loads = data.get('loads', {})
        corner = data.get('corner', 'FL')
        compound = data.get('compound', 'medium')

        # Run thermal simulation
        temperatures = thermal_model.step(dt, loads, corner, compound)

        # Update wear model
        wear_model.update_wear(temperatures, loads, dt)
        wear_effects = wear_model.get_wear_effects(corner)

        # Prepare response
        response = {
            'temperatures': {
                'tread': float(temperatures[0]),
                'carcass': float(temperatures[1]),
                'rim': float(temperatures[2])
            },
            'wear_effects': wear_effects,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/api/thermal/state', methods=['GET'])
def get_thermal_state():
    """API endpoint to get current thermal state"""
    try:
        corner = request.args.get('corner', 'FL')
        thermal_state = thermal_model.get_thermal_state(corner)

        response = {
            'corner': corner,
            'temperatures': {
                'tread': float(thermal_state[0]),
                'carcass': float(thermal_state[1]),
                'rim': float(thermal_state[2])
            },
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/api/wear/summary', methods=['GET'])
def get_wear_summary():
    """API endpoint to get wear summary"""
    try:
        wear_summary = wear_model.get_wear_summary()

        response = {
            'wear_summary': wear_summary,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### **Database Integration**

```python
import sqlite3
import pandas as pd
from src.thermal import ThermalModel, ThermalParams
from datetime import datetime, timedelta

class F1TireDatabase:
    def __init__(self, db_path='f1_tire_data.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create telemetry table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                corner TEXT,
                tread_temp REAL,
                carcass_temp REAL,
                rim_temp REAL,
                wear_level REAL,
                compound TEXT,
                track_temp REAL,
                ambient_temp REAL,
                speed REAL,
                loads TEXT
            )
        ''')

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time DATETIME,
                end_time DATETIME,
                track_name TEXT,
                weather_conditions TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_telemetry(self, telemetry_data):
        """Store telemetry data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO telemetry
            (timestamp, corner, tread_temp, carcass_temp, rim_temp,
             wear_level, compound, track_temp, ambient_temp, speed, loads)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            telemetry_data['timestamp'],
            telemetry_data['corner'],
            telemetry_data['tread_temp'],
            telemetry_data['carcass_temp'],
            telemetry_data['rim_temp'],
            telemetry_data['wear_level'],
            telemetry_data['compound'],
            telemetry_data['track_temp'],
            telemetry_data['ambient_temp'],
            telemetry_data['speed'],
            json.dumps(telemetry_data['loads'])
        ))

        conn.commit()
        conn.close()

    def get_telemetry_history(self, corner=None, start_time=None, end_time=None):
        """Retrieve telemetry history"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM telemetry WHERE 1=1"
        params = []

        if corner:
            query += " AND corner = ?"
            params.append(corner)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_statistics(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get record counts
        cursor.execute("SELECT COUNT(*) FROM telemetry")
        total_records = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT corner) FROM telemetry")
        corners_count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM telemetry")
        time_range = cursor.fetchone()

        conn.close()

        return {
            'total_records': total_records,
            'corners_count': corners_count,
            'time_range': time_range
        }

# Example usage
db = F1TireDatabase()

# Initialize thermal model
thermal_params = ThermalParams()
thermal_model = ThermalModel(thermal_params)

# Simulate and store data
base_time = datetime.now()
for i in range(100):
    timestamp = base_time + timedelta(seconds=i*0.1)

    loads = {'Fx': 1000, 'Fy': 500, 'Fz': 4000, 'Mz': 50}
    temps = thermal_model.step(0.1, loads, 'FL', 'medium')

    telemetry_data = {
        'timestamp': timestamp,
        'corner': 'FL',
        'tread_temp': temps[0],
        'carcass_temp': temps[1],
        'rim_temp': temps[2],
        'wear_level': i * 0.001,
        'compound': 'medium',
        'track_temp': 39.0,
        'ambient_temp': 27.0,
        'speed': 180.0,
        'loads': loads
    }

    db.store_telemetry(telemetry_data)

# Retrieve and analyze data
history_df = db.get_telemetry_history(corner='FL')
stats = db.get_statistics()

print(f"Database Statistics:")
print(f"Total Records: {stats['total_records']}")
print(f"Corners: {stats['corners_count']}")
print(f"Time Range: {stats['time_range'][0]} to {stats['time_range'][1]}")

print(f"\nRecent Data Sample:")
print(history_df.tail())
```

---

## 📞 **Support & Resources**

### **Getting Help**

- **Documentation**: Check `/docs` directory for comprehensive guides
- **API Reference**: See `docs/API_REFERENCE.md` for detailed API documentation
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions in GitHub discussions

### **Contributing Examples**

- **Code Examples**: Submit new examples via pull requests
- **Use Cases**: Share your use cases and implementations
- **Improvements**: Suggest improvements to existing examples

---

**Examples Guide Version**: 2.0  
**Last Updated**: September 2025  
**Next Review**: September 2026

_This examples guide provides practical implementations and use cases for the F1 Tire Temperature Management System._
