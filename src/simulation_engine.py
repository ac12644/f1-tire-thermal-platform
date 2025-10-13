from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import random
from collections import defaultdict
import statistics
try:
    from .config import system_config
except ImportError:
    from config import system_config

class SimulationType(Enum):
    """Types of race simulations."""
    SINGLE_LAP = "single_lap"
    MULTI_LAP = "multi_lap"
    FULL_RACE = "full_race"
    QUALIFYING = "qualifying"
    PRACTICE = "practice"
    SCENARIO_TESTING = "scenario_testing"
    STRATEGY_COMPARISON = "strategy_comparison"
    WEATHER_SIMULATION = "weather_simulation"

class ScenarioType(Enum):
    """Types of race scenarios."""
    NORMAL_RACE = "normal_race"
    WET_RACE = "wet_race"
    DRY_TO_WET = "dry_to_wet"
    WET_TO_DRY = "wet_to_dry"
    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"
    MECHANICAL_FAILURE = "mechanical_failure"
    TIRE_FAILURE = "tire_failure"
    FUEL_SHORTAGE = "fuel_shortage"
    GRID_PENALTY = "grid_penalty"

class SimulationStatus(Enum):
    """Simulation execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SimulationParams:
    """Parameters for race simulation."""
    # Simulation parameters
    simulation_type: SimulationType = SimulationType.FULL_RACE
    scenario_type: ScenarioType = ScenarioType.NORMAL_RACE
    duration_laps: int = None  # Will use config default
    time_step: float = None    # Will use config default
    
    # Race parameters
    track_length: float = None  # Will use config default
    pit_lane_time: float = None  # Will use config default
    safety_car_probability: float = None  # Will use config default
    red_flag_probability: float = None  # Will use config default
    
    # Weather parameters
    weather_change_probability: float = None  # Will use config default
    rain_intensity_range: Tuple[float, float] = None  # Will use config default
    temperature_variation: float = None  # Will use config default
    
    # Performance parameters
    max_concurrent_simulations: int = None  # Will use config default
    simulation_timeout: int = None  # Will use config default
    cache_results: bool = None  # Will use config default
    
    # Advanced parameters
    monte_carlo_runs: int = 1000
    sensitivity_analysis: bool = True
    optimization_enabled: bool = True
    
    def __post_init__(self):
        """Initialize with config defaults if not provided."""
        sim_config = system_config.simulation_config
        
        if self.duration_laps is None:
            self.duration_laps = sim_config.typical_race_laps
        if self.time_step is None:
            self.time_step = sim_config.default_time_step
        if self.track_length is None:
            self.track_length = system_config.track_config.length_meters
        if self.pit_lane_time is None:
            self.pit_lane_time = sim_config.pit_lane_time
        if self.safety_car_probability is None:
            self.safety_car_probability = sim_config.safety_car_probability
        if self.red_flag_probability is None:
            self.red_flag_probability = sim_config.red_flag_probability
        if self.weather_change_probability is None:
            self.weather_change_probability = sim_config.weather_change_probability
        if self.rain_intensity_range is None:
            self.rain_intensity_range = sim_config.rain_intensity_range
        if self.temperature_variation is None:
            self.temperature_variation = sim_config.temperature_variation
        if self.max_concurrent_simulations is None:
            self.max_concurrent_simulations = system_config.max_concurrent_simulations
        if self.simulation_timeout is None:
            self.simulation_timeout = system_config.simulation_timeout
        if self.cache_results is None:
            self.cache_results = system_config.cache_results

class RaceSimulation:
    """
    Advanced race simulation engine for F1 tire temperature management.
    
    Features:
    - Multi-scenario race simulation (normal, wet, safety car, etc.)
    - Real-time simulation with configurable time steps
    - Monte Carlo simulation for statistical analysis
    - Strategy comparison and optimization
    - Weather evolution modeling
    - Safety car and red flag scenarios
    - Performance prediction and analysis
    - Concurrent simulation execution
    """
    
    def __init__(self, params: SimulationParams = None):
        self.p = params or SimulationParams()
        
        # Simulation state
        self.simulation_id = None
        self.status = SimulationStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.current_lap = 0
        self.current_time = 0.0
        
        # Race data
        self.race_data = {}
        self.lap_data = []
        self.telemetry_data = []
        self.strategy_data = {}
        self.weather_data = []
        
        # Simulation components
        self.thermal_model = None
        self.wear_model = None
        self.weather_model = None
        self.driver_profiles = None
        self.decision_engine = None
        
        # Results
        self.simulation_results = {}
        self.performance_metrics = {}
        self.strategy_analysis = {}
        self.weather_impact = {}
        
        # Concurrent execution
        self.executor = ThreadPoolExecutor(max_workers=self.p.max_concurrent_simulations)
        self.active_simulations = {}
        
        # Statistics
        self.simulation_stats = {
            'total_simulations': 0,
            'completed_simulations': 0,
            'failed_simulations': 0,
            'average_duration': 0.0,
            'success_rate': 0.0
        }
    
    def initialize_simulation(self, simulation_id: str, components: Dict[str, Any]):
        """
        Initialize simulation with required components.
        
        Args:
            simulation_id: Unique simulation identifier
            components: Dictionary containing simulation components
        """
        self.simulation_id = simulation_id
        self.status = SimulationStatus.PENDING
        
        # Store components
        self.thermal_model = components.get('thermal_model')
        self.wear_model = components.get('wear_model')
        self.weather_model = components.get('weather_model')
        self.driver_profiles = components.get('driver_profiles')
        self.decision_engine = components.get('decision_engine')
        
        # Initialize race data
        self.race_data = {
            'simulation_id': simulation_id,
            'simulation_type': self.p.simulation_type.value,
            'scenario_type': self.p.scenario_type.value,
            'track_length': self.p.track_length,
            'duration_laps': self.p.duration_laps,
            'start_time': datetime.now(),
            'status': self.status.value
        }
        
        # Initialize data structures
        self.lap_data = []
        self.telemetry_data = []
        self.strategy_data = {}
        self.weather_data = []
        
        # Reset simulation state
        self.current_lap = 0
        self.current_time = 0.0
        
        return True
    
    async def run_simulation(self) -> Dict[str, Any]:
        """
        Run the race simulation asynchronously.
        
        Returns:
            Dictionary with simulation results
        """
        try:
            self.status = SimulationStatus.RUNNING
            self.start_time = datetime.now()
            
            # Initialize simulation state
            await self._initialize_simulation_state()
            
            # Run simulation loop
            if self.p.simulation_type == SimulationType.SINGLE_LAP:
                await self._run_single_lap_simulation()
            elif self.p.simulation_type == SimulationType.MULTI_LAP:
                await self._run_multi_lap_simulation()
            elif self.p.simulation_type == SimulationType.FULL_RACE:
                await self._run_full_race_simulation()
            elif self.p.simulation_type == SimulationType.QUALIFYING:
                await self._run_qualifying_simulation()
            elif self.p.simulation_type == SimulationType.PRACTICE:
                await self._run_practice_simulation()
            elif self.p.simulation_type == SimulationType.SCENARIO_TESTING:
                await self._run_scenario_simulation()
            elif self.p.simulation_type == SimulationType.STRATEGY_COMPARISON:
                await self._run_strategy_comparison_simulation()
            elif self.p.simulation_type == SimulationType.WEATHER_SIMULATION:
                await self._run_weather_simulation()
            
            # Complete simulation
            self.status = SimulationStatus.COMPLETED
            self.end_time = datetime.now()
            
            # Generate results
            results = await self._generate_simulation_results()
            
            # Update statistics
            self._update_simulation_stats()
            
            return results
            
        except Exception as e:
            self.status = SimulationStatus.FAILED
            self.end_time = datetime.now()
            raise Exception(f"Simulation failed: {str(e)}")
    
    async def _initialize_simulation_state(self):
        """Initialize simulation state and components."""
        # Initialize thermal state
        if self.thermal_model:
            self.thermal_model.reset()
        
        # Initialize wear state
        if self.wear_model:
            self.wear_model.reset_wear()
        
        # Initialize weather state
        if self.weather_model:
            self.weather_model.reset()
        
        # Initialize race parameters
        self.current_lap = 0
        self.current_time = 0.0
        
        # Initialize scenario-specific parameters
        await self._initialize_scenario_state()
    
    async def _initialize_scenario_state(self):
        """Initialize scenario-specific simulation state."""
        if self.p.scenario_type == ScenarioType.WET_RACE:
            # Initialize wet race conditions
            if self.weather_model:
                self.weather_model.set_rain_probability(0.8)
                self.weather_model.set_track_temperature(25.0)
        elif self.p.scenario_type == ScenarioType.DRY_TO_WET:
            # Initialize dry-to-wet transition
            if self.weather_model:
                self.weather_model.set_rain_probability(0.1)
                self.weather_model.set_weather_change_probability(0.3)
        elif self.p.scenario_type == ScenarioType.SAFETY_CAR:
            # Initialize safety car scenario
            self.race_data['safety_car_probability'] = 0.5
        elif self.p.scenario_type == ScenarioType.RED_FLAG:
            # Initialize red flag scenario
            self.race_data['red_flag_probability'] = 0.3
        elif self.p.scenario_type == ScenarioType.MECHANICAL_FAILURE:
            # Initialize mechanical failure scenario
            self.race_data['mechanical_failure_probability'] = 0.1
        elif self.p.scenario_type == ScenarioType.TIRE_FAILURE:
            # Initialize tire failure scenario
            self.race_data['tire_failure_probability'] = 0.05
    
    async def _run_single_lap_simulation(self):
        """Run single lap simulation."""
        lap_data = await self._simulate_lap()
        self.lap_data.append(lap_data)
        self.current_lap = 1
    
    async def _run_multi_lap_simulation(self):
        """Run multi-lap simulation."""
        num_laps = min(10, self.p.duration_laps)  # Limit for multi-lap
        
        for lap in range(num_laps):
            lap_data = await self._simulate_lap()
            self.lap_data.append(lap_data)
            self.current_lap = lap + 1
            
            # Check for early termination conditions
            if await self._check_termination_conditions():
                break
    
    async def _run_full_race_simulation(self):
        """Run full race simulation."""
        for lap in range(self.p.duration_laps):
            lap_data = await self._simulate_lap()
            self.lap_data.append(lap_data)
            self.current_lap = lap + 1
            
            # Update weather conditions
            await self._update_weather_conditions()
            
            # Check for race events
            await self._check_race_events()
            
            # Check for early termination conditions
            if await self._check_termination_conditions():
                break
            
            # Add small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
    
    async def _run_qualifying_simulation(self):
        """Run qualifying simulation."""
        # Qualifying has 3 sessions
        sessions = ['Q1', 'Q2', 'Q3']
        
        for session in sessions:
            session_data = {
                'session': session,
                'laps': [],
                'best_lap_time': float('inf'),
                'position': 0
            }
            
            # Simulate session
            num_laps = 5 if session == 'Q1' else (3 if session == 'Q2' else 1)
            
            for lap in range(num_laps):
                lap_data = await self._simulate_lap()
                session_data['laps'].append(lap_data)
                
                # Track best lap time
                if lap_data['lap_time'] < session_data['best_lap_time']:
                    session_data['best_lap_time'] = lap_data['lap_time']
            
            self.lap_data.append(session_data)
    
    async def _run_practice_simulation(self):
        """Run practice session simulation."""
        # Practice sessions are longer but less intense
        num_laps = min(30, self.p.duration_laps)
        
        for lap in range(num_laps):
            lap_data = await self._simulate_lap()
            self.lap_data.append(lap_data)
            self.current_lap = lap + 1
            
            # Practice sessions have more experimental setups
            if lap % 5 == 0:
                await self._simulate_setup_changes()
    
    async def _run_scenario_simulation(self):
        """Run scenario testing simulation."""
        # Test specific scenarios
        scenarios = [
            ScenarioType.NORMAL_RACE,
            ScenarioType.WET_RACE,
            ScenarioType.SAFETY_CAR,
            ScenarioType.MECHANICAL_FAILURE
        ]
        
        for scenario in scenarios:
            # Update scenario
            original_scenario = self.p.scenario_type
            self.p.scenario_type = scenario
            
            # Run simulation for this scenario
            await self._run_multi_lap_simulation()
            
            # Store scenario results
            self.simulation_results[f'scenario_{scenario.value}'] = {
                'lap_data': self.lap_data.copy(),
                'performance_metrics': self._calculate_performance_metrics()
            }
            
            # Reset for next scenario
            self.p.scenario_type = original_scenario
            self.lap_data = []
            self.current_lap = 0
    
    async def _run_strategy_comparison_simulation(self):
        """Run strategy comparison simulation."""
        # Define different strategies
        strategies = [
            {'name': 'conservative', 'pit_stops': 1, 'compound_sequence': ['medium', 'hard']},
            {'name': 'aggressive', 'pit_stops': 2, 'compound_sequence': ['soft', 'soft', 'medium']},
            {'name': 'balanced', 'pit_stops': 2, 'compound_sequence': ['soft', 'medium', 'hard']}
        ]
        
        for strategy in strategies:
            # Initialize strategy
            self.strategy_data = strategy
            
            # Run simulation with this strategy
            await self._run_multi_lap_simulation()
            
            # Store strategy results
            self.simulation_results[f'strategy_{strategy["name"]}'] = {
                'lap_data': self.lap_data.copy(),
                'performance_metrics': self._calculate_performance_metrics(),
                'strategy': strategy
            }
            
            # Reset for next strategy
            self.lap_data = []
            self.current_lap = 0
    
    async def _run_weather_simulation(self):
        """Run weather simulation."""
        # Simulate different weather conditions
        weather_conditions = [
            {'rain_probability': 0.0, 'track_temp': 35.0, 'name': 'dry'},
            {'rain_probability': 0.3, 'track_temp': 30.0, 'name': 'light_rain'},
            {'rain_probability': 0.7, 'track_temp': 25.0, 'name': 'heavy_rain'},
            {'rain_probability': 0.1, 'track_temp': 40.0, 'name': 'hot_dry'}
        ]
        
        for weather in weather_conditions:
            # Set weather conditions
            if self.weather_model:
                self.weather_model.set_rain_probability(weather['rain_probability'])
                self.weather_model.set_track_temperature(weather['track_temp'])
            
            # Run simulation with this weather
            await self._run_multi_lap_simulation()
            
            # Store weather results
            self.simulation_results[f'weather_{weather["name"]}'] = {
                'lap_data': self.lap_data.copy(),
                'performance_metrics': self._calculate_performance_metrics(),
                'weather': weather
            }
            
            # Reset for next weather
            self.lap_data = []
            self.current_lap = 0
    
    async def _simulate_lap(self) -> Dict[str, Any]:
        """Simulate a single lap."""
        lap_start_time = self.current_time
        
        # Initialize lap data
        lap_data = {
            'lap_number': self.current_lap + 1,
            'start_time': lap_start_time,
            'end_time': 0.0,
            'lap_time': 0.0,
            'telemetry': [],
            'events': [],
            'performance_metrics': {}
        }
        
        # Simulate lap progression
        lap_duration = 90.0  # Base lap time in seconds
        time_step = self.p.time_step
        
        for step in range(int(lap_duration / time_step)):
            step_time = lap_start_time + step * time_step
            
            # Get current state
            thermal_state = self.thermal_model.get_state() if self.thermal_model else np.array([90.0, 85.0, 80.0])
            wear_state = self.wear_model.get_wear_summary() if self.wear_model else {}
            weather_state = self.weather_model.get_weather_summary() if self.weather_model else {}
            
            # Simulate step
            step_data = await self._simulate_step(step_time, thermal_state, wear_state, weather_state)
            lap_data['telemetry'].append(step_data)
            
            # Check for events
            events = await self._check_lap_events(step_time, step_data)
            if events:
                lap_data['events'].extend(events)
        
        # Calculate lap metrics
        lap_data['end_time'] = lap_start_time + lap_duration
        lap_data['lap_time'] = lap_duration
        lap_data['performance_metrics'] = self._calculate_lap_metrics(lap_data)
        
        # Update simulation time
        self.current_time = lap_data['end_time']
        
        return lap_data
    
    async def _simulate_step(self, step_time: float, thermal_state: np.ndarray, 
                           wear_state: Dict, weather_state: Dict) -> Dict[str, Any]:
        """Simulate a single time step."""
        step_data = {
            'timestamp': step_time,
            'thermal_state': thermal_state.tolist(),
            'wear_state': wear_state,
            'weather_state': weather_state,
            'position': 1,  # Default position
            'speed': 200.0,  # km/h
            'throttle': 0.8,
            'brake': 0.2,
            'steering': 0.1
        }
        
        # Update thermal model
        if self.thermal_model:
            self.thermal_model.step(thermal_state, wear_state, weather_state)
            step_data['thermal_state'] = self.thermal_model.get_state().tolist()
        
        # Update wear model
        if self.wear_model:
            self.wear_model.update_wear(thermal_state, wear_state, weather_state)
            step_data['wear_state'] = self.wear_model.get_wear_summary()
        
        # Update weather model
        if self.weather_model:
            self.weather_model.update_weather(step_time)
            step_data['weather_state'] = self.weather_model.get_weather_summary()
        
        return step_data
    
    async def _check_lap_events(self, step_time: float, step_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for lap events."""
        events = []
        
        # Check for tire failure
        if self.p.scenario_type == ScenarioType.TIRE_FAILURE:
            if random.random() < self.race_data.get('tire_failure_probability', 0.05):
                events.append({
                    'type': 'tire_failure',
                    'timestamp': step_time,
                    'description': 'Tire failure detected',
                    'severity': 'critical'
                })
        
        # Check for mechanical failure
        if self.p.scenario_type == ScenarioType.MECHANICAL_FAILURE:
            if random.random() < self.race_data.get('mechanical_failure_probability', 0.1):
                events.append({
                    'type': 'mechanical_failure',
                    'timestamp': step_time,
                    'description': 'Mechanical failure detected',
                    'severity': 'high'
                })
        
        # Check for safety car
        if self.p.scenario_type == ScenarioType.SAFETY_CAR:
            if random.random() < self.race_data.get('safety_car_probability', 0.1):
                events.append({
                    'type': 'safety_car',
                    'timestamp': step_time,
                    'description': 'Safety car deployed',
                    'severity': 'medium'
                })
        
        return events
    
    async def _check_race_events(self):
        """Check for race-level events."""
        # Check for red flag
        if self.p.scenario_type == ScenarioType.RED_FLAG:
            if random.random() < self.race_data.get('red_flag_probability', 0.02):
                self.race_data['red_flag'] = True
                self.race_data['red_flag_time'] = self.current_time
    
    async def _check_termination_conditions(self) -> bool:
        """Check if simulation should terminate early."""
        # Check for red flag
        if self.race_data.get('red_flag', False):
            return True
        
        # Check for mechanical failure
        if self.p.scenario_type == ScenarioType.MECHANICAL_FAILURE:
            if random.random() < 0.1:  # 10% chance of early termination
                return True
        
        return False
    
    async def _update_weather_conditions(self):
        """Update weather conditions during simulation."""
        if self.weather_model and random.random() < self.p.weather_change_probability:
            # Simulate weather change
            new_rain_prob = random.uniform(0.0, 1.0)
            new_track_temp = random.uniform(20.0, 40.0)
            
            self.weather_model.set_rain_probability(new_rain_prob)
            self.weather_model.set_track_temperature(new_track_temp)
    
    async def _simulate_setup_changes(self):
        """Simulate setup changes during practice."""
        # Practice sessions allow for setup experimentation
        if self.thermal_model:
            # Simulate setup changes
            setup_changes = {
                'tire_pressure': random.uniform(1.3, 1.7),
                'camber': random.uniform(-3.0, -1.0),
                'toe': random.uniform(-0.5, 0.5)
            }
            
            # Apply setup changes (simplified)
            self.race_data['setup_changes'] = setup_changes
    
    def _calculate_lap_metrics(self, lap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate lap performance metrics."""
        telemetry = lap_data['telemetry']
        
        if not telemetry:
            return {}
        
        # Calculate metrics
        avg_tread_temp = statistics.mean([t['thermal_state'][0] for t in telemetry])
        avg_wear = statistics.mean([t['wear_state'].get('FL', {}).get('wear_level', 0) for t in telemetry])
        avg_speed = statistics.mean([t['speed'] for t in telemetry])
        
        return {
            'avg_tread_temp': avg_tread_temp,
            'avg_wear': avg_wear,
            'avg_speed': avg_speed,
            'lap_time': lap_data['lap_time'],
            'events_count': len(lap_data['events'])
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        if not self.lap_data:
            return {}
        
        # Calculate metrics across all laps
        lap_times = [lap['lap_time'] for lap in self.lap_data if isinstance(lap, dict) and 'lap_time' in lap]
        avg_tread_temps = []
        avg_wear_levels = []
        
        for lap in self.lap_data:
            if isinstance(lap, dict) and 'performance_metrics' in lap:
                metrics = lap['performance_metrics']
                if 'avg_tread_temp' in metrics:
                    avg_tread_temps.append(metrics['avg_tread_temp'])
                if 'avg_wear' in metrics:
                    avg_wear_levels.append(metrics['avg_wear'])
        
        return {
            'total_laps': len(self.lap_data),
            'avg_lap_time': statistics.mean(lap_times) if lap_times else 0.0,
            'best_lap_time': min(lap_times) if lap_times else 0.0,
            'avg_tread_temp': statistics.mean(avg_tread_temps) if avg_tread_temps else 0.0,
            'avg_wear_level': statistics.mean(avg_wear_levels) if avg_wear_levels else 0.0,
            'total_events': sum(len(lap.get('events', [])) for lap in self.lap_data if isinstance(lap, dict)),
            'simulation_duration': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0.0
        }
    
    async def _generate_simulation_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results."""
        results = {
            'simulation_id': self.simulation_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0.0,
            'simulation_type': self.p.simulation_type.value,
            'scenario_type': self.p.scenario_type.value,
            'race_data': self.race_data,
            'lap_data': self.lap_data,
            'performance_metrics': self._calculate_performance_metrics(),
            'strategy_analysis': self.strategy_analysis,
            'weather_impact': self.weather_impact,
            'simulation_results': self.simulation_results
        }
        
        return results
    
    def _update_simulation_stats(self):
        """Update simulation statistics."""
        self.simulation_stats['total_simulations'] += 1
        
        if self.status == SimulationStatus.COMPLETED:
            self.simulation_stats['completed_simulations'] += 1
        elif self.status == SimulationStatus.FAILED:
            self.simulation_stats['failed_simulations'] += 1
        
        # Calculate success rate
        total = self.simulation_stats['total_simulations']
        completed = self.simulation_stats['completed_simulations']
        self.simulation_stats['success_rate'] = completed / total if total > 0 else 0.0
        
        # Update average duration
        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            current_avg = self.simulation_stats['average_duration']
            count = self.simulation_stats['completed_simulations']
            self.simulation_stats['average_duration'] = (current_avg * (count - 1) + duration) / count
    
    async def run_monte_carlo_simulation(self, num_runs: int = None) -> Dict[str, Any]:
        """Run Monte Carlo simulation for statistical analysis."""
        if num_runs is None:
            num_runs = self.p.monte_carlo_runs
        
        results = []
        
        # Run multiple simulations
        for run in range(num_runs):
            # Reset simulation state
            self.status = SimulationStatus.PENDING
            self.lap_data = []
            self.current_lap = 0
            self.current_time = 0.0
            
            # Run simulation
            try:
                result = await self.run_simulation()
                results.append(result)
            except Exception as e:
                print(f"Monte Carlo run {run} failed: {str(e)}")
                continue
        
        # Analyze results
        monte_carlo_analysis = self._analyze_monte_carlo_results(results)
        
        return {
            'monte_carlo_runs': num_runs,
            'successful_runs': len(results),
            'results': results,
            'analysis': monte_carlo_analysis
        }
    
    def _analyze_monte_carlo_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        if not results:
            return {}
        
        # Extract metrics
        lap_times = []
        avg_temps = []
        avg_wear = []
        
        for result in results:
            metrics = result.get('performance_metrics', {})
            if 'avg_lap_time' in metrics:
                lap_times.append(metrics['avg_lap_time'])
            if 'avg_tread_temp' in metrics:
                avg_temps.append(metrics['avg_tread_temp'])
            if 'avg_wear_level' in metrics:
                avg_wear.append(metrics['avg_wear_level'])
        
        # Calculate statistics
        analysis = {
            'lap_time_stats': {
                'mean': statistics.mean(lap_times) if lap_times else 0.0,
                'std': statistics.stdev(lap_times) if len(lap_times) > 1 else 0.0,
                'min': min(lap_times) if lap_times else 0.0,
                'max': max(lap_times) if lap_times else 0.0,
                'percentiles': self._calculate_percentiles(lap_times)
            },
            'temperature_stats': {
                'mean': statistics.mean(avg_temps) if avg_temps else 0.0,
                'std': statistics.stdev(avg_temps) if len(avg_temps) > 1 else 0.0,
                'min': min(avg_temps) if avg_temps else 0.0,
                'max': max(avg_temps) if avg_temps else 0.0
            },
            'wear_stats': {
                'mean': statistics.mean(avg_wear) if avg_wear else 0.0,
                'std': statistics.stdev(avg_wear) if len(avg_wear) > 1 else 0.0,
                'min': min(avg_wear) if avg_wear else 0.0,
                'max': max(avg_wear) if avg_wear else 0.0
            }
        }
        
        return analysis
    
    def _calculate_percentiles(self, data: List[float]) -> Dict[int, float]:
        """Calculate percentiles for data."""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        percentiles = {}
        
        for p in [25, 50, 75, 90, 95]:
            index = int((p / 100) * (len(sorted_data) - 1))
            percentiles[p] = sorted_data[index]
        
        return percentiles
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary."""
        return {
            'simulation_stats': self.simulation_stats,
            'active_simulations': len(self.active_simulations),
            'simulation_params': {
                'simulation_type': self.p.simulation_type.value,
                'scenario_type': self.p.scenario_type.value,
                'duration_laps': self.p.duration_laps,
                'time_step': self.p.time_step,
                'monte_carlo_runs': self.p.monte_carlo_runs
            },
            'current_simulation': {
                'simulation_id': self.simulation_id,
                'status': self.status.value,
                'current_lap': self.current_lap,
                'current_time': self.current_time
            } if self.simulation_id else None
        }
    
    def cancel_simulation(self):
        """Cancel running simulation."""
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.CANCELLED
            self.end_time = datetime.now()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
