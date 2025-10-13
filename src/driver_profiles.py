from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from driver import DriverProfile, DriverParams, DrivingStyle, DriverExperience

class DriverProfiles:
    """
    Manages multiple driver profiles and provides multi-driver analysis.
    
    Features:
    - Multiple driver profile management
    - Driver comparison and analysis
    - Multi-driver race simulation
    - Driver performance tracking
    - Personalized strategy recommendations
    """
    
    def __init__(self):
        self.drivers: Dict[str, DriverProfile] = {}
        self.active_driver: Optional[str] = None
        self.driver_comparison_data = {}
        
        # Initialize with some example drivers
        self._create_example_drivers()
    
    def _create_example_drivers(self):
        """Create example driver profiles for demonstration."""
        # Aggressive Champion Driver
        aggressive_params = DriverParams(
            thermal_aggression=1.4,
            thermal_efficiency=1.2,
            brake_aggression=1.3,
            throttle_aggression=1.4,
            steering_aggression=1.2,
            tire_awareness=1.3,
            pressure_management=1.2,
            compound_adaptation=1.3,
            overtaking_aggression=1.5,
            wet_weather_skill=1.1,
            temperature_adaptation=1.2
        )
        
        self.add_driver("Max Verstappen", aggressive_params, 
                       DrivingStyle.AGGRESSIVE, DriverExperience.CHAMPION)
        
        # Conservative Veteran Driver
        conservative_params = DriverParams(
            thermal_aggression=0.7,
            thermal_efficiency=1.3,
            brake_aggression=0.8,
            throttle_aggression=0.7,
            steering_aggression=0.9,
            tire_awareness=1.4,
            pressure_management=1.3,
            compound_adaptation=1.1,
            overtaking_aggression=0.8,
            wet_weather_skill=1.2,
            temperature_adaptation=1.1
        )
        
        self.add_driver("Lewis Hamilton", conservative_params,
                       DrivingStyle.CONSERVATIVE, DriverExperience.VETERAN)
        
        # Balanced Experienced Driver
        balanced_params = DriverParams(
            thermal_aggression=1.0,
            thermal_efficiency=1.1,
            brake_aggression=1.0,
            throttle_aggression=1.0,
            steering_aggression=1.0,
            tire_awareness=1.1,
            pressure_management=1.1,
            compound_adaptation=1.1,
            overtaking_aggression=1.0,
            wet_weather_skill=1.0,
            temperature_adaptation=1.0
        )
        
        self.add_driver("Charles Leclerc", balanced_params,
                       DrivingStyle.BALANCED, DriverExperience.EXPERIENCED)
        
        # Smooth Rookie Driver
        smooth_params = DriverParams(
            thermal_aggression=0.6,
            thermal_efficiency=0.9,
            brake_aggression=0.7,
            throttle_aggression=0.6,
            steering_aggression=0.8,
            tire_awareness=0.8,
            pressure_management=0.8,
            compound_adaptation=0.7,
            overtaking_aggression=0.6,
            wet_weather_skill=0.8,
            temperature_adaptation=0.8
        )
        
        self.add_driver("Oscar Piastri", smooth_params,
                       DrivingStyle.SMOOTH, DriverExperience.ROOKIE)
        
        # Set default active driver
        self.active_driver = "Lewis Hamilton"
    
    def add_driver(self, name: str, params: DriverParams, 
                   style: DrivingStyle, experience: DriverExperience):
        """Add a new driver profile."""
        driver = DriverProfile(name, params, style, experience)
        self.drivers[name] = driver
        return driver
    
    def get_driver(self, name: str) -> Optional[DriverProfile]:
        """Get driver profile by name."""
        return self.drivers.get(name)
    
    def get_active_driver(self) -> Optional[DriverProfile]:
        """Get currently active driver."""
        if self.active_driver:
            return self.drivers.get(self.active_driver)
        return None
    
    def set_active_driver(self, name: str):
        """Set active driver."""
        if name in self.drivers:
            self.active_driver = name
    
    def get_all_drivers(self) -> List[DriverProfile]:
        """Get all driver profiles."""
        return list(self.drivers.values())
    
    def get_driver_names(self) -> List[str]:
        """Get list of all driver names."""
        return list(self.drivers.keys())
    
    def compare_drivers(self, driver_names: List[str] = None) -> Dict[str, Dict]:
        """
        Compare multiple drivers across various metrics.
        
        Args:
            driver_names: List of driver names to compare (None = all drivers)
            
        Returns:
            Dict with comparison data
        """
        if driver_names is None:
            driver_names = list(self.drivers.keys())
        
        comparison = {}
        
        for name in driver_names:
            if name in self.drivers:
                driver = self.drivers[name]
                summary = driver.get_driver_summary()
                
                comparison[name] = {
                    'style': summary['style'],
                    'experience': summary['experience'],
                    'thermal_signature': summary['thermal_signature'],
                    'thermal_consistency': summary['thermal_consistency'],
                    'recommendation_follow_rate': summary['recommendation_follow_rate'],
                    'session_laps': summary['session_laps'],
                    'average_lap_time': summary['average_lap_time']
                }
        
        return comparison
    
    def get_driver_rankings(self, metric: str = 'thermal_consistency') -> List[Tuple[str, float]]:
        """
        Get driver rankings based on specified metric.
        
        Args:
            metric: Metric to rank by ('thermal_consistency', 'recommendation_follow_rate', etc.)
            
        Returns:
            List of (driver_name, metric_value) tuples sorted by metric
        """
        rankings = []
        
        for name, driver in self.drivers.items():
            summary = driver.get_driver_summary()
            value = summary.get(metric, 0.0)
            rankings.append((name, value))
        
        # Sort by metric value (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_personalized_recommendations(self, driver_name: str, thermal_state: np.ndarray,
                                       conditions: Dict, wear_summary: Dict) -> List[Tuple[str, str]]:
        """
        Get personalized recommendations for specific driver.
        
        Args:
            driver_name: Name of driver
            thermal_state: Current thermal state
            conditions: Environmental conditions
            wear_summary: Wear status
            
        Returns:
            List of personalized recommendations
        """
        driver = self.get_driver(driver_name)
        if driver is None:
            return []
        
        return driver.get_personalized_recommendations(thermal_state, conditions, wear_summary)
    
    def simulate_multi_driver_race(self, conditions: Dict, laps: int = 10) -> Dict[str, List]:
        """
        Simulate a multi-driver race scenario.
        
        Args:
            conditions: Race conditions (weather, track, etc.)
            laps: Number of laps to simulate
            
        Returns:
            Dict with simulation results for each driver
        """
        simulation_results = {}
        
        for name, driver in self.drivers.items():
            driver_results = {
                'lap_times': [],
                'thermal_states': [],
                'recommendations': [],
                'performance_metrics': []
            }
            
            # Simulate race for this driver
            for lap in range(laps):
                # Simulate thermal state based on driver characteristics
                thermal_mult = driver.get_thermal_multipliers(conditions)
                
                # Base thermal state with driver-specific variations
                base_temp = 95.0
                thermal_variation = 5.0 * thermal_mult['thermal_generation']
                thermal_state = np.array([
                    base_temp + thermal_variation * np.random.normal(),
                    base_temp - 5.0 + thermal_variation * np.random.normal(),
                    base_temp - 10.0 + thermal_variation * np.random.normal()
                ])
                
                # Simulate lap time based on driver characteristics
                base_lap_time = 85.0  # Base lap time in seconds
                performance_factor = thermal_mult['thermal_efficiency']
                lap_time = base_lap_time / performance_factor + np.random.normal(0, 0.5)
                
                # Get personalized recommendations
                recommendations = driver.get_personalized_recommendations(
                    thermal_state, conditions, {}
                )
                
                # Store results
                driver_results['lap_times'].append(lap_time)
                driver_results['thermal_states'].append(thermal_state.copy())
                driver_results['recommendations'].append(recommendations)
                
                # Update driver session data
                driver.update_session_data(thermal_state, lap_time, conditions, recommendations)
            
            simulation_results[name] = driver_results
        
        return simulation_results
    
    def get_race_strategy_recommendations(self, driver_name: str, conditions: Dict) -> List[Tuple[str, str]]:
        """
        Get race strategy recommendations for specific driver.
        
        Args:
            driver_name: Name of driver
            conditions: Race conditions
            
        Returns:
            List of strategy recommendations
        """
        driver = self.get_driver(driver_name)
        if driver is None:
            return []
        
        recommendations = []
        
        # Compound strategy based on driver characteristics
        thermal_aggression = driver.params.thermal_aggression
        compound_adaptation = driver.params.compound_adaptation
        
        if thermal_aggression > 1.2:
            recommendations.append(("STRATEGY", f"Aggressive driver: Consider soft compound for qualifying"))
        elif thermal_aggression < 0.8:
            recommendations.append(("STRATEGY", f"Conservative driver: Hard compound for long stints"))
        
        # Weather strategy
        rain_prob = conditions.get('rain_probability', 0.0)
        wet_skill = driver.params.wet_weather_skill
        
        if rain_prob > 0.5:
            if wet_skill > 1.2:
                recommendations.append(("STRATEGY", f"Wet weather expert: Use rain to your advantage"))
            elif wet_skill < 0.8:
                recommendations.append(("STRATEGY", f"Wet conditions: Focus on survival, avoid risks"))
        
        # Experience-based strategy
        if driver.experience == DriverExperience.ROOKIE:
            recommendations.append(("STRATEGY", f"Rookie: Conservative strategy, focus on learning"))
        elif driver.experience == DriverExperience.CHAMPION:
            recommendations.append(("STRATEGY", f"Champion: Aggressive strategy, push boundaries"))
        
        return recommendations
    
    def get_driver_development_insights(self, driver_name: str) -> List[Tuple[str, str]]:
        """
        Get driver development insights and coaching recommendations.
        
        Args:
            driver_name: Name of driver
            
        Returns:
            List of development insights
        """
        driver = self.get_driver(driver_name)
        if driver is None:
            return []
        
        insights = []
        
        # Thermal management insights
        thermal_consistency = driver.get_driver_summary()['thermal_consistency']
        if thermal_consistency < 0.7:
            insights.append(("DEVELOPMENT", f"Thermal consistency: Focus on smoother inputs"))
        elif thermal_consistency > 0.9:
            insights.append(("DEVELOPMENT", f"Excellent thermal management: Maintain current approach"))
        
        # Experience-based development
        if driver.experience == DriverExperience.ROOKIE:
            insights.append(("COACHING", f"Rookie development: Focus on tire awareness and pressure management"))
        elif driver.experience == DriverExperience.EXPERIENCED:
            insights.append(("COACHING", f"Experienced driver: Work on compound adaptation and racecraft"))
        
        # Style-specific development
        if driver.style == DrivingStyle.AGGRESSIVE:
            insights.append(("COACHING", f"Aggressive style: Learn to manage thermal buildup better"))
        elif driver.style == DrivingStyle.CONSERVATIVE:
            insights.append(("COACHING", f"Conservative style: Push boundaries when safe"))
        
        return insights
    
    def reset_all_sessions(self):
        """Reset all driver sessions."""
        for driver in self.drivers.values():
            driver.reset_session()
    
    def get_driver_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics for all drivers."""
        stats = {}
        
        for name, driver in self.drivers.items():
            summary = driver.get_driver_summary()
            stats[name] = {
                'basic_info': {
                    'name': summary['name'],
                    'style': summary['style'],
                    'experience': summary['experience']
                },
                'performance': {
                    'thermal_consistency': summary['thermal_consistency'],
                    'recommendation_follow_rate': summary['recommendation_follow_rate'],
                    'average_lap_time': summary['average_lap_time']
                },
                'characteristics': {
                    'thermal_aggression': driver.params.thermal_aggression,
                    'thermal_efficiency': driver.params.thermal_efficiency,
                    'tire_awareness': driver.params.tire_awareness,
                    'wet_weather_skill': driver.params.wet_weather_skill
                },
                'session_data': {
                    'session_laps': summary['session_laps'],
                    'total_laps': summary['total_laps']
                }
            }
        
        return stats
