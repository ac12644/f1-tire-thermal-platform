from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .config import system_config
except ImportError:
    from config import system_config

class DecisionEngine:
    def __init__(self, compound="medium", wear_model=None, driver_profiles=None, active_driver=None):
        tire_config = system_config.tire_config
        self.band = tire_config.optimal_bands.get(compound, tire_config.optimal_bands["medium"])
        self.compound = compound
        self.wear_model = wear_model
        self.driver_profiles = driver_profiles
        self.active_driver = active_driver
        self.grip_thresholds = tire_config.grip_degradation_thresholds

    def forecast_naive(self, hist, steps=10):
        # simple exponential smoothing trend per state dim
        if len(hist) < 5:
            return np.tile(hist[-1], (steps,1))
        hist = np.array(hist[-30:])
        alpha = 0.35
        level = hist[0]
        for x in hist[1:]:
            level = alpha*x + (1-alpha)*level
        return np.vstack([level for _ in range(steps)])

    def actions(self, est_by_corner, wear_summary=None, weather_summary=None):
        """
        Generate recommendations based on thermal state, wear levels, and driver characteristics.
        
        Args:
            est_by_corner: Dict of corner -> [Tt, Tc, Tr] estimates
            wear_summary: Dict of corner -> wear effects (optional)
            weather_summary: Dict with weather conditions (optional)
        """
        lo, hi = self.band
        recs = []
        
        # Get driver-specific temperature bands if available
        driver_bands = self._get_driver_specific_bands()
        
        for corner, x in est_by_corner.items():
            Tt, Tc, Tr = x
            
            # Use driver-specific bands if available
            corner_lo, corner_hi = driver_bands.get(corner, (lo, hi))
            
            # Thermal-based recommendations
            if Tt > corner_hi:
                recs.append((corner, f"Tread {Tt:.1f}C > {corner_hi}C: brake bias +1, gentler entry T7-T9"))
            elif Tt < corner_lo:
                recs.append((corner, f"Tread {Tt:.1f}C < {corner_lo}C: diff entry -1, short throttle bursts, push on exits"))
            if Tc > corner_hi+2:
                recs.append((corner, f"Carcass {Tc:.1f}C high: consider +0.2 psi next stop"))
            
            # Wear-based recommendations
            if wear_summary and corner in wear_summary:
                wear_data = wear_summary[corner]
                wear_level = wear_data.get("wear_level", 0.0)
                grip_factor = wear_data.get("grip_factor", 1.0)
                
                # Check for significant grip degradation
                grip_threshold = self.grip_thresholds.get(self.compound, 0.4)
                
                if wear_level > grip_threshold:
                    grip_loss_pct = (1.0 - grip_factor) * 100
                    recs.append((corner, f"Wear {wear_level:.1%}: {grip_loss_pct:.0f}% grip loss - consider pit stop"))
                
                # Early warning for approaching wear threshold
                elif wear_level > grip_threshold * 0.7:
                    recs.append((corner, f"Wear {wear_level:.1%}: approaching grip degradation - monitor closely"))
                
                # Stiffness degradation affects handling
                stiffness_factor = wear_data.get("stiffness_factor", 1.0)
                if stiffness_factor < 0.9:
                    stiffness_loss = (1.0 - stiffness_factor) * 100
                    recs.append((corner, f"Stiffness -{stiffness_loss:.0f}%: adjust suspension, reduce kerb usage"))
        
        # Add driver-specific recommendations
        if self.driver_profiles and self.active_driver:
            driver_recs = self._get_driver_specific_recommendations(est_by_corner, wear_summary, weather_summary)
            recs.extend(driver_recs)
        
        return recs
    
    def _get_driver_specific_bands(self) -> Dict[str, Tuple[float, float]]:
        """Get driver-specific temperature bands for each corner."""
        if not self.driver_profiles or not self.active_driver:
            return {}
        
        driver = self.driver_profiles.get_driver(self.active_driver)
        if not driver:
            return {}
        
        # Get driver's thermal signature
        thermal_signature = driver.thermal_signature
        thermal_generation = thermal_signature['thermal_generation']
        
        # Adjust bands based on driver characteristics
        # Aggressive drivers can handle higher temps, conservative drivers prefer lower temps
        temp_adjustment = (thermal_generation - 1.0) * 5.0  # ±5°C adjustment
        
        adjusted_lo = self.band[0] + temp_adjustment
        adjusted_hi = self.band[1] + temp_adjustment
        
        # Apply to all corners
        return {corner: (adjusted_lo, adjusted_hi) for corner in ["FL", "FR", "RL", "RR"]}
    
    def _get_driver_specific_recommendations(self, est_by_corner: Dict, wear_summary: Dict, weather_summary: Dict) -> List[Tuple[str, str]]:
        """Get driver-specific recommendations."""
        driver = self.driver_profiles.get_driver(self.active_driver)
        if not driver:
            return []
        
        # Get average thermal state for driver recommendations
        if not est_by_corner:
            return []
        
        avg_thermal_state = np.mean(list(est_by_corner.values()), axis=0)
        
        # Get personalized recommendations
        personalized_recs = driver.get_personalized_recommendations(
            avg_thermal_state, weather_summary or {}, wear_summary or {}
        )
        
        return personalized_recs
    
    def get_wear_recommendations(self, wear_summary: Dict) -> List[Tuple[str, str]]:
        """
        Generate specific wear-based recommendations.
        
        Args:
            wear_summary: Dict of corner -> wear effects
            
        Returns:
            List of (corner, recommendation) tuples
        """
        recommendations = []
        
        for corner, wear_data in wear_summary.items():
            wear_level = wear_data.get("wear_level", 0.0)
            grip_factor = wear_data.get("grip_factor", 1.0)
            
            # Critical wear level (80%+)
            if wear_level >= 0.8:
                recommendations.append((corner, f"CRITICAL: {wear_level:.1%} wear - immediate pit stop required"))
            
            # High wear level (60-80%)
            elif wear_level >= 0.6:
                recommendations.append((corner, f"HIGH: {wear_level:.1%} wear - pit stop recommended within 2-3 laps"))
            
            # Moderate wear level (40-60%)
            elif wear_level >= 0.4:
                recommendations.append((corner, f"MODERATE: {wear_level:.1%} wear - monitor grip levels, plan pit strategy"))
            
            # Check for uneven wear patterns
            if corner in ["FL", "FR"] and wear_level > 0.3:
                # Front tires wearing faster - adjust setup
                recommendations.append((corner, f"Front wear {wear_level:.1%}: consider reducing front camber/toe"))
        
        return recommendations
    
    def predict_pit_window(self, wear_summary: Dict, current_lap: int, total_laps: int) -> Dict[str, float]:
        """
        Predict optimal pit window based on wear levels.
        
        Args:
            wear_summary: Current wear status
            current_lap: Current lap number
            total_laps: Total race laps
            
        Returns:
            Dict with pit window recommendations
        """
        pit_recommendations = {}
        
        for corner, wear_data in wear_summary.items():
            wear_level = wear_data.get("wear_level", 0.0)
            
            # Calculate laps to critical wear (80%)
            if wear_level > 0.01:  # Avoid division by zero
                wear_rate = wear_level / max(1, current_lap)
                laps_to_critical = (0.8 - wear_level) / wear_rate if wear_rate > 0 else total_laps
            else:
                laps_to_critical = total_laps
            
            # Recommend pit window (5 laps before critical)
            recommended_pit_lap = current_lap + laps_to_critical - 5
            recommended_pit_lap = max(current_lap + 1, min(recommended_pit_lap, total_laps))
            
            pit_recommendations[corner] = {
                "recommended_pit_lap": recommended_pit_lap,
                "laps_to_critical": laps_to_critical,
                "urgency": "HIGH" if laps_to_critical < 5 else "MEDIUM" if laps_to_critical < 10 else "LOW"
            }
        
        return pit_recommendations
