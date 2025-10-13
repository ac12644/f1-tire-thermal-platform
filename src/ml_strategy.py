from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pickle
import json
from collections import deque
import random

class StrategyAction(Enum):
    """Strategy actions for reinforcement learning."""
    MAINTAIN_PACE = "maintain_pace"
    INCREASE_PACE = "increase_pace"
    DECREASE_PACE = "decrease_pace"
    PIT_STOP = "pit_stop"
    ADJUST_PRESSURE = "adjust_pressure"
    CHANGE_COMPOUND = "change_compound"
    ADJUST_SETUP = "adjust_setup"
    CONSERVE_TIRES = "conserve_tires"
    PUSH_HARD = "push_hard"

class RacePhase(Enum):
    """Race phases for strategy optimization."""
    START = "start"
    EARLY_RACE = "early_race"
    MID_RACE = "mid_race"
    LATE_RACE = "late_race"
    FINAL_LAPS = "final_laps"

@dataclass
class MLStrategyParams:
    """Parameters for ML strategy optimization."""
    # Learning parameters
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    
    # Experience replay
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    
    # Reward shaping
    temperature_reward_weight: float = 0.3
    wear_reward_weight: float = 0.2
    position_reward_weight: float = 0.3
    strategy_reward_weight: float = 0.2
    
    # State representation
    state_dimensions: int = 20
    action_dimensions: int = len(StrategyAction)
    
    # Training parameters
    training_frequency: int = 10
    model_save_frequency: int = 1000

class MLStrategyOptimizer:
    """
    Machine Learning-based strategy optimization using reinforcement learning.
    
    Features:
    - Q-Learning for strategy optimization
    - Experience replay for stable learning
    - Reward shaping for multi-objective optimization
    - Adaptive exploration-exploitation balance
    - Real-time strategy recommendations
    - Historical pattern learning
    """
    
    def __init__(self, params: MLStrategyParams = None):
        self.p = params or MLStrategyParams()
        
        # Q-Learning components
        self.q_table = {}
        self.target_q_table = {}
        self.experience_memory = deque(maxlen=self.p.memory_size)
        
        # Learning state
        self.current_exploration_rate = self.p.exploration_rate
        self.training_step = 0
        self.last_action = None
        self.last_state = None
        
        # Performance tracking
        self.reward_history = []
        self.action_history = []
        self.strategy_performance = {}
        
        # Pattern recognition
        self.successful_patterns = {}
        self.failed_patterns = {}
        
    def get_state_representation(self, thermal_state: np.ndarray, wear_summary: Dict,
                               weather_summary: Dict, race_context: Dict) -> str:
        """
        Convert current state to string representation for Q-table.
        
        Args:
            thermal_state: Current thermal state [Tt, Tc, Tr]
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            race_context: Race context (lap, position, etc.)
            
        Returns:
            String representation of state
        """
        Tt, Tc, Tr = thermal_state
        
        # Discretize continuous values for state representation
        temp_bucket = int(Tt // 5)  # 5°C buckets
        wear_bucket = int(np.mean([wear_summary.get(corner, {}).get('wear_level', 0.0) 
                                  for corner in ['FL', 'FR', 'RL', 'RR']]) * 10)
        weather_bucket = int(weather_summary.get('rain_probability', 0.0) * 10)
        lap_bucket = int(race_context.get('current_lap', 0) // 10)  # 10-lap buckets
        position_bucket = int(race_context.get('position', 1) // 2)  # 2-position buckets
        
        # Create state string
        state_str = f"T{temp_bucket}_W{wear_bucket}_WX{weather_bucket}_L{lap_bucket}_P{position_bucket}"
        
        return state_str
    
    def select_action(self, state: str, available_actions: List[StrategyAction] = None) -> StrategyAction:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            available_actions: List of available actions (None = all actions)
            
        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = list(StrategyAction)
        
        # Initialize Q-values for state if not exists
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        
        # Epsilon-greedy action selection
        if random.random() < self.current_exploration_rate:
            # Exploration: random action
            action = random.choice(available_actions)
        else:
            # Exploitation: best action
            q_values = {action: self.q_table[state].get(action, 0.0) 
                       for action in available_actions}
            action = max(q_values, key=q_values.get)
        
        return action
    
    def calculate_reward(self, thermal_state: np.ndarray, wear_summary: Dict,
                        weather_summary: Dict, race_context: Dict, action: StrategyAction) -> float:
        """
        Calculate reward for strategy action.
        
        Args:
            thermal_state: Current thermal state
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            race_context: Race context
            action: Action taken
            
        Returns:
            Reward value
        """
        Tt, Tc, Tr = thermal_state
        
        # Temperature reward (optimal range is 90-100°C)
        optimal_temp = 95.0
        temp_deviation = abs(Tt - optimal_temp)
        temp_reward = max(0, 1.0 - temp_deviation / 50.0)  # Normalize to 0-1
        
        # Wear reward (lower wear is better)
        avg_wear = np.mean([wear_summary.get(corner, {}).get('wear_level', 0.0) 
                           for corner in ['FL', 'FR', 'RL', 'RR']])
        wear_reward = max(0, 1.0 - avg_wear)
        
        # Position reward (higher position is better)
        position = race_context.get('position', 1)
        total_cars = race_context.get('total_cars', 20)
        position_reward = (total_cars - position + 1) / total_cars
        
        # Strategy-specific rewards
        strategy_reward = self._calculate_strategy_specific_reward(action, thermal_state, 
                                                                 wear_summary, race_context)
        
        # Weighted reward
        total_reward = (self.p.temperature_reward_weight * temp_reward +
                       self.p.wear_reward_weight * wear_reward +
                       self.p.position_reward_weight * position_reward +
                       self.p.strategy_reward_weight * strategy_reward)
        
        return total_reward
    
    def _calculate_strategy_specific_reward(self, action: StrategyAction, thermal_state: np.ndarray,
                                          wear_summary: Dict, race_context: Dict) -> float:
        """Calculate strategy-specific reward."""
        Tt, Tc, Tr = thermal_state
        avg_wear = np.mean([wear_summary.get(corner, {}).get('wear_level', 0.0) 
                           for corner in ['FL', 'FR', 'RL', 'RR']])
        
        # Action-specific rewards
        if action == StrategyAction.MAINTAIN_PACE:
            return 0.5  # Neutral reward
        elif action == StrategyAction.INCREASE_PACE:
            if Tt < 90:  # Good to push when temps are low
                return 0.8
            elif Tt > 110:  # Bad to push when temps are high
                return 0.2
            else:
                return 0.5
        elif action == StrategyAction.DECREASE_PACE:
            if Tt > 110:  # Good to conserve when temps are high
                return 0.8
            elif Tt < 90:  # Bad to conserve when temps are low
                return 0.2
            else:
                return 0.5
        elif action == StrategyAction.PIT_STOP:
            if avg_wear > 0.7:  # Good to pit when wear is high
                return 0.9
            elif avg_wear < 0.3:  # Bad to pit when wear is low
                return 0.1
            else:
                return 0.5
        elif action == StrategyAction.CONSERVE_TIRES:
            if avg_wear > 0.5:  # Good to conserve when wear is moderate-high
                return 0.7
            else:
                return 0.3
        elif action == StrategyAction.PUSH_HARD:
            if Tt < 95 and avg_wear < 0.4:  # Good to push when conditions allow
                return 0.8
            else:
                return 0.2
        
        return 0.5  # Default neutral reward
    
    def update_q_value(self, state: str, action: StrategyAction, reward: float, 
                     next_state: str, done: bool = False):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Initialize Q-values if not exist
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in StrategyAction}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in StrategyAction}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target_q = reward + self.p.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.p.learning_rate * (target_q - current_q)
        
        # Store experience for replay
        self.experience_memory.append((state, action, reward, next_state, done))
        
        # Update exploration rate
        self.current_exploration_rate = max(self.p.min_exploration_rate,
                                          self.current_exploration_rate * self.p.exploration_decay)
    
    def train_model(self):
        """Train the model using experience replay."""
        if len(self.experience_memory) < self.p.batch_size:
            return
        
        # Sample batch from experience memory
        batch = random.sample(self.experience_memory, self.p.batch_size)
        
        # Update Q-values for batch
        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.p.target_update_frequency == 0:
            self.target_q_table = self.q_table.copy()
    
    def get_strategy_recommendations(self, thermal_state: np.ndarray, wear_summary: Dict,
                                   weather_summary: Dict, race_context: Dict) -> List[Tuple[str, str]]:
        """
        Get ML-based strategy recommendations.
        
        Args:
            thermal_state: Current thermal state
            wear_summary: Current wear status
            weather_summary: Current weather conditions
            race_context: Race context
            
        Returns:
            List of strategy recommendations
        """
        recommendations = []
        
        # Get current state
        current_state = self.get_state_representation(thermal_state, wear_summary, 
                                                    weather_summary, race_context)
        
        # Get best action from Q-table
        if current_state in self.q_table:
            q_values = self.q_table[current_state]
            best_action = max(q_values, key=q_values.get)
            best_q_value = q_values[best_action]
            
            # Generate recommendations based on best action
            if best_action == StrategyAction.INCREASE_PACE:
                recommendations.append(("ML_STRATEGY", f"AI recommends increasing pace (Q-value: {best_q_value:.3f})"))
            elif best_action == StrategyAction.DECREASE_PACE:
                recommendations.append(("ML_STRATEGY", f"AI recommends decreasing pace (Q-value: {best_q_value:.3f})"))
            elif best_action == StrategyAction.PIT_STOP:
                recommendations.append(("ML_STRATEGY", f"AI recommends pit stop (Q-value: {best_q_value:.3f})"))
            elif best_action == StrategyAction.CONSERVE_TIRES:
                recommendations.append(("ML_STRATEGY", f"AI recommends conserving tires (Q-value: {best_q_value:.3f})"))
            elif best_action == StrategyAction.PUSH_HARD:
                recommendations.append(("ML_STRATEGY", f"AI recommends pushing hard (Q-value: {best_q_value:.3f})"))
        
        # Add pattern-based recommendations
        pattern_recs = self._get_pattern_based_recommendations(thermal_state, wear_summary, 
                                                             weather_summary, race_context)
        recommendations.extend(pattern_recs)
        
        return recommendations
    
    def _get_pattern_based_recommendations(self, thermal_state: np.ndarray, wear_summary: Dict,
                                         weather_summary: Dict, race_context: Dict) -> List[Tuple[str, str]]:
        """Get recommendations based on learned patterns."""
        recommendations = []
        
        # Analyze current conditions for pattern matching
        Tt, Tc, Tr = thermal_state
        avg_wear = np.mean([wear_summary.get(corner, {}).get('wear_level', 0.0) 
                           for corner in ['FL', 'FR', 'RL', 'RR']])
        
        # Pattern: High temperature + high wear
        if Tt > 105 and avg_wear > 0.6:
            if 'high_temp_high_wear' in self.successful_patterns:
                success_rate = self.successful_patterns['high_temp_high_wear']['success_rate']
                if success_rate > 0.7:
                    recommendations.append(("ML_PATTERN", f"Pattern: High temp+wear → Pit stop (success rate: {success_rate:.1%})"))
        
        # Pattern: Low temperature + low wear
        elif Tt < 90 and avg_wear < 0.3:
            if 'low_temp_low_wear' in self.successful_patterns:
                success_rate = self.successful_patterns['low_temp_low_wear']['success_rate']
                if success_rate > 0.7:
                    recommendations.append(("ML_PATTERN", f"Pattern: Low temp+wear → Push hard (success rate: {success_rate:.1%})"))
        
        # Pattern: Rain conditions
        if weather_summary.get('rain_probability', 0.0) > 0.5:
            if 'rain_conditions' in self.successful_patterns:
                success_rate = self.successful_patterns['rain_conditions']['success_rate']
                if success_rate > 0.7:
                    recommendations.append(("ML_PATTERN", f"Pattern: Rain → Conserve tires (success rate: {success_rate:.1%})"))
        
        return recommendations
    
    def update_pattern_success(self, pattern: str, success: bool):
        """Update pattern success tracking."""
        if pattern not in self.successful_patterns:
            self.successful_patterns[pattern] = {'successes': 0, 'total': 0}
        
        self.successful_patterns[pattern]['total'] += 1
        if success:
            self.successful_patterns[pattern]['successes'] += 1
        
        # Calculate success rate
        self.successful_patterns[pattern]['success_rate'] = (
            self.successful_patterns[pattern]['successes'] / 
            self.successful_patterns[pattern]['total']
        )
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML model summary."""
        return {
            'training_step': self.training_step,
            'exploration_rate': self.current_exploration_rate,
            'q_table_size': len(self.q_table),
            'experience_memory_size': len(self.experience_memory),
            'successful_patterns': len(self.successful_patterns),
            'average_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'total_episodes': len(self.reward_history),
            'learning_progress': self._calculate_learning_progress()
        }
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress based on reward improvement."""
        if len(self.reward_history) < 100:
            return 0.0
        
        recent_rewards = self.reward_history[-100:]
        early_rewards = self.reward_history[:100] if len(self.reward_history) >= 200 else recent_rewards
        
        progress = np.mean(recent_rewards) - np.mean(early_rewards)
        return max(0.0, min(1.0, progress))  # Normalize to 0-1
    
    def save_model(self, filepath: str):
        """Save the ML model to file."""
        model_data = {
            'q_table': self.q_table,
            'target_q_table': self.target_q_table,
            'exploration_rate': self.current_exploration_rate,
            'training_step': self.training_step,
            'successful_patterns': self.successful_patterns,
            'reward_history': self.reward_history,
            'params': self.p
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the ML model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.target_q_table = model_data['target_q_table']
        self.current_exploration_rate = model_data['exploration_rate']
        self.training_step = model_data['training_step']
        self.successful_patterns = model_data['successful_patterns']
        self.reward_history = model_data['reward_history']
        self.p = model_data['params']
    
    def reset_model(self):
        """Reset the ML model for new training."""
        self.q_table = {}
        self.target_q_table = {}
        self.experience_memory.clear()
        self.current_exploration_rate = self.p.exploration_rate
        self.training_step = 0
        self.reward_history = []
        self.action_history = []
        self.successful_patterns = {}
        self.failed_patterns = {}
