from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import random
import json
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor
import asyncio

class OptimizationAlgorithm(Enum):
    """Types of optimization algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_DESCENT = "gradient_descent"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class StrategyComponent(Enum):
    """Components of race strategy."""
    PIT_STOPS = "pit_stops"
    COMPOUND_SEQUENCE = "compound_sequence"
    PIT_WINDOWS = "pit_windows"
    FUEL_STRATEGY = "fuel_strategy"
    TIRE_PRESSURE = "tire_pressure"
    DRIVING_STYLE = "driving_style"
    WEATHER_ADAPTATION = "weather_adaptation"

class FitnessMetric(Enum):
    """Fitness metrics for strategy optimization."""
    RACE_TIME = "race_time"
    POSITION_GAIN = "position_gain"
    TIRE_LIFE = "tire_life"
    FUEL_EFFICIENCY = "fuel_efficiency"
    RISK_SCORE = "risk_score"
    CONSISTENCY = "consistency"
    ADAPTABILITY = "adaptability"

@dataclass
class StrategyOptimizationParams:
    """Parameters for strategy optimization."""
    # Algorithm parameters
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # Strategy parameters
    max_pit_stops: int = 3
    min_pit_stops: int = 1
    available_compounds: List[str] = None
    race_duration_laps: int = 58
    
    # Optimization parameters
    fitness_weights: Dict[str, float] = None
    constraint_penalty: float = 1000.0
    convergence_threshold: float = 0.001
    max_iterations: int = 1000
    
    # Performance parameters
    parallel_evaluation: bool = True
    max_workers: int = 4
    cache_evaluations: bool = True
    
    def __post_init__(self):
        if self.available_compounds is None:
            self.available_compounds = ['soft', 'medium', 'hard', 'intermediate', 'wet']
        
        if self.fitness_weights is None:
            self.fitness_weights = {
                'race_time': 0.4,
                'position_gain': 0.3,
                'tire_life': 0.1,
                'fuel_efficiency': 0.1,
                'risk_score': 0.1
            }

class StrategyChromosome:
    """Represents a strategy chromosome for genetic algorithm."""
    
    def __init__(self, params: StrategyOptimizationParams):
        self.p = params
        self.genes = {}
        self.fitness = 0.0
        self.fitness_breakdown = {}
        self.constraints_violated = []
        
        # Initialize genes
        self._initialize_genes()
    
    def _initialize_genes(self):
        """Initialize chromosome genes."""
        # Pit stops
        self.genes['pit_stops'] = random.randint(self.p.min_pit_stops, self.p.max_pit_stops)
        
        # Compound sequence
        num_compounds = self.genes['pit_stops'] + 1
        self.genes['compound_sequence'] = random.choices(
            self.p.available_compounds, k=num_compounds
        )
        
        # Pit windows (lap numbers)
        self.genes['pit_windows'] = self._generate_pit_windows()
        
        # Fuel strategy (percentage per stint)
        self.genes['fuel_strategy'] = self._generate_fuel_strategy()
        
        # Tire pressure (bar)
        self.genes['tire_pressure'] = random.uniform(1.3, 1.7)
        
        # Driving style (aggression level)
        self.genes['driving_style'] = random.uniform(0.3, 0.9)
        
        # Weather adaptation (sensitivity)
        self.genes['weather_adaptation'] = random.uniform(0.1, 0.8)
    
    def _generate_pit_windows(self) -> List[int]:
        """Generate pit window lap numbers."""
        windows = []
        total_laps = self.p.race_duration_laps
        
        if self.genes['pit_stops'] == 1:
            windows.append(random.randint(15, total_laps - 10))
        elif self.genes['pit_stops'] == 2:
            windows.append(random.randint(10, total_laps // 2))
            windows.append(random.randint(total_laps // 2, total_laps - 5))
        elif self.genes['pit_stops'] == 3:
            windows.append(random.randint(8, total_laps // 3))
            windows.append(random.randint(total_laps // 3, 2 * total_laps // 3))
            windows.append(random.randint(2 * total_laps // 3, total_laps - 3))
        
        return sorted(windows)
    
    def _generate_fuel_strategy(self) -> List[float]:
        """Generate fuel strategy percentages."""
        num_stints = self.genes['pit_stops'] + 1
        fuel_percentages = []
        
        # Generate random percentages that sum to 100
        for _ in range(num_stints - 1):
            fuel_percentages.append(random.uniform(20, 40))
        
        # Last stint gets remaining fuel
        remaining = 100 - sum(fuel_percentages)
        fuel_percentages.append(max(20, remaining))
        
        return fuel_percentages
    
    def mutate(self):
        """Apply mutation to chromosome."""
        for gene_name in self.genes:
            if random.random() < self.p.mutation_rate:
                if gene_name == 'pit_stops':
                    self.genes[gene_name] = random.randint(self.p.min_pit_stops, self.p.max_pit_stops)
                    # Update dependent genes
                    self._update_dependent_genes()
                elif gene_name == 'compound_sequence':
                    self.genes[gene_name] = random.choices(
                        self.p.available_compounds, k=len(self.genes[gene_name])
                    )
                elif gene_name == 'pit_windows':
                    self.genes[gene_name] = self._generate_pit_windows()
                elif gene_name == 'fuel_strategy':
                    self.genes[gene_name] = self._generate_fuel_strategy()
                elif gene_name == 'tire_pressure':
                    self.genes[gene_name] = random.uniform(1.3, 1.7)
                elif gene_name == 'driving_style':
                    self.genes[gene_name] = random.uniform(0.3, 0.9)
                elif gene_name == 'weather_adaptation':
                    self.genes[gene_name] = random.uniform(0.1, 0.8)
    
    def _update_dependent_genes(self):
        """Update genes that depend on pit stops."""
        num_compounds = self.genes['pit_stops'] + 1
        self.genes['compound_sequence'] = random.choices(
            self.p.available_compounds, k=num_compounds
        )
        self.genes['pit_windows'] = self._generate_pit_windows()
        self.genes['fuel_strategy'] = self._generate_fuel_strategy()
    
    def crossover(self, other: 'StrategyChromosome') -> Tuple['StrategyChromosome', 'StrategyChromosome']:
        """Perform crossover with another chromosome."""
        child1 = StrategyChromosome(self.p)
        child2 = StrategyChromosome(self.p)
        
        for gene_name in self.genes:
            if random.random() < 0.5:
                child1.genes[gene_name] = self.genes[gene_name].copy() if isinstance(self.genes[gene_name], list) else self.genes[gene_name]
                child2.genes[gene_name] = other.genes[gene_name].copy() if isinstance(other.genes[gene_name], list) else other.genes[gene_name]
            else:
                child1.genes[gene_name] = other.genes[gene_name].copy() if isinstance(other.genes[gene_name], list) else other.genes[gene_name]
                child2.genes[gene_name] = self.genes[gene_name].copy() if isinstance(self.genes[gene_name], list) else self.genes[gene_name]
        
        return child1, child2
    
    def to_strategy_dict(self) -> Dict[str, Any]:
        """Convert chromosome to strategy dictionary."""
        return {
            'pit_stops': self.genes['pit_stops'],
            'compound_sequence': self.genes['compound_sequence'],
            'pit_windows': self.genes['pit_windows'],
            'fuel_strategy': self.genes['fuel_strategy'],
            'tire_pressure': self.genes['tire_pressure'],
            'driving_style': self.genes['driving_style'],
            'weather_adaptation': self.genes['weather_adaptation']
        }

class StrategyOptimizer:
    """
    Advanced strategy optimization system using genetic algorithms and other optimization methods.
    
    Features:
    - Genetic algorithm for strategy optimization
    - Multiple optimization algorithms (PSO, SA, Bayesian)
    - Multi-objective optimization with weighted fitness
    - Constraint handling and penalty functions
    - Parallel evaluation for performance
    - Strategy validation and feasibility checking
    - Real-time optimization during races
    - Historical strategy analysis and learning
    """
    
    def __init__(self, params: StrategyOptimizationParams = None):
        self.p = params or StrategyOptimizationParams()
        
        # Optimization state
        self.population = []
        self.generation = 0
        self.best_chromosome = None
        self.best_fitness_history = []
        self.population_fitness_history = []
        
        # Evaluation cache
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Strategy validation
        self.constraint_validators = {}
        self.fitness_evaluators = {}
        
        # Performance tracking
        self.optimization_stats = {
            'total_evaluations': 0,
            'cache_hit_rate': 0.0,
            'convergence_generation': 0,
            'optimization_time': 0.0
        }
        
        # Initialize validators and evaluators
        self._initialize_validators()
        self._initialize_evaluators()
        
        # Thread pool for parallel evaluation
        if self.p.parallel_evaluation:
            self.executor = ThreadPoolExecutor(max_workers=self.p.max_workers)
        else:
            self.executor = None
    
    def _initialize_validators(self):
        """Initialize constraint validators."""
        self.constraint_validators = {
            'pit_window_validity': self._validate_pit_windows,
            'compound_sequence_validity': self._validate_compound_sequence,
            'fuel_strategy_validity': self._validate_fuel_strategy,
            'tire_pressure_validity': self._validate_tire_pressure,
            'driving_style_validity': self._validate_driving_style
        }
    
    def _initialize_evaluators(self):
        """Initialize fitness evaluators."""
        self.fitness_evaluators = {
            'race_time': self._evaluate_race_time,
            'position_gain': self._evaluate_position_gain,
            'tire_life': self._evaluate_tire_life,
            'fuel_efficiency': self._evaluate_fuel_efficiency,
            'risk_score': self._evaluate_risk_score,
            'consistency': self._evaluate_consistency,
            'adaptability': self._evaluate_adaptability
        }
    
    def optimize_strategy(self, race_context: Dict[str, Any], 
                         simulation_engine=None) -> Dict[str, Any]:
        """
        Optimize race strategy using genetic algorithm.
        
        Args:
            race_context: Race context information
            simulation_engine: Simulation engine for strategy evaluation
            
        Returns:
            Dictionary with optimized strategy and results
        """
        start_time = datetime.now()
        
        # Initialize population
        self._initialize_population()
        
        # Run optimization
        if self.p.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
            results = self._run_genetic_algorithm(race_context, simulation_engine)
        elif self.p.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
            results = self._run_particle_swarm_optimization(race_context, simulation_engine)
        elif self.p.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
            results = self._run_simulated_annealing(race_context, simulation_engine)
        elif self.p.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
            results = self._run_random_search(race_context, simulation_engine)
        else:
            raise ValueError(f"Unsupported optimization algorithm: {self.p.algorithm}")
        
        # Update statistics
        end_time = datetime.now()
        self.optimization_stats['optimization_time'] = (end_time - start_time).total_seconds()
        self.optimization_stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        
        return results
    
    def _initialize_population(self):
        """Initialize population of strategy chromosomes."""
        self.population = []
        
        for _ in range(self.p.population_size):
            chromosome = StrategyChromosome(self.p)
            self.population.append(chromosome)
        
        self.generation = 0
        self.best_fitness_history = []
        self.population_fitness_history = []
    
    def _run_genetic_algorithm(self, race_context: Dict[str, Any], 
                              simulation_engine=None) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        for generation in range(self.p.generations):
            self.generation = generation
            
            # Evaluate population
            self._evaluate_population(race_context, simulation_engine)
            
            # Track best fitness
            best_fitness = max(chromosome.fitness for chromosome in self.population)
            self.best_fitness_history.append(best_fitness)
            
            # Track population fitness
            avg_fitness = statistics.mean(chromosome.fitness for chromosome in self.population)
            self.population_fitness_history.append(avg_fitness)
            
            # Check convergence
            if self._check_convergence():
                self.optimization_stats['convergence_generation'] = generation
                break
            
            # Select parents and create new generation
            new_population = self._create_new_generation()
            self.population = new_population
        
        # Find best chromosome
        self.best_chromosome = max(self.population, key=lambda c: c.fitness)
        
        return {
            'optimized_strategy': self.best_chromosome.to_strategy_dict(),
            'best_fitness': self.best_chromosome.fitness,
            'fitness_breakdown': self.best_chromosome.fitness_breakdown,
            'generations': self.generation + 1,
            'convergence_generation': self.optimization_stats['convergence_generation'],
            'fitness_history': self.best_fitness_history,
            'population_fitness_history': self.population_fitness_history,
            'optimization_stats': self.optimization_stats
        }
    
    def _evaluate_population(self, race_context: Dict[str, Any], simulation_engine=None):
        """Evaluate fitness of all chromosomes in population."""
        if self.p.parallel_evaluation and self.executor:
            # Parallel evaluation
            futures = []
            for chromosome in self.population:
                future = self.executor.submit(self._evaluate_chromosome, chromosome, race_context, simulation_engine)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                self.population[i].fitness, self.population[i].fitness_breakdown = future.result()
        else:
            # Sequential evaluation
            for chromosome in self.population:
                chromosome.fitness, chromosome.fitness_breakdown = self._evaluate_chromosome(
                    chromosome, race_context, simulation_engine
                )
    
    def _evaluate_chromosome(self, chromosome: StrategyChromosome, 
                           race_context: Dict[str, Any], simulation_engine=None) -> Tuple[float, Dict[str, float]]:
        """Evaluate fitness of a single chromosome."""
        # Check cache
        strategy_key = json.dumps(chromosome.to_strategy_dict(), sort_keys=True)
        if self.p.cache_evaluations and strategy_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[strategy_key]
        
        self.cache_misses += 1
        
        # Validate constraints
        constraints_violated = self._validate_strategy(chromosome)
        if constraints_violated:
            # Apply penalty
            penalty = len(constraints_violated) * self.p.constraint_penalty
            fitness_breakdown = {metric.value: -penalty for metric in FitnessMetric}
            fitness = -penalty
        else:
            # Evaluate fitness
            fitness_breakdown = {}
            for metric_name, weight in self.p.fitness_weights.items():
                if metric_name in self.fitness_evaluators:
                    metric_value = self.fitness_evaluators[metric_name](chromosome, race_context, simulation_engine)
                    fitness_breakdown[metric_name] = metric_value
                else:
                    fitness_breakdown[metric_name] = 0.0
            
            # Calculate weighted fitness
            fitness = sum(weight * fitness_breakdown[metric] for metric, weight in self.p.fitness_weights.items())
        
        # Cache result
        if self.p.cache_evaluations:
            self.evaluation_cache[strategy_key] = (fitness, fitness_breakdown)
        
        self.optimization_stats['total_evaluations'] += 1
        
        return fitness, fitness_breakdown
    
    def _validate_strategy(self, chromosome: StrategyChromosome) -> List[str]:
        """Validate strategy constraints."""
        violations = []
        
        for validator_name, validator_func in self.constraint_validators.items():
            if not validator_func(chromosome):
                violations.append(validator_name)
        
        return violations
    
    def _validate_pit_windows(self, chromosome: StrategyChromosome) -> bool:
        """Validate pit window constraints."""
        pit_windows = chromosome.genes['pit_windows']
        
        # Check if pit windows are within race duration
        for window in pit_windows:
            if window < 1 or window > self.p.race_duration_laps:
                return False
        
        # Check if pit windows are properly spaced
        for i in range(len(pit_windows) - 1):
            if pit_windows[i+1] - pit_windows[i] < 5:  # Minimum 5 laps between pits
                return False
        
        return True
    
    def _validate_compound_sequence(self, chromosome: StrategyChromosome) -> bool:
        """Validate compound sequence constraints."""
        compounds = chromosome.genes['compound_sequence']
        
        # Check if all compounds are available
        for compound in compounds:
            if compound not in self.p.available_compounds:
                return False
        
        # Check compound sequence length matches pit stops
        expected_length = chromosome.genes['pit_stops'] + 1
        if len(compounds) != expected_length:
            return False
        
        return True
    
    def _validate_fuel_strategy(self, chromosome: StrategyChromosome) -> bool:
        """Validate fuel strategy constraints."""
        fuel_strategy = chromosome.genes['fuel_strategy']
        
        # Check if fuel percentages sum to 100
        if abs(sum(fuel_strategy) - 100.0) > 0.1:
            return False
        
        # Check if all percentages are positive
        for percentage in fuel_strategy:
            if percentage <= 0:
                return False
        
        return True
    
    def _validate_tire_pressure(self, chromosome: StrategyChromosome) -> bool:
        """Validate tire pressure constraints."""
        pressure = chromosome.genes['tire_pressure']
        return 1.0 <= pressure <= 2.0
    
    def _validate_driving_style(self, chromosome: StrategyChromosome) -> bool:
        """Validate driving style constraints."""
        style = chromosome.genes['driving_style']
        return 0.0 <= style <= 1.0
    
    def _evaluate_race_time(self, chromosome: StrategyChromosome, 
                          race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate race time fitness."""
        # Simplified race time calculation
        base_lap_time = 90.0  # Base lap time in seconds
        
        # Adjust for compound performance
        compound_multipliers = {
            'soft': 0.95,      # Faster
            'medium': 1.0,     # Baseline
            'hard': 1.05,      # Slower
            'intermediate': 1.1,  # Wet conditions
            'wet': 1.15       # Heavy wet
        }
        
        total_time = 0.0
        compounds = chromosome.genes['compound_sequence']
        pit_windows = chromosome.genes['pit_windows']
        
        # Calculate time for each stint
        current_lap = 0
        for i, compound in enumerate(compounds):
            if i < len(pit_windows):
                stint_laps = pit_windows[i] - current_lap
            else:
                stint_laps = self.p.race_duration_laps - current_lap
            
            stint_time = stint_laps * base_lap_time * compound_multipliers.get(compound, 1.0)
            total_time += stint_time
            
            # Add pit stop time
            if i < len(pit_windows):
                total_time += 20.0  # Pit stop time
            
            current_lap = pit_windows[i] if i < len(pit_windows) else self.p.race_duration_laps
        
        # Convert to fitness (lower time = higher fitness)
        max_race_time = self.p.race_duration_laps * base_lap_time * 1.2  # 20% buffer
        fitness = 1.0 - (total_time / max_race_time)
        
        return max(0.0, fitness)
    
    def _evaluate_position_gain(self, chromosome: StrategyChromosome, 
                              race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate position gain fitness."""
        # Simplified position gain calculation
        pit_stops = chromosome.genes['pit_stops']
        driving_style = chromosome.genes['driving_style']
        
        # More pit stops can lead to position gains through fresher tires
        pit_gain = pit_stops * 0.1
        
        # Aggressive driving can lead to position gains
        style_gain = driving_style * 0.2
        
        # But aggressive driving with many pit stops can be risky
        risk_penalty = pit_stops * driving_style * 0.05
        
        fitness = pit_gain + style_gain - risk_penalty
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_tire_life(self, chromosome: StrategyChromosome, 
                          race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate tire life fitness."""
        compounds = chromosome.genes['compound_sequence']
        
        # Harder compounds last longer
        compound_life_multipliers = {
            'soft': 0.7,      # Short life
            'medium': 1.0,    # Baseline
            'hard': 1.3,      # Long life
            'intermediate': 0.8,  # Wet conditions
            'wet': 0.6       # Heavy wet
        }
        
        total_life = 0.0
        for compound in compounds:
            total_life += compound_life_multipliers.get(compound, 1.0)
        
        # Normalize by number of compounds
        avg_life = total_life / len(compounds)
        
        return min(1.0, avg_life)
    
    def _evaluate_fuel_efficiency(self, chromosome: StrategyChromosome, 
                                race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate fuel efficiency fitness."""
        fuel_strategy = chromosome.genes['fuel_strategy']
        driving_style = chromosome.genes['driving_style']
        
        # More balanced fuel strategy is more efficient
        fuel_variance = statistics.variance(fuel_strategy) if len(fuel_strategy) > 1 else 0.0
        efficiency = 1.0 - (fuel_variance / 100.0)  # Normalize variance
        
        # Aggressive driving reduces fuel efficiency
        style_penalty = driving_style * 0.2
        
        fitness = efficiency - style_penalty
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_risk_score(self, chromosome: StrategyChromosome, 
                           race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate risk score fitness."""
        pit_stops = chromosome.genes['pit_stops']
        driving_style = chromosome.genes['driving_style']
        tire_pressure = chromosome.genes['tire_pressure']
        
        # More pit stops = higher risk
        pit_risk = pit_stops * 0.1
        
        # Aggressive driving = higher risk
        style_risk = driving_style * 0.2
        
        # Extreme tire pressure = higher risk
        pressure_risk = abs(tire_pressure - 1.5) * 0.1
        
        total_risk = pit_risk + style_risk + pressure_risk
        
        # Convert risk to fitness (lower risk = higher fitness)
        fitness = 1.0 - min(1.0, total_risk)
        
        return max(0.0, fitness)
    
    def _evaluate_consistency(self, chromosome: StrategyChromosome, 
                            race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate consistency fitness."""
        compounds = chromosome.genes['compound_sequence']
        fuel_strategy = chromosome.genes['fuel_strategy']
        
        # Consistent compound usage
        compound_consistency = 1.0 - len(set(compounds)) / len(compounds)
        
        # Consistent fuel strategy
        fuel_consistency = 1.0 - (max(fuel_strategy) - min(fuel_strategy)) / 100.0
        
        fitness = (compound_consistency + fuel_consistency) / 2.0
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_adaptability(self, chromosome: StrategyChromosome, 
                             race_context: Dict[str, Any], simulation_engine=None) -> float:
        """Evaluate adaptability fitness."""
        weather_adaptation = chromosome.genes['weather_adaptation']
        pit_stops = chromosome.genes['pit_stops']
        
        # More pit stops allow for better adaptation
        pit_adaptability = min(1.0, pit_stops / 3.0)
        
        # Weather adaptation capability
        weather_adaptability = weather_adaptation
        
        fitness = (pit_adaptability + weather_adaptability) / 2.0
        
        return max(0.0, min(1.0, fitness))
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.best_fitness_history) < 10:
            return False
        
        # Check if fitness has not improved significantly in last 10 generations
        recent_fitness = self.best_fitness_history[-10:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < self.p.convergence_threshold
    
    def _create_new_generation(self) -> List[StrategyChromosome]:
        """Create new generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: keep best chromosomes
        sorted_population = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        for i in range(self.p.elite_size):
            new_population.append(sorted_population[i])
        
        # Create remaining population through crossover and mutation
        while len(new_population) < self.p.population_size:
            # Select parents
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.p.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1.mutate()
            child2.mutate()
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.p.population_size]
    
    def _select_parent(self) -> StrategyChromosome:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda c: c.fitness)
    
    def _run_particle_swarm_optimization(self, race_context: Dict[str, Any], 
                                       simulation_engine=None) -> Dict[str, Any]:
        """Run particle swarm optimization."""
        # Simplified PSO implementation
        # This would be a more complex implementation in practice
        
        best_strategy = None
        best_fitness = float('-inf')
        
        for iteration in range(self.p.max_iterations):
            # Generate random strategy
            chromosome = StrategyChromosome(self.p)
            fitness, _ = self._evaluate_chromosome(chromosome, race_context, simulation_engine)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = chromosome
        
        return {
            'optimized_strategy': best_strategy.to_strategy_dict() if best_strategy else {},
            'best_fitness': best_fitness,
            'iterations': self.p.max_iterations,
            'algorithm': 'particle_swarm'
        }
    
    def _run_simulated_annealing(self, race_context: Dict[str, Any], 
                               simulation_engine=None) -> Dict[str, Any]:
        """Run simulated annealing optimization."""
        # Simplified SA implementation
        current = StrategyChromosome(self.p)
        current.fitness, _ = self._evaluate_chromosome(current, race_context, simulation_engine)
        
        best = current
        temperature = 1.0
        
        for iteration in range(self.p.max_iterations):
            # Generate neighbor
            neighbor = StrategyChromosome(self.p)
            neighbor.fitness, _ = self._evaluate_chromosome(neighbor, race_context, simulation_engine)
            
            # Accept or reject
            if neighbor.fitness > current.fitness or random.random() < np.exp((neighbor.fitness - current.fitness) / temperature):
                current = neighbor
            
            # Update best
            if current.fitness > best.fitness:
                best = current
            
            # Cool down
            temperature *= 0.99
        
        return {
            'optimized_strategy': best.to_strategy_dict(),
            'best_fitness': best.fitness,
            'iterations': self.p.max_iterations,
            'algorithm': 'simulated_annealing'
        }
    
    def _run_random_search(self, race_context: Dict[str, Any], 
                         simulation_engine=None) -> Dict[str, Any]:
        """Run random search optimization."""
        best_strategy = None
        best_fitness = float('-inf')
        
        for iteration in range(self.p.max_iterations):
            chromosome = StrategyChromosome(self.p)
            fitness, _ = self._evaluate_chromosome(chromosome, race_context, simulation_engine)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = chromosome
        
        return {
            'optimized_strategy': best_strategy.to_strategy_dict() if best_strategy else {},
            'best_fitness': best_fitness,
            'iterations': self.p.max_iterations,
            'algorithm': 'random_search'
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        return {
            'optimization_stats': self.optimization_stats,
            'algorithm': self.p.algorithm.value,
            'population_size': self.p.population_size,
            'generations': self.p.generations,
            'best_fitness_history': self.best_fitness_history,
            'population_fitness_history': self.population_fitness_history,
            'cache_stats': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.optimization_stats['cache_hit_rate']
            },
            'best_strategy': self.best_chromosome.to_strategy_dict() if self.best_chromosome else None
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
