"""
Monte Carlo Strategy Simulator
Simulates race strategies with tire degradation
"""
import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class TireCompound:
    """Tire compound characteristics"""
    name: str
    base_lap_time: float  # seconds
    degradation_rate: float  # seconds per lap
    optimal_stint_length: int  # laps


class MonteCarloStrategySimulator:
    """Monte Carlo simulator for race strategies"""
    
    # Default tire compounds
    COMPOUNDS = {
        'SOFT': TireCompound('SOFT', 90.0, 0.08, 15),
        'MEDIUM': TireCompound('MEDIUM', 90.5, 0.04, 25),
        'HARD': TireCompound('HARD', 91.0, 0.02, 35)
    }
    
    def __init__(self, total_laps: int = 55, pit_stop_time: float = 25.0):
        self.total_laps = total_laps
        self.pit_stop_time = pit_stop_time
        self.safety_car_probability = 0.05  # 5% per lap
    
    def simulate_stint(self, compound: TireCompound, stint_length: int, start_lap: int) -> Dict:
        """
        Simulate a single tire stint
        
        Returns:
            Dictionary with stint_time, lap_times, safety_car_laps
        """
        stint_time = 0
        lap_times = []
        safety_car_laps = []
        
        for lap in range(stint_length):
            tire_age = lap + 1
            
            # Calculate lap time with degradation
            degradation = compound.degradation_rate * (tire_age ** 1.3)
            lap_time = compound.base_lap_time + degradation
            
            # Add random variance
            lap_time += random.gauss(0, 0.15)
            
            # Check for safety car
            if random.random() < self.safety_car_probability:
                safety_car_laps.append(start_lap + lap)
                lap_time *= 1.3  # Slower under safety car
            
            lap_times.append(lap_time)
            stint_time += lap_time
        
        return {
            'stint_time': stint_time,
            'lap_times': lap_times,
            'safety_car_laps': safety_car_laps
        }
    
    def simulate_strategy(self, strategy: List[Tuple[str, int]], n_simulations: int = 1000) -> Dict:
        """
        Simulate a complete race strategy
        
        Args:
            strategy: List of (compound_name, stint_length) tuples
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Simulating strategy with {n_simulations} iterations...")
        
        race_times = []
        
        for _ in range(n_simulations):
            total_time = 0
            current_lap = 0
            
            for stint_idx, (compound_name, stint_length) in enumerate(strategy):
                # Get compound
                compound = self.COMPOUNDS.get(compound_name, self.COMPOUNDS['MEDIUM'])
                
                # Simulate stint
                stint_result = self.simulate_stint(compound, stint_length, current_lap)
                total_time += stint_result['stint_time']
                current_lap += stint_length
                
                # Add pit stop time (except after final stint)
                if stint_idx < len(strategy) - 1:
                    pit_time = random.lognormvariate(np.log(self.pit_stop_time), 0.1)
                    total_time += pit_time
            
            race_times.append(total_time)
        
        # Calculate statistics
        race_times = np.array(race_times)
        
        result = {
            'mean_time': float(np.mean(race_times)),
            'std_dev': float(np.std(race_times)),
            'best_time': float(np.min(race_times)),
            'worst_time': float(np.max(race_times)),
            'median_time': float(np.median(race_times)),
            'percentile_25': float(np.percentile(race_times, 25)),
            'percentile_75': float(np.percentile(race_times, 75))
        }
        
        logger.info(f"Simulation complete - Mean time: {result['mean_time']:.2f}s")
        
        return result
    
    def optimize_strategy(self, candidate_strategies: List[List[Tuple[str, int]]], 
                         n_simulations: int = 500) -> List[Dict]:
        """
        Compare multiple strategies and rank them
        
        Returns:
            List of strategy results ranked by mean time
        """
        logger.info(f"Optimizing {len(candidate_strategies)} strategies...")
        
        results = []
        
        for idx, strategy in enumerate(candidate_strategies):
            stats = self.simulate_strategy(strategy, n_simulations)
            
            # Calculate risk score
            risk_score = stats['std_dev'] / stats['mean_time']
            
            results.append({
                'strategy_id': idx + 1,
                'strategy': strategy,
                'mean_time': stats['mean_time'],
                'std_dev': stats['std_dev'],
                'best_time': stats['best_time'],
                'worst_time': stats['worst_time'],
                'risk_score': risk_score
            })
        
        # Sort by mean time
        results.sort(key=lambda x: x['mean_time'])
        
        # Add rank
        for idx, result in enumerate(results):
            result['rank'] = idx + 1
        
        logger.info("Strategy optimization complete")
        
        return results