"""
Strategy Simulator API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List, Tuple
from pydantic import BaseModel

from models.strategy_simulator import MonteCarloStrategySimulator

router = APIRouter()


class StrategyRequest(BaseModel):
    strategy: List[Tuple[str, int]]  # [(compound, laps), ...]
    n_simulations: int = 1000
    total_laps: int = 55
    pit_stop_time: float = 25.0


class StrategyOptimizationRequest(BaseModel):
    strategies: List[List[Tuple[str, int]]]
    n_simulations: int = 500
    total_laps: int = 55


@router.post("/simulate")
async def simulate_strategy(request: StrategyRequest):
    """Simulate a single race strategy"""
    try:
        simulator = MonteCarloStrategySimulator(
            total_laps=request.total_laps,
            pit_stop_time=request.pit_stop_time
        )
        
        result = simulator.simulate_strategy(
            request.strategy,
            request.n_simulations
        )
        
        return {
            'success': True,
            'result': result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_strategies(request: StrategyOptimizationRequest):
    """Compare and optimize multiple strategies"""
    try:
        simulator = MonteCarloStrategySimulator(total_laps=request.total_laps)
        
        results = simulator.optimize_strategy(
            request.strategies,
            request.n_simulations
        )
        
        return {
            'success': True,
            'results': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compounds")
async def get_tire_compounds():
    """Get available tire compounds and their characteristics"""
    return {
        'compounds': [
            {
                'name': 'SOFT',
                'base_lap_time': 90.0,
                'degradation_rate': 0.08,
                'optimal_stint_length': 15,
                'color': '#FF0000'
            },
            {
                'name': 'MEDIUM',
                'base_lap_time': 90.5,
                'degradation_rate': 0.04,
                'optimal_stint_length': 25,
                'color': '#FFFF00'
            },
            {
                'name': 'HARD',
                'base_lap_time': 91.0,
                'degradation_rate': 0.02,
                'optimal_stint_length': 35,
                'color': '#FFFFFF'
            }
        ]
    }