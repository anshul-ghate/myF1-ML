"""
Analytics API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from services.supabase_client import get_supabase_client

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard_data(season: Optional[int] = None):
    """Get dashboard analytics data"""
    supabase = get_supabase_client()
    
    try:
        current_season = season or 2024
        
        # Get driver standings
        driver_standings_response = supabase.table('race_results')\
            .select('*, drivers(*), races!inner(*)')\
            .eq('races.season_year', current_season)\
            .execute()
        
        # Calculate driver standings
        driver_standings = {}
        for result in driver_standings_response.data:
            driver_id = result['driver_id']
            if driver_id not in driver_standings:
                driver_standings[driver_id] = {
                    'driver_name': result['drivers']['name'],
                    'points': 0
                }
            driver_standings[driver_id]['points'] += result.get('points', 0)
        
        top_drivers = sorted(
            driver_standings.items(),
            key=lambda x: x[1]['points'],
            reverse=True
        )[:5]
        
        # Get constructor standings
        constructor_standings_response = supabase.table('race_results')\
            .select('*, constructors(*), races!inner(*)')\
            .eq('races.season_year', current_season)\
            .execute()
        
        constructor_standings = {}
        for result in constructor_standings_response.data:
            constructor_id = result['constructor_id']
            if constructor_id not in constructor_standings:
                constructor_standings[constructor_id] = {
                    'constructor_name': result['constructors']['name'],
                    'points': 0
                }
            constructor_standings[constructor_id]['points'] += result.get('points', 0)
        
        top_constructors = sorted(
            constructor_standings.items(),
            key=lambda x: x[1]['points'],
            reverse=True
        )[:3]
        
        return {
            'driver_standings': [
                {'position': idx + 1, **data}
                for idx, (driver_id, data) in enumerate(top_drivers)
            ],
            'constructor_standings': [
                {'position': idx + 1, **data}
                for idx, (constructor_id, data) in enumerate(top_constructors)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))