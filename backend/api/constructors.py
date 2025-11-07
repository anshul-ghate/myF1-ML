"""
Constructors API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from services.supabase_client import get_supabase_client

router = APIRouter()


@router.get("/")
async def get_constructors():
    """Get all constructors"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('constructors')\
            .select('*')\
            .order('name')\
            .execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{constructor_id}")
async def get_constructor(constructor_id: str):
    """Get constructor by ID"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('constructors')\
            .select('*')\
            .eq('id', constructor_id)\
            .single()\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Constructor not found")
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/standings/{season}")
async def get_constructor_standings(season: int):
    """Get constructor championship standings"""
    supabase = get_supabase_client()
    
    try:
        # Get all results for the season
        response = supabase.table('race_results')\
            .select('*, constructors(*), races!inner(*)')\
            .eq('races.season_year', season)\
            .execute()
        
        # Calculate standings
        standings = {}
        for result in response.data:
            constructor_id = result['constructor_id']
            if constructor_id not in standings:
                standings[constructor_id] = {
                    'constructor_id': constructor_id,
                    'constructor_name': result['constructors']['name'],
                    'points': 0,
                    'wins': 0
                }
            
            standings[constructor_id]['points'] += result.get('points', 0)
            if result.get('position') == 1:
                standings[constructor_id]['wins'] += 1
        
        # Sort by points
        standings_list = sorted(
            standings.values(),
            key=lambda x: x['points'],
            reverse=True
        )
        
        # Add positions
        for idx, standing in enumerate(standings_list):
            standing['position'] = idx + 1
        
        return standings_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))