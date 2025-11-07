"""
Drivers API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from services.supabase_client import get_supabase_client

router = APIRouter()


@router.get("/")
async def get_drivers(season: Optional[int] = None):
    """Get all drivers"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('drivers')\
            .select('*')\
            .order('name')\
            .execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{driver_id}")
async def get_driver(driver_id: str):
    """Get driver by ID"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('drivers')\
            .select('*')\
            .eq('id', driver_id)\
            .single()\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Driver not found")
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{driver_id}/results")
async def get_driver_results(driver_id: str, season: Optional[int] = None):
    """Get race results for a driver"""
    supabase = get_supabase_client()
    
    try:
        query = supabase.table('race_results')\
            .select('*, races(*), constructors(*)')\
            .eq('driver_id', driver_id)
        
        if season:
            query = query.eq('races.season_year', season)
        
        response = query.order('races.date').execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/standings/{season}")
async def get_driver_standings(season: int):
    """Get driver championship standings"""
    supabase = get_supabase_client()
    
    try:
        # Get all results for the season
        response = supabase.table('race_results')\
            .select('*, drivers(*), races!inner(*)')\
            .eq('races.season_year', season)\
            .execute()
        
        # Calculate standings
        standings = {}
        for result in response.data:
            driver_id = result['driver_id']
            if driver_id not in standings:
                standings[driver_id] = {
                    'driver_id': driver_id,
                    'driver_name': result['drivers']['name'],
                    'points': 0,
                    'wins': 0,
                    'podiums': 0
                }
            
            standings[driver_id]['points'] += result.get('points', 0)
            if result.get('position') == 1:
                standings[driver_id]['wins'] += 1
            if result.get('position', 99) <= 3:
                standings[driver_id]['podiums'] += 1
        
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