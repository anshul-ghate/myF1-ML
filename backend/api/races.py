"""
Races API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from services.supabase_client import get_supabase_client

router = APIRouter()


class Race(BaseModel):
    id: str
    season_year: int
    round: int
    name: str
    date: str
    time: str
    circuit_id: str


@router.get("/", response_model=List[Race])
async def get_races(season: Optional[int] = None):
    """Get all races, optionally filtered by season"""
    supabase = get_supabase_client()
    
    try:
        query = supabase.table('races').select('*')
        
        if season:
            query = query.eq('season_year', season)
        
        response = query.order('date').execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming")
async def get_upcoming_races():
    """Get upcoming races"""
    supabase = get_supabase_client()
    
    try:
        today = datetime.now().date().isoformat()
        
        response = supabase.table('races')\
            .select('*, circuits(*)')\
            .gte('date', today)\
            .order('date')\
            .limit(5)\
            .execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{race_id}")
async def get_race(race_id: str):
    """Get race by ID"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('races')\
            .select('*, circuits(*)')\
            .eq('id', race_id)\
            .single()\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Race not found")
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{race_id}/results")
async def get_race_results(race_id: str):
    """Get results for a specific race"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('race_results')\
            .select('*, drivers(*), constructors(*)')\
            .eq('race_id', race_id)\
            .order('position')\
            .execute()
        
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))