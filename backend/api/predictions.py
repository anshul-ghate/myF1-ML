"""
Predictions API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

from services.supabase_client import get_supabase_client
from models.race_winner_predictor import RaceWinnerPredictor

router = APIRouter()


class PredictionRequest(BaseModel):
    race_id: str


class PredictionResponse(BaseModel):
    driver_id: str
    driver_name: str
    win_probability: float
    confidence: float


@router.post("/generate", response_model=List[PredictionResponse])
async def generate_predictions(request: PredictionRequest):
    """Generate race winner predictions"""
    supabase = get_supabase_client()
    
    try:
        # Initialize predictor
        predictor = RaceWinnerPredictor()
        
        # Check if model is trained
        if predictor.model is None:
            # Train model if not already trained
            X, y, _ = predictor.prepare_training_data(supabase)
            if len(X) > 0:
                predictor.train(X, y)
        
        # Generate predictions
        predictions = predictor.predict_race(request.race_id, supabase)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{race_id}", response_model=List[PredictionResponse])
async def get_predictions(race_id: str):
    """Get stored predictions for a race"""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table('predictions')\
            .select('*, drivers(*)')\
            .eq('race_id', race_id)\
            .execute()
        
        # Format response
        predictions = []
        for pred in response.data:
            predictions.append({
                'driver_id': pred['driver_id'],
                'driver_name': pred['drivers']['name'],
                'win_probability': pred.get('confidence', 0.5),  # Use confidence as proxy
                'confidence': pred.get('confidence', 0.5)
            })
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model():
    """Train the prediction model"""
    supabase = get_supabase_client()
    
    try:
        predictor = RaceWinnerPredictor()
        
        # Prepare training data
        X, y, _ = predictor.prepare_training_data(supabase)
        
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Train model
        metrics = predictor.train(X, y)
        
        return {
            'success': True,
            'metrics': metrics,
            'message': 'Model trained successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))