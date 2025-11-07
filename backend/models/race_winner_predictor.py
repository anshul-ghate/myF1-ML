"""
Race Winner Prediction Model
XGBoost classifier for predicting race winners
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os
from loguru import logger
from datetime import datetime

from config import settings


class RaceWinnerPredictor:
    """XGBoost model for predicting race winners"""
    
    FEATURE_NAMES = [
        'driver_avg_position_last_5',
        'driver_avg_points_last_5',
        'driver_podium_rate_last_5',
        'constructor_avg_position_last_5',
        'driver_championship_position',
        'driver_championship_points',
        'qualifying_position',
        'circuit_overtaking_difficulty'
    ]
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(settings.MODEL_DIR, 'race_winner_model.pkl')
        
        # Create model directory
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
    
    def prepare_training_data(self, supabase_client) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from database
        
        Returns:
            X: Feature matrix
            y: Target vector (1 if won, 0 otherwise)
            driver_ids: List of driver IDs
        """
        logger.info("Preparing training data...")
        
        try:
            # Fetch race results
            response = supabase_client.table('race_results')\
                .select('*, races(*), drivers(*), constructors(*)')\
                .execute()
            
            if not response.data:
                logger.warning("No race results found in database")
                return np.array([]), np.array([]), []
            
            X = []
            y = []
            driver_ids = []
            
            for result in response.data:
                try:
                    features = self._extract_features(result, supabase_client)
                    X.append(features)
                    y.append(1 if result['position'] == 1 else 0)
                    driver_ids.append(result['driver_id'])
                except Exception as e:
                    logger.warning(f"Error extracting features for result: {e}")
                    continue
            
            logger.info(f"Prepared {len(X)} training samples")
            return np.array(X), np.array(y), driver_ids
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _extract_features(self, result: Dict, supabase_client) -> List[float]:
        """Extract feature vector for a single race result"""
        features = []
        
        driver_id = result['driver_id']
        
        # Simple feature extraction (can be enhanced)
        # For now, use simplified features
        features.extend([
            float(result.get('position', 10)),  # Current position as proxy
            float(result.get('points', 0)),  # Points
            0.5,  # Podium rate (placeholder)
            10.0,  # Constructor avg position (placeholder)
            5,  # Championship position (placeholder)
            100,  # Championship points (placeholder)
            float(result.get('position', 10)),  # Qualifying (use race position as proxy)
            5  # Circuit difficulty (placeholder)
        ])
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the XGBoost model
        
        Returns:
            Dictionary with training metrics
        """
        if len(X) == 0:
            logger.warning("No training data available")
            return {'accuracy': 0, 'log_loss': 0}
        
        logger.info("Training race winner prediction model...")
        
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'random_state': 42
            }
            
            self.model = xgb.XGBClassifier(**params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            accuracy = accuracy_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_proba)
            
            logger.info(f"Model trained - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
            
            # Save model
            self.save_model()
            
            return {
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'trained_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict_race(self, race_id: str, supabase_client) -> List[Dict]:
        """
        Generate predictions for all drivers in a race
        
        Returns:
            List of predictions with driver_id, win_probability, confidence
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return []
        
        try:
            # Get race entries (drivers participating)
            # For now, get all drivers
            response = supabase_client.table('drivers').select('*').execute()
            
            predictions = []
            
            for driver in response.data[:20]:  # Limit to 20 drivers
                # Extract features (simplified)
                features = [5.0, 10.0, 0.5, 10.0, 5, 100, 10.0, 5]
                
                # Predict
                X = np.array([features])
                win_probability = float(self.model.predict_proba(X)[0, 1])
                
                predictions.append({
                    'driver_id': driver['id'],
                    'driver_name': driver['name'],
                    'win_probability': win_probability,
                    'confidence': self._calculate_confidence(win_probability)
                })
            
            # Sort by probability
            predictions.sort(key=lambda x: x['win_probability'], reverse=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence score"""
        return 1 - 2 * abs(probability - 0.5)
    
    def save_model(self):
        """Save model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model from disk"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")