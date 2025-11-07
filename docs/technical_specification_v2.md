# F1 Analytics & Prediction Platform - Technical Specification v2.0

## Executive Summary

This document provides a complete technical specification for rebuilding the F1 Analytics & Prediction Platform with real machine learning capabilities, FastF1 integration, and automated data synchronization. This addresses the critical gaps in the current implementation.

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                           │
│  React + TypeScript + Shadcn-UI + TailwindCSS               │
│  - Dashboard, Predictions, Strategy Simulator, Analytics    │
└──────────────────┬──────────────────────────────────────────┘
                   │ REST API / WebSocket
┌──────────────────▼──────────────────────────────────────────┐
│              Python Backend Service (FastAPI)                │
│  - FastF1 Integration                                        │
│  - ML Prediction Engine                                      │
│  - Strategy Simulator Engine                                 │
│  - Automated Data Sync Agent                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼─────────┐
│   Supabase     │   │   FastF1 API     │
│   PostgreSQL   │   │   Ergast API     │
│   - Storage    │   │   - Live Data    │
│   - Auth       │   │   - Historical   │
└────────────────┘   └──────────────────┘
```

### 1.2 Technology Stack

**Frontend:**
- React 18 with TypeScript
- Shadcn-UI component library
- TailwindCSS for styling
- Recharts for data visualization
- React Query for data fetching
- Zustand for state management

**Backend:**
- Python 3.11+
- FastAPI framework
- FastF1 library (v3.x)
- Uvicorn ASGI server
- APScheduler for automated tasks

**Machine Learning:**
- scikit-learn (Random Forest, XGBoost)
- TensorFlow/Keras (LSTM, Neural Networks)
- pandas for data manipulation
- numpy for numerical operations

**Database & Services:**
- Supabase (PostgreSQL + Auth + Storage)
- Redis for caching
- WebSocket for real-time updates

---

## 2. FastF1 Integration Strategy

### 2.1 Data Sources

**FastF1 Library Capabilities:**
- Session data (Practice, Qualifying, Race)
- Lap timing data with sector times
- Car telemetry (Speed, RPM, Gear, Throttle, Brake, DRS)
- Weather data (Temperature, Humidity, Rainfall, Wind)
- Tire compound data and pit stop information
- Track status (Flags, Safety Car periods)
- Race control messages

**Data Collection Workflow:**

```python
# Example FastF1 Data Collection
import fastf1

# Enable caching for performance
fastf1.Cache.enable_cache('/cache')

# Load session
session = fastf1.get_session(2024, 'Monaco', 'Race')
session.load()

# Get comprehensive data
laps = session.laps
results = session.results
weather = session.weather_data
telemetry = session.car_data
```

### 2.2 Data Sync Agent Architecture

**Automated Background Service:**

```python
# Pseudo-code for Data Sync Agent
class F1DataSyncAgent:
    def __init__(self):
        self.scheduler = APScheduler()
        self.fastf1_client = FastF1Client()
        self.supabase_client = SupabaseClient()
    
    async def sync_schedule(self):
        """Sync race calendar (runs daily)"""
        schedule = await self.fastf1_client.get_schedule(2024)
        await self.supabase_client.upsert_races(schedule)
    
    async def sync_session_data(self, year, race, session_type):
        """Sync session data after each session"""
        session = await self.fastf1_client.load_session(year, race, session_type)
        
        # Extract and store data
        await self.store_lap_data(session.laps)
        await self.store_telemetry(session.car_data)
        await self.store_weather(session.weather_data)
        await self.store_results(session.results)
    
    async def monitor_live_sessions(self):
        """Monitor for live sessions and sync in real-time"""
        while True:
            active_sessions = await self.get_active_sessions()
            for session in active_sessions:
                await self.sync_session_data(session)
            await asyncio.sleep(60)  # Check every minute
    
    def start(self):
        """Start all automated tasks"""
        # Daily schedule sync
        self.scheduler.add_job(self.sync_schedule, 'cron', hour=0)
        
        # Live session monitoring
        self.scheduler.add_job(self.monitor_live_sessions, 'interval', minutes=1)
        
        # Historical data backfill (one-time)
        self.scheduler.add_job(self.backfill_historical_data, trigger='date')
        
        self.scheduler.start()
```

### 2.3 Data Models

**Enhanced Database Schema:**

```sql
-- Lap Times Table
CREATE TABLE lap_times (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID REFERENCES races(id),
    driver_id UUID REFERENCES drivers(id),
    lap_number INTEGER,
    lap_time INTERVAL,
    sector_1_time INTERVAL,
    sector_2_time INTERVAL,
    sector_3_time INTERVAL,
    compound VARCHAR(20),
    tire_life INTEGER,
    track_status VARCHAR(50),
    is_personal_best BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Telemetry Data Table
CREATE TABLE telemetry_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lap_id UUID REFERENCES lap_times(id),
    time_offset FLOAT,
    speed FLOAT,
    rpm INTEGER,
    gear INTEGER,
    throttle FLOAT,
    brake FLOAT,
    drs INTEGER,
    distance FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Weather Data Table
CREATE TABLE weather_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID REFERENCES races(id),
    session_type VARCHAR(20),
    timestamp TIMESTAMP,
    air_temp FLOAT,
    track_temp FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    wind_direction INTEGER,
    rainfall BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pit Stops Table
CREATE TABLE pit_stops (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID REFERENCES races(id),
    driver_id UUID REFERENCES drivers(id),
    stop_number INTEGER,
    lap_number INTEGER,
    pit_time INTERVAL,
    duration INTERVAL,
    compound_before VARCHAR(20),
    compound_after VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 3. Machine Learning Prediction System

### 3.1 Prediction Models Architecture

**Model Pipeline:**

```
Raw Data → Feature Engineering → Model Training → Prediction → Confidence Score
```

**Three-Tier Prediction System:**

1. **Race Winner Prediction** (Classification)
2. **Driver Finishing Position** (Regression)
3. **Lap Time Prediction** (Time Series)

### 3.2 Feature Engineering

**Input Features (50+ features):**

**Driver Features:**
- Recent form (last 5 races average position)
- Season points total
- Win rate at specific circuit
- Average qualifying position
- DNF rate
- Overtaking ability score

**Circuit Features:**
- Circuit type (street, permanent, hybrid)
- Track length
- Number of corners
- Average speed
- Elevation changes
- Overtaking difficulty score

**Car/Constructor Features:**
- Constructor championship position
- Recent performance trend
- Power unit reliability score
- Aerodynamic efficiency rating

**Session Features:**
- Grid position
- Qualifying gap to pole
- Free practice pace
- Tire strategy (planned compounds)

**Environmental Features:**
- Weather forecast (temperature, rain probability)
- Track temperature
- Wind speed and direction

**Historical Features:**
- Driver's best finish at circuit
- Constructor's average position at circuit
- Head-to-head record against competitors

### 3.3 Model Implementations

#### Model 1: Race Winner Prediction (XGBoost Classifier)

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RaceWinnerPredictor:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.feature_columns = []
    
    def prepare_features(self, race_data):
        """Engineer features from raw data"""
        features = pd.DataFrame()
        
        # Driver features
        features['driver_recent_form'] = race_data.groupby('driver_id')['position'].rolling(5).mean()
        features['driver_season_points'] = race_data.groupby('driver_id')['points'].cumsum()
        features['driver_circuit_win_rate'] = self.calculate_circuit_win_rate(race_data)
        
        # Circuit features
        features['circuit_type_encoded'] = self.encode_circuit_type(race_data['circuit_id'])
        features['overtaking_difficulty'] = self.get_overtaking_score(race_data['circuit_id'])
        
        # Constructor features
        features['constructor_form'] = race_data.groupby('constructor_id')['points'].rolling(3).mean()
        
        # Weather features
        features['rain_probability'] = race_data['weather_rain_prob']
        features['temperature'] = race_data['weather_temp']
        
        # Qualifying features
        features['grid_position'] = race_data['grid']
        features['quali_gap_to_pole'] = race_data['quali_time'] - race_data['pole_time']
        
        return features
    
    def train(self, historical_data):
        """Train the model on historical race data"""
        X = self.prepare_features(historical_data)
        y = (historical_data['position'] == 1).astype(int)  # Binary: win or not
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        return self.model
    
    def predict(self, race_features):
        """Predict race winner with confidence"""
        X = self.prepare_features(race_features)
        
        # Get probabilities for all drivers
        probabilities = self.model.predict_proba(X)[:, 1]
        
        predictions = pd.DataFrame({
            'driver_id': race_features['driver_id'],
            'win_probability': probabilities,
            'confidence_score': self.calculate_confidence(probabilities)
        })
        
        return predictions.sort_values('win_probability', ascending=False)
```

#### Model 2: Finishing Position Prediction (Random Forest Regressor)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class FinishingPositionPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_features(self, race_data):
        """Feature engineering for position prediction"""
        features = pd.DataFrame()
        
        # Grid position (strongest predictor)
        features['grid_position'] = race_data['grid']
        
        # Recent performance
        features['avg_position_last_5'] = race_data.groupby('driver_id')['position'].rolling(5).mean()
        features['avg_position_last_3'] = race_data.groupby('driver_id')['position'].rolling(3).mean()
        
        # Qualifying performance
        features['quali_position'] = race_data['quali_position']
        features['quali_gap'] = race_data['quali_gap_to_pole']
        
        # Circuit-specific performance
        features['avg_position_at_circuit'] = self.get_circuit_average(race_data)
        features['best_finish_at_circuit'] = self.get_circuit_best(race_data)
        
        # Constructor strength
        features['constructor_avg_position'] = race_data.groupby('constructor_id')['position'].rolling(5).mean()
        
        # Reliability
        features['dnf_rate'] = self.calculate_dnf_rate(race_data)
        
        # Weather impact
        features['rain_expected'] = race_data['weather_rain_prob'] > 0.3
        features['temperature'] = race_data['weather_temp']
        
        return features
    
    def train(self, historical_data):
        """Train position prediction model"""
        X = self.prepare_features(historical_data)
        y = historical_data['position']
        
        # Remove DNFs from training
        mask = y <= 20
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.2f} positions")
        print(f"R² Score: {r2:.4f}")
        
        return self.model
    
    def predict(self, race_features):
        """Predict finishing positions"""
        X = self.prepare_features(race_features)
        predicted_positions = self.model.predict(X)
        
        # Calculate prediction intervals
        predictions = pd.DataFrame({
            'driver_id': race_features['driver_id'],
            'predicted_position': predicted_positions,
            'lower_bound': predicted_positions - 2,  # 95% confidence
            'upper_bound': predicted_positions + 2
        })
        
        return predictions.sort_values('predicted_position')
```

#### Model 3: Lap Time Prediction (LSTM Neural Network)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LapTimePredictor:
    def __init__(self):
        self.model = None
        self.sequence_length = 10  # Use last 10 laps
    
    def build_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = keras.Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Predict lap time
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, lap_data):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for driver in lap_data['driver_id'].unique():
            driver_laps = lap_data[lap_data['driver_id'] == driver].sort_values('lap_number')
            
            # Features: lap_time, tire_age, fuel_load, track_temp, position
            features = driver_laps[['lap_time_seconds', 'tire_age', 'fuel_load', 
                                   'track_temp', 'position']].values
            
            for i in range(len(features) - self.sequence_length):
                sequences.append(features[i:i+self.sequence_length])
                targets.append(features[i+self.sequence_length, 0])  # Next lap time
        
        return np.array(sequences), np.array(targets)
    
    def train(self, historical_lap_data):
        """Train LSTM model"""
        X, y = self.prepare_sequences(historical_lap_data)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5)
            ]
        )
        
        return self.model, history
    
    def predict(self, recent_laps):
        """Predict next lap time"""
        X = self.prepare_sequences(recent_laps)[0]
        predicted_lap_times = self.model.predict(X)
        
        return predicted_lap_times
```

### 3.4 Model Training Pipeline

```python
class MLPipeline:
    def __init__(self):
        self.winner_model = RaceWinnerPredictor()
        self.position_model = FinishingPositionPredictor()
        self.laptime_model = LapTimePredictor()
        self.data_loader = FastF1DataLoader()
    
    async def train_all_models(self):
        """Complete training pipeline"""
        print("Loading historical data...")
        historical_data = await self.data_loader.load_historical_races(
            start_year=2018,
            end_year=2024
        )
        
        print("Training Race Winner Model...")
        self.winner_model.train(historical_data)
        
        print("Training Position Predictor...")
        self.position_model.train(historical_data)
        
        print("Training Lap Time Predictor...")
        lap_data = await self.data_loader.load_lap_times(historical_data)
        self.laptime_model.train(lap_data)
        
        print("Saving models...")
        self.save_models()
        
        print("Model training complete!")
    
    def save_models(self):
        """Persist trained models"""
        joblib.dump(self.winner_model, 'models/winner_model.pkl')
        joblib.dump(self.position_model, 'models/position_model.pkl')
        self.laptime_model.model.save('models/laptime_model.h5')
    
    async def generate_race_predictions(self, race_id):
        """Generate predictions for upcoming race"""
        race_data = await self.data_loader.load_race_data(race_id)
        
        # Winner prediction
        winner_probs = self.winner_model.predict(race_data)
        
        # Position predictions
        positions = self.position_model.predict(race_data)
        
        # Combine results
        predictions = pd.merge(winner_probs, positions, on='driver_id')
        
        return predictions
```

---

## 4. Strategy Simulator Engine

### 4.1 Tire Strategy Optimization

```python
class TireStrategySimulator:
    def __init__(self):
        self.tire_degradation_models = {
            'SOFT': {'base_wear': 0.15, 'performance_drop': 0.05},
            'MEDIUM': {'base_wear': 0.10, 'performance_drop': 0.03},
            'HARD': {'base_wear': 0.07, 'performance_drop': 0.02}
        }
        self.pit_stop_loss = 22.0  # seconds lost in pit stop
    
    def simulate_race(self, strategy, race_params):
        """
        Simulate a race with given tire strategy
        
        strategy: List of (compound, stint_length) tuples
        race_params: Dict with circuit, weather, traffic info
        """
        total_time = 0
        current_lap = 0
        lap_times = []
        
        for compound, stint_length in strategy:
            stint_time = self.simulate_stint(
                compound=compound,
                laps=stint_length,
                start_lap=current_lap,
                race_params=race_params
            )
            
            total_time += stint_time
            current_lap += stint_length
            
            # Add pit stop time (except last stint)
            if current_lap < race_params['total_laps']:
                total_time += self.pit_stop_loss
                lap_times.append({'lap': current_lap, 'event': 'PIT_STOP'})
        
        return {
            'total_race_time': total_time,
            'lap_times': lap_times,
            'strategy': strategy
        }
    
    def simulate_stint(self, compound, laps, start_lap, race_params):
        """Simulate a tire stint with degradation"""
        base_lap_time = race_params['base_lap_time']
        degradation = self.tire_degradation_models[compound]
        
        stint_time = 0
        
        for lap in range(laps):
            # Tire degradation effect
            tire_age = lap + 1
            degradation_factor = 1 + (degradation['base_wear'] * tire_age / 10)
            performance_loss = degradation['performance_drop'] * tire_age
            
            # Fuel effect (car gets lighter)
            fuel_effect = -0.03 * (race_params['total_laps'] - (start_lap + lap)) / race_params['total_laps']
            
            # Traffic effect
            traffic_factor = self.calculate_traffic_impact(start_lap + lap, race_params)
            
            lap_time = base_lap_time * degradation_factor + performance_loss + fuel_effect + traffic_factor
            stint_time += lap_time
        
        return stint_time
    
    def optimize_strategy(self, race_params):
        """Find optimal tire strategy using dynamic programming"""
        total_laps = race_params['total_laps']
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        
        # Must use at least 2 different compounds
        best_strategies = []
        
        # Generate all valid strategies (1-3 stops)
        for num_stops in range(1, 4):
            strategies = self.generate_strategies(total_laps, num_stops, compounds)
            
            for strategy in strategies:
                result = self.simulate_race(strategy, race_params)
                best_strategies.append({
                    'strategy': strategy,
                    'total_time': result['total_race_time'],
                    'num_stops': num_stops
                })
        
        # Sort by total time
        best_strategies.sort(key=lambda x: x['total_time'])
        
        return best_strategies[:5]  # Return top 5 strategies
    
    def calculate_undercut_window(self, race_params, current_lap):
        """Calculate optimal lap for undercut"""
        # Simulate staying out vs pitting
        stay_out_time = self.simulate_stint('MEDIUM', 5, current_lap, race_params)
        pit_and_new_tires = self.pit_stop_loss + self.simulate_stint('SOFT', 5, current_lap, race_params)
        
        undercut_advantage = stay_out_time - pit_and_new_tires
        
        return {
            'undercut_advantage': undercut_advantage,
            'recommended': undercut_advantage > 0
        }
```

### 4.2 Race Pace Simulator

```python
class RacePaceSimulator:
    def __init__(self):
        self.overtaking_difficulty = {}  # Circuit-specific
    
    def simulate_race_progression(self, starting_grid, strategies, race_params):
        """Simulate full race with all drivers"""
        positions = starting_grid.copy()
        lap_by_lap = []
        
        for lap in range(1, race_params['total_laps'] + 1):
            # Calculate lap times for each driver
            lap_times = {}
            for driver in positions:
                lap_times[driver] = self.calculate_lap_time(
                    driver=driver,
                    lap=lap,
                    position=positions[driver],
                    strategy=strategies[driver],
                    race_params=race_params
                )
            
            # Update positions based on lap times
            positions = self.update_positions(positions, lap_times)
            
            # Handle pit stops
            positions = self.process_pit_stops(positions, lap, strategies)
            
            lap_by_lap.append({
                'lap': lap,
                'positions': positions.copy(),
                'lap_times': lap_times.copy()
            })
        
        return lap_by_lap
    
    def calculate_overtaking_probability(self, attacker, defender, circuit):
        """Calculate probability of successful overtake"""
        # Factors: pace difference, DRS availability, circuit difficulty
        pace_diff = attacker['lap_time'] - defender['lap_time']
        
        base_probability = 0.1  # Base 10% chance
        
        # Pace advantage
        if pace_diff < -0.5:  # 0.5s faster
            base_probability += 0.3
        
        # DRS effect
        if attacker['drs_available']:
            base_probability += 0.4
        
        # Circuit difficulty
        circuit_factor = self.overtaking_difficulty.get(circuit, 0.5)
        base_probability *= circuit_factor
        
        return min(base_probability, 0.9)  # Cap at 90%
```

---

## 5. Backend API Design

### 5.1 FastAPI Application Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Configuration
│   ├── models/                 # Pydantic models
│   │   ├── __init__.py
│   │   ├── race.py
│   │   ├── driver.py
│   │   ├── prediction.py
│   │   └── strategy.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── fastf1_service.py
│   │   ├── ml_service.py
│   │   ├── strategy_service.py
│   │   └── sync_agent.py
│   ├── api/                    # API routes
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── races.py
│   │   │   ├── drivers.py
│   │   │   ├── predictions.py
│   │   │   └── strategy.py
│   ├── db/                     # Database
│   │   ├── __init__.py
│   │   ├── supabase.py
│   │   └── queries.py
│   └── ml/                     # ML models
│       ├── __init__.py
│       ├── models/
│       │   ├── winner_predictor.py
│       │   ├── position_predictor.py
│       │   └── laptime_predictor.py
│       └── training/
│           └── train_pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 5.2 Core API Endpoints

```python
# app/main.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import races, drivers, predictions, strategy
from app.services.sync_agent import F1DataSyncAgent

app = FastAPI(title="F1 Analytics API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(races.router, prefix="/api/v1/races", tags=["races"])
app.include_router(drivers.router, prefix="/api/v1/drivers", tags=["drivers"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["strategy"])

# Initialize sync agent
sync_agent = F1DataSyncAgent()

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    # Start automated data sync
    sync_agent.start()
    
    # Initial data sync if database is empty
    background_tasks = BackgroundTasks()
    background_tasks.add_task(sync_agent.initial_sync)

@app.get("/")
async def root():
    return {"message": "F1 Analytics API v2.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

```python
# app/api/v1/predictions.py
from fastapi import APIRouter, HTTPException
from app.services.ml_service import MLService
from app.models.prediction import PredictionRequest, PredictionResponse

router = APIRouter()
ml_service = MLService()

@router.post("/race/{race_id}", response_model=PredictionResponse)
async def predict_race(race_id: str):
    """Generate predictions for upcoming race"""
    try:
        predictions = await ml_service.generate_race_predictions(race_id)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/race/{race_id}/winner")
async def predict_winner(race_id: str):
    """Predict race winner with probabilities"""
    predictions = await ml_service.predict_winner(race_id)
    return predictions

@router.get("/race/{race_id}/positions")
async def predict_positions(race_id: str):
    """Predict finishing positions for all drivers"""
    predictions = await ml_service.predict_positions(race_id)
    return predictions

@router.post("/race/{race_id}/lap-times")
async def predict_lap_times(race_id: str, lap_number: int):
    """Predict lap times for specific lap"""
    predictions = await ml_service.predict_lap_times(race_id, lap_number)
    return predictions
```

```python
# app/api/v1/strategy.py
from fastapi import APIRouter
from app.services.strategy_service import StrategyService
from app.models.strategy import StrategyRequest, StrategyResponse

router = APIRouter()
strategy_service = StrategyService()

@router.post("/optimize", response_model=StrategyResponse)
async def optimize_strategy(request: StrategyRequest):
    """Optimize tire strategy for race"""
    optimal_strategies = await strategy_service.optimize_tire_strategy(
        race_id=request.race_id,
        driver_id=request.driver_id,
        race_params=request.race_params
    )
    return optimal_strategies

@router.post("/simulate")
async def simulate_race(request: StrategyRequest):
    """Simulate race with given strategy"""
    simulation = await strategy_service.simulate_race(
        race_id=request.race_id,
        strategies=request.strategies
    )
    return simulation

@router.get("/undercut/{race_id}")
async def calculate_undercut(race_id: str, current_lap: int):
    """Calculate undercut window"""
    undercut_data = await strategy_service.calculate_undercut_window(
        race_id=race_id,
        current_lap=current_lap
    )
    return undercut_data
```

---

## 6. Frontend Integration

### 6.1 API Service Layer

```typescript
// src/lib/apiClient.ts
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Predictions API
export const predictionsAPI = {
  getRacePredictions: async (raceId: string) => {
    const response = await apiClient.post(`/predictions/race/${raceId}`);
    return response.data;
  },
  
  getWinnerPrediction: async (raceId: string) => {
    const response = await apiClient.get(`/predictions/race/${raceId}/winner`);
    return response.data;
  },
  
  getPositionPredictions: async (raceId: string) => {
    const response = await apiClient.get(`/predictions/race/${raceId}/positions`);
    return response.data;
  },
};

// Strategy API
export const strategyAPI = {
  optimizeStrategy: async (raceId: string, driverId: string, params: any) => {
    const response = await apiClient.post('/strategy/optimize', {
      race_id: raceId,
      driver_id: driverId,
      race_params: params,
    });
    return response.data;
  },
  
  simulateRace: async (raceId: string, strategies: any) => {
    const response = await apiClient.post('/strategy/simulate', {
      race_id: raceId,
      strategies,
    });
    return response.data;
  },
};

// Races API
export const racesAPI = {
  getCurrentSeason: async () => {
    const response = await apiClient.get('/races/season/current');
    return response.data;
  },
  
  getNextRace: async () => {
    const response = await apiClient.get('/races/next');
    return response.data;
  },
  
  getRaceResults: async (raceId: string) => {
    const response = await apiClient.get(`/races/${raceId}/results`);
    return response.data;
  },
};

// Drivers API
export const driversAPI = {
  getAllDrivers: async () => {
    const response = await apiClient.get('/drivers');
    return response.data;
  },
  
  getDriverDetails: async (driverId: string) => {
    const response = await apiClient.get(`/drivers/${driverId}`);
    return response.data;
  },
  
  getDriverStandings: async () => {
    const response = await apiClient.get('/drivers/standings');
    return response.data;
  },
};
```

### 6.2 React Query Integration

```typescript
// src/hooks/usePredictions.ts
import { useQuery, useMutation } from '@tanstack/react-query';
import { predictionsAPI } from '@/lib/apiClient';

export const usePredictions = (raceId: string) => {
  return useQuery({
    queryKey: ['predictions', raceId],
    queryFn: () => predictionsAPI.getRacePredictions(raceId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    enabled: !!raceId,
  });
};

export const useWinnerPrediction = (raceId: string) => {
  return useQuery({
    queryKey: ['winner-prediction', raceId],
    queryFn: () => predictionsAPI.getWinnerPrediction(raceId),
    enabled: !!raceId,
  });
};

// src/hooks/useStrategy.ts
export const useStrategyOptimization = () => {
  return useMutation({
    mutationFn: ({ raceId, driverId, params }: any) =>
      strategyAPI.optimizeStrategy(raceId, driverId, params),
  });
};
```

### 6.3 Enhanced Dashboard Components

```typescript
// src/components/dashboard/PredictionsCard.tsx
import { usePredictions } from '@/hooks/usePredictions';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

export const PredictionsCard = ({ raceId }: { raceId: string }) => {
  const { data: predictions, isLoading, error } = usePredictions(raceId);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-48">
          <Loader2 className="h-8 w-8 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="text-red-500">
          Error loading predictions
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Race Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {predictions?.drivers.map((driver: any) => (
            <div key={driver.id} className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="font-bold">{driver.predicted_position}</span>
                <span>{driver.name}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                {(driver.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
```

---

## 7. Deployment Architecture

### 7.1 Docker Configuration

```dockerfile
# Dockerfile for Python Backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create cache directory for FastF1
RUN mkdir -p /app/cache

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./backend:/app
      - fastf1-cache:/app/cache
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000/api/v1
      - VITE_SUPABASE_URL=${SUPABASE_URL}
      - VITE_SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  fastf1-cache:
  redis-data:
```

### 7.2 Production Deployment

**Backend Deployment (Railway/Render/Fly.io):**
- Python FastAPI service
- Environment variables for Supabase
- Persistent volume for FastF1 cache
- Redis for caching

**Frontend Deployment (Vercel/Netlify):**
- React application
- Environment variables for API URL
- Automatic deployments from Git

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Python FastAPI backend structure
- [ ] Integrate FastF1 library
- [ ] Create database schema in Supabase
- [ ] Implement basic data sync agent
- [ ] Set up Docker environment

### Phase 2: Data Pipeline (Week 3-4)
- [ ] Implement automated data sync from FastF1
- [ ] Build historical data backfill process
- [ ] Create data transformation pipelines
- [ ] Set up Redis caching
- [ ] Test data quality and consistency

### Phase 3: ML Models (Week 5-7)
- [ ] Collect and prepare training data
- [ ] Implement Race Winner Predictor (XGBoost)
- [ ] Implement Position Predictor (Random Forest)
- [ ] Implement Lap Time Predictor (LSTM)
- [ ] Train models on historical data
- [ ] Validate model accuracy
- [ ] Create model serving endpoints

### Phase 4: Strategy Simulator (Week 8-9)
- [ ] Implement tire degradation models
- [ ] Build race pace simulator
- [ ] Create strategy optimization algorithms
- [ ] Add undercut/overcut calculations
- [ ] Test simulator accuracy

### Phase 5: API Development (Week 10-11)
- [ ] Build all REST API endpoints
- [ ] Implement WebSocket for real-time updates
- [ ] Add authentication and authorization
- [ ] Create API documentation
- [ ] Performance optimization

### Phase 6: Frontend Integration (Week 12-14)
- [ ] Update all dashboard components
- [ ] Build Predictions page
- [ ] Build Strategy Simulator page
- [ ] Build detailed Driver/Constructor pages
- [ ] Implement real-time data updates
- [ ] Add data visualizations

### Phase 7: Testing & Optimization (Week 15-16)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Load testing
- [ ] Bug fixes
- [ ] Documentation

### Phase 8: Deployment (Week 17-18)
- [ ] Set up production environment
- [ ] Deploy backend service
- [ ] Deploy frontend application
- [ ] Configure monitoring and logging
- [ ] Final testing in production

---

## 9. Success Metrics

### Technical Metrics
- **API Response Time**: < 500ms for 95% of requests
- **Model Accuracy**: 
  - Winner prediction: > 60% accuracy
  - Position prediction: MAE < 2 positions
  - Lap time prediction: MAE < 0.5 seconds
- **Data Freshness**: < 5 minutes delay from live sessions
- **System Uptime**: > 99.5%

### User Experience Metrics
- **Page Load Time**: < 3 seconds
- **Data Availability**: 100% of races have predictions
- **Feature Completeness**: All promised features functional
- **User Satisfaction**: Positive feedback on accuracy

---

## 10. Risk Mitigation

### Technical Risks
1. **FastF1 API Rate Limits**
   - Mitigation: Implement caching, use historical data
   
2. **Model Training Time**
   - Mitigation: Use pre-trained models, incremental learning
   
3. **Real-time Data Delays**
   - Mitigation: Fallback to cached data, show staleness indicators

4. **Deployment Complexity**
   - Mitigation: Use Docker, comprehensive documentation

### Data Risks
1. **Missing Historical Data**
   - Mitigation: Multiple data sources, data validation
   
2. **API Changes**
   - Mitigation: Version pinning, adapter pattern

---

## 11. Conclusion

This technical specification provides a complete blueprint for rebuilding the F1 Analytics & Prediction Platform with:

✅ **Real FastF1 Integration** - Rich telemetry and timing data
✅ **Actual ML Models** - XGBoost, Random Forest, LSTM for predictions
✅ **Automated Data Sync** - Background agent for continuous updates
✅ **Strategy Simulator** - Tire optimization and race simulation
✅ **Complete Feature Set** - All promised functionality implemented
✅ **Production-Ready Architecture** - Scalable, maintainable, deployable

The implementation will take approximately 18 weeks with a dedicated team, resulting in a fully functional, intelligent F1 analytics application that meets all user requirements.