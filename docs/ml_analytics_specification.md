# ML Analytics Specification: F1 Analytics Platform

**Version**: 1.0  
**Date**: 2025-11-07  
**Author**: David, Data Analyst  
**Status**: Implementation Ready

---

## Executive Summary

This document provides a comprehensive technical specification for implementing the F1 Analytics Platform with real machine learning, FastF1 integration, and all planned features. This specification addresses all gaps identified in the gap analysis and provides detailed implementation guidance for the development team.

### Key Objectives:
1. **Replace mock data with real F1 data** using FastF1 library
2. **Implement genuine ML models** for predictions (not random numbers)
3. **Build autonomous agent system** for automatic data sync and model updates
4. **Complete all missing features** (Predictions, Strategy Simulator, Drivers, Live Race)
5. **Create production-ready analytics** with proper error handling and monitoring

---

## Table of Contents

1. [FastF1 Data Ingestion Strategy](#1-fastf1-data-ingestion-strategy)
2. [ML Model Architectures](#2-ml-model-architectures)
3. [Analytics Algorithms](#3-analytics-algorithms)
4. [Strategy Simulator Implementation](#4-strategy-simulator-implementation)
5. [Intelligent Agent Design](#5-intelligent-agent-design)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Technical Stack](#7-technical-stack)
8. [Deployment Architecture](#8-deployment-architecture)

---

## 1. FastF1 Data Ingestion Strategy

### 1.1 Overview

FastF1 is a Python library providing access to Formula 1 timing data and telemetry. It serves as our primary data source, replacing the limited Ergast API.

**Key Capabilities:**
- Historical race data (2018-present with full telemetry)
- Live timing data during race weekends
- Lap-by-lap telemetry (speed, throttle, brake, gear, DRS)
- Weather data (track temperature, air temperature)
- Pit stop information
- Session results (Practice, Qualifying, Sprint, Race)

### 1.2 Installation & Setup

**Python Environment Setup:**
```bash
# Create Python service directory
mkdir -p /workspace/python-services/data-ingestion

# Install dependencies
pip install fastf1 pandas numpy scikit-learn xgboost python-dotenv supabase-py
```

**Package Requirements (`requirements.txt`):**
```
fastf1==3.3.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
python-dotenv==1.0.0
supabase==2.3.0
requests==2.31.0
```

**FastF1 Configuration:**
```python
import fastf1
import os

# Enable caching to improve performance and reduce API calls
cache_dir = os.path.join(os.getcwd(), 'fastf1_cache')
fastf1.Cache.enable_cache(cache_dir)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
```

### 1.3 Data Ingestion Service Architecture

**File Structure:**
```
/workspace/python-services/data-ingestion/
├── main.py                 # Entry point
├── config.py               # Configuration and environment variables
├── models/
│   ├── database.py         # Supabase connection and models
│   └── schemas.py          # Data schemas
├── services/
│   ├── fastf1_service.py   # FastF1 data fetching
│   ├── ergast_service.py   # Ergast API (for historical data pre-2018)
│   └── supabase_service.py # Database operations
├── utils/
│   ├── data_transformer.py # Data transformation utilities
│   └── logger.py           # Logging configuration
└── requirements.txt
```

### 1.4 Core Data Fetching Functions

**1.4.1 Fetch Season Schedule:**
```python
import fastf1

def fetch_season_schedule(year: int) -> pd.DataFrame:
    """
    Fetch complete race schedule for a season.
    
    Returns:
        DataFrame with columns: RoundNumber, Country, Location, 
        EventName, EventDate, EventFormat, Session1-5
    """
    schedule = fastf1.get_event_schedule(year)
    return schedule

# Example usage:
schedule_2024 = fetch_season_schedule(2024)
```

**1.4.2 Fetch Race Session Data:**
```python
def fetch_race_session(year: int, round_number: int, session_type: str = 'R'):
    """
    Fetch detailed session data.
    
    Args:
        year: Season year
        round_number: Race round number
        session_type: 'FP1', 'FP2', 'FP3', 'Q', 'S' (Sprint), 'R' (Race)
    
    Returns:
        FastF1 Session object with laps, results, weather data
    """
    session = fastf1.get_session(year, round_number, session_type)
    session.load()  # Load all data (laps, telemetry, weather)
    return session

# Example usage:
race = fetch_race_session(2024, 1, 'R')  # Bahrain GP Race
laps = race.laps  # All lap data
results = race.results  # Final results
weather = race.weather_data  # Weather conditions
```

**1.4.3 Fetch Driver Lap Data:**
```python
def fetch_driver_laps(session, driver_code: str) -> pd.DataFrame:
    """
    Get all laps for a specific driver in a session.
    
    Returns:
        DataFrame with: LapNumber, LapTime, Sector1Time, Sector2Time, 
        Sector3Time, Compound, TyreLife, TrackStatus, Position
    """
    driver_laps = session.laps.pick_driver(driver_code)
    return driver_laps

# Example usage:
verstappen_laps = fetch_driver_laps(race, 'VER')
fastest_lap = verstappen_laps.pick_fastest()
```

**1.4.4 Fetch Telemetry Data:**
```python
def fetch_lap_telemetry(lap) -> pd.DataFrame:
    """
    Get detailed telemetry for a specific lap.
    
    Returns:
        DataFrame with: Time, Speed, RPM, nGear, Throttle, Brake, 
        DRS, Distance (at 100Hz sampling rate)
    """
    telemetry = lap.get_car_data()
    telemetry = telemetry.add_distance()  # Add distance traveled
    return telemetry

# Example usage:
telemetry = fetch_lap_telemetry(fastest_lap)
# telemetry now has ~6000 rows (60 second lap * 100Hz)
```

### 1.5 Data Transformation Pipeline

**Transform FastF1 data to Supabase schema:**

```python
from typing import Dict, List
import uuid

class DataTransformer:
    """Transform FastF1 data to Supabase database schema."""
    
    @staticmethod
    def transform_driver(driver_info: pd.Series) -> Dict:
        """Transform driver data to database format."""
        return {
            'id': str(uuid.uuid4()),
            'name': driver_info['FullName'],
            'code': driver_info['Abbreviation'],
            'permanent_number': int(driver_info['DriverNumber']),
            'nationality': driver_info.get('CountryCode', 'Unknown'),
            'dob': None,  # Not available in FastF1, fetch from Ergast
            'headshot_url': None  # Fetch from external source
        }
    
    @staticmethod
    def transform_race_result(lap_data: pd.DataFrame, 
                             session, 
                             race_id: str) -> List[Dict]:
        """Transform race results to database format."""
        results = []
        
        # Get final lap for each driver
        for driver in lap_data['Driver'].unique():
            driver_laps = lap_data[lap_data['Driver'] == driver]
            final_lap = driver_laps.iloc[-1]
            
            result = {
                'race_id': race_id,
                'driver_id': driver,  # Will be mapped to UUID
                'constructor_id': final_lap['Team'],  # Will be mapped to UUID
                'position': int(final_lap['Position']),
                'points': calculate_points(final_lap['Position']),
                'status': 'Finished' if pd.notna(final_lap['Position']) else 'DNF',
                'fastest_lap_time': driver_laps['LapTime'].min(),
                'fastest_lap_rank': None  # Calculate separately
            }
            results.append(result)
        
        return results
    
    @staticmethod
    def transform_lap_data_for_ml(laps: pd.DataFrame) -> pd.DataFrame:
        """
        Transform lap data into ML-ready features.
        
        Features:
        - lap_number: Current lap
        - lap_time_seconds: Lap time in seconds
        - sector_1_time, sector_2_time, sector_3_time
        - tire_compound: Encoded (Soft=0, Medium=1, Hard=2)
        - tire_age: Laps on current tire
        - position: Current race position
        - track_status: Encoded (Green=0, Yellow=1, Red=2, SC=3)
        - fuel_load_estimate: Estimated based on lap number
        """
        ml_data = laps.copy()
        
        # Convert lap times to seconds
        ml_data['lap_time_seconds'] = ml_data['LapTime'].dt.total_seconds()
        ml_data['sector_1_seconds'] = ml_data['Sector1Time'].dt.total_seconds()
        ml_data['sector_2_seconds'] = ml_data['Sector2Time'].dt.total_seconds()
        ml_data['sector_3_seconds'] = ml_data['Sector3Time'].dt.total_seconds()
        
        # Encode tire compound
        compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
        ml_data['tire_compound_encoded'] = ml_data['Compound'].map(compound_map)
        
        # Encode track status
        status_map = {1: 0, 2: 1, 4: 2, 6: 3}  # FastF1 status codes
        ml_data['track_status_encoded'] = ml_data['TrackStatus'].map(status_map)
        
        # Estimate fuel load (decreases linearly over race)
        total_laps = ml_data['LapNumber'].max()
        ml_data['fuel_load_estimate'] = 100 * (1 - ml_data['LapNumber'] / total_laps)
        
        return ml_data

def calculate_points(position: int) -> float:
    """Calculate F1 points based on finishing position."""
    points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return points_system.get(position, 0)
```

### 1.6 Database Population Strategy

**Initial Data Load (Historical):**

```python
import asyncio
from supabase import create_client, Client
from datetime import datetime

class DataIngestionService:
    """Service for ingesting F1 data into Supabase."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.transformer = DataTransformer()
    
    async def populate_historical_data(self, start_year: int = 2023, 
                                      end_year: int = 2024):
        """
        Populate database with historical data.
        
        Steps:
        1. Fetch season schedules
        2. For each race:
           - Fetch session data (Qualifying, Race)
           - Transform and insert drivers, constructors, circuits
           - Insert race results
           - Store lap data for ML training
        """
        for year in range(start_year, end_year + 1):
            print(f"Processing season {year}...")
            
            # Insert season
            await self.insert_season(year)
            
            # Fetch schedule
            schedule = fetch_season_schedule(year)
            
            for _, event in schedule.iterrows():
                try:
                    # Skip if race hasn't happened yet
                    if event['EventDate'] > datetime.now():
                        continue
                    
                    print(f"  Processing {event['EventName']}...")
                    
                    # Insert circuit
                    circuit_id = await self.insert_circuit(event)
                    
                    # Insert race
                    race_id = await self.insert_race(event, year, circuit_id)
                    
                    # Fetch and insert race results
                    race_session = fetch_race_session(year, event['RoundNumber'], 'R')
                    await self.insert_race_results(race_session, race_id)
                    
                    # Store lap data for ML
                    await self.store_lap_data_for_ml(race_session, race_id)
                    
                except Exception as e:
                    print(f"    Error processing {event['EventName']}: {e}")
                    continue
    
    async def insert_season(self, year: int):
        """Insert season into database."""
        data = {'year': year, 'url': f'https://ergast.com/api/f1/{year}'}
        self.supabase.table('seasons').upsert(data).execute()
    
    async def insert_circuit(self, event: pd.Series) -> str:
        """Insert circuit and return ID."""
        circuit_data = {
            'id': str(uuid.uuid4()),
            'name': event['Location'],
            'location': event['Location'],
            'country': event['Country']
        }
        
        # Check if exists
        existing = self.supabase.table('circuits')\
            .select('id')\
            .eq('name', circuit_data['name'])\
            .execute()
        
        if existing.data:
            return existing.data[0]['id']
        
        result = self.supabase.table('circuits').insert(circuit_data).execute()
        return result.data[0]['id']
    
    async def insert_race(self, event: pd.Series, year: int, 
                         circuit_id: str) -> str:
        """Insert race and return ID."""
        race_data = {
            'id': str(uuid.uuid4()),
            'season_year': year,
            'round': int(event['RoundNumber']),
            'name': event['EventName'],
            'date': event['EventDate'].date(),
            'time': event.get('Session5Date', event['EventDate']).time(),
            'circuit_id': circuit_id
        }
        
        result = self.supabase.table('races').insert(race_data).execute()
        return result.data[0]['id']
    
    async def insert_race_results(self, session, race_id: str):
        """Insert race results for all drivers."""
        laps = session.laps
        results = self.transformer.transform_race_result(laps, session, race_id)
        
        # Batch insert
        self.supabase.table('race_results').insert(results).execute()
    
    async def store_lap_data_for_ml(self, session, race_id: str):
        """
        Store lap-by-lap data in a format suitable for ML training.
        This creates a rich dataset for feature engineering.
        """
        laps = session.laps
        ml_data = self.transformer.transform_lap_data_for_ml(laps)
        
        # Store in a separate table or as JSON in race_results
        # For now, we'll store key aggregates
        pass

# Usage:
async def main():
    service = DataIngestionService(
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    await service.populate_historical_data(2023, 2024)

if __name__ == '__main__':
    asyncio.run(main())
```

### 1.7 Automatic Data Sync on App Startup

**Supabase Edge Function: `initialize_app_data`**

```typescript
// supabase/functions/initialize_app_data/index.ts

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )
    
    // Check if database has data
    const { data: driversCount } = await supabaseClient
      .from('drivers')
      .select('id', { count: 'exact', head: true })
    
    const { data: racesCount } = await supabaseClient
      .from('races')
      .select('id', { count: 'exact', head: true })
    
    // If empty, trigger Python data ingestion service
    if (driversCount === 0 || racesCount === 0) {
      console.log('Database is empty. Triggering data ingestion...')
      
      // Call Python service endpoint
      const response = await fetch(
        Deno.env.get('PYTHON_SERVICE_URL') + '/ingest-historical',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ years: [2023, 2024] })
        }
      )
      
      if (!response.ok) {
        throw new Error('Data ingestion failed')
      }
      
      return new Response(
        JSON.stringify({ 
          message: 'Data ingestion initiated',
          status: 'in_progress'
        }),
        { headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Check for updates (new races, driver changes)
    const lastUpdate = await supabaseClient
      .from('system_metadata')
      .select('last_sync_date')
      .single()
    
    const needsUpdate = shouldUpdate(lastUpdate?.data?.last_sync_date)
    
    if (needsUpdate) {
      // Trigger incremental update
      await fetch(
        Deno.env.get('PYTHON_SERVICE_URL') + '/sync-latest',
        { method: 'POST' }
      )
    }
    
    return new Response(
      JSON.stringify({ 
        message: 'Data is up to date',
        status: 'ready'
      }),
      { headers: { 'Content-Type': 'application/json' } }
    )
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
})

function shouldUpdate(lastSyncDate: string | null): boolean {
  if (!lastSyncDate) return true
  
  const lastSync = new Date(lastSyncDate)
  const now = new Date()
  const hoursSinceSync = (now.getTime() - lastSync.getTime()) / (1000 * 60 * 60)
  
  // Update if more than 24 hours since last sync
  return hoursSinceSync > 24
}
```

**Frontend Integration:**

```typescript
// src/lib/dataInitializer.ts

export async function initializeAppData(): Promise<void> {
  try {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_SUPABASE_URL}/functions/v1/initialize_app_data`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY}`
        }
      }
    )
    
    const result = await response.json()
    
    if (result.status === 'in_progress') {
      // Show loading state
      console.log('Data ingestion in progress...')
      // Poll for completion
      await pollForDataReady()
    }
  } catch (error) {
    console.error('Failed to initialize app data:', error)
  }
}

async function pollForDataReady(maxAttempts: number = 30): Promise<void> {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(resolve => setTimeout(resolve, 2000)) // Wait 2 seconds
    
    const { data } = await supabase
      .from('drivers')
      .select('id', { count: 'exact', head: true })
    
    if (data && data > 0) {
      console.log('Data ready!')
      return
    }
  }
  
  throw new Error('Data initialization timeout')
}

// Call on app mount
// src/pages/_app.tsx
useEffect(() => {
  initializeAppData()
}, [])
```

### 1.8 Live Data Streaming (During Race Weekends)

**Real-time Data Ingestion:**

```python
import time
from datetime import datetime

class LiveDataStreamer:
    """Stream live race data during sessions."""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.is_streaming = False
    
    async def stream_live_session(self, year: int, round_number: int):
        """
        Stream live data during a race session.
        Polls FastF1 every 5 seconds and broadcasts updates.
        """
        self.is_streaming = True
        session = fastf1.get_session(year, round_number, 'R')
        
        print(f"Starting live stream for {session.event['EventName']}...")
        
        last_lap_count = 0
        
        while self.is_streaming:
            try:
                # Reload session data
                session.load(laps=True, telemetry=False)  # Skip telemetry for speed
                
                current_lap_count = len(session.laps)
                
                # Check for new laps
                if current_lap_count > last_lap_count:
                    new_laps = session.laps[last_lap_count:]
                    
                    # Process new laps
                    for _, lap in new_laps.iterrows():
                        await self.broadcast_lap_update(lap)
                    
                    last_lap_count = current_lap_count
                
                # Wait 5 seconds before next poll
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Error in live stream: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def broadcast_lap_update(self, lap: pd.Series):
        """
        Broadcast lap update via Supabase Realtime.
        """
        update_data = {
            'type': 'LAP_UPDATE',
            'driver': lap['Driver'],
            'lap_number': int(lap['LapNumber']),
            'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
            'position': int(lap['Position']) if pd.notna(lap['Position']) else None,
            'tire_compound': lap['Compound'],
            'tire_age': int(lap['TyreLife']) if pd.notna(lap['TyreLife']) else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to Supabase Realtime channel
        await self.supabase.realtime.send(
            channel='live_race',
            event='lap_update',
            payload=update_data
        )
        
        print(f"Broadcasted: {lap['Driver']} - Lap {lap['LapNumber']}")
    
    def stop_streaming(self):
        """Stop the live stream."""
        self.is_streaming = False
```

---

## 2. ML Model Architectures

### 2.1 Overview

We will implement three primary ML models:

1. **Race Winner Prediction**: XGBoost Classifier
2. **Qualifying Position Prediction**: Random Forest Regressor
3. **Tire Degradation Model**: Polynomial Regression + LSTM

### 2.2 Race Winner Prediction Model

**Model Type**: Gradient Boosting Classifier (XGBoost)

**Objective**: Predict probability of each driver winning the race

**Features** (20 features):

```python
RACE_WINNER_FEATURES = [
    # Driver Performance (last 5 races)
    'driver_avg_position_last_5',
    'driver_avg_points_last_5',
    'driver_avg_grid_last_5',
    'driver_dnf_rate_last_5',
    'driver_podium_rate_last_5',
    
    # Constructor Performance (last 5 races)
    'constructor_avg_position_last_5',
    'constructor_avg_points_last_5',
    'constructor_reliability_last_5',
    
    # Circuit-Specific Performance
    'driver_avg_position_at_circuit',
    'driver_best_position_at_circuit',
    'constructor_avg_position_at_circuit',
    
    # Current Season Standing
    'driver_championship_position',
    'driver_championship_points',
    'constructor_championship_position',
    
    # Qualifying Performance
    'qualifying_position',
    'qualifying_gap_to_pole',
    
    # Track Characteristics
    'circuit_type',  # Street/Permanent
    'circuit_length',
    'circuit_corners',
    'circuit_overtaking_difficulty'
]
```

**Implementation:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib

class RaceWinnerPredictor:
    """XGBoost model for predicting race winners."""
    
    def __init__(self):
        self.model = None
        self.feature_names = RACE_WINNER_FEATURES
    
    def prepare_training_data(self, supabase_client) -> tuple:
        """
        Fetch and prepare training data from database.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (1 if driver won, 0 otherwise)
            driver_ids: List of driver IDs for each sample
        """
        # Fetch race results from database
        results = supabase_client.table('race_results')\
            .select('*, races(*), drivers(*), constructors(*)')\
            .execute()
        
        X = []
        y = []
        driver_ids = []
        
        for result in results.data:
            features = self.extract_features(result, supabase_client)
            X.append(features)
            y.append(1 if result['position'] == 1 else 0)
            driver_ids.append(result['driver_id'])
        
        return np.array(X), np.array(y), driver_ids
    
    def extract_features(self, result: dict, supabase_client) -> list:
        """Extract feature vector for a single race result."""
        features = []
        
        driver_id = result['driver_id']
        race_date = result['races']['date']
        circuit_id = result['races']['circuit_id']
        
        # Calculate rolling averages (last 5 races)
        last_5_races = self.get_last_n_races(
            supabase_client, driver_id, race_date, n=5
        )
        
        features.extend([
            np.mean([r['position'] for r in last_5_races]),
            np.mean([r['points'] for r in last_5_races]),
            np.mean([r['grid_position'] for r in last_5_races]),
            sum(1 for r in last_5_races if r['status'] != 'Finished') / 5,
            sum(1 for r in last_5_races if r['position'] <= 3) / 5
        ])
        
        # Constructor performance
        constructor_id = result['constructor_id']
        constructor_last_5 = self.get_constructor_last_n_races(
            supabase_client, constructor_id, race_date, n=5
        )
        
        features.extend([
            np.mean([r['avg_position'] for r in constructor_last_5]),
            np.mean([r['total_points'] for r in constructor_last_5]),
            sum(1 for r in constructor_last_5 if r['both_finished']) / 5
        ])
        
        # Circuit-specific performance
        circuit_history = self.get_driver_circuit_history(
            supabase_client, driver_id, circuit_id
        )
        
        features.extend([
            np.mean([r['position'] for r in circuit_history]) if circuit_history else 10,
            min([r['position'] for r in circuit_history]) if circuit_history else 20,
            np.mean([r['constructor_avg'] for r in circuit_history]) if circuit_history else 10
        ])
        
        # Championship standings (at time of race)
        standings = self.get_championship_standings_before_race(
            supabase_client, driver_id, race_date
        )
        
        features.extend([
            standings['position'],
            standings['points'],
            standings['constructor_position']
        ])
        
        # Qualifying performance
        features.extend([
            result.get('grid_position', 20),
            result.get('qualifying_gap', 1.0)
        ])
        
        # Circuit characteristics
        circuit = supabase_client.table('circuits')\
            .select('*')\
            .eq('id', circuit_id)\
            .single()\
            .execute()
        
        features.extend([
            circuit.data.get('type_encoded', 0),
            circuit.data.get('length', 5.0),
            circuit.data.get('corners', 15),
            circuit.data.get('overtaking_difficulty', 5)
        ])
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the XGBoost model."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=True
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred_proba)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Log Loss: {logloss:.4f}")
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss
        }
    
    def predict_race(self, race_id: str, supabase_client) -> list:
        """
        Generate predictions for all drivers in a race.
        
        Returns:
            List of dicts with driver_id, win_probability, confidence
        """
        # Fetch all drivers participating in the race
        race_entries = supabase_client.table('race_entries')\
            .select('driver_id')\
            .eq('race_id', race_id)\
            .execute()
        
        predictions = []
        
        for entry in race_entries.data:
            # Extract features for this driver
            features = self.extract_features_for_prediction(
                race_id, entry['driver_id'], supabase_client
            )
            
            # Predict probability
            X = np.array([features])
            win_probability = self.model.predict_proba(X)[0, 1]
            
            predictions.append({
                'driver_id': entry['driver_id'],
                'win_probability': float(win_probability),
                'confidence': self.calculate_confidence(win_probability)
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return predictions
    
    def calculate_confidence(self, probability: float) -> float:
        """
        Calculate confidence score based on probability.
        High confidence if probability is very high or very low.
        """
        return 1 - 2 * abs(probability - 0.5)
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
```

**Model Training Script:**

```python
# scripts/train_race_winner_model.py

import os
from supabase import create_client
from models.race_winner_predictor import RaceWinnerPredictor

def main():
    # Initialize Supabase client
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    # Initialize predictor
    predictor = RaceWinnerPredictor()
    
    # Prepare training data
    print("Preparing training data...")
    X, y, driver_ids = predictor.prepare_training_data(supabase)
    print(f"Training samples: {len(X)}")
    
    # Train model
    print("Training model...")
    metrics = predictor.train(X, y)
    
    # Save model
    model_path = 'models/artifacts/race_winner_v1.pkl'
    predictor.save_model(model_path)
    
    # Upload to Supabase Storage
    with open(model_path, 'rb') as f:
        supabase.storage.from_('ml-models').upload(
            'race_winner_v1.pkl',
            f.read(),
            {'content-type': 'application/octet-stream'}
        )
    
    print("Model training complete!")
    print(f"Metrics: {metrics}")

if __name__ == '__main__':
    main()
```

### 2.3 Tire Degradation Model

**Model Type**: Polynomial Regression + LSTM

**Objective**: Predict lap time degradation based on tire age

**Implementation:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras

class TireDegradationModel:
    """
    Hybrid model for tire degradation prediction.
    - Polynomial Regression for overall trend
    - LSTM for capturing non-linear patterns
    """
    
    def __init__(self):
        self.poly_model = None
        self.lstm_model = None
        self.poly_features = PolynomialFeatures(degree=2)
    
    def prepare_tire_data(self, supabase_client) -> dict:
        """
        Prepare tire degradation data by compound.
        
        Returns:
            Dict with keys: 'SOFT', 'MEDIUM', 'HARD'
            Each contains: tire_ages, lap_times, track_temps
        """
        # Fetch all laps with tire information
        laps = supabase_client.table('lap_data')\
            .select('*')\
            .execute()
        
        data_by_compound = {'SOFT': [], 'MEDIUM': [], 'HARD': []}
        
        for lap in laps.data:
            compound = lap['tire_compound']
            if compound in data_by_compound:
                data_by_compound[compound].append({
                    'tire_age': lap['tire_age'],
                    'lap_time': lap['lap_time_seconds'],
                    'track_temp': lap['track_temperature'],
                    'fuel_load': lap['fuel_load_estimate']
                })
        
        return data_by_compound
    
    def train_polynomial_model(self, tire_ages: np.ndarray, 
                               lap_times: np.ndarray):
        """
        Train polynomial regression model.
        Models: lap_time = a + b*tire_age + c*tire_age^2
        """
        X = tire_ages.reshape(-1, 1)
        X_poly = self.poly_features.fit_transform(X)
        
        self.poly_model = LinearRegression()
        self.poly_model.fit(X_poly, lap_times)
        
        # Calculate R² score
        score = self.poly_model.score(X_poly, lap_times)
        print(f"Polynomial model R² score: {score:.4f}")
        
        return self.poly_model
    
    def build_lstm_model(self, sequence_length: int = 10):
        """
        Build LSTM model for sequence prediction.
        Input: Last N lap times
        Output: Next lap time
        """
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(sequence_length, 1), 
                            return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, lap_times: np.ndarray, 
                        sequence_length: int = 10):
        """Train LSTM on lap time sequences."""
        # Create sequences
        X, y = [], []
        for i in range(len(lap_times) - sequence_length):
            X.append(lap_times[i:i+sequence_length])
            y.append(lap_times[i+sequence_length])
        
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        
        # Split train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Build and train model
        self.lstm_model = self.build_lstm_model(sequence_length)
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict_degradation(self, tire_age: int, 
                           compound: str,
                           recent_lap_times: list = None) -> float:
        """
        Predict lap time for given tire age.
        Uses ensemble of polynomial and LSTM predictions.
        """
        # Polynomial prediction
        X_poly = self.poly_features.transform([[tire_age]])
        poly_pred = self.poly_model.predict(X_poly)[0]
        
        # LSTM prediction (if recent laps available)
        if recent_lap_times and len(recent_lap_times) >= 10:
            X_lstm = np.array(recent_lap_times[-10:]).reshape(1, 10, 1)
            lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)[0, 0]
            
            # Ensemble: weighted average
            prediction = 0.6 * poly_pred + 0.4 * lstm_pred
        else:
            prediction = poly_pred
        
        return float(prediction)
    
    def calculate_optimal_pit_window(self, current_tire_age: int,
                                     compound: str,
                                     pit_stop_time: float = 25.0) -> dict:
        """
        Calculate optimal pit stop lap based on degradation.
        
        Returns:
            optimal_lap: Best lap to pit
            time_saved: Expected time saved by pitting at optimal lap
        """
        # Predict lap times for next 20 laps
        future_lap_times = []
        for age in range(current_tire_age, current_tire_age + 20):
            lap_time = self.predict_degradation(age, compound)
            future_lap_times.append(lap_time)
        
        # Calculate cumulative time loss vs pitting now
        baseline_time = future_lap_times[0]
        cumulative_loss = np.cumsum([t - baseline_time for t in future_lap_times])
        
        # Find when cumulative loss exceeds pit stop time
        optimal_lap_offset = np.argmax(cumulative_loss > pit_stop_time)
        
        if optimal_lap_offset == 0:
            optimal_lap_offset = 10  # Default to 10 laps if no crossover
        
        return {
            'optimal_lap': current_tire_age + optimal_lap_offset,
            'time_saved': float(cumulative_loss[optimal_lap_offset] - pit_stop_time),
            'predicted_lap_times': future_lap_times
        }
```

### 2.4 Model Deployment & Versioning

**Model Storage Structure in Supabase Storage:**

```
ml-models/
├── race_winner/
│   ├── v1.0.0/
│   │   ├── model.pkl
│   │   ├── metadata.json
│   │   └── performance_metrics.json
│   └── latest -> v1.0.0
├── tire_degradation/
│   ├── v1.0.0/
│   │   ├── poly_model.pkl
│   │   ├── lstm_model.h5
│   │   └── metadata.json
│   └── latest -> v1.0.0
└── qualifying_predictor/
    └── ...
```

**Model Metadata Format:**

```json
{
  "model_name": "race_winner_predictor",
  "version": "1.0.0",
  "created_at": "2024-11-07T10:30:00Z",
  "training_data": {
    "start_date": "2023-01-01",
    "end_date": "2024-11-01",
    "num_samples": 1200,
    "num_features": 20
  },
  "performance_metrics": {
    "accuracy": 0.78,
    "log_loss": 0.45,
    "precision": 0.82,
    "recall": 0.75
  },
  "feature_names": [
    "driver_avg_position_last_5",
    "driver_avg_points_last_5",
    ...
  ],
  "hyperparameters": {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200
  }
}
```

---

## 3. Analytics Algorithms

### 3.1 Driver Performance Metrics

**Consistency Score:**

```python
def calculate_consistency_score(lap_times: list) -> float:
    """
    Calculate driver consistency based on lap time variance.
    
    Score = 1 / (1 + coefficient_of_variation)
    Higher score = more consistent
    """
    if len(lap_times) < 5:
        return 0.0
    
    mean_time = np.mean(lap_times)
    std_time = np.std(lap_times)
    
    coefficient_of_variation = std_time / mean_time
    consistency_score = 1 / (1 + coefficient_of_variation)
    
    return consistency_score
```

**Overtaking Ability:**

```python
def calculate_overtaking_metrics(race_results: pd.DataFrame) -> dict:
    """
    Calculate overtaking statistics for a driver.
    
    Returns:
        - total_overtakes: Number of positions gained
        - overtake_success_rate: Percentage of successful overtakes
        - avg_overtakes_per_race: Average positions gained per race
    """
    position_changes = []
    
    for _, race in race_results.iterrows():
        grid_position = race['grid_position']
        final_position = race['position']
        
        if pd.notna(grid_position) and pd.notna(final_position):
            change = grid_position - final_position  # Positive = gained positions
            position_changes.append(change)
    
    total_overtakes = sum(c for c in position_changes if c > 0)
    total_races = len(position_changes)
    
    return {
        'total_overtakes': total_overtakes,
        'avg_overtakes_per_race': total_overtakes / total_races if total_races > 0 else 0,
        'position_changes': position_changes
    }
```

**Pace Analysis:**

```python
def analyze_driver_pace(driver_laps: pd.DataFrame, 
                       session_fastest: float) -> dict:
    """
    Analyze driver pace relative to session fastest.
    
    Returns:
        - avg_gap_to_fastest: Average time gap to fastest lap
        - pace_percentage: Percentage of session fastest pace
        - sector_strengths: Best performing sectors
    """
    driver_fastest = driver_laps['LapTime'].min().total_seconds()
    avg_lap_time = driver_laps['LapTime'].mean().total_seconds()
    
    gap_to_fastest = driver_fastest - session_fastest
    pace_percentage = (session_fastest / driver_fastest) * 100
    
    # Sector analysis
    sector_gaps = {
        'sector_1': driver_laps['Sector1Time'].min().total_seconds(),
        'sector_2': driver_laps['Sector2Time'].min().total_seconds(),
        'sector_3': driver_laps['Sector3Time'].min().total_seconds()
    }
    
    return {
        'fastest_lap': driver_fastest,
        'avg_lap_time': avg_lap_time,
        'gap_to_fastest': gap_to_fastest,
        'pace_percentage': pace_percentage,
        'sector_times': sector_gaps
    }
```

### 3.2 Constructor Performance Analysis

```python
class ConstructorAnalyzer:
    """Analyze constructor (team) performance."""
    
    @staticmethod
    def calculate_constructor_points(race_results: pd.DataFrame) -> dict:
        """Calculate total points and average position for constructor."""
        total_points = race_results['points'].sum()
        avg_position = race_results['position'].mean()
        
        # Reliability: percentage of races both cars finished
        both_finished = race_results.groupby('race_id').apply(
            lambda x: (x['status'] == 'Finished').all()
        ).mean()
        
        return {
            'total_points': total_points,
            'avg_position': avg_position,
            'reliability_rate': both_finished,
            'races_completed': len(race_results)
        }
    
    @staticmethod
    def compare_constructors(constructor1_data: dict, 
                            constructor2_data: dict) -> dict:
        """Compare two constructors across multiple metrics."""
        comparison = {
            'points_difference': constructor1_data['total_points'] - constructor2_data['total_points'],
            'position_difference': constructor1_data['avg_position'] - constructor2_data['avg_position'],
            'reliability_difference': constructor1_data['reliability_rate'] - constructor2_data['reliability_rate']
        }
        
        # Determine which is better
        comparison['better_team'] = 1 if comparison['points_difference'] > 0 else 2
        
        return comparison
```

### 3.3 Circuit Analysis

```python
def analyze_circuit_characteristics(circuit_id: str, 
                                   historical_races: pd.DataFrame) -> dict:
    """
    Analyze circuit characteristics based on historical data.
    
    Returns:
        - avg_winning_margin: Average time gap between P1 and P2
        - overtaking_frequency: Average overtakes per race
        - safety_car_probability: Likelihood of safety car
        - tire_strategy_distribution: Most common strategies
    """
    circuit_races = historical_races[historical_races['circuit_id'] == circuit_id]
    
    # Winning margin
    winning_margins = []
    for race_id in circuit_races['race_id'].unique():
        race_results = circuit_races[circuit_races['race_id'] == race_id]
        if len(race_results) >= 2:
            p1_time = race_results[race_results['position'] == 1]['total_time'].iloc[0]
            p2_time = race_results[race_results['position'] == 2]['total_time'].iloc[0]
            winning_margins.append(p2_time - p1_time)
    
    avg_winning_margin = np.mean(winning_margins) if winning_margins else 0
    
    # Overtaking frequency (position changes from grid to finish)
    overtakes = []
    for race_id in circuit_races['race_id'].unique():
        race_results = circuit_races[circuit_races['race_id'] == race_id]
        total_position_changes = abs(
            race_results['grid_position'] - race_results['position']
        ).sum()
        overtakes.append(total_position_changes)
    
    avg_overtakes = np.mean(overtakes) if overtakes else 0
    
    # Safety car probability
    safety_car_races = circuit_races['had_safety_car'].sum()
    safety_car_prob = safety_car_races / len(circuit_races) if len(circuit_races) > 0 else 0
    
    return {
        'avg_winning_margin': avg_winning_margin,
        'avg_overtakes': avg_overtakes,
        'safety_car_probability': safety_car_prob,
        'total_races_analyzed': len(circuit_races)
    }
```

---

## 4. Strategy Simulator Implementation

### 4.1 Monte Carlo Race Simulator

**Core Algorithm:**

```python
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TireCompound:
    name: str
    base_lap_time: float  # seconds
    degradation_rate: float  # seconds per lap
    optimal_stint_length: int  # laps

@dataclass
class RaceConfig:
    total_laps: int
    pit_stop_time: float  # seconds
    compounds: List[TireCompound]
    safety_car_probability: float  # per lap

class MonteCarloRaceSimulator:
    """
    Simulate F1 race strategies using Monte Carlo method.
    Runs thousands of simulations to evaluate strategy outcomes.
    """
    
    def __init__(self, race_config: RaceConfig):
        self.config = race_config
    
    def simulate_stint(self, compound: TireCompound, 
                      stint_length: int,
                      start_lap: int) -> dict:
        """
        Simulate a single tire stint.
        
        Returns:
            stint_time: Total time for stint
            lap_times: List of individual lap times
            safety_car_laps: Laps with safety car
        """
        stint_time = 0
        lap_times = []
        safety_car_laps = []
        
        for lap in range(stint_length):
            tire_age = lap + 1
            
            # Base lap time + degradation
            degradation = compound.degradation_rate * (tire_age ** 1.5)
            lap_time = compound.base_lap_time + degradation
            
            # Add random variance (±0.2 seconds)
            lap_time += random.gauss(0, 0.1)
            
            # Check for safety car
            if random.random() < self.config.safety_car_probability:
                safety_car_laps.append(start_lap + lap)
                lap_time *= 1.3  # Slower under safety car
            
            lap_times.append(lap_time)
            stint_time += lap_time
        
        return {
            'stint_time': stint_time,
            'lap_times': lap_times,
            'safety_car_laps': safety_car_laps
        }
    
    def simulate_strategy(self, strategy: List[Tuple[str, int]], 
                         n_simulations: int = 1000) -> dict:
        """
        Evaluate a complete race strategy through Monte Carlo simulation.
        
        Args:
            strategy: List of (compound_name, stint_length) tuples
            n_simulations: Number of simulation runs
        
        Returns:
            Statistics: mean_time, std_dev, best_time, worst_time, 
                       distribution, safety_car_impact
        """
        race_times = []
        safety_car_impacts = []
        
        for _ in range(n_simulations):
            total_time = 0
            current_lap = 0
            had_safety_car = False
            
            for stint_idx, (compound_name, stint_length) in enumerate(strategy):
                # Get compound object
                compound = next(c for c in self.config.compounds 
                              if c.name == compound_name)
                
                # Simulate stint
                stint_result = self.simulate_stint(
                    compound, stint_length, current_lap
                )
                
                total_time += stint_result['stint_time']
                current_lap += stint_length
                
                # Add pit stop time (except after final stint)
                if stint_idx < len(strategy) - 1:
                    # Pit stop time varies (log-logistic distribution)
                    pit_time = random.lognormvariate(
                        np.log(self.config.pit_stop_time), 0.1
                    )
                    total_time += pit_time
                
                # Track safety car occurrence
                if stint_result['safety_car_laps']:
                    had_safety_car = True
            
            race_times.append(total_time)
            safety_car_impacts.append(1 if had_safety_car else 0)
        
        # Calculate statistics
        race_times = np.array(race_times)
        
        return {
            'mean_time': float(np.mean(race_times)),
            'std_dev': float(np.std(race_times)),
            'best_time': float(np.min(race_times)),
            'worst_time': float(np.max(race_times)),
            'median_time': float(np.median(race_times)),
            'percentile_25': float(np.percentile(race_times, 25)),
            'percentile_75': float(np.percentile(race_times, 75)),
            'safety_car_frequency': float(np.mean(safety_car_impacts)),
            'distribution': race_times.tolist()
        }
    
    def optimize_strategy(self, 
                         candidate_strategies: List[List[Tuple[str, int]]],
                         n_simulations: int = 1000) -> pd.DataFrame:
        """
        Compare multiple strategies and rank by performance.
        
        Returns:
            DataFrame with strategy rankings, mean times, risk scores
        """
        results = []
        
        for idx, strategy in enumerate(candidate_strategies):
            print(f"Simulating strategy {idx + 1}/{len(candidate_strategies)}...")
            
            stats = self.simulate_strategy(strategy, n_simulations)
            
            # Calculate risk score (lower is better)
            risk_score = stats['std_dev'] / stats['mean_time']
            
            results.append({
                'strategy_id': idx + 1,
                'strategy': strategy,
                'mean_time': stats['mean_time'],
                'std_dev': stats['std_dev'],
                'best_time': stats['best_time'],
                'worst_time': stats['worst_time'],
                'risk_score': risk_score,
                'safety_car_frequency': stats['safety_car_frequency']
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('mean_time')
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def simulate_live_decision(self, 
                              current_lap: int,
                              current_tire_age: int,
                              current_compound: str,
                              laps_remaining: int) -> dict:
        """
        Simulate pit stop decision during live race.
        
        Returns:
            recommendation: 'PIT_NOW' or 'CONTINUE'
            expected_gain: Time gained/lost by pitting
            confidence: Confidence in recommendation
        """
        # Strategy 1: Pit now
        strategy_pit_now = [
            (current_compound, current_tire_age),
            ('MEDIUM', laps_remaining)  # Assume medium for remaining laps
        ]
        
        # Strategy 2: Continue 5 more laps, then pit
        strategy_continue = [
            (current_compound, current_tire_age + 5),
            ('MEDIUM', laps_remaining - 5)
        ]
        
        # Simulate both strategies
        result_pit_now = self.simulate_strategy(strategy_pit_now, n_simulations=500)
        result_continue = self.simulate_strategy(strategy_continue, n_simulations=500)
        
        # Compare expected times
        time_difference = result_continue['mean_time'] - result_pit_now['mean_time']
        
        if time_difference > 0:
            recommendation = 'PIT_NOW'
            expected_gain = time_difference
        else:
            recommendation = 'CONTINUE'
            expected_gain = -time_difference
        
        # Calculate confidence based on overlap of distributions
        overlap = self.calculate_distribution_overlap(
            result_pit_now['distribution'],
            result_continue['distribution']
        )
        confidence = 1 - overlap
        
        return {
            'recommendation': recommendation,
            'expected_gain': expected_gain,
            'confidence': confidence,
            'pit_now_mean': result_pit_now['mean_time'],
            'continue_mean': result_continue['mean_time']
        }
    
    @staticmethod
    def calculate_distribution_overlap(dist1: list, dist2: list) -> float:
        """Calculate overlap between two distributions."""
        # Use KDE to estimate probability density functions
        from scipy.stats import gaussian_kde
        
        kde1 = gaussian_kde(dist1)
        kde2 = gaussian_kde(dist2)
        
        # Calculate overlap integral
        x_min = min(min(dist1), min(dist2))
        x_max = max(max(dist1), max(dist2))
        x = np.linspace(x_min, x_max, 1000)
        
        overlap = np.trapz(np.minimum(kde1(x), kde2(x)), x)
        
        return overlap
```

### 4.2 Strategy Simulator API

**FastAPI Endpoint:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

app = FastAPI()

class StrategyRequest(BaseModel):
    race_id: str
    strategies: List[List[Tuple[str, int]]]
    n_simulations: int = 1000

class LiveDecisionRequest(BaseModel):
    race_id: str
    current_lap: int
    current_tire_age: int
    current_compound: str

@app.post("/api/strategy/simulate")
async def simulate_strategies(request: StrategyRequest):
    """
    Simulate and compare multiple race strategies.
    """
    try:
        # Fetch race configuration
        race_config = get_race_config(request.race_id)
        
        # Initialize simulator
        simulator = MonteCarloRaceSimulator(race_config)
        
        # Run simulations
        results = simulator.optimize_strategy(
            request.strategies,
            request.n_simulations
        )
        
        return {
            'success': True,
            'results': results.to_dict(orient='records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/live-decision")
async def get_live_decision(request: LiveDecisionRequest):
    """
    Get real-time pit stop recommendation during race.
    """
    try:
        race_config = get_race_config(request.race_id)
        simulator = MonteCarloRaceSimulator(race_config)
        
        laps_remaining = race_config.total_laps - request.current_lap
        
        decision = simulator.simulate_live_decision(
            request.current_lap,
            request.current_tire_age,
            request.current_compound,
            laps_remaining
        )
        
        return {
            'success': True,
            'decision': decision
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4.3 Frontend Integration

**Strategy Simulator Component:**

```typescript
// src/components/strategy-simulator/StrategySimulator.tsx

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select } from '@/components/ui/select';

interface Stint {
  compound: 'SOFT' | 'MEDIUM' | 'HARD';
  laps: number;
}

interface SimulationResult {
  strategy_id: number;
  mean_time: number;
  std_dev: number;
  rank: number;
  risk_score: number;
}

export function StrategySimulator({ raceId }: { raceId: string }) {
  const [stints, setStints] = useState<Stint[]>([
    { compound: 'MEDIUM', laps: 25 },
    { compound: 'HARD', laps: 30 }
  ]);
  
  const [results, setResults] = useState<SimulationResult[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  
  const addStint = () => {
    setStints([...stints, { compound: 'MEDIUM', laps: 10 }]);
  };
  
  const updateStint = (index: number, field: keyof Stint, value: any) => {
    const newStints = [...stints];
    newStints[index][field] = value;
    setStints(newStints);
  };
  
  const removeStint = (index: number) => {
    setStints(stints.filter((_, i) => i !== index));
  };
  
  const runSimulation = async () => {
    setIsSimulating(true);
    
    try {
      // Convert stints to strategy format
      const strategy = stints.map(s => [s.compound, s.laps]);
      
      const response = await fetch('/api/strategy/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          race_id: raceId,
          strategies: [strategy],  // Can add multiple strategies
          n_simulations: 1000
        })
      });
      
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Simulation failed:', error);
    } finally {
      setIsSimulating(false);
    }
  };
  
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Strategy Builder</h2>
        
        {/* Stint Editor */}
        <div className="space-y-4">
          {stints.map((stint, index) => (
            <div key={index} className="flex gap-4 items-center">
              <span className="font-semibold">Stint {index + 1}:</span>
              
              <Select
                value={stint.compound}
                onChange={(e) => updateStint(index, 'compound', e.target.value)}
              >
                <option value="SOFT">Soft</option>
                <option value="MEDIUM">Medium</option>
                <option value="HARD">Hard</option>
              </Select>
              
              <input
                type="number"
                value={stint.laps}
                onChange={(e) => updateStint(index, 'laps', parseInt(e.target.value))}
                className="w-20 px-2 py-1 border rounded"
                min={1}
                max={60}
              />
              <span>laps</span>
              
              <Button
                variant="destructive"
                size="sm"
                onClick={() => removeStint(index)}
              >
                Remove
              </Button>
            </div>
          ))}
          
          <Button onClick={addStint}>Add Stint</Button>
        </div>
        
        {/* Simulate Button */}
        <Button
          onClick={runSimulation}
          disabled={isSimulating}
          className="mt-6"
        >
          {isSimulating ? 'Simulating...' : 'Run Simulation'}
        </Button>
      </Card>
      
      {/* Results */}
      {results.length > 0 && (
        <Card className="p-6">
          <h2 className="text-2xl font-bold mb-4">Simulation Results</h2>
          
          <div className="space-y-4">
            {results.map((result) => (
              <div key={result.strategy_id} className="border-b pb-4">
                <div className="flex justify-between">
                  <span className="font-semibold">
                    Strategy #{result.strategy_id} (Rank: {result.rank})
                  </span>
                  <span>Mean Time: {result.mean_time.toFixed(2)}s</span>
                </div>
                
                <div className="text-sm text-gray-600 mt-2">
                  <span>Std Dev: ±{result.std_dev.toFixed(2)}s</span>
                  <span className="ml-4">Risk Score: {result.risk_score.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
```

---

## 5. Intelligent Agent Design

### 5.1 Agent Architecture Overview

The intelligent agent system consists of three autonomous agents:

1. **Data Sync Agent**: Monitors and updates F1 data automatically
2. **Model Retraining Agent**: Retrains ML models after each race
3. **Prediction Generation Agent**: Generates predictions for upcoming races

### 5.2 Data Sync Agent

**Implementation:**

```python
import schedule
import time
from datetime import datetime, timedelta

class DataSyncAgent:
    """
    Autonomous agent for keeping F1 data up-to-date.
    Runs as a background service, checking for updates periodically.
    """
    
    def __init__(self, supabase_client, fastf1_service):
        self.supabase = supabase_client
        self.fastf1 = fastf1_service
        self.last_sync = None
    
    def check_for_updates(self):
        """
        Check if there are any updates needed.
        
        Checks:
        1. New races completed
        2. Driver lineup changes
        3. Constructor name changes
        4. Schedule updates
        """
        print(f"[{datetime.now()}] Data Sync Agent: Checking for updates...")
        
        updates_needed = []
        
        # Check for new race results
        if self.has_new_race_results():
            updates_needed.append('race_results')
        
        # Check for driver changes
        if self.has_driver_changes():
            updates_needed.append('drivers')
        
        # Check for schedule updates
        if self.has_schedule_updates():
            updates_needed.append('schedule')
        
        if updates_needed:
            print(f"Updates needed: {updates_needed}")
            self.perform_updates(updates_needed)
        else:
            print("No updates needed.")
    
    def has_new_race_results(self) -> bool:
        """Check if there are new race results to fetch."""
        # Get latest race in database
        latest_race = self.supabase.table('races')\
            .select('date')\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        
        if not latest_race.data:
            return True  # No races in DB, need to fetch
        
        latest_date = datetime.fromisoformat(latest_race.data[0]['date'])
        
        # Check if there's a race that happened after latest_date
        current_year = datetime.now().year
        schedule = fastf1.get_event_schedule(current_year)
        
        for _, event in schedule.iterrows():
            event_date = event['EventDate']
            if event_date > latest_date and event_date < datetime.now():
                # Race happened but not in database
                return True
        
        return False
    
    def has_driver_changes(self) -> bool:
        """Check if any drivers have changed teams."""
        # Fetch current driver lineups from FastF1
        current_year = datetime.now().year
        session = fastf1.get_session(current_year, 1, 'R')
        session.load()
        
        current_drivers = session.results[['DriverNumber', 'TeamName']].to_dict('records')
        
        # Compare with database
        for driver in current_drivers:
            db_driver = self.supabase.table('drivers')\
                .select('*, constructors(name)')\
                .eq('permanent_number', driver['DriverNumber'])\
                .single()\
                .execute()
            
            if db_driver.data:
                if db_driver.data['constructors']['name'] != driver['TeamName']:
                    return True  # Driver changed team
        
        return False
    
    def has_schedule_updates(self) -> bool:
        """Check if race schedule has been updated."""
        current_year = datetime.now().year
        
        # Fetch schedule from FastF1
        schedule = fastf1.get_event_schedule(current_year)
        
        # Compare with database
        db_races = self.supabase.table('races')\
            .select('round, date')\
            .eq('season_year', current_year)\
            .execute()
        
        if len(db_races.data) != len(schedule):
            return True  # Number of races changed
        
        # Check for date changes
        for _, event in schedule.iterrows():
            db_race = next(
                (r for r in db_races.data if r['round'] == event['RoundNumber']),
                None
            )
            
            if db_race:
                if db_race['date'] != event['EventDate'].date().isoformat():
                    return True  # Race date changed
        
        return False
    
    def perform_updates(self, update_types: list):
        """Perform the necessary updates."""
        for update_type in update_types:
            try:
                if update_type == 'race_results':
                    self.update_race_results()
                elif update_type == 'drivers':
                    self.update_drivers()
                elif update_type == 'schedule':
                    self.update_schedule()
                
                print(f"✓ Updated: {update_type}")
            
            except Exception as e:
                print(f"✗ Failed to update {update_type}: {e}")
        
        # Update last sync timestamp
        self.last_sync = datetime.now()
        self.supabase.table('system_metadata')\
            .upsert({'key': 'last_sync', 'value': self.last_sync.isoformat()})\
            .execute()
    
    def update_race_results(self):
        """Fetch and insert new race results."""
        # Implementation similar to initial data load
        pass
    
    def update_drivers(self):
        """Update driver information and team changes."""
        pass
    
    def update_schedule(self):
        """Update race schedule."""
        pass
    
    def run(self):
        """
        Run the agent continuously.
        Checks for updates every hour.
        """
        # Schedule checks
        schedule.every().hour.do(self.check_for_updates)
        
        # Also check immediately on startup
        self.check_for_updates()
        
        print("Data Sync Agent started. Running checks every hour...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute if scheduled task is due
```

### 5.3 Model Retraining Agent

```python
class ModelRetrainingAgent:
    """
    Autonomous agent for retraining ML models after each race.
    Triggered by database hooks when new race results are inserted.
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.models = {
            'race_winner': RaceWinnerPredictor(),
            'tire_degradation': TireDegradationModel()
        }
    
    def on_race_completed(self, race_id: str):
        """
        Triggered when a race is completed and results are inserted.
        
        Workflow:
        1. Validate new data
        2. Retrain models
        3. Evaluate new models
        4. Compare with current models (drift detection)
        5. Deploy if improved
        6. Generate predictions for next race
        """
        print(f"[{datetime.now()}] Model Retraining Agent: Race {race_id} completed")
        
        try:
            # Step 1: Validate data
            if not self.validate_race_data(race_id):
                print("Data validation failed. Skipping retraining.")
                return
            
            # Step 2: Retrain models
            print("Retraining models...")
            new_models = self.retrain_all_models()
            
            # Step 3: Evaluate new models
            print("Evaluating new models...")
            new_metrics = self.evaluate_models(new_models)
            
            # Step 4: Compare with current models (drift detection)
            print("Comparing with current models...")
            should_deploy = self.should_deploy_new_models(new_metrics)
            
            if should_deploy:
                # Step 5: Deploy new models
                print("Deploying new models...")
                self.deploy_models(new_models, new_metrics)
                
                # Step 6: Generate predictions for next race
                print("Generating predictions for next race...")
                self.generate_next_race_predictions()
            else:
                print("New models did not improve. Keeping current models.")
        
        except Exception as e:
            print(f"Error in model retraining: {e}")
            # Log error to monitoring system
            self.log_error(e)
    
    def validate_race_data(self, race_id: str) -> bool:
        """Validate that race data is complete and correct."""
        race_results = self.supabase.table('race_results')\
            .select('*')\
            .eq('race_id', race_id)\
            .execute()
        
        # Check if we have results for at least 15 drivers
        if len(race_results.data) < 15:
            return False
        
        # Check if positions are valid (1-20)
        positions = [r['position'] for r in race_results.data]
        if any(p < 1 or p > 20 for p in positions if p is not None):
            return False
        
        return True
    
    def retrain_all_models(self) -> dict:
        """Retrain all ML models with updated data."""
        new_models = {}
        
        # Retrain race winner predictor
        print("  Retraining race winner predictor...")
        X, y, _ = self.models['race_winner'].prepare_training_data(self.supabase)
        self.models['race_winner'].train(X, y)
        new_models['race_winner'] = self.models['race_winner']
        
        # Retrain tire degradation model
        print("  Retraining tire degradation model...")
        tire_data = self.models['tire_degradation'].prepare_tire_data(self.supabase)
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            if compound in tire_data and tire_data[compound]:
                self.models['tire_degradation'].train_polynomial_model(
                    tire_data[compound]['tire_ages'],
                    tire_data[compound]['lap_times']
                )
        new_models['tire_degradation'] = self.models['tire_degradation']
        
        return new_models
    
    def evaluate_models(self, models: dict) -> dict:
        """Evaluate model performance on hold-out dataset."""
        metrics = {}
        
        # Evaluate race winner predictor
        # Use last 3 races as test set
        test_races = self.get_last_n_races(3)
        
        predictions = []
        actuals = []
        
        for race in test_races:
            race_predictions = models['race_winner'].predict_race(
                race['id'], self.supabase
            )
            
            # Get actual winner
            actual_winner = self.get_race_winner(race['id'])
            
            # Check if predicted winner matches actual
            predicted_winner = race_predictions[0]['driver_id']
            predictions.append(predicted_winner)
            actuals.append(actual_winner)
        
        accuracy = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
        
        metrics['race_winner'] = {
            'accuracy': accuracy,
            'test_races': len(test_races)
        }
        
        return metrics
    
    def should_deploy_new_models(self, new_metrics: dict) -> bool:
        """
        Determine if new models should be deployed.
        Uses drift detection to compare with current models.
        """
        # Fetch current model metrics
        current_metrics = self.supabase.table('model_metadata')\
            .select('metrics')\
            .eq('status', 'deployed')\
            .execute()
        
        if not current_metrics.data:
            return True  # No current model, deploy new one
        
        current_accuracy = current_metrics.data[0]['metrics']['race_winner']['accuracy']
        new_accuracy = new_metrics['race_winner']['accuracy']
        
        # Deploy if new model is at least 2% better
        improvement = new_accuracy - current_accuracy
        
        if improvement > 0.02:
            print(f"Model improved by {improvement:.2%}. Deploying.")
            return True
        elif improvement < -0.05:
            print(f"Model degraded by {-improvement:.2%}. Not deploying.")
            return False
        else:
            print(f"Model change: {improvement:.2%}. Not significant.")
            return False
    
    def deploy_models(self, models: dict, metrics: dict):
        """Deploy new models to production."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"v{timestamp}"
        
        for model_name, model in models.items():
            # Save model artifact
            model_path = f"models/artifacts/{model_name}_{version}.pkl"
            model.save_model(model_path)
            
            # Upload to Supabase Storage
            with open(model_path, 'rb') as f:
                self.supabase.storage.from_('ml-models').upload(
                    f"{model_name}/{version}/model.pkl",
                    f.read(),
                    {'content-type': 'application/octet-stream'}
                )
            
            # Update model metadata
            metadata = {
                'model_name': model_name,
                'version': version,
                'status': 'deployed',
                'metrics': metrics.get(model_name, {}),
                'deployed_at': datetime.now().isoformat()
            }
            
            self.supabase.table('model_metadata').insert(metadata).execute()
            
            print(f"✓ Deployed {model_name} {version}")
    
    def generate_next_race_predictions(self):
        """Generate predictions for the next upcoming race."""
        # Get next race
        next_race = self.supabase.table('races')\
            .select('*')\
            .gt('date', datetime.now().date().isoformat())\
            .order('date')\
            .limit(1)\
            .execute()
        
        if not next_race.data:
            print("No upcoming races to predict.")
            return
        
        race_id = next_race.data[0]['id']
        
        # Generate predictions
        predictions = self.models['race_winner'].predict_race(race_id, self.supabase)
        
        # Insert into database
        for pred in predictions:
            prediction_data = {
                'race_id': race_id,
                'driver_id': pred['driver_id'],
                'prediction_type': 'RACE_WINNER',
                'predicted_position': 1 if pred == predictions[0] else None,
                'confidence': pred['confidence'],
                'model_version': 'latest',
                'created_at': datetime.now().isoformat()
            }
            
            self.supabase.table('predictions').insert(prediction_data).execute()
        
        print(f"✓ Generated predictions for race {race_id}")
    
    def log_error(self, error: Exception):
        """Log errors to monitoring system."""
        self.supabase.table('agent_logs').insert({
            'agent_name': 'ModelRetrainingAgent',
            'level': 'ERROR',
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }).execute()
```

### 5.4 Agent Deployment

**Docker Container for Agents:**

```dockerfile
# Dockerfile for Python agents

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY python-services/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run agents
CMD ["python", "agents/main.py"]
```

**Agent Orchestration:**

```python
# agents/main.py

import os
import threading
from data_sync_agent import DataSyncAgent
from model_retraining_agent import ModelRetrainingAgent
from supabase import create_client

def main():
    # Initialize Supabase client
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    # Initialize agents
    data_sync_agent = DataSyncAgent(supabase, fastf1_service=None)
    model_retraining_agent = ModelRetrainingAgent(supabase)
    
    # Run Data Sync Agent in separate thread
    sync_thread = threading.Thread(target=data_sync_agent.run, daemon=True)
    sync_thread.start()
    
    # Listen for race completion events (Supabase Realtime)
    def on_race_completed(payload):
        race_id = payload['record']['id']
        model_retraining_agent.on_race_completed(race_id)
    
    supabase.realtime.channel('race_results')\
        .on('INSERT', on_race_completed)\
        .subscribe()
    
    print("Agents started successfully.")
    
    # Keep main thread alive
    sync_thread.join()

if __name__ == '__main__':
    main()
```

---

## 6. Implementation Roadmap

### Phase 1: Data Foundation (Week 1)
**Priority: P0 - Blocking**

**Days 1-2: Environment Setup**
- [ ] Set up Python environment in project
- [ ] Install FastF1, pandas, numpy, scikit-learn, xgboost
- [ ] Configure FastF1 caching
- [ ] Test FastF1 API access

**Days 3-4: Data Ingestion Service**
- [ ] Build DataIngestionService class
- [ ] Implement fetch functions for seasons, races, laps
- [ ] Create data transformation pipeline
- [ ] Test with 2023-2024 seasons

**Days 5-6: Database Population**
- [ ] Run initial data load script
- [ ] Verify data integrity
- [ ] Create indexes for query optimization
- [ ] Test data retrieval from frontend

**Day 7: Automatic Sync**
- [ ] Implement `initialize_app_data` Edge Function
- [ ] Integrate with frontend app startup
- [ ] Test automatic data initialization
- [ ] Add loading states and error handling

**Deliverables:**
- ✅ Database populated with 2023-2024 F1 data
- ✅ Dashboard shows real standings and race results
- ✅ Automatic data sync on app startup

---

### Phase 2: ML Models (Week 2)
**Priority: P0 - Core Value**

**Days 1-3: Race Winner Prediction**
- [ ] Implement feature extraction pipeline
- [ ] Build RaceWinnerPredictor class
- [ ] Train XGBoost model on historical data
- [ ] Evaluate model performance (target: >70% accuracy)
- [ ] Save model to Supabase Storage

**Days 4-5: Tire Degradation Model**
- [ ] Implement TireDegradationModel class
- [ ] Train polynomial regression models by compound
- [ ] Build LSTM for sequence prediction
- [ ] Test degradation predictions

**Days 6-7: Prediction Generation & UI**
- [ ] Create prediction generation script
- [ ] Generate predictions for upcoming races
- [ ] Build Predictions Tab UI component
- [ ] Display predictions with confidence scores
- [ ] Add charts for probability visualization

**Deliverables:**
- ✅ Real ML models generating predictions
- ✅ Predictions Tab showing race winner probabilities
- ✅ Model artifacts stored and versioned

---

### Phase 3: Strategy Simulator (Week 3)
**Priority: P0 - Unique Feature**

**Days 1-3: Monte Carlo Simulator**
- [ ] Implement MonteCarloRaceSimulator class
- [ ] Build stint simulation logic
- [ ] Add tire degradation and safety car modeling
- [ ] Test with sample strategies

**Days 4-5: Strategy API**
- [ ] Create FastAPI endpoints for simulation
- [ ] Implement strategy optimization algorithm
- [ ] Add live decision recommendation
- [ ] Test API with Postman

**Days 6-7: Frontend UI**
- [ ] Build StrategySimulator component
- [ ] Create stint editor interface
- [ ] Add results visualization
- [ ] Integrate with backend API

**Deliverables:**
- ✅ Working Strategy Simulator with Monte Carlo
- ✅ Users can create and test pit strategies
- ✅ Visual results showing time distributions

---

### Phase 4: Drivers Analytics (Week 4)
**Priority: P1 - Enhanced Engagement**

**Days 1-2: Performance Metrics**
- [ ] Implement consistency score calculation
- [ ] Build overtaking metrics analyzer
- [ ] Create pace analysis functions
- [ ] Calculate metrics for all drivers

**Days 3-4: Drivers Tab UI**
- [ ] Create `/drivers` route and page
- [ ] Build driver list component
- [ ] Create individual driver detail pages
- [ ] Add performance charts

**Days 5-6: Head-to-Head Comparison**
- [ ] Implement comparison algorithm
- [ ] Build comparison UI component
- [ ] Add telemetry visualization
- [ ] Create sector analysis charts

**Day 7: Testing & Polish**
- [ ] Test all driver features
- [ ] Optimize query performance
- [ ] Add loading states
- [ ] Fix bugs

**Deliverables:**
- ✅ Drivers Tab with comprehensive analytics
- ✅ Head-to-head comparison tool
- ✅ Performance metrics for all drivers

---

### Phase 5: Live Race Features (Week 5)
**Priority: P1 - Real-time Engagement**

**Days 1-2: Live Data Streaming**
- [ ] Implement LiveDataStreamer class
- [ ] Set up Supabase Realtime channels
- [ ] Test live data broadcasting
- [ ] Add error handling and reconnection

**Days 3-4: Live Race View UI**
- [ ] Create `/live-race` route and page
- [ ] Build real-time leaderboard component
- [ ] Add track map visualization
- [ ] Create event feed component

**Days 5-6: Live Predictions**
- [ ] Implement real-time prediction updates
- [ ] Add battle forecast feature
- [ ] Create championship impact calculator
- [ ] Build live predictions panel

**Day 7: Integration & Testing**
- [ ] Test full live race flow
- [ ] Optimize WebSocket performance
- [ ] Add fallback for connection issues
- [ ] Polish UI/UX

**Deliverables:**
- ✅ Live Race View with real-time updates
- ✅ Live predictions during races
- ✅ Interactive track map

---

### Phase 6: Agent System (Week 6)
**Priority: P1 - Automation**

**Days 1-2: Data Sync Agent**
- [ ] Implement DataSyncAgent class
- [ ] Add update detection logic
- [ ] Create scheduled job (pg_cron)
- [ ] Test automatic updates

**Days 3-4: Model Retraining Agent**
- [ ] Implement ModelRetrainingAgent class
- [ ] Add drift detection algorithm
- [ ] Create database triggers
- [ ] Test post-race retraining

**Days 5-6: Deployment**
- [ ] Create Docker container for agents
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Set up monitoring and logging
- [ ] Configure alerts

**Day 7: Testing & Monitoring**
- [ ] Test full agent workflow
- [ ] Verify automatic data updates
- [ ] Check model retraining after race
- [ ] Monitor agent logs

**Deliverables:**
- ✅ Autonomous agents running in production
- ✅ Automatic data sync without manual intervention
- ✅ Self-improving ML models

---

## 7. Technical Stack

### 7.1 Backend

**Python Services:**
- **FastF1**: 3.3.0 - F1 data access
- **Pandas**: 2.1.4 - Data manipulation
- **NumPy**: 1.26.2 - Numerical computing
- **Scikit-learn**: 1.3.2 - ML algorithms
- **XGBoost**: 2.0.3 - Gradient boosting
- **TensorFlow**: 2.15.0 - Deep learning (LSTM)
- **FastAPI**: 0.104.1 - API framework
- **Supabase-py**: 2.3.0 - Database client
- **Schedule**: 1.2.0 - Task scheduling

**Supabase:**
- **PostgreSQL**: Database
- **Auth**: User authentication
- **Realtime**: Live data streaming
- **Edge Functions**: Serverless compute (Deno)
- **Storage**: Model artifacts

### 7.2 Frontend

**Core:**
- **Next.js**: 14.0.4 - React framework
- **TypeScript**: 5.3.3 - Type safety
- **React**: 18.2.0 - UI library

**UI Components:**
- **Shadcn-ui**: Component library
- **Tailwind CSS**: 3.4.0 - Styling
- **Radix UI**: Headless components
- **Lucide React**: Icons

**Data Visualization:**
- **Recharts**: 2.10.0 - Charts
- **D3.js**: 7.8.5 - Advanced visualizations
- **React-Plotly**: Interactive plots

**State Management:**
- **Zustand**: 4.4.7 - Global state
- **React Query**: 5.12.0 - Server state

### 7.3 Infrastructure

**Deployment:**
- **Vercel**: Frontend hosting
- **AWS/GCP**: Python services (Docker containers)
- **Supabase Cloud**: Backend services

**Monitoring:**
- **Sentry**: Error tracking
- **Datadog**: Performance monitoring
- **Supabase Logs**: Built-in logging

**CI/CD:**
- **GitHub Actions**: Automated testing and deployment
- **Docker**: Containerization
- **Terraform**: Infrastructure as code (optional)

---

## 8. Deployment Architecture

### 8.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Vercel)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Next.js App                                              │  │
│  │  - Dashboard                                              │  │
│  │  - Predictions Tab                                        │  │
│  │  - Strategy Simulator                                     │  │
│  │  - Drivers Tab                                            │  │
│  │  - Live Race View                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Supabase Backend                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PostgreSQL  │  │   Realtime   │  │     Auth     │         │
│  │   Database   │  │   Channels   │  │   (JWT)      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Edge Functions│  │   Storage    │  │   pg_cron    │         │
│  │  (Deno)      │  │ (ML Models)  │  │  (Scheduler) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ API Calls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Python Services (AWS/GCP)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Ingestion Service                                   │  │
│  │  - FastF1 integration                                     │  │
│  │  - Historical data fetching                               │  │
│  │  - Live data streaming                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ML Training Service                                      │  │
│  │  - Model training                                         │  │
│  │  - Model evaluation                                       │  │
│  │  - Model deployment                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Strategy Simulation API (FastAPI)                        │  │
│  │  - Monte Carlo simulator                                  │  │
│  │  - Strategy optimization                                  │  │
│  │  - Live decision recommendations                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Intelligent Agents                                       │  │
│  │  - Data Sync Agent                                        │  │
│  │  - Model Retraining Agent                                 │  │
│  │  - Prediction Generation Agent                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ API Calls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External APIs                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   FastF1     │  │  Ergast API  │  │  OpenAI API  │         │
│  │     API      │  │ (Historical) │  │  (AI Chat)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Data Flow

**1. Initial App Load:**
```
User opens app → Frontend calls initialize_app_data → 
Edge Function checks DB → If empty, triggers Python service →
Python service fetches from FastF1 → Populates Supabase DB →
Frontend receives data → Displays dashboard
```

**2. Live Race:**
```
Race starts → Python LiveDataStreamer polls FastF1 (every 5s) →
Broadcasts to Supabase Realtime → Frontend subscribes to channel →
Receives lap updates → Updates leaderboard in real-time
```

**3. Prediction Generation:**
```
User navigates to Predictions Tab → Frontend fetches predictions from DB →
If no predictions, triggers generation → Python service loads ML model →
Generates predictions → Stores in DB → Returns to frontend → Displays with charts
```

**4. Strategy Simulation:**
```
User creates strategy → Frontend sends to FastAPI endpoint →
Python MonteCarloSimulator runs 1000 simulations →
Returns statistics → Frontend displays results with charts
```

**5. Automatic Model Retraining:**
```
Race completes → Results inserted to DB → Database trigger fires →
Calls ModelRetrainingAgent → Agent retrains models →
Evaluates performance → If improved, deploys new version →
Generates predictions for next race
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Python Services:**
```python
# tests/test_data_ingestion.py

import pytest
from services.data_ingestion import DataIngestionService

def test_fetch_season_schedule():
    service = DataIngestionService()
    schedule = service.fetch_season_schedule(2024)
    
    assert len(schedule) > 20  # At least 20 races
    assert 'EventName' in schedule.columns
    assert 'EventDate' in schedule.columns

def test_transform_race_result():
    transformer = DataTransformer()
    
    sample_lap = {
        'Driver': 'VER',
        'Position': 1,
        'LapTime': pd.Timedelta(seconds=90.5),
        'Team': 'Red Bull Racing'
    }
    
    result = transformer.transform_race_result([sample_lap], None, 'race_123')
    
    assert result[0]['position'] == 1
    assert result[0]['points'] == 25
```

**ML Models:**
```python
# tests/test_ml_models.py

def test_race_winner_predictor():
    predictor = RaceWinnerPredictor()
    
    # Mock training data
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    
    metrics = predictor.train(X, y)
    
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0.5  # Better than random

def test_tire_degradation_model():
    model = TireDegradationModel()
    
    tire_ages = np.array([1, 2, 3, 4, 5, 10, 15, 20])
    lap_times = np.array([90.0, 90.1, 90.3, 90.6, 91.0, 92.5, 94.0, 96.0])
    
    model.train_polynomial_model(tire_ages, lap_times)
    
    predicted_time = model.predict_degradation(25, 'SOFT')
    
    assert predicted_time > 96.0  # Should be slower than lap 20
```

### 9.2 Integration Tests

```python
# tests/test_integration.py

def test_end_to_end_prediction_flow():
    """Test complete flow from data fetch to prediction generation."""
    
    # 1. Fetch data
    service = DataIngestionService(supabase_client)
    service.populate_historical_data(2024, 2024)
    
    # 2. Train model
    predictor = RaceWinnerPredictor()
    X, y, _ = predictor.prepare_training_data(supabase_client)
    predictor.train(X, y)
    
    # 3. Generate predictions
    next_race_id = get_next_race_id(supabase_client)
    predictions = predictor.predict_race(next_race_id, supabase_client)
    
    # 4. Verify predictions
    assert len(predictions) > 0
    assert all('win_probability' in p for p in predictions)
    assert sum(p['win_probability'] for p in predictions) <= 1.0
```

### 9.3 Frontend Tests

```typescript
// tests/components/StrategySimulator.test.tsx

import { render, screen, fireEvent } from '@testing-library/react';
import { StrategySimulator } from '@/components/strategy-simulator/StrategySimulator';

describe('StrategySimulator', () => {
  it('renders stint editor', () => {
    render(<StrategySimulator raceId="race_123" />);
    
    expect(screen.getByText('Strategy Builder')).toBeInTheDocument();
    expect(screen.getByText('Stint 1:')).toBeInTheDocument();
  });
  
  it('allows adding stints', () => {
    render(<StrategySimulator raceId="race_123" />);
    
    const addButton = screen.getByText('Add Stint');
    fireEvent.click(addButton);
    
    expect(screen.getByText('Stint 2:')).toBeInTheDocument();
  });
  
  it('runs simulation on button click', async () => {
    const mockFetch = jest.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ results: [{ mean_time: 5400 }] })
      })
    );
    global.fetch = mockFetch as any;
    
    render(<StrategySimulator raceId="race_123" />);
    
    const simulateButton = screen.getByText('Run Simulation');
    fireEvent.click(simulateButton);
    
    expect(mockFetch).toHaveBeenCalledWith('/api/strategy/simulate', expect.any(Object));
  });
});
```

---

## 10. Monitoring & Observability

### 10.1 Logging

**Python Services:**
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/f1_analytics.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('F1Analytics')

# Usage
logger.info("Data ingestion started")
logger.error(f"Failed to fetch race data: {error}")
```

**Supabase Edge Functions:**
```typescript
// Log to Supabase
await supabase.from('logs').insert({
  level: 'INFO',
  service: 'initialize_app_data',
  message: 'Data initialization completed',
  timestamp: new Date().toISOString()
})
```

### 10.2 Performance Monitoring

**Key Metrics to Track:**
- API response times
- Database query performance
- ML model inference latency
- Data ingestion throughput
- Agent execution time
- Frontend page load times

**Tools:**
- **Datadog**: Application performance monitoring
- **Sentry**: Error tracking and alerting
- **Supabase Dashboard**: Database performance metrics

### 10.3 Alerts

**Critical Alerts:**
- Data ingestion failures
- Model training errors
- Prediction generation failures
- Agent crashes
- Database connection issues
- API rate limit exceeded

**Alert Channels:**
- Email notifications
- Slack integration
- PagerDuty for critical issues

---

## 11. Security Considerations

### 11.1 API Key Management

- Store all API keys in environment variables
- Use Supabase Secrets for Edge Functions
- Never commit keys to version control
- Rotate keys periodically

### 11.2 Database Security

- Enable Row Level Security (RLS) on all tables
- Use service role key only in backend services
- Implement proper authentication for user-facing features
- Sanitize all user inputs

### 11.3 Rate Limiting

- Implement rate limiting on API endpoints
- Cache FastF1 data to reduce API calls
- Use Supabase's built-in rate limiting

---

## 12. Cost Optimization

### 12.1 Supabase

- Use free tier for development
- Optimize database queries with indexes
- Archive old data to reduce storage costs
- Monitor API usage

### 12.2 Python Services

- Use spot instances for non-critical workloads
- Scale down during off-season
- Optimize Docker images for smaller size
- Use caching to reduce compute

### 12.3 External APIs

- Cache FastF1 data aggressively
- Batch requests when possible
- Use Ergast API for historical data (free)
- Monitor OpenAI API usage

---

## 13. Future Enhancements

### 13.1 Advanced ML Models

- **LSTM for Lap Time Prediction**: More accurate lap time forecasting
- **Reinforcement Learning for Strategy**: Deep Q-Network for optimal pit decisions
- **Ensemble Models**: Combine multiple models for better accuracy
- **Transfer Learning**: Use pre-trained models for faster training

### 13.2 Additional Features

- **Telemetry Visualization**: Speed traces, throttle/brake overlays
- **Track Maps**: Interactive circuit maps with data overlays
- **Championship Simulator**: Monte Carlo simulation for title race
- **Fantasy F1 Integration**: Optimize fantasy team selection
- **Mobile App**: Native iOS/Android apps
- **Push Notifications**: Race alerts and prediction updates

### 13.3 Data Enhancements

- **Weather Data Integration**: More detailed weather forecasts
- **Social Media Sentiment**: Analyze fan sentiment for predictions
- **Betting Odds Integration**: Compare predictions with bookmaker odds
- **Team Radio Transcripts**: Analyze team communications

---

## 14. Conclusion

This comprehensive specification provides a complete roadmap for implementing a production-ready F1 Analytics Platform with real machine learning, FastF1 integration, and autonomous agents. The implementation is structured in 6 phases over 6 weeks, delivering incremental value at each stage.

**Key Success Factors:**
1. **Real Data**: FastF1 provides rich, detailed F1 data
2. **Real ML**: XGBoost and LSTM models for accurate predictions
3. **Automation**: Intelligent agents for data sync and model updates
4. **User Experience**: Intuitive UI with interactive features
5. **Scalability**: Cloud-native architecture for growth

**Next Steps:**
1. Review and approve this specification
2. Set up development environment
3. Begin Phase 1: Data Foundation
4. Iterate based on user feedback

**Document Version**: 1.0  
**Last Updated**: 2025-11-07  
**Status**: Ready for Implementation

---

**Prepared by**: David, Data Analyst  
**For**: F1 Analytics Platform Development Team  
**Contact**: david@f1analytics.dev