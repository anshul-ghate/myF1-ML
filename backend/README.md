# F1 Analytics Backend

Complete Python FastAPI backend with FastF1 integration, ML models, and automated data synchronization.

## Features

- ✅ **FastF1 Integration**: Real F1 data with telemetry and lap times
- ✅ **Automated Data Sync**: Populates database automatically on startup
- ✅ **ML Predictions**: XGBoost race winner predictor
- ✅ **Strategy Simulator**: Monte Carlo pit stop strategy optimization
- ✅ **REST API**: Complete endpoints for all features
- ✅ **Background Scheduler**: Continuous data updates

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
```

### 3. Run the Server

```bash
python main.py
```

The server will:
1. Start on http://localhost:8000
2. Automatically check if database is empty
3. If empty, populate with 2023-2024 F1 data (takes 5-10 minutes)
4. Start background sync scheduler
5. Train ML models

### 4. Access API Documentation

Open http://localhost:8000/docs for interactive API documentation.

## API Endpoints

### Races
- `GET /api/v1/races` - Get all races
- `GET /api/v1/races/upcoming` - Get upcoming races
- `GET /api/v1/races/{race_id}` - Get race details
- `GET /api/v1/races/{race_id}/results` - Get race results

### Predictions
- `POST /api/v1/predictions/generate` - Generate predictions
- `GET /api/v1/predictions/{race_id}` - Get predictions
- `POST /api/v1/predictions/train` - Train ML model

### Strategy
- `POST /api/v1/strategy/simulate` - Simulate strategy
- `POST /api/v1/strategy/optimize` - Optimize strategies
- `GET /api/v1/strategy/compounds` - Get tire compounds

### Drivers
- `GET /api/v1/drivers` - Get all drivers
- `GET /api/v1/drivers/{driver_id}` - Get driver details
- `GET /api/v1/drivers/{driver_id}/results` - Get driver results
- `GET /api/v1/drivers/standings/{season}` - Get standings

### Constructors
- `GET /api/v1/constructors` - Get all constructors
- `GET /api/v1/constructors/{constructor_id}` - Get constructor details
- `GET /api/v1/constructors/standings/{season}` - Get standings

### Analytics
- `GET /api/v1/analytics/dashboard` - Get dashboard data

## Architecture

```
backend/
├── main.py                 # FastAPI app with automated startup
├── manual_sync.py          # Manual data sync script
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── services/
│   ├── supabase_client.py  # Supabase connection
│   ├── fastf1_service.py   # FastF1 data fetching
│   └── data_sync_agent.py  # Automated data sync
├── models/
│   ├── race_winner_predictor.py  # XGBoost ML model
│   └── strategy_simulator.py     # Monte Carlo simulator
└── api/
    ├── races.py            # Race endpoints
    ├── predictions.py      # Prediction endpoints
    ├── strategy.py         # Strategy endpoints
    ├── drivers.py          # Driver endpoints
    ├── constructors.py     # Constructor endpoints
    └── analytics.py        # Analytics endpoints
```

## Automated Agent

The Data Sync Agent runs automatically on startup:

1. **Checks database status**
2. **If empty**: Fetches 2023-2024 F1 data from FastF1
3. **Populates**: Drivers, constructors, circuits, races, results
4. **Schedules**: Background sync every 24 hours
5. **Updates**: Checks for new races and driver changes

No manual intervention required!

## ML Models

### Race Winner Predictor
- **Algorithm**: XGBoost Classifier
- **Features**: Driver form, constructor performance, circuit history
- **Training**: Automatic on first prediction request
- **Accuracy**: Target 70%+

### Strategy Simulator
- **Algorithm**: Monte Carlo simulation
- **Simulations**: 1000+ iterations per strategy
- **Models**: Tire degradation, safety car probability
- **Output**: Mean time, std dev, risk score

## Development

### Run in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pytest tests/
```

### Train ML Model Manually

```bash
curl -X POST http://localhost:8000/api/v1/predictions/train
```

### Manual Data Sync

To manually populate or update the database, run the `manual_sync.py` script:

```bash
python manual_sync.py
```

This script will perform the same initial data load or update check that normally runs on server startup. Logs will be saved to `logs/manual_sync.log`.

## Deployment

### Docker

```bash
docker build -t f1-analytics-backend .
docker run -p 8000:8000 --env-file .env f1-analytics-backend
```

### Railway/Render/Fly.io

1. Push to GitHub
2. Connect repository to hosting platform
3. Set environment variables
4. Deploy

The automated agent will run on first deployment and populate the database.

## Troubleshooting

### Database Not Populating

Check logs for errors:
```bash
tail -f logs/f1_analytics.log
```
Or for the manual script:
```bash
tail -f logs/manual_sync.log
```

### FastF1 Cache Issues

Clear cache:
```bash
rm -rf fastf1_cache/
```

### Model Training Fails

Ensure database has race results:
```bash
curl http://localhost:8000/api/v1/races
```

## Support

For issues, check:
- API docs: http://localhost:8000/docs
- Logs: `logs/f1_analytics.log`
- FastF1 docs: https://docs.fastf1.dev/

## License

MIT
