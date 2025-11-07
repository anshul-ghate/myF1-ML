# Frontend-Backend Integration Guide

Complete guide for integrating the React frontend with the Python FastAPI backend.

## Overview

The frontend now connects to a real Python backend with FastF1 integration and ML models, replacing all mock data.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Next.js)                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Components (Dashboard, Predictions, Strategy, etc.)   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  React Hooks (useDriverStandings, etc.)                │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Backend Data Service                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  API Client                                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Python FastAPI Backend                          │
│  - FastF1 Integration                                        │
│  - ML Models (XGBoost, LSTM)                                 │
│  - Strategy Simulator                                        │
│  - Automated Data Sync                                       │
└─────────────────────────────────────────────────────────────┘
```

## New Files Created

### 1. API Client (`src/lib/apiClient.ts`)
- Low-level HTTP client
- Handles all API requests
- Type-safe endpoints
- Error handling

### 2. Backend Data Service (`src/lib/backendDataService.ts`)
- High-level data service
- Business logic layer
- Typed interfaces
- Caching support

### 3. React Hooks (`src/hooks/useBackendData.ts`)
- `useDriverStandings()` - Fetch driver standings
- `useConstructorStandings()` - Fetch constructor standings
- `useUpcomingRaces()` - Fetch upcoming races
- `useRacePredictions()` - Fetch ML predictions
- `useDashboardData()` - Fetch dashboard data
- `useDrivers()` - Fetch all drivers
- `useConstructors()` - Fetch all constructors
- `useBackendHealth()` - Check backend status

### 4. Backend Status Component (`src/components/BackendStatus.tsx`)
- Shows connection status
- Real-time health checks
- Error messages with instructions

### 5. Environment Configuration (`.env.local.example`)
- Backend API URL configuration
- Feature flags
- Supabase configuration

## Setup Instructions

### 1. Start the Backend

First, ensure the Python backend is running:

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Supabase credentials
python main.py
```

The backend will:
- Start on http://localhost:8000
- Automatically populate database with F1 data (5-10 minutes first time)
- Train ML models
- Start background sync

### 2. Configure Frontend

Create `.env.local` in the root directory:

```bash
cp .env.local.example .env.local
```

Edit `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### 3. Install Frontend Dependencies

```bash
npm install
```

### 4. Run Frontend

```bash
npm run dev
```

Open http://localhost:3000

## Usage Examples

### Fetching Driver Standings

```typescript
import { useDriverStandings } from '@/hooks/useBackendData';

function DriverStandingsComponent() {
  const { standings, loading, error } = useDriverStandings(2024);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      {standings.map((standing) => (
        <div key={standing.driver_id}>
          {standing.position}. {standing.driver_name} - {standing.points} pts
        </div>
      ))}
    </div>
  );
}
```

### Generating Predictions

```typescript
import { backendDataService } from '@/lib/backendDataService';

async function generatePredictions(raceId: string) {
  const predictions = await backendDataService.getRacePredictions(raceId);
  
  console.log('Top 3 predictions:');
  predictions.slice(0, 3).forEach((pred, idx) => {
    console.log(`${idx + 1}. ${pred.driver_name}: ${(pred.win_probability * 100).toFixed(1)}%`);
  });
}
```

### Simulating Strategy

```typescript
import { backendDataService } from '@/lib/backendDataService';

async function simulateStrategy() {
  // Strategy: Soft for 15 laps, then Medium for 40 laps
  const strategy: Array<[string, number]> = [
    ['SOFT', 15],
    ['MEDIUM', 40]
  ];

  const result = await backendDataService.simulateStrategy(strategy, {
    n_simulations: 1000,
    total_laps: 55,
    pit_stop_time: 25.0
  });

  if (result) {
    console.log(`Mean race time: ${result.mean_time.toFixed(2)}s`);
    console.log(`Best case: ${result.best_time.toFixed(2)}s`);
    console.log(`Worst case: ${result.worst_time.toFixed(2)}s`);
  }
}
```

### Checking Backend Health

```typescript
import { useBackendHealth } from '@/hooks/useBackendData';

function App() {
  const { isHealthy, checking } = useBackendHealth();

  return (
    <div>
      {checking && <p>Checking backend...</p>}
      {!checking && isHealthy && <p>✅ Backend connected</p>}
      {!checking && !isHealthy && <p>❌ Backend offline</p>}
    </div>
  );
}
```

## Updating Existing Components

### Dashboard Component

Replace mock data with real data:

```typescript
// Before
import { mockDriverStandings } from '@/data/mockData';

// After
import { useDriverStandings } from '@/hooks/useBackendData';

function Dashboard() {
  const { standings, loading, error } = useDriverStandings(2024);
  
  // Use standings instead of mockDriverStandings
}
```

### Race Results Component

```typescript
// Before
import { mockRaceResults } from '@/data/mockData';

// After
import { useState, useEffect } from 'react';
import { backendDataService } from '@/lib/backendDataService';

function RaceResults({ raceId }: { raceId: string }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    backendDataService.getRaceResults(raceId).then(data => {
      setResults(data);
      setLoading(false);
    });
  }, [raceId]);

  // Use results
}
```

## API Endpoints Reference

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

## Error Handling

All hooks and services include error handling:

```typescript
const { data, loading, error } = useDriverStandings(2024);

if (error) {
  // Handle error
  console.error('Failed to fetch standings:', error);
  return <ErrorComponent message={error.message} />;
}
```

## Loading States

All hooks provide loading states:

```typescript
const { standings, loading } = useDriverStandings(2024);

if (loading) {
  return <LoadingSpinner />;
}

return <StandingsTable data={standings} />;
```

## Caching

The backend includes caching:
- FastF1 data is cached locally
- ML models are persisted
- API responses can be cached in frontend

To implement frontend caching, use React Query or SWR:

```bash
npm install @tanstack/react-query
```

```typescript
import { useQuery } from '@tanstack/react-query';
import { backendDataService } from '@/lib/backendDataService';

function useDriverStandings(season: number) {
  return useQuery({
    queryKey: ['driverStandings', season],
    queryFn: () => backendDataService.getDriverStandings(season),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
```

## Troubleshooting

### Backend Not Responding

1. Check if backend is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check backend logs:
   ```bash
   cd backend
   tail -f logs/f1_analytics.log
   ```

3. Verify environment variables:
   ```bash
   cat .env.local
   ```

### CORS Issues

If you see CORS errors, ensure the backend allows your frontend origin:

In `backend/config.py`:
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    # Add your production domain
]
```

### Data Not Loading

1. Check if database is populated:
   ```bash
   curl http://localhost:8000/api/v1/drivers
   ```

2. If empty, wait for initial data sync (5-10 minutes)

3. Check backend logs for sync progress

### Predictions Not Working

1. Ensure database has race results:
   ```bash
   curl http://localhost:8000/api/v1/races
   ```

2. Train model manually:
   ```bash
   curl -X POST http://localhost:8000/api/v1/predictions/train
   ```

## Production Deployment

### Backend

Deploy backend to Railway/Render/Fly.io (see `backend/DEPLOYMENT.md`)

### Frontend

1. Update `.env.production`:
   ```bash
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app/api/v1
   ```

2. Deploy to Vercel:
   ```bash
   vercel --prod
   ```

### Environment Variables

Set in Vercel dashboard:
- `NEXT_PUBLIC_API_URL` - Your backend URL
- `NEXT_PUBLIC_SUPABASE_URL` - Supabase URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Supabase key

## Testing

### Test Backend Connection

```typescript
import { backendDataService } from '@/lib/backendDataService';

async function testConnection() {
  const isHealthy = await backendDataService.checkHealth();
  console.log('Backend healthy:', isHealthy);
}
```

### Test Data Fetching

```typescript
import { backendDataService } from '@/lib/backendDataService';

async function testDataFetching() {
  const drivers = await backendDataService.getDrivers();
  console.log(`Fetched ${drivers.length} drivers`);
  
  const standings = await backendDataService.getDriverStandings(2024);
  console.log(`Fetched ${standings.length} standings`);
}
```

## Next Steps

1. ✅ Backend is running with real F1 data
2. ✅ Frontend has API client and hooks
3. ✅ Update existing components to use real data
4. ✅ Add loading and error states
5. ✅ Test all features
6. ✅ Deploy to production

## Support

- Backend API docs: http://localhost:8000/docs
- Backend health: http://localhost:8000/health
- Frontend: http://localhost:3000

For issues, check:
- Backend logs: `backend/logs/f1_analytics.log`
- Browser console for frontend errors
- Network tab for API requests

## License

MIT