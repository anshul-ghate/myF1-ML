/**
 * API Client for FastAPI Backend
 * Handles all HTTP requests to the Python backend
 */
import type {
  Race,
  RaceResult,
  DriverStanding,
  ConstructorStanding,
  Driver,
  Constructor,
  Prediction,
  StrategySimulation,
  TireCompound,
  HealthCheck,
  PitStop,
} from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_PREFIX = '/api/v1';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${API_PREFIX}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // Health check
  async healthCheck(): Promise<HealthCheck> {
    return this.request('/health');
  }

  // Races
  async getRaces(season?: number): Promise<Race[]> {
    const query = season ? `?season=${season}` : '';
    return this.request(`/races${query}`);
  }

  async getUpcomingRaces(): Promise<Race[]> {
    return this.request('/races/upcoming');
  }

  async getRaceById(raceId: string): Promise<Race> {
    return this.request(`/races/${raceId}`);
  }

  async getRaceResults(raceId: string): Promise<RaceResult[]> {
    return this.request(`/races/${raceId}/results`);
  }

  // Predictions
  async generatePredictions(raceId: string): Promise<Prediction[]> {
    return this.request('/predictions/generate', {
      method: 'POST',
      body: JSON.stringify({ race_id: raceId }),
    });
  }

  async getPredictions(raceId: string): Promise<Prediction[]> {
    return this.request(`/predictions/${raceId}`);
  }

  async trainModel(): Promise<{ message: string; status: string }> {
    return this.request('/predictions/train', {
      method: 'POST',
    });
  }

  // Strategy Simulator
  async simulateStrategy(data: {
    race_id: string;
    driver_id: string;
    pit_stops: PitStop[];
  }): Promise<StrategySimulation> {
    return this.request('/strategy/simulate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async optimizeStrategies(data: {
    race_id: string;
    driver_id: string;
    num_strategies?: number;
  }): Promise<StrategySimulation[]> {
    return this.request('/strategy/optimize', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getTireCompounds(): Promise<TireCompound[]> {
    return this.request('/strategy/compounds');
  }

  // Drivers
  async getDrivers(season?: number): Promise<Driver[]> {
    const query = season ? `?season=${season}` : '';
    return this.request(`/drivers${query}`);
  }

  async getDriverById(driverId: string): Promise<Driver> {
    return this.request(`/drivers/${driverId}`);
  }

  async getDriverResults(driverId: string, season?: number): Promise<RaceResult[]> {
    const query = season ? `?season=${season}` : '';
    return this.request(`/drivers/${driverId}/results${query}`);
  }

  async getDriverStandings(season: number = new Date().getFullYear()): Promise<DriverStanding[]> {
    return this.request(`/drivers/standings/${season}`);
  }

  // Constructors
  async getConstructors(season?: number): Promise<Constructor[]> {
    const query = season ? `?season=${season}` : '';
    return this.request(`/constructors${query}`);
  }

  async getConstructorById(constructorId: string): Promise<Constructor> {
    return this.request(`/constructors/${constructorId}`);
  }

  async getConstructorStandings(season: number = new Date().getFullYear()): Promise<ConstructorStanding[]> {
    return this.request(`/constructors/standings/${season}`);
  }

  // Analytics
  async getDashboardData(): Promise<{
    upcoming_races: Race[];
    driver_standings: DriverStanding[];
    constructor_standings: ConstructorStanding[];
    last_race_results: RaceResult[];
  }> {
    return this.request('/analytics/dashboard');
  }
}

export const apiClient = new ApiClient();
export default apiClient;