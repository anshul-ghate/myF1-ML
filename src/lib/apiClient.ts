/**
 * API Client for F1 Analytics Backend
 * Connects to Python FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Races API
  async getRaces(season?: number) {
    const query = season ? `?season=${season}` : '';
    return this.request(`/races${query}`);
  }

  async getUpcomingRaces() {
    return this.request('/races/upcoming');
  }

  async getRace(raceId: string) {
    return this.request(`/races/${raceId}`);
  }

  async getRaceResults(raceId: string) {
    return this.request(`/races/${raceId}/results`);
  }

  // Predictions API
  async generatePredictions(raceId: string) {
    return this.request('/predictions/generate', {
      method: 'POST',
      body: JSON.stringify({ race_id: raceId }),
    });
  }

  async getPredictions(raceId: string) {
    return this.request(`/predictions/${raceId}`);
  }

  async trainModel() {
    return this.request('/predictions/train', {
      method: 'POST',
    });
  }

  // Strategy API
  async simulateStrategy(strategy: Array<[string, number]>, options?: {
    n_simulations?: number;
    total_laps?: number;
    pit_stop_time?: number;
  }) {
    return this.request('/strategy/simulate', {
      method: 'POST',
      body: JSON.stringify({
        strategy,
        n_simulations: options?.n_simulations || 1000,
        total_laps: options?.total_laps || 55,
        pit_stop_time: options?.pit_stop_time || 25.0,
      }),
    });
  }

  async optimizeStrategies(strategies: Array<Array<[string, number]>>, options?: {
    n_simulations?: number;
    total_laps?: number;
  }) {
    return this.request('/strategy/optimize', {
      method: 'POST',
      body: JSON.stringify({
        strategies,
        n_simulations: options?.n_simulations || 500,
        total_laps: options?.total_laps || 55,
      }),
    });
  }

  async getTireCompounds() {
    return this.request('/strategy/compounds');
  }

  // Drivers API
  async getDrivers(season?: number) {
    const query = season ? `?season=${season}` : '';
    return this.request(`/drivers${query}`);
  }

  async getDriver(driverId: string) {
    return this.request(`/drivers/${driverId}`);
  }

  async getDriverResults(driverId: string, season?: number) {
    const query = season ? `?season=${season}` : '';
    return this.request(`/drivers/${driverId}/results${query}`);
  }

  async getDriverStandings(season: number) {
    return this.request(`/drivers/standings/${season}`);
  }

  // Constructors API
  async getConstructors() {
    return this.request('/constructors');
  }

  async getConstructor(constructorId: string) {
    return this.request(`/constructors/${constructorId}`);
  }

  async getConstructorStandings(season: number) {
    return this.request(`/constructors/standings/${season}`);
  }

  // Analytics API
  async getDashboardData(season?: number) {
    const query = season ? `?season=${season}` : '';
    return this.request(`/analytics/dashboard${query}`);
  }

  // Health check
  async healthCheck() {
    return fetch(`${this.baseUrl.replace('/api/v1', '')}/health`).then(r => r.json());
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export class for testing
export default ApiClient;