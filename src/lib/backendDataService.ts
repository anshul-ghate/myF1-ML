/**
 * Backend Data Service
 * Replaces the old mock data service with real API calls
 */

import { apiClient } from './apiClient';

export interface Driver {
  id: string;
  name: string;
  code: string;
  permanent_number: number;
  nationality: string;
  dob?: string;
  headshot_url?: string;
}

export interface Constructor {
  id: string;
  name: string;
  nationality: string;
  logo_url?: string;
}

export interface Race {
  id: string;
  season_year: number;
  round: number;
  name: string;
  date: string;
  time: string;
  circuit_id: string;
  circuits?: {
    name: string;
    location: string;
    country: string;
  };
}

export interface RaceResult {
  race_id: string;
  driver_id: string;
  constructor_id: string;
  position: number;
  points: number;
  status: string;
  fastest_lap_time?: string;
  drivers?: Driver;
  constructors?: Constructor;
}

export interface Prediction {
  driver_id: string;
  driver_name: string;
  win_probability: number;
  confidence: number;
}

export interface DriverStanding {
  position: number;
  driver_id: string;
  driver_name: string;
  points: number;
  wins: number;
  podiums: number;
}

export interface ConstructorStanding {
  position: number;
  constructor_id: string;
  constructor_name: string;
  points: number;
  wins: number;
}

export interface StrategyResult {
  mean_time: number;
  std_dev: number;
  best_time: number;
  worst_time: number;
  median_time: number;
  percentile_25: number;
  percentile_75: number;
}

export interface DashboardData {
  driver_standings: DriverStanding[];
  constructor_standings: ConstructorStanding[];
}

class BackendDataService {
  /**
   * Get current season driver standings
   */
  async getDriverStandings(season: number = 2024): Promise<DriverStanding[]> {
    try {
      const standings = await apiClient.getDriverStandings(season);
      return standings;
    } catch (error) {
      console.error('Error fetching driver standings:', error);
      return [];
    }
  }

  /**
   * Get current season constructor standings
   */
  async getConstructorStandings(season: number = 2024): Promise<ConstructorStanding[]> {
    try {
      const standings = await apiClient.getConstructorStandings(season);
      return standings;
    } catch (error) {
      console.error('Error fetching constructor standings:', error);
      return [];
    }
  }

  /**
   * Get upcoming races
   */
  async getUpcomingRaces(): Promise<Race[]> {
    try {
      const races = await apiClient.getUpcomingRaces();
      return races;
    } catch (error) {
      console.error('Error fetching upcoming races:', error);
      return [];
    }
  }

  /**
   * Get race results
   */
  async getRaceResults(raceId: string): Promise<RaceResult[]> {
    try {
      const results = await apiClient.getRaceResults(raceId);
      return results;
    } catch (error) {
      console.error('Error fetching race results:', error);
      return [];
    }
  }

  /**
   * Get predictions for a race
   */
  async getRacePredictions(raceId: string): Promise<Prediction[]> {
    try {
      // Try to get existing predictions
      let predictions = await apiClient.getPredictions(raceId);
      
      // If no predictions exist, generate them
      if (!predictions || predictions.length === 0) {
        predictions = await apiClient.generatePredictions(raceId);
      }
      
      return predictions;
    } catch (error) {
      console.error('Error fetching predictions:', error);
      return [];
    }
  }

  /**
   * Simulate a race strategy
   */
  async simulateStrategy(
    strategy: Array<[string, number]>,
    options?: {
      n_simulations?: number;
      total_laps?: number;
      pit_stop_time?: number;
    }
  ): Promise<StrategyResult | null> {
    try {
      const response = await apiClient.simulateStrategy(strategy, options);
      return response.result;
    } catch (error) {
      console.error('Error simulating strategy:', error);
      return null;
    }
  }

  /**
   * Optimize multiple strategies
   */
  async optimizeStrategies(
    strategies: Array<Array<[string, number]>>,
    options?: {
      n_simulations?: number;
      total_laps?: number;
    }
  ): Promise<any[]> {
    try {
      const response = await apiClient.optimizeStrategies(strategies, options);
      return response.results;
    } catch (error) {
      console.error('Error optimizing strategies:', error);
      return [];
    }
  }

  /**
   * Get all drivers
   */
  async getDrivers(season?: number): Promise<Driver[]> {
    try {
      const drivers = await apiClient.getDrivers(season);
      return drivers;
    } catch (error) {
      console.error('Error fetching drivers:', error);
      return [];
    }
  }

  /**
   * Get driver details
   */
  async getDriver(driverId: string): Promise<Driver | null> {
    try {
      const driver = await apiClient.getDriver(driverId);
      return driver;
    } catch (error) {
      console.error('Error fetching driver:', error);
      return null;
    }
  }

  /**
   * Get driver race results
   */
  async getDriverResults(driverId: string, season?: number): Promise<RaceResult[]> {
    try {
      const results = await apiClient.getDriverResults(driverId, season);
      return results;
    } catch (error) {
      console.error('Error fetching driver results:', error);
      return [];
    }
  }

  /**
   * Get all constructors
   */
  async getConstructors(): Promise<Constructor[]> {
    try {
      const constructors = await apiClient.getConstructors();
      return constructors;
    } catch (error) {
      console.error('Error fetching constructors:', error);
      return [];
    }
  }

  /**
   * Get dashboard data
   */
  async getDashboardData(season?: number): Promise<DashboardData | null> {
    try {
      const data = await apiClient.getDashboardData(season);
      return data;
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      return null;
    }
  }

  /**
   * Get all races for a season
   */
  async getRaces(season?: number): Promise<Race[]> {
    try {
      const races = await apiClient.getRaces(season);
      return races;
    } catch (error) {
      console.error('Error fetching races:', error);
      return [];
    }
  }

  /**
   * Get tire compounds
   */
  async getTireCompounds() {
    try {
      const response = await apiClient.getTireCompounds();
      return response.compounds;
    } catch (error) {
      console.error('Error fetching tire compounds:', error);
      return [];
    }
  }

  /**
   * Check backend health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const health = await apiClient.healthCheck();
      return health.status === 'healthy';
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  /**
   * Train ML model
   */
  async trainModel(): Promise<any> {
    try {
      const response = await apiClient.trainModel();
      return response;
    } catch (error) {
      console.error('Error training model:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const backendDataService = new BackendDataService();

// Export class for testing
export default BackendDataService;