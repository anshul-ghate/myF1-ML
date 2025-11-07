/**
 * API Type Definitions
 * Type definitions for backend API responses
 */

export interface Race {
  id: string;
  season: number;
  round: number;
  race_name: string;
  circuit_id: string;
  circuit_name: string;
  date: string;
  time?: string;
  url?: string;
  results?: RaceResult[];
}

export interface RaceResult {
  id: string;
  race_id: string;
  driver_id: string;
  driver_name: string;
  constructor_id: string;
  constructor_name: string;
  position: number;
  points: number;
  laps: number;
  status: string;
  grid?: number;
  fastest_lap?: number;
}

export interface DriverStanding {
  driver_id: string;
  driver_name: string;
  constructor_id: string;
  constructor_name: string;
  position: number;
  points: number;
  wins: number;
}

export interface ConstructorStanding {
  constructor_id: string;
  constructor_name: string;
  position: number;
  points: number;
  wins: number;
}

export interface Driver {
  id: string;
  driver_ref: string;
  code?: string;
  permanent_number?: number;
  given_name: string;
  family_name: string;
  date_of_birth?: string;
  nationality: string;
  url?: string;
}

export interface Constructor {
  id: string;
  constructor_ref: string;
  name: string;
  nationality: string;
  url?: string;
}

export interface Prediction {
  driver_id: string;
  driver_name: string;
  constructor_id: string;
  constructor_name: string;
  predicted_position: number;
  confidence: number;
}

export interface StrategySimulation {
  mean_time: number;
  std_dev: number;
  risk_score: number;
  pit_stops?: PitStop[];
}

export interface PitStop {
  lap: number;
  compound: string;
}

export interface TireCompound {
  name: string;
  color: string;
}

export interface HealthCheck {
  status: string;
  version: string;
  service: string;
}