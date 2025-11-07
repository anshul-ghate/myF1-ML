/**
 * Custom hooks for fetching data from FastAPI backend
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/apiClient';

// Dashboard data
export function useDashboardData() {
  return useQuery({
    queryKey: ['dashboard'],
    queryFn: () => apiClient.getDashboardData(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true,
  });
}

// Races
export function useRaces(season?: number) {
  return useQuery({
    queryKey: ['races', season],
    queryFn: () => apiClient.getRaces(season),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

export function useUpcomingRaces() {
  return useQuery({
    queryKey: ['races', 'upcoming'],
    queryFn: () => apiClient.getUpcomingRaces(),
    staleTime: 5 * 60 * 1000,
    refetchInterval: 60 * 1000, // Refresh every minute
  });
}

export function useRaceById(raceId: string) {
  return useQuery({
    queryKey: ['race', raceId],
    queryFn: () => apiClient.getRaceById(raceId),
    enabled: !!raceId,
  });
}

export function useRaceResults(raceId: string) {
  return useQuery({
    queryKey: ['race-results', raceId],
    queryFn: () => apiClient.getRaceResults(raceId),
    enabled: !!raceId,
  });
}

// Driver Standings
export function useDriverStandings(season?: number) {
  return useQuery({
    queryKey: ['driver-standings', season || new Date().getFullYear()],
    queryFn: () => apiClient.getDriverStandings(season),
    staleTime: 5 * 60 * 1000,
  });
}

// Constructor Standings
export function useConstructorStandings(season?: number) {
  return useQuery({
    queryKey: ['constructor-standings', season || new Date().getFullYear()],
    queryFn: () => apiClient.getConstructorStandings(season),
    staleTime: 5 * 60 * 1000,
  });
}

// Drivers
export function useDrivers(season?: number) {
  return useQuery({
    queryKey: ['drivers', season],
    queryFn: () => apiClient.getDrivers(season),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

export function useDriverById(driverId: string) {
  return useQuery({
    queryKey: ['driver', driverId],
    queryFn: () => apiClient.getDriverById(driverId),
    enabled: !!driverId,
  });
}

export function useDriverResults(driverId: string, season?: number) {
  return useQuery({
    queryKey: ['driver-results', driverId, season],
    queryFn: () => apiClient.getDriverResults(driverId, season),
    enabled: !!driverId,
  });
}

// Constructors
export function useConstructors(season?: number) {
  return useQuery({
    queryKey: ['constructors', season],
    queryFn: () => apiClient.getConstructors(season),
    staleTime: 30 * 60 * 1000,
  });
}

export function useConstructorById(constructorId: string) {
  return useQuery({
    queryKey: ['constructor', constructorId],
    queryFn: () => apiClient.getConstructorById(constructorId),
    enabled: !!constructorId,
  });
}

// Predictions
export function usePredictions(raceId: string) {
  return useQuery({
    queryKey: ['predictions', raceId],
    queryFn: () => apiClient.getPredictions(raceId),
    enabled: !!raceId,
    staleTime: 10 * 60 * 1000,
  });
}

export function useGeneratePredictions() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (raceId: string) => apiClient.generatePredictions(raceId),
    onSuccess: (_, raceId) => {
      queryClient.invalidateQueries({ queryKey: ['predictions', raceId] });
    },
  });
}

export function useTrainModel() {
  return useMutation({
    mutationFn: () => apiClient.trainModel(),
  });
}

// Strategy Simulator
export function useSimulateStrategy() {
  return useMutation({
    mutationFn: (data: {
      race_id: string;
      driver_id: string;
      pit_stops: Array<{ lap: number; compound: string }>;
    }) => apiClient.simulateStrategy(data),
  });
}

export function useOptimizeStrategies() {
  return useMutation({
    mutationFn: (data: {
      race_id: string;
      driver_id: string;
      num_strategies?: number;
    }) => apiClient.optimizeStrategies(data),
  });
}

export function useTireCompounds() {
  return useQuery({
    queryKey: ['tire-compounds'],
    queryFn: () => apiClient.getTireCompounds(),
    staleTime: Infinity, // Tire compounds rarely change
  });
}

// Backend health check
export function useBackendHealth() {
  return useQuery({
    queryKey: ['backend-health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 30 * 1000, // Check every 30 seconds
    retry: 3,
  });
}