/**
 * React hooks for fetching data from backend
 */

import { useState, useEffect } from 'react';
import { backendDataService } from '@/lib/backendDataService';
import type {
  Driver,
  Constructor,
  Race,
  RaceResult,
  Prediction,
  DriverStanding,
  ConstructorStanding,
  DashboardData,
} from '@/lib/backendDataService';

/**
 * Hook for fetching driver standings
 */
export function useDriverStandings(season: number = 2024) {
  const [standings, setStandings] = useState<DriverStanding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchStandings = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getDriverStandings(season);
        if (mounted) {
          setStandings(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchStandings();

    return () => {
      mounted = false;
    };
  }, [season]);

  return { standings, loading, error };
}

/**
 * Hook for fetching constructor standings
 */
export function useConstructorStandings(season: number = 2024) {
  const [standings, setStandings] = useState<ConstructorStanding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchStandings = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getConstructorStandings(season);
        if (mounted) {
          setStandings(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchStandings();

    return () => {
      mounted = false;
    };
  }, [season]);

  return { standings, loading, error };
}

/**
 * Hook for fetching upcoming races
 */
export function useUpcomingRaces() {
  const [races, setRaces] = useState<Race[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchRaces = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getUpcomingRaces();
        if (mounted) {
          setRaces(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchRaces();

    return () => {
      mounted = false;
    };
  }, []);

  return { races, loading, error };
}

/**
 * Hook for fetching race predictions
 */
export function useRacePredictions(raceId: string | null) {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!raceId) return;

    let mounted = true;

    const fetchPredictions = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getRacePredictions(raceId);
        if (mounted) {
          setPredictions(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchPredictions();

    return () => {
      mounted = false;
    };
  }, [raceId]);

  return { predictions, loading, error };
}

/**
 * Hook for fetching dashboard data
 */
export function useDashboardData(season?: number) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchData = async () => {
      try {
        setLoading(true);
        const dashboardData = await backendDataService.getDashboardData(season);
        if (mounted) {
          setData(dashboardData);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      mounted = false;
    };
  }, [season]);

  return { data, loading, error };
}

/**
 * Hook for fetching all drivers
 */
export function useDrivers(season?: number) {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchDrivers = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getDrivers(season);
        if (mounted) {
          setDrivers(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchDrivers();

    return () => {
      mounted = false;
    };
  }, [season]);

  return { drivers, loading, error };
}

/**
 * Hook for fetching all constructors
 */
export function useConstructors() {
  const [constructors, setConstructors] = useState<Constructor[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchConstructors = async () => {
      try {
        setLoading(true);
        const data = await backendDataService.getConstructors();
        if (mounted) {
          setConstructors(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err as Error);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchConstructors();

    return () => {
      mounted = false;
    };
  }, []);

  return { constructors, loading, error };
}

/**
 * Hook for checking backend health
 */
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let mounted = true;

    const checkHealth = async () => {
      try {
        setChecking(true);
        const healthy = await backendDataService.checkHealth();
        if (mounted) {
          setIsHealthy(healthy);
        }
      } catch (err) {
        if (mounted) {
          setIsHealthy(false);
        }
      } finally {
        if (mounted) {
          setChecking(false);
        }
      }
    };

    checkHealth();

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return { isHealthy, checking };
}