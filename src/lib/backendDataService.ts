// Backend data service that uses Supabase instead of Ergast API
import {
  fetchDrivers,
  fetchConstructors,
  fetchRaces,
  fetchNextRace,
  fetchLastRace,
  fetchRaceResults,
  fetchDriverStandings,
  fetchConstructorStandings,
  fetchPredictions,
  Driver,
  Constructor,
  Race,
  RaceResult,
} from './supabaseService';

// Transform backend data to match frontend types
export async function getDriverStandings(year: number = new Date().getFullYear()) {
  const standings = await fetchDriverStandings(year);
  
  return standings.map(standing => ({
    position: standing.position.toString(),
    Driver: {
      driverId: standing.driver.ergast_driver_id,
      code: standing.driver.code,
      givenName: standing.driver.given_name,
      familyName: standing.driver.family_name,
      dateOfBirth: standing.driver.date_of_birth,
      nationality: standing.driver.nationality,
    },
    Constructor: {
      constructorId: standing.constructor.ergast_constructor_id,
      name: standing.constructor.name,
      nationality: standing.constructor.nationality,
    },
    points: standing.points.toString(),
    wins: standing.wins.toString(),
  }));
}

export async function getConstructorStandings(year: number = new Date().getFullYear()) {
  const standings = await fetchConstructorStandings(year);
  
  return standings.map(standing => ({
    position: standing.position.toString(),
    Constructor: {
      constructorId: standing.constructor.ergast_constructor_id,
      name: standing.constructor.name,
      nationality: standing.constructor.nationality,
    },
    points: standing.points.toString(),
    wins: standing.wins.toString(),
  }));
}

export async function getLastRaceResults() {
  const lastRace = await fetchLastRace();
  if (!lastRace) return [];

  const results = await fetchRaceResults(lastRace.id);
  
  return {
    raceName: lastRace.name,
    Circuit: {
      circuitName: lastRace.circuit?.name || '',
      Location: {
        locality: lastRace.circuit?.location || '',
        country: lastRace.circuit?.country || '',
      },
    },
    date: lastRace.date,
    Results: results.map(result => ({
      position: result.position?.toString() || result.position_text,
      Driver: {
        driverId: result.driver?.ergast_driver_id || '',
        code: result.driver?.code || '',
        givenName: result.driver?.given_name || '',
        familyName: result.driver?.family_name || '',
      },
      Constructor: {
        constructorId: result.constructor?.ergast_constructor_id || '',
        name: result.constructor?.name || '',
      },
      grid: result.grid?.toString() || '',
      laps: result.laps?.toString() || '',
      status: result.status,
      Time: result.fastest_lap_rank ? { time: 'N/A' } : undefined,
      FastestLap: result.fastest_lap_rank ? {
        rank: result.fastest_lap_rank.toString(),
        Time: { time: 'N/A' },
      } : undefined,
    })),
  };
}

export async function getRaceSchedule(year: number = new Date().getFullYear()) {
  const races = await fetchRaces(year);
  
  return races.map(race => ({
    round: race.round.toString(),
    raceName: race.name,
    Circuit: {
      circuitId: race.circuit?.ergast_circuit_id || '',
      circuitName: race.circuit?.name || '',
      Location: {
        locality: race.circuit?.location || '',
        country: race.circuit?.country || '',
      },
    },
    date: race.date,
    time: race.time || '',
  }));
}

export async function getNextRace() {
  const nextRace = await fetchNextRace();
  if (!nextRace) return null;

  return {
    round: nextRace.round.toString(),
    raceName: nextRace.name,
    Circuit: {
      circuitId: nextRace.circuit?.ergast_circuit_id || '',
      circuitName: nextRace.circuit?.name || '',
      Location: {
        locality: nextRace.circuit?.location || '',
        country: nextRace.circuit?.country || '',
      },
    },
    date: nextRace.date,
    time: nextRace.time || '',
  };
}

// Get predictions for next race
export async function getNextRacePredictions() {
  const nextRace = await fetchNextRace();
  if (!nextRace) return [];

  const predictions = await fetchPredictions(nextRace.id);
  
  return predictions.map(pred => ({
    position: pred.predicted_position,
    driver: {
      code: pred.driver?.code || '',
      givenName: pred.driver?.given_name || '',
      familyName: pred.driver?.family_name || '',
    },
    confidence: pred.confidence,
  }));
}