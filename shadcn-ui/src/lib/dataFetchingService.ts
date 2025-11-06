import {
  DriverStanding,
  ConstructorStanding,
  Race,
  RaceWithResults,
} from '@/data/types';

const BASE_URL = 'https://ergast.com/api/f1';

async function fetchData(endpoint: string) {
  const response = await fetch(`${BASE_URL}/${endpoint}`);
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  const data = await response.json();
  return data.MRData;
}

export const DataFetchingService = {
  async getDriverStandings(): Promise<DriverStanding[]> {
    const data = await fetchData('current/driverStandings.json');
    const standings = data.StandingsTable.StandingsLists[0].DriverStandings;
    // The 'wins' property is part of the standings data from this endpoint
    return standings;
  },

  async getConstructorStandings(): Promise<ConstructorStanding[]> {
    const data = await fetchData('current/constructorStandings.json');
    const standings = data.StandingsTable.StandingsLists[0].ConstructorStandings;
    // The 'wins' property is part of the standings data from this endpoint
    return standings;
  },

  async getNextRace(): Promise<Race> {
    const data = await fetchData('current/next.json');
    return data.RaceTable.Races[0];
  },

  async getLastRaceResults(): Promise<RaceWithResults> {
    const data = await fetchData('current/last/results.json');
    return data.RaceTable.Races[0];
  },

  async getRaceSchedule(): Promise<Race[]> {
    const data = await fetchData('current.json');
    return data.RaceTable.Races;
  },
};