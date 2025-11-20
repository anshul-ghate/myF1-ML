// Data fetching service - now uses Supabase backend
import {
  getDriverStandings,
  getConstructorStandings,
  getLastRaceResults,
  getRaceSchedule,
  getNextRace,
  getNextRacePredictions,
} from './backendDataService';

export async function fetchDriverStandings(year: number = new Date().getFullYear()) {
  try {
    return await getDriverStandings(year);
  } catch (error) {
    console.error('Error fetching driver standings:', error);
    return [];
  }
}

export async function fetchConstructorStandings(year: number = new Date().getFullYear()) {
  try {
    return await getConstructorStandings(year);
  } catch (error) {
    console.error('Error fetching constructor standings:', error);
    return [];
  }
}

export async function fetchLastRaceResults() {
  try {
    return await getLastRaceResults();
  } catch (error) {
    console.error('Error fetching last race results:', error);
    return null;
  }
}

export async function fetchRaceSchedule(year: number = new Date().getFullYear()) {
  try {
    return await getRaceSchedule(year);
  } catch (error) {
    console.error('Error fetching race schedule:', error);
    return [];
  }
}

export async function fetchNextRace() {
  try {
    return await getNextRace();
  } catch (error) {
    console.error('Error fetching next race:', error);
    return null;
  }
}

export async function fetchPredictions() {
  try {
    return await getNextRacePredictions();
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return [];
  }
}