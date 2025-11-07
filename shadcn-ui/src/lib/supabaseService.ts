import { supabase } from './supabaseClient';

// Types for database entities
export interface Driver {
  id: string;
  ergast_driver_id: string;
  code: string;
  permanent_number: number;
  given_name: string;
  family_name: string;
  date_of_birth: string;
  nationality: string;
  url: string;
  headshot_url?: string;
}

export interface Constructor {
  id: string;
  ergast_constructor_id: string;
  name: string;
  nationality: string;
  url: string;
  logo_url?: string;
}

export interface Circuit {
  id: string;
  ergast_circuit_id: string;
  name: string;
  location: string;
  country: string;
  lat: number;
  lng: number;
  url: string;
}

export interface Race {
  id: string;
  ergast_race_id: string;
  season_year: number;
  round: number;
  name: string;
  circuit_id: string;
  date: string;
  time?: string;
  url: string;
  circuit?: Circuit;
}

export interface RaceResult {
  id: string;
  race_id: string;
  driver_id: string;
  constructor_id: string;
  position?: number;
  position_text: string;
  points: number;
  grid: number;
  laps: number;
  status: string;
  fastest_lap_rank?: number;
  driver?: Driver;
  constructor?: Constructor;
}

export interface Prediction {
  id: string;
  race_id: string;
  driver_id: string;
  prediction_type: string;
  predicted_position: number;
  confidence: number;
  model_version: string;
  driver?: Driver;
}

export interface UserProfile {
  user_id: string;
  username?: string;
  favorite_driver_id?: string;
  favorite_constructor_id?: string;
  favorite_driver?: Driver;
  favorite_constructor?: Constructor;
}

export interface DriverStanding {
  position: number;
  driver: Driver;
  constructor: Constructor;
  points: number;
  wins: number;
}

export interface ConstructorStanding {
  position: number;
  constructor: Constructor;
  points: number;
  wins: number;
}

// Fetch all drivers
export async function fetchDrivers(): Promise<Driver[]> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_drivers')
    .select('*')
    .order('family_name', { ascending: true });

  if (error) throw error;
  return data || [];
}

// Fetch all constructors
export async function fetchConstructors(): Promise<Constructor[]> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_constructors')
    .select('*')
    .order('name', { ascending: true });

  if (error) throw error;
  return data || [];
}

// Fetch current season races
export async function fetchRaces(year: number): Promise<Race[]> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_races')
    .select(`
      *,
      circuit:app_b64c9980ff_circuits(*)
    `)
    .eq('season_year', year)
    .order('round', { ascending: true });

  if (error) throw error;
  return data || [];
}

// Fetch next upcoming race
export async function fetchNextRace(): Promise<Race | null> {
  const today = new Date().toISOString().split('T')[0];
  
  const { data, error } = await supabase
    .from('app_b64c9980ff_races')
    .select(`
      *,
      circuit:app_b64c9980ff_circuits(*)
    `)
    .gte('date', today)
    .order('date', { ascending: true })
    .limit(1)
    .single();

  if (error && error.code !== 'PGRST116') throw error;
  return data || null;
}

// Fetch last completed race
export async function fetchLastRace(): Promise<Race | null> {
  const today = new Date().toISOString().split('T')[0];
  
  const { data, error } = await supabase
    .from('app_b64c9980ff_races')
    .select(`
      *,
      circuit:app_b64c9980ff_circuits(*)
    `)
    .lt('date', today)
    .order('date', { ascending: false })
    .limit(1)
    .single();

  if (error && error.code !== 'PGRST116') throw error;
  return data || null;
}

// Fetch race results for a specific race
export async function fetchRaceResults(raceId: string): Promise<RaceResult[]> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_race_results')
    .select(`
      *,
      driver:app_b64c9980ff_drivers(*),
      constructor:app_b64c9980ff_constructors(*)
    `)
    .eq('race_id', raceId)
    .order('position', { ascending: true });

  if (error) throw error;
  return data || [];
}

// Fetch driver standings (aggregated points)
export async function fetchDriverStandings(year: number): Promise<DriverStanding[]> {
  // Get all race results for the season
  const { data: races } = await supabase
    .from('app_b64c9980ff_races')
    .select('id')
    .eq('season_year', year);

  if (!races) return [];

  const raceIds = races.map(r => r.id);

  const { data, error } = await supabase
    .from('app_b64c9980ff_race_results')
    .select(`
      driver_id,
      points,
      position,
      driver:app_b64c9980ff_drivers(*),
      constructor:app_b64c9980ff_constructors(*)
    `)
    .in('race_id', raceIds);

  if (error) throw error;

  // Aggregate points by driver
  const standingsMap = new Map<string, DriverStanding>();
  
  for (const result of data || []) {
    const driverId = result.driver_id;
    if (!standingsMap.has(driverId)) {
      standingsMap.set(driverId, {
        driver: result.driver,
        constructor: result.constructor,
        points: 0,
        wins: 0,
        position: 0,
      });
    }
    
    const standing = standingsMap.get(driverId)!;
    standing.points += result.points || 0;
    if (result.position === 1) standing.wins += 1;
  }

  // Convert to array and sort by points
  return Array.from(standingsMap.values())
    .sort((a, b) => b.points - a.points)
    .map((standing, index) => ({
      ...standing,
      position: index + 1,
    }));
}

// Fetch constructor standings (aggregated points)
export async function fetchConstructorStandings(year: number): Promise<ConstructorStanding[]> {
  const { data: races } = await supabase
    .from('app_b64c9980ff_races')
    .select('id')
    .eq('season_year', year);

  if (!races) return [];

  const raceIds = races.map(r => r.id);

  const { data, error } = await supabase
    .from('app_b64c9980ff_race_results')
    .select(`
      constructor_id,
      points,
      position,
      constructor:app_b64c9980ff_constructors(*)
    `)
    .in('race_id', raceIds);

  if (error) throw error;

  // Aggregate points by constructor
  const standingsMap = new Map<string, ConstructorStanding>();
  
  for (const result of data || []) {
    const constructorId = result.constructor_id;
    if (!standingsMap.has(constructorId)) {
      standingsMap.set(constructorId, {
        constructor: result.constructor,
        points: 0,
        wins: 0,
        position: 0,
      });
    }
    
    const standing = standingsMap.get(constructorId)!;
    standing.points += result.points || 0;
    if (result.position === 1) standing.wins += 1;
  }

  return Array.from(standingsMap.values())
    .sort((a, b) => b.points - a.points)
    .map((standing, index) => ({
      ...standing,
      position: index + 1,
    }));
}

// Fetch predictions for a race
export async function fetchPredictions(raceId: string): Promise<Prediction[]> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_predictions')
    .select(`
      *,
      driver:app_b64c9980ff_drivers(*)
    `)
    .eq('race_id', raceId)
    .order('predicted_position', { ascending: true });

  if (error) throw error;
  return data || [];
}

// Fetch user profile
export async function fetchUserProfile(userId: string): Promise<UserProfile | null> {
  const { data, error } = await supabase
    .from('app_b64c9980ff_profiles')
    .select(`
      *,
      favorite_driver:app_b64c9980ff_drivers(*),
      favorite_constructor:app_b64c9980ff_constructors(*)
    `)
    .eq('user_id', userId)
    .single();

  if (error && error.code !== 'PGRST116') throw error;
  return data || null;
}

// Update user profile
export async function updateUserProfile(
  userId: string,
  updates: Partial<UserProfile>
): Promise<void> {
  const { error } = await supabase
    .from('app_b64c9980ff_profiles')
    .upsert({
      user_id: userId,
      ...updates,
      updated_at: new Date().toISOString(),
    });

  if (error) throw error;
}

// Call AI Assistant edge function
export async function askAIAssistant(query: string, userId?: string): Promise<string> {
  const { data, error } = await supabase.functions.invoke('app_b64c9980ff_ai_assistant', {
    body: { query, userId },
  });

  if (error) throw error;
  return data.response;
}

// Trigger data sync
export async function triggerDataSync(): Promise<void> {
  const { error } = await supabase.functions.invoke('app_b64c9980ff_data_sync');
  if (error) throw error;
}

// Trigger prediction generation
export async function triggerPredictionGeneration(): Promise<void> {
  const { error } = await supabase.functions.invoke('app_b64c9980ff_generate_predictions');
  if (error) throw error;
}