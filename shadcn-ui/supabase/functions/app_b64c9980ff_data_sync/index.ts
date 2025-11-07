import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': '*',
};

Deno.serve(async (req) => {
  const requestId = crypto.randomUUID();
  console.log(`[${requestId}] Data sync request received:`, req.method);

  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  try {
    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    const currentYear = new Date().getFullYear();
    console.log(`[${requestId}] Syncing data for season ${currentYear}`);

    // Fetch current season data from Ergast API
    const ergastBaseUrl = 'https://ergast.com/api/f1';
    
    // 1. Sync Drivers
    console.log(`[${requestId}] Fetching drivers from Ergast API`);
    const driversResponse = await fetch(`${ergastBaseUrl}/${currentYear}/drivers.json`);
    const driversData = await driversResponse.json();
    const drivers = driversData.MRData.DriverTable.Drivers;

    console.log(`[${requestId}] Syncing ${drivers.length} drivers`);
    for (const driver of drivers) {
      const { error } = await supabase
        .from('app_b64c9980ff_drivers')
        .upsert({
          ergast_driver_id: driver.driverId,
          code: driver.code,
          permanent_number: driver.permanentNumber ? parseInt(driver.permanentNumber) : null,
          given_name: driver.givenName,
          family_name: driver.familyName,
          date_of_birth: driver.dateOfBirth,
          nationality: driver.nationality,
          url: driver.url,
          updated_at: new Date().toISOString(),
        }, {
          onConflict: 'ergast_driver_id',
        });

      if (error) {
        console.error(`[${requestId}] Error upserting driver ${driver.driverId}:`, error);
      }
    }

    // 2. Sync Constructors
    console.log(`[${requestId}] Fetching constructors from Ergast API`);
    const constructorsResponse = await fetch(`${ergastBaseUrl}/${currentYear}/constructors.json`);
    const constructorsData = await constructorsResponse.json();
    const constructors = constructorsData.MRData.ConstructorTable.Constructors;

    console.log(`[${requestId}] Syncing ${constructors.length} constructors`);
    for (const constructor of constructors) {
      const { error } = await supabase
        .from('app_b64c9980ff_constructors')
        .upsert({
          ergast_constructor_id: constructor.constructorId,
          name: constructor.name,
          nationality: constructor.nationality,
          url: constructor.url,
          updated_at: new Date().toISOString(),
        }, {
          onConflict: 'ergast_constructor_id',
        });

      if (error) {
        console.error(`[${requestId}] Error upserting constructor ${constructor.constructorId}:`, error);
      }
    }

    // 3. Sync Circuits
    console.log(`[${requestId}] Fetching circuits from Ergast API`);
    const circuitsResponse = await fetch(`${ergastBaseUrl}/${currentYear}/circuits.json`);
    const circuitsData = await circuitsResponse.json();
    const circuits = circuitsData.MRData.CircuitTable.Circuits;

    console.log(`[${requestId}] Syncing ${circuits.length} circuits`);
    for (const circuit of circuits) {
      const { error } = await supabase
        .from('app_b64c9980ff_circuits')
        .upsert({
          ergast_circuit_id: circuit.circuitId,
          name: circuit.circuitName,
          location: circuit.Location.locality,
          country: circuit.Location.country,
          lat: parseFloat(circuit.Location.lat),
          lng: parseFloat(circuit.Location.long),
          url: circuit.url,
          updated_at: new Date().toISOString(),
        }, {
          onConflict: 'ergast_circuit_id',
        });

      if (error) {
        console.error(`[${requestId}] Error upserting circuit ${circuit.circuitId}:`, error);
      }
    }

    // 4. Sync Season
    console.log(`[${requestId}] Syncing season ${currentYear}`);
    await supabase
      .from('app_b64c9980ff_seasons')
      .upsert({
        year: currentYear,
        url: `${ergastBaseUrl}/${currentYear}.json`,
      }, {
        onConflict: 'year',
      });

    // 5. Sync Race Schedule
    console.log(`[${requestId}] Fetching race schedule from Ergast API`);
    const scheduleResponse = await fetch(`${ergastBaseUrl}/${currentYear}.json`);
    const scheduleData = await scheduleResponse.json();
    const races = scheduleData.MRData.RaceTable.Races;

    console.log(`[${requestId}] Syncing ${races.length} races`);
    for (const race of races) {
      // Get circuit UUID
      const { data: circuitData } = await supabase
        .from('app_b64c9980ff_circuits')
        .select('id')
        .eq('ergast_circuit_id', race.Circuit.circuitId)
        .single();

      if (!circuitData) {
        console.error(`[${requestId}] Circuit not found for race: ${race.Circuit.circuitId}`);
        continue;
      }

      const { error } = await supabase
        .from('app_b64c9980ff_races')
        .upsert({
          ergast_race_id: `${currentYear}_${race.round}`,
          season_year: currentYear,
          round: parseInt(race.round),
          name: race.raceName,
          circuit_id: circuitData.id,
          date: race.date,
          time: race.time ? race.time.substring(0, 8) : null,
          url: race.url,
          updated_at: new Date().toISOString(),
        }, {
          onConflict: 'ergast_race_id',
        });

      if (error) {
        console.error(`[${requestId}] Error upserting race ${race.raceName}:`, error);
      }
    }

    // 6. Sync Race Results (for completed races)
    console.log(`[${requestId}] Syncing race results`);
    const resultsResponse = await fetch(`${ergastBaseUrl}/${currentYear}/results.json?limit=1000`);
    const resultsData = await resultsResponse.json();
    const raceResults = resultsData.MRData.RaceTable.Races;

    for (const race of raceResults) {
      // Get race UUID
      const { data: raceData } = await supabase
        .from('app_b64c9980ff_races')
        .select('id')
        .eq('ergast_race_id', `${currentYear}_${race.round}`)
        .single();

      if (!raceData) continue;

      for (const result of race.Results) {
        // Get driver and constructor UUIDs
        const { data: driverData } = await supabase
          .from('app_b64c9980ff_drivers')
          .select('id')
          .eq('ergast_driver_id', result.Driver.driverId)
          .single();

        const { data: constructorData } = await supabase
          .from('app_b64c9980ff_constructors')
          .select('id')
          .eq('ergast_constructor_id', result.Constructor.constructorId)
          .single();

        if (!driverData || !constructorData) continue;

        const { error } = await supabase
          .from('app_b64c9980ff_race_results')
          .upsert({
            race_id: raceData.id,
            driver_id: driverData.id,
            constructor_id: constructorData.id,
            position: result.position ? parseInt(result.position) : null,
            position_text: result.positionText,
            points: parseFloat(result.points),
            grid: parseInt(result.grid),
            laps: parseInt(result.laps),
            status: result.status,
            fastest_lap_rank: result.FastestLap?.rank ? parseInt(result.FastestLap.rank) : null,
          }, {
            onConflict: 'race_id,driver_id',
          });

        if (error) {
          console.error(`[${requestId}] Error upserting race result:`, error);
        }
      }
    }

    console.log(`[${requestId}] Data sync completed successfully`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: 'Data sync completed',
        stats: {
          drivers: drivers.length,
          constructors: constructors.length,
          circuits: circuits.length,
          races: races.length,
        }
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error(`[${requestId}] Error during data sync:`, error);
    return new Response(
      JSON.stringify({ 
        error: error.message || 'Internal server error',
        requestId 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});