import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': '*',
};

Deno.serve(async (req) => {
  const requestId = crypto.randomUUID();
  console.log(`[${requestId}] Prediction generation request received:`, req.method);

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
    console.log(`[${requestId}] Generating predictions for season ${currentYear}`);

    // Get the next upcoming race
    const { data: upcomingRaces, error: raceError } = await supabase
      .from('app_b64c9980ff_races')
      .select('*')
      .eq('season_year', currentYear)
      .gte('date', new Date().toISOString().split('T')[0])
      .order('date', { ascending: true })
      .limit(1);

    if (raceError || !upcomingRaces || upcomingRaces.length === 0) {
      console.log(`[${requestId}] No upcoming races found`);
      return new Response(
        JSON.stringify({ message: 'No upcoming races to predict' }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const upcomingRace = upcomingRaces[0];
    console.log(`[${requestId}] Generating predictions for race: ${upcomingRace.name}`);

    // Get all drivers and their recent performance
    const { data: drivers, error: driversError } = await supabase
      .from('app_b64c9980ff_drivers')
      .select('*');

    if (driversError || !drivers) {
      throw new Error('Failed to fetch drivers');
    }

    // Get recent race results for each driver (last 5 races)
    const { data: recentResults, error: resultsError } = await supabase
      .from('app_b64c9980ff_race_results')
      .select(`
        *,
        race:app_b64c9980ff_races(date, round)
      `)
      .eq('race.season_year', currentYear)
      .order('race.date', { ascending: false })
      .limit(100);

    if (resultsError) {
      console.error(`[${requestId}] Error fetching recent results:`, resultsError);
    }

    // Calculate driver performance scores
    const driverScores = new Map();
    
    for (const driver of drivers) {
      const driverResults = recentResults?.filter(r => r.driver_id === driver.id) || [];
      
      // Calculate average position and points
      let totalPoints = 0;
      let totalPosition = 0;
      let raceCount = 0;

      for (const result of driverResults.slice(0, 5)) {
        totalPoints += result.points || 0;
        if (result.position) {
          totalPosition += result.position;
          raceCount++;
        }
      }

      const avgPoints = raceCount > 0 ? totalPoints / raceCount : 0;
      const avgPosition = raceCount > 0 ? totalPosition / raceCount : 20;

      // Simple scoring: lower average position and higher points = better score
      // Normalize to 0-1 range where 1 is best
      const positionScore = Math.max(0, (20 - avgPosition) / 20);
      const pointsScore = Math.min(1, avgPoints / 25);
      
      // Combined score with weights
      const score = (positionScore * 0.6) + (pointsScore * 0.4);
      
      driverScores.set(driver.id, {
        driver,
        score,
        avgPosition,
        avgPoints,
        recentRaces: raceCount,
      });
    }

    // Sort drivers by score
    const sortedDrivers = Array.from(driverScores.values())
      .sort((a, b) => b.score - a.score);

    // Generate predictions with confidence scores
    console.log(`[${requestId}] Generating predictions for ${sortedDrivers.length} drivers`);
    
    const predictions = [];
    for (let i = 0; i < sortedDrivers.length; i++) {
      const driverData = sortedDrivers[i];
      
      // Confidence decreases as predicted position increases
      // Top drivers have higher confidence
      const baseConfidence = driverData.score;
      const positionPenalty = i * 0.02; // Reduce confidence for lower positions
      const confidence = Math.max(0.1, Math.min(0.95, baseConfidence - positionPenalty));

      predictions.push({
        race_id: upcomingRace.id,
        driver_id: driverData.driver.id,
        prediction_type: 'RACE_FINISH',
        predicted_position: i + 1,
        confidence: parseFloat(confidence.toFixed(4)),
        model_version: 'v1.0_simple_avg',
      });
    }

    // Delete existing predictions for this race
    await supabase
      .from('app_b64c9980ff_predictions')
      .delete()
      .eq('race_id', upcomingRace.id);

    // Insert new predictions
    const { error: insertError } = await supabase
      .from('app_b64c9980ff_predictions')
      .insert(predictions);

    if (insertError) {
      console.error(`[${requestId}] Error inserting predictions:`, insertError);
      throw insertError;
    }

    console.log(`[${requestId}] Successfully generated ${predictions.length} predictions`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: 'Predictions generated successfully',
        race: upcomingRace.name,
        predictionsCount: predictions.length,
        topPredictions: predictions.slice(0, 5).map(p => ({
          position: p.predicted_position,
          confidence: p.confidence,
        })),
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error(`[${requestId}] Error generating predictions:`, error);
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