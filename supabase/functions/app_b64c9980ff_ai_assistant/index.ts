import { createClient } from 'npm:@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': '*',
};

Deno.serve(async (req) => {
  const requestId = crypto.randomUUID();
  console.log(`[${requestId}] Request received:`, req.method, req.url);

  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  try {
    // Parse request body
    let body;
    try {
      body = await req.json();
    } catch (e) {
      console.error(`[${requestId}] Failed to parse request body:`, e);
      return new Response(
        JSON.stringify({ error: 'Invalid request body' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const { query, userId } = body;

    if (!query) {
      return new Response(
        JSON.stringify({ error: 'Query is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`[${requestId}] Processing query for user:`, userId);

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Fetch context data from database
    console.log(`[${requestId}] Fetching context data`);
    
    // Get current season and upcoming race
    const currentYear = new Date().getFullYear();
    const { data: races, error: racesError } = await supabase
      .from('app_b64c9980ff_races')
      .select(`
        *,
        circuit:app_b64c9980ff_circuits(*)
      `)
      .eq('season_year', currentYear)
      .gte('date', new Date().toISOString().split('T')[0])
      .order('date', { ascending: true })
      .limit(1);

    if (racesError) {
      console.error(`[${requestId}] Error fetching races:`, racesError);
    }

    // Get latest driver standings
    const { data: driverStandings, error: driverError } = await supabase
      .from('app_b64c9980ff_race_results')
      .select(`
        driver_id,
        driver:app_b64c9980ff_drivers(*),
        constructor:app_b64c9980ff_constructors(*),
        points
      `)
      .order('points', { ascending: false })
      .limit(10);

    if (driverError) {
      console.error(`[${requestId}] Error fetching driver standings:`, driverError);
    }

    // Get user profile if userId provided
    let userProfile = null;
    if (userId) {
      const { data: profile } = await supabase
        .from('app_b64c9980ff_profiles')
        .select(`
          *,
          favorite_driver:app_b64c9980ff_drivers(*),
          favorite_constructor:app_b64c9980ff_constructors(*)
        `)
        .eq('user_id', userId)
        .single();
      
      userProfile = profile;
    }

    // Build context for LLM
    const context = {
      currentYear,
      upcomingRace: races?.[0] || null,
      driverStandings: driverStandings || [],
      userProfile,
    };

    console.log(`[${requestId}] Context assembled, calling OpenAI`);

    // Call OpenAI API
    const openaiApiKey = Deno.env.get('OPENAI_API_KEY');
    if (!openaiApiKey) {
      throw new Error('OPENAI_API_KEY not configured');
    }

    const systemPrompt = `You are an expert F1 analyst and enthusiastic fan assistant. You have deep knowledge of Formula 1 racing, including current season data, driver performances, team strategies, and historical context.

Current Context:
- Season: ${currentYear}
- Upcoming Race: ${context.upcomingRace ? `${context.upcomingRace.name} at ${context.upcomingRace.circuit?.name}` : 'No upcoming race'}
${userProfile?.favorite_driver ? `- User's Favorite Driver: ${userProfile.favorite_driver.given_name} ${userProfile.favorite_driver.family_name}` : ''}

Provide insightful, accurate, and engaging responses about F1. Use data when available, but also share your passion for the sport.`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: query }
        ],
        temperature: 0.7,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[${requestId}] OpenAI API error:`, response.status, errorText);
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const aiResponse = await response.json();
    console.log(`[${requestId}] AI response received`);

    return new Response(
      JSON.stringify({
        response: aiResponse.choices[0].message.content,
        context: {
          upcomingRace: context.upcomingRace?.name,
          favoriteDriver: userProfile?.favorite_driver ? 
            `${userProfile.favorite_driver.given_name} ${userProfile.favorite_driver.family_name}` : null,
        }
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error(`[${requestId}] Error:`, error);
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