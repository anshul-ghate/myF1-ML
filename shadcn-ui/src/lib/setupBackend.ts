// Backend setup utility
import { triggerDataSync, triggerPredictionGeneration } from './supabaseService';

export async function initializeBackend() {
  try {
    console.log('Starting backend initialization...');
    
    // Step 1: Sync F1 data from Ergast API
    console.log('Syncing F1 data...');
    await triggerDataSync();
    console.log('Data sync completed');
    
    // Step 2: Generate predictions for upcoming race
    console.log('Generating predictions...');
    await triggerPredictionGeneration();
    console.log('Predictions generated');
    
    console.log('Backend initialization completed successfully');
    return { success: true };
  } catch (error) {
    console.error('Backend initialization failed:', error);
    return { success: false, error };
  }
}